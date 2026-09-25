from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._allocator import get_global_state_pointer
from triton._C.libtriton.gsan_testing import thread_state_address, SHADOW_GRANULARITY_BYTES, PER_DEVICE_STATE_STRIDE_BYTES, GLOBAL_STATE_SIZE_BYTES, shadow_cell_address, thread_state_stride_bytes, SHADOW_CELL_SIZE_BYTES
from ._testing import (decode_global_state_tensor, decode_shadow_cell_tensor, decode_thread_state_tensor)
from ._utils import uint8_cuda_tensor_from_ptr


@triton.jit
def nanosleep(duration):
    duration = tl.to_tensor(duration)
    tl.inline_asm_elementwise("nanosleep.u32 $1; mov.b32 $0, 0;", "=r, r", [duration], tl.int32, is_pure=False, pack=1)


@triton.jit
def atomic_poll(ptr, expect, sem: tl.constexpr = "relaxed", scope: tl.constexpr = "gpu"):
    while tl.atomic_add(ptr, 0, sem=sem, scope=scope) != expect:
        nanosleep(100)


def shadow_cell_tensor_from_address(real_address: int, *, device_index: int | None = None) -> torch.Tensor:
    if device_index is None:
        device_index = torch.cuda.current_device()
    shadow_ptr = shadow_cell_address(real_address)
    return uint8_cuda_tensor_from_ptr(shadow_ptr, SHADOW_CELL_SIZE_BYTES, device_index)


def shadow_cell_from_address(real_address: int, *, device_index: int | None = None):
    return decode_shadow_cell_tensor(shadow_cell_tensor_from_address(real_address, device_index=device_index))


def global_state_tensor(*, device_index: int | None = None) -> torch.Tensor:
    if device_index is None:
        device_index = torch.cuda.current_device()
    ptr = get_global_state_pointer() + device_index * PER_DEVICE_STATE_STRIDE_BYTES
    return uint8_cuda_tensor_from_ptr(ptr, GLOBAL_STATE_SIZE_BYTES, device_index)


def global_state(*, device_index: int | None = None):
    return decode_global_state_tensor(global_state_tensor(device_index=device_index))


def thread_state_tensor(smid: int, *, device_index: int | None = None) -> torch.Tensor:
    if device_index is None:
        device_index = torch.cuda.current_device()
    gs = global_state(device_index=device_index)
    ptr = thread_state_address(
        get_global_state_pointer() + device_index * PER_DEVICE_STATE_STRIDE_BYTES,
        gs.num_threads,
        gs.clock_buffer_size,
        smid,
    )
    size = thread_state_stride_bytes(gs.num_threads, gs.clock_buffer_size)
    return uint8_cuda_tensor_from_ptr(ptr, size, device_index)


def thread_state_from_smid(smid: int, *, device_index: int | None = None):
    if device_index is None:
        device_index = torch.cuda.current_device()
    gs = global_state(device_index=device_index)
    return decode_thread_state_tensor(
        thread_state_tensor(smid, device_index=device_index),
        gs.num_threads,
        gs.clock_buffer_size,
    )


def shadow_tensor_for(real: torch.Tensor) -> torch.Tensor:
    shadow_ptr = shadow_cell_address(real.data_ptr())
    nbytes = real.untyped_storage().nbytes()
    num_cells = triton.cdiv(nbytes, SHADOW_GRANULARITY_BYTES)
    shadow_size = num_cells * SHADOW_CELL_SIZE_BYTES
    return uint8_cuda_tensor_from_ptr(shadow_ptr, shadow_size, real.device.index)


@triton.jit
def store_one_i32(ptr):
    tl.store(ptr, 1)


@triton.jit
def load_one_i32(ptr, out_ptr):
    value = tl.load(ptr)
    tl.store(out_ptr, value)


class ExplicitGraph:
    """Small native graph launcher for GSan regression tests (no stream capture)."""

    def __init__(self, plan):
        import ctypes as C

        self.C = C
        self.cuda = C.CDLL("libcuda.so.1")
        self.plan = plan
        self.graph = C.c_void_p()
        self.executable = C.c_void_p()
        self.nodes = []
        self.compiled = []
        self.parameters = []

        class Params(C.Structure):
            _fields_ = [("func", C.c_void_p), ("grid_x", C.c_uint), ("grid_y", C.c_uint), ("grid_z", C.c_uint),
                        ("block_x", C.c_uint), ("block_y", C.c_uint), ("block_z", C.c_uint), ("shared", C.c_uint),
                        ("args", C.POINTER(C.c_void_p)), ("extra", C.c_void_p), ("kernel", C.c_void_p),
                        ("context", C.c_void_p)]

        self.Params = Params
        self._call("cuGraphCreate", C.byref(self.graph), C.c_uint(0))
        begin, self.end_spec = plan.boundary_kernels
        self.begin_node = self._add(begin, ())

    def _call(self, name, *args):
        status = getattr(self.cuda, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with CUDA error {status}")

    def _params(self, spec):
        C = self.C
        values = (C.c_uint64 * len(spec.arguments))(*spec.arguments)
        pointers = (C.c_void_p * len(values))(*(C.addressof(values) + i * 8 for i in range(len(values))))
        params = self.Params(spec.function, spec.grid, 1, 1, spec.block, 1, 1, spec.shared, pointers, None, None, None)
        return params, values, pointers

    def _add(self, spec, dependencies):
        C = self.C
        node = C.c_void_p()
        deps = (C.c_void_p * len(dependencies))(*(dep.value for dep in dependencies))
        params, values, pointers = self._params(spec)
        self._call("cuGraphAddKernelNode_v2", C.byref(node), self.graph, deps, C.c_size_t(len(dependencies)),
                   C.byref(params))
        self.parameters.append((params, values, pointers))
        return node

    def add_kernel(self, compiled, arguments, grid):
        from .graph import KernelLaunch

        C = self.C
        index = len(self.nodes)
        node = self.plan.nodes[index]
        assert compiled.metadata.num_ctas == 1
        assert not compiled.metadata.global_scratch_size and not compiled.metadata.profile_scratch_size
        compiled._init_handles()
        self.compiled.append(compiled)
        values = tuple(arg.data_ptr() if hasattr(arg, "data_ptr") else arg for arg in arguments)
        spec = KernelLaunch(compiled.function, grid, compiled.metadata.num_warps * 32, compiled.metadata.shared,
                            (*values, *self.plan.node_launch_args(index), 0, 0))
        deps = tuple(self.nodes[j] for j in node.dependencies)
        if not node.dependencies and not node.programmatic_dependencies:
            deps = (self.begin_node, )
        result = self._add(spec, deps)
        for predecessor in node.programmatic_dependencies:
            # from_port=programmatic, to_port=default, type=programmatic.
            edge = (C.c_ubyte * 8)(1, 0, 1, 0, 0, 0, 0, 0)
            self._call("cuGraphAddDependencies_v2", self.graph, C.byref(self.nodes[predecessor]), C.byref(result),
                       C.byref(edge), C.c_size_t(1))
        self.nodes.append(result)

    def instantiate(self):
        C = self.C
        assert len(self.nodes) == len(self.plan.nodes)
        self.end_node = self._add(self.end_spec, self.nodes or (self.begin_node, ))
        self._call("cuGraphInstantiateWithFlags", C.byref(self.executable), self.graph, C.c_uint64(0))

    def launch(self, stream=None, dependencies=()):
        C = self.C
        if stream is None:
            stream = torch.cuda.current_stream()
        prepared = self.plan.prepare_launch(stream.cuda_stream, dependencies)
        try:
            for node, spec in ((self.begin_node, prepared.begin), (self.end_node, prepared.end)):
                params, _values, _pointers = self._params(spec)
                self._call("cuGraphExecKernelNodeSetParams_v2", self.executable, node, C.byref(params))
            self._call("cuGraphLaunch", self.executable, C.c_void_p(stream.cuda_stream))
        except BaseException:
            prepared.finish(False)
            raise
        prepared.finish()
        return prepared.completion

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self.executable.value:
            self._call("cuGraphExecDestroy", self.executable)
        self._call("cuGraphDestroy", self.graph)
        self.plan.close()
