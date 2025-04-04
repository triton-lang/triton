import functools
from typing import Dict, List, Optional, Tuple, Any

import triton
from triton._C.libtriton import ir
from triton._C.libtriton import proton as triton_proton
from triton._C.libproton import proton as libproton
from triton.compiler import LazyDict
from triton.runtime.jit import JITFunction
from triton.runtime._allocation import set_profile_allocator, NullAllocator
from triton.backends import backends

from .hook import Hook
from ..flags import set_instrumentation_on, set_instrumentation_off


class CudaAllocator:

    def __init__(self, instrumentation_hook):
        self.instrumentation_hook = instrumentation_hook

    def __call__(self, size: int, alignment: int, stream: Optional[int]):
        import torch

        # Ensure proper alignment and minimum size
        # `alignment` and `size` are specified by the instrumentation engine to specify the minimum required
        # alignment and size of the buffer.
        # Then, we make it match the default profile buffer alignment and size here, taking the maximum of the two.
        alignment = max(alignment, self.instrumentation_hook.profile_buffer_alignment)
        aligned_size = (size + alignment - 1) // alignment * alignment
        aligned_size = max(aligned_size, self.instrumentation_hook.profile_buffer_size)

        # Create the buffer
        # FIXME(Keren): This is not correct in general, as the buffer will be deallocated by the Torch runtime.
        # We should use a custom allocator to manage the buffer lifecycle.
        buffer = torch.empty((aligned_size, ), dtype=torch.uint8, device="cuda")
        self.instrumentation_hook.buffer = buffer
        return buffer


class Instrumentation:

    def __init__(self, ir_map: Dict[str, Any]):
        self.loaded = False
        self.manager = ir_map

    def register(self, ir: str, func):
        if ir in self.manager:
            raise RuntimeError(f"IR already registered: {ir}")
        self.manager[ir] = func

    def patch(self, ir: str, pm, context):
        self.load_dialects(context)
        if ir in self.manager:
            self.manager[ir](pm)

    def load_dialects(self, ctx):
        if self.loaded:
            return
        self.loaded = True
        triton_proton.load_dialects(ctx)


class InstrumentationHook(Hook):
    # It's important to note that only one instance of the instrumentation hook can be active at a time.
    active_count = 0
    profile_mem = None

    def __init__(self, mode: str,  #
                 profile_buffer_size: int = 16 * 1024 * 1024,  #
                 profile_buffer_alignment: int = 128,  #
                 ):
        # Mapping of function objects to their scope ID pairs
        self.function_scope_ids: Dict[Any, List[Tuple[int, int]]] = {}
        self.mode = mode

        # Default buffer configuration
        self.profile_buffer_size = profile_buffer_size
        self.profile_buffer_alignment = profile_buffer_alignment
        self.allocator = CudaAllocator(self)
        self.buffer = None

    def activate(self):
        if InstrumentationHook.active_count > 0:
            raise RuntimeError("Only one instance of the instrumentation hook can be active at a time.")

        InstrumentationHook.active_count += 1

        set_instrumentation_on()

        backend = triton.runtime.driver.active.get_current_target().backend

        def to_llvmir_passes(pm, backend_pass):
            triton_proton.add_allocate_proton_global_scratch_buffer(pm)
            backend_pass(pm)

        def to_ttgpuir_passes(pm):
            triton_proton.add_convert_proton_to_protongpu(pm)
            triton_proton.add_allocate_proton_shared_memory(pm)

        if backend == "cuda":
            backend_name = "nvidia"
            ttgpuir_func = lambda pm: to_ttgpuir_passes(pm)
            llvmir_func = lambda pm: to_llvmir_passes(pm, triton_proton.add_convert_proton_nvidia_gpu_to_llvm)
        elif backend == "hip":
            backend_name = "amd"
            ttgpuir_func = lambda pm: to_ttgpuir_passes(pm)
            llvmir_func = lambda pm: to_llvmir_passes(pm, triton_proton.add_convert_proton_amd_gpu_to_llvm)
        else:
            raise RuntimeError(f"Unsupported backend: {backend}")

        backends[backend_name].compiler.instrumentation = Instrumentation({
            "ttgpuir": ttgpuir_func,
            "llvmir": llvmir_func,
        })

        # Set up the profiling allocator
        set_profile_allocator(self.allocator)

        original_run = JITFunction.run

        mode = self.mode

        @functools.wraps(original_run)
        def instrumented_run(self, *args, **kwargs):
            kwargs["instrumentation_mode"] = mode
            return original_run(self, *args, **kwargs)

        JITFunction.run = instrumented_run

    def deactivate(self):
        if InstrumentationHook.active_count == 0:
            return

        InstrumentationHook.active_count -= 1

        backend = triton.runtime.driver.active.get_current_target().backend
        if backend == "cuda":
            backend_name = "nvidia"
        elif backend == "hip":
            backend_name = "amd"
        else:
            raise RuntimeError(f"Unsupported backend: {backend}")

        backends[backend_name].compiler.instrumentation = {}

        set_instrumentation_off()

        # Restore original JIT function run method
        if hasattr(JITFunction.run, "__wrapped__"):
            JITFunction.run = JITFunction.run.__wrapped__

        # Reset profile allocator
        set_profile_allocator(NullAllocator())

        # Reset host memory for external processing
        InstrumentationHook.profile_mem = None

    def init_handle(self, function: Any, module: Any, metadata_group: Dict[str, str]) -> None:
        if function and function not in self.function_scope_ids:
            # Find the IR path in metadata
            ir_path = next((path for key, path in metadata_group.items() if key.endswith(("ttgir", "ttir"))), None)

            if ir_path:
                # Parse the MLIR module to extract scope IDs
                context = ir.context()
                ir.load_dialects(context)
                triton_proton.load_dialects(context)
                module = ir.parse_mlir_module(ir_path, context)
                module.context = context
                self.function_scope_ids[function] = triton_proton.get_scope_id_pairs(module)

        scope_id_pairs = self.function_scope_ids.get(function, [])
        libproton.init_scope_ids(function, scope_id_pairs)

    def enter(self, lazy_dict: LazyDict) -> None:
        libproton.enter_instrumented_op(lazy_dict.data.get("function", None), self.buffer.data_ptr(),
                                        self.profile_buffer_size)
        InstrumentationHook.profile_mem = None

    def exit(self, lazy_dict: LazyDict) -> None:
        # FIXME(Keren): exit_instrumented_op will sync the device and copy the buffer back to the host.
        # But this is not necessary if we can delay the copy to reduce the overhead.
        # We should fix it after the profiling buffer is managed by a custom allocator.
        libproton.exit_instrumented_op(lazy_dict.data.get("function", None), self.buffer.data_ptr(),
                                       self.profile_buffer_size)

        # Copy the profiling buffer to the host for external processing (e.g., 3rd party tools).
        # FIXME(fywkevin): We should provide a config option to control on/off this behavior.
        import torch
        InstrumentationHook.profile_mem = torch.empty_like(self.buffer, device='cpu')
        InstrumentationHook.profile_mem.copy_(self.buffer)

        # Release the profiling buffer for recycling
        self.buffer = None
