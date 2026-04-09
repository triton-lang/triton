from typing import Dict, Optional, Union, Any

import triton
from triton._C.libtriton import ir as triton_ir
from triton._C.libtriton import proton as triton_proton
from triton._C.libtriton import amd as triton_amd
from triton._C.libtriton import nvidia as triton_nvidia
from triton._C.libtriton import passes as triton_passes
from triton._C.libproton import proton as libproton
from triton.compiler import LazyDict
from triton.runtime._allocation import set_profile_allocator, NullAllocator
from triton.backends import backends

from .hook import Hook
from ..flags import flags
from ..state import enter_state, exit_state, COMPUTE_METADATA_SCOPE_NAME
from .. import mode

# TODO(fywkevin): add support for major.minor
VERSION = 1


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _is_current_cuda_graph_capture() -> bool:
    import torch
    if _get_backend_name() != "nvidia":
        return False
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if is_capturing is None:
        return False
    return bool(is_capturing())


class ProfileScratchAllocation:

    def __init__(self, slot: "StepBufferSlot", offset: int, size: int):
        self.slot = slot
        self.offset = offset
        self.size = size

    def data_ptr(self) -> int:
        return self.slot.buffer.data_ptr() + self.offset

    def element_size(self) -> int:
        return 1

    def numel(self) -> int:
        return self.size

    def byte_range(self):
        return self.slot.buffer[self.offset:self.offset + self.size]


class StepBufferSlot:

    def __init__(self, buffer: Any):
        self.buffer = buffer
        self.capacity = int(buffer.element_size() * buffer.numel())
        self.token = buffer.data_ptr()
        self.reset()

    def reset(self) -> None:
        self.offset = 0
        self.sealed = False
        self.used = False


class StepBufferRing:

    def __init__(self, instrumentation_hook: "InstrumentationHook", slot_size: int, slot_count: int):
        self.instrumentation_hook = instrumentation_hook
        self.slots = [self._allocate_slot(slot_size) for _ in range(slot_count)]
        self.current_slot_idx: Optional[int] = None
        self.next_slot_idx = 0

    def _allocate_slot(self, slot_size: int) -> StepBufferSlot:
        import torch
        device = triton.runtime.driver.active.get_active_torch_device()
        enter_state(COMPUTE_METADATA_SCOPE_NAME)
        buffer = torch.zeros((slot_size, ), dtype=torch.uint8, device=device)
        exit_state()
        return StepBufferSlot(buffer)

    def allocate(self, size: int, alignment: int, stream: int) -> ProfileScratchAllocation:
        slot = self._ensure_current_slot(stream)
        offset = _round_up(slot.offset, alignment)
        if offset + size > slot.capacity:
            raise RuntimeError(
                "Profiling step buffer is full; call proton.data.advance_phase() sooner or increase "
                "TRITON_PROFILE_BUFFER_SIZE")
        slot.offset = offset + size
        slot.used = True
        return ProfileScratchAllocation(slot, offset, size)

    def mark_step(self, stream: int) -> None:
        if self.current_slot_idx is None:
            return
        slot = self.slots[self.current_slot_idx]
        if not slot.used:
            return
        libproton.mark_step(stream, slot.token)
        slot.sealed = True
        self.current_slot_idx = None
        self.next_slot_idx = (self.next_slot_idx + 1) % len(self.slots)

    def _ensure_current_slot(self, stream: int) -> StepBufferSlot:
        if self.current_slot_idx is None:
            self.current_slot_idx = self._acquire_slot(stream)
        return self.slots[self.current_slot_idx]

    def _acquire_slot(self, stream: int) -> int:
        slot_idx = self.next_slot_idx
        slot = self.slots[slot_idx]
        if slot.sealed:
            # advance_phase() already seals the slot and schedules async
            # draining. We only gate the compute stream when the ring wraps and
            # this exact slot must be reused before its copy has completed.
            libproton.wait_step_buffer(stream, slot.token)
        slot.reset()
        return slot_idx


class CudaAllocator:

    def __init__(self, instrumentation_hook):
        self.instrumentation_hook = instrumentation_hook

    def __call__(self, size: int, alignment: int, stream: Optional[int]):
        if alignment != self.instrumentation_hook.profile_buffer_alignment:
            raise RuntimeError(
                f"Alignment mismatch: {alignment} != {self.instrumentation_hook.profile_buffer_alignment}")
        aligned_size = _round_up(size, alignment)

        import torch
        if stream is None:
            device = triton.runtime.driver.active.get_current_device()
            stream = triton.runtime.driver.active.get_current_stream(device)
        if self.instrumentation_hook.mode.trace_mode == "kernel" and _is_current_cuda_graph_capture():
            raise RuntimeError(
                "trace_mode='kernel' is not supported during CUDA graph capture yet; "
                "captured launches bake in fixed profile scratch pointers, so replay-time "
                "slot reuse and phase attribution are unresolved.")
        if InstrumentationHook.enable_host_buffer:
            enter_state(COMPUTE_METADATA_SCOPE_NAME)
            buffer = torch.zeros((max(aligned_size, self.instrumentation_hook.profile_buffer_size), ),
                                 dtype=torch.uint8, device=triton.runtime.driver.active.get_active_torch_device())
            exit_state()
        else:
            if aligned_size > self.instrumentation_hook.profile_buffer_size:
                raise RuntimeError(
                    f"Kernel requested {aligned_size} bytes of profile scratch, which exceeds the preallocated step "
                    f"slot size {self.instrumentation_hook.profile_buffer_size}. Increase TRITON_PROFILE_BUFFER_SIZE.")
            buffer = self.instrumentation_hook.get_step_buffer_ring().allocate(aligned_size, alignment, stream)
        if self.instrumentation_hook.mode.trace_mode == "kernel":
            self.instrumentation_hook.initialize_kernel_trace_record(buffer, stream)
        self.instrumentation_hook.current_buffer = buffer
        return buffer


class Instrumentation:

    def __init__(self, ir_map: Dict[str, Any]):
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
        triton_proton.load_dialects(ctx)


def _interpret_mode(mode_obj: Union[str, mode.InstrumentationMode]) -> mode.InstrumentationMode:
    if isinstance(mode_obj, mode.InstrumentationMode):
        return mode_obj
    elif not mode_obj:
        mode_obj = "default"

    parts = mode_obj.split(":")
    mode_name = parts[0]
    opts: Dict[str, str] = {}
    for opt in parts[1:]:
        if "=" in opt:
            key, val = opt.split("=", 1)
            opts[key] = val
        else:
            raise ValueError(f"Malformed instrumentation option: '{opt}'")

    # Get option values or empty strings
    options = {
        "metric_type": opts.get("metric_type", "cycle"), "trace_mode": opts.get("trace_mode", "scope"),
        "buffer_type": opts.get("buffer_type", "shared"),
        "buffer_strategy": opts.get("buffer_strategy", "circular"), "buffer_size": int(opts.get("buffer_size", "0")),
        "granularity": opts.get("granularity", "warp"), "sampling_strategy": opts.get("sampling_strategy", "none"),
        "sampling_options": opts.get("sampling_options", ""), "optimizations": opts.get("optimizations", "")
    }

    if options["trace_mode"] not in {"scope", "kernel"}:
        raise ValueError(f"Unknown trace_mode: {options['trace_mode']}")

    # Helper function to validate and map options to their enum values
    def get_option_value(opt_name, mapping):
        value = options[opt_name]
        if value and value not in mapping:
            raise ValueError(f"Unknown {opt_name}: {value}")
        return mapping[value] if value else value

    # Look up enum values for each option
    options["metric_type"] = get_option_value("metric_type", mode.metric_types)
    options["buffer_type"] = get_option_value("buffer_type", mode.buffer_types)
    options["buffer_strategy"] = get_option_value("buffer_strategy", mode.buffer_strategies)
    options["granularity"] = get_option_value("granularity", mode.granularities)
    options["sampling_strategy"] = get_option_value("sampling_strategy", mode.sampling_strategies)

    values = ([value.strip()
               for value in options["optimizations"].split(",")] if len(options["optimizations"]) > 0 else [])
    for value in values:
        if value not in mode.optimizations:
            raise ValueError(f"Unknown optimization: {value}")
    options["optimizations"] = [mode.optimizations[value] for value in values]

    # Create the appropriate mode instance
    if mode_name == "default":
        return mode.Default(**options)
    elif mode_name == "mma":
        return mode.MMA(**options)
    else:
        raise ValueError(f"Unknown mode: {mode_obj}")


def _get_backend_name() -> str:
    backend = triton.runtime.driver.active.get_current_target().backend
    if backend == "cuda":
        return "nvidia"
    elif backend == "hip":
        return "amd"
    else:
        raise RuntimeError(f"Unsupported backend: {backend}")


class InstrumentationHook(Hook):
    priority: int = 0
    # It's important to note that only one instance of the instrumentation hook can be active at a time.
    active_count: int = 0
    enable_host_buffer: bool = False
    host_buffer: Optional[Any] = None
    # FIXME(fywkevin): change to a more reasonable value after we have support for periodic buffer dumping.
    profile_buffer_size: int = 1
    profile_buffer_slots: int = 2
    profile_buffer_alignment: int = 128

    def __init__(self, mode_obj: Union[None, str, mode.InstrumentationMode]):
        # Mapping of function objects to their scope ID pairs
        self.mode: mode.InstrumentationMode = _interpret_mode(mode_obj)

        self.allocator = CudaAllocator(self)
        self.current_buffer: Optional[Any] = None
        self.step_buffer_rings: Dict[int, StepBufferRing] = {}
        self.metadata_path: Dict[Any, Optional[str]] = {}

    def activate(self):
        if InstrumentationHook.active_count > 0:
            raise RuntimeError("Only one instance of the instrumentation hook can be active at a time.")

        InstrumentationHook.active_count += 1

        flags.instrumentation_on = True
        self.profile_buffer_size = triton.knobs.proton.profile_buffer_size
        self.profile_buffer_slots = triton.knobs.proton.profile_buffer_slots

        device = triton.runtime.driver.active.get_current_device()
        max_shared_mem = triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]
        backend_name = _get_backend_name()

        def to_llvmir_passes(pm):
            is_long_clk = False if mode.Optimize.CLOCK32 in self.mode.optimizations else True
            triton_proton.add_convert_proton_to_protongpu(pm, self.mode.metric_type, self.mode.sampling_strategy,
                                                          self.mode.sampling_options, self.mode.granularity,
                                                          self.mode.buffer_strategy, self.mode.buffer_type,
                                                          self.mode.buffer_size, max_shared_mem,
                                                          self.profile_buffer_size, self.profile_buffer_alignment,
                                                          self.mode.trace_mode == "kernel",
                                                          is_long_clk)
            triton_passes.common.add_cse(pm)

            if mode.Optimize.SCHED_STORES in self.mode.optimizations:
                triton_proton.add_schedule_buffer_store(pm)

            triton_proton.add_allocate_proton_shared_memory(pm)

            if mode.Optimize.SCHED_BARRIERS in self.mode.optimizations and backend_name == "amd":
                triton_proton.add_sched_barriers(pm)

        def to_llvm_passes(pm):
            if backend_name == "nvidia":
                triton_proton.add_convert_proton_nvidia_gpu_to_llvm(pm)
            elif backend_name == "amd":
                arch = triton.runtime.driver.active.utils.get_device_properties(device)["arch"].split(":")[0]
                triton_proton.add_convert_proton_amd_gpu_to_llvm(pm, arch)

        backends[backend_name].compiler.instrumentation = Instrumentation({
            "ttgpuir_to_llvmir":
            lambda pm: to_llvmir_passes(pm),
            "llvmir_to_llvm":
            lambda pm: to_llvm_passes(pm),
        })

        # Set up the profiling allocator
        set_profile_allocator(self.allocator)

        # Set the instrumentation mode
        triton.knobs.compilation.instrumentation_mode = str(self.mode)

        # Preallocate the current device's step-buffer ring up front so the
        # first profiled launch does not need to allocate device buffers while
        # a CUDA graph capture may already be in progress.
        if not InstrumentationHook.enable_host_buffer:
            self.get_step_buffer_ring()

    def deactivate(self):
        if InstrumentationHook.active_count == 0:
            return

        InstrumentationHook.active_count -= 1

        backend_name = _get_backend_name()

        # No instrumentation passes are registered anymore
        backends[backend_name].compiler.instrumentation = {}

        # No runtime instrumentation hook is active anymore
        flags.instrumentation_on = False

        # Restore the instrumentation mode
        triton.knobs.compilation.instrumentation_mode = ""

        # Reset profile allocator
        set_profile_allocator(NullAllocator())

        # Reset host memory for external processing
        InstrumentationHook.host_buffer = None

        self.current_buffer = None
        self.step_buffer_rings = {}

    def mark_step(self, stream: int) -> None:
        device = triton.runtime.driver.active.get_current_device()
        ring = self.step_buffer_rings.get(device)
        if ring is not None:
            ring.mark_step(stream)

    def get_step_buffer_ring(self) -> StepBufferRing:
        device = triton.runtime.driver.active.get_current_device()
        if device not in self.step_buffer_rings:
            self.step_buffer_rings[device] = StepBufferRing(self, self.profile_buffer_size, self.profile_buffer_slots)
        return self.step_buffer_rings[device]

    def initialize_kernel_trace_record(self, buffer: Any, stream: int) -> None:
        device = triton.runtime.driver.active.get_active_torch_device()
        device_interface = triton.runtime.driver.active.get_device_interface()
        launch_stream = device_interface.ExternalStream(stream, device=device)
        buffer_view = buffer.byte_range() if isinstance(buffer, ProfileScratchAllocation) else buffer
        with device_interface.stream(launch_stream):
            buffer_view.zero_()
            buffer_view[:8].fill_(0xFF)

    def init_handle(self, module: Any, function: Any, name: str, metadata_group: Dict[str, str], hash: str) -> None:
        if not function:
            return

        # Find the IR path in metadata
        ir_path = next((path for key, path in metadata_group.items() if key.endswith(("ttgir"))), None)
        metadata_path = next((path for key, path in metadata_group.items() if key.endswith(("json"))), None)
        self.metadata_path[function] = metadata_path

        if ir_path:
            context = triton_ir.context()
            triton_ir.load_dialects(context)
            backend_name = _get_backend_name()
            if backend_name == "nvidia":
                triton_nvidia.load_dialects(context)
            elif backend_name == "amd":
                triton_amd.load_dialects(context)
            triton_proton.load_dialects(context)
            module = triton_ir.parse_mlir_module(ir_path, context)
            module.context = context

            scope_id_names = triton_proton.get_scope_id_names(module)
            scope_id_parents = triton_proton.get_scope_id_parents(module)
            libproton.init_function_metadata(function, name, scope_id_names, scope_id_parents, metadata_path)
        else:
            raise RuntimeError(f"IR path not found in metadata for function {function}")

    def destroy_handle(self, module: Any, function: Any, name: str, metadata_group: Dict[str, str], hash: str) -> None:
        if not function:
            return
        self.metadata_path.pop(function, None)
        libproton.destroy_function_metadata(function)

    def enter(self, metadata: LazyDict) -> None:
        func = metadata.data.get("function")
        stream = metadata.data.get("stream")
        buffer = self.current_buffer
        alloc_size = 0 if buffer is None else buffer.element_size() * buffer.numel()
        data_ptr = 0 if buffer is None else buffer.data_ptr()
        libproton.enter_instrumented_op(stream, func, data_ptr, alloc_size)
        if InstrumentationHook.enable_host_buffer:
            InstrumentationHook.host_buffer = None

    def exit(self, metadata: LazyDict) -> None:
        func = metadata.data.get("function")
        stream = metadata.data.get("stream")
        buffer = self.current_buffer
        alloc_size = 0 if buffer is None else buffer.element_size() * buffer.numel()
        data_ptr = 0 if buffer is None else buffer.data_ptr()
        libproton.exit_instrumented_op(stream, func, data_ptr, alloc_size)

        if InstrumentationHook.enable_host_buffer:
            self._populate_host_buffer(func, buffer)

        self.current_buffer = None

    def _populate_host_buffer(self, function: Any, buffer: Optional[Any]) -> None:
        if function and self.metadata_path[function]:
            if buffer is None:
                return
            import torch
            import struct
            import json

            def encode_target(target: Dict[str, Any]) -> int:
                #TODO(fywkevin): also account for `arch`
                if target["backend"] == "cuda":
                    return 1
                elif target["backend"] == "hip":
                    return 2
                return 0

            alloc_size = 0 if buffer is None else buffer.element_size() * buffer.numel()
            sampled_warps = self.mode.sampling_options.strip().split(",")
            data = {}
            with open(self.metadata_path[function], 'r') as file:
                data = json.load(file)

            device_type = encode_target(data["target"])
            scratch_mem_size = data["profile_scratch_size"]
            total_unit = data["num_warps"]
            uid_num = total_unit if self.mode.sampling_strategy == triton_proton.SAMPLING_STRATEGY.NONE else len(
                sampled_warps)
            block_num = int(alloc_size / scratch_mem_size)

            # Binary trace layout:
            # +------------------+
            # |     version      |  4 bytes
            # +------------------+
            # |  header_offset   |  4 bytes
            # +------------------+
            # |   header_size    |  4 bytes
            # +------------------+
            # |  payload_offset  |  4 bytes
            # +------------------+
            # |   payload_size   |  4 bytes
            # +------------------+
            # |   device_type    |  4 bytes
            # +------------------+
            # |    block_num     |  4 bytes
            # +------------------+
            # |   total_unit     |  4 bytes
            # +------------------+
            # | scratch_mem_size |  4 bytes
            # +------------------+
            # |     uid_num      |  4 bytes
            # +------------------+
            # |                  |
            # |     uid_vec      |  uid_num * 4 bytes
            # |                  |
            # +------------------+
            # |                  |
            # |     payload      |  size_payload bytes
            # |                  |
            # +------------------+

            is_all_warps = self.mode.sampling_options == "" and self.mode.granularity == triton_proton.GRANULARITY.WARP
            if is_all_warps:
                uid_vec = [i for i in range(total_unit)]
            else:
                uid_vec = [int(i) for i in sampled_warps]

            header_size = 40 + uid_num * 4
            header_offset = 4
            payload_offset = header_size
            payload_size = alloc_size
            header_values = [
                VERSION, header_offset, header_size, payload_offset, payload_size, device_type, block_num, total_unit,
                scratch_mem_size, uid_num, *uid_vec
            ]
            header_bytes = struct.pack("I" * len(header_values), *header_values)

            InstrumentationHook.host_buffer = torch.empty(header_size + alloc_size, dtype=torch.uint8, device="cpu")
            config_portion = InstrumentationHook.host_buffer[:header_size]
            config_portion.copy_(torch.tensor(list(header_bytes), dtype=torch.uint8))
            data_portion = InstrumentationHook.host_buffer[header_size:].view_as(buffer)
            data_portion.copy_(buffer.cpu())
