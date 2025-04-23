import functools
from typing import Dict, List, Optional, Tuple, Union, Any

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
from .. import mode


class CudaAllocator:

    def __init__(self, instrumentation_hook):
        self.instrumentation_hook = instrumentation_hook

    def __call__(self, size: int, alignment: int, stream: Optional[int]):
        # Ensure proper alignment and minimum size
        # `alignment` and `size` are specified by the instrumentation engine to specify the minimum required
        # alignment and size of the buffer.
        # Then, we make it match the default profile buffer alignment and size here, taking the maximum of the two.
        alignment = max(alignment, self.instrumentation_hook.profile_buffer_alignment)
        aligned_size = (size + alignment - 1) // alignment * alignment
        aligned_size = max(aligned_size, self.instrumentation_hook.profile_buffer_size)

        # Create the buffer
        import torch
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
        "metric_type": opts.get("metric_type", "cycle"),
        "buffer_type": opts.get("buffer_type", "shared"),
        "buffer_strategy": opts.get("buffer_strategy", "circular"),
        "buffer_size": int(opts.get("buffer_size", "0")),
        "granularity": opts.get("granularity", "warp"),
        "sampling_strategy": opts.get("sampling_strategy", "none"),
        "sampling_options": opts.get("sampling_options", ""),
    }

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

    # Create the appropriate mode instance
    if mode_name == "default":
        return mode.Default(**options)
    elif mode_name == "mma":
        return mode.MMA(**options)
    else:
        raise ValueError(f"Unknown mode: {mode_obj}")


class InstrumentationHook(Hook):
    # It's important to note that only one instance of the instrumentation hook can be active at a time.
    active_count: int = 0
    enable_host_buffer: bool = False
    host_buffer: Optional[Any] = None
    profile_buffer_size: int = 16 * 1024 * 1024
    profile_buffer_alignment: int = 128

    def __init__(self, mode_obj: Union[None, str, mode.InstrumentationMode]):
        # Mapping of function objects to their scope ID pairs
        self.function_scope_ids: Dict[Any, List[Tuple[int, int]]] = {}
        self.mode: mode.InstrumentationMode = _interpret_mode(mode_obj)

        self.allocator = CudaAllocator(self)
        self.buffer = None

    def activate(self):
        if InstrumentationHook.active_count > 0:
            raise RuntimeError("Only one instance of the instrumentation hook can be active at a time.")

        InstrumentationHook.active_count += 1

        set_instrumentation_on()

        backend = triton.runtime.driver.active.get_current_target().backend
        device = triton.runtime.driver.active.get_current_device()
        max_shared_mem = triton.runtime.driver.active.utils.get_device_properties(device)["max_shared_mem"]

        def to_llvmir_passes(pm, backend_pass):
            triton_proton.add_allocate_proton_global_scratch_buffer(pm)
            backend_pass(pm)

        def to_ttgpuir_passes(pm):
            triton_proton.add_convert_proton_to_protongpu(pm, self.mode.metric_type, self.mode.sampling_strategy,
                                                          self.mode.sampling_options, self.mode.granularity,
                                                          self.mode.buffer_strategy, self.mode.buffer_type,
                                                          self.mode.buffer_size, max_shared_mem,
                                                          self.profile_buffer_size, self.profile_buffer_alignment)

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
            kwargs["instrumentation_mode"] = str(mode)
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
        if InstrumentationHook.enable_host_buffer:
            InstrumentationHook.host_buffer = None
        # Reset the buffer reference
        self.buffer = None

    def init_handle(self, function: Any, module: Any, metadata_group: Dict[str, str]) -> None:
        if function and function not in self.function_scope_ids:
            # Find the IR path in metadata
            ir_path = next((path for key, path in metadata_group.items() if key.endswith(("ttgir", "ttir"))), None)

            if ir_path:
                context = ir.context()
                ir.load_dialects(context)
                triton_proton.load_dialects(context)
                module = ir.parse_mlir_module(ir_path, context)
                module.context = context
                self.function_scope_ids[function] = triton_proton.get_scope_id_pairs(module)

        scope_id_pairs = self.function_scope_ids.get(function, [])
        libproton.init_scope_ids(function, scope_id_pairs)

    def _data_ptr(self) -> int:
        return 0 if self.buffer is None else self.buffer.data_ptr()

    def enter(self, lazy_dict: LazyDict) -> None:
        func = lazy_dict.data.get("function")
        libproton.enter_instrumented_op(func, self._data_ptr(), self.profile_buffer_size)
        if InstrumentationHook.enable_host_buffer:
            InstrumentationHook.host_buffer = None

    def exit(self, lazy_dict: LazyDict) -> None:
        func = lazy_dict.data.get("function")
        libproton.exit_instrumented_op(func, self._data_ptr(), self.profile_buffer_size)

        if InstrumentationHook.enable_host_buffer:
            # copy profiling buffer to CPU for external processing
            InstrumentationHook.host_buffer = self.buffer.cpu().clone()
