import functools
from typing import Dict, List, Optional, Tuple, Any

from triton._C.libtriton import ir
from triton._C.libtriton import proton as triton_proton
from triton._C.libproton import proton as libproton
from triton.compiler import CompiledKernel, LazyDict
from triton.runtime.jit import JITFunction
from triton.language import constexpr
from triton.runtime._allocation import set_profile_allocator, NullAllocator

from .hook import TritonHook


class TritonInstrumentationHook:
    # Mapping of function objects to their scope ID pairs
    function_scope_ids: Dict[Any, List[Tuple[int, int]]] = {}

    # Reference to the basic Triton hook for delegation
    triton_hook: Optional[TritonHook] = None

    # Default buffer configuration
    profile_buffer_size = 16 * 1024 * 1024  # 16 MB
    profile_buffer_alignment = 128

    @staticmethod
    def init_handle(function: Any, module: Any, metadata_group: Dict[str, str]) -> None:
        if not function or function in TritonInstrumentationHook.function_scope_ids:
            return

        # Find the IR path in metadata
        ir_path = next((path for key, path in metadata_group.items() if key.endswith(("ttgir", "ttir"))), None)

        if ir_path:
            # Parse the MLIR module to extract scope IDs
            context = ir.context()
            ir.load_dialects(context)
            module = ir.parse_mlir_module(ir_path, context)
            module.context = context
            TritonInstrumentationHook.function_scope_ids[function] = triton_proton.get_scope_id_pairs(module)

    @staticmethod
    def enter(lazy_dict: LazyDict) -> None:
        # Set up scope ID mapping for the kernel
        function = lazy_dict.data.get("function")
        scope_id_pairs = TritonInstrumentationHook.function_scope_ids.get(function, [])
        libproton.map_scope_ids(scope_id_pairs)

        # Delegate to the base hook if available
        if TritonInstrumentationHook.triton_hook:
            TritonInstrumentationHook.triton_hook.enter(lazy_dict)

    @staticmethod
    def exit(lazy_dict: LazyDict) -> None:
        # Clean up resources
        libproton.unset_scope_ids()
        libproton.unset_scratch_buffer()

        # Delegate to the base hook if available
        if TritonInstrumentationHook.triton_hook:
            TritonInstrumentationHook.triton_hook.exit(lazy_dict)


class CudaAllocator:

    def __call__(self, size: int, alignment: int, stream: Optional[int]):
        import torch

        # Ensure proper alignment and minimum size
        alignment = max(alignment, TritonInstrumentationHook.profile_buffer_alignment)
        aligned_size = (size + alignment - 1) // alignment * alignment
        aligned_size = max(aligned_size, TritonInstrumentationHook.profile_buffer_size)

        # Create the buffer
        buffer = torch.empty(aligned_size, dtype=torch.uint8, device="cuda", stream=stream)
        libproton.set_profile_buffer(buffer.data_ptr(), size, alignment)
        return buffer


def register_instrumentation_hook(mode: str) -> None:
    # Register init handle hook
    if CompiledKernel.init_handle_hook is not None:
        raise RuntimeError("Triton instrumentation hook is already registered.")
    CompiledKernel.init_handle_hook = TritonInstrumentationHook.init_handle

    # Register launch hooks
    if CompiledKernel.launch_enter_hook is not None:
        TritonInstrumentationHook.triton_hook = TritonHook
        CompiledKernel.launch_enter_hook = TritonInstrumentationHook.enter
        CompiledKernel.launch_exit_hook = TritonInstrumentationHook.exit

    # Set up JIT function instrumentation

    # Monkey patch JITFunction.run to include instrumentation mode
    original_run = JITFunction.run

    @functools.wraps(original_run)
    def instrumented_run(self, *args, **kwargs):
        kwargs["instrumentation"] = constexpr(mode)
        return original_run(self, *args, **kwargs)

    JITFunction.run = instrumented_run

    # Set up the profiling allocator
    set_profile_allocator(CudaAllocator())


def unregister_instrumentation_hook() -> None:
    # Clean up hook references
    CompiledKernel.init_handle_hook = None

    # Restore original hooks if needed
    if TritonInstrumentationHook.triton_hook:
        CompiledKernel.launch_enter_hook = TritonHook.enter
        CompiledKernel.launch_exit_hook = TritonHook.exit
    else:
        CompiledKernel.launch_enter_hook = None
        CompiledKernel.launch_exit_hook = None

    # Restore original JIT function run method
    if hasattr(JITFunction.run, "__wrapped__"):
        JITFunction.run = JITFunction.run.__wrapped__

    # Reset profile allocator
    set_profile_allocator(NullAllocator())


def config_profile_buffer(size: int, alignment: int) -> None:
    TritonInstrumentationHook.profile_buffer_size = size
    TritonInstrumentationHook.profile_buffer_alignment = alignment
