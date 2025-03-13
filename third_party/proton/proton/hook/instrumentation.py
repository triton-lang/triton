import functools
from typing import Dict, List, Optional, Tuple, Any

from triton._C.libtriton import ir
from triton._C.libtriton import proton as triton_proton
from triton._C.libproton import proton as libproton
from triton.compiler import LazyDict
from triton.runtime.jit import JITFunction
from triton.language import constexpr
from triton.runtime._allocation import set_profile_allocator, NullAllocator

from .hook import Hook


class CudaAllocator:

    def __init__(self, instrumentation_hook):
        self.instrumentation_hook = instrumentation_hook

    def __call__(self, size: int, alignment: int, stream: Optional[int]):
        import torch

        # Ensure proper alignment and minimum size
        alignment = max(alignment, self.instrumentation_hook.profile_buffer_alignment)
        aligned_size = (size + alignment - 1) // alignment * alignment
        aligned_size = max(aligned_size, self.instrumentation_hook.profile_buffer_size)

        # Create the buffer
        buffer = torch.empty(aligned_size, dtype=torch.uint8, device="cuda", stream=stream)
        self.instrumentation_hook.buffer = buffer.data_ptr()
        return buffer


class InstrumentationHook(Hook):
    # It's important to note that only one active session can use the instrumentation hook at a time.
    # The check is enforced by the proton library.

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
        # Set up the profiling allocator
        set_profile_allocator(self.allocator)

        original_run = JITFunction.run

        @functools.wraps(original_run)
        def instrumented_run(self, *args, **kwargs):
            kwargs["instrumentation"] = constexpr(self.mode)
            return original_run(self, *args, **kwargs)

        JITFunction.run = instrumented_run

    def deactivate(self):
        # Restore original JIT function run method
        if hasattr(JITFunction.run, "__wrapped__"):
            JITFunction.run = JITFunction.run.__wrapped__

        # Reset profile allocator
        set_profile_allocator(NullAllocator())

    def init_handle(self, function: Any, module: Any, metadata_group: Dict[str, str]) -> None:
        if function and function not in self.function_scope_ids:
            # Find the IR path in metadata
            ir_path = next((path for key, path in metadata_group.items() if key.endswith(("ttgir", "ttir"))), None)

            if ir_path:
                # Parse the MLIR module to extract scope IDs
                context = ir.context()
                ir.load_dialects(context)
                module = ir.parse_mlir_module(ir_path, context)
                module.context = context
                self.function_scope_ids[function] = triton_proton.get_scope_id_pairs(module)

        scope_id_pairs = self.function_scope_ids.get(function, [])
        libproton.init_scope_ids(function, scope_id_pairs)

    def enter(self, lazy_dict: LazyDict) -> None:
        libproton.enter_instrumented_op(lazy_dict.get("function"), self.buffer, self.profile_buffer_size)

    def exit(self, lazy_dict: LazyDict) -> None:
        libproton.exit_instrumented_op(lazy_dict.get("function"), self.buffer, self.profile_buffer_size)
        # Release the profiling buffer for recycling
        self.buffer = None
