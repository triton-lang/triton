import functools

from .state import enter_state, exit_state
from .scope import enter_scope, exit_scope

from triton._C.libtriton import ir
from triton._C.libtriton import proton as triton_proton
from triton._C.libproton import proton as libproton

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"


class TritonHook:
    from triton.compiler import LazyDict

    flops_width = [8, 16, 32, 64]
    metrics = [f"flops{width}" for width in flops_width] + ["bytes"] + ["flops"]

    @staticmethod
    def enter(lazy_dict: LazyDict) -> None:
        enter_state(COMPUTE_METADATA_SCOPE_NAME)
        metadata = lazy_dict.get()
        exit_state()
        fn_metrics = {k: metadata[k] for k in TritonHook.metrics if k in metadata}
        enter_scope(metadata["name"], triton_op=True, metrics=fn_metrics)

    @staticmethod
    def exit(lazy_dict: LazyDict) -> None:
        exit_scope(triton_op=True)


class TritonInstrumentationHook:
    from triton.compiler import LazyDict

    function_scope_ids: dict = {}
    triton_hook: TritonHook = None
    profile_buffer_size = 16 * 1024 * 1024  # 16 MB
    profile_buffer_alignment = 128

    @staticmethod
    def init_handle(function, module, metadata_group):
        if function and function not in TritonInstrumentationHook.function_scope_ids:
            ir_path = next((path for key, path in metadata_group.items() if key.endswith(("ttgir", "ttir"))), None)
            if ir_path:
                context = ir.context()
                ir.load_dialects(context)
                module = ir.parse_mlir_module(ir_path, context)
                module.context = context
                TritonInstrumentationHook.function_scope_ids[function] = triton_proton.get_scope_id_pairs(module)

    @staticmethod
    def enter(lazy_dict: LazyDict, scratch_buffer) -> None:
        function = lazy_dict.data.get("function")
        scope_id_pairs = TritonInstrumentationHook.function_scope_ids.get(function, [])
        libproton.map_scope_ids(scope_id_pairs)
        libproton.set_scratch_buffer(
            scratch_buffer,
            TritonInstrumentationHook.profile_buffer_size,
            TritonInstrumentationHook.profile_buffer_alignment,
        )
        if TritonInstrumentationHook.triton_hook:
            TritonInstrumentationHook.triton_hook.enter(lazy_dict)

    @staticmethod
    def exit(lazy_dict: LazyDict) -> None:
        libproton.unset_scope_ids()
        libproton.unset_scratch_buffer()
        if TritonInstrumentationHook.triton_hook:
            TritonInstrumentationHook.triton_hook.exit(lazy_dict)


def register_launch_hook() -> None:
    from triton.compiler import CompiledKernel

    if CompiledKernel.launch_enter_hook is None:
        CompiledKernel.launch_enter_hook = TritonHook.enter
        CompiledKernel.launch_exit_hook = TritonHook.exit
    else:
        raise RuntimeError("Triton launch hook is already registered.")


def unregister_launch_hook() -> None:
    from triton.compiler import CompiledKernel

    CompiledKernel.launch_enter_hook = None
    CompiledKernel.launch_exit_hook = None


def register_instrumentation_hook(mode: str) -> None:
    from triton.compiler import CompiledKernel

    if CompiledKernel.init_handle_hook is not None:
        raise RuntimeError("Triton instrumentation hook is already registered.")
    CompiledKernel.init_handle_hook = TritonInstrumentationHook.init_handle

    if CompiledKernel.launch_enter_hook is not None:
        TritonInstrumentationHook.triton_hook = TritonHook
        CompiledKernel.launch_enter_hook = TritonInstrumentationHook.enter
        CompiledKernel.launch_exit_hook = TritonInstrumentationHook.exit

    from triton.runtime.jit import JITFunction
    from triton.language import constexpr
    # Append the instrumentation mode to kwargs
    @functools.wraps(JITFunction.run)
    def run(self, *args, **kwargs):
        kwargs["instrumentation"] = constexpr(mode)
        return self.run(*args, **kwargs)

    JITFunction.run = run


def unregister_instrumentation_hook() -> None:
    from triton.compiler import CompiledKernel

    CompiledKernel.init_handle_hook = None
    CompiledKernel.launch_enter_hook = None
    CompiledKernel.launch_exit_hook = None
    if TritonInstrumentationHook.triton_hook:
        CompiledKernel.launch_enter_hook = TritonHook.enter
        CompiledKernel.launch_exit_hook = TritonHook.exit

    from triton.runtime.jit import JITFunction
    JITFunction.run = JITFunction.run.__wrapped__
