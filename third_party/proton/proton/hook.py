from .state import enter_state, exit_state
from .scope import enter_scope, exit_scope

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
