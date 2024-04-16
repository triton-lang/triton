from .scope import enter_scope, exit_scope
from triton.compiler import CompiledKernel, LazyDict

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"


class TritonHook:
    metrics = ["flops8", "flops16", "flops32", "flops64", "bytes"]

    @staticmethod
    def enter(metadata: LazyDict) -> None:
        enter_scope(COMPUTE_METADATA_SCOPE_NAME)
        metadata = metadata.get()
        exit_scope()
        fn_metrics = {k: metadata[k] for k in TritonHook.metrics if k in metadata}
        enter_scope(metadata["name"], triton_op=True, metrics=fn_metrics)

    @staticmethod
    def exit(metadata: LazyDict) -> None:
        exit_scope(triton_op=True)


def register_triton_hook() -> None:
    if CompiledKernel.launch_enter_hook is None:
        CompiledKernel.launch_enter_hook = TritonHook.enter
        CompiledKernel.launch_exit_hook = TritonHook.exit


def unregister_triton_hook() -> None:
    if CompiledKernel.launch_enter_hook == TritonHook.enter:
        CompiledKernel.launch_enter_hook = None
        CompiledKernel.launch_exit_hook = None
