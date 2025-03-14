from ..state import enter_state, exit_state
from ..scope import enter_scope, exit_scope
from triton.compiler import LazyDict
from .hook import Hook

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"


class LaunchHook(Hook):
    flops_width = [8, 16, 32, 64]
    metrics = [f"flops{width}" for width in flops_width] + ["bytes"] + ["flops"]

    def __init__(self):
        pass

    def init_handle(self, function, module, metadata_group):
        pass

    def activate(self):
        pass

    def deactivate(self):
        pass

    def enter(self, lazy_dict: LazyDict) -> None:
        enter_state(COMPUTE_METADATA_SCOPE_NAME)
        metadata = lazy_dict.get()
        exit_state()
        fn_metrics = {k: metadata[k] for k in LaunchHook.metrics if k in metadata}
        enter_scope(metadata["name"], triton_op=True, metrics=fn_metrics)

    def exit(self, lazy_dict: LazyDict) -> None:
        exit_scope(triton_op=True)
