from ..state import enter_state, exit_state
from triton.compiler import LazyDict
from .hook import Hook
from triton._C.libproton import proton as libproton
from contextvars import ContextVar

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"

op_name = ContextVar("op_name", default=None)
id = ContextVar("id", default=None)


class LaunchHook(Hook):
    # Highest priority
    priority = 100
    # This is a singleton class
    _instance = None
    flops_width = [8, 16, 32, 64]
    metrics = [f"flops{width}" for width in flops_width] + ["bytes"] + ["flops"]

    def __init__(self):
        pass

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LaunchHook, cls).__new__(cls)
        return cls._instance

    def init_handle(self, module, function, name: str, metadata_group: dict, hash: str) -> None:
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
        op_name.set(metadata["name"])
        id.set(libproton.record_scope())
        libproton.enter_op(id.get(), metadata["name"])
        libproton.add_metrics(id.get(), fn_metrics)

    def exit(self, lazy_dict: LazyDict) -> None:
        libproton.exit_op(id.get(), op_name.get())
