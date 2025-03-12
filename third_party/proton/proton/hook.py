from .state import enter_state, exit_state
from .scope import enter_scope, exit_scope
from triton.compiler import CompiledKernel, LazyDict
from triton._C.libproton import proton as libproton
from typing import Dict, Optional

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"


class TritonHook:
    sessions = Dict[int, bool]
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


def activate_launch_hook(session: Optional[int] = None) -> None:
    if session and session not in TritonHook.sessions:
        return

    if libproton.get_num_active_session() > 0:
        raise ValueError("Only one session can be activated with launch hook.")

    if CompiledKernel.launch_enter_hook is None:
        CompiledKernel.launch_enter_hook = TritonHook.enter
        CompiledKernel.launch_exit_hook = TritonHook.exit
    else:
        raise RuntimeError("Triton launch hook is already registered.")


def deactivate_launch_hook(session: int) -> None:
    if session not in TritonHook.sessions:
        return

    CompiledKernel.launch_enter_hook = None
    CompiledKernel.launch_exit_hook = None


def register_launch_hook(session: int) -> None:
    TritonHook.sessions[session] = True
    activate_launch_hook(session)


def unregister_launch_hook(session: Optional[int] = None) -> None:
    if session is None:
        TritonHook.sessions.clear()
    elif session in TritonHook.sessions:
        del TritonHook.sessions[session]
    deactivate_launch_hook(session)
