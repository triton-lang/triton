from .state import enter_state, exit_state
from .scope import enter_scope, exit_scope

from triton._C.libtriton import ir, proton

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"


class TritonHook:
    from triton.compiler import LazyDict

    flops_width = [8, 16, 32, 64]
    metrics = [f"flops{width}" for width in flops_width] + \
        ["bytes"] + ["flops"]

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


class TritonInitHandleHook:
    function_scope_ids: dict = {}

    @staticmethod
    def map_scope_ids(function, module, metadata_group):
        if function and function not in TritonInitHandleHook.function_scope_ids:
            ir_path = None
            if "ttgir" in metadata_group:
                ir_path = metadata_group["ttgir"]
            elif "ttir" in metadata_group:
                ir_path = metadata_group["ttir"]
            if ir_path:
                context = ir.context()
                module = ir.parse_mlir_module(ir_path, context)
                module.context = context
                scope_id_pairs = proton.get_scope_id_pairs(module)


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


def register_init_handle_hook() -> None:
    from triton.compiler import CompiledKernel
    if CompiledKernel.init_handle_hook is not None:
        raise RuntimeError("Triton init handle hook is already registered.")
    CompiledKernel.init_handle_hook = TritonInitHandleHook.map_scope_ids


def unregister_init_handle_hook() -> None:
    from triton.compiler import CompiledKernel
    CompiledKernel.init_handle_hook = None
