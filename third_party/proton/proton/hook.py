from .state import enter_state, exit_state
from .scope import enter_scope, exit_scope
from triton import knobs
from triton.compiler import LazyDict

from triton._C.libtriton import ir
from triton._C.libtriton import proton as triton_proton
from triton._C.libproton import proton as libproton

COMPUTE_METADATA_SCOPE_NAME = "__proton_launch_metadata"


class TritonHook:
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
    if knobs.runtime.launch_enter_hook is None:
        knobs.runtime.launch_enter_hook = TritonHook.enter
        knobs.runtime.launch_exit_hook = TritonHook.exit


def unregister_launch_hook() -> None:
    if knobs.runtime.launch_enter_hook == TritonHook.enter:
        knobs.runtime.launch_enter_hook = None
        knobs.runtime.launch_exit_hook = None


class TritonInitHandleHook:
    function_scope_ids: dict = {}
    triton_hook: TritonHook = None

    @staticmethod
    def map_scope_ids(function, module, metadata_group) -> None:
        if function and function not in TritonInitHandleHook.function_scope_ids:
            ir_path = next((path for key, path in metadata_group.items() if key.endswith(("ttgir", "ttir"))), None)
            if ir_path:
                context = ir.context()
                ir.load_dialects(context)
                module = ir.parse_mlir_module(ir_path, context)
                module.context = context
                TritonInitHandleHook.function_scope_ids[function] = triton_proton.get_scope_id_pairs(module)

    @staticmethod
    def enter(lazy_dict: LazyDict) -> None:
        function = lazy_dict.data.get("function")
        scope_id_pairs = TritonInitHandleHook.function_scope_ids.get(function, [])
        libproton.map_scope_ids(scope_id_pairs)
        if TritonInitHandleHook.triton_hook:
            TritonInitHandleHook.triton_hook.enter(lazy_dict)

    @staticmethod
    def exit(lazy_dict: LazyDict) -> None:
        libproton.unmap_scope_ids()
        if TritonInitHandleHook.triton_hook:
            TritonInitHandleHook.triton_hook.exit(lazy_dict)


def register_init_handle_hook() -> None:
    if knobs.runtime.init_handle_hook is not None:
        raise RuntimeError("Triton init handle hook is already registered.")
    knobs.runtime.init_handle_hook = TritonInitHandleHook.map_scope_ids


def unregister_init_handle_hook() -> None:
    if knobs.runtime.init_handle_hook is None:
        raise RuntimeError("Triton init handle hook is not registered.")
    knobs.runtime.init_handle_hook = None
