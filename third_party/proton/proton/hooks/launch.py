from ..state import enter_state, exit_state, COMPUTE_METADATA_SCOPE_NAME
from ..metric import transform_tensor_metrics, set_metric_kernels
from triton.compiler import LazyDict
from .hook import Hook
from triton._C.libproton import proton as libproton
from contextvars import ContextVar
from numbers import Number
import re
from typing import Optional

op_name = ContextVar("op_name", default=None)
id = ContextVar("id", default=None)
enabled = ContextVar("enabled", default=False)


class LaunchHook(Hook):
    # Highest priority
    priority = 100
    flops_width = [8, 16, 32, 64]
    # Historical/derived metrics (e.g., used by viewer utilization computations).
    # Launch metadata can carry *additional* metrics; see _extract_metrics().
    metrics = [f"flops{width}" for width in flops_width] + ["bytes"] + ["flops"]

    # Reserved keys that Triton’s runtime always attaches to launch_metadata.
    # We never treat these as metrics.
    _reserved_metadata_keys = {"name", "function", "stream"}

    # LaunchHook is intended to be a process-wide singleton. HookManager dedupes
    # by identity (object instance), so we must ensure repeated LaunchHook()
    # constructions return the same instance to avoid double registration.
    _instance = None

    def configure(self, *, include: Optional[str] = None, exclude: Optional[str] = None) -> None:
        # Regexes over the compiled kernel name (metadata.data["name"]).
        self._include_pattern = include
        self._exclude_pattern = exclude
        self._include_re = re.compile(include) if include else None
        self._exclude_re = re.compile(exclude) if exclude else None

    def _matches_kernel_name(self, kernel_name: str) -> bool:
        if self._include_re is not None and self._include_re.match(kernel_name) is None:
            return False
        if self._exclude_re is not None and self._exclude_re.match(kernel_name) is not None:
            return False
        return True

    @staticmethod
    def _is_supported_metric_value(value) -> bool:
        # Supported scalar: Python/numpy number-like (bools are allowed but not very useful).
        # Supported tensor: objects with a data_ptr() method (e.g., torch.Tensor).
        if value is None:
            return False
        if hasattr(value, "data_ptr"):
            return True
        return isinstance(value, Number)

    @staticmethod
    def _extract_metrics(lazy_metadata: dict) -> dict:
        # Accept arbitrary metrics from launch_metadata while filtering out reserved fields
        # and unsupported values (e.g., objects/functions).
        return {
            k: v
            for k, v in lazy_metadata.items()
            if k not in LaunchHook._reserved_metadata_keys and LaunchHook._is_supported_metric_value(v)
        }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        # Singleton: __init__ is invoked on every construction even when __new__
        # returns an existing instance.
        if getattr(self, "_initialized", False):
            return
        # Ensure filter state is always initialized even if configure() isn't called.
        self.configure(include=None, exclude=None)
        self._initialized = True

    def init_handle(self, module, function, name: str, metadata_group: dict, hash: str) -> None:
        pass

    def activate(self):
        pass

    def deactivate(self):
        pass

    def enter(self, metadata: LazyDict) -> None:
        # Fast path: if the kernel name is already available without evaluating launch_metadata,
        # apply include/exclude filters and potentially skip metadata evaluation entirely.
        kernel_name = metadata.data.get("name")
        if not self._matches_kernel_name(kernel_name):
            enabled.set(False)
            return
        enter_state(COMPUTE_METADATA_SCOPE_NAME)
        lazy_metadata = metadata.get()
        exit_state()

        kernel_name = lazy_metadata["name"]
        # If name wasn't available (or changed), apply filters using the evaluated name.
        if not self._matches_kernel_name(kernel_name):
            enabled.set(False)
            return

        enabled.set(True)
        fn_metrics = LaunchHook._extract_metrics(lazy_metadata)
        op_name.set(kernel_name)
        id.set(libproton.record_scope())
        if fn_metrics:
            set_metric_kernels()
        scalar_metrics, tensor_metrics = transform_tensor_metrics(fn_metrics)
        libproton.enter_op(id.get(), lazy_metadata["name"])
        libproton.add_metrics(id.get(), scalar_metrics, tensor_metrics)

    def exit(self, metadata: LazyDict) -> None:
        if not enabled.get():
            return
        libproton.exit_op(id.get(), op_name.get())
