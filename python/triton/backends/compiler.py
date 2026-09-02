from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping
from typing import Any, Dict, Optional, Union
from types import ModuleType


@dataclass(frozen=True)
class GPUTarget(object):
    """A Triton compilation target.

    ``warp_size`` used to be mandatory because CUDA and HIP were the only
    in-tree targets.  It is intentionally optional now: backends such as
    Gaudi execute logical Triton programs on an index space and a wide vector
    machine rather than a SIMT warp.  The historical name remains part of the
    public API for source and cache compatibility.
    """

    # Target backend, e.g., cuda, hip, gaudi
    backend: str
    # Target architecture, e.g., 90 (CUDA), gfx940 (HIP), gaudi2 (Gaudi)
    arch: Union[int, str]
    warp_size: Optional[int] = None

    def __post_init__(self):
        if not self.backend:
            raise ValueError("target backend must be non-empty")
        if self.warp_size is not None:
            try:
                if int(self.warp_size) <= 0:
                    raise ValueError("warp_size must be positive when specified")
            except (TypeError, ValueError) as exc:
                raise ValueError("warp_size must be an integer when specified") from exc

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GPUTarget":
        """Restore a target from version-tolerant JSON metadata."""
        return cls(value["backend"], value["arch"], value.get("warp_size"))


# New code may use the hardware-neutral spelling.  Keep GPUTarget as the
# canonical class so existing imports, isinstance checks, and serialized cache
# metadata continue to work.
Target = GPUTarget


class Language(Enum):
    """The input language being compiled by the backend."""
    TRITON = 0
    GLUON = 1


class BaseBackend(metaclass=ABCMeta):
    supports_native_tensor_specialization = True

    def __init__(self, target: GPUTarget) -> None:
        self.target = target
        assert self.supports_target(target)

    def normalize_options(self, options: dict) -> dict:
        """Flatten target-specific options before backend parsing.

        ``triton.Config(backend_options={...})`` gives non-SIMT backends a
        namespace without adding every target knob to Triton's public launch
        signature.  Top-level launch options win so explicit call-site values
        remain authoritative.
        """
        normalized = dict(options)
        backend_options = normalized.pop("backend_options", None)
        if backend_options is None:
            return normalized
        if not isinstance(backend_options, Mapping):
            raise TypeError("backend_options must be a mapping")
        return {**backend_options, **normalized}

    def get_launch_options(self, options: object) -> dict:
        """Return concise backend options exposed to JIT/profiler hooks."""
        common = (
            "num_warps",
            "num_ctas",
            "num_stages",
            "enable_fp_fusion",
            "launch_cooperative_grid",
        )
        return {name: getattr(options, name) for name in common if hasattr(options, name)}

    def validate_kernel_resources(self, metadata, device_properties: dict, *, n_max_threads=None) -> None:
        """Validate target resources before and after loading a binary.

        CUDA/HIP retain the historical shared/tensor-memory/thread checks.
        Backends with a different execution model override this method.
        """
        from triton.runtime.autotuner import OutOfResources

        shared = getattr(metadata, "shared", 0)
        max_shared = device_properties.get("max_shared_mem")
        if max_shared is not None and shared > max_shared:
            raise OutOfResources(shared, max_shared, "shared memory")

        tmem_size = getattr(metadata, "tmem_size", None)
        if tmem_size is not None:
            # Blackwell tensor memory is currently not reported by the driver.
            max_tmem_size = 576 if metadata.target.arch == 107 else 512
            if tmem_size > max_tmem_size:
                raise OutOfResources(tmem_size, max_tmem_size, "tensor memory")

        warp_size = metadata.target.warp_size
        num_warps = getattr(metadata, "num_warps", None)
        if n_max_threads is not None and warp_size is not None and num_warps is not None:
            required_threads = num_warps * warp_size
            if required_threads > n_max_threads:
                raise OutOfResources(required_threads, n_max_threads, "threads")

    @staticmethod
    @abstractmethod
    def supports_target(target: GPUTarget):
        raise NotImplementedError

    @abstractmethod
    def hash(self) -> str:
        """Returns a unique identifier for this backend"""
        raise NotImplementedError

    @abstractmethod
    def parse_options(self, options: dict) -> object:
        """
        Converts an `options` dictionary into an arbitrary object and returns it.
        This function may contain target-specific heuristics and check the legality of the provided options
        """
        raise NotImplementedError

    @abstractmethod
    def add_stages(self, stages: dict, options: object) -> None:
        """
        Populates `stages` dictionary with entries of the form:
        ir_name [str] => Function[(src: str, metadata: dict) -> str|bytes]
        The value of each entry may populate a `metadata` dictionary.
        Stages will be run sequentially (in inseriton order) and can communicate using `metadata`.
        All stages are expected to return a `str` object, except for the last stage which returns
        a `bytes` object for execution by the launcher.
        """
        raise NotImplementedError

    @abstractmethod
    def load_dialects(self, context):
        """
        Load additional MLIR dialects into the provided `context`
        """
        raise NotImplementedError

    @abstractmethod
    def get_module_map(self) -> Dict[str, ModuleType]:
        """
        Return a map of interface modules to their device-specific implementations
        """
        raise NotImplementedError

    @staticmethod
    def parse_attr(desc):
        assert isinstance(desc, str)
        ret = []
        if "D" in desc:
            ret += [["tt.divisibility", 16]]
        return ret

    @staticmethod
    def get_int_specialization(arg, **kwargs):
        if arg % 16 == 0 and kwargs.get("align", False):
            return "D"
        return ""

    @staticmethod
    def get_tensor_specialization(arg, **kwargs):
        if arg.data_ptr() % 16 == 0 and kwargs.get("align", False):
            return "D"
        return ""
