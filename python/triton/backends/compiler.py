from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union, Optional
from types import ModuleType


class CUDADeviceVariant(str, Enum):
    H100 = "h100"
    H200 = "h200"


class CUDATargetAlias(str, Enum):
    """Hardware-specific names that compile for an existing CUDA architecture."""

    CUDA_90_H200 = ("cuda-90-h200", 90, CUDADeviceVariant.H200)
    SM90_H200 = ("sm90-h200", 90, CUDADeviceVariant.H200)

    compute_capability: int
    device_variant: CUDADeviceVariant

    def __new__(cls, value: str, compute_capability: int, device_variant: CUDADeviceVariant):
        alias = str.__new__(cls, value)
        alias._value_ = value
        alias.compute_capability = compute_capability
        alias.device_variant = device_variant
        return alias

    @classmethod
    def parse(cls, value: object) -> Optional["CUDATargetAlias"]:
        try:
            return cls(value)
        except ValueError:
            return None


@dataclass(frozen=True)
class GPUTarget(object):
    # Target backend, e.g., cuda, hip
    backend: str
    # Target architecture, e.g., 90 (for cuda compute capability), gfx940 (for hip)
    arch: Union[int, str]
    warp_size: int
    device_variant: Optional[CUDADeviceVariant] = None

    def __post_init__(self):
        if self.backend == "cuda":
            variant = CUDADeviceVariant(self.device_variant) if self.device_variant is not None else None
            alias = CUDATargetAlias.parse(self.arch)
            if alias is not None:
                if variant is not None and variant != alias.device_variant:
                    raise ValueError("CUDA target alias conflicts with device_variant")
                object.__setattr__(self, "arch", alias.compute_capability)
                variant = alias.device_variant
            if variant is not None and int(self.arch) != 90:
                raise ValueError("H100 and H200 device variants require CUDA arch 90")
            object.__setattr__(self, "device_variant", variant)


class Language(Enum):
    """The input language being compiled by the backend."""
    TRITON = 0
    GLUON = 1


class BaseBackend(metaclass=ABCMeta):
    supports_native_tensor_specialization = True

    def __init__(self, target: GPUTarget) -> None:
        self.target = target
        assert self.supports_target(target)

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
    def add_stages(self, stages: dict, options: object, language: Language) -> None:
        """
        Populates `stages` dictionary with entries of the form:
        ir_name [str] => Function[(src: str, metadata: dict) -> str|bytes]
        The value of each entry may populate a `metadata` dictionary.
        Stages will be run sequentially (in inseriton order) and can communicate using `metadata`.
        All stages are expected to return a `str` object, except for the last stage which returns
        a `bytes` object for execution by the launcher.
        `language` is the frontend that produced `src` (e.g. `Language.TRITON` or `Language.GLUON`).
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
