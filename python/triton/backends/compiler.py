from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Union
from types import ModuleType


@dataclass(frozen=True)
class GPUTarget(object):
    # Target backend, e.g., cuda, hip
    backend: str
    # Target architecture, e.g., 90 (for cuda compute capability), gfx940 (for hip)
    arch: Union[int, str]
    warp_size: int


class Language(Enum):
    """The input language being compiled by the backend."""
    TRITON = 0
    GLUON = 1


class BaseBackend(metaclass=ABCMeta):
    supports_native_tensor_specialization = True
    specialize_on_32_byte_alignment = False

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
        elif desc in ("2", "4", "8", "32"):
            ret += [["tt.divisibility", int(desc)]]
        return ret

    @staticmethod
    def get_int_specialization(arg, **kwargs):
        if kwargs.get("align", False):
            if kwargs.get("specialize_on_32_byte_alignment", False) and arg % 32 == 0:
                return "32"
            for divisor in (16, 8, 4, 2):
                if arg % divisor == 0:
                    return "D" if divisor == 16 else str(divisor)
        return ""

    @staticmethod
    def get_tensor_specialization(arg, **kwargs):
        return BaseBackend.get_int_specialization(arg.data_ptr(), **kwargs)
