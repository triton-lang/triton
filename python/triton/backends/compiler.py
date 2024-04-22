import os
import re
import subprocess

from abc import ABCMeta, abstractmethod, abstractclassmethod
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class GPUTarget(object):
    # Target backend, e.g., cuda, hip
    backend: str
    # Target architecture, e.g., 90 (for cuda compute capability), gfx940 (for hip)
    arch: Union[int, str]
    warp_size: int


class BaseBackend(metaclass=ABCMeta):

    def __init__(self, target: GPUTarget) -> None:
        self.target = target
        assert self.supports_target(target)

    @staticmethod
    def _path_to_binary(binary: str):
        base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
        paths = [
            os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
            os.path.join(base_dir, "third_party", "cuda", "bin", binary),
        ]
        for p in paths:
            bin = p.split(" ")[0]
            if os.path.exists(bin) and os.path.isfile(bin):
                result = subprocess.check_output([bin, "--version"], stderr=subprocess.STDOUT)
                if result is not None:
                    version = re.search(r".*release (\d+\.\d+).*", result.decode("utf-8"), flags=re.MULTILINE)
                    if version is not None:
                        return p, version.group(1)
        raise RuntimeError(f"Cannot find {binary}")

    @abstractclassmethod
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
