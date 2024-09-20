import os
import re
import hashlib
import subprocess

from abc import ABCMeta, abstractmethod, abstractclassmethod
from dataclasses import dataclass
from typing import Dict, Union
from types import ModuleType


# This class handles the properties for the given function parameters
# Different backends can add more properties to the common ones
class AttrsDescriptor:
    arg_properties: dict[str, list]
    property_val: dict[str, int]

    def __init__(self, params=None, args=None):
        self.arg_properties = {}
        self.property_val = {"tt.divisibility": 16, "tt.equal_to_1": 1}

        if (params is None) or (args is None):
            return

        assert (len(params) == len(args))

        # Divisibility property
        self.arg_properties["tt.divisibility"] = [
            param.num for param, arg in zip(params, args) if AttrsDescriptor.is_divisible_by_16(arg)
            and not param.do_not_specialize and not param.do_not_specialize_on_alignment
        ]

        # Equal to 1 property
        self.arg_properties["tt.equal_to_1"] = [
            param.num for param, arg in zip(params, args) if AttrsDescriptor.is_1(arg) and not param.do_not_specialize
        ]

    # Get the function attributes as a dict like:
    # {
    #   "arg0" : [(prop_name00, val00), (prop_name01, val01), ...)]}
    #   "arg1" : [(prop_name10, val10), (prop_name11, val11), ...)]}
    # }
    def get_fn_attrs(self):
        attrs = {}
        for prop_name, arg_set in self.arg_properties.items():
            for arg in arg_set:
                attrs[arg] = attrs.get(arg, []) + [(prop_name, 16)]
        return attrs

    # Return the same object, without the given attribute `attr_name`
    def erase_property(self, attr_name):
        import copy
        c = copy.deepcopy(self)
        if attr_name in c.arg_properties:
            c.arg_properties.pop(attr_name)
        return c

    def __getitem__(self, attr_name):
        if attr_name in self.arg_properties:
            return self.arg_properties[attr_name]
        return None

    def hash(self):
        key = str([sorted(x) for x in self.__dict__.values()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def to_dict(self):
        return self.arg_properties

    # Create the class from a set of hints that are passed out. So instead
    # of calling the specializations, it's the user that tells us what property
    # each argument has
    @staticmethod
    def from_hints(hints):
        attrsDescriptor = AttrsDescriptor()
        for prop_name, prop_val in attrsDescriptor.property_val.items():
            attrsDescriptor.arg_properties[prop_name] = [i for i, h in hints.items() if h == prop_val]
        return attrsDescriptor

    @staticmethod
    def is_divisible_by_16(x):
        if hasattr(x, "data_ptr"):
            return x.data_ptr() % 16 == 0
        elif isinstance(x, int):
            return x % 16 == 0
        if x is None:
            return True
        return False

    @staticmethod
    def is_equal_to_1(x):
        return True if isinstance(x, int) and not isinstance(x, bool) and x == 1 else False

    @staticmethod
    def from_dict(data):
        return AttrsDescriptor(data)

    @staticmethod
    def get_property_key(val, align):
        if align and AttrsDescriptor.is_divisible_by_16(val):
            return "D"
        if AttrsDescriptor.is_equal_to_1(val):
            return "1"
        return "N"


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

    @abstractmethod
    def get_module_map(self) -> Dict[str, ModuleType]:
        """
        Return a map of interface modules to their device-specific implementations.
        """
        raise NotImplementedError

    def get_attrs_descriptor(self, params, args):
        """
        Return an attribute descriptor: given a set of parameters and arguments
        the descriptor stores a set of compile time properties that can improve code
        generation. Different backends might benefit from different properties
        """
        return AttrsDescriptor(params, args)

    def compute_spec_key(self, arg, align):
        """
        Return the ascii key for a given argument with a given set of properties
        """
        return AttrsDescriptor.get_property_key(arg, align)
