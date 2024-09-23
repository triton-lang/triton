import os
import re
import hashlib
import subprocess

from abc import ABCMeta, abstractmethod, abstractclassmethod
from dataclasses import dataclass
from typing import Dict, Union
from types import ModuleType


class AttrsDescriptor:
    """
    This class handles the compile-time properties for the given function
    parameters. Different backends can add more properties to the common ones.
    The class contains two fields:

    `arg_properties`: a dictionary containing the different compile-time properties for different
        parameters. I.e., the dictionary will look like:
        {
        "prop0": (0, 2, 3)
        "prop1": (0, 4, 5)
        }
        Different backend might need different properties on those paraemters to enable
        specific optimizations. The common compile time properties contained in this class
        are :
        - "tt.divisibility", i.e., is the given parameter divisible by 16
        - "tt.equal_to_1", i.e., is the given parameter an integer constant 1

    `arg_value`: a dictionary containing the value of the different compile-time properties, like:
        {
            "prop0": val0
            "prop1": val1
        }

    """
    __slots__ = ('arg_properties', 'property_val', '__dict__')

    def __init__(self, params=None, values=None):
        """
        We can initialize the AttrsDescriptor class by passing the list of params
        of the function and their `values`. The function will try to apply the properties
        to the values and save the parameters in the `arg_properties` list. If we don't pass
        either the `params` or the `values` we should initialize the class via an alternative method
        (see `from_dict` or `from_hints`)
        """
        self.property_val = {"tt.divisibility": 16, "tt.equal_to_1": 1}
        self.arg_properties = {}
        if (params is None) or (values is None):
            return

        assert (len(params) == len(values))

        # Divisibility property
        self.arg_properties["tt.divisibility"] = [
            param.num for param, arg in zip(params, values) if AttrsDescriptor.is_divisible_by_16(arg)
            and not param.do_not_specialize and not param.do_not_specialize_on_alignment
        ]

        # Equal to 1 property
        self.arg_properties["tt.equal_to_1"] = [
            param.num
            for param, arg in zip(params, values)
            if AttrsDescriptor.is_equal_to_1(arg) and not param.do_not_specialize
        ]

    def get_fn_attrs(self):
        """
        Get the function attributes as a dict like:
            {
            "arg0" : [(prop_name00, val00), (prop_name01, val01), ...)]}
            "arg1" : [(prop_name10, val10), (prop_name11, val11), ...)]}
            }
        """
        attrs = {}
        for prop_name, arg_set in self.arg_properties.items():
            prop_val = self.property_val[prop_name]
            for arg in arg_set:
                attrs[arg] = attrs.get(arg, []) + [(prop_name, prop_val)]
        return attrs

    def filter_out_property(self, attr_name):
        """ Return the same object, without the given attribute `attr_name`"""
        import copy
        c = copy.deepcopy(self)
        if attr_name in c.arg_properties:
            c.arg_properties.pop(attr_name)
        return c

    def __getitem__(self, attr_name):
        if attr_name in self.arg_properties:
            return self.arg_properties[attr_name]
        return []

    def hash(self):
        key = str([sorted(x) for x in self.__dict__.values()])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def to_dict(self):
        return self.arg_properties

    @staticmethod
    def from_hints(hints: list[tuple[int, int]]):
        """
        Create the class from a set of hints that are passed in. So, instead
        of deducing the properties from a list of paramaters and values, the user
        can pass in a list of `hints=[(param_index, val)]` and if `val` matches
        one of the values of the properties (e.g., `prop_val[prop0]`), then we insert
        `param_index` into the correct list (e.g., in `arg_properties[prop0]`)
        """
        attrsDescriptor = AttrsDescriptor()
        for prop_name, prop_val in attrsDescriptor.property_val.items():
            attrsDescriptor.arg_properties[prop_name] = [i for i, h in hints.items() if h == prop_val]
        return attrsDescriptor

    @staticmethod
    def is_divisible_by_16(x):
        """ Return if the argument is a multiple of 16"""
        if hasattr(x, "data_ptr"):
            return x.data_ptr() % 16 == 0
        elif isinstance(x, int):
            return x % 16 == 0
        if x is None:
            return True
        return False

    @staticmethod
    def is_equal_to_1(x):
        """ Return if the argument is a constant 1"""
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
        Return a map of interface modules to their device-specific implementations
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
