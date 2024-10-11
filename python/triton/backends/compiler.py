import os
import re
import hashlib
import subprocess

from abc import ABCMeta, abstractmethod, abstractclassmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from types import ModuleType

# Table that associates strings to AttrsDescriptor (sub)classes.
# In this way we can dynamically select the correct class
# constructor
_descriptor_table = {}


def register_descriptor(cls):
    """
    Register a descriptor into the descriptor table
    """
    _descriptor_table[cls.__name__] = cls
    return cls


@register_descriptor
class AttrsDescriptor:
    """
    This class handles compile-time properties for specific function parameters.

    Different backends can add more properties to the common ones. The class
    contains two fields:

    `arg_properties`: a dictionary containing the different compile-time properties for different
        parameters. I.e., the dictionary is a map from property names to parameter indices
        {
        "prop0": (0, 2, 3)
        "prop1": (0, 4, 5)
        }
        Different backends might need different properties on those paraemters to enable
        specific optimizations. The common compile time properties contained in this class
        are :
        - "tt.divisibility", i.e., is the given parameter divisible by 16
        - "tt.equal_to_1", i.e., is the given parameter an integer constant 1

    `property_values`: a dictionary containing the value of the different compile-time properties, like:
        {
            "prop0": val0
            "prop1": val1
        }

    `constant_properties`: a set containing the properties that can be used to determine if a parameter is constant

    """
    __slots__ = ('divisibility_16', 'equal_to_1', 'arg_properties', 'property_values', 'constant_properties')

    def __init__(self, params=None, values=None):
        """
        Initialize the compile-time properties

        We can initialize the AttrsDescriptor class by passing the list of params
        of the function and their `values`. The function will try to apply the properties
        to the values and save the parameters in the `arg_properties` list. If we don't pass
        either the `params` or the `values` we should initialize the class via an alternative method
        (see `from_dict` or `from_hints`)
        """
        # Default initialization
        self.arg_properties = {}
        self.property_values = {}
        self.constant_properties = set()

        self._add_common_properties(params, values)
        self._add_backend_properties(params, values)
        self._init_slots()

    def _add_common_properties(self, params, values):
        """ Add common compile-time properties """
        self.property_values["tt.divisibility"] = 16
        self.property_values["tt.equal_to"] = 1
        self.constant_properties.add("tt.equal_to")

        if (params is None) or (values is None):
            return

        # Compile properties deduction
        assert (len(params) == len(values))

        # Divisibility property
        self.arg_properties["tt.divisibility"] = [
            param.num for param, arg in zip(params, values) if AttrsDescriptor.is_divisible_by_16(arg)
            and not param.do_not_specialize and not param.do_not_specialize_on_alignment
        ]

        # Equal to 1 property
        self.arg_properties["tt.equal_to"] = [
            param.num
            for param, arg in zip(params, values)
            if AttrsDescriptor.is_equal_to_1(arg) and not param.do_not_specialize
        ]

    def _add_backend_properties(self, params=None, values=None):
        """ This method is for different subclasses to implement their own compile-time properties """
        pass

    def _init_slots(self):
        """ Initialize the slots of this class """
        for name, val in self.arg_properties.items():
            setattr(self, name.removeprefix('tt.') + '_' + str(self.property_values[name]), val)

    def get_fn_attrs(self) -> Dict:
        """
        Get the function attributes as a dictionary.

        The returned dictionary will look like :
            {
            "arg0" : [(prop_name00, val00), (prop_name01, val01), ...)]}
            "arg1" : [(prop_name10, val10), (prop_name11, val11), ...)]}
            }
        """
        attrs = {}
        for prop_name, arg_set in self.arg_properties.items():
            prop_val = self.property_values[prop_name]
            for arg in arg_set:
                attrs[arg] = attrs.get(arg, []) + [(prop_name, prop_val)]
        return attrs

    def get_constants(self) -> Dict:
        """ Return a mapping of constant parameters to their values """
        constants = {}
        for prop_name in self.constant_properties:
            for p in self.arg_properties.get(prop_name, []):
                constants[p] = self.property_values[prop_name]
        return constants

    def filter_out_constants(self):
        """ Return the same object, without properties marked as constants"""
        import copy
        c = copy.deepcopy(self)
        for prop_name in c.constant_properties:
            c.arg_properties.pop(prop_name, None)
            c.property_values.pop(prop_name, None)
        c.constant_properties = {}
        return c

    def hash(self):
        values = [sorted(self.arg_properties.values())]
        values += [sorted(self.property_values.values())]
        values += [sorted(self.constant_properties)]
        key = str(values)
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def to_dict(self):
        """
        Store the fields of this class in a serializable dictionary
        """
        # We need to only store the `arg_properties` field. To initialize the
        # other fields we relay on the class type. We store it as a string in
        # the dictionary so that we can use it to invoke the appropriate
        # (sub)class constructor in the `from_dict` method.
        return {"arg_properties": self.arg_properties, "cls": type(self).__name__}

    @staticmethod
    def from_dict(data):
        """
        Create the object from a serializable dictionary
        """
        attrs_descriptor = _descriptor_table[data["cls"]]()
        for prop_name, param_ids in data["arg_properties"].items():
            attrs_descriptor.arg_properties[prop_name] = param_ids
        attrs_descriptor._init_slots()
        return attrs_descriptor

    @classmethod
    def from_hints(cls, hints: List[Tuple[int, int]]):
        """
        Create the class from a set of hints that are passed in.

        Instead of deducing the properties from a list of paramaters and values,
        the user can pass in a list of `hints=[(param_index, val)]` and if `val`
        matches one of the values of the properties (e.g., `prop_val[prop0]`),
        then we insert `param_index` into the correct list (e.g., in
        `arg_properties[prop0]`)
        """
        attrs_descriptor = cls()
        for prop_name, prop_val in attrs_descriptor.property_values.items():
            attrs_descriptor.arg_properties[prop_name] = [i for i, h in hints.items() if h == prop_val]
        attrs_descriptor._init_slots()
        return attrs_descriptor

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
    def get_property_key(val, align):
        if align and AttrsDescriptor.is_divisible_by_16(val):
            return "D"
        if AttrsDescriptor.is_equal_to_1(val):
            return "1"
        return "N"

    def __repr__(self):
        return f"AttrsDescriptor.from_dict({self.to_dict()})"


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
