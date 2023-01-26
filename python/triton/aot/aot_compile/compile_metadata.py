from argparse import Namespace
from dataclasses import dataclass
import inspect
import textwrap
from typing import Any, Callable, Dict, Sequence, Union
from enum import Enum

from aot_compile.static_analysis import JITStub

# import triton.language as tl
# from triton.compiler import instance_descriptor, kernel_suffix
# from triton import JITFunction


def _exists(v):
    return v is not None


class Specializations(Enum):
    """Parameter size specializations"""

    NONE = 0
    ONE = 1
    SIXTEEN = 16
    UNKNOWN = -1


def extract_source(fn: Callable):
    """
    Extract source code from a function.

    In case of executing a script from a string (like we do for the AOT compilation) inspect can't get the source code.
    In that case we pass the source to the scope of the function with __AOT_COMPILE_src
    """
    try:
        src = inspect.getsource(fn)
    except OSError:
        try:
            # TODO: make `__AOT_COMPILE_src` into something maintainable
            src = fn.__globals__["__AOT_COMPILE_src"][fn.__name__]
        except:
            raise OSError(f"Could get source code for {fn.__name__}")

    return textwrap.dedent(src)


def jit(fn):
    return FakeJITFunction(fn=fn)


# ASTGeneratingObject = JITFunction
ASTGeneratingObject = Any
""" Alias for `JITFunction` """
# TritonConstantExpr = tl.constexpr
TritonConstantExpr = Any


def _is_int_type(ann):
    """
    i1-i64, u1-u64, B are int types
    """
    if ann.startswith("i"):
        return True
    if ann.startswith("u"):
        return True

    if ann == "B":
        return True

    return False


def _valid_triton_ty_ann(ann) -> Union[str, None]:
    """
    Takes in a string for type annotation. If string is valid return the type (without pointer prefix)
    """
    if ann[0] == "*":
        ann = ann[1:]
    tys = {
        "fp8": "triton.language.float8",
        "fp16": "triton.language.float16",
        "bf16": "triton.language.bfloat16",
        "fp32": "triton.language.float32",
        "fp64": "triton.language.float64",
        "i1": "triton.language.int1",
        "i8": "triton.language.int8",
        "i16": "triton.language.int16",
        "i32": "triton.language.int32",
        "i64": "triton.language.int64",
        "u8": "triton.language.uint8",
        "u16": "triton.language.uint16",
        "u32": "triton.language.uint32",
        "u64": "triton.language.uint64",
        "B": "triton.language.int1",
    }
    if ann in tys:
        return ann
    return


class InputArgMeta:
    MISSING_TOKEN = "<ADD>"
    NONE_TOKEN = "_"


@dataclass
class Param(InputArgMeta):
    name: str
    argnum: int
    type_ann: str = InputArgMeta.MISSING_TOKEN
    """ Type annotaion that goes as input to signature param in triton.compile"""
    specialization: Specializations = Specializations.UNKNOWN

    @property
    def is_int(self):
        return _is_int_type(self.type_ann)

    def __repr__(self) -> str:
        if self.specialization == Specializations.UNKNOWN:
            spec = Param.MISSING_TOKEN
        elif self.specialization == Specializations.NONE:
            spec = Param.NONE_TOKEN
        else:
            spec = int(self.specialization.value)

        return f"{self.name}:{self.type_ann}({spec})"

    @classmethod
    def from_args(cls, name: str, type_ann: str, specialization: str, argnum: int):
        _err_msgs = [f"For {name}:"]
        ty = _valid_triton_ty_ann(type_ann)
        if not ty:
            _err_msgs.append(f"{type_ann} is not a valid Triton type annotation")

        if specialization == Param.NONE_TOKEN:
            specialization = Specializations.NONE
        else:
            try:
                specialization = Specializations(int(specialization))
            except ValueError:
                _err_msgs.append(
                    f"{specialization} is not a valid Specialization value"
                )

        if len(_err_msgs) > 1:
            raise ValueError("\n\t".join(_err_msgs))

        return cls(
            name=name,
            type_ann=type_ann,
            specialization=specialization,
            argnum=argnum,
        )

    @classmethod
    def from_str(cls, repr: str, argnums: Dict[str, int]):
        name, rest = repr.split(":")
        type_ann, rest = rest.split("(")
        raw_spec = rest.replace(")", "")
        argnum = argnums[name]
        return cls.from_args(
            name=name, type_ann=type_ann, specialization=raw_spec, argnum=argnum
        )


@dataclass
class Constant(InputArgMeta):
    name: str
    argnum: int
    val: Any = None

    def __repr__(self) -> str:
        return f"{self.name}: {self.val}"


def _pointer_convention(arg_name):

    # convention ends with _ptr
    if arg_name.endswith("_ptr"):
        return "*"

    # convention, first letter is upper means pointer
    if arg_name[0].isupper() and (not arg_name[1:] or arg_name[1:].islower()):
        return "*"

    return None


def try_infer_triton_ty(
    arg: inspect.Parameter, argnum: int
) -> Union[Param, Constant, None]:

    _make_const = lambda: Constant(
        name=arg.name,
        val=arg.default if arg.default != inspect._empty else None,
        argnum=argnum,
    )

    if isinstance(arg.annotation, TritonConstantExpr):
        return _make_const()

    if isinstance(arg.annotation, str):
        cur_ty = _valid_triton_ty_ann(arg.annotation)

        if _exists(cur_ty):
            # TODO: assuming cuda memory always 32/64 aligned. this assumption might break for other targets
            specialization = (
                Specializations.SIXTEEN if cur_ty[0] == "*" else Specializations.UNKNOWN
            )
            return Param(arg.name, cur_ty, specialization=specialization, argnum=argnum)

    cur_ty = _pointer_convention(arg.name)
    if _exists(cur_ty):
        return Param(
            arg.name,
            type_ann=f"*{Param.MISSING_TOKEN}",
            specialization=Specializations.SIXTEEN,
            argnum=argnum,
        )

    # constant convention ALL CAPS
    if arg.name.isupper():
        return _make_const()

    return


@dataclass
class AOTKernelMetadata:
    name: str
    params: Sequence[Param]
    constants: Sequence[Constant]

    def __repr__(self) -> str:
        to_str = lambda lst: ", ".join([str(x) for x in lst])

        params = to_str(self.params)
        consts = to_str(self.constants)

        return f"[{self.name}]\n\tparams:\t\t{params}\n\tconstants:\t{consts}"

    @staticmethod
    def make_dict(params: Sequence[Param], constants: Sequence[Constant]):

        argnums = [None for _ in range(len(params) + len(constants))]
        for p in params:
            argnums[p.argnum] = p.name
        for c in constants:
            argnums[c.argnum] = c.name

        return {
            "params": " ".join([str(p) for p in params]),
            "constants": {c.name: [] for c in constants},
            "arg_order": ",".join(argnums),
        }

    def to_dict(self) -> Dict:
        res = AOTKernelMetadata.make_dict(params=self.params, constants=self.constants)
        return {self.name: res}

    @classmethod
    def prase_config(cls, name: str, conf_dict: Dict) -> Sequence["AOTKernelMetadata"]:
        _err_msgs = [f"For {name}:"]
        argnums = {k: i for i, k in enumerate(conf_dict["arg_order"].split(","))}
        params = [
            Param.from_str(pstr, argnums) for pstr in conf_dict["params"].split(" ")
        ]

        prev_len = None
        const_variants = []
        for c_name, vals in conf_dict["constants"].items():
            if prev_len is None or len(vals) == prev_len:
                prev_len = len(vals)
                const_variants.append(
                    [Constant(name=c_name, val=v, argnum=argnums[c_name]) for v in vals]
                )
                continue
            _err_msgs.append(f"Not all constants have the same value count")
        if prev_len == 0:
            _err_msgs.append(f"Must provide constant values for {name}")
        if len(_err_msgs) > 1:
            raise ValueError("\n\t".join(_err_msgs))

        # NOTE: in case we produce multiple varinats of same kernel add serial number to the
        #       kernel name. need to make sure that the triton compiler gets this name as well
        variants = [
            cls(
                name=f"{name}{i}",
                params=params,
                constants=[cv[i] for cv in const_variants],
            )
            for i in range(prev_len)
        ]
        return variants

    @classmethod
    def parse_args(cls, args: Namespace):
        const_dict = vars(args)
        params = []

        for argnum, (arg_str, arg) in enumerate(zip(args.signature, ker_args)):
            # e.g.: *fp32:16
            type_ann, spec = arg_str.split(":")
            param = Param.from_args(
                name=arg, type_ann=type_ann, specialization=spec, argnum=argnum
            )
            params.append(param)

        const_dict.pop("signature")
        consts = [
            Constant(name=k, argnum=arg_names.index(k), val=v)
            for k, v in const_dict.items()
        ]

        return AOTKernelMetadata(name=kernel.__name__, params=params, constants=consts)


def infer_triton_signature(jit_fn: ASTGeneratingObject) -> str:
    # TODO: this needs to output AOTKernelMetadata and turning to dict is optional
    # TODO: (idea) make a config helper where it goes over parts that can;t be inferred and prompts user of annotation
    sig = inspect.signature(jit_fn.fn)
    params = []
    consts = []
    for argnum, arg in enumerate(sig.parameters.values()):
        cur_ty = try_infer_triton_ty(arg, argnum=argnum)
        if isinstance(cur_ty, Param):
            params.append(cur_ty)

        elif isinstance(cur_ty, Constant):
            consts.append(cur_ty)
        else:
            params.append(Param(name=arg.name, argnum=argnum))

    return AOTKernelMetadata.make_dict(params=params, constants=consts)


@dataclass
class CompileMetadata:
    arg_names: Sequence[str]
    """ Names of input arguments """
    signature: Sequence[str]
    """ Triton type annotations of function argument """
    constants: Dict[str, Union[int, JITStub]]
    specializations: instance_descriptor
    compiled_function_name: str
    """ kernel name as generetaed by compiler """
    docstr: str
    """ Represents argument types and constant values"""


def parse_signature(type_anns: Sequence[str]) -> Dict[int, str]:
    signature = {}
    for argnum, type_ann in enumerate(type_anns):
        assert _exists(
            _valid_triton_ty_ann(type_ann)
        ), f"Bad type annotaoin {type_ann} is not valid"
        signature[argnum] = type_ann.strip()
    return signature


def parse_specializations(spec_ann: Sequence[str]) -> instance_descriptor:

    # index by spec_val % 16
    div_16 = set()
    is_1 = set()

    for argnum, spec in enumerate(spec_ann):
        try:
            spec_val = int(spec)
            if spec_val == 16:
                div_16.add(argnum)
            elif spec_val == 1:
                is_1.add(argnum)
        except ValueError:
            # No specializations
            continue

    return instance_descriptor(divisible_by_16=div_16, equal_to_1=is_1)


def compilation_metadata_from_args(
    kernel: JITStub, kernel_args: Sequence[str]
) -> CompileMetadata:
    from argparse import ArgumentParser

    parser = ArgumentParser(description=f"Meta params for {kernel.__name__}")
    arg_names = kernel.arg_names
    consts = []
    ker_arg_names = []

    for arg_num, arg in enumerate(arg_names):
        if arg_num in kernel.constants:
            consts.append(arg)
        else:
            ker_arg_names.append(arg)

    sig = ",".join(ker_arg_names)
    parser.add_argument(
        "--signature",
        "-s",
        nargs=len(ker_arg_names),
        type=str,
        required=False,
        metavar=tuple(ker_arg_names),
        help=f"Provide annotations for the following (in order) {sig} e.g. *fp32:16, i32:1",
    )

    # TODO: add support for defaults
    for c in consts:
        parser.add_argument(
            f"--{c}",
            type=int,
            help="Constant value for kernel compilation",
            required=False,
        )

    args = parser.parse_args(args=kernel_args)

    signature = []
    specializations = []

    doc_str = []

    for argnum, arg_str in enumerate(args.signature):
        type_ann, spec = arg_str.split(":")
        doc_str.append(f"{ker_arg_names[arg_num]}: {arg_str}")
        signature.append(type_ann)
        specializations.append(spec)

    arg_docstr = ",".join(doc_str)

    const_dict = vars(args)
    const_dict.pop("signature")
    conts_docstr = ",".join([f"{k}: {v}" for k, v in const_dict.items()])

    specials = parse_specializations(specializations)
    function_name = f"{kernel.__name__}_{kernel_suffix(signature=ker_arg_names, specialization=specials)}"

    docstr = "\n".join([kernel.__name__, "---", arg_docstr, conts_docstr])

    return CompileMetadata(
        arg_names=ker_arg_names,
        signature=parse_signature(signature),
        constants=const_dict,
        specializations=specials,
        compiled_function_name=function_name,
        docstr=docstr,
    )


def make_compile_metadata(meta: AOTKernelMetadata) -> CompileMetadata:
    kwargs = {}
    type_anns = [p.type_ann for p in meta.params]
    kwargs["signature"] = ",".join(type_anns)

    div_16 = set()
    is_1 = set()
    for p in meta.params:
        if p.specialization == Specializations.SIXTEEN:
            div_16.add(p.argnum)
            continue

        if p.specialization == Specializations.ONE and p.is_int:
            is_1.add(p.argnum)

    # param specialization
    specials = instance_descriptor(divisible_by_16=div_16, equal_to_1=is_1)
    kwargs["configs"] = [specials]

    kwargs["constants"] = {c.name: c.val for c in meta.constants}

    function_name = (
        f"{meta.name}_{kernel_suffix(signature=type_anns, specialization=specials)}"
    )
    return CompileMetadata(compile_kwargs=kwargs, compiled_function_name=function_name)
