from __future__ import annotations

from functools import reduce
from typing import Any, Callable, TYPE_CHECKING, Union, List, Dict

if TYPE_CHECKING:
    from .language import core
    IterableType = Union[list[Any], tuple[Any, ...], core.tuple, core.tuple_type]
    ObjPath = tuple[int, ...]

TRITON_MAX_TENSOR_NUMEL = 1048576


def get_iterable_path(iterable: IterableType, path: ObjPath) -> Any:
    return reduce(lambda a, idx: a[idx], path, iterable)  # type: ignore[index]


def set_iterable_path(iterable: IterableType, path: tuple[int, ...], val: Any):
    from .language import core
    assert len(path) != 0
    prev = iterable if len(path) == 1 else get_iterable_path(iterable, path[:-1])
    assert isinstance(prev, core.tuple)
    prev._setitem(path[-1], val)


def is_iterable(x):
    from .language import core
    return isinstance(x, (list, tuple, core.tuple, core.tuple_type))


def apply_with_path(value: Any, fn: Callable[[ObjPath, Any], None], _path=None) -> None:
    if _path is None:
        _path = ()

    if is_iterable(value):
        for idx, item in enumerate(value):
            apply_with_path(item, fn, _path=(*_path, idx))
    else:
        fn(_path, value)


def find_paths_if(iterable: Union[IterableType, Any], pred: Callable[[ObjPath, Any], bool]) -> list[ObjPath]:
    # We need to use dict so that ordering is maintained, while set doesn't guarantee order
    ret: dict[ObjPath, None] = {}

    def _impl(path: tuple[int, ...], current: Any):
        if is_iterable(current):
            for idx, item in enumerate(current):
                _impl((*path, idx), item)
        elif pred(path, current):
            ret[path] = None

    _impl((), iterable)

    return list(ret.keys())


def is_power_of_two(x):
    return (x & (x - 1)) == 0


def validate_block_shape(shape: List[int]):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]")
        if not is_power_of_two(d):
            raise ValueError(f"Shape element {i} must be a power of 2")
        numel *= d

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return numel


type_canonicalisation_dict = {
    # we canonicalise all bools to be unsigned:
    "bool": "u1",
    "int1": "u1",
    "uint1": "u1",
    "i1": "u1",
    # floating-point dtypes:
    "float8e4nv": "fp8e4nv",
    "float8e5": "fp8e5",
    "float8e4b15": "fp8e4b15",
    "float8_e4m3fn": "fp8e4nv",
    "float8e4b8": "fp8e4b8",
    "float8_e4m3fnuz": "fp8e4b8",
    "float8_e5m2": "fp8e5",
    "float8e5b16": "fp8e5b16",
    "float8_e5m2fnuz": "fp8e5b16",
    "half": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float": "fp32",
    "float32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    # signed integers:
    "int8": "i8",
    "int16": "i16",
    "int": "i32",
    "int32": "i32",
    "int64": "i64",
    # unsigned integers:
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "void": "void",
}

for v in list(type_canonicalisation_dict.values()):
    type_canonicalisation_dict[v] = v


def canonicalize_dtype(dtype):
    dtype_str = str(dtype).split(".")[-1]
    return type_canonicalisation_dict[dtype_str]


def canonicalize_ptr_dtype(dtype, is_const):
    return f"{'*k' if is_const else '*'}{canonicalize_dtype(dtype)}"


BITWIDTH_DICT: Dict[str, int] = {
    **{f"u{n}": n
       for n in (1, 8, 16, 32, 64)},
    **{f"i{n}": n
       for n in (1, 8, 16, 32, 64)},
    **{f"fp{n}": n
       for n in (16, 32, 64)},
    **{f"fp8{suffix}": 8
       for suffix in ("e4nv", "e4b15", "e4b8", "e5", "e5b16")},
    "bf16": 16,
    "void": 0,
}

for k, v in type_canonicalisation_dict.items():
    BITWIDTH_DICT[k] = BITWIDTH_DICT[v]


def get_primitive_bitwidth(dtype: str) -> int:
    return BITWIDTH_DICT[dtype]


def is_namedtuple(val):
    return isinstance(val, type) and issubclass(val, tuple) and hasattr(val, "_fields")


def _tuple_create(arg, contents):
    # NamedTuples and tuples have different construction semantics. NamedTuple
    # has a constructor that takes individual arguments, while tuple takes an
    # iterable. Both have type "tuple" making it difficult to distinguish
    # between them, but only NamedTuple has "_fields" and apparently this is how
    # everyone does the check.
    return type(arg)(*contents) if hasattr(arg, "_fields") else type(arg)(contents)
