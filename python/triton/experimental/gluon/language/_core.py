from __future__ import annotations
from typing import TypeVar, List, TYPE_CHECKING, Tuple
from functools import wraps

if TYPE_CHECKING:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
    from ._semantic import GluonSemantic

from ._layouts import SharedLayout, DistributedLayout
from triton._C.libtriton import ir
import triton.language.core as tl_core
from triton.language.core import (
    constexpr,
    base_value,
    base_type,
    dtype,
    block_type,  # TODO: block type with layout info
    pointer_type,
    tuple_type,
    void,
    int1,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    uint64,
    float8e5,
    float8e5b16,
    float8e4nv,
    float8e4b8,
    float8e4b15,
    float16,
    bfloat16,
    float32,
    float64,
    _unwrap_if_constexpr,
    _unwrap_shape,
    tensor,
)

_IMPORT_FROM_TRITON: List[str] = [
    "expand_dims",  # NOQA: F822
    "load",  # NOQA: F822
    "program_id",  # NOQA: F822
    "reduce",  # NOQA: F822
    "static_assert",  # NOQA: F822
    "store",  # NOQA: F822
    "to_tensor",  # NOQA: F822
    "where",  # NOQA: F822
    "maximum",  # NOQA: F822
    "minimum",  # NOQA: F822
    "permute",
]

__all__ = [
    "constexpr",
    "base_value",
    "base_type",
    "dtype",
    "block_type",
    "pointer_type",
    "tuple_type",
    "void",
    "int1",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float8e5",
    "float8e5b16",
    "float8e4nv",
    "float8e4b8",
    "float8e4b8",
    "float8e4b15",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "_unwrap_if_constexpr",
    "tensor",
    "arange",
    "full",
    "convert_layout",
    "allocate_shared_memory",
    "shared_memory_descriptor",
    "warp_specialize",
    *_IMPORT_FROM_TRITON,
]

T = TypeVar("T")

# TODO: split these
GLUON_BUILTIN = "__triton_builtin__"


class distributed_type(block_type):

    def __init__(self, element_ty: dtype, shape: List[int], layout):
        super().__init__(element_ty, shape)
        self.layout = layout
        self.name = f"<{self.shape}, {self.element_ty}, {self.layout}>"
        assert isinstance(layout, DistributedLayout)

    def to_ir(self, builder: ir.builder) -> ir.type:
        elem_ty = self.element_ty.to_ir(builder)
        layout = self.layout._to_ir(builder)
        return builder.get_distributed_ty(elem_ty, self.shape, layout)

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = "_".join(map(str, self.shape))
        layout = self.layout.mangle()
        return f"{elt}S{shape}SL{layout}L"

    def with_element_ty(self, scalar_ty: dtype) -> block_type:
        return distributed_type(scalar_ty, self.shape, self.layout)


def builtin(fn: T) -> T:
    """Mark a function as a builtin."""
    assert callable(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if "_semantic" not in kwargs or kwargs["_semantic"] is None:
            raise ValueError("Did you forget to add @triton.gluon.jit ? "
                             "(`_semantic` argument must be provided outside of JIT functions.)")
        return fn(*args, **kwargs)

    setattr(wrapper, GLUON_BUILTIN, True)

    return wrapper


class shared_memory_descriptor_type(base_type):

    def __init__(self, element_ty, shape, layout, alloc_shape):
        self.element_ty = element_ty
        self.shape = shape
        self.layout = layout
        self.alloc_shape = alloc_shape
        assert isinstance(layout, SharedLayout)

    def to_ir(self, builder: GluonOpBuilder) -> None:
        return builder.get_shared_mem_desc_ty(
            self.element_ty.to_ir(builder),
            self.shape,
            self.layout._to_ir(builder),
            self.alloc_shape,
        )

    def _unflatten_ir(self, handles: List[ir.Value], cursor: int) -> Tuple[shared_memory_descriptor, int]:
        value = shared_memory_descriptor(handles[cursor], self.element_ty, self.shape, self.layout, self.alloc_shape)
        return value, cursor + 1

    def _flatten_ir_types(self, builder: GluonOpBuilder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def __str__(self) -> str:
        return f"shared_memory_descriptor<{self.element_ty}, {self.shape}, {self.layout}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.layout == other.layout
                and self.alloc_shape == other.alloc_shape)

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        shape_str = "_".join([str(s) for s in self.shape])
        return f"MD{self.element_ty.mangle()}S{shape_str}SL{self.layout.mangle()}LAS{self.alloc_shape}ASMD"


class shared_memory_descriptor(base_value):

    def __init__(self, handle, element_ty, shape, layout, alloc_shape):
        self.handle = handle
        self.type = shared_memory_descriptor_type(element_ty, shape, layout, alloc_shape)

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)

    @property
    def dtype(self):
        return self.type.element_ty

    @property
    def shape(self):
        return self.type.shape

    @property
    def rank(self):
        return len(self.shape)

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def load(self, layout, _semantic: GluonSemantic) -> tensor:
        layout = _unwrap_if_constexpr(layout)
        return _semantic.shared_load(self, layout)

    @builtin
    def store(self, value, _semantic: GluonSemantic) -> None:
        return _semantic.shared_store(self, value)

    @builtin
    def split(self, offset, size, dim=None, layout=None, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        if layout is None:
            layout = self.type.layout
        if dim is None:
            dim = 0

        offset = _unwrap_if_constexpr(offset)
        size = _unwrap_if_constexpr(size)
        dim = _unwrap_if_constexpr(dim)
        layout = _unwrap_if_constexpr(layout)

        return _semantic.memdesc_split(self, offset, size, dim, layout)

    @builtin
    def subslice(self, index, shape=None, layout=None, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        if layout is None:
            layout = self.type.layout
        if shape is None:
            shape = self.shape[1:]

        index = _unwrap_if_constexpr(index)
        shape = [_unwrap_if_constexpr(s) for s in shape]
        layout = _unwrap_if_constexpr(layout)

        return _semantic.memdesc_slice(self, index, shape, layout)

    @builtin
    def permute(self, order, layout, _semantic: GluonSemantic) -> shared_memory_descriptor:
        order = [_unwrap_if_constexpr(o) for o in order]
        layout = _unwrap_if_constexpr(layout)

        return _semantic.memdesc_trans(self, order, layout)

    @builtin
    def reshape(self, shape, layout, _semantic: GluonSemantic) -> shared_memory_descriptor:
        shape = [_unwrap_if_constexpr(s) for s in shape]
        layout = _unwrap_if_constexpr(layout)

        return _semantic.memdesc_reshape(self, shape, layout)

    @builtin
    def _reinterpret(self, dtype, shape, layout, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        dtype = _unwrap_if_constexpr(dtype)
        shape = [_unwrap_if_constexpr(s) for s in shape]
        layout = _unwrap_if_constexpr(layout)

        return _semantic.memdesc_reinterpret(self, dtype, shape, layout)

    @builtin
    def _keep_alive(self, _semantic: GluonSemantic = None) -> None:
        return _semantic.shared_dealloc(self)


for name in _IMPORT_FROM_TRITON:
    fn = getattr(tl_core, name)
    globals()[name] = builtin(fn)


@builtin
def arange(start, end, layout, _semantic=None):
    start = _unwrap_if_constexpr(start)
    end = _unwrap_if_constexpr(end)
    layout = _unwrap_if_constexpr(layout)
    return _semantic.arange(start, end, layout)


@builtin
def convert_layout(value, layout, _semantic=None):
    layout = _unwrap_if_constexpr(layout)
    return _semantic.convert_layout(value, layout)


@builtin
def full(shape, value, dtype, layout, _semantic=None):
    shape = _unwrap_shape(shape)
    value = _unwrap_if_constexpr(value)
    dtype = _unwrap_if_constexpr(dtype)
    layout = _unwrap_if_constexpr(layout)
    return _semantic.full(shape, value, dtype, layout)


@builtin
def allocate_shared_memory(element_ty, shape, layout, value=None, _semantic=None):
    element_ty = _unwrap_if_constexpr(element_ty)
    shape = _unwrap_if_constexpr(shape)
    shape = [_unwrap_if_constexpr(s) for s in shape]
    layout = _unwrap_if_constexpr(layout)
    return _semantic.allocate_shared(element_ty, shape, layout, value)


@builtin
def warp_specialize(args, default_partition, worker_partitions, worker_num_warps, worker_num_regs,  #
                    _semantic=None, _generator=None):
    worker_num_warps = [_unwrap_if_constexpr(w) for w in worker_num_warps]
    worker_num_regs = [_unwrap_if_constexpr(r) for r in worker_num_regs]
    return _semantic.warp_specialize(args, default_partition, worker_partitions, worker_num_warps,  #
                                     worker_num_regs, _generator)
