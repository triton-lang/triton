from __future__ import annotations
import inspect
import math
from typing import TypeVar, List, TYPE_CHECKING, Tuple
from functools import wraps
import warnings

if TYPE_CHECKING:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
    from ._semantic import GluonSemantic

from ._layouts import SharedLayout, DistributedLayout, BlockedLayout, DotOperandLayout, AutoLayout, CoalescedLayout
from triton._C.libtriton import ir
import triton.language.core as tl_core
from triton.language.core import (
    constexpr,
    base_value,
    base_type,
    dtype,
    block_type,  # TODO: block type with layout info
    pointer_type,
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
    static_range,
    tensor,
    tuple,
    tuple_type,
)

# We define __all__ only to appease the python linter, these are not used in
# this file but we want to import them anyway so they are importable from here.
__all__ = [
    "constexpr",
    "pointer_type",
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
    "float8e4b15",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "distributed_type",
    "shared_memory_descriptor_type",
    "static_range",
    "tuple",
    "tuple_type",
    "num_ctas",
]

T = TypeVar("T")

# TODO: split these
GLUON_BUILTIN = "__triton_builtin__"


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
    wrapper.signature = inspect.signature(fn)

    return wrapper


# Explicitly import forwarded Triton language symbols so mypy sees them.
add = builtin(tl_core.add)
associative_scan = builtin(tl_core.associative_scan)
assume = builtin(tl_core.assume)
atomic_add = builtin(tl_core.atomic_add)
atomic_and = builtin(tl_core.atomic_and)
atomic_cas = builtin(tl_core.atomic_cas)
atomic_max = builtin(tl_core.atomic_max)
atomic_min = builtin(tl_core.atomic_min)
atomic_or = builtin(tl_core.atomic_or)
atomic_xchg = builtin(tl_core.atomic_xchg)
atomic_xor = builtin(tl_core.atomic_xor)
broadcast = builtin(tl_core.broadcast)
cast = builtin(tl_core.cast)
device_assert = builtin(tl_core.device_assert)
device_print = builtin(tl_core.device_print)
expand_dims = builtin(tl_core.expand_dims)
gather = builtin(tl_core.gather)
inline_asm_elementwise = builtin(tl_core.inline_asm_elementwise)
join = builtin(tl_core.join)
load = builtin(tl_core.load)
map_elementwise = builtin(tl_core.map_elementwise)
max_constancy = builtin(tl_core.max_constancy)
max_contiguous = builtin(tl_core.max_contiguous)
maximum = builtin(tl_core.maximum)
minimum = builtin(tl_core.minimum)
mul = builtin(tl_core.mul)
multiple_of = builtin(tl_core.multiple_of)
num_programs = builtin(tl_core.num_programs)
permute = builtin(tl_core.permute)
program_id = builtin(tl_core.program_id)
reduce = builtin(tl_core.reduce)
reshape = builtin(tl_core.reshape)
split = builtin(tl_core.split)
static_assert = builtin(tl_core.static_assert)
static_print = builtin(tl_core.static_print)
store = builtin(tl_core.store)
sub = builtin(tl_core.sub)
to_tensor = builtin(tl_core.to_tensor)
where = builtin(tl_core.where)


class distributed_type(block_type):

    def __init__(self, element_ty: dtype, shape: List[int], layout):
        layout = _unwrap_if_constexpr(layout)
        shape = _unwrap_if_constexpr(shape)
        super().__init__(element_ty, shape)
        self.layout = layout
        self.name = f"<{self.shape}, {self.element_ty}, {self.layout}>"
        assert isinstance(layout, DistributedLayout), "tensor layout must be a DistributedLayout"
        if not isinstance(layout, (AutoLayout, CoalescedLayout)):
            assert len(
                shape
            ) == layout.rank, f"tensor shape and layout rank mismatch: shape={shape}, layout={layout}, shape rank={len(shape)}, layout rank={layout.rank}"

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

    def __eq__(self, other) -> bool:
        if not isinstance(other, distributed_type):
            return False
        return super().__eq__(other) and self.layout == other.layout


class shared_memory_descriptor_type(base_type):

    def __init__(self, element_ty, shape, layout, alloc_shape):
        shape = _unwrap_if_constexpr(shape)
        alloc_shape = _unwrap_if_constexpr(alloc_shape)
        layout = _unwrap_if_constexpr(layout)
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
        return f"shared_memory_descriptor<{self.element_ty}, {self.shape}, {self.layout}, {self.alloc_shape}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.layout == other.layout
                and self.alloc_shape == other.alloc_shape)

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        shape_str = "_".join([str(s) for s in self.shape])
        alloc_shape_str = "_".join([str(s) for s in self.alloc_shape])
        return f"MD{self.element_ty.mangle()}S{shape_str}SL{self.layout.mangle()}LAS{alloc_shape_str}ASMD"


class shared_memory_descriptor(base_value):
    """
    Represents a handle to a shared memory allocation in Gluon IR.
    """

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

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def layout(self):
        return self.type.layout

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def load(self, layout, _semantic: GluonSemantic = None) -> tensor:
        """
        Load a tensor from shared memory.

        Args:
            layout (DistributedLayout): The destination layout of the tensor.

        Returns:
            tensor: A Gluon tensor containing the loaded data.
        """
        layout = _unwrap_if_constexpr(layout)
        return _semantic.shared_load(self, layout)

    @builtin
    def store(self, value, _semantic: GluonSemantic = None) -> None:
        """
        Store a tensor into shared memory.

        Args:
            value (tensor): The tensor whose contents to store.
        """
        return _semantic.shared_store(self, value)

    @builtin
    def slice(self, start, length, dim=0, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        """
        Create a subview of shared memory by slicing along a given dimension.

        Args:
            start (int): The starting index of the slice.
            length (int): The length of the slice.
            dim (int): The dimension to slice (default: 0).

        Returns:
            shared_memory_descriptor: Descriptor for the sliced subview.
        """
        start = _unwrap_if_constexpr(start)
        length = _unwrap_if_constexpr(length)
        dim = _unwrap_if_constexpr(dim)
        return _semantic.memdesc_slice(self, start, length, dim)

    @builtin
    def index(self, index, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        """
        Create a subview of shared memory by indexing along the first dimension.

        Args:
            index (int): The index at which to take the subview.

        Returns:
            shared_memory_descriptor: Descriptor for the indexed subview.
        """
        index = _unwrap_if_constexpr(index)
        return _semantic.memdesc_index(self, index)

    @builtin
    def permute(self, order, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        """
        Permute the dimensions of the shared memory descriptor.

        Args:
            order (List[int]): The new ordering of dimensions.

        Returns:
            shared_memory_descriptor: Descriptor with permuted dimensions.
        """
        order = [_unwrap_if_constexpr(o) for o in order]
        return _semantic.memdesc_trans(self, order)

    @builtin
    def reshape(self, shape, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        """
        Reshape the shared memory descriptor to a new shape and layout.

        Args:
            shape (List[int]): The target shape.

        Returns:
            shared_memory_descriptor: Descriptor with the new shape and layout.
        """
        shape = [_unwrap_if_constexpr(s) for s in shape]

        return _semantic.memdesc_reshape(self, shape)

    @builtin
    def _reinterpret(self, dtype, shape, layout, _semantic: GluonSemantic = None) -> shared_memory_descriptor:
        """
        Reinterpret the shared memory descriptor as a different dtype, shape, or layout.

        Args:
            dtype (dtype): The new data type.
            shape (List[int]): The new shape.
            layout (SharedLayout): The new layout.

        Returns:
            shared_memory_descriptor: Descriptor with updated type and layout.
        """
        dtype = _unwrap_if_constexpr(dtype)
        shape = [_unwrap_if_constexpr(s) for s in shape]
        layout = _unwrap_if_constexpr(layout)

        return _semantic.memdesc_reinterpret(self, dtype, shape, layout)

    @builtin
    def _keep_alive(self, _semantic: GluonSemantic = None) -> None:
        """
        Dummy use to keep the shared memory descriptor alive.
        """
        return _semantic.shared_dealloc(self)


@builtin
def arange(start, end, layout=None, _semantic=None):
    """
    Generate a sequence tensor with values in [start, end) using a specified layout.

    Args:
        start (int): Inclusive start of the sequence.
        end (int): Exclusive end of the sequence.
        layout (DistributedLayout): The layout of the output tensor. Defaults to AutoLayout.

    Returns:
        tensor: A 1D tensor containing sequential values.
    """
    start = _unwrap_if_constexpr(start)
    end = _unwrap_if_constexpr(end)
    layout = _unwrap_if_constexpr(layout)
    return _semantic.arange(start, end, layout)


@builtin
def convert_layout(value, layout, assert_trivial=False, _semantic=None):
    """
    Convert a tensor to a different distributed layout.

    Args:
        value (tensor): The input tensor.
        layout (DistributedLayout): The target layout.
        assert_trivial (bool): If True, asserts that the conversion is trivial (no data movement).

    Returns:
        tensor: The tensor with the new layout.
    """
    layout = _unwrap_if_constexpr(layout)
    return _semantic.convert_layout(value, layout, assert_trivial)


@builtin
def full(shape, value, dtype, layout=None, _semantic=None):
    """
    Create a tensor filled with a scalar value, with specified shape, dtype, and layout.

    Args:
        shape (Sequence[int]): The shape of the tensor.
        value (int or float): The fill value.
        dtype (dtype): The data type for the tensor.
        layout (Optional[DistributedLayout]): The layout of the output tensor, defaults to AutoLayout().

    Returns:
        tensor: A tensor where every element equals value.
    """
    shape = _unwrap_shape(shape)
    value = _unwrap_if_constexpr(value)
    dtype = _unwrap_if_constexpr(dtype)
    layout = _unwrap_if_constexpr(layout)
    return _semantic.full(shape, value, dtype, layout)


@builtin
def histogram(input, num_bins, mask=None, layout=None, _semantic=None, _generator=None):
    """
    Compute a histogram of a 1D integer tensor.

    Args:
        input (tensor): 1D tensor of integer values.
        num_bins (int): Number of bins. Bins have width 1 and start at 0.
        mask (Optional[tensor]): Boolean mask to exclude elements when False.
        layout (DistributedLayout): Destination layout of the output histogram.

    Returns:
        tensor: 1D int32 tensor of length `num_bins` with the requested layout.
    """
    num_bins = _unwrap_if_constexpr(num_bins)
    layout = _unwrap_if_constexpr(layout)
    if mask is not None:
        mask = _semantic.to_tensor(mask)
    return _semantic.histogram(input, num_bins, mask, layout)


@builtin
def allocate_shared_memory(element_ty, shape, layout, value=None, _semantic=None) -> shared_memory_descriptor:
    """
    Allocate shared memory for a tensor with the given element type, shape, and layout.

    Args:
        element_ty (dtype): The element data type.
        shape (Sequence[int]): The dimensions of the shared memory.
        layout (SharedLayout): The shared memory layout.
        value (tensor, optional): Initial value to copy into shared memory.

    Returns:
        shared_memory_descriptor: Descriptor for the allocated memory.
    """
    element_ty = _unwrap_if_constexpr(element_ty)
    shape = _unwrap_if_constexpr(shape)
    shape = [_unwrap_if_constexpr(s) for s in shape]
    layout = _unwrap_if_constexpr(layout)
    return _semantic.allocate_shared(element_ty, shape, layout, value)


@builtin
def set_auto_layout(value, layout, _semantic=None):
    """
    Set a tensor with AutoLayout to a concrete layout

    Args:
        value (tensor): The input tensor.
        layout (DistribtedLayout): The target layout.

    Returns:
        tensor: The tensor with the new layout.
    """
    layout = _unwrap_if_constexpr(layout)
    return _semantic.set_auto_layout(value, layout)


@builtin
def fp4_to_fp(src, elem_type, axis, _semantic=None):
    """
    Upcast a tensor from fp4 (e2m1) to another floating point type.
    """
    axis = _unwrap_if_constexpr(axis)
    elem_type = _unwrap_if_constexpr(elem_type)
    return _semantic.fp4_to_fp(src, elem_type, axis)


@builtin
def warp_specialize(functions_and_args, worker_num_warps, worker_num_regs=None, _semantic=None, _generator=None):
    """
    Create a warp-specialized execution region, partitioning work across warps.

    This forks the current execution into a "default partition" and an arbitrary number of
    "worker partitons". The default partition is executed in the same :code:`num_warps` warps as
    the parent region, and may accept tensor arguments and return tensors. Worker partitions are
    executed in additional warps, which sit idle while executing the parent region.

    Note that calling warp_specialize recursively is not supported.

    Args:
        functions_and_args (List[Tuple[Callable, Any]]): List of functions and arguments for each partition. The first of which is the default partition.
        worker_num_warps (List[int]): Number of warps used for each worker partition.
        worker_num_regs (List[int], optional): Number of registers for each worker partition.
            If not None, will be used by backend for dynamic register reallocation.

    Returns:
        Tuple[Any, ...]: Results from the default partition.
    """
    worker_num_warps = [_unwrap_if_constexpr(w) for w in worker_num_warps]
    if worker_num_regs is not None:
        worker_num_regs = [_unwrap_if_constexpr(r) for r in worker_num_regs]
    return _semantic.warp_specialize(functions_and_args, worker_num_warps, worker_num_regs, _generator)


@builtin
def num_warps(_semantic=None, _generator=None):
    """
    Returns the number of warps that execute the current context, including in warp-specialized regions.
    """
    return _semantic.num_warps(_generator)


@builtin
def num_ctas(_semantic=None):
    """
    Returns the number of CTAs in the current kernel
    """
    return _semantic.num_ctas()


@builtin
def thread_barrier(_semantic=None):
    """
    Insert a barrier to synchronize threads within a CTA.
    """
    return _semantic.debug_barrier()


@builtin
def bank_conflicts(distr_ty, shared_ty, _semantic=None) -> int:
    """
    Count the bank conflicts per wavefront of each instruction generated when
    reading/writing the distributed tensor from/to the shared memory descriptor
    using ld.shared/st.shared instructions.

    We define a bank conflict of N to be the excess number of memory accesses that each
    wavefront needs to access the shared memory descriptor. When one uses no ld/st
    vectorization, this is equal to t he number of excess memory accesses per instruction.

    Args:
        distr_ty (distributed_type): The distributed tensor.
        shared_ty (shared_memory_descriptor_type): The shared memory descriptor.

    Returns:
        int: The number of bank conflicts.
    """
    distr_ty = _unwrap_if_constexpr(distr_ty)
    shared_ty = _unwrap_if_constexpr(shared_ty)
    return _semantic.bank_conflicts(distr_ty, shared_ty)


@builtin
def to_linear_layout(layout, shape, _semantic=None):
    layout = _unwrap_if_constexpr(layout)
    shape = _unwrap_shape(shape)
    return _semantic.to_linear_layout(layout, shape)


@builtin
def dot_fma(a, b, acc, _semantic=None):
    assert isinstance(a, tensor), "a must be a tensor"
    assert isinstance(b, tensor), "b must be a tensor"
    assert isinstance(acc, tensor), "acc must be a tensor"

    mma_layout = acc.type.layout
    assert isinstance(mma_layout, BlockedLayout), "acc must have a BlockedLayout"
    assert isinstance(a.type.layout, DotOperandLayout), "a must have a DotOperandLayout"
    assert isinstance(b.type.layout, DotOperandLayout), "b must have a DotOperandLayout"
    assert a.type.layout.parent == mma_layout, "a's parent layout must be the same as acc's layout"
    assert b.type.layout.parent == mma_layout, "b's parent layout must be the same as acc's layout"
    assert a.type.layout.operand_index == 0, "a's operand index must be 0"
    assert b.type.layout.operand_index == 1, "b's operand index must be 1"

    M, N = acc.shape
    K = a.shape[1]
    if M * N * K > 2**19:
        warnings.warn(f"Large dot FMA instruction size {M}x{N}x{K} may have slow compile times")

    handle = _semantic.dot(a, b, acc, input_precision=None, max_num_imprecise_acc=None, out_dtype=acc.dtype).handle
    return tensor(handle, acc.type)
