from __future__ import annotations
from typing import Optional, Tuple, List, TYPE_CHECKING

from dataclasses import dataclass
from triton.runtime.jit import constexpr_function
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._core import builtin, base_type, base_value, _unwrap_if_constexpr
from triton.experimental.gluon.language._semantic import _check, _compute_tmem_reg_layout

from . import tma
from ..hopper import fence_async_shared, mbarrier
from ..ampere import async_copy, mma_v2

from triton._C.libtriton import ir
import triton._C.libtriton.gluon_ir as gluon_ir
if TYPE_CHECKING:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
    from ..._semantic import GluonSemantic

__all__ = [
    "allocate_tensor_memory",
    "async_copy",
    "fence_async_shared",
    "get_tmem_reg_layout",
    "mbarrier",
    "mma_v2",
    "tensor_memory_descriptor",
    "TensorMemoryLayout",
    "TensorMemoryScalesLayout",
    "tma",
    "_TensorMemoryLinearLayout",
]


@dataclass(frozen=True, eq=True)
class TensorMemoryLayout:
    """
    Describes the layout for tensor memory in Blackwell architecture.

    Args:
        block (Tuple[int, int]): Number of contiguous elements per row / column in a CTA.
        col_stride (int): Number of 32-bit columns to advance between logically
            adjacent columns. Packed layouts use a stride of 1. Unpacked
            layouts use ``32 / bitwidth``.
        cta_split_num (Optional[Tuple[int, int]]): CTA split factors. Defaults to None.
        two_ctas (bool): Whether the layout is for two-CTA mode. Defaults to False.
    """
    block: Tuple[int, int]
    col_stride: int
    cta_split_num: Optional[Tuple[int, int]] = None
    two_ctas: bool = False

    def __post_init__(self):
        super().__setattr__("block", _unwrap_if_constexpr(self.block))
        super().__setattr__("col_stride", _unwrap_if_constexpr(self.col_stride))
        super().__setattr__("cta_split_num", _unwrap_if_constexpr(self.cta_split_num))
        super().__setattr__("two_ctas", _unwrap_if_constexpr(self.two_ctas))
        assert len(self.block) == 2
        assert self.cta_split_num is None or len(self.cta_split_num) == 2
        assert self.col_stride >= 1 and (self.col_stride &
                                         (self.col_stride - 1)) == 0, "tensor memory col_stride must be a power of two"

    def _to_ir(self, builder):
        cta_split_num = list(self.cta_split_num) if self.cta_split_num else [1, 1]
        return builder.get_tensor_memory_layout(
            self.block,
            self.col_stride,
            cta_split_num,
            self.two_ctas,
        )

    def mangle(self) -> str:
        block_str = f"{self.block[0]}x{self.block[1]}"
        stride_str = f"C{self.col_stride}"
        cta_split_str = (f"CS{self.cta_split_num[0]}x{self.cta_split_num[1]}" if self.cta_split_num else "")
        two_ctas_str = "2CT" if self.two_ctas else ""
        return f"TL{block_str}{stride_str}{cta_split_str}{two_ctas_str}TL"

    def __hash__(self):
        return hash((self.block, self.col_stride, self.cta_split_num, self.two_ctas))


@dataclass(frozen=True, eq=True)
class TensorMemoryScalesLayout:
    """
    Describes the layout for tensor memory scales in Blackwell architecture.

    Args:
        cta_split_num (Optional[Tuple[int, int]]): CTA split factors. Defaults to None.
    """
    cta_split_num: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        super().__setattr__("cta_split_num", _unwrap_if_constexpr(self.cta_split_num))
        assert self.cta_split_num is None or len(self.cta_split_num) == 2

    def _to_ir(self, builder):
        cta_split_num = list(self.cta_split_num) if self.cta_split_num else [1, 1]
        return builder.get_tensor_memory_scales_layout(cta_split_num)

    def mangle(self) -> str:
        cta_split_str = f"CS{self.cta_split_num[0]}x{self.cta_split_num[1]}" if self.cta_split_num else ""
        return f"TLS{cta_split_str}TLS"

    def __hash__(self):
        return hash(self.cta_split_num)


@dataclass(frozen=True)
class _TensorMemoryLinearLayout:
    """
    Print-only linear layout for TMEM (row/col -> dim0/dim1).
    """
    rows: List[List[int]]
    cols: List[List[int]]
    shape: List[int]

    def _to_ir(self, builder):
        raise RuntimeError("TensorMemoryLinearLayout is print-only; IR materialization is unsupported")

    def mangle(self):
        return f"TMLL_{self.shape}_TMLL"

    def __hash__(self):
        return hash((tuple(map(tuple, self.rows)), tuple(map(tuple, self.cols)), tuple(self.shape)))


@constexpr_function
def get_tmem_reg_layout(
        element_ty,
        shape,
        layout,
        num_warps,
        instr_variant="32x32b",
        cga_layout=(),
):
    """
    Returns a DistributedLinearLayout compatible with TMEM load/store instructions.

    Args:
        element_ty (dtype): Element type stored in tensor memory.
        shape (Sequence[int]): Global tensor shape addressed by the TMEM descriptor.
        layout (TensorMemoryLayout): Tensor memory layout descriptor.
        num_warps (int): Number of warps participating in the operation.
        instr_variant (str): TMEM instruction variant (e.g. ``\"32x32b\"``).
        cga_layout (Sequence[Sequence[int]]): CGA layout bases describing CTA distribution.
    """

    def _unwrap(x):
        if isinstance(x, ttgl.constexpr):
            return _unwrap(x.value)
        if isinstance(x, list):
            return [_unwrap(i) for i in x]
        if isinstance(x, tuple):
            return tuple(_unwrap(i) for i in x)
        return x

    return _compute_tmem_reg_layout(
        _unwrap(element_ty),
        _unwrap(shape),
        _unwrap(layout),
        _unwrap(num_warps),
        _unwrap(instr_variant),
        _unwrap(cga_layout),
    )


class tensor_memory_descriptor_type(base_type):

    def __init__(self, element_ty, shape, layout, alloc_shape):
        self.element_ty = element_ty
        self.shape = shape
        self.layout = layout
        self.alloc_shape = alloc_shape
        assert isinstance(layout, TensorMemoryLayout) or isinstance(layout, TensorMemoryScalesLayout)

    def to_ir(self, builder: GluonOpBuilder) -> None:
        return builder.get_tensor_mem_desc_ty(
            self.element_ty.to_ir(builder),
            self.shape,
            self.layout._to_ir(builder),
            self.alloc_shape,
        )

    def _unflatten_ir(self, handles: List[ir.Value], cursor: int) -> Tuple[tensor_memory_descriptor, int]:
        value = tensor_memory_descriptor(handles[cursor], self.element_ty, self.shape, self.layout, self.alloc_shape)
        return value, cursor + 1

    def _flatten_ir_types(self, builder: GluonOpBuilder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def __str__(self) -> str:
        return f"tensor_memory_descriptor<{self.element_ty}, {self.shape}, {self.layout}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape and self.layout == other.layout
                and self.alloc_shape == other.alloc_shape)

    def __neq__(self, other) -> bool:
        return not (self == other)

    def mangle(self) -> str:
        shape_str = "_".join([str(s) for s in self.shape])
        return f"MD{self.element_ty.mangle()}S{shape_str}SL{self.layout.mangle()}LAS{self.alloc_shape}ASMD"


class tensor_memory_descriptor(base_value):
    """
    Represents a tensor memory descriptor handle for Tensor Core Gen5 operations.
    """

    def __init__(self, handle, element_ty, shape, layout, alloc_shape):
        self.handle = handle
        self.type = tensor_memory_descriptor_type(element_ty, shape, layout, alloc_shape)

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
    def layout(self):
        return self.type.layout

    def __str__(self) -> str:
        return str(self.type)

    @builtin
    def load(self, layout, _semantic: GluonSemantic = None) -> ttgl.tensor:
        """
        Load a tensor from tensor memory.

        Args:
            layout (DistributedLayout): Destination layout of the tensor.

        Returns:
            tensor: A distributed tensor containing the loaded data.
        """
        layout = _unwrap_if_constexpr(layout)
        ret_ty = ttgl.distributed_type(self.dtype, self.shape, layout)
        builder = _semantic.builder
        handle = builder.create_tmem_load(ret_ty.to_ir(builder), self.handle)
        return ttgl.tensor(handle, ret_ty)

    def _load_red(self, layout, red_op, abs, propagate_nan, _semantic: GluonSemantic):
        #   red_op: MIN/MAX reduction operation
        #   abs (bool): If True, reduce absolute values.
        #   propagate_nan (NONE): If ALL, propagate NaN in specified reduction operation.
        layout = _unwrap_if_constexpr(layout)
        abs_flag = _unwrap_if_constexpr(abs)
        propagate_nan = _unwrap_if_constexpr(propagate_nan)

        ret_ty = ttgl.distributed_type(self.dtype, self.shape, layout)
        builder = _semantic.builder

        result, reduced, red_layout = builder.create_tmem_load(ret_ty.to_ir(builder), self.handle, red_op, abs_flag,
                                                               propagate_nan)

        red_shape = [self.shape[0]]  # [M] for [M,N] input
        red_ty = ttgl.distributed_type(self.dtype, red_shape, red_layout)

        return (ttgl.tensor(result, ret_ty), ttgl.tensor(reduced, red_ty))

    @builtin
    def load_min(self, layout, abs=False, propagate_nan=ir.PROPAGATE_NAN.NONE, _semantic: GluonSemantic = None):
        """
        Load a tensor from tensor memory with MIN reduction along the N-dimension.

        Args:
            layout (DistributedLayout): Destination layout of the tensor.
            abs (bool): If True, reduce absolute values. Defaults to False.
            propagate_nan (PROPAGATE_NAN): If ALL, propagate NaN in the reduction operation. Defaults to NONE.

        Returns:
            tuple: A tuple containing (tensor, reduced_tensor) where tensor is the loaded data
                   and reduced_tensor is the result of MIN reduction along the N-dimension of loaded data
        """
        return self._load_red(layout, gluon_ir.TMEM_LOAD_REDUCE_MODIFIER.MIN, abs, propagate_nan, _semantic)

    @builtin
    def load_max(self, layout, abs=False, propagate_nan=ir.PROPAGATE_NAN.NONE, _semantic: GluonSemantic = None):
        """
        Load a tensor from tensor memory with MAX reduction along the N-dimension.

        Args:
            layout (DistributedLayout): Destination layout of the tensor.
            abs (bool): If True, reduce absolute values. Defaults to False.
            propagate_nan (PROPAGATE_NAN): If ALL, propagate NaN in the reduction operation. Defaults to NONE.

        Returns:
            tuple: A tuple containing (tensor, reduced_tensor) where tensor is the loaded data
                   and reduced_tensor is the result of MAX reduction along the N-dimension of loaded data.
        """
        return self._load_red(layout, gluon_ir.TMEM_LOAD_REDUCE_MODIFIER.MAX, abs, propagate_nan, _semantic)

    @builtin
    def store(self, value, pred=True, _semantic: GluonSemantic = None) -> None:
        """
        Store a tensor into tensor memory.

        Args:
            value (tensor): The tensor to store.
            pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        """
        pred = _unwrap_if_constexpr(pred)
        pred = _semantic.to_tensor(pred)
        assert value.shape == self.shape, f"source shape {value.shape} does not match destination shape {self.shape}"
        assert value.dtype == self.dtype, f"source dtype {value.dtype} does not match destination dtype {self.dtype}"
        _semantic.builder.create_tmem_store(self.handle, value.handle, pred.handle)

    @builtin
    def slice(self, start, length, _semantic: GluonSemantic = None) -> None:
        """
        Create a slice of the tensor memory descriptor along the last dimension.

        Args:
            start (int): The starting index for subslice.
            length (int): The length of the subslice.

        Returns:
            tensor_memory_descriptor: Descriptor for the subslice.
        """
        start = _unwrap_if_constexpr(start)
        length = _unwrap_if_constexpr(length)
        _check(isinstance(start, int), lambda: "start must be a constant int")
        _check(isinstance(length, int), lambda: "length must be a constant int")
        shape = self.shape[:-1] + [length]
        layout = self.type.layout
        layout = TensorMemoryLayout(
            (layout.block[0], min(layout.block[1], length)),
            layout.col_stride,
            layout.cta_split_num,
            layout.two_ctas,
        )
        ret = tensor_memory_descriptor(None, self.dtype, shape, layout, self.type.alloc_shape)
        builder = _semantic.builder
        ret.handle = builder.create_tmem_subslice(ret.type.to_ir(builder), self.handle, start)
        return ret

    @builtin
    def index(self, index, _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        """
        Create a subview of tensor memory by indexing the first dimension.

        Args:
            index (tensor): The index tensor for the subview.

        Returns:
            tensor_memory_descriptor: Descriptor for the indexed subview.
        """
        index = _semantic.to_tensor(index)
        builder = _semantic.builder
        shape = self.shape[1:]
        layout = self.layout
        ret = tensor_memory_descriptor(None, self.dtype, shape, layout, shape)
        ret.handle = builder.create_memdesc_index(ret.type.to_ir(builder), self.handle, index.handle)
        return ret

    @builtin
    def _reinterpret(self, dtype, shape, layout, _semantic: GluonSemantic = None) -> tensor_memory_descriptor:
        """
        Reinterpret tensor memory descriptor with a new dtype, shape, and layout.

        Args:
            dtype (dtype): The new data type.
            shape (Sequence[int]): The new shape.
            layout (TensorMemoryLayout): The new layout.

        Returns:
            tensor_memory_descriptor: Descriptor with updated type and layout.
        """
        dtype = _unwrap_if_constexpr(dtype)
        shape = [_unwrap_if_constexpr(s) for s in shape]
        layout = _unwrap_if_constexpr(layout)

        ty = tensor_memory_descriptor_type(dtype, shape, layout, shape)
        handle = _semantic.builder.create_memdesc_reinterpret(ty.to_ir(_semantic.builder), self.handle)
        return tensor_memory_descriptor(handle, **ty.__dict__)


@builtin
def allocate_tensor_memory(element_ty, shape, layout, value=None, _semantic=None):
    """
    Allocate tensor memory.

    Args:
        element_ty (dtype): The element data type.
        shape (Sequence[int]): The descriptor shape.
        layout (TensorMemoryLayout): The layout of the tensor memory.
        value (tensor, optional): Initial tensor to copy. Defaults to None.

    Returns:
        tensor_memory_descriptor: Descriptor for the allocated memory.
    """
    element_ty = _unwrap_if_constexpr(element_ty)
    shape = _unwrap_if_constexpr(shape)
    layout = _unwrap_if_constexpr(layout)
    value = value.handle if value is not None else None

    ty = tensor_memory_descriptor_type(element_ty, shape, layout, shape)
    builder = _semantic.builder
    handle = builder.create_tmem_alloc(ty.to_ir(builder), value)
    return tensor_memory_descriptor(handle, element_ty, shape, layout, shape)


@builtin
def tcgen05_copy(src, dst, _semantic=None):
    """
    Start an asynchronous copy from shared memory to tensor memory.

    Args:
        src (shared_memory_descriptor): Shared memory to copy from.
        dst (tensor_memory_descriptor): Tensor memory to copy to.
    """
    assert isinstance(src, ttgl.shared_memory_descriptor), "source must be a shared memory descriptor"
    assert isinstance(dst, tensor_memory_descriptor), "destination must be a tensor memory descriptor"
    _semantic.builder.create_tmem_copy(src.handle, dst.handle)


@builtin
def tcgen05_mma(a, b, acc, *, use_acc=True, pred=True, multicast=False, mbarriers=None, mbarrier_preds=None,
                _semantic=None):
    """
    Emit a 5th generation TensorCore MMA instruction.
    acc = a * b + (acc if use_acc else 0)

    Args:
        a (shared_memory_descriptor): Left hand side operand in shared memory.
        b (shared_memory_descriptor or tensor_memory_descriptor): Right hand side operand in shared or tensor memory.
        acc (tensor_memory_descriptor): Accumulator value in tensor memory (mutated).
        use_acc (bool): Whether to use the initial value of the accumulator. Defaults to True.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        multicast (bool): Whether tcgen05 commit should multicast across a CTA cluster. Defaults to False.
        mbarriers (Sequence[shared_memory_descriptor], optional): Barriers to signal when the operation is complete. If None, mma is synchronous. Defaults to None.
        mbarrier_preds (Sequence[bool], optional): Predicates for barriers. Defaults to None.
    """
    use_acc = _semantic.to_tensor(use_acc)
    pred = _semantic.to_tensor(pred)

    if mbarriers is None:
        assert mbarrier_preds is None
        mbarriers = []
        mbarrier_preds = []
    else:
        mbarriers = [bar.handle for bar in mbarriers]
        if mbarrier_preds is None:
            true = _semantic.to_tensor(True)
            mbarrier_preds = [true.handle] * len(mbarriers)
        else:
            mbarrier_preds = _semantic._convert_to_ir_values(mbarrier_preds, require_i64=False)

    multicast = _unwrap_if_constexpr(multicast)
    _semantic.builder.create_tcgen05_mma(a.handle, b.handle, acc.handle, use_acc.handle, pred.handle, mbarriers,
                                         mbarrier_preds, acc.layout.two_ctas, multicast)


@builtin
def tcgen05_mma_scaled(a, b, acc, a_scale, b_scale, a_type, b_type, *, use_acc=True, pred=True, mbarriers=None,
                       mbarrier_preds=None, _semantic=None):
    """
    Emit a 5th generation TensorCore MMA scaled instruction.
    acc = (a * a_scale) * (b * b_scale) + (acc if use_acc else 0)

    Args:
        a (shared_memory_descriptor): Left hand side operand in shared memory.
        b (shared_memory_descriptor or tensor_memory_descriptor): Right hand side operand in shared or tensor memory.
        acc (tensor_memory_descriptor): Accumulator value in tensor memory (mutated).
        a_scale (tensor): Scale factor for operand A.
        b_scale (tensor): Scale factor for operand B.
        a_type (str): Type of operand A. One of {"e2m1", "e4m3", "e5m2"}.
        b_type (str): Type of operand B. One of {"e2m1", "e4m3", "e5m2"}.
        use_acc (bool): Whether to use the initial value of the accumulator. Defaults to True.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        mbarriers (Sequence[mbarrier], optional): Barriers to signal when the operation is complete. If None, mma is synchronous. Defaults to None.
        mbarrier_preds (Sequence[bool], optional): Predicates for barriers. Defaults to None.
    """
    use_acc = _semantic.to_tensor(use_acc)
    pred = _semantic.to_tensor(pred)
    assert acc.type.layout.block[0] != 64, "tcgen05_mma_scaled does not support blockM=64"

    if mbarriers is None:
        assert mbarrier_preds is None
        mbarriers = []
        mbarrier_preds = []
    else:
        mbarriers = [bar.handle for bar in mbarriers]
        if mbarrier_preds is None:
            true = _semantic.to_tensor(True)
            mbarrier_preds = [true.handle] * len(mbarriers)
        else:
            mbarrier_preds = _semantic._convert_to_ir_values(mbarrier_preds, require_i64=False)

    allowed_formats = {"e2m1", "e4m3", "e5m2"}
    assert a_type.value in allowed_formats, f"Unsupported lhs_format: {a_type.value}"
    assert b_type.value in allowed_formats, f"Unsupported rhs_format: {b_type.value}"
    a_type = _semantic._str_to_fp_type(a_type.value)
    b_type = _semantic._str_to_fp_type(b_type.value)
    _semantic.builder.create_tcgen05_mma_scaled(a.handle, b.handle, acc.handle, a_scale.handle, b_scale.handle, a_type,
                                                b_type, use_acc.handle, pred.handle, mbarriers, mbarrier_preds)


@constexpr_function
def tcgen05_mma_barrier_count(smems, multicast):
    """
    Calculate the number of CTAs that will commit the tcgen05 MMA instruction.

    Args:
        smems (Sequence[shared_memory_descriptor]): Shared memory descriptors used in the tcgen05 instruction.
        multicast (bool): Whether the tcgen05 instruction is multicast.

    Returns:
        int: The number of CTAs that will commit the tcgen05 MMA instruction.
    """
    assert 0 <= len(smems) <= 2, "tcgen05_mma_barrier_count supports 0, 1, or 2 smem descriptors"
    if not smems or not multicast:
        return 1

    def basis_is_zero(basis):
        return all(b == 0 for b in basis)

    def num_broadcast_bits(smem):
        return sum(basis_is_zero(basis) for basis in smem.layout.cga_layout)

    if len(smems) == 1:
        return 2**num_broadcast_bits(smems[0])

    assert len(smems) == 2
    num_broadcast_bits_a = num_broadcast_bits(smems[0])
    num_broadcast_bits_b = num_broadcast_bits(smems[1])
    # Asser that for every basis, at least one of them is non-zero
    # so that the inclusion-exclusion principle below works
    # This can be generalised if needed by substracting below 2**size_intersection
    for i in range(len(smems[0].layout.cga_layout)):
        assert not basis_is_zero(smems[0].layout.cga_layout[i]) or not basis_is_zero(smems[1].layout.cga_layout[i])

    # Inclusion-exclusion
    num_cta_commits = 2**num_broadcast_bits_a + 2**num_broadcast_bits_b - 1
    return num_cta_commits


@builtin
def tcgen05_commit(barrier, pred=True, descs=(), _semantic=None):
    """
    This instruction causes the provided mbarrier to be arrived-on with a count
    of 1 when all async tcgen05 MMA and copy instructions previously issued by
    the thread are complete.

    If `descs` are provided, the commit will be multicast across the CTA cluster
    based on the shared layouts of those descriptors. This should be used when
    the inputs to the tcgen5 MMA come from TMA descriptors using multicast.

    Args:
        barrier (shared_memory_descriptor): The barrier to track completion of tcgen05 MMA and copy instructions.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        descs (Sequence[shared_memory_descriptor]): Shared memory descriptors for
            the preceding multiplication inputs. Defaults to ().
    """
    pred = _semantic.to_tensor(pred)
    descs = _unwrap_if_constexpr(descs)
    descs = [d.handle for d in descs]
    _semantic.builder.create_tcgen05_commit(barrier.handle, pred.handle, descs)
