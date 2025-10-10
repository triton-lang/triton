from __future__ import annotations
from typing import Optional, Tuple, List, TYPE_CHECKING

from dataclasses import dataclass
from triton.runtime.jit import constexpr_function
from triton.experimental.gluon.language import _core as ttgl
from triton.experimental.gluon.language._core import builtin, base_type, base_value
from triton.experimental.gluon.language._layouts import DistributedLinearLayout, DistributedLinearLayout
from triton.experimental.gluon.language._semantic import _check

from . import tma
from ..hopper import fence_async_shared, mbarrier
from ..ampere import async_copy, mma_v2

from triton._C.libtriton import ir
if TYPE_CHECKING:
    from triton._C.libtriton.gluon_ir import GluonOpBuilder
    from ..._semantic import GluonSemantic


def _load_dialects(ctx):
    ir.load_dialects(ctx)


_load_dialects.__triton_builtin__ = True

__all__ = [
    "allocate_tensor_memory",
    "async_copy",
    "fence_async_shared",
    "get_tmem_reg_layout",
    "get_tmem_scales_reg_layout",
    "mbarrier",
    "mma_v2",
    "tensor_memory_descriptor",
    "TensorMemoryLayout",
    "tma",
]


@dataclass(frozen=True, eq=True)
class TensorMemoryLayout:
    """
    Describes the layout for tensor memory in Blackwell architecture.

    Args:
        block (Tuple[int, int]): Tiling block dimensions (M/rows, N/cols).
        col_stride (int): Number of 32-bit columns to advance between logically
            adjacent columns. Packed layouts use a stride of 1. Unpacked
            layouts use ``32 / bitwidth``.
        cta_split_num (Optional[Tuple[int, int]]): CTA split factors. Defaults to None.
    """
    block: Tuple[int, int]
    col_stride: int
    cta_split_num: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        assert len(self.block) == 2
        assert self.cta_split_num is None or len(self.cta_split_num) == 2
        assert self.col_stride >= 1 and (self.col_stride &
                                         (self.col_stride - 1)) == 0, "tensor memory col_stride must be a power of two"

    def _to_ir(self, builder):
        cta_split_num = self.cta_split_num or [1, 1]
        return builder.get_tensor_memory_layout(
            self.block,
            self.col_stride,
            cta_split_num,
        )

    def mangle(self) -> str:
        block_str = f"{self.block[0]}x{self.block[1]}"
        stride_str = f"C{self.col_stride}"
        cta_split_str = (f"CS{self.cta_split_num[0]}x{self.cta_split_num[1]}" if self.cta_split_num else "")
        return f"TL{block_str}{stride_str}{cta_split_str}TL"


@dataclass(frozen=True, eq=True)
class TensorMemoryScalesLayout:
    """
    Describes the layout for tensor memory scales in Blackwell architecture.

    Args:
        cta_split_num (Optional[Tuple[int, int]]): CTA split factors. Defaults to None.
    """
    cta_split_num: Optional[Tuple[int, int]] = None

    def __post_init__(self):
        assert self.cta_split_num is None or len(self.cta_split_num) == 2

    def _to_ir(self, builder):
        cta_split_num = self.cta_split_num or [1, 1]
        return builder.get_tensor_memory_scales_layout(cta_split_num, )

    def mangle(self) -> str:
        cta_split_str = f"CS{self.cta_split_num[0]}x{self.cta_split_num[1]}" if self.cta_split_num else ""
        return f"TLS{cta_split_str}TLS"


@constexpr_function
def get_tmem_reg_layout(
    element_ty,
    shape,
    layout,
    num_warps,
    instr_variant="32x32b",
    ctas_per_cga=None,
    cta_split_num=None,
    cta_order=None,
):
    """
    Returns a DistributedLinearLayout compatible with TMEM load/store instructions.

    Args:
        element_ty (dtype): Element type stored in tensor memory.
        shape (Sequence[int]): Global tensor shape addressed by the TMEM descriptor.
        layout (TensorMemoryLayout): Tensor memory layout descriptor.
        num_warps (int): Number of warps participating in the operation.
        instr_variant (str): TMEM instruction variant (e.g. ``\"32x32b\"``).
        ctas_per_cga (Optional[Sequence[int]]): CTA grouping along each dimension.
        cta_split_num (Optional[Sequence[int]]): CTA split factors along each dimension.
        cta_order (Optional[Sequence[int]]): CTA visitation order.
    """
    from triton._C.libtriton.gluon_ir import GluonOpBuilder

    def _unwrap(value):
        while hasattr(value, "value"):
            value = value.value
        return value

    element_ty = _unwrap(element_ty)
    layout = _unwrap(layout)
    num_warps = _unwrap(num_warps)
    instr_variant = _unwrap(instr_variant)
    shape = [_unwrap(s) for s in shape]

    if not isinstance(layout, TensorMemoryLayout):
        raise TypeError("layout must be a TensorMemoryLayout")
    if not hasattr(element_ty, "to_ir"):
        raise TypeError("element_ty must be a Triton dtype")
    if not isinstance(instr_variant, str):
        raise TypeError("instr_variant must be a string")
    if num_warps < 4 or (num_warps & (num_warps - 1)) != 0:
        raise ValueError("num_warps must be a power of two and >= 4")

    rank = len(shape)
    if rank != 2:
        raise ValueError("expected a 2D tensor")

    if ctas_per_cga is None:
        ctas_per_cga = [1] * rank
    else:
        ctas_per_cga = [_unwrap(x) for x in ctas_per_cga]
    if cta_split_num is None:
        cta_split_num = [1] * rank
    else:
        cta_split_num = [_unwrap(x) for x in cta_split_num]
    if cta_order is None:
        cta_order = list(reversed(range(rank)))
    else:
        cta_order = [_unwrap(x) for x in cta_order]

    if len(ctas_per_cga) != rank:
        raise ValueError("ctas_per_cga rank mismatch")
    if len(cta_split_num) != rank:
        raise ValueError("cta_split_num rank mismatch")
    if len(cta_order) != rank:
        raise ValueError("cta_order rank mismatch")

    context = ir.context()
    _load_dialects(context)
    builder = GluonOpBuilder(context)
    element_ty_ir = element_ty.to_ir(builder)
    layout_attr = layout._to_ir(builder)
    mem_desc_ty = builder.get_tensor_mem_desc_ty(element_ty_ir, shape, layout_attr, shape)
    cta_layout_attr = builder.get_cta_layout(ctas_per_cga, cta_split_num, cta_order)
    result = builder.get_distributed_layout_for_tmem_ldst(mem_desc_ty, instr_variant, num_warps, cta_layout_attr)
    if result is None:
        raise ValueError(f"TMEM layout '{instr_variant}' unsupported for shape {shape} and num_warps {num_warps}")
    assert isinstance(result, DistributedLinearLayout)
    return result


@constexpr_function
def get_tmem_scales_reg_layout(M, N, shape, num_warps, ctas_per_cga=None, cta_split_num=None, cta_order=None):
    """Return a linear layout that is compatible with tmem scaled layout.
    """
    assert len(shape) == 2, "expected a 2D tensor"
    assert num_warps in [4, 8], "expected 4 or 8 warps"

    # Use per-CTA shape to build the linear layout bases
    shape_per_cta = _get_shape_per_cta(shape, cta_split_num)
    M_cta, N_cta = shape_per_cta[0], shape_per_cta[1]

    # Register bases: pack 4 scales together along N; if fewer than 4, replicate.
    reg_bases = []
    i = 1
    while i < 4:
        if i >= N_cta:
            reg_bases.append([0, 0])
        else:
            size_per_thread = [1, _cdiv(N, 2)]
            warps_per_cta = [4 * min(blocks_per_tile[0], num_warp_groups)]
            warps_per_cta.append(_cdiv(num_warp_groups, warps_per_cta[0] // 4))
    else:
        cta_split_num = [_unwrap(x) for x in cta_split_num]
    if cta_order is None:
        cta_order = list(reversed(range(rank)))
    else:
        cta_order = [_unwrap(x) for x in cta_order]

    if len(ctas_per_cga) != rank:
        raise ValueError("ctas_per_cga rank mismatch")
    if len(cta_split_num) != rank:
        raise ValueError("cta_split_num rank mismatch")
    if len(cta_order) != rank:
        raise ValueError("cta_order rank mismatch")

    context = ir.context()
    _load_dialects(context)
    builder = GluonOpBuilder(context)
    element_ty_ir = element_ty.to_ir(builder)
    layout_attr = layout._to_ir(builder)
    mem_desc_ty = builder.get_tensor_mem_desc_ty(element_ty_ir, shape, layout_attr, shape)
    cta_layout_attr = builder.get_cta_layout(ctas_per_cga, cta_split_num, cta_order)
    result = builder.get_distributed_layout_for_tmem_ldst(mem_desc_ty, instr_variant, num_warps, cta_layout_attr)
    if result is None:
        raise ValueError(f"TMEM layout '{instr_variant}' unsupported for shape {shape} and num_warps {num_warps}")
    assert isinstance(result, DistributedLinearLayout)
    return result


@constexpr_function
def get_tmem_scales_reg_layout(M, N, shape, num_warps, ctas_per_cga=None, cta_split_num=None, cta_order=None):
    """Return a linear layout that is compatible with tmem scaled layout.
    """
    assert len(shape) == 2, "expected a 2D tensor"
    assert num_warps in [4, 8], "expected 4 or 8 warps"

    # Use per-CTA shape to build the linear layout bases
    shape_per_cta = _get_shape_per_cta(shape, cta_split_num)
    M_cta, N_cta = shape_per_cta[0], shape_per_cta[1]

    # Register bases: pack 4 scales together along N; if fewer than 4, replicate.
    reg_bases = []
    i = 1
    while i < 4:
        if i >= N_cta:
            reg_bases.append([0, 0])
        else:
            reg_bases.append([0, i])
        i <<= 1

    # Lane bases: distribute 32 rows of M along a warp.
    lane_bases = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]]

    # Warp bases: replicate across warps within a warpgroup by default.
    warp_bases = [[0, 0], [0, 0]]

    # Extend register bases for larger M and N beyond the initial pack.
    i = 32
    while i < M_cta:
        reg_bases.append([i, 0])
        i <<= 1

    i = 4
    while i < N_cta:
        reg_bases.append([0, i])
        i <<= 1

    # For 8 warps, distribute the last dimension on the second warpgoup.
    if num_warps == 8:
        warp_bases.append(reg_bases[-1])
        reg_bases.pop()

    # No explicit CTA mapping here; the register layout is per-CTA.
    return DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=lane_bases,
        warp_bases=warp_bases,
        block_bases=[],
        shape=shape_per_cta,
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
    def load(self, layout, _semantic: GluonSemantic) -> ttgl.tensor:
        """
        Load a tensor from tensor memory.

        Args:
            layout (DistributedLayout): Destination layout of the tensor.

        Returns:
            tensor: A distributed tensor containing the loaded data.
        """
        layout = ttgl._unwrap_if_constexpr(layout)
        ret_ty = ttgl.distributed_type(self.dtype, self.shape, layout)
        builder = _semantic.builder
        handle = builder.create_tmem_load(ret_ty.to_ir(builder), self.handle)
        return ttgl.tensor(handle, ret_ty)

    @builtin
    def store(self, value, pred=True, _semantic: GluonSemantic = None) -> None:
        """
        Store a tensor into tensor memory.

        Args:
            value (tensor): The tensor to store.
            pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
        """
        pred = ttgl._unwrap_if_constexpr(pred)
        pred = _semantic.to_tensor(pred)
        assert value.shape == self.shape, f"source shape {value.shape} does not match destination shape {self.shape}"
        assert value.dtype == self.dtype, f"source dtype {value.dtype} does not match destination dtype {self.dtype}"
        _semantic.builder.create_tmem_store(self.handle, value.handle, pred.handle)

    @builtin
    def slice(self, start, length, _semantic: GluonSemantic) -> None:
        """
        Create a slice of the tensor memory descriptor along the last dimension.

        Args:
            start (int): The starting index for subslice.
            length (int): The length of the subslice.

        Returns:
            tensor_memory_descriptor: Descriptor for the subslice.
        """
        start = ttgl._unwrap_if_constexpr(start)
        length = ttgl._unwrap_if_constexpr(length)
        _check(isinstance(start, int), lambda: "start must be a constant int")
        _check(isinstance(length, int), lambda: "length must be a constant int")
        shape = self.shape[:-1] + [length]
        layout = self.type.layout
        layout = TensorMemoryLayout(
            (layout.block[0], min(layout.block[1], length)),
            layout.col_stride,
            layout.cta_split_num,
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
        ret = tensor_memory_descriptor(None, self.dtype, shape, layout, self.type.alloc_shape)
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
        dtype = ttgl._unwrap_if_constexpr(dtype)
        shape = [ttgl._unwrap_if_constexpr(s) for s in shape]
        layout = ttgl._unwrap_if_constexpr(layout)

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
    element_ty = ttgl._unwrap_if_constexpr(element_ty)
    shape = ttgl._unwrap_if_constexpr(shape)
    layout = ttgl._unwrap_if_constexpr(layout)
    value = value.handle if value is not None else None

    ty = tensor_memory_descriptor_type(element_ty, shape, layout, shape)
    builder = _semantic.builder
    handle = builder.create_tmem_alloc(ty.to_ir(builder), value)
    return tensor_memory_descriptor(handle, element_ty, shape, layout, shape)


@builtin
def tcgen05_copy(src, dst, _semantic=None):
    """
    Start an asynchronous copy from shared memory to tensor memory.

    WARNING: The current semantics of the instruction are not well defined and
    the API will change in the future. Use at your own risk.

    Args:
        src (shared_memory_descriptor): Shared memory to copy from.
        dst (tensor_memory_descriptor): Tensor memory to copy to.
    """
    assert isinstance(src, ttgl.shared_memory_descriptor), "source must be a shared memory descriptor"
    assert isinstance(dst, tensor_memory_descriptor), "destination must be a tensor memory descriptor"
    _semantic.builder.create_tmem_copy(src.handle, dst.handle)


@builtin
def tcgen05_mma(a, b, acc, *, use_acc=True, pred=True, mbarriers=None, mbarrier_preds=None, _semantic=None):
    """
    Emit a 5th generation TensorCore MMA instruction.
    acc = a * b + (acc if use_acc else 0)

    Args:
        a (shared_memory_descriptor): Left hand side operand in shared memory.
        b (shared_memory_descriptor or tensor_memory_descriptor): Right hand side operand in shared or tensor memory.
        acc (tensor_memory_descriptor): Accumulator value in tensor memory (mutated).
        use_acc (bool): Whether to use the initial value of the accumulator. Defaults to True.
        pred (bool): Scalar predicate. Operation is skipped if predicate is False. Defaults to True.
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

    _semantic.builder.create_tcgen05_mma(a.handle, b.handle, acc.handle, use_acc.handle, pred.handle, mbarriers,
                                         mbarrier_preds)


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


@builtin
def tcgen05_commit(barrier, _semantic=None):
    """
    This instruction causes the provided mbarrier to be arrived-on with a count
    of 1 when all async tcgen05 MMA and copy instructions previously issued by
    the thread are complete.

    Args:
        barrier (shared_memory_descriptor): The barrier to track completion of tcgen05 MMA and copy instructions.
    """
    _semantic.builder.create_tcgen05_commit(barrier.handle)
