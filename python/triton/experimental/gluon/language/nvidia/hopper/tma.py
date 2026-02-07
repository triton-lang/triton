from __future__ import annotations
from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from triton.language.core import base_type, base_value
import triton.experimental.gluon.language._core as ttgl
from triton.experimental.gluon.language._layouts import NVMMASharedLayout
from triton.experimental.gluon.language._core import builtin, _unwrap_if_constexpr

if TYPE_CHECKING:
    from triton._C import ir

__all__ = [
    "async_copy_global_to_shared",
    "async_copy_global_to_shared_im2col",
    "async_copy_shared_to_global",
    "store_wait",
]


@dataclass(eq=True)
class _tensor_descriptor_type_base(base_type):
    """Base class for tensor descriptor types (tiled and im2col)."""
    block_type: ttgl.block_type
    shape_type: ttgl.tuple_type
    strides_type: ttgl.tuple_type
    layout: NVMMASharedLayout

    # Subclasses must override these
    _type_name: str = ""
    _mangle_prefix: str = ""

    def __str__(self) -> str:
        return f"{self._type_name}<{self.block_type}, {self.layout}>"

    @property
    def nbytes_per_cta(self) -> int:
        cga_layout = self.layout.cga_layout
        if len(cga_layout) == 0:
            return self.block_type.nbytes
        num_cta_splits = 2**sum(any(x != 0 for x in basis) for basis in cga_layout)
        return self.block_type.nbytes // num_cta_splits

    def _to_ir(self, builder: ir.builder) -> ir.type:
        raise NotImplementedError

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        raise NotImplementedError

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self._to_ir(builder))
        self.shape_type._flatten_ir_types(builder, out)
        self.strides_type._flatten_ir_types(builder, out)

    def mangle(self) -> str:
        return f"{self._mangle_prefix}{self.block_type.mangle()}_{self.layout.mangle()}{self._mangle_prefix}"


@dataclass(eq=True)
class tensor_descriptor_type(_tensor_descriptor_type_base):
    """Type for tiled tensor descriptors."""
    _type_name: str = "tensor_descriptor"
    _mangle_prefix: str = "TD"

    def _to_ir(self, builder: ir.builder) -> ir.type:
        is_signed = self.block_type.element_ty.is_int_signed()
        return builder.get_tensor_descriptor_layout_type(self.block_type.to_ir(builder), is_signed,
                                                         self.layout._to_ir(builder))

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        value = tensor_descriptor(handle, shape, strides, self.block_type, layout=self.layout)
        return value, cursor


@dataclass(eq=True)
class tensor_descriptor_im2col_type(_tensor_descriptor_type_base):
    """Type for im2col tensor descriptors (convolution-friendly access patterns)."""
    _type_name: str = "tensor_descriptor_im2col"
    _mangle_prefix: str = "TDI"

    def _to_ir(self, builder: ir.builder) -> ir.type:
        is_signed = self.block_type.element_ty.is_int_signed()
        return builder.get_tensor_descriptor_im2col_layout_type(self.block_type.to_ir(builder), is_signed,
                                                                self.layout._to_ir(builder))

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[base_value, int]:
        handle = handles[cursor]
        cursor += 1
        shape, cursor = self.shape_type._unflatten_ir(handles, cursor)
        strides, cursor = self.strides_type._unflatten_ir(handles, cursor)
        value = tensor_descriptor_im2col(handle, shape, strides, self.block_type, layout=self.layout)
        return value, cursor


class _tensor_descriptor_value_base(base_value):

    def __init__(self, handle, shape: List[ttgl.tensor], strides: List[ttgl.tensor], block_type: ttgl.block_type,
                 layout: NVMMASharedLayout, type_cls):
        self.handle = handle
        self.shape = ttgl.tuple(shape)
        self.strides = ttgl.tuple(strides)
        self.type = type_cls(block_type, shape_type=self.shape.type, strides_type=self.strides.type, layout=layout)

    def _flatten_ir(self, handles: List[ir.value]) -> None:
        handles.append(self.handle)
        self.shape._flatten_ir(handles)
        self.strides._flatten_ir(handles)

    @property
    def nbytes_per_cta(self):
        return self.type.nbytes_per_cta

    @property
    def block_type(self):
        return self.type.block_type

    @property
    def block_shape(self):
        return self.type.block_type.shape

    @property
    def dtype(self):
        return self.type.block_type.element_ty

    @property
    def layout(self):
        return self.type.layout


class tensor_descriptor(_tensor_descriptor_value_base):

    def __init__(self, handle, shape: List[ttgl.tensor], strides: List[ttgl.tensor], block_type: ttgl.block_type,
                 layout: NVMMASharedLayout):
        super().__init__(handle, shape, strides, block_type, layout, tensor_descriptor_type)


class tensor_descriptor_im2col(_tensor_descriptor_value_base):

    def __init__(self, handle, shape: List[ttgl.tensor], strides: List[ttgl.tensor], block_type: ttgl.block_type,
                 layout: NVMMASharedLayout):
        super().__init__(handle, shape, strides, block_type, layout, tensor_descriptor_im2col_type)


def _emit_alignment_check(desc, coord, fn_name: str, arg_name: str, _semantic=None):
    coord = list(coord)[-1]
    align_bytes = 16
    if desc.layout.fp4_padded:
        align_bytes = 64
    dtype = desc.dtype
    assert dtype.primitive_bitwidth % 8 == 0, f"unexpected sub-byte dtype {dtype}"
    elem_bytes = dtype.primitive_bitwidth // 8
    align = align_bytes // elem_bytes

    align_val = ttgl.to_tensor(align, _semantic=_semantic)
    zero = ttgl.to_tensor(0, _semantic=_semantic)

    coord = ttgl.to_tensor(coord, _semantic=_semantic)
    rem = coord.__mod__(align_val, _semantic=_semantic)
    is_zero = rem.__eq__(zero, _semantic=_semantic)

    fp4_padded = "with fp4_padded=True " if desc.layout.fp4_padded else ""
    ttgl.device_assert(
        is_zero, f"{fn_name} {fp4_padded}{arg_name} must be {align_bytes}-byte aligned, "
        f"i.e. a multiple of {align} for dtype={dtype.codegen_name()}", _semantic=_semantic)


def _convert_im2col_offsets(offsets, _semantic):
    offsets_ir = []
    for offset in offsets:
        offset = _unwrap_if_constexpr(offset)
        if isinstance(offset, int):
            offsets_ir.append(_semantic.builder.get_int16(offset))
        elif hasattr(offset, "handle"):
            offsets_ir.append(offset.handle)
        else:
            raise ValueError(f"Unsupported offset type: {type(offset)}")
    return offsets_ir


@builtin
def async_copy_global_to_shared(tensor_desc, coord, barrier, result, pred=True, multicast=False, _semantic=None):
    """
    Copy data from global memory to shared memory using TMA.

    Args:
        tensor_desc: Tensor descriptor (tiled)
        coord: Coordinates in the source tensor
        barrier: Barrier for synchronization
        result: Destination memory descriptor
        pred: Predicate for conditional execution
        multicast: Enable multicast
    """
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, coord, "async_copy_global_to_shared", "innermost coordinate",
                              _semantic=_semantic)

    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    pred = _semantic.to_tensor(pred)
    multicast = _unwrap_if_constexpr(multicast)

    _semantic.builder.create_async_tma_copy_global_to_local(
        tensor_desc.handle,
        coord,
        barrier.handle,
        result.handle,
        pred.handle,
        multicast,
        None,
    )


@builtin
def async_copy_global_to_shared_im2col(tensor_desc, coord, offsets, barrier, result, pred=True, multicast=False,
                                       _semantic=None):
    """
    Copy data from global memory to shared memory using TMA in im2col mode.

    Args:
        tensor_desc: Tensor descriptor (im2col)
        coord: Coordinates in the source tensor
        offsets: Im2col offsets (must be i16 values)
            - For 3D tensors: 1 offset
            - For 4D tensors: 2 offsets
            - For 5D tensors: 3 offsets
        barrier: Barrier for synchronization
        result: Destination memory descriptor
        pred: Predicate for conditional execution
        multicast: Enable multicast
    """
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, coord, "async_copy_global_to_shared_im2col", "innermost coordinate",
                              _semantic=_semantic)

    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    pred = _semantic.to_tensor(pred)
    multicast = _unwrap_if_constexpr(multicast)
    offsets_ir = _convert_im2col_offsets(offsets, _semantic)

    _semantic.builder.create_async_tma_copy_global_to_local(
        tensor_desc.handle,
        coord,
        barrier.handle,
        result.handle,
        pred.handle,
        multicast,
        offsets_ir,
    )


@builtin
def async_copy_shared_to_global(tensor_desc, coord, src, _semantic=None):
    if _semantic.builder.options.enable_iisan:
        _emit_alignment_check(tensor_desc, coord, "async_copy_shared_to_global", "innermost coordinate",
                              _semantic=_semantic)
    coord = _semantic._convert_to_ir_values(coord, require_i64=False)
    _semantic.builder.create_async_tma_copy_local_to_global(tensor_desc.handle, coord, src.handle)


@builtin
def store_wait(pendings, _semantic=None):
    pendings = _unwrap_if_constexpr(pendings)
    _semantic.builder.create_async_tma_store_wait(pendings)


@builtin
def make_tensor_descriptor(
    base: ttgl.tensor,
    shape: List[ttgl.tensor],
    strides: List[ttgl.tensor],
    block_shape: List[ttgl.constexpr],
    layout: NVMMASharedLayout,
    padding_option="zero",
    _semantic=None,
) -> tensor_descriptor:
    padding_option = _unwrap_if_constexpr(padding_option)
    block_shape = _unwrap_if_constexpr(block_shape)

    ndim = len(shape)
    if not (1 <= ndim <= 5):
        raise ValueError(f"Expected 1 <= ndim <= 5 but got {ndim} dimensions")
    if len(strides) != ndim:
        raise ValueError(f"Expected {ndim} strides but got {len(strides)}")
    if len(block_shape) != ndim:
        raise ValueError(f"Expected block_shape to have {ndim} dimensions but got {len(block_shape)}")
    assert isinstance(base.dtype, ttgl.pointer_type)
    elem_size = base.dtype.element_ty.primitive_bitwidth // 8
    contig_dim_size = ttgl._unwrap_if_constexpr(block_shape[-1])
    if contig_dim_size * elem_size < 16:
        raise ValueError(
            f"Descriptor block shape must have at least 16 bytes in the last dimension, but got {contig_dim_size} * {elem_size} = {contig_dim_size * elem_size} bytes"
        )

    last_stride = ttgl._unwrap_if_constexpr(strides[-1])
    if last_stride != 1:
        raise ValueError(f"Tensor descriptor last dim must be 1 but got {last_stride}")

    shape = [_semantic.make_scalar(x, ttgl.int32) for x in shape]
    strides = [_semantic.make_scalar(ttgl._unwrap_if_constexpr(x), ttgl.int64) for x in strides]

    # Check whether `block_shape` is static
    block_shape = ttgl._unwrap_shape(block_shape)

    assert isinstance(base.type, ttgl.pointer_type)
    block_type = ttgl.block_type(base.type.element_ty, block_shape)
    base_handle = base.handle

    padding = _semantic._str_to_padding_option(padding_option)

    layout = _unwrap_if_constexpr(layout)
    assert isinstance(layout, NVMMASharedLayout), \
        "Expected layout to be a NVMMASharedLayout"

    shape_type = ttgl.tuple(shape).type
    strides_type = ttgl.tuple(strides).type
    ty = tensor_descriptor_type(block_type, shape_type, strides_type, layout)

    if base.type.element_ty.is_int() and padding == ttgl.ir.PADDING_OPTION.PAD_NAN:
        raise ValueError("Padding option `nan` is not supported for integer blocks")
    handle = _semantic.builder.create_make_tensor_descriptor(
        ty._to_ir(_semantic.builder),
        base_handle,
        [s.handle for s in shape],
        [s.handle for s in strides],
        padding,
    )
    return tensor_descriptor(handle, shape, strides, block_type, layout)
