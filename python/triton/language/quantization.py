from __future__ import division, annotations

from triton import language as tl


def _dequantize(
    input: tl.tensor,
    scale: tl.tensor,
    shift: tl.tensor,
    nbit: int,
    dst_ty: tl.dtype,
    builder: tl.ir.builder,
) -> tl.tensor:
    input_ty = input.type
    assert isinstance(input_ty, tl.block_type)
    assert input_ty.element_ty.is_int32() or input_ty.element_ty.is_int16()
    assert nbit in [2, 4, 8]
    assert dst_ty == tl.float16

    shape = input_ty.get_block_shapes()
    factor = input_ty.element_ty.primitive_bitwidth // nbit
    dst_shape = shape[:-1] + (factor * shape[-1],)

    dst_ty = tl.block_type(dst_ty, dst_shape)
    return tl.tensor(
        builder.create_dequantize(
            input.handle,
            scale.handle,
            shift.handle,
            dst_ty.to_ir(builder),
        ),
        dst_ty,
    )


@tl.builtin
def dequantize(
    input: tl.tensor,
    scale: tl.tensor,
    shift: tl.tensor,
    nbit: int,
    dst_ty: tl.dtype = tl.float16,
    _builder: tl.ir.builder = None,
):
    """
    Tries to dequantize the input to given dtype
    """
    nbit = tl._constexpr_to_value(nbit)
    return _dequantize(input, scale, shift, nbit, dst_ty, _builder)
