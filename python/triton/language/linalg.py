import triton.language as tl


def _i_dot(
    lhs: tl.tensor,
    rhs: tl.tensor,
    allow_tf32: bool,
    trans_a: bool,
    trans_b: bool,
    builder: tl.ir.builder,
) -> tl.tensor:
    assert lhs.type.is_block() and rhs.type.is_block()
    if lhs.type.scalar.is_int():
        _0 = builder.get_int32(0)
        ret_scalar_ty = tl.int32
    else:
        _0 = builder.get_float32(0)
        ret_scalar_ty = tl.float32
    M = lhs.type.shape[1 if trans_a else 0]
    N = rhs.type.shape[0 if trans_b else 1]
    _0 = builder.create_splat(_0, [M, N])
    ret_ty = tl.block_type(ret_scalar_ty, [M, N])
    return tl.tensor(
        builder.create_dot(lhs.handle, rhs.handle, _0, allow_tf32, trans_a, trans_b),
        ret_ty,
    )


@tl.builtin
def dot(input, other, allow_tf32=True, trans_a=False, trans_b=False, _builder=None):
    """
    Returns the matrix product of two blocks.

    The two blocks must be two-dimensional and have compatible inner dimensions.

    :param input: The first tensor to be multiplied.
    :type input: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    """
    allow_tf32 = tl._constexpr_to_value(allow_tf32)
    return _i_dot(input, other, allow_tf32, trans_a, trans_b, _builder)
