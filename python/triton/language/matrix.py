from __future__ import division, annotations

import triton.language as tl


def _dot(
    a: tl.tensor,
    b: tl.tensor,
    trans_a: bool,
    trans_b: bool,
    allow_tf32: bool,
    builder: tl.ir.builder,
) -> tl.tensor:
    in_a = 1 if not trans_a else 0
    in_b = 1 if trans_b else 0
    assert a.type.is_block() and b.type.is_block()
    assert len(a.shape) == 2 and len(b.shape) == 2
    assert a.shape[in_a] == b.shape[in_b]
    assert (
        a.shape[0] >= 16 and a.shape[1] >= 16 and b.shape[1] >= 16
    ), "small blocks not supported!"
    if a.type.scalar.is_int():
        _0 = builder.get_int32(0)
        ret_scalar_ty = tl.int32
    else:
        _0 = builder.get_float32(0)
        ret_scalar_ty = tl.float32
    M = a.type.shape[in_a ^ 1]
    N = b.type.shape[in_b ^ 1]
    _0 = builder.create_splat(_0, [M, N])
    ret_ty = tl.block_type(ret_scalar_ty, [M, N])
    ret = builder.create_dot(
        a.handle,
        b.handle,
        _0,
        trans_a,
        trans_b,
        allow_tf32,
    )
    return tl.tensor(ret, ret_ty)


@tl.builtin
def dot(
    input,
    other,
    trans_a=False,
    trans_b=False,
    allow_tf32=True,
    _builder: tl.ir.builder = None,
):
    """
    Returns the matrix product of two blocks.

    The two blocks must be two dimensionals and have compatible inner dimensions.

    :param input: The first tensor to be multiplied.
    :type input: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    :param other: The second tensor to be multiplied.
    :type other: 2D tensor of scalar-type in {:code:`float16`, :code:`bfloat16`, :code:`float32`}
    """
    allow_tf32 = tl._constexpr_to_value(allow_tf32)
    return _dot(input, other, trans_a, trans_b, allow_tf32, _builder)
