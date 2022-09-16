import triton
import triton.language as tl


@triton.jit
def cast_check():
    zero_0d = tl.zeros([], dtype=tl.float32)
    zero_1d = tl.zeros([2], dtype=tl.float32)
    zero_2d_21 = tl.zeros([2, 1], dtype=tl.float32)
    zero_2d_22 = tl.zeros([2, 2], dtype=tl.float32)

    # scalar + scalar -> scalar
    a0 = 0.0 + 0.0
    # scalar + 0D -> 0D
    a1 = 0.0 + zero_0d
    a2 = zero_0d + 0.0
    # scalar + 1D -> 1D
    a3 = 0.0 + zero_1d
    a4 = zero_1d + 0.0
    # scalar + 2D -> 2D
    a5 = 0.0 + zero_2d_22
    a6 = zero_2d_22 + 0.0

    # 0D + 0D -> 0D
    b1 = zero_0d + zero_0d
    # 0D + 1D -> 1D
    b2 = zero_0d + zero_1d
    b3 = zero_1d + zero_0d
    # 0D + 2D -> 2D
    b4 = zero_0d + zero_2d_22
    b5 = zero_2d_22 + zero_0d

    # 1D + 1D -> 1D
    c1 = zero_1d + zero_1d
    # 1D + 2D -> 2D
    c2 = zero_1d + zero_2d_21
    c3 = zero_1d + zero_2d_22
    c4 = zero_2d_21 + zero_1d
    c5 = zero_2d_22 + zero_1d

    # 2D + 2D -> 2D
    d1 = zero_2d_21 + zero_2d_21
    d2 = zero_2d_22 + zero_2d_22
    d3 = zero_2d_21 + zero_2d_22
    d4 = zero_2d_22 + zero_2d_21

    return a0, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, d1, d2, d3, d4


def test_cast_check():
    kernel = triton.compile(cast_check,
                            signature="",
                            device=0,
                            output="ttir")
    assert (kernel)
    # TODO: Check types of the results
