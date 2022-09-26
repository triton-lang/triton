
import triton
import triton.language as tl


@triton.jit
def math_kernel(x1_ptr, x2_ptr, x3_ptr, x4_ptr, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    x1 = tl.load(x1_ptr + offsets, mask=offsets < n)
    x2 = tl.load(x2_ptr + offsets, mask=offsets < n)
    x3 = tl.load(x3_ptr + offsets, mask=offsets < n)
    x4 = tl.load(x4_ptr + offsets, mask=offsets < n)

    y1 = tl.sin(x1)
    y2 = tl.libdevice.sin(x2)
    y3 = tl.libdevice.fdiv_rn(x3, x3)
    y4 = tl.libdevice.fmaf_rd(x4, x4, x4)

    tl.store(x1_ptr + offsets, y1, mask=offsets < n)
    tl.store(x2_ptr + offsets, y2, mask=offsets < n)
    tl.store(x3_ptr + offsets, y3, mask=offsets < n)
    tl.store(x4_ptr + offsets, y4, mask=offsets < n)


def test_empty_kernel_cubin_compile():
    kernel = triton.compiler._compile(math_kernel,
                                      "*fp32,*fp32,*fp32,*fp32,i32",
                                      device=0,
                                      constants={"BLOCK_SIZE": 256},
                                      output="ttgir")  # "cubin"
    assert kernel
    # TODO: Check if the values are correct.
    # TODO: Cover all the math operators
