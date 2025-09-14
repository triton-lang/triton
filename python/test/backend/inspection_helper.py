import triton
import triton.language as tl
import compiler_inspection  # NOQA


@triton.jit
def dummy_kernel(n_elements, BLOCK_SIZE: tl.constexpr):
    pass


n_elements = 98432
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
dummy_kernel[grid](n_elements, BLOCK_SIZE=1024, num_warps=1)
