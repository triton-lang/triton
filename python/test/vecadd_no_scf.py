import triton
import triton.language as tl

NUM_WARPS = 4

# triton kernel


@triton.jit
def kernel(x_ptr, stride_xn,
           y_ptr, stride_yn,
           z_ptr, stride_zn,
           BLOCK_SIZE_N: tl.constexpr):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    x_ptrs = x_ptr + offset
    y_ptrs = y_ptr + offset
    x = tl.load(x_ptrs)
    y = tl.load(y_ptrs)
    z = x + y
    z_ptrs = z_ptr + offset
    tl.store(z_ptrs, z)


ret = triton.compile(kernel, "*fp32,i32,*fp32,i32,*fp32,i32", constants={"BLOCK_SIZE_N": 256}, num_warps=NUM_WARPS, device=0, output="ptx")

print(ret)

# TODO: base class for python end2end tests,
#      runtime execution, correctness comparison etc.
