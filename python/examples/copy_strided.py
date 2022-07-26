
import triton
import triton.language as tl


# triton kernel
@triton.jit
def kernel(X, stride_xm, stride_xn,
           Z, stride_zm, stride_zn,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn
    Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
    tl.store(Zs, tl.load(Xs))


ret = triton.compile(kernel, "*fp32,i32,i32,*fp32,i32,i32", constants={"BLOCK_M": 128, "BLOCK_N": 128}, output="ttgir")
print(ret)
