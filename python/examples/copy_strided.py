import triton
import triton.language as tl


# triton kernel
@triton.jit
def kernel(X, stride_xm,
           Z, stride_zn,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * 1
    Zs = Z + off_m[:, None] * 1 + off_n[None, :] * stride_zn
    tl.store(Zs, tl.load(Xs))


ret = triton.compile(kernel, signature="*fp32,i32,*fp32,i32", constants={"BLOCK_M": 64, "BLOCK_N": 64})
print(ret.asm["ttgir"])
