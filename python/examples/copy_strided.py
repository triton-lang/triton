import triton
import triton.language as tl
import triton.compiler as tc


# triton kernel
@triton.jit
def kernel(X, stride_xm,  #
           Z, stride_zn,  #
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    off_m = tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, BLOCK_N)
    Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * 1
    Zs = Z + off_m[:, None] * 1 + off_n[None, :] * stride_zn
    tl.store(Zs, tl.load(Xs))


src = tc.ASTSource(
    fn=kernel,
    constants={"BLOCK_M": 64, "BLOCK_N": 64},
    signature="*fp32,i32,*fp32,i32",
)

ret = triton.compile(src)
print(ret.asm["ttgir"])
