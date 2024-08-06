import triton
import triton.language as tl


# Kernel that implements GEMM with explicit multiply-reduce instructions for small block sizes.
# Based on **tune_gemm** `matmul_kernel` from commit `cf44637` (see `triton-mlir` branch).
@triton.jit
def multreduce_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                             stride_cm, stride_cn, stride_bias, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                             BLOCK_SIZE_K: tl.constexpr, BIAS: tl.constexpr, EVEN_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    if BIAS:
        bias_ptrs = bias_ptr + offs_am * stride_bias
        bias = tl.load(bias_ptrs, mask=offs_am < M, other=0.0)
    acc_dtype = tl.float32 if a_ptr.type.element_ty != tl.int8 else tl.int32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Dot product implemented as explicit multiply-reduce:
        a = tl.reshape(a, (BLOCK_SIZE_M, BLOCK_SIZE_K, 1)).to(acc_dtype)
        b = tl.reshape(b, (1, BLOCK_SIZE_K, BLOCK_SIZE_N)).to(acc_dtype)
        accumulator += tl.sum(a * b, axis=1)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(c_ptr.type.element_ty)
    if BIAS:
        c += bias[:, None]
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)
