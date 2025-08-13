import torch
import triton
import triton.language as tl


@triton.jit
def _reduce_standard(X, stride_xb, stride_xm, stride_xn,  #
                     Y, stride_ym, stride_yn,  #
                     B, M, N,  #
                     BLOCK_MN: tl.constexpr,  #
                     ):
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    total = M * N
    base = tl.arange(0, BLOCK_MN) + pid * BLOCK_MN
    for start in tl.range(0, total, BLOCK_MN * num_pid):
        idxs = start + base
        mask = idxs < total
        # Assumption: last two dims [M, N] are contiguous -> linear offset `idxs`
        ptrs = X + idxs
        acc = tl.zeros([BLOCK_MN], dtype=tl.float32)
        for b in tl.range(B, num_stages=2):
            vals = tl.load(ptrs + b * stride_xb, mask=mask, other=0.0)
            acc += vals.to(tl.float32)
        out_ptrs = Y + idxs
        tl.store(out_ptrs, acc, mask=mask)


def reduce_standard(x: torch.Tensor, dim: int = 0):
    if dim != 0 or x.ndim != 3:
        raise NotImplementedError("reduce_standard only supports 3D inputs with dim=0")
    B, M, N = x.shape
    # Require last two dims to be contiguous so [M, N] can be linearly indexed
    if not (x.stride(2) == 1 and x.stride(1) == N):
        raise NotImplementedError("reduce_standard requires contiguous last two dims to collapse [M, N]")
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_MN = 32768
    grid0 = min(256, triton.cdiv(M * N, BLOCK_MN))
    grid = (grid0, )
    _reduce_standard[grid](
        x, x.stride(0), x.stride(1), x.stride(2),  #
        y, y.stride(0), y.stride(1),  #
        B, M, N,  #
        BLOCK_MN=BLOCK_MN,  #
        num_warps=32,  #
    )
    return y


def reduce_standard_torch(x: torch.Tensor, dim: int = 0):
    if dim != 0 or x.ndim != 3:
        raise NotImplementedError("reduce_standard_torch only supports 3D inputs with dim=0")
    return x.sum(dim=0)
