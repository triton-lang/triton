import torch
import triton.language as tl
import triton

'''
[split_k, m, n] -> [m, n]
'''
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32},  num_warps=4),
        triton.Config({'BLOCK_M': 64 , 'BLOCK_N': 32},  num_warps=2),
        triton.Config({'BLOCK_M': 32 , 'BLOCK_N': 64},  num_warps=2),
    ],
    key=['SPLIT_K', 'M', 'N'],
)
@triton.jit
def _block_reduce(src, dst, SPLIT_K, M, N, stride_k, stride_m, stride_n, **META):
  BLOCK_M = META['BLOCK_M']
  BLOCK_N = META['BLOCK_N']
  pid_m = tl.program_id(0)
  pid_n = tl.program_id(1)

  rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

  mask = (rm < M)[:, None] & (rn < N)[None, :]
  src += rm[:, None] * stride_m + rn[None, :] * stride_n

  acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  for k in range(0, SPLIT_K, 1):
    acc += tl.load(src, mask=mask)
    src += stride_k

  dst += rm[:, None] * stride_m + rn[None, :] * stride_n

  tl.store(dst, acc, mask=mask)

def block_reduce(src):
  assert src.ndim == 3

  SPLIT_K, M, N = src.shape
  device, dtype = src.device, src.dtype
  dst = torch.empty((M, N), device=device, dtype=dtype)

  grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']), )
  _block_reduce[grid](src, dst, SPLIT_K, M, N,
                      src.stride(0), src.stride(1), src.stride(2))
  
  return dst