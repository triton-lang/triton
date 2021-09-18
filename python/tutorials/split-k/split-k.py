import torch
import triton
import triton.language as tl

N_BLOCK_M = 32
N_BLOCK_N = 32
N_SPLIT_K = 32
N_WARP = 1

@triton.heuristics({
    'EVEN_K': lambda *args, **meta: args[5] % (meta['BLOCK_K'] * meta['SPLIT_K']) == 0,
})
@triton.jit
def _matmul_atomic(A, B, C, M, N, K, 
            stride_am, stride_ak, 
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            **META):
  # extract meta-parameters
  BLOCK_M = META['BLOCK_M']
  BLOCK_N = META['BLOCK_N']
  BLOCK_K = META['BLOCK_K']
  GROUP_M = META['GROUP_M']
  SPLIT_K = META['SPLIT_K']

  # pid
  pid = tl.program_id(0)
  pid_z = tl.program_id(1)
  grid_m = (M + BLOCK_M - 1) // BLOCK_M
  grid_n = (N + BLOCK_N - 1) // BLOCK_N
  # L2 swizzling
  width = GROUP_M * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
  pid_m = group_id * GROUP_M + (pid % group_size)
  pid_n = (pid % width) // (group_size)
  # do matrix multiplication
  rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
  rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
  rk = pid_z*BLOCK_K + tl.arange(0, BLOCK_K)
  # pointers
  A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
  B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
  acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  for k in range(K, 0, -BLOCK_K*SPLIT_K):
    if META['EVEN_K']:
      a = tl.load(A)
      b = tl.load(B)
    else:
      a = tl.load(A, mask=rk[None, :] < k, other=0.)
      b = tl.load(B, mask=rk[:, None] < k, other=0.)
    acc += tl.dot(a, b)
    A += BLOCK_K * SPLIT_K * stride_ak
    B += BLOCK_K * SPLIT_K * stride_bk
  acc = acc.to(tl.float16)
  # rematerialize rm and rn to save registers
  rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
  mask = (rm < M)[:, None] & (rn < N)[None, :]
  # handles write-back with reduction-splitting
  # if SPLIT_K == 1:
  #   tl.store(C, acc, mask=mask)
  # else:
  # tl.store(C, acc, mask=mask)
  tl.atomic_add(C, acc, mask=mask)
    # LOCKS = LOCKS + tl.program_id(0)
    # COUNT = LOCKS + tl.num_programs(0)
    # while tl.atomic_cas(LOCKS, 0, 1) == 1:
    #     pass
    # count = tl.load(COUNT)
    # if count == 0:
    #     tl.store(C, acc, mask=mask)
    # else:
    #     curr = tl.load(C, mask=mask, other=0.)
    #     tl.store(C, acc + curr, mask=mask)
    # tl.atomic_xchg(COUNT, (count + 1) % SPLIT_K)
    # tl.atomic_xchg(LOCKS, 0)

def matmul_atomic(a, b):
  device = a.device
  dtype = a.dtype
  M, K = a.shape
  _, N = b.shape
  
  # allocates output
  c = torch.zeros((M, N), device=device, dtype=dtype)

  grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * 
                       triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
  pgm = _matmul_atomic[grid](a, b, c, M, N, K, 
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_M=N_BLOCK_M, BLOCK_N=N_BLOCK_N, BLOCK_K=32, 
                SPLIT_K=N_SPLIT_K, GROUP_M=8, num_warps=N_WARP)

  return c

@triton.heuristics({
    'EVEN_K': lambda *args, **meta: args[5] % (meta['BLOCK_K'] * meta['SPLIT_K']) == 0,
})
@triton.jit
def _matmul_locks(A, B, C, M, N, K, 
            stride_am, stride_ak, 
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            LOCKS,
            **META):
  # extract meta-parameters
  BLOCK_M = META['BLOCK_M']
  BLOCK_N = META['BLOCK_N']
  BLOCK_K = META['BLOCK_K']
  GROUP_M = META['GROUP_M']
  SPLIT_K = META['SPLIT_K']

  # pid
  pid = tl.program_id(0)
  pid_z = tl.program_id(1)
  grid_m = (M + BLOCK_M - 1) // BLOCK_M
  grid_n = (N + BLOCK_N - 1) // BLOCK_N
  # L2 swizzling
  width = GROUP_M * grid_n
  group_id = pid // width
  group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
  pid_m = group_id * GROUP_M + (pid % group_size)
  pid_n = (pid % width) // (group_size)
  # do matrix multiplication
  rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
  rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
  rk = pid_z*BLOCK_K + tl.arange(0, BLOCK_K)
  # pointers
  A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
  B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
  acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
  for k in range(K, 0, -BLOCK_K*SPLIT_K):
    if META['EVEN_K']:
      a = tl.load(A)
      b = tl.load(B)
    else:
      a = tl.load(A, mask=rk[None, :] < k, other=0.)
      b = tl.load(B, mask=rk[:, None] < k, other=0.)
    acc += tl.dot(a, b)
    A += BLOCK_K * SPLIT_K * stride_ak
    B += BLOCK_K * SPLIT_K * stride_bk
  acc = acc.to(tl.float16)
  # rematerialize rm and rn to save registers
  rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
  rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
  C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
  mask = (rm < M)[:, None] & (rn < N)[None, :]
  # handles write-back with reduction-splitting
  # if SPLIT_K == 1:
  #   tl.store(C, acc, mask=mask)
  # else:
  # tl.store(C, acc, mask=mask)
  LOCKS = LOCKS + tl.program_id(0)
  COUNT = LOCKS + tl.num_programs(0)
  while tl.atomic_cas(LOCKS, 0, 1) == 1:
      pass
  count = tl.load(COUNT)
  if count == 0:
      tl.store(C, acc, mask=mask)
  else:
      curr = tl.load(C, mask=mask, other=0.)
      tl.store(C, acc + curr, mask=mask)
  tl.atomic_xchg(COUNT, (count + 1) % SPLIT_K)
  tl.atomic_xchg(LOCKS, 0)

def matmul_locks(a, b):
  device = a.device
  dtype = a.dtype
  M, K = a.shape
  _, N = b.shape
  
  # allocates output
  c = torch.zeros((M, N), device=device, dtype=dtype)
  locks = torch.zeros(1024 * 1024, dtype=torch.int32, device=device)

  grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * 
                       triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
  pgm = _matmul_locks[grid](a, b, c, M, N, K, 
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                locks,
                BLOCK_M=N_BLOCK_M, BLOCK_N=N_BLOCK_N, BLOCK_K=32, 
                SPLIT_K=N_SPLIT_K, GROUP_M=8, num_warps=N_WARP)

  return c

# test
mm_geo = [
  # (64, 64, 65536),
  # (64, 64, 16384),
  # (128, 256, 65536),
  # (256, 128, 16384),
  # (64, 64, 8192),
  (32, 32, 16384),
]

for M, N, K in mm_geo:
  a = torch.randn((M, K), device='cuda', dtype=torch.float16)
  b = torch.randn((K, N), device='cuda', dtype=torch.float16)

  c_ref = torch.matmul(a, b)
  c_atomic = matmul_atomic(a, b)
  c_locks = matmul_locks(a, b)

  torch_ms, _, _ = triton.testing.do_bench(lambda: torch.matmul(a, b))
  atomic_ms, _, _ = triton.testing.do_bench(lambda: matmul_atomic(a, b))
  lock_ms, _, _ = triton.testing.do_bench(lambda: matmul_locks(a, b))

  perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)

  print(f'{M}, {N}, {K}')
  print(f'torch: {perf(torch_ms):.2f} TFLOPS')
  print(f'atomic: {perf(atomic_ms):.2f} TFLOPS')
  print(f'lock: {perf(lock_ms):.2f} TFLOPS')

  print('')