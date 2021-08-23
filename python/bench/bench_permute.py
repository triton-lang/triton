import torch
import triton
import triton.language as tl

@triton.jit
def _copy(x_ptr, x_stride0, x_stride1,
          y_ptr, y_stride0, y_stride1,
          **meta):
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)
    BLOCK_M = meta['BLOCK_M']
    BLOCK_N = meta['BLOCK_N']
    off_m = pid_0 * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_1 * BLOCK_N + tl.arange(0, BLOCK_N)
    x_ptrs = x_ptr + off_m[:, None] * x_stride0 + off_n[None, :] * x_stride1
    y_ptrs = y_ptr + off_m[:, None] * y_stride0 + off_n[None, :] * y_stride1
    tl.store(y_ptrs, tl.load(x_ptrs), mask=True)

def copy(x, perm):
    M, N = x.shape
    y = torch.empty_like(x).permute(perm)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),\
                         triton.cdiv(N, meta['BLOCK_N']))
    pgm = _copy[grid](x, x.stride(0), x.stride(1),
          y, y.stride(0), y.stride(1),
          BLOCK_M=128, BLOCK_N=128)
    print(pgm.asm('ptx'))
    return y

x = torch.randn((8192, 8192), device='cuda')
gbps = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
# ms, _, _= triton.testing.do_bench(lambda: copy(x, (1, 0)))
# print(gbps(ms))

print('---')
print(copy(x, (1, 0)))
print(x)
print('---')