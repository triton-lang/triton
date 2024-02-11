import torch
import triton
import triton.language as tl


@triton.jit
def topk(X, Vals, Idx, N, BLOCK_M: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    off_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    Xs = X + off_m[:, None] * N + tl.arange(0, K)
    ret = tl.full((BLOCK_M, K), float("-inf"), dtype=tl.float32)
    ret = (ret.to(tl.int32, bitcast=True) & 0x0000FFFF).to(tl.float32, bitcast=True)
    for k in range(0, N, K):
        # cur = (16-bit (val), 16-bit (idx))
        cur = tl.zeros((BLOCK_M, K), dtype=tl.uint32)
        cur = cur | (tl.load(Xs).to(tl.uint16, bitcast=True).to(tl.uint32) << 16)
        cur = cur | (k + tl.arange(0, K))
        cur = cur.to(tl.float32, bitcast=True)
        # one step of bitonic-merge sort
        cur = tl.sort(cur, dim=1, descending=0)
        ret = ret + cur
        ret = tl.sort(ret, dim=1, descending=1)
        # increment pointer
        Xs += K
    # unpack values and indices into separate variables
    vals = (ret.to(tl.int32, bitcast=True) >> 16).to(tl.int16).to(tl.float16, bitcast=True)
    idx = (ret.to(tl.int32, bitcast=True) & 0x0000FFFF).to(tl.int16)
    tl.store(Vals + off_m[:, None] * K + tl.arange(0, K), vals)
    tl.store(Idx + off_m[:, None] * K + tl.arange(0, K), idx)


M = 2048
N = 16384
K = 64
BLOCK_M = 8
x = torch.randn((M, N), dtype=torch.float16, device="cuda")
y = torch.empty((M, K), dtype=torch.float16, device="cuda")
i = torch.empty((M, K), dtype=torch.int16, device="cuda")
grid = (triton.cdiv(M, BLOCK_M), )
fn = lambda: topk[grid](x, y, i, N, BLOCK_M, K)
tri_ms = triton.testing.do_bench(fn)
ref_ms = triton.testing.do_bench(lambda: torch.topk(x, K))
# h = fn()
print(tri_ms, ref_ms)
# print(h.asm["ttgir"])
# print(h.asm["ttgir"])
# print(y - torch.topk(x, k=K).values)
# print(i - torch.topk(x, k=K).indices)
# print(y)
# print(i)
