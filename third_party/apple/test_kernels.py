"""
Apple MPS Triton backend — curated kernel tests.
Run: python third_party/apple/test_kernels.py

Each test prints "name: OK" or "name: FAIL <details>".
Add new tests as features are implemented.
"""
import torch
import triton
import triton.language as tl

DEVICE = 'mps'
results = []

def check(name, cond, detail=''):
    status = 'OK' if cond else f'FAIL {detail}'
    print(f'{name}: {status}')
    results.append((name, cond))


# ── 1. Element-wise add (scalar load/store, program_id) ───────────────────────
@triton.jit
def add_kernel(x, y, out, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    tl.store(out + offs, tl.load(x + offs) + tl.load(y + offs))

x = torch.ones(128, device=DEVICE)
y = torch.full((128,), 2., device=DEVICE)
out = torch.zeros(128, device=DEVICE)
add_kernel[(1,)](x, y, out, 128)
check('add', (out - 3.).abs().max().item() == 0)


# ── 2. Multi-program dispatch (4 CTAs, each handles 64 elements) ──────────────
@triton.jit
def multiprog_kernel(x, out, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    tl.store(out + offs, tl.load(x + offs) * 2.)

x = torch.arange(256, dtype=torch.float32, device=DEVICE)
out = torch.zeros(256, device=DEVICE)
multiprog_kernel[(4,)](x, out, 64)
check('multiprog', (out - x * 2).abs().max().item() == 0)


# ── 3. Matrix multiply via tl.dot (simdgroup MMA) ─────────────────────────────
@triton.jit
def dot_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    a = tl.load(A + offs_m[:, None] * K + offs_k[None, :])
    b = tl.load(B + offs_k[:, None] * N + offs_n[None, :])
    tl.store(C + offs_m[:, None] * N + offs_n[None, :], tl.dot(a, b))

A = torch.ones(8, 8, device=DEVICE)
B = torch.ones(8, 8, device=DEVICE)
C = torch.zeros(8, 8, device=DEVICE)
dot_kernel[(1,)](A, B, C, 8, 8, 8)
check('dot (8x8x8)', C.max().item() == 8.0)


# ── 4. Element-wise float ops on tensors (struct<(f32)> unpack/pack) ──────────
@triton.jit
def float_ops_kernel(x, out, N: tl.constexpr):
    offs = tl.arange(0, N)
    v = tl.load(x + offs)
    v = v * 2.0      # MulFOp
    v = v + 1.0      # AddFOp
    v = v - 0.5      # SubFOp
    v = v / 2.0      # DivFOp
    tl.store(out + offs, v)

x = torch.ones(32, device=DEVICE)
out = torch.zeros(32, device=DEVICE)
float_ops_kernel[(1,)](x, out, 32)
# (1*2+1-0.5)/2 = 1.25
check('float_ops', (out - 1.25).abs().max().item() < 1e-5)


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f'\n{passed}/{total} passed')
