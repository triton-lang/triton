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

# 16×16 dot
A16 = torch.ones(16, 16, device=DEVICE)
B16 = torch.ones(16, 16, device=DEVICE)
C16 = torch.zeros(16, 16, device=DEVICE)
try:
    dot_kernel[(1,)](A16, B16, C16, 16, 16, 16)
    check('dot (16x16x16)', C16.max().item() == 16.0, f'max={C16.max().item()}')
except Exception as e:
    check('dot (16x16x16)', False, str(e)[:200])

# 32×32 dot
A32 = torch.ones(32, 32, device=DEVICE)
B32 = torch.ones(32, 32, device=DEVICE)
C32 = torch.zeros(32, 32, device=DEVICE)
try:
    dot_kernel[(1,)](A32, B32, C32, 32, 32, 32)
    check('dot (32x32x32)', C32.max().item() == 32.0, f'max={C32.max().item()}')
except Exception as e:
    check('dot (32x32x32)', False, str(e)[:200])

# Random 16×16 correctness check
torch.manual_seed(42)
Ar = torch.randn(16, 16, device=DEVICE)
Br = torch.randn(16, 16, device=DEVICE)
Cr = torch.zeros(16, 16, device=DEVICE)
Cref = Ar @ Br
try:
    dot_kernel[(1,)](Ar, Br, Cr, 16, 16, 16)
    err = (Cr - Cref).abs().max().item()
    check('dot (16x16 random)', err < 1e-3, f'max_err={err}')
except Exception as e:
    check('dot (16x16 random)', False, str(e)[:200])

# Random 32×32 correctness check
torch.manual_seed(123)
Ar32 = torch.randn(32, 32, device=DEVICE)
Br32 = torch.randn(32, 32, device=DEVICE)
Cr32 = torch.zeros(32, 32, device=DEVICE)
Cref32 = Ar32 @ Br32
try:
    dot_kernel[(1,)](Ar32, Br32, Cr32, 32, 32, 32)
    err32 = (Cr32 - Cref32).abs().max().item()
    check('dot (32x32 random)', err32 < 1e-2, f'max_err={err32}')
except Exception as e:
    check('dot (32x32 random)', False, str(e)[:200])


# ── 3b. Tiled GEMM (multi-CTA, K-loop) ────────────────────────────────────────
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)

    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)

# 64×64 GEMM with 16×16 tiles
M_g, N_g, K_g = 64, 64, 64
BM, BN, BK = 16, 16, 16
torch.manual_seed(7)
Ag = torch.randn(M_g, K_g, device=DEVICE)
Bg = torch.randn(K_g, N_g, device=DEVICE)
Cg = torch.zeros(M_g, N_g, device=DEVICE)
Cref_g = Ag @ Bg
grid = (M_g // BM, N_g // BN)
try:
    matmul_kernel[grid](
        Ag, Bg, Cg,
        M_g, N_g, K_g,
        Ag.stride(0), Ag.stride(1),
        Bg.stride(0), Bg.stride(1),
        Cg.stride(0), Cg.stride(1),
        BM, BN, BK,
    )
    err_g = (Cg - Cref_g).abs().max().item()
    check('gemm (64x64, 16x16 tiles)', err_g < 1e-2, f'max_err={err_g}')
except Exception as e:
    check('gemm (64x64, 16x16 tiles)', False, str(e)[:200])


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


# ── 5. Masked load/store (scalar runtime arg) ─────────────────────────────────
@triton.jit
def masked_load_kernel(x, out, N: tl.constexpr, sz):
    offs = tl.arange(0, N)
    mask = offs < sz
    v = tl.load(x + offs, mask=mask, other=0.)
    tl.store(out + offs, v, mask=mask)

x = torch.ones(64, device=DEVICE)
out = torch.zeros(64, device=DEVICE)
masked_load_kernel[(1,)](x, out, 64, 50)
print(f'  [dbg] out[48:52]={out[48:52].tolist()}, sum50={out[:50].sum().item()}, tail={out[50:].sum().item()}')
check('masked load', out[:50].sum().item() == 50. and out[50:].sum().item() == 0.)


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f'\n{passed}/{total} passed')
