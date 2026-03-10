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

# Multi-CTA dot (64 threadgroups, 16x16 tiles — regression test)
@triton.jit
def dot_multi_cta_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = N // BN
    pid_m = pid // num_n
    pid_n = pid % num_n
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, K)
    a = tl.load(A_ptr + rm[:, None] * K + rk[None, :])
    b = tl.load(B_ptr + rk[:, None] * N + rn[None, :])
    acc = tl.dot(a, b)
    tl.store(C_ptr + rm[:, None] * N + rn[None, :], acc)

torch.manual_seed(42)
_M, _N, _K = 128, 128, 16
_BM, _BN = 16, 16
_A = torch.randn(_M, _K, device=DEVICE)
_B = torch.randn(_K, _N, device=DEVICE)
_C = torch.zeros(_M, _N, device=DEVICE)
try:
    dot_multi_cta_kernel[(_M // _BM * (_N // _BN),)](_A, _B, _C, _M, _N, _K, _BM, _BN)
    torch.mps.synchronize()
    _ref = _A @ _B
    _err = (_C - _ref).abs().max().item()
    check('dot (16x16, 64 CTAs)', _err < 1e-2, f'max_err={_err}')
except Exception as e:
    check('dot (16x16, 64 CTAs)', False, str(e)[:200])


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


# ── 4b. Math intrinsics ───────────────────────────────────────────────────────
@triton.jit
def math_exp_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.exp(tl.load(x_ptr + offs)))

@triton.jit
def math_log_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.log(tl.load(x_ptr + offs)))

@triton.jit
def math_sqrt_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.sqrt(tl.load(x_ptr + offs)))

@triton.jit
def math_sin_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.sin(tl.load(x_ptr + offs)))

@triton.jit
def math_cos_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.cos(tl.load(x_ptr + offs)))

@triton.jit
def math_abs_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    tl.store(out_ptr + offs, tl.abs(tl.load(x_ptr + offs)))

torch.manual_seed(0)
x_rand = torch.randn(32, device=DEVICE)
x_pos = torch.rand(32, device=DEVICE) + 0.1

for name, kern, x, ref_fn in [
    ('exp',  math_exp_kernel,  x_rand, torch.exp),
    ('log',  math_log_kernel,  x_pos,  torch.log),
    ('sqrt', math_sqrt_kernel, x_pos,  torch.sqrt),
    ('sin',  math_sin_kernel,  x_rand, torch.sin),
    ('cos',  math_cos_kernel,  x_rand, torch.cos),
    ('abs',  math_abs_kernel,  x_rand, torch.abs),
]:
    out = torch.zeros(32, device=DEVICE)
    kern[(1,)](x, out, 32)
    ref = ref_fn(x)
    ok = torch.allclose(out, ref, atol=1e-3, rtol=1e-3)
    check(f'math_{name}', ok, f'max_err={(out-ref).abs().max().item():.6f}' if not ok else '')


# ── 4c. Type casts ────────────────────────────────────────────────────────────
@triton.jit
def cast_f32_f16(x, o, N: tl.constexpr):
    i = tl.arange(0, N); tl.store(o+i, tl.load(x+i).to(tl.float16))

@triton.jit
def cast_f16_f32(x, o, N: tl.constexpr):
    i = tl.arange(0, N); tl.store(o+i, tl.load(x+i).to(tl.float32))

@triton.jit
def cast_f32_bf16(x, o, N: tl.constexpr):
    i = tl.arange(0, N); tl.store(o+i, tl.load(x+i).to(tl.bfloat16))

@triton.jit
def cast_bf16_f32(x, o, N: tl.constexpr):
    i = tl.arange(0, N); tl.store(o+i, tl.load(x+i).to(tl.float32))

@triton.jit
def cast_i32_f32(x, o, N: tl.constexpr):
    i = tl.arange(0, N); tl.store(o+i, tl.load(x+i).to(tl.float32))

@triton.jit
def cast_f32_i32(x, o, N: tl.constexpr):
    i = tl.arange(0, N); tl.store(o+i, tl.load(x+i).to(tl.int32))

x_f32 = torch.randn(32, device=DEVICE)

o = torch.zeros(32, device=DEVICE, dtype=torch.float16)
cast_f32_f16[(1,)](x_f32, o, 32)
check('cast f32→f16', torch.allclose(o.float(), x_f32.half().float(), atol=1e-3))

o = torch.zeros(32, device=DEVICE)
cast_f16_f32[(1,)](x_f32.half(), o, 32)
check('cast f16→f32', torch.allclose(o, x_f32.half().float(), atol=1e-5))

o = torch.zeros(32, device=DEVICE, dtype=torch.bfloat16)
cast_f32_bf16[(1,)](x_f32, o, 32)
check('cast f32→bf16', torch.allclose(o.float(), x_f32.bfloat16().float(), atol=1e-2))

o = torch.zeros(32, device=DEVICE)
cast_bf16_f32[(1,)](x_f32.bfloat16(), o, 32)
check('cast bf16→f32', torch.allclose(o, x_f32.bfloat16().float(), atol=1e-5))

x_i32 = torch.arange(32, device=DEVICE, dtype=torch.int32)
o = torch.zeros(32, device=DEVICE)
cast_i32_f32[(1,)](x_i32, o, 32)
check('cast i32→f32', torch.allclose(o, x_i32.float()))

x_flt = torch.tensor([1.7, -2.3, 0., 100.9]*8, device=DEVICE)
o = torch.zeros(32, device=DEVICE, dtype=torch.int32)
cast_f32_i32[(1,)](x_flt, o, 32)
check('cast f32→i32', torch.equal(o, x_flt.int()))


# ── 4b. Mixed dtype binary ops (bf16+int, regression for cast paths) ──────────
@triton.jit
def mixed_add_kernel(X, Y, Z, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.load(Y + offs)
    z = x + y
    tl.store(Z + offs, z)

for int_dtype, int_name in [(torch.int8, 'i8'), (torch.int16, 'i16'), (torch.int32, 'i32')]:
    for float_dtype, float_name in [(torch.bfloat16, 'bf16'), (torch.float16, 'f16'), (torch.float32, 'f32')]:
        x_int = torch.arange(1, 33, device=DEVICE, dtype=int_dtype)
        y_float = torch.ones(32, device=DEVICE, dtype=float_dtype)
        z_out = torch.zeros(32, device=DEVICE, dtype=float_dtype)
        test_name = f'mixed add ({int_name}+{float_name})'
        try:
            mixed_add_kernel[(1,)](x_int, y_float, z_out, 32)
            torch.mps.synchronize()
            ref = x_int.to(float_dtype) + y_float
            err = (z_out - ref).abs().max().item()
            check(test_name, err < 1e-2, f'max_err={err}')
        except Exception as e:
            check(test_name, False, str(e)[:120])

# Scalar + tensor mixed dtype (regression for test_bin_op[1-bfloat16-int8-+])
@triton.jit
def mixed_scalar_add_kernel(X, Y, Z, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    y = tl.load(Y)  # scalar load
    z = x + y
    tl.store(Z + offs, z)

for int_dtype, int_name in [(torch.int8, 'i8'), (torch.int16, 'i16')]:
    x_bf16 = torch.ones(32, device=DEVICE, dtype=torch.bfloat16) * 2.0
    y_int = torch.tensor([3], device=DEVICE, dtype=int_dtype)
    z_out = torch.zeros(32, device=DEVICE, dtype=torch.bfloat16)
    test_name = f'mixed scalar add (bf16+{int_name})'
    try:
        mixed_scalar_add_kernel[(1,)](x_bf16, y_int, z_out, 32)
        torch.mps.synchronize()
        ref = x_bf16 + y_int.to(torch.bfloat16)
        err = (z_out - ref).abs().max().item()
        check(test_name, err < 1e-2, f'max_err={err}')
    except Exception as e:
        check(test_name, False, str(e)[:120])


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


# ── 11. Reduce sum (single warp, 32 elements) ────────────────────────────────
@triton.jit
def reduce_sum_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = tl.sum(x, axis=0)
    tl.store(output_ptr + pid, result)

x = torch.ones(32, device=DEVICE)
out = torch.zeros(1, device=DEVICE)
reduce_sum_kernel[(1,)](x, out, 32, BLOCK_SIZE=32)
check('reduce sum', out.item() == 32.0, f'got {out.item()}')


# ── 12. Reduce max (single warp, 32 elements) ────────────────────────────────
@triton.jit
def reduce_max_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=float('-inf'))
    result = tl.max(x, axis=0)
    tl.store(output_ptr + pid, result)

x = torch.arange(32, device=DEVICE, dtype=torch.float32)
out = torch.zeros(1, device=DEVICE)
reduce_max_kernel[(1,)](x, out, 32, BLOCK_SIZE=32)
check('reduce max', out.item() == 31.0, f'got {out.item()}')


# ── 13. Reduce min (single warp, 32 elements) ────────────────────────────────
@triton.jit
def reduce_min_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=float('inf'))
    result = tl.min(x, axis=0)
    tl.store(output_ptr + pid, result)

x = torch.arange(32, device=DEVICE, dtype=torch.float32)
out = torch.zeros(1, device=DEVICE)
reduce_min_kernel[(1,)](x, out, 32, BLOCK_SIZE=32)
check('reduce min', out.item() == 0.0, f'got {out.item()}')


# ── 14. Reduce sum multi-warp (128 elements, 4 warps) ──────────────────────
x = torch.ones(128, device=DEVICE)
out = torch.zeros(1, device=DEVICE)
try:
    reduce_sum_kernel[(1,)](x, out, 128, BLOCK_SIZE=128)
    check('reduce sum (multi-warp)', out.item() == 128.0, f'got {out.item()}')
except Exception as e:
    check('reduce sum (multi-warp)', False, str(e)[:200])


# ── 15. Reduce stress: random inputs, sum/max/min, multi-warp ─────────────
torch.manual_seed(42)

# Sum: random floats, multi-warp, check against torch.sum
x = torch.randn(128, device=DEVICE)
out = torch.zeros(1, device=DEVICE)
reduce_sum_kernel[(1,)](x, out, 128, BLOCK_SIZE=128)
expected = x.sum().item()
check('reduce sum (random, multi-warp)',
      abs(out.item() - expected) < 1e-3,
      f'got {out.item()}, expected {expected}')

# Max: random floats, multi-warp
@triton.jit
def reduce_max_mw_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=float('-inf'))
    result = tl.max(x, axis=0)
    tl.store(output_ptr + pid, result)

x = torch.randn(128, device=DEVICE)
out = torch.full((1,), float('-inf'), device=DEVICE)
reduce_max_mw_kernel[(1,)](x, out, 128, BLOCK_SIZE=128)
expected = x.max().item()
check('reduce max (random, multi-warp)',
      abs(out.item() - expected) < 1e-5,
      f'got {out.item()}, expected {expected}')

# Min: random floats, multi-warp
@triton.jit
def reduce_min_mw_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=float('inf'))
    result = tl.min(x, axis=0)
    tl.store(output_ptr + pid, result)

x = torch.randn(128, device=DEVICE)
out = torch.full((1,), float('inf'), device=DEVICE)
reduce_min_mw_kernel[(1,)](x, out, 128, BLOCK_SIZE=128)
expected = x.min().item()
check('reduce min (random, multi-warp)',
      abs(out.item() - expected) < 1e-5,
      f'got {out.item()}, expected {expected}')

# Sum: multi-block reduction (4 blocks × 128 threads each)
x = torch.randn(512, device=DEVICE)
out = torch.zeros(4, device=DEVICE)
reduce_sum_kernel[(4,)](x, out, 512, BLOCK_SIZE=128)
for i in range(4):
    expected = x[i*128:(i+1)*128].sum().item()
    ok = abs(out[i].item() - expected) < 1e-2
    if not ok:
        check('reduce sum (multi-block)', False,
              f'block {i}: got {out[i].item()}, expected {expected}')
        break
else:
    check('reduce sum (multi-block)', True, '')


# ── Atomic operations ─────────────────────────────────────────────────────────
@triton.jit
def atomic_add_kernel(ptr, val, N: tl.constexpr):
    pid = tl.program_id(0)
    tl.atomic_add(ptr, val)

# Single block: 128 threads all atomically add 1.0
out = torch.zeros(1, device=DEVICE)
atomic_add_kernel[(1,)](out, 1.0, N=128, num_warps=4)
torch.mps.synchronize()
check('atomic_add (f32, single block)', abs(out.item() - 128.0) < 1e-3,
      f'got {out.item()}, expected 128.0')

# Multi-block: 4 blocks × 128 threads
out = torch.zeros(1, device=DEVICE)
atomic_add_kernel[(4,)](out, 1.0, N=128, num_warps=4)
torch.mps.synchronize()
check('atomic_add (f32, multi-block)', abs(out.item() - 512.0) < 1e-3,
      f'got {out.item()}, expected 512.0')

# Integer atomic add
@triton.jit
def atomic_add_i32_kernel(ptr, val, N: tl.constexpr):
    tl.atomic_add(ptr, val)

out_i = torch.zeros(1, dtype=torch.int32, device=DEVICE)
atomic_add_i32_kernel[(4,)](out_i, 1, N=128, num_warps=4)
torch.mps.synchronize()
check('atomic_add (i32, multi-block)', out_i.item() == 512,
      f'got {out_i.item()}, expected 512')


# Integer atomic max
@triton.jit
def atomic_max_i32_kernel(ptr, val, N: tl.constexpr):
    tl.atomic_max(ptr, val)

out_i = torch.full((1,), -999, dtype=torch.int32, device=DEVICE)
atomic_max_i32_kernel[(4,)](out_i, 42, N=128, num_warps=4)
torch.mps.synchronize()
check('atomic_max (i32)', out_i.item() == 42,
      f'got {out_i.item()}, expected 42')

# Integer atomic min
@triton.jit
def atomic_min_i32_kernel(ptr, val, N: tl.constexpr):
    tl.atomic_min(ptr, val)

out_i = torch.full((1,), 999, dtype=torch.int32, device=DEVICE)
atomic_min_i32_kernel[(4,)](out_i, 7, N=128, num_warps=4)
torch.mps.synchronize()
check('atomic_min (i32)', out_i.item() == 7,
      f'got {out_i.item()}, expected 7')

# Integer atomic xchg
@triton.jit
def atomic_xchg_i32_kernel(ptr, val, N: tl.constexpr):
    tl.atomic_xchg(ptr, val)

out_i = torch.zeros(1, dtype=torch.int32, device=DEVICE)
atomic_xchg_i32_kernel[(1,)](out_i, 42, N=128, num_warps=4)
torch.mps.synchronize()
check('atomic_xchg (i32)', out_i.item() == 42,
      f'got {out_i.item()}, expected 42')

# Float atomic xchg
@triton.jit
def atomic_xchg_f32_kernel(ptr, val, N: tl.constexpr):
    tl.atomic_xchg(ptr, val)

out = torch.zeros(1, device=DEVICE)
atomic_xchg_f32_kernel[(1,)](out, 3.14, N=128, num_warps=4)
torch.mps.synchronize()
check('atomic_xchg (f32)', abs(out.item() - 3.14) < 1e-2,
      f'got {out.item()}, expected 3.14')

# Integer atomic and/or/xor
@triton.jit
def atomic_and_kernel(ptr, val, N: tl.constexpr):
    tl.atomic_and(ptr, val)

out_i = torch.full((1,), 0xFF, dtype=torch.int32, device=DEVICE)
atomic_and_kernel[(1,)](out_i, 0x0F, N=128, num_warps=4)
torch.mps.synchronize()
check('atomic_and (i32)', out_i.item() == 0x0F,
      f'got {out_i.item()}, expected {0x0F}')

@triton.jit
def atomic_or_kernel(ptr, val, N: tl.constexpr):
    tl.atomic_or(ptr, val)

out_i = torch.full((1,), 0xF0, dtype=torch.int32, device=DEVICE)
atomic_or_kernel[(1,)](out_i, 0x0F, N=128, num_warps=4)
torch.mps.synchronize()
check('atomic_or (i32)', out_i.item() == 0xFF,
      f'got {out_i.item()}, expected {0xFF}')

@triton.jit
def atomic_xor_kernel(ptr, val, N: tl.constexpr):
    tl.atomic_xor(ptr, val)

# 128 threads XOR with same value: even count → cancels out
out_i = torch.full((1,), 0xAA, dtype=torch.int32, device=DEVICE)
atomic_xor_kernel[(1,)](out_i, 0xFF, N=128, num_warps=4)
torch.mps.synchronize()
# 128 threads XOR 0xFF: even number of XORs cancels, result = 0xAA
check('atomic_xor (i32)', out_i.item() == 0xAA,
      f'got {out_i.item()}, expected {0xAA}')


# ── tl.where (select) ────────────────────────────────────────────────────────
@triton.jit
def where_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    # clamp negative values to zero (equivalent to relu)
    out = tl.where(x > 0, x, 0.0)
    tl.store(out_ptr + offs, out)

x = torch.randn(128, device=DEVICE)
out = torch.zeros(128, device=DEVICE)
where_kernel[(1,)](x, out, 128)
torch.mps.synchronize()
ref = torch.clamp(x, min=0)
check('where (relu)', (out - ref).abs().max().item() < 1e-5,
      f'max err {(out - ref).abs().max().item()}')


# ── Softmax ──────────────────────────────────────────────────────────────────
@triton.jit
def softmax_kernel(x_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    out = num / den
    tl.store(out_ptr + offs, out)

x = torch.randn(128, device=DEVICE)
out = torch.zeros(128, device=DEVICE)
softmax_kernel[(1,)](x, out, 128)
torch.mps.synchronize()
ref = torch.softmax(x, dim=0)
check('softmax', (out - ref).abs().max().item() < 1e-5,
      f'max err {(out - ref).abs().max().item()}')


# ── Batched softmax (one row per program) ────────────────────────────────────
@triton.jit
def softmax_rows_kernel(x_ptr, out_ptr, stride, N: tl.constexpr):
    row = tl.program_id(0)
    offs = row * stride + tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + offs, num / den)

M, N = 64, 128
x = torch.randn(M, N, device=DEVICE)
out = torch.zeros_like(x)
softmax_rows_kernel[(M,)](x, out, N, N=N)
torch.mps.synchronize()
ref = torch.softmax(x, dim=1)
check('softmax (64 rows)', (out - ref).abs().max().item() < 1e-5,
      f'max err {(out - ref).abs().max().item()}')


# ── Masked softmax (causal-style: mask upper triangle to -inf) ───────────────
@triton.jit
def masked_softmax_kernel(x_ptr, out_ptr, N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + row * N + offs)
    # causal mask: keep positions <= row
    mask = offs <= row
    x = tl.where(mask, x, float('-inf'))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    tl.store(out_ptr + row * N + offs, num / den)

N = 64
x = torch.randn(N, N, device=DEVICE)
out = torch.zeros_like(x)
masked_softmax_kernel[(N,)](x, out, N=N)
torch.mps.synchronize()
# reference: mask upper triangle then softmax
mask = torch.triu(torch.ones(N, N, device=DEVICE), diagonal=1).bool()
x_ref = x.masked_fill(mask, float('-inf'))
ref = torch.softmax(x_ref, dim=1)
check('masked softmax (causal)', (out - ref).abs().max().item() < 1e-5,
      f'max err {(out - ref).abs().max().item()}')


# ── Layer norm ────────────────────────────────────────────────────────────────
@triton.jit
def layernorm_kernel(x_ptr, w_ptr, b_ptr, out_ptr, stride, N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + row * stride + offs)
    mean = tl.sum(x, axis=0) / N
    x_zm = x - mean
    var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    x_hat = x_zm * rstd
    w = tl.load(w_ptr + offs)
    b = tl.load(b_ptr + offs)
    tl.store(out_ptr + row * stride + offs, x_hat * w + b)

M, N = 32, 128
x = torch.randn(M, N, device=DEVICE)
w = torch.randn(N, device=DEVICE)
b = torch.randn(N, device=DEVICE)
out = torch.zeros_like(x)
layernorm_kernel[(M,)](x, w, b, out, N, N=N)
torch.mps.synchronize()
ref = torch.nn.functional.layer_norm(x, [N], w, b, eps=1e-5)
check('layernorm (32x128)', (out - ref).abs().max().item() < 1e-4,
      f'max err {(out - ref).abs().max().item()}')


# ── ReLU (element-wise, multi-program) ───────────────────────────────────────
@triton.jit
def relu_kernel(x_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * N + tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    tl.store(out_ptr + offs, tl.where(x > 0, x, 0.0))

x = torch.randn(1024, device=DEVICE)
out = torch.zeros_like(x)
relu_kernel[(8,)](x, out, 128)
torch.mps.synchronize()
ref = torch.relu(x)
check('relu (1024 elems)', (out - ref).abs().max().item() == 0)


# ── Cross-entropy loss (per-row: -log(softmax[label])) ──────────────────────
@triton.jit
def cross_entropy_kernel(logits_ptr, labels_ptr, loss_ptr, stride, N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    logits = tl.load(logits_ptr + row * stride + offs)
    mx = tl.max(logits, axis=0)
    logits = logits - mx
    log_sum_exp = tl.log(tl.sum(tl.exp(logits), axis=0))
    label = tl.load(labels_ptr + row)
    log_prob = tl.load(logits_ptr + row * stride + label) - mx - log_sum_exp
    tl.store(loss_ptr + row, -log_prob)

M, N = 16, 64
logits = torch.randn(M, N, device=DEVICE)
labels = torch.randint(0, N, (M,), device=DEVICE, dtype=torch.int32)
loss = torch.zeros(M, device=DEVICE)
cross_entropy_kernel[(M,)](logits, labels, loss, N, N=N)
torch.mps.synchronize()
ref = torch.nn.functional.cross_entropy(logits, labels.long(), reduction='none')
check('cross_entropy (16x64)', (loss - ref).abs().max().item() < 1e-4,
      f'max err {(loss - ref).abs().max().item()}')


# ── Fused attention (Q @ K^T → causal mask → softmax → @ V) ─────────────────
@triton.jit
def fused_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_q, stride_k, stride_v, stride_o,
    N: tl.constexpr, D: tl.constexpr,
):
    row = tl.program_id(0)
    d_offs = tl.arange(0, D)

    # Load Q[row, :]
    q = tl.load(Q_ptr + row * stride_q + d_offs)

    # Compute attention scores: Q[row] @ K^T → (N,)
    col_offs = tl.arange(0, N)
    scores = tl.zeros([N], dtype=tl.float32)
    for j in range(D):
        q_j = tl.load(Q_ptr + row * stride_q + j)
        k_col = tl.load(K_ptr + col_offs * stride_k + j)
        scores += q_j * k_col

    # Scale
    scale: tl.constexpr = 1.0 / (D ** 0.5)
    scores = scores * scale

    # Causal mask
    mask = col_offs <= row
    scores = tl.where(mask, scores, float('-inf'))

    # Softmax
    mx = tl.max(scores, axis=0)
    scores = scores - mx
    exp_s = tl.exp(scores)
    sum_s = tl.sum(exp_s, axis=0)
    attn = exp_s / sum_s

    # Weighted sum of V
    out = tl.zeros([D], dtype=tl.float32)
    for j in range(N):
        a_j = tl.load(K_ptr + j * stride_k + 0 - 0)  # dummy to keep j in scope
        a_j = tl.sum(tl.where(col_offs == j, attn, 0.0), axis=0)
        v_row = tl.load(V_ptr + j * stride_v + d_offs)
        out += a_j * v_row

    tl.store(O_ptr + row * stride_o + d_offs, out)

N_seq, D_head = 32, 32
Q = torch.randn(N_seq, D_head, device=DEVICE)
K = torch.randn(N_seq, D_head, device=DEVICE)
V = torch.randn(N_seq, D_head, device=DEVICE)
O = torch.zeros_like(Q)
fused_attention_kernel[(N_seq,)](Q, K, V, O, D_head, D_head, D_head, D_head,
                                 N=N_seq, D=D_head)
torch.mps.synchronize()
# Reference: scaled dot-product attention with causal mask
scores_ref = (Q @ K.T) / (D_head ** 0.5)
causal = torch.triu(torch.ones(N_seq, N_seq, device=DEVICE), diagonal=1).bool()
scores_ref = scores_ref.masked_fill(causal, float('-inf'))
attn_ref = torch.softmax(scores_ref, dim=1)
O_ref = attn_ref @ V
check('fused_attention (32x32, causal)', (O - O_ref).abs().max().item() < 1e-3,
      f'max err {(O - O_ref).abs().max().item()}')


# ── Tiled GEMM (Triton tutorial style: multi-CTA, tiled) ────────────────────
@triton.jit
def gemm_tiled_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = N // BN
    pid_m = pid // num_n
    pid_n = pid % num_n

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)

    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k in range(0, K, BK):
        rk = k + tl.arange(0, BK)
        a = tl.load(A_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(B_ptr + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b)

    tl.store(C_ptr + rm[:, None] * N + rn[None, :], acc)

M, N, K = 64, 64, 64
BM, BN, BK = 16, 16, 16
A = torch.randn(M, K, device=DEVICE)
B = torch.randn(K, N, device=DEVICE)
C = torch.zeros(M, N, device=DEVICE)
grid = (M // BM * (N // BN),)
gemm_tiled_kernel[grid](A, B, C, M=M, N=N, K=K, BM=BM, BN=BN, BK=BK)
torch.mps.synchronize()
ref = A @ B
check('gemm_tiled (64x64x64, 16x16 tiles)', (C - ref).abs().max().item() < 1e-2,
      f'max err {(C - ref).abs().max().item()}')


# ── GEMM larger (128x128x64) ────────────────────────────────────────────────
M, N, K = 128, 128, 64
A = torch.randn(M, K, device=DEVICE)
B = torch.randn(K, N, device=DEVICE)
C = torch.zeros(M, N, device=DEVICE)
grid = (M // BM * (N // BN),)
gemm_tiled_kernel[grid](A, B, C, M=M, N=N, K=K, BM=BM, BN=BN, BK=BK)
torch.mps.synchronize()
ref = A @ B
check('gemm_tiled (128x128x64)', (C - ref).abs().max().item() < 1e-1,
      f'max err {(C - ref).abs().max().item()}')


# ── Summary ───────────────────────────────────────────────────────────────────
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f'\n{passed}/{total} passed')
