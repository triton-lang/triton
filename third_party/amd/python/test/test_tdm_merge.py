"""Tests for adjacent TDM-copy merging on gfx1250.

Adjacent `async_load`s with compatible `warp_used_hint` masks should fuse into
one `tensor_load_to_lds`.  The prepare pass can also generate hints for adjacent
unhinted copies unless `TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS=1` is set.

Compile-only tests count AMDGCN instructions; runtime tests compare against a
torch CPU reference and are skipped off gfx1250.
"""

import re

import pytest
import torch

import triton
from triton.backends.compiler import GPUTarget
from triton._internal_testing import is_hip_gfx1250
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


def _use_tdm_hint(request):
    if isinstance(request, bool):
        return request
    return not request.config.getoption("--tdm-disable-hint")


def _hints(use_hint, *hints):
    """Return real hints when enabled, otherwise unhinted loads."""
    return hints if use_hint else (None, ) * len(hints)


# Shared kernel building blocks.


@gluon.jit
def _stage_input(ptr, M, N, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr, layout: ttgl.constexpr):
    """Build a descriptor and independent staging buffer for one TDM copy."""
    desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=ptr, shape=(M, N), strides=(N, 1),
                                                       block_shape=(BLOCK_M, BLOCK_N), layout=layout)
    buf = ttgl.allocate_shared_memory(desc.dtype, [1] + desc.block_shape, desc.layout).index(0)
    return desc, buf


@gluon.jit
def _store_tile(out_ptr, value, off_m, off_n, M, N, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr,
                layout: ttgl.constexpr):
    """Masked store of a [BLOCK_M, BLOCK_N] result tile at (off_m, off_n)."""
    offs_m = off_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, layout))
    offs_n = off_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, layout))
    offs = (offs_m[:, None] * N) + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ttgl.store(out_ptr + offs, value, mask=mask)


# 2-way merge: adjacent loads fuse when their hints are disjoint.


@gluon.jit
def vector_add_tdm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    HINT_A: ttgl.constexpr,
    HINT_B: ttgl.constexpr,
):
    """Two-tile vector add with adjacent TDM loads."""
    num_warps: ttgl.constexpr = ttgl.num_warps()
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc, a_buf = _stage_input(a_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)
    b_desc, b_buf = _stage_input(b_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)

    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_wait(0)  # "wait for everything"; correct under any merge

    c = a_buf.load(layout=BLOCKED_LAYOUT) + b_buf.load(layout=BLOCKED_LAYOUT)
    _store_tile(c_ptr, c, off_m, off_n, M, N, BLOCK_M, BLOCK_N, BLOCKED_LAYOUT)


# Layout: (HINT_A, HINT_B, expected_merge, id).  Bit `i` selects warp `i`.
_HINT_PARAMS = [
    # minimal mergeable pair: K=1 each, union {0,1}
    (0b00000001, 0b00000010, True, "merge_single_warp_pair"),
    # split warps in half: K=4 each, union covers all 8 warps.
    (0b00001111, 0b11110000, True, "merge_lo_hi"),
    # strided cosets: every-other warp + complement.
    (0b01010101, 0b10101010, True, "merge_strided"),
    # lo/hi pair cosets: basis {0,2}, two K=4 quartets.
    (0b00110011, 0b11001100, True, "merge_lo_hi_pairs"),
    # partial coverage: K=2 each, union covers 4 of 8 warps (rest idle).
    (0b00000011, 0b00001100, True, "merge_partial_K4_idle"),
    # pairwise-disjoint hints whose union is not itself a coset.
    (0b00000001, 0b00001000, True, "merge_disjoint_K1_noncoset_union"),
]


def _param_args(p):
    """Strip trailing id from a parametrised entry."""
    return p[:-1]


def _param_id(p):
    return p[-1]


def _tensor_load_to_lds_lines(amdgcn: str) -> list[str]:
    """Return emitted TDM load instruction lines, ignoring metadata/comments."""
    lines = []
    for line in amdgcn.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith((".", "#", ";")):
            continue
        if re.search(r"\btensor_load_to_lds\b", stripped):
            lines.append(stripped)
    return lines


def _assert_tensor_load_count(amdgcn: str, expected: int, context: str):
    lines = _tensor_load_to_lds_lines(amdgcn)
    actual = len(lines)
    assert actual == expected, (f"expected {expected} tensor_load_to_lds instruction(s) for {context}, "
                                f"got {actual}\nMatched tensor_load_to_lds lines:\n" + "\n".join(lines))


def _compile_amdgcn(fn, ptr_names, constexprs, *, ptr_ty="*fp16", num_warps=8) -> str:
    """Compile `fn` for gfx1250 and return its AMDGCN asm.

    The signature is `{ptrs: ptr_ty, M/N: i32, <constexpr keys>: constexpr}`,
    matching every kernel here (pointer args, then M, N, then constexprs).
    """
    signature = {p: ptr_ty for p in ptr_names}
    signature["M"] = signature["N"] = "i32"
    signature.update({name: "constexpr" for name in constexprs})
    k = triton.compile(
        gluon._runtime.GluonASTSource(fn=fn, signature=signature, constexprs=constexprs),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": num_warps},
    )
    return k.asm["amdgcn"]


def _run_vector_add(kernel, n_inputs, block, hints, *, num_warps=8):
    """Launch `kernel(*inputs, out, M, N, BLOCK_M, BLOCK_N, *hints)` and check
    `out == sum(inputs)` against a torch-CPU reference.  Merging is perf-only,
    so the result is identical whether or not the copies fuse.
    """
    M, N = 256, 512
    BLOCK_M, BLOCK_N = block
    torch.manual_seed(0)
    inputs = [torch.rand((M, N), dtype=torch.float16) for _ in range(n_inputs)]
    dev_inputs = [t.cuda() for t in inputs]
    out = torch.empty((M, N), dtype=torch.float16, device="cuda")

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    kernel[grid](*dev_inputs, out, M, N, BLOCK_M, BLOCK_N, *hints, num_warps=num_warps)

    expected = inputs[0]
    for t in inputs[1:]:
        expected = expected + t
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


_COMPILE_BLOCK_SHAPES = [(64, 64), (32, 128)]


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "HINT_A,HINT_B,expected_merge",
    [_param_args(p) for p in _HINT_PARAMS],
    ids=[_param_id(p) for p in _HINT_PARAMS],
)
def test_compile_vector_add_tdm(BLOCK_M, BLOCK_N, HINT_A, HINT_B, expected_merge, request):
    """Compile-only: asserts 1 fused vs 2 separate `tensor_load_to_lds`."""
    use_tdm_hint = _use_tdm_hint(request)
    ha, hb = _hints(use_tdm_hint, HINT_A, HINT_B)
    amdgcn = _compile_amdgcn(vector_add_tdm_kernel, ["a_ptr", "b_ptr", "c_ptr"],
                             {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "HINT_A": ha, "HINT_B": hb})
    context = f"HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}"
    fused = use_tdm_hint and expected_merge
    _assert_tensor_load_count(amdgcn, 1 if fused else 2, context)


def test_compile_vector_add_tdm_auto_merge_env_toggle(monkeypatch):
    """Compile-only: env toggles generated hints for adjacent unhinted copies."""
    env_var = "TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS"

    def compile_unhinted(block_m, block_n):
        return _compile_amdgcn(vector_add_tdm_kernel, ["a_ptr", "b_ptr", "c_ptr"],
                               {"BLOCK_M": block_m, "BLOCK_N": block_n, "HINT_A": None, "HINT_B": None})

    for env, block, expected in [("1", (64, 64), 2), ("0", (32, 128), 1), ("1", (128, 64), 2)]:
        monkeypatch.setenv(env_var, env)
        _assert_tensor_load_count(compile_unhinted(*block), expected, f"env={env} generated hints")


_RUNTIME_BLOCK_SHAPES = [(64, 64), (128, 64)]


@pytest.mark.skipif(not is_hip_gfx1250(), reason="TDM is only tested on gfx1250.")
@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _RUNTIME_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "HINT_A,HINT_B,expected_merge",
    [_param_args(p) for p in _HINT_PARAMS],
    ids=[_param_id(p) for p in _HINT_PARAMS],
)
def test_runtime_vector_add_tdm(BLOCK_M, BLOCK_N, HINT_A, HINT_B, expected_merge, request):
    """Runtime: c = a + b vs torch CPU reference; merging is perf-only."""
    _run_vector_add(vector_add_tdm_kernel, 2, (BLOCK_M, BLOCK_N), _hints(_use_tdm_hint(request), HINT_A, HINT_B))


# 3-way merge: covers the non-uniform generated-hint shape.
#   num_warps=4: {0b0011, 0b0100, 0b1000}
#   num_warps=8: {0b00001111, 0b00110000, 0b11000000}


@gluon.jit
def vector_add_tdm_kernel_3way(
    a_ptr,
    b_ptr,
    c_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    HINT_A: ttgl.constexpr,
    HINT_B: ttgl.constexpr,
    HINT_C: ttgl.constexpr,
):
    """Sum three tiles via three adjacent TDM copies."""
    num_warps: ttgl.constexpr = ttgl.num_warps()
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc, a_buf = _stage_input(a_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)
    b_desc, b_buf = _stage_input(b_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)
    c_desc, c_buf = _stage_input(c_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)

    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_load(c_desc, [off_m, off_n], c_buf, warp_used_hint=HINT_C)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    out = a_buf.load(layout=BLOCKED_LAYOUT) + b_buf.load(layout=BLOCKED_LAYOUT) + c_buf.load(layout=BLOCKED_LAYOUT)
    _store_tile(out_ptr, out, off_m, off_n, M, N, BLOCK_M, BLOCK_N, BLOCKED_LAYOUT)


_HINT_PARAMS_3WAY = [
    (8, 0b00001111, 0b00110000, 0b11000000, True, "three_way_8w_generated_shape"),
    (4, 0b0011, 0b0100, 0b1000, True, "three_way_4w_generated_shape"),
    # Pairwise-disjoint hints whose union is not a coset.
    (8, 0b00000001, 0b00001000, 0b00010000, True, "three_way_noncoset_union"),
]


def _compile_3way(num_warps, hints=(None, None, None), block=(64, 64)) -> str:
    a, b, c = hints
    return _compile_amdgcn(vector_add_tdm_kernel_3way, ["a_ptr", "b_ptr", "c_ptr", "out_ptr"],
                           {"BLOCK_M": block[0], "BLOCK_N": block[1], "HINT_A": a, "HINT_B": b, "HINT_C": c},
                           num_warps=num_warps)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "NUM_WARPS,HINT_A,HINT_B,HINT_C,expected_merge",
    [_param_args(p) for p in _HINT_PARAMS_3WAY],
    ids=[_param_id(p) for p in _HINT_PARAMS_3WAY],
)
def test_compile_vector_add_tdm_3way(BLOCK_M, BLOCK_N, NUM_WARPS, HINT_A, HINT_B, HINT_C, expected_merge, request):
    """Compile-only: 3 adjacent copies fuse when their hints are pairwise disjoint."""
    use_tdm_hint = _use_tdm_hint(request)
    amdgcn = _compile_3way(NUM_WARPS, _hints(use_tdm_hint, HINT_A, HINT_B, HINT_C), (BLOCK_M, BLOCK_N))
    context = f"NUM_WARPS={NUM_WARPS}, HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}, HINT_C=0b{HINT_C:08b}"
    fused = use_tdm_hint and expected_merge
    _assert_tensor_load_count(amdgcn, 1 if fused else 3, context)


def test_compile_vector_add_tdm_3way_auto_merge_env_toggle(monkeypatch):
    """Compile-only: env can generate 3-way hints for adjacent unhinted copies."""
    env_var = "TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS"
    # (env, num_warps, block, expected): generation runs for 4 and 8 warps.
    for env, warps, block, expected in [("1", 8, (64, 64), 3), ("0", 8, (32, 128), 1), ("0", 4, (128, 64), 1),
                                        ("1", 4, (64, 128), 3)]:
        monkeypatch.setenv(env_var, env)
        amdgcn = _compile_3way(warps, block=block)
        _assert_tensor_load_count(amdgcn, expected, f"3-way env={env} warps={warps}")


# 4-way merge: exercises the N=4 member-predicate path.


@gluon.jit
def vector_add_tdm_kernel_4way(
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    out_ptr,
    M,
    N,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    HINT_A: ttgl.constexpr,
    HINT_B: ttgl.constexpr,
    HINT_C: ttgl.constexpr,
    HINT_D: ttgl.constexpr,
):
    """Sum four tiles via four adjacent TDM copies."""
    num_warps: ttgl.constexpr = ttgl.num_warps()
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc, a_buf = _stage_input(a_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)
    b_desc, b_buf = _stage_input(b_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)
    c_desc, c_buf = _stage_input(c_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)
    d_desc, d_buf = _stage_input(d_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)

    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_load(c_desc, [off_m, off_n], c_buf, warp_used_hint=HINT_C)
    ttgl.amd.gfx1250.tdm.async_load(d_desc, [off_m, off_n], d_buf, warp_used_hint=HINT_D)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    out = (a_buf.load(layout=BLOCKED_LAYOUT) + b_buf.load(layout=BLOCKED_LAYOUT) + c_buf.load(layout=BLOCKED_LAYOUT) +
           d_buf.load(layout=BLOCKED_LAYOUT))
    _store_tile(out_ptr, out, off_m, off_n, M, N, BLOCK_M, BLOCK_N, BLOCKED_LAYOUT)


# 4-way positive cases: K=1 partial coverage and K=2 full coverage.
_HINT_PARAMS_4WAY = [
    (0b00000001, 0b00000010, 0b00000100, 0b00001000, "four_K1_lo4_union"),
    (0b00000011, 0b00001100, 0b00110000, 0b11000000, "four_K2_full_union"),
]


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "HINT_A,HINT_B,HINT_C,HINT_D",
    [_param_args(p) for p in _HINT_PARAMS_4WAY],
    ids=[_param_id(p) for p in _HINT_PARAMS_4WAY],
)
def test_compile_vector_add_tdm_4way(BLOCK_M, BLOCK_N, HINT_A, HINT_B, HINT_C, HINT_D, request):
    """Compile-only: every quadruple must fuse to a single intrinsic."""
    use_tdm_hint = _use_tdm_hint(request)
    ha, hb, hc, hd = _hints(use_tdm_hint, HINT_A, HINT_B, HINT_C, HINT_D)
    amdgcn = _compile_amdgcn(
        vector_add_tdm_kernel_4way, ["a_ptr", "b_ptr", "c_ptr", "d_ptr", "out_ptr"],
        {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "HINT_A": ha, "HINT_B": hb, "HINT_C": hc, "HINT_D": hd})
    context = (f"HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}, "
               f"HINT_C=0b{HINT_C:08b}, HINT_D=0b{HINT_D:08b}")
    _assert_tensor_load_count(amdgcn, 1 if use_tdm_hint else 4, context)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="TDM is only tested on gfx1250.")
@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _RUNTIME_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "HINT_A,HINT_B,HINT_C,HINT_D",
    [_param_args(p) for p in _HINT_PARAMS_4WAY],
    ids=[_param_id(p) for p in _HINT_PARAMS_4WAY],
)
def test_runtime_vector_add_tdm_4way(BLOCK_M, BLOCK_N, HINT_A, HINT_B, HINT_C, HINT_D, request):
    """Runtime: out = a+b+c+d; checks the 4-way fused intrinsic routes
    each member's bytes to its own buffer."""
    _run_vector_add(vector_add_tdm_kernel_4way, 4, (BLOCK_M, BLOCK_N),
                    _hints(_use_tdm_hint(request), HINT_A, HINT_B, HINT_C, HINT_D))


# Heterogeneous destination MemDescTypes should still fuse.


@gluon.jit
def heterogeneous_tdm_merge_kernel(
    a_ptr,
    b_ptr,
    as_ptr,
    bs_ptr,
    M,
    N,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    BLOCK_N_B: ttgl.constexpr,
    BLOCK_SCALE_M: ttgl.constexpr,
    BLOCK_SCALE_N: ttgl.constexpr,
    HINT_A: ttgl.constexpr,
    HINT_B: ttgl.constexpr,
    HINT_AS: ttgl.constexpr,
    HINT_BS: ttgl.constexpr,
):
    A_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])
    B_SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    SCALE_SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc, a_buf = _stage_input(a_ptr, M, N, BLOCK_M, BLOCK_N, A_SHARED_LAYOUT)
    b_desc, b_buf = _stage_input(b_ptr, M, N, BLOCK_M, BLOCK_N_B, B_SHARED_LAYOUT)
    as_desc, as_buf = _stage_input(as_ptr, M, N, BLOCK_SCALE_M, BLOCK_SCALE_N, SCALE_SHARED_LAYOUT)
    bs_desc, bs_buf = _stage_input(bs_ptr, M, N, BLOCK_SCALE_M, BLOCK_SCALE_N, SCALE_SHARED_LAYOUT)

    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_load(as_desc, [off_m, off_n], as_buf, warp_used_hint=HINT_AS)
    ttgl.amd.gfx1250.tdm.async_load(bs_desc, [off_m, off_n], bs_buf, warp_used_hint=HINT_BS)
    ttgl.amd.gfx1250.tdm.async_wait(0)


def test_compile_heterogeneous_tdm_merge(request):
    """Compile-only: A/B/AS/BS destination MemDescTypes still fuse."""
    use_tdm_hint = _use_tdm_hint(request)
    ha, hb, has_, hbs = _hints(use_tdm_hint, 0b00010001, 0b00100010, 0b01000100, 0b10001000)
    amdgcn = _compile_amdgcn(
        heterogeneous_tdm_merge_kernel, ["a_ptr", "b_ptr", "as_ptr", "bs_ptr"], {
            "BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_N_B": 2048, "BLOCK_SCALE_M": 64, "BLOCK_SCALE_N": 32, "HINT_A": ha,
            "HINT_B": hb, "HINT_AS": has_, "HINT_BS": hbs
        }, ptr_ty="*i8")
    _assert_tensor_load_count(amdgcn, 1 if use_tdm_hint else 4, "heterogeneous A/B/AS/BS destination MemDescTypes")


# Cache modifiers must match for a merge.


@gluon.jit
def vector_add_tdm_kernel_cache(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    BLOCK_M: ttgl.constexpr,
    BLOCK_N: ttgl.constexpr,
    HINT_A: ttgl.constexpr,
    HINT_B: ttgl.constexpr,
    CACHE_A: ttgl.constexpr,
    CACHE_B: ttgl.constexpr,
):
    """Two-tile vector add with explicit per-copy cache modifiers (same strings
    as `tt.load`'s `cache_modifier`: `""`, `.ca`, `.cg`, ...)."""
    num_warps: ttgl.constexpr = ttgl.num_warps()
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc, a_buf = _stage_input(a_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)
    b_desc, b_buf = _stage_input(b_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)

    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A, cache_modifier=CACHE_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B, cache_modifier=CACHE_B)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    c = a_buf.load(layout=BLOCKED_LAYOUT) + b_buf.load(layout=BLOCKED_LAYOUT)
    _store_tile(c_ptr, c, off_m, off_n, M, N, BLOCK_M, BLOCK_N, BLOCKED_LAYOUT)


# Layout: (CACHE_A, CACHE_B, expected_merge, id).  Hints are pinned.
_CACHE_PARAMS = [
    # same cache: rule 7 satisfied
    ("", "", True, "same_default"),
    (".cg", ".cg", True, "same_cg"),
    # mismatched cache: rule 7 forces a split
    ("", ".cg", False, "default_vs_cg"),
    (".ca", ".cg", False, "ca_vs_cg"),
]


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "CACHE_A,CACHE_B,expected_merge",
    [_param_args(p) for p in _CACHE_PARAMS],
    ids=[_param_id(p) for p in _CACHE_PARAMS],
)
def test_compile_vector_add_tdm_cache(BLOCK_M, BLOCK_N, CACHE_A, CACHE_B, expected_merge, request):
    """Compile-only: asserts rule 7 (matching cache modifiers)."""
    use_tdm_hint = _use_tdm_hint(request)
    # Hints pinned to the lo/hi split so only the cache modifier decides fusion.
    ha, hb = _hints(use_tdm_hint, 0b00001111, 0b11110000)
    amdgcn = _compile_amdgcn(
        vector_add_tdm_kernel_cache, ["a_ptr", "b_ptr", "c_ptr"],
        {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "HINT_A": ha, "HINT_B": hb, "CACHE_A": CACHE_A, "CACHE_B": CACHE_B})
    context = f"CACHE_A={CACHE_A!r}, CACHE_B={CACHE_B!r}"
    fused = use_tdm_hint and expected_merge
    _assert_tensor_load_count(amdgcn, 1 if fused else 2, context)


if __name__ == "__main__":
    # Smoke test: iterate all cookbook entries at one block size.
    # Run as `python test_tdm_merge.py` on a gfx1250 device.
    if not is_hip_gfx1250():
        raise SystemExit("This script requires a gfx1250 device.")
    print("[2-way: vector_add_tdm_kernel]")
    for p in _HINT_PARAMS:
        ha, hb, em, ident = p
        print(f"-- {ident}: HINT_A=0b{ha:08b}, HINT_B=0b{hb:08b}, expected_merge={em}")
        test_runtime_vector_add_tdm(64, 64, ha, hb, em, True)
        print("   OK")
    print("[4-way: vector_add_tdm_kernel_4way]")
    for p in _HINT_PARAMS_4WAY:
        ha, hb, hc, hd, ident = p
        print(f"-- {ident}: A=0b{ha:08b} B=0b{hb:08b} C=0b{hc:08b} D=0b{hd:08b}")
        test_runtime_vector_add_tdm_4way(64, 64, ha, hb, hc, hd, True)
        print("   OK")
