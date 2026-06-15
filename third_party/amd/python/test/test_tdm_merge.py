"""Tests for TDM-copy materialization on gfx1250.

Explicit `async_load_fused` calls lower to one `tensor_load_to_lds`.  Regular
`async_load`s with user-provided `warp_used_hint` masks stay separate; the
materialization pass only auto-merges adjacent unhinted copies unless
`TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS=1` is set.

Compile-only tests count AMDGCN instructions; runtime tests are opt-in for
gfx1250 and compare against a torch CPU reference.
"""

import re
import os

import pytest
import torch

import triton
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl


def _require_tdm_runtime():
    if os.environ.get("TRITON_AMD_RUN_TDM_RUNTIME_TESTS") != "1":
        pytest.skip("TDM runtime tests require gfx1250; set TRITON_AMD_RUN_TDM_RUNTIME_TESTS=1 to run them")


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


@gluon.jit
def _position_input(desc, off_m, off_n):
    """Position a descriptor before the adjacent TDM-copy run."""
    return ttgl.amd.gfx1250.tdm.update_tensor_descriptor(desc, add_offsets=[off_m, off_n], pred=True)


# 2-way copies: explicit fused loads merge, while regular hinted loads do not.


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
    a_desc = _position_input(a_desc, off_m, off_n)
    b_desc = _position_input(b_desc, off_m, off_n)

    ttgl.amd.gfx1250.tdm.async_load(a_desc, dest=a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, dest=b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_wait(0)  # "wait for everything"; correct under any merge

    c = a_buf.load(layout=BLOCKED_LAYOUT) + b_buf.load(layout=BLOCKED_LAYOUT)
    _store_tile(c_ptr, c, off_m, off_n, M, N, BLOCK_M, BLOCK_N, BLOCKED_LAYOUT)


@gluon.jit
def vector_add_tdm_explicit_fused_kernel(
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
    """Two-tile vector add using the explicit fused TDM load API."""
    num_warps: ttgl.constexpr = ttgl.num_warps()
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc, a_buf = _stage_input(a_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)
    b_desc, b_buf = _stage_input(b_ptr, M, N, BLOCK_M, BLOCK_N, SHARED_LAYOUT)
    a_desc = ttgl.amd.gfx1250.tdm.update_tensor_descriptor(a_desc, add_offsets=[off_m, off_n], pred=True)
    b_desc = ttgl.amd.gfx1250.tdm.update_tensor_descriptor(b_desc, add_offsets=[off_m, off_n], pred=True)

    ttgl.amd.gfx1250.tdm.async_load_fused([(a_desc, a_buf, HINT_A), (b_desc, b_buf, HINT_B)])
    ttgl.amd.gfx1250.tdm.async_wait(0)

    c = a_buf.load(layout=BLOCKED_LAYOUT) + b_buf.load(layout=BLOCKED_LAYOUT)
    _store_tile(c_ptr, c, off_m, off_n, M, N, BLOCK_M, BLOCK_N, BLOCKED_LAYOUT)


# Layout: (HINT_A, HINT_B, id).  Bit `i` selects warp `i`.
_HINT_PARAMS = [
    # minimal legal pair: K=1 each, union {0,1}
    (0b00000001, 0b00000010, "single_warp_pair"),
    # split warps in half: K=4 each, union covers all 8 warps.
    (0b00001111, 0b11110000, "lo_hi"),
    # strided cosets: every-other warp + complement.
    (0b01010101, 0b10101010, "strided"),
    # lo/hi pair cosets: basis {0,2}, two K=4 quartets.
    (0b00110011, 0b11001100, "lo_hi_pairs"),
    # partial coverage: K=2 each, union covers 4 of 8 warps (rest idle).
    (0b00000011, 0b00001100, "partial_K4_idle"),
    # pairwise-disjoint hints whose union is not itself a coset.
    (0b00000001, 0b00001000, "disjoint_K1_noncoset_union"),
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
    # These tests assert current codegen details.  The Triton disk-cache key
    # includes libtriton and language files, but not the Python backend pipeline,
    # so force recompilation to avoid stale assembly after pass-pipeline edits.
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.always_compile = True
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
    "HINT_A,HINT_B",
    [_param_args(p) for p in _HINT_PARAMS],
    ids=[_param_id(p) for p in _HINT_PARAMS],
)
def test_compile_vector_add_tdm(BLOCK_M, BLOCK_N, HINT_A, HINT_B):
    """Compile-only: regular hinted loads stay separate."""
    amdgcn = _compile_amdgcn(
        vector_add_tdm_kernel, ["a_ptr", "b_ptr", "c_ptr"], {
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "HINT_A": HINT_A, "HINT_B": HINT_B
        })
    context = f"HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}"
    _assert_tensor_load_count(amdgcn, 2, context)


def test_compile_vector_add_tdm_explicit_fused():
    """Compile-only: explicit fused Gluon API lowers to one TDM intrinsic."""
    amdgcn = _compile_amdgcn(
        vector_add_tdm_explicit_fused_kernel, ["a_ptr", "b_ptr", "c_ptr"], {
            "BLOCK_M": 64, "BLOCK_N": 64, "HINT_A": 0b00001111, "HINT_B": 0b11110000
        })
    _assert_tensor_load_count(amdgcn, 1, "explicit async_load_fused")


def test_compile_vector_add_tdm_auto_merge_env_toggle(monkeypatch):
    """Compile-only: env toggles generated hints for adjacent unhinted copies."""
    env_var = "TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS"

    def compile_unhinted(block_m, block_n):
        return _compile_amdgcn(
            vector_add_tdm_kernel, ["a_ptr", "b_ptr", "c_ptr"], {
                "BLOCK_M": block_m, "BLOCK_N": block_n, "HINT_A": None, "HINT_B": None
            })

    for env, block, expected in [("1", (64, 64), 2), ("0", (32, 128), 1), ("1", (128, 64), 2)]:
        monkeypatch.setenv(env_var, env)
        _assert_tensor_load_count(compile_unhinted(*block), expected, f"env={env} generated hints")


_RUNTIME_BLOCK_SHAPES = [(64, 64), (128, 64)]


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _RUNTIME_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "HINT_A,HINT_B",
    [_param_args(p) for p in _HINT_PARAMS],
    ids=[_param_id(p) for p in _HINT_PARAMS],
)
def test_runtime_vector_add_tdm(BLOCK_M, BLOCK_N, HINT_A, HINT_B):
    """Runtime: c = a + b vs torch CPU reference; merging is perf-only."""
    _require_tdm_runtime()
    _run_vector_add(vector_add_tdm_kernel, 2, (BLOCK_M, BLOCK_N), (HINT_A, HINT_B))


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
    a_desc = _position_input(a_desc, off_m, off_n)
    b_desc = _position_input(b_desc, off_m, off_n)
    c_desc = _position_input(c_desc, off_m, off_n)

    ttgl.amd.gfx1250.tdm.async_load(a_desc, dest=a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, dest=b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_load(c_desc, dest=c_buf, warp_used_hint=HINT_C)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    out = a_buf.load(layout=BLOCKED_LAYOUT) + b_buf.load(layout=BLOCKED_LAYOUT) + c_buf.load(layout=BLOCKED_LAYOUT)
    _store_tile(out_ptr, out, off_m, off_n, M, N, BLOCK_M, BLOCK_N, BLOCKED_LAYOUT)


_HINT_PARAMS_3WAY = [
    (8, 0b00001111, 0b00110000, 0b11000000, "three_way_8w_generated_shape"),
    (4, 0b0011, 0b0100, 0b1000, "three_way_4w_generated_shape"),
    # Pairwise-disjoint hints whose union is not a coset.
    (8, 0b00000001, 0b00001000, 0b00010000, "three_way_noncoset_union"),
]


def _compile_3way(num_warps, hints=(None, None, None), block=(64, 64)) -> str:
    a, b, c = hints
    return _compile_amdgcn(
        vector_add_tdm_kernel_3way, ["a_ptr", "b_ptr", "c_ptr", "out_ptr"], {
            "BLOCK_M": block[0], "BLOCK_N": block[1], "HINT_A": a, "HINT_B": b, "HINT_C": c
        }, num_warps=num_warps)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "NUM_WARPS,HINT_A,HINT_B,HINT_C",
    [_param_args(p) for p in _HINT_PARAMS_3WAY],
    ids=[_param_id(p) for p in _HINT_PARAMS_3WAY],
)
def test_compile_vector_add_tdm_3way(BLOCK_M, BLOCK_N, NUM_WARPS, HINT_A, HINT_B, HINT_C):
    """Compile-only: regular hinted 3-way loads stay separate."""
    amdgcn = _compile_3way(NUM_WARPS, (HINT_A, HINT_B, HINT_C), (BLOCK_M, BLOCK_N))
    context = f"NUM_WARPS={NUM_WARPS}, HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}, HINT_C=0b{HINT_C:08b}"
    _assert_tensor_load_count(amdgcn, 3, context)


def test_compile_vector_add_tdm_3way_auto_merge_env_toggle(monkeypatch):
    """Compile-only: env can generate 3-way hints for adjacent unhinted copies."""
    env_var = "TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS"
    # (env, num_warps, block, expected): generation runs for 4 and 8 warps.
    for env, warps, block, expected in [("1", 8, (64, 64), 3), ("0", 8, (32, 128), 1),
                                        ("0", 4, (128, 64), 1), ("1", 4, (64, 128), 3)]:
        monkeypatch.setenv(env_var, env)
        amdgcn = _compile_3way(warps, block=block)
        _assert_tensor_load_count(amdgcn, expected, f"3-way env={env} warps={warps}")


# 4-way auto merge: exercises the N=4 member-predicate path.


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
    a_desc = _position_input(a_desc, off_m, off_n)
    b_desc = _position_input(b_desc, off_m, off_n)
    c_desc = _position_input(c_desc, off_m, off_n)
    d_desc = _position_input(d_desc, off_m, off_n)

    ttgl.amd.gfx1250.tdm.async_load(a_desc, dest=a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, dest=b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_load(c_desc, dest=c_buf, warp_used_hint=HINT_C)
    ttgl.amd.gfx1250.tdm.async_load(d_desc, dest=d_buf, warp_used_hint=HINT_D)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    out = (a_buf.load(layout=BLOCKED_LAYOUT) + b_buf.load(layout=BLOCKED_LAYOUT) +
           c_buf.load(layout=BLOCKED_LAYOUT) + d_buf.load(layout=BLOCKED_LAYOUT))
    _store_tile(out_ptr, out, off_m, off_n, M, N, BLOCK_M, BLOCK_N, BLOCKED_LAYOUT)


# 4-way legal hint cases: K=1 partial coverage and K=2 full coverage.
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
def test_compile_vector_add_tdm_4way(BLOCK_M, BLOCK_N, HINT_A, HINT_B, HINT_C, HINT_D):
    """Compile-only: regular hinted 4-way loads stay separate."""
    amdgcn = _compile_amdgcn(
        vector_add_tdm_kernel_4way, ["a_ptr", "b_ptr", "c_ptr", "d_ptr", "out_ptr"], {
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "HINT_A": HINT_A, "HINT_B": HINT_B,
            "HINT_C": HINT_C, "HINT_D": HINT_D
        })
    context = (f"HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}, "
               f"HINT_C=0b{HINT_C:08b}, HINT_D=0b{HINT_D:08b}")
    _assert_tensor_load_count(amdgcn, 4, context)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _RUNTIME_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "HINT_A,HINT_B,HINT_C,HINT_D",
    [_param_args(p) for p in _HINT_PARAMS_4WAY],
    ids=[_param_id(p) for p in _HINT_PARAMS_4WAY],
)
def test_runtime_vector_add_tdm_4way(BLOCK_M, BLOCK_N, HINT_A, HINT_B, HINT_C, HINT_D):
    """Runtime: out = a+b+c+d; merging is perf-only."""
    _require_tdm_runtime()
    _run_vector_add(vector_add_tdm_kernel_4way, 4, (BLOCK_M, BLOCK_N), (HINT_A, HINT_B, HINT_C, HINT_D))


# Auto merge can handle heterogeneous destination MemDescTypes.


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
    a_desc = _position_input(a_desc, off_m, off_n)
    b_desc = _position_input(b_desc, off_m, off_n)
    as_desc = _position_input(as_desc, off_m, off_n)
    bs_desc = _position_input(bs_desc, off_m, off_n)

    ttgl.amd.gfx1250.tdm.async_load(a_desc, dest=a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, dest=b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_load(as_desc, dest=as_buf, warp_used_hint=HINT_AS)
    ttgl.amd.gfx1250.tdm.async_load(bs_desc, dest=bs_buf, warp_used_hint=HINT_BS)
    ttgl.amd.gfx1250.tdm.async_wait(0)


def test_compile_heterogeneous_tdm_hints_stay_separate():
    """Compile-only: heterogeneous hinted loads stay separate."""
    amdgcn = _compile_amdgcn(
        heterogeneous_tdm_merge_kernel, ["a_ptr", "b_ptr", "as_ptr", "bs_ptr"], {
            "BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_N_B": 2048,
            "BLOCK_SCALE_M": 64, "BLOCK_SCALE_N": 32,
            "HINT_A": 0b00010001, "HINT_B": 0b00100010,
            "HINT_AS": 0b01000100, "HINT_BS": 0b10001000
        }, ptr_ty="*i8")
    _assert_tensor_load_count(amdgcn, 4, "heterogeneous hinted A/B/AS/BS loads")


def test_compile_heterogeneous_tdm_auto_merge():
    """Compile-only: heterogeneous unhinted loads auto-fuse."""
    amdgcn = _compile_amdgcn(
        heterogeneous_tdm_merge_kernel, ["a_ptr", "b_ptr", "as_ptr", "bs_ptr"], {
            "BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_N_B": 2048,
            "BLOCK_SCALE_M": 64, "BLOCK_SCALE_N": 32,
            "HINT_A": None, "HINT_B": None, "HINT_AS": None, "HINT_BS": None
        }, ptr_ty="*i8")
    _assert_tensor_load_count(amdgcn, 1, "heterogeneous unhinted A/B/AS/BS loads")


# Cache modifiers must match for auto merge.


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
    a_desc = _position_input(a_desc, off_m, off_n)
    b_desc = _position_input(b_desc, off_m, off_n)

    ttgl.amd.gfx1250.tdm.async_load(a_desc, dest=a_buf, warp_used_hint=HINT_A, cache_modifier=CACHE_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, dest=b_buf, warp_used_hint=HINT_B, cache_modifier=CACHE_B)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    c = a_buf.load(layout=BLOCKED_LAYOUT) + b_buf.load(layout=BLOCKED_LAYOUT)
    _store_tile(c_ptr, c, off_m, off_n, M, N, BLOCK_M, BLOCK_N, BLOCKED_LAYOUT)


# Layout: (CACHE_A, CACHE_B, expected_auto_merge, id).
_CACHE_PARAMS = [
    # same cache: auto-merge rule 6 satisfied
    ("", "", True, "same_default"),
    (".cg", ".cg", True, "same_cg"),
    # mismatched cache: auto-merge rule 6 forces a split
    ("", ".cg", False, "default_vs_cg"),
    (".ca", ".cg", False, "ca_vs_cg"),
]


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "CACHE_A,CACHE_B,expected_auto_merge",
    [_param_args(p) for p in _CACHE_PARAMS],
    ids=[_param_id(p) for p in _CACHE_PARAMS],
)
def test_compile_vector_add_tdm_cache(BLOCK_M, BLOCK_N, CACHE_A, CACHE_B, expected_auto_merge):
    """Compile-only: asserts matching cache modifiers for auto merge."""
    amdgcn = _compile_amdgcn(
        vector_add_tdm_kernel_cache, ["a_ptr", "b_ptr", "c_ptr"], {
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "HINT_A": None, "HINT_B": None,
            "CACHE_A": CACHE_A, "CACHE_B": CACHE_B
        })
    context = f"CACHE_A={CACHE_A!r}, CACHE_B={CACHE_B!r}"
    _assert_tensor_load_count(amdgcn, 1 if expected_auto_merge else 2, context)


if __name__ == "__main__":
    # Smoke test: iterate all cookbook entries at one block size.
    # Run as `python test_tdm_merge.py` on a gfx1250 device.
    if os.environ.get("TRITON_AMD_RUN_TDM_RUNTIME_TESTS") != "1":
        raise SystemExit("Set TRITON_AMD_RUN_TDM_RUNTIME_TESTS=1 on a gfx1250 device.")
    print("[2-way: vector_add_tdm_kernel]")
    for p in _HINT_PARAMS:
        ha, hb, ident = p
        print(f"-- {ident}: HINT_A=0b{ha:08b}, HINT_B=0b{hb:08b}")
        test_runtime_vector_add_tdm(64, 64, ha, hb)
        print("   OK")
    print("[4-way: vector_add_tdm_kernel_4way]")
    for p in _HINT_PARAMS_4WAY:
        ha, hb, hc, hd, ident = p
        print(f"-- {ident}: A=0b{ha:08b} B=0b{hb:08b} C=0b{hc:08b} D=0b{hd:08b}")
        test_runtime_vector_add_tdm_4way(64, 64, ha, hb, hc, hd)
        print("   OK")
