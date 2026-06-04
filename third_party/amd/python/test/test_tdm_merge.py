"""Manual + tests for adjacent TDM-copy merging on gfx1250.

Canonical reference for **implicit merging** of adjacent `async_load`s with
compatible `warp_used_hint`s.  Adjacent hinted copies with pairwise-disjoint
hints fuse into one `llvm.amdgcn.tensor.load.to.lds` during TDM->LLVM lowering;
each wave `select`s its own descriptor (no source rewrite).  Unless
`TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS=1`, the compiler also auto-generates
hints for runs of already-adjacent unhinted copies (it only adds attributes;
copies separated by another op, e.g. interleaved `memdesc_index` destinations,
are left alone).  The env knob gates only auto-generation; user-provided
compatible hints always merge.

For `warp_used_hint` legality (K, i0, axis-aligned coset) and a hint cookbook,
see `test_tdm_copy.py`.  Authoritative mergeability rules live in
`TDMUtility.h::TDMMergeGroupInfo`; in brief, the N (2..4) members must be
consecutive, hinted, mbarrier-free, pairwise-disjoint, same-rank and same-cache
(any intervening op, including a workgroup barrier, ends the run).  Only
pairwise disjointness is required -- the union need not be a coset (e.g. K=1
hints {0b0001,0b0010,0b0100,0b1000} fuse as N=4).  Destination MemDescTypes may
differ; metadata is lowered per member.  Kernels use `async_wait(0)` ("wait for
everything"), correct under any merge outcome.

Worked examples (kernel -> what it exercises):
  * vector_add_tdm_kernel          -- 2-way merge + decline cookbook
  * vector_add_tdm_kernel_3way     -- 3-way merge (non-uniform generated shape)
  * vector_add_tdm_kernel_4way     -- 4-way member-predicate path
  * heterogeneous_tdm_merge_kernel -- differing destination MemDescTypes
  * vector_add_tdm_kernel_cache    -- cache_modifier gates merge (rule 7)

Compile-only tests count `tensor_load_to_lds` in the AMDGCN asm (no GPU);
runtime tests compare against a torch-on-CPU reference and are skipped off
gfx1250.
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


# ===========================================================================
# Worked example #1: 2-way merge
# ===========================================================================
# Covers mergeable adjacent pairs and the "decline to merge" path for
# merge-rule violations such as overlapping (non-disjoint) hints.
#
#     ttgl.amd.gfx1250.tdm.async_load(a_desc, [m, n], a_buf,
#                                     warp_used_hint=HINT_A)
#     ttgl.amd.gfx1250.tdm.async_load(b_desc, [m, n], b_buf,
#                                     warp_used_hint=HINT_B)
#     ttgl.amd.gfx1250.tdm.async_wait(0)
#
# Both hints legal + pairwise disjoint -> fused into one
# `tensor_load_to_lds`.
# ===========================================================================


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
    USE_HINT: ttgl.constexpr,
):
    """Two-tile vector add via TDM, parametrised on hints A and B.

    The two hinted loads sit back-to-back so the merge analyser can
    consider them; actual fusion depends on rules 1-7 documented at
    the top of this file.
    """
    num_warps: ttgl.constexpr = ttgl.num_warps()
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)

    # One allocation per copy: distinct buffers let membar see disjoint
    # destinations, so no workgroup barrier splits the adjacent copies.
    a_stage = ttgl.allocate_shared_memory(a_desc.dtype, [1] + a_desc.block_shape, a_desc.layout)
    b_stage = ttgl.allocate_shared_memory(b_desc.dtype, [1] + b_desc.block_shape, b_desc.layout)
    a_buf = a_stage.index(0)
    b_buf = b_stage.index(0)

    if not USE_HINT:
        HINT_A = None
        HINT_B = None
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_wait(0)  # "wait for everything"; correct under any merge

    a = a_buf.load(layout=BLOCKED_LAYOUT)
    b = b_buf.load(layout=BLOCKED_LAYOUT)
    c = a + b

    offs_m = off_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_n = off_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs = (offs_m[:, None] * N) + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ttgl.store(c_ptr + offs, c, mask=mask)


# ---------------------------------------------------------------------------
# Hint cookbook for `vector_add_tdm_kernel`.  Layout:
# (HINT_A, HINT_B, expected_merge, id).  Bit `i` => warp `i`.
# Every pair here is pairwise-disjoint, so all merge (rule 3 only requires
# disjoint hints, not a coset union); other failing-rule cases live in the
# cache cookbook below.  `expected_merge` is kept in the schema for symmetry.
# ---------------------------------------------------------------------------

_HINT_PARAMS = [
    # minimal mergeable pair: K=1 each, union {0,1}
    (0b00000001, 0b00000010, True, "merge_single_warp_pair"),
    # split warps in half: K=4 each, union covers all 8 warps.  Useful
    # for double-buffered staging.
    (0b00001111, 0b11110000, True, "merge_lo_hi"),
    # strided cosets: every-other warp + complement.  Useful for
    # interleaved producer/consumer patterns.
    (0b01010101, 0b10101010, True, "merge_strided"),
    # lo/hi pair cosets: basis {0,2}, two K=4 quartets.
    (0b00110011, 0b11001100, True, "merge_lo_hi_pairs"),
    # partial coverage: K=2 each, union covers 4 of 8 warps (rest idle).
    (0b00000011, 0b00001100, True, "merge_partial_K4_idle"),
    # disjoint K=1 hints whose union {0,3} is not itself a coset.  Rule 3
    # only requires pairwise disjointness, so this still merges: warp 0 picks
    # member A, every other warp falls through to member B (hint 0b1000),
    # which predicates off all but warp 3 -- matching standalone emission.
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


def _count_tensor_load_to_lds(amdgcn: str) -> int:
    return len(_tensor_load_to_lds_lines(amdgcn))


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


# ---------------------------------------------------------------------------
# Compile-only test: counts `tensor_load_to_lds` instructions in the
# AMDGCN asm.  No GPU required.
# ---------------------------------------------------------------------------

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
    amdgcn = _compile_amdgcn(
        vector_add_tdm_kernel, ["a_ptr", "b_ptr", "c_ptr"], {
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N,
            "HINT_A": HINT_A, "HINT_B": HINT_B, "USE_HINT": use_tdm_hint
        })
    context = f"HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}"
    fused = use_tdm_hint and expected_merge
    _assert_tensor_load_count(amdgcn, 1 if fused else 2, context)


def test_compile_vector_add_tdm_auto_merge_env_toggle(monkeypatch):
    """Compile-only: env toggles generated hints for adjacent unhinted copies."""
    env_var = "TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS"

    def compile_unhinted(block_m, block_n):
        return _compile_amdgcn(
            vector_add_tdm_kernel, ["a_ptr", "b_ptr", "c_ptr"], {
                "BLOCK_M": block_m, "BLOCK_N": block_n,
                "HINT_A": 0, "HINT_B": 0, "USE_HINT": False
            })

    for env, block, expected in [("1", (64, 64), 2), ("0", (32, 128), 1), ("1", (128, 64), 2)]:
        monkeypatch.setenv(env_var, env)
        _assert_tensor_load_count(compile_unhinted(*block), expected, f"env={env} generated hints")


# ---------------------------------------------------------------------------
# Runtime correctness test on gfx1250 hardware.
# ---------------------------------------------------------------------------

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
    M, N = 256, 512
    NUM_WARPS = 8
    use_tdm_hint = _use_tdm_hint(request)

    # Keep all torch math on CPU: torch's HIP runtime often lacks
    # kernels for gfx1250 (rand/add/zeros_like hit hipErrorInvalidImage).
    # `.cuda()` is a pure memcpy and `torch.empty` only allocates.
    torch.manual_seed(0)
    a_cpu = torch.rand((M, N), dtype=torch.float16)
    b_cpu = torch.rand((M, N), dtype=torch.float16)
    a = a_cpu.cuda()
    b = b_cpu.cuda()
    c = torch.empty((M, N), dtype=torch.float16, device="cuda")

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    vector_add_tdm_kernel[grid](a, b, c, M, N, BLOCK_M, BLOCK_N, HINT_A, HINT_B,
                                use_tdm_hint, num_warps=NUM_WARPS)

    expected = a_cpu + b_cpu
    torch.testing.assert_close(c.cpu(), expected, atol=1e-3, rtol=1e-3)


# ===========================================================================
# Worked example #2: 3-way merge across three descriptors
# ===========================================================================
# Three back-to-back TDM copies can fuse into one hardware op when their hints
# are pairwise disjoint.  This covers the non-uniform generated-hint shape:
#   num_warps=4: {0b0011, 0b0100, 0b1000}
#   num_warps=8: {0b00001111, 0b00110000, 0b11000000}
# ===========================================================================


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
    USE_HINT: ttgl.constexpr,
):
    """Sum three tiles via three adjacent TDM copies."""
    num_warps: ttgl.constexpr = ttgl.num_warps()
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    c_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)

    # One allocation per copy so membar sees disjoint destinations (no barrier
    # between the adjacent copies to split the current run).
    a_stage = ttgl.allocate_shared_memory(a_desc.dtype, [1] + a_desc.block_shape, a_desc.layout)
    b_stage = ttgl.allocate_shared_memory(b_desc.dtype, [1] + b_desc.block_shape, b_desc.layout)
    c_stage = ttgl.allocate_shared_memory(c_desc.dtype, [1] + c_desc.block_shape, c_desc.layout)
    a_buf = a_stage.index(0)
    b_buf = b_stage.index(0)
    c_buf = c_stage.index(0)

    if not USE_HINT:
        HINT_A = None
        HINT_B = None
        HINT_C = None
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_load(c_desc, [off_m, off_n], c_buf, warp_used_hint=HINT_C)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    a = a_buf.load(layout=BLOCKED_LAYOUT)
    b = b_buf.load(layout=BLOCKED_LAYOUT)
    c = c_buf.load(layout=BLOCKED_LAYOUT)
    out = a + b + c

    offs_m = off_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_n = off_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs = (offs_m[:, None] * N) + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ttgl.store(out_ptr + offs, out, mask=mask)


_HINT_PARAMS_3WAY = [
    (8, 0b00001111, 0b00110000, 0b11000000, True, "three_way_8w_generated_shape"),
    (4, 0b0011, 0b0100, 0b1000, True, "three_way_4w_generated_shape"),
    # Disjoint K=1 hints whose union {0,3,4} is not a coset; rule 3 only
    # requires pairwise disjointness, so they still fuse.
    (8, 0b00000001, 0b00001000, 0b00010000, True, "three_way_noncoset_union"),
]


def _compile_3way(num_warps, use_hint, hints=(0, 0, 0), block=(64, 64)) -> str:
    a, b, c = hints
    return _compile_amdgcn(
        vector_add_tdm_kernel_3way, ["a_ptr", "b_ptr", "c_ptr", "out_ptr"], {
            "BLOCK_M": block[0], "BLOCK_N": block[1],
            "HINT_A": a, "HINT_B": b, "HINT_C": c, "USE_HINT": use_hint
        }, num_warps=num_warps)


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "NUM_WARPS,HINT_A,HINT_B,HINT_C,expected_merge",
    [_param_args(p) for p in _HINT_PARAMS_3WAY],
    ids=[_param_id(p) for p in _HINT_PARAMS_3WAY],
)
def test_compile_vector_add_tdm_3way(BLOCK_M, BLOCK_N, NUM_WARPS, HINT_A, HINT_B, HINT_C, expected_merge, request):
    """Compile-only: 3 adjacent copies fuse when their hints are pairwise disjoint."""
    use_tdm_hint = _use_tdm_hint(request)
    amdgcn = _compile_3way(NUM_WARPS, use_tdm_hint, (HINT_A, HINT_B, HINT_C), (BLOCK_M, BLOCK_N))
    context = f"NUM_WARPS={NUM_WARPS}, HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}, HINT_C=0b{HINT_C:08b}"
    fused = use_tdm_hint and expected_merge
    _assert_tensor_load_count(amdgcn, 1 if fused else 3, context)


def test_compile_vector_add_tdm_3way_auto_merge_env_toggle(monkeypatch):
    """Compile-only: env can generate 3-way hints for adjacent unhinted copies."""
    env_var = "TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS"
    # (env, num_warps, block, expected): generation runs for 4 and 8 warps.
    for env, warps, block, expected in [("1", 8, (64, 64), 3), ("0", 8, (32, 128), 1),
                                        ("0", 4, (128, 64), 1), ("1", 4, (64, 128), 3)]:
        monkeypatch.setenv(env_var, env)
        amdgcn = _compile_3way(warps, use_hint=False, block=block)
        _assert_tensor_load_count(amdgcn, expected, f"3-way env={env} warps={warps}")


# ===========================================================================
# Worked example #3: 4-way merge across four descriptors
# ===========================================================================
# Four back-to-back TDM copies fused into one hardware op, exercising
# the N=4 member-predicate path.  Only pairwise disjointness is required; the
# union need not be a coset (e.g. K=1 hints {0b0001, 0b0010, 0b0100, 0b1000}
# fuse even though no proper sub-union is a coset).
# ===========================================================================


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
    USE_HINT: ttgl.constexpr,
):
    """Sum four tiles via four adjacent TDM copies.

    Distinct buffers but shared layout/block shape (rule 6).  When the
    four hints satisfy rules 1-7, lowering fuses into one intrinsic.
    """
    num_warps: ttgl.constexpr = ttgl.num_warps()
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    c_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    d_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=d_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)

    # One allocation per copy so membar sees disjoint destinations (no barrier
    # between the adjacent copies to split the current run).
    a_stage = ttgl.allocate_shared_memory(a_desc.dtype, [1] + a_desc.block_shape, a_desc.layout)
    b_stage = ttgl.allocate_shared_memory(b_desc.dtype, [1] + b_desc.block_shape, b_desc.layout)
    c_stage = ttgl.allocate_shared_memory(c_desc.dtype, [1] + c_desc.block_shape, c_desc.layout)
    d_stage = ttgl.allocate_shared_memory(d_desc.dtype, [1] + d_desc.block_shape, d_desc.layout)
    a_buf = a_stage.index(0)
    b_buf = b_stage.index(0)
    c_buf = c_stage.index(0)
    d_buf = d_stage.index(0)

    # See `_HINT_PARAMS_4WAY` below for legal hint quadruples.
    if not USE_HINT:
        HINT_A = None
        HINT_B = None
        HINT_C = None
        HINT_D = None
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_load(c_desc, [off_m, off_n], c_buf, warp_used_hint=HINT_C)
    ttgl.amd.gfx1250.tdm.async_load(d_desc, [off_m, off_n], d_buf, warp_used_hint=HINT_D)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    a = a_buf.load(layout=BLOCKED_LAYOUT)
    b = b_buf.load(layout=BLOCKED_LAYOUT)
    c = c_buf.load(layout=BLOCKED_LAYOUT)
    d = d_buf.load(layout=BLOCKED_LAYOUT)
    out = a + b + c + d

    offs_m = off_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_n = off_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs = (offs_m[:, None] * N) + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ttgl.store(out_ptr + offs, out, mask=mask)


# 4-way merge cookbook (all positive / must-fuse cases):
#   - four_K1_lo4_union: K=1 each, active warps {0,1,2,3}.  Shows the
#     analyser tolerating non-coset intermediate union 0b0111.
#   - four_K2_full_union: K=2 each, union covers all 8 warps -- fan-out
#     where every warp loads exactly one of four tiles.
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
    amdgcn = _compile_amdgcn(
        vector_add_tdm_kernel_4way, ["a_ptr", "b_ptr", "c_ptr", "d_ptr", "out_ptr"], {
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "HINT_A": HINT_A, "HINT_B": HINT_B,
            "HINT_C": HINT_C, "HINT_D": HINT_D, "USE_HINT": use_tdm_hint
        })
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
    M, N = 256, 512
    NUM_WARPS = 8
    use_tdm_hint = _use_tdm_hint(request)

    torch.manual_seed(0)
    a_cpu = torch.rand((M, N), dtype=torch.float16)
    b_cpu = torch.rand((M, N), dtype=torch.float16)
    c_cpu = torch.rand((M, N), dtype=torch.float16)
    d_cpu = torch.rand((M, N), dtype=torch.float16)
    a = a_cpu.cuda()
    b = b_cpu.cuda()
    c = c_cpu.cuda()
    d = d_cpu.cuda()
    out = torch.empty((M, N), dtype=torch.float16, device="cuda")

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    vector_add_tdm_kernel_4way[grid](a, b, c, d, out, M, N, BLOCK_M, BLOCK_N,
                                     HINT_A, HINT_B, HINT_C, HINT_D, use_tdm_hint, num_warps=NUM_WARPS)
    expected = a_cpu + b_cpu + c_cpu + d_cpu
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


# ===========================================================================
# Worked example #4: heterogeneous destination MemDescTypes
# ===========================================================================
# Four adjacent TDM copies with different destination shared-memory shapes and
# layouts still fuse when the remaining merge rules hold.  This mirrors the
# MXFP A/B/AS/BS load group and guards against accidentally deriving all
# descriptor-fill metadata from the first member.
# ===========================================================================


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
    USE_HINT: ttgl.constexpr,
):
    A_SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])
    B_SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    SCALE_SHARED_LAYOUT: ttgl.constexpr = ttgl.SwizzledSharedLayout(1, 1, 1, [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=A_SHARED_LAYOUT)
    b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N_B), layout=B_SHARED_LAYOUT)
    as_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=as_ptr, shape=(M, N), strides=(N, 1),
                                                          block_shape=(BLOCK_SCALE_M, BLOCK_SCALE_N), layout=SCALE_SHARED_LAYOUT)
    bs_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=bs_ptr, shape=(M, N), strides=(N, 1),
                                                          block_shape=(BLOCK_SCALE_M, BLOCK_SCALE_N), layout=SCALE_SHARED_LAYOUT)

    a_stage = ttgl.allocate_shared_memory(a_desc.dtype, [1] + a_desc.block_shape, a_desc.layout)
    b_stage = ttgl.allocate_shared_memory(b_desc.dtype, [1] + b_desc.block_shape, b_desc.layout)
    as_stage = ttgl.allocate_shared_memory(as_desc.dtype, [1] + as_desc.block_shape, as_desc.layout)
    bs_stage = ttgl.allocate_shared_memory(bs_desc.dtype, [1] + bs_desc.block_shape, bs_desc.layout)
    a_buf = a_stage.index(0)
    b_buf = b_stage.index(0)
    as_buf = as_stage.index(0)
    bs_buf = bs_stage.index(0)

    if not USE_HINT:
        HINT_A = None
        HINT_B = None
        HINT_AS = None
        HINT_BS = None
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B)
    ttgl.amd.gfx1250.tdm.async_load(as_desc, [off_m, off_n], as_buf, warp_used_hint=HINT_AS)
    ttgl.amd.gfx1250.tdm.async_load(bs_desc, [off_m, off_n], bs_buf, warp_used_hint=HINT_BS)
    ttgl.amd.gfx1250.tdm.async_wait(0)


def test_compile_heterogeneous_tdm_merge(request):
    """Compile-only: A/B/AS/BS destination MemDescTypes still fuse."""
    use_tdm_hint = _use_tdm_hint(request)
    amdgcn = _compile_amdgcn(
        heterogeneous_tdm_merge_kernel, ["a_ptr", "b_ptr", "as_ptr", "bs_ptr"], {
            "BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_N_B": 2048,
            "BLOCK_SCALE_M": 64, "BLOCK_SCALE_N": 32,
            "HINT_A": 0b00010001, "HINT_B": 0b00100010,
            "HINT_AS": 0b01000100, "HINT_BS": 0b10001000, "USE_HINT": use_tdm_hint
        }, ptr_ty="*i8")
    _assert_tensor_load_count(amdgcn, 1 if use_tdm_hint else 4,
                              "heterogeneous A/B/AS/BS destination MemDescTypes")


# ===========================================================================
# Worked example #5: cache_modifier interaction with merging
# ===========================================================================
# Rule 7: members must share the same `cache_modifier` (the fused
# intrinsic has one auxBits field).  Hints are pinned to the canonical
# "split warps in half" pair so only the cache modifier decides fusion.
# ===========================================================================


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
    USE_HINT: ttgl.constexpr,
):
    """Two-tile vector add with explicit cache modifiers per copy.

    `CACHE_A` / `CACHE_B` use the same strings as `tt.load`'s
    `cache_modifier` (`""`, `.ca`, `.cg`, ...; see
    `_str_to_load_cache_modifier`).
    """
    num_warps: ttgl.constexpr = ttgl.num_warps()
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])

    pid_m = ttgl.program_id(axis=0)
    pid_n = ttgl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=b_ptr, shape=(M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)

    # One allocation per copy so membar sees disjoint destinations (no barrier
    # between the adjacent copies to split the current run).
    a_stage = ttgl.allocate_shared_memory(a_desc.dtype, [1] + a_desc.block_shape, a_desc.layout)
    b_stage = ttgl.allocate_shared_memory(b_desc.dtype, [1] + b_desc.block_shape, b_desc.layout)
    a_buf = a_stage.index(0)
    b_buf = b_stage.index(0)

    if not USE_HINT:
        HINT_A = None
        HINT_B = None
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A, cache_modifier=CACHE_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B, cache_modifier=CACHE_B)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    a = a_buf.load(layout=BLOCKED_LAYOUT)
    b = b_buf.load(layout=BLOCKED_LAYOUT)
    c = a + b

    offs_m = off_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_n = off_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs = (offs_m[:, None] * N) + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ttgl.store(c_ptr + offs, c, mask=mask)


# Cache-modifier cookbook.  Layout: (CACHE_A, CACHE_B, expected_merge,
# id).  Hints pinned; only the cache string varies.
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
    amdgcn = _compile_amdgcn(
        vector_add_tdm_kernel_cache, ["a_ptr", "b_ptr", "c_ptr"], {
            "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "HINT_A": 0b00001111, "HINT_B": 0b11110000,
            "CACHE_A": CACHE_A, "CACHE_B": CACHE_B, "USE_HINT": use_tdm_hint
        })
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
