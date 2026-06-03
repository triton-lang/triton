"""User-facing manual + test suite for adjacent TDM-copy merging on gfx1250.

This is the canonical reference for **implicit merging** of adjacent
`async_load`s with compatible `warp_used_hint` values on AMD gfx1250.
The environment knob only controls whether the compiler auto-generates hints
for adjacent unhinted copies; user-provided compatible hints remain mergeable.
General predicated/partial-copy coverage lives with the TDM copy tests;
this file keeps only merge-specific coverage.

Worked examples by shape:

  * Two descriptors merged          -> `vector_add_tdm_kernel`,
                                       merge-eligible rows
  * Two descriptors not merged      -> `vector_add_tdm_kernel`,
                                       merge-rule violation rows
  * Three descriptors fused         -> `vector_add_tdm_kernel_3way`
  * Four descriptors fused          -> `vector_add_tdm_kernel_4way`
  * Heterogeneous destinations      -> `heterogeneous_tdm_merge_kernel`
  * Cache-modifier vs merge         -> `vector_add_tdm_kernel_cache`

================================================================
Manual: terminology
================================================================

For `warp_used_hint` legality (K, i0, axis-aligned coset) and a cookbook of
legal/illegal hints and the generic hint formulas, see the manual in
`test_tdm_copy.py`; this file does not repeat it.  One merge-specific term:

  * **current run**: consecutive `async_load`s the merge analyser considers for
    fusion.  From the run head it picks the largest supported N (2, 3, or 4)
    with pairwise-disjoint hints.

================================================================
Manual: implicit op-merging across adjacent copies
================================================================

Two or more adjacent `async_load`s with compatible hints fuse into one
`llvm.amdgcn.tensor.load.to.lds` during TDM->LLVM lowering.  Each wave
selects its member via SGPR-uniform selection; no source-level rewrite is
needed.  Unless `TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS=1`, the compiler can
also create compatible hints for adjacent unhinted copies.

There are two separate capabilities:

  1. **Fundamental lowering merge**: if the source already provides compatible
     `warp_used_hint`s, TDM-to-LLVM can merge the copies even when they write to
     independent shared allocations, provided the allocation ops are outside the
     consecutive copy run.
  2. **Automatic hint generation**: unless the disable env var is set, the
     compiler can synthesize hints only for the canonical indexed-destination
     shape with non-partitioned destinations:

         memdesc_index A; async_tdm_copy A;
         memdesc_index B; async_tdm_copy B; ...

     The auto pass does not synthesize hints for arbitrary independent
     `allocate_shared_memory` destinations or partitioned destinations; provide
     hints explicitly for those.

Mergeability rules (authoritative list in
`TDMUtility.h::TDMMergeGroupInfo`):

  1. Every member has a verifier-legal `warp_used_hint`. Unhinted copies end
     the current run unless auto hint generation is enabled and the copies match
     the generated-hint pattern.
  2. No member has an `mbarrier`.  Such ops lower as singletons and
     end the current run.
  3. Members have pairwise-disjoint hints. The union does not need to be a
     verifier-legal `warp_used_hint`.  Members may have different K.
  4. Group size N is 2, 3, or 4.
  5. Members are consecutive in the same block; any intervening op
     (TDM or not) ends the current run.
  6. Members have same-rank descriptors that can be represented by a compatible
     hardware descriptor group form for the fused intrinsic. Destination
     `MemDescType`s may differ; shape/layout/type metadata is lowered per
     member.
  7. Members share the same `cache_modifier`.

The analyser picks, from the head of the current run, the largest
supported N (up to 4) whose hints stay pairwise disjoint.  Op order, not
warp order: it picks the first N `async_load`s, not the first N warps.
The union need not be a coset, so e.g. K=1 hints {0b0001, 0b0010,
0b0100, 0b1000} fuse as one N=4 group.

Example: two adjacent loads that *will* fuse:

    # Op A's hint activates warps {0,1,2,3} (the lower half).
    # Op B's hint activates warps {4,5,6,7} (the upper half).
    # K=4 each, pairwise disjoint -> fuses into one `tensor_load_to_lds` op.
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [m, n], a_buf,
                                    warp_used_hint=0b00001111)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [m, n], b_buf,
                                    warp_used_hint=0b11110000)
    ttgl.amd.gfx1250.tdm.async_wait(0)  # one outstanding fused TDM op

Example: two adjacent loads that *will not* fuse:

    ttgl.amd.gfx1250.tdm.async_load(a_desc, [m, n], a_buf,
                                    warp_used_hint=0b00000011)   # warps {0,1}
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [m, n], b_buf,
                                    warp_used_hint=0b00000110)   # warps {1,2}
    # Rule 3 violation: the hints overlap on warp 1, so they are not
    # pairwise disjoint.  Lowered as two separate intrinsics.

================================================================
Manual: `async_wait` is user-owned
================================================================

The lowering doesn't adjust wait counts; size them on the post-merge
intrinsic count.  `async_wait(0)` ("wait for everything") is correct
under any merge outcome and is what every kernel here uses.

================================================================
Manual: cache modifier interaction with merge
================================================================

`async_load(..., cache_modifier=".cg")` lowers to the single auxBits
immediate on the fused intrinsic, so mismatched cache modifiers block
fusion (rule 7).  See `vector_add_tdm_kernel_cache` for a worked
example covering same-cache and mismatched-cache sides.

================================================================
What this file actually tests
================================================================

  * Compile-only tests count `tensor_load_to_lds` instructions in
    the AMDGCN asm to assert the merge analyser's decisions.
  * Runtime tests on gfx1250 compare against a torch-on-CPU reference.

Runtime tests are skipped on non-gfx1250 hosts.
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

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr,
        shape=(M, N),
        strides=(N, 1),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=SHARED_LAYOUT,
    )
    b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr,
        shape=(M, N),
        strides=(N, 1),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=SHARED_LAYOUT,
    )

    # One allocation per copy: distinct buffers let the membar analysis see the
    # destinations as disjoint, so no workgroup barrier is inserted between the
    # adjacent copies (a barrier would split the current run).  Fusion then
    # depends only on the hint merge rules below.
    a_stage = ttgl.allocate_shared_memory(a_desc.dtype, [1] + a_desc.block_shape, a_desc.layout)
    b_stage = ttgl.allocate_shared_memory(b_desc.dtype, [1] + b_desc.block_shape, b_desc.layout)
    a_buf = a_stage.index(0)
    b_buf = b_stage.index(0)

    if not USE_HINT:
        HINT_A = None
        HINT_B = None
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B)

    # `async_wait(0)` = "wait for everything", correct under any merge
    # outcome.
    ttgl.amd.gfx1250.tdm.async_wait(0)

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


def _compile_vector_add_tdm_amdgcn(BLOCK_M: int, BLOCK_N: int, USE_HINT: bool) -> str:
    NUM_WARPS = 8
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "HINT_A": "constexpr",
        "HINT_B": "constexpr",
        "USE_HINT": "constexpr",
    }
    constexprs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "HINT_A": 0,
        "HINT_B": 0,
        "USE_HINT": USE_HINT,
    }
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=vector_add_tdm_kernel,
            signature=signature,
            constexprs=constexprs,
        ),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": NUM_WARPS},
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
    NUM_WARPS = 8
    use_tdm_hint = _use_tdm_hint(request)
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "HINT_A": "constexpr",
        "HINT_B": "constexpr",
        "USE_HINT": "constexpr",
    }
    constexprs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "HINT_A": HINT_A,
        "HINT_B": HINT_B,
        "USE_HINT": use_tdm_hint,
    }
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=vector_add_tdm_kernel,
            signature=signature,
            constexprs=constexprs,
        ),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": NUM_WARPS},
    )

    amdgcn = k.asm["amdgcn"]
    context = f"HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}"
    if not use_tdm_hint:
        _assert_tensor_load_count(amdgcn, 2, f"unhinted {context}")
    elif expected_merge:
        _assert_tensor_load_count(amdgcn, 1, f"fused {context}")
    else:
        _assert_tensor_load_count(amdgcn, 2, context)


def test_compile_vector_add_tdm_auto_merge_env_toggle(monkeypatch):
    """Compile-only: env toggles generated hints for adjacent unhinted copies."""
    env_var = "TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS"

    monkeypatch.setenv(env_var, "1")
    amdgcn = _compile_vector_add_tdm_amdgcn(64, 64, USE_HINT=False)
    _assert_tensor_load_count(amdgcn, 2, "env-disabled generated hints")

    monkeypatch.setenv(env_var, "0")
    amdgcn = _compile_vector_add_tdm_amdgcn(32, 128, USE_HINT=False)
    _assert_tensor_load_count(amdgcn, 1, "env-enabled generated hints")

    monkeypatch.setenv(env_var, "1")
    amdgcn = _compile_vector_add_tdm_amdgcn(128, 64, USE_HINT=False)
    _assert_tensor_load_count(amdgcn, 2, "env-reset generated hints")


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
    vector_add_tdm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        HINT_A,
        HINT_B,
        use_tdm_hint,
        num_warps=NUM_WARPS,
    )

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


def _compile_vector_add_tdm_3way_amdgcn(
    num_warps: int,
    use_hint: bool,
    hint_a: int = 0,
    hint_b: int = 0,
    hint_c: int = 0,
    block_m: int = 64,
    block_n: int = 64,
) -> str:
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "out_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "HINT_A": "constexpr",
        "HINT_B": "constexpr",
        "HINT_C": "constexpr",
        "USE_HINT": "constexpr",
    }
    constexprs = {
        "BLOCK_M": block_m,
        "BLOCK_N": block_n,
        "HINT_A": hint_a,
        "HINT_B": hint_b,
        "HINT_C": hint_c,
        "USE_HINT": use_hint,
    }
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=vector_add_tdm_kernel_3way,
            signature=signature,
            constexprs=constexprs,
        ),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": num_warps},
    )
    return k.asm["amdgcn"]


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "NUM_WARPS,HINT_A,HINT_B,HINT_C,expected_merge",
    [_param_args(p) for p in _HINT_PARAMS_3WAY],
    ids=[_param_id(p) for p in _HINT_PARAMS_3WAY],
)
def test_compile_vector_add_tdm_3way(BLOCK_M, BLOCK_N, NUM_WARPS, HINT_A, HINT_B, HINT_C, expected_merge, request):
    """Compile-only: 3 adjacent copies fuse when their hints are pairwise disjoint."""
    use_tdm_hint = _use_tdm_hint(request)
    amdgcn = _compile_vector_add_tdm_3way_amdgcn(
        NUM_WARPS, use_tdm_hint, HINT_A, HINT_B, HINT_C, BLOCK_M, BLOCK_N)
    context = (
        f"NUM_WARPS={NUM_WARPS}, HINT_A=0b{HINT_A:08b}, "
        f"HINT_B=0b{HINT_B:08b}, HINT_C=0b{HINT_C:08b}"
    )
    if not use_tdm_hint:
        _assert_tensor_load_count(amdgcn, 3, f"unhinted {context}")
    elif expected_merge:
        _assert_tensor_load_count(amdgcn, 1, f"fused {context}")
    else:
        _assert_tensor_load_count(amdgcn, 3, context)


def test_compile_vector_add_tdm_3way_auto_merge_env_toggle(monkeypatch):
    """Compile-only: env can generate 3-way hints for adjacent unhinted copies."""
    env_var = "TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS"

    monkeypatch.setenv(env_var, "1")
    amdgcn = _compile_vector_add_tdm_3way_amdgcn(8, use_hint=False, block_m=64, block_n=64)
    _assert_tensor_load_count(amdgcn, 3, "3-way env-disabled generated hints")

    monkeypatch.setenv(env_var, "0")
    amdgcn = _compile_vector_add_tdm_3way_amdgcn(8, use_hint=False, block_m=32, block_n=128)
    _assert_tensor_load_count(amdgcn, 1, "3-way env-enabled generated hints for 8 warps")

    monkeypatch.setenv(env_var, "0")
    amdgcn = _compile_vector_add_tdm_3way_amdgcn(4, use_hint=False, block_m=128, block_n=64)
    _assert_tensor_load_count(amdgcn, 1, "3-way env-enabled generated hints for 4 warps")

    monkeypatch.setenv(env_var, "1")
    amdgcn = _compile_vector_add_tdm_3way_amdgcn(4, use_hint=False, block_m=64, block_n=128)
    _assert_tensor_load_count(amdgcn, 3, "3-way env-reset generated hints")


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
    NUM_WARPS = 8
    use_tdm_hint = _use_tdm_hint(request)
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "d_ptr": "*fp16",
        "out_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "HINT_A": "constexpr",
        "HINT_B": "constexpr",
        "HINT_C": "constexpr",
        "HINT_D": "constexpr",
        "USE_HINT": "constexpr",
    }
    constexprs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "HINT_A": HINT_A,
        "HINT_B": HINT_B,
        "HINT_C": HINT_C,
        "HINT_D": HINT_D,
        "USE_HINT": use_tdm_hint,
    }
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=vector_add_tdm_kernel_4way,
            signature=signature,
            constexprs=constexprs,
        ),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": NUM_WARPS},
    )
    expected_tdm = 1 if use_tdm_hint else 4
    amdgcn = k.asm["amdgcn"]
    _assert_tensor_load_count(
        amdgcn,
        expected_tdm,
        f"HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}, "
        f"HINT_C=0b{HINT_C:08b}, HINT_D=0b{HINT_D:08b}",
    )


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
    vector_add_tdm_kernel_4way[grid](
        a,
        b,
        c,
        d,
        out,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        HINT_A,
        HINT_B,
        HINT_C,
        HINT_D,
        use_tdm_hint,
        num_warps=NUM_WARPS,
    )
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

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr,
        shape=(M, N),
        strides=(N, 1),
        block_shape=(BLOCK_M, BLOCK_N),
        layout=A_SHARED_LAYOUT,
    )
    b_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr,
        shape=(M, N),
        strides=(N, 1),
        block_shape=(BLOCK_M, BLOCK_N_B),
        layout=B_SHARED_LAYOUT,
    )
    as_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=as_ptr,
        shape=(M, N),
        strides=(N, 1),
        block_shape=(BLOCK_SCALE_M, BLOCK_SCALE_N),
        layout=SCALE_SHARED_LAYOUT,
    )
    bs_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=bs_ptr,
        shape=(M, N),
        strides=(N, 1),
        block_shape=(BLOCK_SCALE_M, BLOCK_SCALE_N),
        layout=SCALE_SHARED_LAYOUT,
    )

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
    NUM_WARPS = 8
    use_tdm_hint = _use_tdm_hint(request)
    HINT_A = 0b00010001
    HINT_B = 0b00100010
    HINT_AS = 0b01000100
    HINT_BS = 0b10001000
    signature = {
        "a_ptr": "*i8",
        "b_ptr": "*i8",
        "as_ptr": "*i8",
        "bs_ptr": "*i8",
        "M": "i32",
        "N": "i32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "BLOCK_N_B": "constexpr",
        "BLOCK_SCALE_M": "constexpr",
        "BLOCK_SCALE_N": "constexpr",
        "HINT_A": "constexpr",
        "HINT_B": "constexpr",
        "HINT_AS": "constexpr",
        "HINT_BS": "constexpr",
        "USE_HINT": "constexpr",
    }
    constexprs = {
        "BLOCK_M": 256,
        "BLOCK_N": 128,
        "BLOCK_N_B": 2048,
        "BLOCK_SCALE_M": 64,
        "BLOCK_SCALE_N": 32,
        "HINT_A": HINT_A,
        "HINT_B": HINT_B,
        "HINT_AS": HINT_AS,
        "HINT_BS": HINT_BS,
        "USE_HINT": use_tdm_hint,
    }
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=heterogeneous_tdm_merge_kernel,
            signature=signature,
            constexprs=constexprs,
        ),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": NUM_WARPS},
    )
    expected_tdm = 1 if use_tdm_hint else 4
    amdgcn = k.asm["amdgcn"]
    _assert_tensor_load_count(amdgcn, expected_tdm, "heterogeneous A/B/AS/BS destination MemDescTypes")


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
    NUM_WARPS = 8
    use_tdm_hint = _use_tdm_hint(request)
    HINT_A = 0b00001111  # warps {0,1,2,3}
    HINT_B = 0b11110000  # warps {4,5,6,7}; union covers all 8 warps
    signature = {
        "a_ptr": "*fp16",
        "b_ptr": "*fp16",
        "c_ptr": "*fp16",
        "M": "i32",
        "N": "i32",
        "BLOCK_M": "constexpr",
        "BLOCK_N": "constexpr",
        "HINT_A": "constexpr",
        "HINT_B": "constexpr",
        "CACHE_A": "constexpr",
        "CACHE_B": "constexpr",
        "USE_HINT": "constexpr",
    }
    constexprs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "HINT_A": HINT_A,
        "HINT_B": HINT_B,
        "CACHE_A": CACHE_A,
        "CACHE_B": CACHE_B,
        "USE_HINT": use_tdm_hint,
    }
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=vector_add_tdm_kernel_cache,
            signature=signature,
            constexprs=constexprs,
        ),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": NUM_WARPS},
    )
    amdgcn = k.asm["amdgcn"]
    context = f"CACHE_A={CACHE_A!r}, CACHE_B={CACHE_B!r}"
    if not use_tdm_hint:
        _assert_tensor_load_count(amdgcn, 2, f"unhinted {context}")
    elif expected_merge:
        _assert_tensor_load_count(amdgcn, 1, f"fused {context}")
    else:
        _assert_tensor_load_count(amdgcn, 2, context)


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
