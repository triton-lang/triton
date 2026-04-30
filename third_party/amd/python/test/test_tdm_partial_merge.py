"""User-facing manual + test suite for partial TDM copies and adjacent-copy
merging on gfx1250.

This is the canonical reference for two Gluon-level features on AMD
gfx1250:

  1. **Partial TDM copy** -- `async_load(..., warp_used_hint=H)` selects
     which warps participate in a single TDM transfer.
  2. **Implicit merging** -- adjacent `async_load`s with compatible
     hints fuse into one hardware intrinsic during lowering, with no
     IR/op annotation; user code is unchanged.

Worked examples by shape:

  * Partial warps, one descriptor   -> `vector_add_tdm_kernel`, rows
                                       where one HINT is 0
  * Two descriptors merged          -> `vector_add_tdm_kernel`,
                                       merge-eligible rows
  * Four descriptors fused          -> `vector_add_tdm_kernel_4way`
  * Cache-modifier vs merge         -> `vector_add_tdm_kernel_cache`

================================================================
Manual: terminology
================================================================

  * **K** = `popcount(hint)` -- number of active warps.  Power of two,
    `1 <= K <= num_warps`.
  * **i0** = smallest active warp INDEX (`countr_zero(hint)`).
  * **basis bits** = set bits of `support = OR over s in S of (s XOR i0)`.
    They are bit positions; the basis vectors are `1 << basis_bit`.
  * **axis-aligned coset** = active set written as `i0 XOR
    span(basis_bits)`.  Equivalent to "passes the verifier".
  * **candidate batch** (merge-only term): consecutive `async_load`s
    the merge analyser considers for fusion.  From the batch head it
    picks the largest power-of-two N with a legal-coset union.

================================================================
Manual: when to reach for `warp_used_hint`
================================================================

Default `async_load(desc, idx, buf)` slices the tile across all
`num_warps` warps.  Reach for `warp_used_hint=H` to:

  * Free warps for unrelated work when the tile is small.
  * Stage independent back-to-back tiles that the lowering can fuse
    (merging requires every member to carry a hint).
  * Partition warps by role (producer/consumer pipelines).

`H` is an `i32` bitmask: bit `i` set => warp `i` participates.  Omit
or pass `None` for "all warps"; an explicit `warp_used_hint=0` is
rejected by the verifier.

================================================================
Manual: constructing a legal hint
================================================================

The verifier (`validateWarpUsedHint` in `Dialect.cpp`) requires the
active set `S = { i : bit i of H is 1 }` to be an *axis-aligned coset*:

  - `K = popcount(H)` is a power of two with `1 <= K <= num_warps`.
  - `S` shifted by its anchor (smallest active index `i0`) is an
    F_2-linear subspace whose basis vectors are single powers of two.

Pass the hint as a binary-literal `int`.  Cookbook of legal 8-warp
hints (bit 7 on the left, bit 0 on the right):

  +-------------+---+----+--------------------------------------------+
  | Bitmask     | K | i0 | Active warps / shape                       |
  +-------------+---+----+--------------------------------------------+
  | 0b00001111  | 4 |  0 | warps {0,1,2,3}    -- low 4 warps          |
  | 0b11110000  | 4 |  4 | warps {4,5,6,7}    -- high 4 warps         |
  | 0b01010101  | 4 |  0 | warps {0,2,4,6}    -- every other, low i0  |
  | 0b10101010  | 4 |  1 | warps {1,3,5,7}    -- every other, high i0 |
  | 0b00110011  | 4 |  0 | warps {0,1,4,5}    -- two K=2 lanes paired |
  | 0b00000011  | 2 |  0 | warps {0,1}        -- low 2 warps          |
  | 0b00010000  | 1 |  4 | warps {4}          -- single warp          |
  +-------------+---+----+--------------------------------------------+

Common rejected patterns (DO NOT pass these):

  +-------------+----------------------------------------------------+
  | 0           | rejected: must select at least one warp; pass None |
  | 0b00000111  | rejected: K=3 is not a power of two                |
  | 0b00010001  | rejected: warps 0 and 4 cannot form a coset alone  |
  +-------------+----------------------------------------------------+

If you need to write a hint generically (e.g. parameterised on the
number of active warps `K`), the formulas are:

  * "first K warps":              `(1 << K) - 1`
  * "K warps starting at off":    `((1 << K) - 1) << off`
                                  (off must be a multiple of K)
  * "K warps spaced by 2**s":     `sum(1 << (i << s) for i in range(K))`

================================================================
Manual: implicit op-merging across adjacent copies
================================================================

Two or more adjacent `async_load`s with compatible hints fuse into
one `llvm.amdgcn.tensor.load.to.lds` during TDM->LLVM lowering.  Each
wave selects its member via SGPR-uniform `s_cselect_b32` chains; no
source-level rewrite is needed.

Mergeability rules (authoritative list in
`TDMUtility.h::TDMMergeGroupInfo`):

  1. Every member has a verifier-legal `warp_used_hint`.  Hint-less
     ops are not candidates and flush the in-flight batch.
  2. No member has an `mbarrier`.  Such ops lower as singletons and
     flush the batch.
  3. Members share K = popcount(hint), are pairwise disjoint, and
     their union is itself a verifier-legal coset.
  4. Group size N is a power of two >= 2.
  5. Members are consecutive in the same block; arith/index math
     threads through, side-effecting non-TDM ops (`async_wait`,
     `barrier`, ...) flush.
  6. Members write to the same shared encoding + shapePerCTA; SSA
     destinations are pairwise distinct.
  7. Members share the same `cache_modifier`.

The analyser picks, from the head of the candidate batch, the largest
power-of-two N whose first-N union is a legal coset.  Op order, not
warp order: it picks the first N `async_load`s, not the first N warps.
Intermediate unions need not be cosets; e.g. K=1 hints
{0b0001, 0b0010, 0b0100, 0b1000} have non-coset 0b0111 mid-way but
legal 0b1111 at the end, and still fuse as one N=4 group.

Example: two adjacent loads that *will* fuse:

    # Op A's hint activates warps {0,1,2,3} (the lower half).
    # Op B's hint activates warps {4,5,6,7} (the upper half).
    # K=4 each, disjoint, union = 0b11111111 covers all 8 warps and
    # is a legal coset -> fuses into one `tensor_load_to_lds` op.
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [m, n], a_buf,
                                    warp_used_hint=0b00001111)
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [m, n], b_buf,
                                    warp_used_hint=0b11110000)
    ttgl.amd.gfx1250.tdm.async_wait(0)  # one outstanding fused TDM op

Example: two adjacent loads that *will not* fuse:

    ttgl.amd.gfx1250.tdm.async_load(a_desc, [m, n], a_buf,
                                    warp_used_hint=0b00000011)   # K=2
    ttgl.amd.gfx1250.tdm.async_load(b_desc, [m, n], b_buf,
                                    warp_used_hint=0b00010000)   # K=1
    # Rule 3 violation (different K). Lowered as two separate intrinsics.

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

# ===========================================================================
# Worked example #1: 2-way partial copy (with optional merge)
# ===========================================================================
# Covers `warp_used_hint` on a single load, mergeable adjacent pairs,
# and the "decline to merge" path (mixed hint+no-hint, or non-coset
# union).
#
#     ttgl.amd.gfx1250.tdm.async_load(a_desc, [m, n], a_buf,
#                                     warp_used_hint=HINT_A)
#     ttgl.amd.gfx1250.tdm.async_load(b_desc, [m, n], b_buf,
#                                     warp_used_hint=HINT_B)
#     ttgl.amd.gfx1250.tdm.async_wait(0)
#
# Both hints legal + disjoint + equal-K + legal-union -> fused into one
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
):
    """Two-tile vector add via TDM, parametrised on hints A and B.

    `HINT == 0` is mapped to the kwarg-free `async_load` (the verifier
    rejects an explicit `warp_used_hint = 0`).  The two loads sit
    back-to-back so the merge analyser can consider them; actual
    fusion depends on rules 1-7 documented at the top of this file.
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

    a_buf = ttgl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    b_buf = ttgl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)

    # Two adjacent TDM copies into distinct shared buffers with the same
    # encoding (rule 6).  Fuses iff both carry merge-legal hints.
    if HINT_A == 0:
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf)
    else:
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    if HINT_B == 0:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf)
    else:
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
# (HINT_A, HINT_B, expected_merge, id).  Bit `i` => warp `i`.  `0` =
# "no hint" (kernel maps to kwarg-free load; an explicit zero is
# rejected by the verifier).  `expected_merge=False` rows fail rule 1
# (unhinted) or rule 3 (illegal union).
# ---------------------------------------------------------------------------

_HINT_PARAMS = [
    # baseline: no hints (singleton fast-path)
    (0b00000000, 0b00000000, False, "no_hint"),
    # mixed hint + no-hint: rule 1 forbids merging
    (0b11111111, 0b00000000, False, "full_a_unhinted_b"),
    (0b00001111, 0b00000000, False, "lo4_a_unhinted_b"),
    (0b11110000, 0b00000000, False, "hi4_a_unhinted_b"),
    (0b01010101, 0b00000000, False, "strided_a_unhinted_b"),
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
    # mismatched K (popcount(union)=3): rule 3 forbids merging.
    (0b00000011, 0b00010000, False, "disjoint_but_union_illegal"),
]


def _param_args(p):
    """Strip trailing id from a parametrised entry."""
    return p[:-1]


def _param_id(p):
    return p[-1]


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
def test_compile_vector_add_tdm(BLOCK_M, BLOCK_N, HINT_A, HINT_B, expected_merge):
    """Compile-only: asserts 1 fused vs 2 separate `tensor_load_to_lds`."""
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
    }
    constexprs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "HINT_A": HINT_A,
        "HINT_B": HINT_B,
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
    n_tdm = len(re.findall(r"tensor_load_to_lds", amdgcn))
    if expected_merge:
        assert n_tdm == 1, (f"expected fused single tensor_load_to_lds for "
                            f"HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}, got {n_tdm}\n{amdgcn}")
    else:
        assert n_tdm == 2, (f"expected two tensor_load_to_lds for HINT_A=0b{HINT_A:08b}, "
                            f"HINT_B=0b{HINT_B:08b}, got {n_tdm}\n{amdgcn}")


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
def test_runtime_vector_add_tdm(BLOCK_M, BLOCK_N, HINT_A, HINT_B, expected_merge):
    """Runtime: c = a + b vs torch CPU reference; merging is perf-only."""
    M, N = 256, 512
    NUM_WARPS = 8

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
        num_warps=NUM_WARPS,
    )

    expected = a_cpu + b_cpu
    torch.testing.assert_close(c.cpu(), expected, atol=1e-3, rtol=1e-3)


# ===========================================================================
# Worked example #2: 4-way merge across four descriptors
# ===========================================================================
# Four back-to-back TDM copies fused into one hardware op, exercising
# the N=4 selector path (2-bit per-wave selector, 3 chained
# `s_cselect_b32`s).  The full union of all four hints must be a legal
# coset; intermediate unions need not be (e.g. K=1 hints {0b0001,
# 0b0010, 0b0100, 0b1000} have non-coset 0b0111 mid-way but legal
# 0b1111 at the end -- still fuses).
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

    a_buf = ttgl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    b_buf = ttgl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)
    c_buf = ttgl.allocate_shared_memory(c_desc.dtype, c_desc.block_shape, c_desc.layout)
    d_buf = ttgl.allocate_shared_memory(d_desc.dtype, d_desc.block_shape, d_desc.layout)

    # See `_HINT_PARAMS_4WAY` below for legal hint quadruples.
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
def test_compile_vector_add_tdm_4way(BLOCK_M, BLOCK_N, HINT_A, HINT_B, HINT_C, HINT_D):
    """Compile-only: every quadruple must fuse to a single intrinsic."""
    NUM_WARPS = 8
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
    }
    constexprs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "HINT_A": HINT_A,
        "HINT_B": HINT_B,
        "HINT_C": HINT_C,
        "HINT_D": HINT_D,
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
    amdgcn = k.asm["amdgcn"]
    n_tdm = len(re.findall(r"tensor_load_to_lds", amdgcn))
    assert n_tdm == 1, (f"expected fused single tensor_load_to_lds for "
                        f"HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}, "
                        f"HINT_C=0b{HINT_C:08b}, HINT_D=0b{HINT_D:08b}, got {n_tdm}\n{amdgcn}")


@pytest.mark.skipif(not is_hip_gfx1250(), reason="TDM is only tested on gfx1250.")
@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _RUNTIME_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "HINT_A,HINT_B,HINT_C,HINT_D",
    [_param_args(p) for p in _HINT_PARAMS_4WAY],
    ids=[_param_id(p) for p in _HINT_PARAMS_4WAY],
)
def test_runtime_vector_add_tdm_4way(BLOCK_M, BLOCK_N, HINT_A, HINT_B, HINT_C, HINT_D):
    """Runtime: out = a+b+c+d; checks the 4-way fused intrinsic routes
    each member's bytes to its own buffer."""
    M, N = 256, 512
    NUM_WARPS = 8

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
        num_warps=NUM_WARPS,
    )
    expected = a_cpu + b_cpu + c_cpu + d_cpu
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


# ===========================================================================
# Worked example #3: cache_modifier interaction with merging
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

    a_buf = ttgl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)
    b_buf = ttgl.allocate_shared_memory(b_desc.dtype, b_desc.block_shape, b_desc.layout)

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
def test_compile_vector_add_tdm_cache(BLOCK_M, BLOCK_N, CACHE_A, CACHE_B, expected_merge):
    """Compile-only: asserts rule 7 (matching cache modifiers)."""
    NUM_WARPS = 8
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
    }
    constexprs = {
        "BLOCK_M": BLOCK_M,
        "BLOCK_N": BLOCK_N,
        "HINT_A": HINT_A,
        "HINT_B": HINT_B,
        "CACHE_A": CACHE_A,
        "CACHE_B": CACHE_B,
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
    n_tdm = len(re.findall(r"tensor_load_to_lds", amdgcn))
    if expected_merge:
        assert n_tdm == 1, (f"expected fused single tensor_load_to_lds for "
                            f"CACHE_A={CACHE_A!r}, CACHE_B={CACHE_B!r}, got {n_tdm}\n{amdgcn}")
    else:
        assert n_tdm == 2, (f"expected two tensor_load_to_lds for CACHE_A={CACHE_A!r}, "
                            f"CACHE_B={CACHE_B!r}, got {n_tdm}\n{amdgcn}")


if __name__ == "__main__":
    # Smoke test: iterate all cookbook entries at one block size.
    # Run as `python test_tdm_partial_merge.py` on a gfx1250 device.
    if not is_hip_gfx1250():
        raise SystemExit("This script requires a gfx1250 device.")
    print("[2-way: vector_add_tdm_kernel]")
    for p in _HINT_PARAMS:
        ha, hb, em, ident = p
        print(f"-- {ident}: HINT_A=0b{ha:08b}, HINT_B=0b{hb:08b}, expected_merge={em}")
        test_runtime_vector_add_tdm(64, 64, ha, hb, em)
        print("   OK")
    print("[4-way: vector_add_tdm_kernel_4way]")
    for p in _HINT_PARAMS_4WAY:
        ha, hb, hc, hd, ident = p
        print(f"-- {ident}: A=0b{ha:08b} B=0b{hb:08b} C=0b{hc:08b} D=0b{hd:08b}")
        test_runtime_vector_add_tdm_4way(64, 64, ha, hb, hc, hd)
        print("   OK")
