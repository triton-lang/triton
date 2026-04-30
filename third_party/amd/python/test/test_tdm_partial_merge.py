"""Vector-add correctness tests for `warp_used_hint` (partial TDM copy)
and the implicit lowering-only merge for adjacent TDM copies.

The kernel under test loads two source tensors via
`async_tdm_copy_global_to_local` into separate shared-memory buffers,
optionally annotating each load with a per-op `warp_used_hint`, then
computes ``c = a + b`` in registers and stores `c`.  Each parametrised
case picks a different hint pattern (none / partial / pairwise-disjoint
mergeable / pairwise-disjoint with an illegal union) so the suite
exercises:

  * The singleton no-hint baseline (no merge analysis).
  * Singleton hinted copies (verifier-legal axis-aligned cosets, including
    canonical prefixes, shifted prefixes, and strided cosets).
  * Mixed hinted/unhinted pairs (merge analysis declines because one op
    has no hint).
  * Mergeable pairs whose union is also a verifier-legal coset (lowering
    must fuse to a single `tensor_load_to_lds` intrinsic).
  * Pairwise-disjoint pairs whose union is *not* a legal coset (lowering
    must keep them as two intrinsics).

The runtime test compares against ``a + b`` from torch on gfx1250
hardware; the compile-only test inspects the AMDGCN asm to assert the
expected fused/un-fused instruction count.
"""

import re

import pytest
import torch

import triton
from triton.backends.compiler import GPUTarget
from triton._internal_testing import is_hip_gfx1250
from triton.experimental import gluon
import triton.experimental.gluon.language as ttgl

# ---------------------------------------------------------------------------
# Kernel: load two NxN tiles via TDM (with optional hints), add and store.
# Two adjacent `async_load` ops are emitted to exercise merge-eligible
# lowering patterns.
# ---------------------------------------------------------------------------


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

    # Two adjacent TDM copies.  When both carry a hint and the pair is
    # merge-legal, the lowering implicitly fuses them into a single TDM
    # intrinsic; otherwise each emits its own intrinsic.  In all cases
    # the SSA destinations are different shared-memory buffers, so there
    # is no aliasing concern.  The kernel deliberately uses the same
    # shared encoding for both buffers so the mergeability filter can
    # accept the pair.
    if HINT_A == 0:
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf)
    else:
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    if HINT_B == 0:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf)
    else:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B)

    # User-owned waitcnt: count is "remaining outstanding TDM ops", which
    # depends on whether the pair was merged.  Wait for everything (0).
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
# Hint table.  Each entry is (HINT_A, HINT_B, expected_merge, ID).
#
# `K = popcount(hint)` is the number of active warps a hint selects (one
# bit = one warp); the verifier requires `K` to be a power of two and
# the active set to be an axis-aligned coset.  See
# `AsyncTDMCopyGlobalToLocalOp::validateWarpUsedHint` in
# `third_party/amd/lib/Dialect/TritonAMDGPU/IR/Dialect.cpp` for the full
# rule.
#
# `expected_merge=True` means the pair is mergeable: both hints are
# non-zero, pairwise disjoint, have a verifier-legal axis-aligned-coset
# union, share the same shared encoding and shape, carry no mbarrier, and
# have no intervening side-effecting op.
#
# Examples chosen to exercise different basis-bit / i0 placements
# (bit i set => warp i is active):
#   - 0b00001111 + 0b11110000   (canonical prefix + high-half coset, basis {0,1})
#   - 0b01010101 + 0b10101010   (strided cosets, basis {1,2})
#   - 0b00110011 + 0b11001100   (lo/hi pair cosets, basis {0,2})
#   - 0b00000011 + 0b00001100   (K=2 each, union K=4 prefix; 4 warps idle)
# Negative example:
#   - 0b00000011 + 0b00010000   (K=2 + K=1, union popcount=3 -> not power of two)
#
# Two of the mergeable entries below are also constructed via the public
# `tdm.warp_prefix` / `tdm.warp_shifted_prefix` / `tdm.warp_strided`
# helpers (rather than a raw bitmask literal) so this file doubles as a
# usage example for the recommended hint constructors.
# ---------------------------------------------------------------------------

# Hint-constructor helpers (also available as `ttgl.amd.gfx1250.tdm.*`):
# always produce verifier-legal axis-aligned cosets.
_warp_prefix = ttgl.amd.gfx1250.tdm.warp_prefix
_warp_shifted_prefix = ttgl.amd.gfx1250.tdm.warp_shifted_prefix
_warp_strided = ttgl.amd.gfx1250.tdm.warp_strided

_HINT_PARAMS = [
    # (HINT_A, HINT_B, expected_merge, id).  The kernel maps HINT == 0 to the
    # no-kwarg `async_load` (i.e. the `warp_used_hint` attribute is absent on
    # the op), since the verifier rejects an explicit `warp_used_hint = 0`.
    (0b00000000, 0b00000000, False, "no_hint"),
    (0b11111111, 0b00000000, False, "full_a_unhinted_b"),
    (0b00001111, 0b00000000, False, "lo_prefix_a_unhinted_b"),
    (0b11110000, 0b00000000, False, "hi_prefix_a_unhinted_b"),
    (0b01010101, 0b00000000, False, "strided_a_unhinted_b"),
    (0b00000001, 0b00000010, True, "merge_single_warp_pair"),
    # 0b00001111 + 0b11110000 via helpers: warp_prefix(4)=0x0f selects
    # warps 0..3, warp_shifted_prefix(4, 4)=0xf0 selects warps 4..7.
    (_warp_prefix(4), _warp_shifted_prefix(4, 4), True, "merge_lo_hi_prefix"),
    # 0b01010101 + 0b10101010 via helpers: warp_strided(4, 1)=0x55 picks
    # every other warp starting from 0; the second hint shifts i0 by 1.
    (_warp_strided(4, 1), _warp_strided(4, 1) << 1, True, "merge_strided"),
    (0b00110011, 0b11001100, True, "merge_lo_hi_pairs"),
    (0b00000011, 0b00001100, True, "merge_partial_K4_idle"),
    (0b00000011, 0b00010000, False, "disjoint_but_union_illegal"),
]


def _hint_param_args(p):
    """Strip the trailing id off a parametrised entry."""
    return p[:-1]


def _hint_id(p):
    return p[-1]


# ---------------------------------------------------------------------------
# Compile-only test: validates the lowering produces the expected number
# of `tensor_load_to_lds` instructions.  Runs without a GPU (just compiles
# for gfx1250).
# ---------------------------------------------------------------------------

_COMPILE_BLOCK_SHAPES = [(64, 64), (32, 128)]


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "HINT_A,HINT_B,expected_merge",
    [_hint_param_args(p) for p in _HINT_PARAMS],
    ids=[_hint_id(p) for p in _HINT_PARAMS],
)
def test_compile_vector_add_tdm(BLOCK_M, BLOCK_N, HINT_A, HINT_B, expected_merge):
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
    [_hint_param_args(p) for p in _HINT_PARAMS],
    ids=[_hint_id(p) for p in _HINT_PARAMS],
)
def test_runtime_vector_add_tdm(BLOCK_M, BLOCK_N, HINT_A, HINT_B, expected_merge):
    M, N = 256, 512
    NUM_WARPS = 8

    # All torch math happens on CPU — torch's bundled HIP runtime
    # often lacks kernels for gfx1250 (rand/add/zeros_like all hit
    # `hipErrorInvalidImage`), so we keep the only GPU kernel that
    # runs the triton vector-add itself.  Inputs are generated on
    # host, moved to device with `.cuda()` (a pure memcpy, no
    # kernel), and the output buffer is `torch.empty` (allocation
    # only).  The reference is computed on host and compared after
    # `.cpu()`-ing the kernel output.
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


# ---------------------------------------------------------------------------
# 4-way merge kernel and tests.
#
# The 2-way kernel above only exercises N=2 fusion (one `s_cselect_b32`).
# This kernel issues four adjacent hinted `async_load`s into four shared
# buffers and sums them, exercising the N=4 lowering path which chains
# three `s_cselect_b32`s and uses a 2-bit per-wave selector.  Every case
# below is intentionally positive (mergeable): negative behaviour is
# already covered by the lit suite (`tritongpu_tdm_to_llvm.mlir`) and by
# the 2-way table above.
# ---------------------------------------------------------------------------


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

    # Four adjacent hinted TDM copies.  Mergeability requires:
    #   * every member carries a verifier-legal `warp_used_hint`,
    #   * pairwise disjoint with equal popcount K per member,
    #   * union forms a verifier-legal axis-aligned coset,
    #   * same destination shared encoding + identical block shape
    #     (enforced by the common SHARED_LAYOUT and block_shape above),
    #   * no intervening side-effect, no mbarrier on any member.
    # When met, the lowering fuses all four into a single
    # `tensor_load_to_lds` intrinsic via a 2-bit per-wave selector.
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


# Positive 4-way merge cases.  Every entry's hints must be pairwise
# disjoint with equal K and a verifier-legal union coset, so the
# lowering fuses to a single `tensor_load_to_lds` instruction.
#
#   - four_K1_prefix_union: K=1 each, union 0b00001111 (K=4 prefix).
#   - four_K2_full_union:   K=2 each, union 0b11111111 (K=8, all warps).
_HINT_PARAMS_4WAY = [
    (0b00000001, 0b00000010, 0b00000100, 0b00001000, "four_K1_prefix_union"),
    (0b00000011, 0b00001100, 0b00110000, 0b11000000, "four_K2_full_union"),
]


def _hint4_args(p):
    return p[:-1]


def _hint4_id(p):
    return p[-1]


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "HINT_A,HINT_B,HINT_C,HINT_D",
    [_hint4_args(p) for p in _HINT_PARAMS_4WAY],
    ids=[_hint4_id(p) for p in _HINT_PARAMS_4WAY],
)
def test_compile_vector_add_tdm_4way(BLOCK_M, BLOCK_N, HINT_A, HINT_B, HINT_C, HINT_D):
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
    [_hint4_args(p) for p in _HINT_PARAMS_4WAY],
    ids=[_hint4_id(p) for p in _HINT_PARAMS_4WAY],
)
def test_runtime_vector_add_tdm_4way(BLOCK_M, BLOCK_N, HINT_A, HINT_B, HINT_C, HINT_D):
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
        a, b, c, d, out, M, N, BLOCK_M, BLOCK_N, HINT_A, HINT_B, HINT_C, HINT_D,
        num_warps=NUM_WARPS,
    )
    expected = a_cpu + b_cpu + c_cpu + d_cpu
    torch.testing.assert_close(out.cpu(), expected, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    # Convenience entry for ad-hoc runs:
    #   python test_tdm_partial_merge.py
    # Iterates every hint param against a single block size for fast
    # smoke-testing on a connected gfx1250 device.
    if not is_hip_gfx1250():
        raise SystemExit("This script requires a gfx1250 device.")
    for p in _HINT_PARAMS:
        ha, hb, em, ident = p
        print(f"-- {ident}: HINT_A=0b{ha:08b}, HINT_B=0b{hb:08b}, expected_merge={em}")
        test_runtime_vector_add_tdm(64, 64, ha, hb, em)
        print("   OK")
    for p in _HINT_PARAMS_4WAY:
        ha, hb, hc, hd, ident = p
        print(f"-- {ident}: A=0b{ha:08b} B=0b{hb:08b} C=0b{hc:08b} D=0b{hd:08b}")
        test_runtime_vector_add_tdm_4way(64, 64, ha, hb, hc, hd)
        print("   OK")
