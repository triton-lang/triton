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
# `expected_merge=True` means the pair is mergeable: both hints are
# non-zero, pairwise disjoint, have a verifier-legal axis-aligned-coset
# union, share the same shared encoding and shape, carry no mbarrier, and
# have no intervening side-effecting op.
#
# Examples chosen to exercise different basis-bit / i0 placements:
#   - 0x0F + 0xF0     (canonical prefix + high-half coset, basis {0,1})
#   - 0x55 + 0xAA     (strided cosets, basis {1,2})
#   - 0x33 + 0xCC     (lo/hi pair cosets, basis {0,2})
#   - 0x03 + 0x0C     (K=2 each, union K=4 prefix; 4 warps idle)
# Negative example:
#   - 0x03 + 0x10     (K=2 + K=1, union popcount=3 -> not power of two)
# ---------------------------------------------------------------------------

_HINT_PARAMS = [
    # (HINT_A, HINT_B, expected_merge, id)
    (0, 0, False, "no_hint"),
    (0xFF, 0, False, "full_a_unhinted_b"),
    (0x0F, 0, False, "lo_prefix_a_unhinted_b"),
    (0xF0, 0, False, "hi_prefix_a_unhinted_b"),
    (0x55, 0, False, "strided_a_unhinted_b"),
    (0x01, 0x02, True, "merge_single_warp_pair"),
    (0x0F, 0xF0, True, "merge_lo_hi_prefix"),
    (0x55, 0xAA, True, "merge_strided"),
    (0x33, 0xCC, True, "merge_lo_hi_pairs"),
    (0x03, 0x0C, True, "merge_partial_K4_idle"),
    (0x03, 0x10, False, "disjoint_but_union_illegal"),
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
                            f"HINT_A=0x{HINT_A:x}, HINT_B=0x{HINT_B:x}, got {n_tdm}\n{amdgcn}")
    else:
        assert n_tdm == 2, (f"expected two tensor_load_to_lds for HINT_A=0x{HINT_A:x}, "
                            f"HINT_B=0x{HINT_B:x}, got {n_tdm}\n{amdgcn}")


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


if __name__ == "__main__":
    # Convenience entry for ad-hoc runs:
    #   python test_tdm_partial_merge.py
    # Iterates every hint param against a single block size for fast
    # smoke-testing on a connected gfx1250 device.
    if not is_hip_gfx1250():
        raise SystemExit("This script requires a gfx1250 device.")
    for p in _HINT_PARAMS:
        ha, hb, em, ident = p
        print(f"-- {ident}: HINT_A=0x{ha:x}, HINT_B=0x{hb:x}, expected_merge={em}")
        test_runtime_vector_add_tdm(64, 64, ha, hb, em)
        print("   OK")
