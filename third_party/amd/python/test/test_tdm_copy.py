"""User-facing manual + test suite for partial TDM copies on gfx1250.

This is the canonical reference for `async_load(..., warp_used_hint=H)`
on AMD gfx1250.  `H` selects which warps participate in a single TDM
transfer; cleared warps become hardware no-ops.  The data deposited
in shared memory is unchanged -- only the work split changes.

Worked example: `vector_add_tdm_kernel` issues two independent partial
TDM copies into distinct shared buffers and then sums them.

================================================================
The rule: `H` must be axis-aligned
================================================================

`H` is legal iff its active warps share fixed values on some warpId
bit positions and vary freely on the others.  Equivalently: pick
which warpId bits are free, pin the others to fixed values; the
active warps are exactly those whose warpId matches the pinned bits.
K = popcount(H) is therefore a power of two (= 2^(number of free
bits)) with `1 <= K <= num_warps`.

For full details and rationale, see `AsyncTDMCopyGlobalToLocalOp`
in `third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td`.

================================================================
Cookbook: legal 8-warp hints
================================================================

Bit 7 on the left, bit 0 on the right.

  +-------------+---+--------------------------------------------+
  | Bitmask     | K | Active warps      (pinned bits)            |
  +-------------+---+--------------------------------------------+
  | 0b00001111  | 4 | {0,1,2,3}         (bit 2 = 0)              |
  | 0b11110000  | 4 | {4,5,6,7}         (bit 2 = 1)              |
  | 0b01010101  | 4 | {0,2,4,6}         (bit 0 = 0)              |
  | 0b10101010  | 4 | {1,3,5,7}         (bit 0 = 1)              |
  | 0b00110011  | 4 | {0,1,4,5}         (bit 1 = 0)              |
  | 0b00000011  | 2 | {0,1}             (bits 1,2 = 0)           |
  | 0b00010000  | 1 | {4}               (single warp)            |
  +-------------+---+--------------------------------------------+

Rejected patterns:

  +-------------+----------------------------------------------------+
  | 0           | rejected: must select at least one warp; pass None |
  | 0b00000111  | rejected: K=3 is not a power of two                |
  | 0b01101001  | rejected: warps {0,3,5,6} are not axis-aligned     |
  +-------------+----------------------------------------------------+

================================================================
When to reach for `warp_used_hint`
================================================================

Default `async_load(desc, idx, buf)` slices the tile across all
`num_warps` warps.  Reach for `warp_used_hint=H` to:

  * Free warps for unrelated work when the tile is small.
  * Partition warps by role (producer/consumer pipelines).

Omit or pass `None` for "all warps"; an explicit `warp_used_hint=0`
is rejected by the verifier.

================================================================
Constructing a hint generically
================================================================

If `K` (number of active warps) is parameterised:

  * "first K warps":              `(1 << K) - 1`
  * "K warps starting at off":    `((1 << K) - 1) << off`
                                  (off must be a multiple of K)
  * "K warps spaced by 2**s":     `sum(1 << (i << s) for i in range(K))`

================================================================
What this file actually tests
================================================================

  * Compile-only test on the AMDGCN asm: every parametrisation yields
    two `tensor_load_to_lds` instructions (one per `async_load`).
  * Runtime test on gfx1250 compares against a torch-on-CPU reference.

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
    rejects an explicit `warp_used_hint = 0`).  Each load issues an
    independent partial TDM copy into its own shared buffer.
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

    if HINT_A == 0:
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf)
    else:
        ttgl.amd.gfx1250.tdm.async_load(a_desc, [off_m, off_n], a_buf, warp_used_hint=HINT_A)
    if HINT_B == 0:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf)
    else:
        ttgl.amd.gfx1250.tdm.async_load(b_desc, [off_m, off_n], b_buf, warp_used_hint=HINT_B)

    ttgl.amd.gfx1250.tdm.async_wait(0)

    a = a_buf.load(layout=BLOCKED_LAYOUT)
    b = b_buf.load(layout=BLOCKED_LAYOUT)
    c = a + b

    offs_m = off_m + ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_n = off_n + ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs = (offs_m[:, None] * N) + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    ttgl.store(c_ptr + offs, c, mask=mask)


# Hint cookbook for `vector_add_tdm_kernel`.  Layout: (HINT_A, HINT_B,
# id).  Bit `i` => warp `i`; `0` selects the kwarg-free load (an
# explicit zero is rejected by the verifier).  All entries below are
# verifier-legal.
_HINT_PARAMS = [
    (0b00000000, 0b00000000, "no_hint"),
    (0b11111111, 0b00000000, "full_a_unhinted_b"),
    (0b00001111, 0b11110000, "lo4_hi4"),
    (0b01010101, 0b10101010, "strided_pair"),
    (0b00110011, 0b11001100, "lo_hi_pairs"),
    (0b00000011, 0b00001100, "K2_low_warps"),
    (0b00000001, 0b00000010, "single_warp_pair"),
]


def _param_args(p):
    """Strip trailing id from a parametrised entry."""
    return p[:-1]


def _param_id(p):
    return p[-1]


_COMPILE_BLOCK_SHAPES = [(64, 64), (32, 128)]


@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _COMPILE_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "HINT_A,HINT_B",
    [_param_args(p) for p in _HINT_PARAMS],
    ids=[_param_id(p) for p in _HINT_PARAMS],
)
def test_compile_vector_add_tdm(BLOCK_M, BLOCK_N, HINT_A, HINT_B):
    """Compile-only: each `async_load` lowers to one `tensor_load_to_lds`."""
    NUM_WARPS = 8
    signature = {
        "a_ptr": "*i32",
        "b_ptr": "*i32",
        "c_ptr": "*i32",
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
    assert n_tdm == 2, (f"expected two tensor_load_to_lds for HINT_A=0b{HINT_A:08b}, "
                        f"HINT_B=0b{HINT_B:08b}, got {n_tdm}\n{amdgcn}")


_RUNTIME_BLOCK_SHAPES = [(64, 64), (128, 64)]


@pytest.mark.skipif(not is_hip_gfx1250(), reason="TDM is only tested on gfx1250.")
@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _RUNTIME_BLOCK_SHAPES)
@pytest.mark.parametrize(
    "HINT_A,HINT_B",
    [_param_args(p) for p in _HINT_PARAMS],
    ids=[_param_id(p) for p in _HINT_PARAMS],
)
def test_runtime_vector_add_tdm(BLOCK_M, BLOCK_N, HINT_A, HINT_B):
    """Runtime: integer c = a + b vs torch CPU reference."""
    M, N = 256, 512
    NUM_WARPS = 8

    torch.manual_seed(0)
    # FIXME: Switch to native GPU-side initialization once public PyTorch
    # supports gfx1250 kernels.
    a_cpu = torch.randint(0, 128, (M, N), dtype=torch.int32)
    b_cpu = torch.randint(0, 128, (M, N), dtype=torch.int32)
    a = a_cpu.cuda()
    b = b_cpu.cuda()
    c = torch.empty((M, N), dtype=torch.int32, device="cuda")

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
    assert torch.equal(c.cpu(), expected)
