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


@gluon.jit
def tdm_clamp_kernel(a_ptr, M, N, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr):
    """Single TDM load whose offsets exercise the tensor_dim OOB clamp."""
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])

    off_m = ttgl.program_id(axis=0) * BLOCK_M
    off_n = ttgl.program_id(axis=1) * BLOCK_N
    desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(M, N), strides=(N, 1),
                                                       block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    buf = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    ttgl.amd.gfx1250.tdm.async_load(desc, [off_m, off_n], buf)
    ttgl.amd.gfx1250.tdm.async_wait(0)
    buf.load(layout=BLOCKED_LAYOUT)


def test_compile_tdm_clamp_no_readfirstlane():
    """The TDM tensor_dim clamp must stay uniform (SGPR).

    A non-uniform clamp is demoted to VALU and has to be read back into an SGPR
    with ``v_readfirstlane`` next to each ``tensor_load_to_lds``, adding latency
    to every TDM load.  A TDM load whose offsets exercise the clamp must lower
    with none.
    """
    signature = {"a_ptr": "*fp16", "M": "i32", "N": "i32", "BLOCK_M": "constexpr", "BLOCK_N": "constexpr"}
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=tdm_clamp_kernel,
            signature=signature,
            constexprs={"BLOCK_M": 64, "BLOCK_N": 64},
        ),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": 4},
    )
    amdgcn = k.asm["amdgcn"]
    assert "v_readfirstlane" not in amdgcn, f"TDM tensor_dim clamp regressed to VALU (v_readfirstlane emitted)\n{amdgcn}"


@gluon.jit
def tdm_gather_loop_kernel(ptr, optr, M, N, K, BLOCK_N: ttgl.constexpr, NUM_INDICES: ttgl.constexpr):
    """Gather NUM_INDICES rows into shared memory each iteration via TDM, advancing
    the descriptor column per iteration (the v3_tdm/MoE pattern)."""
    num_warps: ttgl.constexpr = ttgl.num_warps()
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [NUM_INDICES, BLOCK_N], [1, 0])
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([NUM_INDICES, 1], [1, 32], [1, num_warps], [1, 0])
    ROW_IDX_LAYOUT: ttgl.constexpr = ttgl.SliceLayout(1, BLOCKED_LAYOUT)

    desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=ptr, shape=(M, N), strides=(N, 1),
                                                       block_shape=(NUM_INDICES, BLOCK_N), layout=SHARED_LAYOUT)
    row_indices = ttgl.arange(0, NUM_INDICES, layout=ROW_IDX_LAYOUT)
    buf = ttgl.allocate_shared_memory(desc.dtype, desc.block_shape, desc.layout)
    acc = ttgl.zeros([NUM_INDICES, BLOCK_N], ttgl.float32, layout=BLOCKED_LAYOUT)
    for _ in range(0, K, BLOCK_N):
        desc = ttgl.amd.gfx1250.tdm.update_tensor_descriptor(desc, add_offsets=[0, BLOCK_N])
        ttgl.amd.gfx1250.tdm.async_gather(desc, src_row_indices=row_indices, dst=buf)
        ttgl.amd.gfx1250.tdm.async_wait(0)
        acc += buf.load(layout=BLOCKED_LAYOUT).to(ttgl.float32)

    offs_m = ttgl.arange(0, NUM_INDICES, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    offs_n = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    ttgl.store(optr + offs_m[:, None] * N + offs_n[None, :], acc.to(ttgl.float16))


def test_compile_tdm_gather_shared_descriptor_sgprs():
    """The base-hoist makes a multi-chunk gather reuse one descriptor across chunks.

    The gather is placed in a loop with the descriptor advancing per iteration, so
    the descriptor is not a compile-time constant (the v3_tdm/MoE pattern).  32 row
    indices lower to four ``tensor_load_to_lds`` per iteration; the base-hoist makes
    all four reuse one group0 SGPR tuple (only the per-chunk lds_addr lane and index
    operands change).  Without it each chunk rebuilds group0 into its own SGPR
    window, giving one distinct group0 per chunk -- this is the part of PR2 that a
    straight-line gather cannot exercise, because there LLVM CSE folds the per-chunk
    rebuilds back together regardless.
    """
    signature = {
        "ptr": "*fp16", "optr": "*fp16", "M": "i32", "N": "i32", "K": "i32", "BLOCK_N": "constexpr", "NUM_INDICES":
        "constexpr"
    }
    k = triton.compile(
        gluon._runtime.GluonASTSource(
            fn=tdm_gather_loop_kernel,
            signature=signature,
            constexprs={"BLOCK_N": 64, "NUM_INDICES": 32},
        ),
        target=GPUTarget("hip", "gfx1250", 32),
        options={"num_warps": 4},
    )
    amdgcn = k.asm["amdgcn"]

    loads = [ln.split() for ln in amdgcn.splitlines() if ln.strip().startswith("tensor_load_to_lds")]
    assert len(loads) >= 4, f"expected a 4-chunk gather, got {len(loads)} TDM instruction(s)\n{amdgcn}"

    # operand 1 = group0 (4 SGPRs).  The base-hoist makes every chunk of the
    # in-loop gather reuse one group0 tuple; without it each chunk rebuilds group0
    # into its own SGPR window (one distinct group0 per chunk).
    group0 = {toks[1].rstrip(",") for toks in loads}
    assert len(group0) == 1, (f"chunks must share one group0 SGPR tuple (base-hoist); got {len(group0)} distinct "
                              f"group0 tuples for {len(loads)} loads: {sorted(group0)}\n{amdgcn}")


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


@gluon.jit
def update_clamp_bounds_kernel(a_ptr, c_ptr, LOG_M, N, OFF_M, BLOCK_M: ttgl.constexpr, BLOCK_N: ttgl.constexpr):
    """Advance a descriptor by OFF_M rows with clamp_bounds=True, then load a full
    BLOCK_M x BLOCK_N tile and store it unmasked.

    clamp_bounds shrinks tensor_dim to the advanced tile's valid extent
    (LOG_M - OFF_M rows), so the hardware loads only the in-bounds rows and
    zero-fills the rest.  The store is intentionally unmasked so the out-of-bounds
    tile region is observable: without the clamp those rows would over-read the
    (physically present) memory past the logical tensor instead of reading zero.
    """
    num_warps: ttgl.constexpr = ttgl.num_warps()
    BLOCKED_LAYOUT: ttgl.constexpr = ttgl.BlockedLayout([1, 8], [4, 8], [num_warps, 1], [1, 0])
    SHARED_LAYOUT: ttgl.constexpr = ttgl.PaddedSharedLayout.with_identity_for([[32, 4]], [BLOCK_M, BLOCK_N], [1, 0])

    a_desc = ttgl.amd.gfx1250.tdm.make_tensor_descriptor(base=a_ptr, shape=(LOG_M, N), strides=(N, 1),
                                                         block_shape=(BLOCK_M, BLOCK_N), layout=SHARED_LAYOUT)
    a_buf = ttgl.allocate_shared_memory(a_desc.dtype, a_desc.block_shape, a_desc.layout)

    a_desc = ttgl.amd.gfx1250.tdm.update_tensor_descriptor(a_desc, add_offsets=[OFF_M, 0], pred=1, clamp_bounds=True)
    ttgl.amd.gfx1250.tdm.async_load(a_desc, [0, 0], a_buf)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    a = a_buf.load(layout=BLOCKED_LAYOUT)
    rm = ttgl.arange(0, BLOCK_M, layout=ttgl.SliceLayout(1, BLOCKED_LAYOUT))
    rn = ttgl.arange(0, BLOCK_N, layout=ttgl.SliceLayout(0, BLOCKED_LAYOUT))
    offs = (rm[:, None] * BLOCK_N) + rn[None, :]
    ttgl.store(c_ptr + offs, a)


@pytest.mark.skipif(not is_hip_gfx1250(), reason="TDM is only tested on gfx1250.")
@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _RUNTIME_BLOCK_SHAPES)
def test_runtime_update_clamp_bounds(BLOCK_M, BLOCK_N):
    """Runtime: update_tensor_descriptor(clamp_bounds=True) shrinks tensor_dim to
    the advanced tile's valid extent, so an edge tile loads only the in-bounds
    rows and the hardware zero-fills the rest.

    The physical source buffer is taller than the logical tensor and filled with
    non-zero data, so a missing/broken clamp is observable end-to-end: the
    out-of-bounds tile rows would come back as the (non-zero) memory past the
    logical tensor instead of zero.  Compared against a torch CPU reference.
    """
    N = BLOCK_N
    LOG_M = BLOCK_M  # logical tensor height
    OFF_M = BLOCK_M // 4  # advance into the tile: valid remaining = 3/4 of a block
    PHYS_M = 2 * BLOCK_M  # physical buffer is taller and entirely non-zero
    NUM_WARPS = 4

    torch.manual_seed(0)
    # FIXME: Switch to native GPU-side initialization once public PyTorch
    # supports gfx1250 kernels.  randint(1, ...) keeps every value non-zero so an
    # over-read is distinguishable from the hardware's OOB zero-fill.
    src_cpu = torch.randint(1, 128, (PHYS_M, N), dtype=torch.int32)
    src = src_cpu.cuda()
    c = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.int32, device="cuda")

    update_clamp_bounds_kernel[(1, )](src, c, LOG_M, N, OFF_M, BLOCK_M, BLOCK_N, num_warps=NUM_WARPS)

    valid = LOG_M - OFF_M
    expected = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.int32)
    expected[:valid, :] = src_cpu[OFF_M:LOG_M, :]
    assert torch.equal(c.cpu(), expected)
