"""User-facing manual + test suite for partial TDM copies on gfx1250.

This is the canonical reference for `async_load(..., warp_used_hint=H)`
on AMD gfx1250.  `H` selects the active warp subset used for the
per-warp descriptor layout.  The data deposited in shared memory is
unchanged -- only the work split changes.

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

  * Use a subset of warps when the tile is small.
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
Explicit fused copies
================================================================

Use `async_load_fused([(desc0, buf0, H0), (desc1, buf1, H1), ...])`
when several independent TDM loads should be represented as one
explicit fused operation.  The API accepts 2-4 members; each member
is a `(descriptor, destination, warp_used_hint)` tuple.

Fused-copy hints follow the same axis-aligned rule as regular
`async_load(..., warp_used_hint=H)`, and the member hints must be
pairwise disjoint.  Regular `async_load`s stay separate by contract;
manual fusion should use `async_load_fused`.

================================================================
What this file actually tests
================================================================

  * Compile-only tests on the AMDGCN asm: standalone loads yield one
    `tensor_load_to_lds` instruction per `async_load`, while explicit fused
    copies yield one instruction per fused op.
  * Compile-only tests for explicit fused cache modifier propagation.
  * Runtime tests on gfx1250 compare against torch-on-CPU references.

Runtime tests are skipped on non-gfx1250 hosts.

This file is the standalone per-`async_load` reference and the explicit
fused-copy reference.  Hinted and unhinted `async_load`s stay as separate
copies; explicit fused copies are covered below.
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
# individually verifier-legal.
#
# This is the *standalone* per-`async_load` reference, so every pair must
# stay unfused. Explicit fusion is requested with `async_load_fused`.
_HINT_PARAMS = [
    (0b00000000, 0b00000000, "no_hint"),
    (0b00001111, 0b00000011, "lo4_overlap_lo2"),
    (0b01010101, 0b00010001, "strided_overlap_lane"),
    (0b00110011, 0b00000011, "lohi_overlap_lo2"),
    (0b00001111, 0b00001111, "identical_lo4"),
    (0b00000011, 0b00000001, "lo2_overlap_single"),
    (0b11111111, 0b00001111, "full_overlap_lo4"),
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
    """Compile-only: each `async_load` lowers to one `tensor_load_to_lds`.

    Regular copies are standalone by contract.  Explicit fused lowering is
    exercised below.
    """
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
    with triton.knobs.compilation.scope():
        triton.knobs.compilation.always_compile = True
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
    CACHE: ttgl.constexpr,
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
    a_desc = _position_input(a_desc, off_m, off_n)
    b_desc = _position_input(b_desc, off_m, off_n)

    ttgl.amd.gfx1250.tdm.async_load_fused([(a_desc, a_buf, HINT_A), (b_desc, b_buf, HINT_B)], cache_modifier=CACHE)
    ttgl.amd.gfx1250.tdm.async_wait(0)

    c = a_buf.load(layout=BLOCKED_LAYOUT) + b_buf.load(layout=BLOCKED_LAYOUT)
    _store_tile(c_ptr, c, off_m, off_n, M, N, BLOCK_M, BLOCK_N, BLOCKED_LAYOUT)


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


def _tdm_load_llir_calls(llir: str) -> list[str]:
    return re.findall(r"(?:tail )?call void @llvm\.amdgcn\.tensor\.load\.to\.lds[^\n]+", llir)


def _compile_gfx1250(fn, ptr_names, constexprs, *, ptr_ty="*fp16", num_warps=8):
    """Compile `fn` for gfx1250.

    The signature is `{ptrs: ptr_ty, M/N: i32, <constexpr keys>: constexpr}`,
    matching every fused-copy kernel below.
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
    return k


def _compile_amdgcn(fn, ptr_names, constexprs, *, ptr_ty="*fp16", num_warps=8) -> str:
    """Compile `fn` for gfx1250 and return its AMDGCN asm."""
    k = _compile_gfx1250(fn, ptr_names, constexprs, ptr_ty=ptr_ty, num_warps=num_warps)
    return k.asm["amdgcn"]


def test_compile_vector_add_tdm_explicit_fused():
    """Compile-only: explicit fused Gluon API lowers to one TDM intrinsic."""
    amdgcn = _compile_amdgcn(
        vector_add_tdm_explicit_fused_kernel,
        ["a_ptr", "b_ptr", "c_ptr"],
        {"BLOCK_M": 64, "BLOCK_N": 64, "HINT_A": 0b00001111, "HINT_B": 0b11110000, "CACHE": ""},
    )
    _assert_tensor_load_count(amdgcn, 1, "explicit async_load_fused")


def test_compile_vector_add_tdm_explicit_fused_cache_modifier():
    """Compile-only: explicit fused Gluon API propagates cache modifiers."""
    default_kernel = _compile_gfx1250(
        vector_add_tdm_explicit_fused_kernel,
        ["a_ptr", "b_ptr", "c_ptr"],
        {"BLOCK_M": 64, "BLOCK_N": 64, "HINT_A": 0b00001111, "HINT_B": 0b11110000, "CACHE": ""},
    )
    cg_kernel = _compile_gfx1250(
        vector_add_tdm_explicit_fused_kernel,
        ["a_ptr", "b_ptr", "c_ptr"],
        {"BLOCK_M": 64, "BLOCK_N": 64, "HINT_A": 0b00001111, "HINT_B": 0b11110000, "CACHE": ".cg"},
    )
    _assert_tensor_load_count(cg_kernel.asm["amdgcn"], 1, "explicit async_load_fused cache modifier")

    # On gfx1250, `.cg` lowers to DEV scope / regular cache behavior: aux = 16.
    # The default cache modifier lowers to aux = 0.
    default_calls = _tdm_load_llir_calls(default_kernel.asm["llir"])
    cg_calls = _tdm_load_llir_calls(cg_kernel.asm["llir"])
    assert len(default_calls) == 1
    assert len(cg_calls) == 1
    assert re.search(r", i32 0\)(?:, .*)?$", default_calls[0])
    assert re.search(r", i32 16\)(?:, .*)?$", cg_calls[0])


# 3-way copies: covers regular hinted separation.


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
    (8, 0b00001111, 0b00110000, 0b11000000, "three_way_8w_disjoint"),
    (4, 0b0011, 0b0100, 0b1000, "three_way_4w_disjoint"),
    # Pairwise-disjoint hints whose union is not a coset.
    (8, 0b00000001, 0b00001000, 0b00010000, "three_way_noncoset_union"),
]


def _compile_3way(num_warps, hints=(None, None, None), block=(64, 64)) -> str:
    a, b, c = hints
    return _compile_amdgcn(vector_add_tdm_kernel_3way, ["a_ptr", "b_ptr", "c_ptr", "out_ptr"],
                           {"BLOCK_M": block[0], "BLOCK_N": block[1], "HINT_A": a, "HINT_B": b, "HINT_C": c},
                           num_warps=num_warps)


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


# 4-way copies: exercises the N=4 member-predicate path.


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

    out = (a_buf.load(layout=BLOCKED_LAYOUT) + b_buf.load(layout=BLOCKED_LAYOUT) + c_buf.load(layout=BLOCKED_LAYOUT) +
           d_buf.load(layout=BLOCKED_LAYOUT))
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
    amdgcn = _compile_amdgcn(vector_add_tdm_kernel_4way, ["a_ptr", "b_ptr", "c_ptr", "d_ptr", "out_ptr"], {
        "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "HINT_A": HINT_A, "HINT_B": HINT_B, "HINT_C": HINT_C, "HINT_D": HINT_D
    })
    context = (f"HINT_A=0b{HINT_A:08b}, HINT_B=0b{HINT_B:08b}, "
               f"HINT_C=0b{HINT_C:08b}, HINT_D=0b{HINT_D:08b}")
    _assert_tensor_load_count(amdgcn, 4, context)


# Heterogeneous destination MemDescTypes.


@gluon.jit
def heterogeneous_tdm_kernel(
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
        heterogeneous_tdm_kernel, ["a_ptr", "b_ptr", "as_ptr", "bs_ptr"], {
            "BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_N_B": 2048, "BLOCK_SCALE_M": 64, "BLOCK_SCALE_N": 32, "HINT_A":
            0b00010001, "HINT_B": 0b00100010, "HINT_AS": 0b01000100, "HINT_BS": 0b10001000
        }, ptr_ty="*i8")
    _assert_tensor_load_count(amdgcn, 4, "heterogeneous hinted A/B/AS/BS loads")


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


@pytest.mark.skipif(not is_hip_gfx1250(), reason="TDM is only tested on gfx1250.")
@pytest.mark.parametrize("BLOCK_M,BLOCK_N", _RUNTIME_BLOCK_SHAPES)
def test_runtime_vector_add_tdm_explicit_fused(BLOCK_M, BLOCK_N):
    """Runtime: explicit fused TDM API produces the same result as two loads."""
    M, N = 256, 512
    NUM_WARPS = 8

    torch.manual_seed(0)
    a_cpu = torch.randint(0, 128, (M, N), dtype=torch.int32)
    b_cpu = torch.randint(0, 128, (M, N), dtype=torch.int32)
    a = a_cpu.cuda()
    b = b_cpu.cuda()
    c = torch.empty((M, N), dtype=torch.int32, device="cuda")

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    vector_add_tdm_explicit_fused_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        BLOCK_M,
        BLOCK_N,
        0b00001111,
        0b11110000,
        "",
        num_warps=NUM_WARPS,
    )

    assert torch.equal(c.cpu(), a_cpu + b_cpu)


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
