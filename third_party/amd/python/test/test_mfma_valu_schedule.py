"""warp_pipeline_stage(schedule_hint="mfma-valu") and delay=N on gfx950.

Compile-only: the hint has to reach the LLVM IR as a sched.group.barrier
declaration (MFMA / VALU / TRANS groups, marker gone), a stage that does not
qualify has to be dropped with a warning, and delay=N has to become s_nops at
the head of the stage.
"""
import re

import triton
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout, cdna4, warp_pipeline_stage

NUM_WARPS = 4
BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 32

SIGNATURE = {
    "A": "*fp16", "B": "*fp16", "Q": "*fp32", "Out": "*fp32", "N": "i32", "BLOCK_M": "constexpr", "BLOCK_N":
    "constexpr", "BLOCK_K": "constexpr"
}
CONSTEXPRS = {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K}
SGB = re.compile(r"sched\.group\.barrier\(i32 (\d+), i32 (\d+), i32 (\d+)\)")
MASK_VALU, MASK_MFMA, MASK_TRANS = 2, 8, 1024


def _compile(fn, arch="gfx950"):
    with knobs.runtime.scope():
        knobs.runtime.override_arch = None
        return triton.compile(src=gluon._runtime.GluonASTSource(fn, SIGNATURE, CONSTEXPRS),
                              target=GPUTarget("hip", arch, 64), options={"num_warps": NUM_WARPS})


@gluon.jit
def _operands(A, B, Q, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr):
    mma: gl.constexpr = AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                      warps_per_cta=[gl.num_warps(), 1])
    a_layout: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    b_layout: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[16, 4],
                                             warps_per_cta=[gl.num_warps(), 1], order=[1, 0])
    rows_a = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked))
    cols_a = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked))
    a = gl.load(A + rows_a[:, None] * BLOCK_K + cols_a[None, :])
    rows_b = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked))
    cols_b = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked))
    b = gl.load(B + rows_b[:, None] * BLOCK_N + cols_b[None, :])
    rows_q = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked))
    cols_q = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked))
    q = gl.load(Q + rows_q[:, None] * BLOCK_N + cols_q[None, :])
    return gl.convert_layout(a, a_layout), gl.convert_layout(b, b_layout), gl.convert_layout(q, mma)


@gluon.jit
def _store(Out, acc, q, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[16, 4],
                                             warps_per_cta=[gl.num_warps(), 1], order=[1, 0])
    rows = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, blocked))
    cols = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, blocked))
    out = gl.convert_layout(acc + q, blocked)
    gl.store(Out + rows[:, None] * BLOCK_N + cols[None, :], out)


@gluon.jit
def coexec_kernel(A, B, Q, Out, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr):
    """Two-stage loop: a dot stage whose softmax-like VALU work is independent of its dot."""
    a, b, q = _operands(A, B, Q, BLOCK_M, BLOCK_N, BLOCK_K)
    acc = gl.zeros_like(q)
    for _ in range(N):
        with warp_pipeline_stage("dot", schedule_hint="mfma-valu"):
            acc = cdna4.mfma(a, b, acc)
            # VALU and TRANS work on the previous stage's value, not on this stage's dot.
            p = gl.exp2(q * 0.5 - 1.0)
        with warp_pipeline_stage("rescale", delay=16):
            q = p * 0.25 + acc * 0.0
    _store(Out, acc, q, BLOCK_M, BLOCK_N)


@gluon.jit
def dependent_kernel(A, B, Q, Out, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr):
    """Same loop, but the exp2 consumes the stage's own dot result: the hint has to be dropped."""
    a, b, q = _operands(A, B, Q, BLOCK_M, BLOCK_N, BLOCK_K)
    acc = gl.zeros_like(q)
    for _ in range(N):
        with warp_pipeline_stage("dot", schedule_hint="mfma-valu"):
            acc = cdna4.mfma(a, b, acc)
            p = gl.exp2(acc * 0.5)
        with warp_pipeline_stage("rescale"):
            q = p * 0.25
    _store(Out, acc, q, BLOCK_M, BLOCK_N)


def _groups(llir):
    return [(int(m), int(n), int(s)) for m, n, s in SGB.findall(llir)]


def test_hint_declares_groups():
    k = _compile(coexec_kernel)
    groups = _groups(k.asm["llir"])
    assert groups, "the mfma-valu hint must leave a sched.group.barrier declaration in the LLIR"
    assert all(mask != 0 for mask, _, _ in groups), "the stage marker must not survive the pass"
    masks = {mask for mask, _, _ in groups}
    assert MASK_MFMA in masks and MASK_VALU in masks and MASK_TRANS in masks, masks
    # One MFMA per window. LLVM may unroll the loop, so the stage can appear more than once, each
    # copy declared under its own sync id with exactly the stage's MFMAs: one 16x16x32 instruction per
    # tile per wave.
    assert all(n == 1 for mask, n, _ in groups if mask == MASK_MFMA)
    mfmas_per_stage = (BLOCK_M // 16) * (BLOCK_N // 16) // NUM_WARPS
    for sync_id in {s for _, _, s in groups}:
        stage = [(mask, n) for mask, n, s in groups if s == sync_id]
        assert sum(1 for mask, _ in stage if mask == MASK_MFMA) == mfmas_per_stage, stage
        assert any(mask == MASK_TRANS for mask, _ in stage) and any(mask == MASK_VALU for mask, _ in stage), stage


def test_delay_emits_nops():
    k = _compile(coexec_kernel)
    amdgcn = k.asm["amdgcn"]
    # delay=16 is two s_nop 7 (8 cycles each), kept at the head of the stage.
    assert amdgcn.count("s_nop 7") >= 2, amdgcn.count("s_nop 7")


def test_dependent_stage_drops_hint(capfd, monkeypatch):
    # Triton shows MLIR warnings only when asked.
    monkeypatch.setenv("MLIR_ENABLE_DIAGNOSTICS", "warnings")
    k = _compile(dependent_kernel)
    err = capfd.readouterr().err
    assert 'schedule_hint="mfma-valu" dropped' in err, err
    assert "consumes the stage's tt.dot result" in err, err
    assert not _groups(k.asm["llir"]), "a dropped hint must not leave any declaration"
