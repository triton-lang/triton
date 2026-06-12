import re

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import warp_pipeline_stage

NUM_WARPS = 4
BLOCK = 128


def _compile_gluon(fn, signature, constexprs, arch="gfx950"):
    """Compile a Gluon kernel for gfx950 and return the compiled object."""
    return triton.compile(src=gluon._runtime.GluonASTSource(fn, signature, constexprs),
                          target=GPUTarget("hip", arch, 64), options={"num_warps": NUM_WARPS})


@gluon.jit
def pipeline_simple(A, B, BLOCK: gl.constexpr):
    """Minimal 2-stage pipeline -- baseline for s_setprio check."""
    blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[16, 4],
                                             warps_per_cta=[gl.num_warps(), 1], order=[1, 0])
    offs_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked)
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=offs_layout)

    for i in tl.range(0, 4):
        with warp_pipeline_stage("stage1", priority=1):
            a = gl.load(A + offs)

        with warp_pipeline_stage("stage2", priority=0):
            gl.store(B + offs, a)


@gluon.jit
def pipeline_unrolled(A, B, N, BLOCK: gl.constexpr):
    """2-stage pipeline with loop unrolling -- IV used inside stages.

    The IV must be referenced inside execute_region bodies so only the
    scalar IV-remap ops (arith.muli + arith.addi on the index) land
    between the cloned clusters after unrolling.  Gluon skips make_ttir,
    so loop_unroll_factor survives until after add_warp_pipeline.
    """
    blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[16, 4],
                                             warps_per_cta=[gl.num_warps(), 1], order=[1, 0])
    offs_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked)
    base = gl.arange(0, BLOCK, layout=offs_layout)

    for i in tl.range(0, N, loop_unroll_factor=2):
        with warp_pipeline_stage("stage1", priority=1):
            offs = base + i * BLOCK
            a = gl.load(A + offs)

        with warp_pipeline_stage("stage2", priority=0):
            gl.store(B + offs, a)


def test_simple_pipeline_setprio():
    """Baseline: a simple 2-stage pipeline emits s_setprio."""
    signature = {"A": "*fp16", "B": "*fp16", "BLOCK": "constexpr"}
    constexprs = {"BLOCK": BLOCK}
    k = _compile_gluon(pipeline_simple, signature, constexprs)
    amdgcn = k.asm["amdgcn"]
    assert amdgcn.count("s_setprio") > 0, "Simple pipeline must emit s_setprio."


def test_pipeline_with_unroll():
    """Warp pipeline + loop unrolling: IV remap ops between stages must not break conversion.

    Gluon kernels skip make_ttir, so loop_unroll_factor survives until the
    Gluon pipeline where add_loop_unroll runs AFTER add_warp_pipeline.  The
    MLIR unroller emits arith.muli+arith.addi for IV remapping between the
    cloned execute_region clusters -- ConvertWarpPipeline must tolerate these.
    """
    signature = {"A": "*fp16", "B": "*fp16", "N": "i32", "BLOCK": "constexpr"}
    constexprs = {"BLOCK": BLOCK}
    k = _compile_gluon(pipeline_unrolled, signature, constexprs)
    amdgcn = k.asm["amdgcn"]
    # Each successfully converted loop emits s_setprio instructions.
    # With unroll factor 2 and dynamic bound, we get a main loop + epilogue.
    # Both must convert: without the fix only the epilogue converts (4 s_setprio),
    # with the fix both convert (10 s_setprio).
    assert amdgcn.count("s_setprio") > 4, \
        f"Expected > 4 s_setprio (both main+epilogue converted), got {amdgcn.count('s_setprio')}"


@gluon.jit
def pipeline_static_range_unrolled(A, B, N, BLOCK: gl.constexpr):
    """Outer loop_unroll_factor + static_range wrapping both stages.

    The outer tl.range has loop_unroll_factor=2, so MLIR unrolling injects
    scalar IV remap ops between cloned bodies.  Inside each iteration,
    static_range(2) duplicates both pipeline stages at the AST level.
    This is the worst-case combo for ConvertWarpPipeline: the loop body
    has 4 execute_region clusters with scalar ops between them.
    """
    blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[16, 4],
                                             warps_per_cta=[gl.num_warps(), 1], order=[1, 0])
    offs_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked)
    base = gl.arange(0, BLOCK, layout=offs_layout)

    for i in tl.range(0, N, loop_unroll_factor=2):
        for j in gl.static_range(2):
            with warp_pipeline_stage("stage1", priority=1):
                offs = base + (i + j) * BLOCK
                a = gl.load(A + offs)

            with warp_pipeline_stage("stage2", priority=0):
                gl.store(B + offs, a)


def test_pipeline_static_range_with_unroll():
    """static_range wrapping both stages + outer loop_unroll_factor must convert."""
    signature = {"A": "*fp16", "B": "*fp16", "N": "i32", "BLOCK": "constexpr"}
    constexprs = {"BLOCK": BLOCK}
    k = _compile_gluon(pipeline_static_range_unrolled, signature, constexprs)
    ttgir = k.asm["ttgir"]
    pipelined_loop_pattern = r"scf\.for\b.*?\n    \} \{[^}]*triton\.warp_pipeline\.pipelined_for[^}]*\}"
    pipelined_loops = re.findall(pipelined_loop_pattern, ttgir, re.S)
    stage_counts = [loop.count("triton.warp_pipeline.stage") for loop in pipelined_loops]
    priority_counts = [(loop.count("triton.warp_pipeline.priority = 1"),
                        loop.count("triton.warp_pipeline.priority = 0")) for loop in pipelined_loops]
    # Main loop has 8 clusters (static_range(2) * 2 stages * unroll(2)).
    # Epilogue has 4 clusters (static_range(2) * 2 stages).
    assert stage_counts == [8, 4], \
        f"Expected main+epilogue warp-pipelined loops with 8 and 4 stages, got {stage_counts}"
    assert priority_counts == [(4, 4), (2, 2)], \
        f"Expected balanced stage priorities in ttgir, got {priority_counts}"
