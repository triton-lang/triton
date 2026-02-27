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
    """Minimal 2-stage pipeline — baseline for s_setprio check."""
    blocked: gl.constexpr = gl.BlockedLayout(size_per_thread=[1, 8], threads_per_warp=[16, 4],
                                             warps_per_cta=[gl.num_warps(), 1], order=[1, 0])
    offs_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked)
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=offs_layout)

    for i in tl.range(0, 4):
        with warp_pipeline_stage("stage1", priority=1):
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
