import expecttest
import re

from triton._filecheck import run_parser
from triton.backends.compiler import GPUTarget
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

HIP_TARGET_CDNA4 = GPUTarget("hip", "gfx950", 64)

_TARGET_PAT = re.compile(r'ttg\.target = "[^"]*"')


def _anonymize_ir(ir):
    return _TARGET_PAT.sub('ttg.target = "..."', ir)


def _make_args(*args, **kwargs):
    return args, kwargs


def test_compute_efficient_padded_shared_layout_op_a_fp16():

    @gluon.jit
    def kernel():
        mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True,
                                                             warps_per_cta=[2, 2])
        dot_op_a: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)
        shared_a: ttgl.constexpr = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(
            dot_op_a, [128, 64], elem_bytes=2)
        ttgl.allocate_shared_memory(ttgl.float16, [128, 64], shared_a)

    module = run_parser(kernel, *_make_args(num_warps=4), target=HIP_TARGET_CDNA4)
    expecttest.assert_expected_inline(
        _anonymize_ir(module.str_nodebug()), """\
#shared = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0], [1, 0], [2, 0], [4, 0], [8, 0]], block = []}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
""")


def test_compute_efficient_padded_shared_layout_op_b_fp8():

    @gluon.jit
    def kernel():
        mfma_layout: ttgl.constexpr = ttgl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 128], transposed=True,
                                                             warps_per_cta=[2, 2])
        dot_op_b: ttgl.constexpr = ttgl.DotOperandLayout(operand_index=1, parent=mfma_layout, k_width=16)
        shared_b: ttgl.constexpr = ttgl.amd.cdna4.compute_efficient_padded_shared_layout(
            dot_op_b, [128, 128], elem_bytes=1)
        ttgl.allocate_shared_memory(ttgl.float8e4nv, [128, 128], shared_b)

    module = run_parser(kernel, *_make_args(num_warps=4), target=HIP_TARGET_CDNA4)
    expecttest.assert_expected_inline(
        _anonymize_ir(module.str_nodebug()), """\
#shared = #ttg.padded_shared<[1024:+32] {offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [0, 16], [0, 32], [0, 64], [0, 1], [0, 2], [0, 4], [0, 8]], block = []}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "...", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @kernel() attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem, mutable>
    tt.return
  }
}
""")


def test_compute_efficient_padded_shared_layout_invalid_returns_none():
    mfma_layout = ttgl.amd.AMDMFMALayout(version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2])
    dot_op = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=8)

    # k_width=2 is outside {4, 8, 16}.
    bad_kwidth_dot_op = ttgl.DotOperandLayout(operand_index=0, parent=mfma_layout, k_width=2)
    assert ttgl.amd.cdna4.compute_efficient_padded_shared_layout(bad_kwidth_dot_op, [128, 64], elem_bytes=2) is None

    # elem_bytes=4 (e.g. fp32) is outside {1, 2}.
    assert ttgl.amd.cdna4.compute_efficient_padded_shared_layout(dot_op, [128, 64], elem_bytes=4) is None
