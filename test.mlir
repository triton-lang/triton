
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// CHECK-LABEL: cvt_mma_to_dot_fp8
// CHECK: prmt.b32
// CHECK: prmt.b32
// CHECK: nvvm.shfl.sync
// CHECK: nvvm.shfl.sync
// CHECK: prmt.b32
// CHECK: prmt.b32
  tt.func @cvt_mma_to_dot_fp8(%a: tensor<128x64xf8E5M2, #mma>) {
    %opA = ttg.convert_layout %a : tensor<128x64xf8E5M2, #mma> -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}
