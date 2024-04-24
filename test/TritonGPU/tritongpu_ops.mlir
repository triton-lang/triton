// RUN: triton-opt %s | triton-opt | FileCheck %s

#shared0 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], hasLeadingOffset=true}>

module attributes {"triton_gpu.target" = "cuda:0", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: memdesc
  // CHECK-SAME: !tt.memdesc<1x64x16xf16, #{{.+}}>
  tt.func @memdesc(%d : !tt.memdesc<1x64x16xf16, #shared0>) {
    tt.return
  }
}
