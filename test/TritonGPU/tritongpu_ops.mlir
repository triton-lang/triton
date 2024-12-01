// RUN: triton-opt %s | triton-opt | FileCheck %s

#shared0 = #ttg.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], hasLeadingOffset=true}>

module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: memdesc
  // CHECK-SAME: !ttg.memdesc<1x64x16xf16, #{{.+}}>
  tt.func @memdesc(%d : !ttg.memdesc<1x64x16xf16, #shared0>) {
    tt.return
  }
}
