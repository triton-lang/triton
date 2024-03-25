// RUN: triton-opt %s -split-input-file --decompose-unsupported-conversions --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=90 2>&1 | FileCheck %s

#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>
#shared = #triton_gpu.shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 16, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func @dot_zero_acc_operand(%a: !tt.memdesc<128x128xf8E5M2, #shared>, %b: !tt.memdesc<128x128xf8E5M2, #shared1>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %m = tt.dot %a, %b, %cst {allowTF32 = true, maxNumImpreciseAcc = 64 : i32} :
      !tt.memdesc<128x128xf8E5M2, #shared> * !tt.memdesc<128x128xf8E5M2, #shared1> -> tensor<128x128xf32, #mma>
    tt.return
  }
}
