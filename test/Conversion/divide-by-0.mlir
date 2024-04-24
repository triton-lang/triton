// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm --cse | FileCheck %s

// CHECK-LABEL: dont_divide_0
// CHECK: %[[C0:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NOT: llvm.urem %{{.*}}, %[[C0]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 8]}>
module attributes {"triton_gpu.target" = "cuda:80", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @dont_divide_0() attributes {noinline = false} {
    %zero = arith.constant dense<0.000000e+00> : tensor<16x1xf32, #mma>
    %cvt = triton_gpu.convert_layout %zero : tensor<16x1xf32, #mma> -> tensor<16x1xf32, #blocked>
    tt.return
  }
}
