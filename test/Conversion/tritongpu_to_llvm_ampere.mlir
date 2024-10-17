// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm=compute-capability=80 2>&1 | FileCheck %s


#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 3072 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @ampere_s8_to_fp16_conversion_opIdx1(%1 : tensor<16x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) attributes {noinline = false} {
    // CHECK-LABEL: ampere_s8_to_fp16_conversion_opIdx1
    // CHECK: llvm.sitofp %{{.*}} : i8 to f16
    %2 = arith.sitofp %1 : tensor<16x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> to tensor<16x32xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    tt.return
  }
}


// -----

#mma = #triton_gpu.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.shared = 3072 : i32, triton_gpu.target = "cuda:80", "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @ampere_s8_to_fp16_conversion_opIdx0(%1 : tensor<32x16xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>) attributes {noinline = false} {
    // CHECK-LABEL: @ampere_s8_to_fp16_conversion_opIdx0
    // CHECK: llvm.sitofp %{{.*}} : i8 to f16
    %2 = arith.sitofp %1 : tensor<32x16xi8, #triton_gpu.dot_op<{opIdx = 0 , parent = #mma, kWidth = 4}>> to tensor<32x16xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    tt.return
  }
}
