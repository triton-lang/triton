// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=120 -cse | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [16, 8]}>
module attributes {"ttg.num-warps" = 8 : i32, ttg.target = "cuda:120"} {
  // CHECK-LABEL: sm120_mmav2_e5m2_e5m2_fp16
  tt.func public @sm120_mmav2_e5m2_e5m2_fp16(%arg0: tensor<32x32xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, %arg1: tensor<32x32xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, %arg2: tensor<32x32xf16, #mma>) {
    // CHECK: mma.{{.*}}.col.f16.e5m2.e5m2.f16
    %0 = tt.dot %arg0, %arg1, %arg2 {maxNumImpreciseAcc = 1073741824 : i32} : tensor<32x32xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x32xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf16, #mma>
    tt.return
  }

  // CHECK-LABEL: sm120_mmav2_e5m2_e4m3_fp16
  tt.func public @sm120_mmav2_e5m2_e4m3_fp16(%arg0: tensor<32x32xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, %arg1: tensor<32x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, %arg2: tensor<32x32xf16, #mma>) {
    // CHECK: mma.{{.*}}.col.f16.e5m2.e4m3.f16
    %0 = tt.dot %arg0, %arg1, %arg2 {maxNumImpreciseAcc = 1073741824 : i32} : tensor<32x32xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf16, #mma>
    tt.return
  }

  // CHECK-LABEL: sm120_mmav2_e4m3_e5m2_fp16
  tt.func public @sm120_mmav2_e4m3_e5m2_fp16(%arg0: tensor<32x32xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, %arg1: tensor<32x32xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, %arg2: tensor<32x32xf16, #mma>) {
    // CHECK: mma.{{.*}}.col.f16.e4m3.e5m2.f16
    %0 = tt.dot %arg0, %arg1, %arg2 {maxNumImpreciseAcc = 1073741824 : i32} : tensor<32x32xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x32xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf16, #mma>
    tt.return
  }

  // CHECK-LABEL: sm120_mmav2_e4m3_e4m3_fp16
  tt.func public @sm120_mmav2_e4m3_e4m3_fp16(%arg0: tensor<32x32xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, %arg1: tensor<32x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, %arg2: tensor<32x32xf16, #mma>) {
    // CHECK: mma.{{.*}}.col.f16.e4m3.e4m3.f16
    %0 = tt.dot %arg0, %arg1, %arg2 {maxNumImpreciseAcc = 1073741824 : i32} : tensor<32x32xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<32x32xf16, #mma>
    tt.return
  }
}
