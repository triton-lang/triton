// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm | FileCheck %s


#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared0 = #triton_gpu.shared<{vec = 1, perPhase=2, maxPhase=8 ,order = [1, 0]}>
#mma0 = #triton_gpu.mma<{version=2, warpsPerCTA=[1,1]}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: convert_dot
  func @convert_dot(%A: tensor<16x16xf16, #blocked0>, %B: tensor<16x16xf16, #blocked0>) {
    %AA = triton_gpu.convert_layout %A : (tensor<16x16xf16, #blocked0>) -> tensor<16x16xf16, #shared0>
    %BB = triton_gpu.convert_layout %B : (tensor<16x16xf16, #blocked0>) -> tensor<16x16xf16, #shared0>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma0>
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ldmatrix.sync.aligned.m8n8.x4
    // CHECK: llvm.inline_asm
    // CHECK-SAME: ldmatrix.sync.aligned.m8n8.x4

    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    // CHECK: llvm.inline_asm
    // CHECK-SAME: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    %D = tt.dot %AA, %BB, %cst0 {allowTF32 = true, transA = false, transB = false} : tensor<16x16xf16, #shared0> * tensor<16x16xf16, #shared0> -> tensor<16x16xf32, #mma0>

    return
  }
}
