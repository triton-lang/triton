// RUN: triton-opt %s -split-input-file -tritongpu-swizzle | FileCheck %s


#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #triton_gpu.shared<{vec=1, perPhase=1, maxPhase=1 ,order = [1, 0]}>
#mma = #triton_gpu.mma<{version=2, warpsPerCTA=[1,1]}>

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: swizzle_mma_f16_64x64x64
  func @swizzle_mma_f16_64x64x64(%A: tensor<64x64xf16, #blocked>, %B: tensor<64x64xf16, #blocked>) {
    %AA = triton_gpu.convert_layout %A : (tensor<64x64xf16, #blocked>) -> tensor<64x64xf16, #shared>
    %BB = triton_gpu.convert_layout %B : (tensor<64x64xf16, #blocked>) -> tensor<64x64xf16, #shared>
    %cst0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %D = tt.dot %AA, %BB, %cst0 {allowTF32 = true} : tensor<64x64xf16, #shared> * tensor<64x64xf16, #shared> -> tensor<64x64xf32, #mma>
    return
  }
}
