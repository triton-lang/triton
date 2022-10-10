// RUN: triton-opt %s -split-input-file -tritongpu-swizzle | FileCheck %s

#shared = #triton_gpu.shared<{vec=1, perPhase=1, maxPhase=1 ,order = [1, 0]}>
#mma1w = #triton_gpu.mma<{version=2, warpsPerCTA=[1, 1]}>
#mma2w = #triton_gpu.mma<{version=2, warpsPerCTA=[1, 2]}>
#mma4w = #triton_gpu.mma<{version=2, warpsPerCTA=[2, 2]}>
#mma8w = #triton_gpu.mma<{version=2, warpsPerCTA=[2, 4]}>

// A w8 128x64 [vec, per, max] 8 1 8
// B w8 64x256 [vec, per, max] 8 1 8
module attributes {"triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: swizzle_mma_f16_128x256x64_w8
  func @swizzle_mma_f16_128x256x64_w8(%A: tensor<128x64xf16, #shared>, %B: tensor<64x256xf16, #shared>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma8w>
    %D = tt.dot %A, %B, %cst0 {allowTF32 = true} : tensor<128x64xf16, #shared> * tensor<64x256xf16, #shared> -> tensor<128x256xf32, #mma8w>
    return
  }
}


// A w4 128x64 [vec, per, max] 8 1 8
// B w4 64x128 [vec, per, max] 8 1 8
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: swizzle_mma_f16_128x128x64_w4
  func @swizzle_mma_f16_128x128x64_w4(%A: tensor<128x64xf16, #shared>, %B: tensor<64x128xf16, #shared>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma4w>
    %D = tt.dot %A, %B, %cst0 {allowTF32 = true} : tensor<128x64xf16, #shared> * tensor<64x128xf16, #shared> -> tensor<128x128xf32, #mma4w>
    return
  }
}

// A w4 128x32 [vec, per, max] 8 2 4
// B w4 32x128 [vec, per, max] 8 1 8
module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: swizzle_mma_f16_128x128x32_w4
  func @swizzle_mma_f16_128x128x32_w4(%A: tensor<128x32xf16, #shared>, %B: tensor<32x128xf16, #shared>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma4w>
    %D = tt.dot %A, %B, %cst0 {allowTF32 = true} : tensor<128x32xf16, #shared> * tensor<32x128xf16, #shared> -> tensor<128x128xf32, #mma4w>
    return
  }
}

// A w2 32x32 [vec, per, max] 8 2 4
// B w2 32x32 [vec, per, max] 8 2 4
module attributes {"triton_gpu.num-warps" = 2 : i32} {
  // CHECK-LABEL: swizzle_mma_f16_32x32x32_w2
  func @swizzle_mma_f16_32x32x32_w2(%A: tensor<32x32xf16, #shared>, %B: tensor<32x32xf16, #shared>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma2w>
    %D = tt.dot %A, %B, %cst0 {allowTF32 = true} : tensor<32x32xf16, #shared> * tensor<32x32xf16, #shared> -> tensor<32x32xf32, #mma2w>
    return
  }
}

// A w1 16x16 [vec, per, max] 8 4 2
// B w1 16x16 [vec, per, max] 8 4 2
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: swizzle_mma_f16_16x16x32_w1
  func @swizzle_mma_f16_16x16x32_w1(%A: tensor<16x16xf16, #shared>, %B: tensor<16x16xf16, #shared>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma1w>
    %D = tt.dot %A, %B, %cst0 {allowTF32 = true} : tensor<16x16xf16, #shared> * tensor<16x16xf16, #shared> -> tensor<16x16xf32, #mma1w>
    return
  }
}
