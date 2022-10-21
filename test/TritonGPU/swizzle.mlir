// RUN: triton-opt %s -split-input-file -tritongpu-swizzle | FileCheck %s

#shared = #triton_gpu.shared<{vec=1, perPhase=1, maxPhase=1 ,order = [1, 0]}>
#mma1w = #triton_gpu.mma<{version=2, warpsPerCTA=[1, 1]}>
#mma2w = #triton_gpu.mma<{version=2, warpsPerCTA=[1, 2]}>
#mma4w = #triton_gpu.mma<{version=2, warpsPerCTA=[2, 2]}>
#mma8w = #triton_gpu.mma<{version=2, warpsPerCTA=[2, 4]}>

// CHECK: [[shared_v8p1m8:#.*]] = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
// CHECK: [[shared_v8p2m4:#.*]] = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
// CHECK: [[shared_v8p4m2:#.*]] = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>

#shared2 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#shared3 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0]}>


module attributes {"triton_gpu.num-warps" = 8 : i32} {
  // CHECK-LABEL: swizzle_mma_f16_128x256x64_w8
  func @swizzle_mma_f16_128x256x64_w8(%A: tensor<128x64xf16, #shared>, %B: tensor<64x256xf16, #shared>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma8w>
    // CHECK: {{.*}} = triton_gpu.convert_layout {{.*}} : (tensor<128x64xf16, {{.*}}>) -> tensor<128x64xf16, [[shared_v8p1m8]]>
    // CHECK: {{.*}} = triton_gpu.convert_layout {{.*}} : (tensor<64x256xf16, {{.*}}>) -> tensor<64x256xf16, [[shared_v8p1m8]]>
    %D = tt.dot %A, %B, %cst0 {allowTF32 = true, transA = false, transB = false} : tensor<128x64xf16, #shared> * tensor<64x256xf16, #shared> -> tensor<128x256xf32, #mma8w>
    return
  }
}


module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: swizzle_mma_f16_128x128x64_w4
  func @swizzle_mma_f16_128x128x64_w4(%A: tensor<128x64xf16, #shared>, %B: tensor<64x128xf16, #shared>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma4w>
    // CHECK: {{.*}} = triton_gpu.convert_layout {{.*}} : (tensor<128x64xf16, {{.*}}>) -> tensor<128x64xf16, [[shared_v8p1m8]]>
    // CHECK: {{.*}} = triton_gpu.convert_layout {{.*}} : (tensor<64x128xf16, {{.*}}>) -> tensor<64x128xf16, [[shared_v8p1m8]]>
    %D = tt.dot %A, %B, %cst0 {allowTF32 = true, transA = false, transB = false} : tensor<128x64xf16, #shared> * tensor<64x128xf16, #shared> -> tensor<128x128xf32, #mma4w>
    return
  }
}

module attributes {"triton_gpu.num-warps" = 4 : i32} {
  // CHECK-LABEL: swizzle_mma_f16_128x128x32_w4
  func @swizzle_mma_f16_128x128x32_w4(%A: tensor<128x32xf16, #shared>, %B: tensor<32x128xf16, #shared>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma4w>
    // CHECK: {{.*}} = triton_gpu.convert_layout {{.*}} : (tensor<128x32xf16, {{.*}}>) -> tensor<128x32xf16, [[shared_v8p2m4]]>
    // CHECK: {{.*}} = triton_gpu.convert_layout {{.*}} : (tensor<32x128xf16, {{.*}}>) -> tensor<32x128xf16, [[shared_v8p1m8]]>
    %D = tt.dot %A, %B, %cst0 {allowTF32 = true, transA = false, transB = false} : tensor<128x32xf16, #shared> * tensor<32x128xf16, #shared> -> tensor<128x128xf32, #mma4w>
    return
  }
}

module attributes {"triton_gpu.num-warps" = 2 : i32} {
  // CHECK-LABEL: swizzle_mma_f16_32x32x32_w2
  func @swizzle_mma_f16_32x32x32_w2(%A: tensor<32x32xf16, #shared>, %B: tensor<32x32xf16, #shared>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma2w>
    // CHECK: {{.*}} = triton_gpu.convert_layout {{.*}} : (tensor<32x32xf16, {{.*}}>) -> tensor<32x32xf16, [[shared_v8p2m4]]>
    // CHECK: {{.*}} = triton_gpu.convert_layout {{.*}} : (tensor<32x32xf16, {{.*}}>) -> tensor<32x32xf16, [[shared_v8p2m4]]>
    %D = tt.dot %A, %B, %cst0 {allowTF32 = true, transA = false, transB = false} : tensor<32x32xf16, #shared> * tensor<32x32xf16, #shared> -> tensor<32x32xf32, #mma2w>
    return
  }
}

module attributes {"triton_gpu.num-warps" = 1 : i32} {
  // CHECK-LABEL: swizzle_mma_f16_16x16x16_w1
  func @swizzle_mma_f16_16x16x16_w1(%A: tensor<16x16xf16, #shared>, %B: tensor<16x16xf16, #shared>) {
    %cst0 = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma1w>
    // CHECK: {{.*}} = triton_gpu.convert_layout {{.*}} : (tensor<16x16xf16, {{.*}}>) -> tensor<16x16xf16, [[shared_v8p4m2]]>
    // CHECK: {{.*}} = triton_gpu.convert_layout {{.*}} : (tensor<16x16xf16, {{.*}}>) -> tensor<16x16xf16, [[shared_v8p4m2]]>
    %D = tt.dot %A, %B, %cst0 {allowTF32 = true, transA = false, transB = false} : tensor<16x16xf16, #shared> * tensor<16x16xf16, #shared> -> tensor<16x16xf32, #mma1w>
    return
  }
}
