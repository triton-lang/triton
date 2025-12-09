// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx1250" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
// CHECK{LITERAL}: #linear1 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
// CHECK{LITERAL}: #mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape = [16, 16, 128]}>
// CHECK{LITERAL}: #mma1 = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape = [16, 16, 64]}>
// CHECK-LABEL: wmma_dot_scaled_mxfp4_mxfp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_mxfp4_mxfp4(
      %arg0: tensor<32x64xi8, #blocked>,
      %arg1: tensor<64x32xi8, #blocked1>,
      %arg2: tensor<32x4xi8, #blocked2>,
      %arg3: tensor<32x4xi8, #blocked2>,
      %arg4: tensor<32x32x!tt.ptr<f32>, #blocked3>
      ) {
    // CHECK-NOT: arith.constant dense<127> : tensor<32x4xi8, #linear>
    // CHECK-NOT: arith.constant dense<127> : tensor<32x4xi8, #linear1>
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[C:.+]] = ttg.convert_layout {{.*}} : tensor<32x32xf32, #blocked3> -> tensor<32x32xf32, #mma>
    // CHECK: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<32x64xi8, #blocked> -> tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>
    // CHECK: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<64x32xi8, #blocked1> -> tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma1, kWidth = 16}>>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : tensor<32x4xi8, #blocked2> -> tensor<32x4xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : tensor<32x4xi8, #blocked2> -> tensor<32x4xi8, #linear1>
    // CHECK: tt.dot_scaled %[[A]] scale %[[SCALE0]], %[[B]] scale %[[SCALE1]], %[[C]] lhs = e2m1 rhs = e2m1
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked3>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x64xi8, #blocked>, tensor<32x4xi8, #blocked2> * tensor<64x32xi8, #blocked1>, tensor<32x4xi8, #blocked2> -> tensor<32x32xf32, #blocked3>
    tt.store %arg4, %1 : tensor<32x32x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
// CHECK{LITERAL}: #linear1 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
// CHECK{LITERAL}: #mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape = [16, 16, 128]}>
// CHECK{LITERAL}: #mma1 = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape = [16, 16, 64]}>
// CHECK-LABEL: wmma_dot_scaled_mxfp4_mxfp8
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_mxfp4_mxfp8(
      %arg0: tensor<32x64xi8, #blocked>,
      %arg1: tensor<128x32xf8E4M3FN, #blocked1>,
      %arg2: tensor<32x4xi8, #blocked2>,
      %arg3: tensor<32x4xi8, #blocked2>,
      %arg4: tensor<32x32x!tt.ptr<f32>, #blocked3>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[C:.+]] = ttg.convert_layout {{.*}} : tensor<32x32xf32, #blocked3> -> tensor<32x32xf32, #mma>
    // CHECK: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<32x64xi8, #blocked> -> tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma1, kWidth = 16}>>
    // CHECK: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<128x32xf8E4M3FN, #blocked1> -> tensor<128x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : tensor<32x4xi8, #blocked2> -> tensor<32x4xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : tensor<32x4xi8, #blocked2> -> tensor<32x4xi8, #linear1>
    // CHECK: tt.dot_scaled %[[A]] scale %[[SCALE0]], %[[B]] scale %[[SCALE1]], %[[C]] lhs = e2m1 rhs = e4m3
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked3>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e2m1 rhs = e4m3 {fastMath = false} : tensor<32x64xi8, #blocked>, tensor<32x4xi8, #blocked2> * tensor<128x32xf8E4M3FN, #blocked1>, tensor<32x4xi8, #blocked2> -> tensor<32x32xf32, #blocked3>
    tt.store %arg4, %1 : tensor<32x32x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
// CHECK{LITERAL}: #linear1 = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
// CHECK{LITERAL}: #mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape = [16, 16, 128]}>
// CHECK-LABEL: wmma_dot_scaled_mxfp8
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_mxfp8(
      %arg0: tensor<32x128xf8E4M3FN, #blocked>,
      %arg1: tensor<128x32xf8E4M3FN, #blocked1>,
      %arg2: tensor<32x4xi8, #blocked2>,
      %arg3: tensor<32x4xi8, #blocked2>,
      %arg4: tensor<32x32x!tt.ptr<f32>, #blocked3>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[C:.+]] = ttg.convert_layout {{.*}} : tensor<32x32xf32, #blocked3> -> tensor<32x32xf32, #mma>
    // CHECK: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<32x128xf8E4M3FN, #blocked> -> tensor<32x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<128x32xf8E4M3FN, #blocked1> -> tensor<128x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : tensor<32x4xi8, #blocked2> -> tensor<32x4xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : tensor<32x4xi8, #blocked2> -> tensor<32x4xi8, #linear1>
    // CHECK: tt.dot_scaled %[[A]] scale %[[SCALE0]], %[[B]] scale %[[SCALE1]], %[[C]] lhs = e4m3 rhs = e4m3
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked3>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x128xf8E4M3FN, #blocked>, tensor<32x4xi8, #blocked2> * tensor<128x32xf8E4M3FN, #blocked1>, tensor<32x4xi8, #blocked2> -> tensor<32x32xf32, #blocked3>
    tt.store %arg4, %1 : tensor<32x32x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
// CHECK{LITERAL}: #linear1 = #ttg.linear<{register = [[0, 1], [0, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
// CHECK{LITERAL}: #mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape = [16, 16, 128]}>
// CHECK-LABEL: wmma_dot_scaled_mxfp8_k64
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_mxfp8_k64(
      %arg0: tensor<32x64xf8E4M3FN, #blocked>,
      %arg1: tensor<64x32xf8E4M3FN, #blocked1>,
      %arg2: tensor<32x2xi8, #blocked2>,
      %arg3: tensor<32x2xi8, #blocked2>,
      %arg4: tensor<32x32x!tt.ptr<f32>, #blocked3>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[C:.+]] = ttg.convert_layout {{.*}} : tensor<32x32xf32, #blocked3> -> tensor<32x32xf32, #mma>
    // CHECK: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<32x64xf8E4M3FN, #blocked> -> tensor<32x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<64x32xf8E4M3FN, #blocked1> -> tensor<64x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : tensor<32x2xi8, #blocked2> -> tensor<32x2xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : tensor<32x2xi8, #blocked2> -> tensor<32x2xi8, #linear1>
    // CHECK: tt.dot_scaled %[[A]] scale %[[SCALE0]], %[[B]] scale %[[SCALE1]], %[[C]] lhs = e4m3 rhs = e4m3
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked3>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x64xf8E4M3FN, #blocked>, tensor<32x2xi8, #blocked2> * tensor<64x32xf8E4M3FN, #blocked1>, tensor<32x2xi8, #blocked2> -> tensor<32x32xf32, #blocked3>
    tt.store %arg4, %1 : tensor<32x32x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
// CHECK{LITERAL}: #linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
// CHECK{LITERAL}: #mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape = [16, 16, 128]}>
// CHECK-LABEL: wmma_dot_scaled_mxfp8_repeat_k
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_mxfp8_repeat_k(
      %arg0: tensor<32x256xf8E4M3FN, #blocked>,
      %arg1: tensor<256x32xf8E4M3FN, #blocked1>,
      %arg2: tensor<32x8xi8, #blocked2>,
      %arg3: tensor<32x8xi8, #blocked2>,
      %arg4: tensor<32x32x!tt.ptr<f32>, #blocked3>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[C:.+]] = ttg.convert_layout {{.*}} : tensor<32x32xf32, #blocked3> -> tensor<32x32xf32, #mma>
    // CHECK: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<32x256xf8E4M3FN, #blocked> -> tensor<32x256xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<256x32xf8E4M3FN, #blocked1> -> tensor<256x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : tensor<32x8xi8, #blocked2> -> tensor<32x8xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : tensor<32x8xi8, #blocked2> -> tensor<32x8xi8, #linear1>
    // CHECK: tt.dot_scaled %[[A]] scale %[[SCALE0]], %[[B]] scale %[[SCALE1]], %[[C]] lhs = e4m3 rhs = e4m3
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked3>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<32x256xf8E4M3FN, #blocked>, tensor<32x8xi8, #blocked2> * tensor<256x32xf8E4M3FN, #blocked1>, tensor<32x8xi8, #blocked2> -> tensor<32x32xf32, #blocked3>
    tt.store %arg4, %1 : tensor<32x32x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[0, 0], [16, 0]], block = []}>
// CHECK{LITERAL}: #linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 0]], warp = [[16, 0], [0, 0]], block = []}>
// CHECK{LITERAL}: #mma = #ttg.amd_wmma<{version = 3, isTranspose = true, warpsPerCTA = [2, 2], instrShape = [16, 16, 128]}>
// CHECK-LABEL: wmma_dot_scaled_mxfp8_repeat_mn
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_mxfp8_repeat_mn(
      %arg0: tensor<64x128xf8E4M3FN, #blocked>,
      %arg1: tensor<128x64xf8E4M3FN, #blocked1>,
      %arg2: tensor<64x4xi8, #blocked2>,
      %arg3: tensor<64x4xi8, #blocked2>,
      %arg4: tensor<64x64x!tt.ptr<f32>, #blocked3>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[C:.+]] = ttg.convert_layout {{.*}} : tensor<64x64xf32, #blocked3> -> tensor<64x64xf32, #mma>
    // CHECK: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<128x64xf8E4M3FN, #blocked1> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : tensor<64x4xi8, #blocked2> -> tensor<64x4xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : tensor<64x4xi8, #blocked2> -> tensor<64x4xi8, #linear1>
    // CHECK: tt.dot_scaled %[[A]] scale %[[SCALE0]], %[[B]] scale %[[SCALE1]], %[[C]] lhs = e4m3 rhs = e4m3
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked3>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<64x128xf8E4M3FN, #blocked>, tensor<64x4xi8, #blocked2> * tensor<128x64xf8E4M3FN, #blocked1>, tensor<64x4xi8, #blocked2> -> tensor<64x64xf32, #blocked3>
    tt.store %arg4, %1 : tensor<64x64x!tt.ptr<f32>, #blocked3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[0, 32], [0, 64], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
// CHECK-LABEL: wmma_dot_scaled_mxfp8_bf16
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_mxfp8_bf16(
      %arg0: tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked4>,
      %arg1: tensor<32x4x!tt.ptr<i8>, #blocked2>,
      %arg2: tensor<128x32x!tt.ptr<bf16>, #blocked>,
      %output: tensor<32x32x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK: tt.load %arg1 {amdg.decomposed_dot_scaled_source = true} : tensor<32x4x!tt.ptr<i8>, #blocked1>
    // CHECK: %[[SCALE:.*]] = tt.reshape {{.*}} : tensor<32x4x32xi8, #blocked3> -> tensor<32x128xi8, #linear>
    // CHECK: %[[CVT0:.*]]  = ttg.convert_layout %[[SCALE]] : tensor<32x128xi8, #linear> -> tensor<32x128xi8, #blocked>
    // CHECK: %[[UPCASTED:.*]] = amdg.scaled_upcast_fp8 {{.*}} scale %[[CVT0]] : tensor<32x128xf8E4M3FN, #blocked>, tensor<32x128xi8, #blocked> -> tensor<32x128xbf16, #blocked>
    // CHECK: %[[SEL:.*]] = arith.select {{.*}}, {{.*}}, %[[UPCASTED]]
    // CHECK: %[[CVT1:.*]] = ttg.convert_layout %[[SEL]] : tensor<32x128xbf16, #blocked> -> tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    // CHECK: %[[OPND0:.*]] = ttg.convert_layout %[[CVT1]] : tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    // CHECK: tt.dot %[[OPND0]]
    %a = tt.load %arg0 : tensor<32x128x!tt.ptr<f8E4M3FN>, #blocked4>
    %scale = tt.load %arg1 : tensor<32x4x!tt.ptr<i8>, #blocked2>
    %b = tt.load %arg2 : tensor<128x32x!tt.ptr<bf16>, #blocked>
    %c = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %res = tt.dot_scaled %a scale %scale, %b, %c lhs = e4m3 rhs = bf16 {fastMath = false} : tensor<32x128xf8E4M3FN, #blocked4>, tensor<32x4xi8, #blocked2> * tensor<128x32xbf16, #blocked> -> tensor<32x32xf32, #blocked>

    tt.store %output, %res : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[32, 0], [64, 0]], block = []}>
// CHECK-LABEL: wmma_dot_scaled_f16_mxfp8
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_f16_mxfp8(
      %arg0: tensor<32x128x!tt.ptr<f16>, #blocked4>,
      %arg1: tensor<32x4x!tt.ptr<i8>, #blocked2>,
      %arg2: tensor<128x32x!tt.ptr<f8E5M2>, #blocked>,
      %output: tensor<32x32x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK: %[[TRANS:.*]] = tt.trans {{.*}} {order = array<i32: 0, 2, 1>} : tensor<4x32x32xi8, #blocked4> -> tensor<4x32x32xi8, #blocked5>
    // CHECK: %[[SCALE:.*]] = tt.reshape %[[TRANS]] : tensor<4x32x32xi8, #blocked5> -> tensor<128x32xi8, #linear>
    // CHECK: %[[CVT0:.*]] = ttg.convert_layout %[[SCALE]] : tensor<128x32xi8, #linear> -> tensor<128x32xi8, #blocked2>
    // CHECK: %[[UPCASTED:.*]] = amdg.scaled_upcast_fp8 {{.*}} scale %[[CVT0]] : tensor<128x32xf8E5M2, #blocked2>, tensor<128x32xi8, #blocked2> -> tensor<128x32xf16, #blocked2>
    // CHECK: %[[SEL:.*]] = arith.select {{.*}}, %cst, %[[UPCASTED]] : tensor<128x32xi1, #blocked2>, tensor<128x32xf16, #blocked2>
    // CHECK: %[[CVT1:.*]] = ttg.convert_layout %[[SEL]] : tensor<128x32xf16, #blocked2> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    // CHECK: %[[OPND1:.*]] = ttg.convert_layout %[[CVT1]] : tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: = tt.dot {{.*}}, %[[OPND1]]
    %a = tt.load %arg0 : tensor<32x128x!tt.ptr<f16>, #blocked4>
    %scale = tt.load %arg1 : tensor<32x4x!tt.ptr<i8>, #blocked2>
    %b = tt.load %arg2 : tensor<128x32x!tt.ptr<f8E5M2>, #blocked>
    %c = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %res = tt.dot_scaled %a, %b scale %scale, %c lhs = fp16 rhs = e5m2 {fastMath = false} : tensor<32x128xf16, #blocked4> * tensor<128x32xf8E5M2, #blocked>,  tensor<32x4xi8, #blocked2> -> tensor<32x32xf32, #blocked>

    tt.store %output, %res : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], lane = [[0, 32], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[0, 0], [0, 0]], block = []}>
// CHECK-LABEL: wmma_dot_scaled_mxfp4_bf16
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_mxfp4_bf16(
      %arg0: tensor<16x32x!tt.ptr<i8>, #blocked5>,
      %arg1: tensor<16x2x!tt.ptr<i8>, #blocked2>,
      %arg2: tensor<64x16x!tt.ptr<bf16>, #blocked>,
      %output: tensor<16x16x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK: tt.load %arg1 {amdg.decomposed_dot_scaled_source = true} : tensor<16x2x!tt.ptr<i8>, #blocked1>
    // CHECK: %[[SCALE:.*]] = tt.reshape {{.*}} : tensor<16x2x32xi8, #blocked3> -> tensor<16x64xi8, #linear>
    // CHECK: %[[CVT0:.*]] = ttg.convert_layout %[[SCALE]] : tensor<16x64xi8, #linear> -> tensor<16x64xi8, #blocked>
    // CHECK: %[[UPCASTED:.*]] = amdg.scaled_upcast_fp4 {{.*}} scale %[[CVT0]] {axis = 1 : i32} : tensor<16x32xi8, #blocked>, tensor<16x64xi8, #blocked> -> tensor<16x64xbf16, #blocked>
    // CHECK: %[[SEL:.*]] = arith.select {{.*}}, %{{.*}}, %[[UPCASTED]] : tensor<16x64xi1, #blocked>, tensor<16x64xbf16, #blocked>
    // CHECK: %[[CVT1:.*]] = ttg.convert_layout %[[SEL]] : tensor<16x64xbf16, #blocked> -> tensor<16x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    // CHECK: %[[OPND0:.*]] = ttg.convert_layout %[[CVT1]] : tensor<16x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> -> tensor<16x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    // CHECK: tt.dot %[[OPND0]]
    %a = tt.load %arg0 : tensor<16x32x!tt.ptr<i8>, #blocked5>
    %scale = tt.load %arg1 : tensor<16x2x!tt.ptr<i8>, #blocked2>
    %b = tt.load %arg2 : tensor<64x16x!tt.ptr<bf16>, #blocked>
    %c = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %res = tt.dot_scaled %a scale %scale, %b, %c lhs = e2m1 rhs = bf16 {fastMath = false} : tensor<16x32xi8, #blocked5>, tensor<16x2xi8, #blocked2> * tensor<64x16xbf16, #blocked> -> tensor<16x16xf32, #blocked>

    tt.store %output, %res : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [32, 0]], warp = [[0, 0], [0, 0]], block = []}>
// CHECK-LABEL: wmma_dot_scaled_fp16_mxfp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_scaled_fp16_mxfp4(
      %arg0: tensor<16x64x!tt.ptr<f16>, #blocked5>,
      %arg1: tensor<16x2x!tt.ptr<i8>, #blocked2>,
      %arg2: tensor<32x16x!tt.ptr<i8>, #blocked>,
      %output: tensor<16x16x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK: tt.load %arg1 {amdg.decomposed_dot_scaled_source = true} : tensor<16x2x!tt.ptr<i8>, #blocked1>
    // CHECK: %[[SCALE:.*]] = tt.reshape {{.*}} : tensor<2x32x16xi8, #blocked5> -> tensor<64x16xi8, #linear>
    // CHECK: %[[CVT0:.*]] = ttg.convert_layout %[[SCALE]] : tensor<64x16xi8, #linear> -> tensor<64x16xi8, #blocked2>
    // CHECK: %[[UPCASTED:.*]] = amdg.scaled_upcast_fp4 {{.*}} scale %[[CVT0]] {axis = 0 : i32} : tensor<32x16xi8, #blocked2>, tensor<64x16xi8, #blocked2> -> tensor<64x16xf16, #blocked2>
    // CHECK: %[[SEL:.*]] = arith.select {{.*}}, %cst, %[[UPCASTED]] : tensor<64x16xi1, #blocked2>, tensor<64x16xf16, #blocked2>
    // CHECK: %[[CVT1:.*]] = ttg.convert_layout %[[SEL]] : tensor<64x16xf16, #blocked2> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    // CHECK: %[[OPND1:.*]] = ttg.convert_layout %[[CVT1]] : tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: tt.dot {{.*}}, %[[OPND1]]
    %a = tt.load %arg0 : tensor<16x64x!tt.ptr<f16>, #blocked5>
    %scale = tt.load %arg1 : tensor<16x2x!tt.ptr<i8>, #blocked2>
    %b = tt.load %arg2 : tensor<32x16x!tt.ptr<i8>, #blocked>
    %c = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %res = tt.dot_scaled %a, %b scale %scale, %c lhs = fp16 rhs = e2m1 {fastMath = false} : tensor<16x64xf16, #blocked5> * tensor<32x16xi8, #blocked>, tensor<16x2xi8, #blocked2> -> tensor<16x16xf32, #blocked>

    tt.store %output, %res : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
