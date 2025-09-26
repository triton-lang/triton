// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx1250 matrix-instruction-size=16" | FileCheck %s --check-prefixes CHECK

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
