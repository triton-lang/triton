// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx950 matrix-instruction-size=0' | FileCheck %s --check-prefixes CHECK

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 32], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 16]], warp = [[0, 0], [32, 0]], block = []}>
// CHECK{LITERAL}: #linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [32, 0], [0, 64]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [16, 0]], warp = [[0, 32], [0, 0]], block = []}>
// CHECK{LITERAL}: #linear2 = #ttg.linear<{register = [[0, 2], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[0, 0], [32, 0]], block = []}>
// CHECK{LITERAL}: #linear3 = #ttg.linear<{register = [[0, 2], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[32, 0], [0, 0]], block = []}>
// CHECK-LABEL: mfma_dot_scaled_mxfp4_mxfp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_mxfp4_mxfp4(
      %arg0: tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x4xi8>,
      %arg3: tensor<128x4xi8>,
      %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK-NOT: arith.constant dense<127> : tensor<128x4xi8, #linear1>
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[C:.+]] = ttg.convert_layout {{.*}} : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #mma>
    // CHECK: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<128x64xi8, #linear>
    // CHECK: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x128xi8, #linear1>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : {{.*}} -> tensor<128x4xi8, #linear2>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : {{.*}} -> tensor<128x4xi8, #linear3>
    // CHECK: tt.dot_scaled %[[A]] scale %[[SCALE0]], %[[B]] scale %[[SCALE1]], %[[C]] lhs = e2m1 rhs = e2m1
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, tensor<128x4xi8> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, tensor<128x4xi8> -> tensor<128x128xf32, #blocked>
    tt.store %arg4, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_mxfp4_fp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_mxfp4_fp4(
      %arg0: tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x4xi8>,
      %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[CST1:.+]] = arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : {{.*}} -> tensor<128x4xi8, #linear3>
    // CHECK: tt.dot_scaled {{.*}} scale %[[SCALE0]], {{.*}} scale %[[CST1]], {{.*}} lhs = e2m1 rhs = e2m1
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, tensor<128x4xi8> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg3, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_fp4_mxfp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_fp4_mxfp4(
      %arg0: tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x4xi8>,
      %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[CST0:.+]] = arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : {{.*}} -> tensor<128x4xi8, #linear3>
    // CHECK: tt.dot_scaled {{.*}} scale %[[CST0]], {{.*}} scale %[[SCALE1]], {{.*}} lhs = e2m1 rhs = e2m1
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot_scaled %arg0, %arg1 scale %arg2, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, tensor<128x4xi8> -> tensor<128x128xf32, #blocked>
    tt.store %arg3, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_fp4_fp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_fp4_fp4(
      %arg0: tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK-DAG: %[[CST0:.+]] = arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK-DAG: %[[CST1:.+]] = arith.constant dense<127> : tensor<128x4xi8, #linear1>
    // CHECK: tt.dot_scaled {{.*}} scale %[[CST1]], {{.*}} scale %[[CST0]], {{.*}} lhs = e2m1 rhs = e2m1
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot_scaled %arg0, %arg1, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 16]], warp = [[0, 0], [32, 0]], block = []}>
// CHECK{LITERAL}: #linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [32, 0], [64, 0], [0, 64]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [16, 0]], warp = [[0, 32], [0, 0]], block = []}>
// CHECK{LITERAL}: #linear2 = #ttg.linear<{register = [[0, 2], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[0, 0], [32, 0]], block = []}>
// CHECK{LITERAL}: #linear3 = #ttg.linear<{register = [[0, 2], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[32, 0], [0, 0]], block = []}>
// CHECK-LABEL: mfma_dot_scaled_mxfp8e4_mxfp8e4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_mxfp8e4_mxfp8e4(
      %arg0: tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x4xi8>,
      %arg3: tensor<128x4xi8>,
      %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK-NOT: arith.constant dense<127> : tensor<128x4xi8, #linear1>
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[C:.+]] = ttg.convert_layout {{.*}} : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #mma>
    // CHECK: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<128x128xf8E4M3FN, #linear>
    // CHECK: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf8E4M3FN, #linear1>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : {{.*}} -> tensor<128x4xi8, #linear2>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : {{.*}} -> tensor<128x4xi8, #linear3>
    // CHECK: tt.dot_scaled %[[A]] scale %[[SCALE0]], %[[B]] scale %[[SCALE1]], %[[C]] lhs = e4m3 rhs = e4m3
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, tensor<128x4xi8> * tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, tensor<128x4xi8> -> tensor<128x128xf32, #blocked>
    tt.store %arg4, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_fp8e4_mxfp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_fp8e4_mxfp4(
      %arg0: tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x4xi8>,
      %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[CST0:.+]] = arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : {{.*}} -> tensor<128x4xi8, #linear3>
    // CHECK: tt.dot_scaled {{.*}} scale %[[CST0]], {{.*}} scale %[[SCALE1]], {{.*}} lhs = e4m3 rhs = e2m1
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot_scaled %arg0, %arg1 scale %arg2, %cst lhs = e4m3 rhs = e2m1 {fastMath = false} : tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>, tensor<128x4xi8> -> tensor<128x128xf32, #blocked>
    tt.store %arg3, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_mxfp4_fp8e5
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_mxfp4_fp8e5(
      %arg0: tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x4xi8>,
      %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[CST1:.+]] = arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : {{.*}} -> tensor<128x4xi8, #linear3>
    // CHECK: tt.dot_scaled {{.*}} scale %[[SCALE0]], {{.*}} scale %[[CST1]], {{.*}} lhs = e2m1 rhs = e5m2
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1, %cst lhs = e2m1 rhs = e5m2 {fastMath = false} : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>, tensor<128x4xi8> * tensor<128x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg3, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
