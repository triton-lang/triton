// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx950 matrix-instruction-size=0" | FileCheck %s --check-prefixes CHECK
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx950 matrix-instruction-size=16" | FileCheck %s --check-prefixes MFMA16

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 2], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[0, 0], [32, 0]], block = []}>
// CHECK{LITERAL}: #linear1 = #ttg.linear<{register = [[0, 2], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[32, 0], [0, 0]], block = []}>
// CHECK-LABEL: mfma_dot_scaled_mxfp4_mxfp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_mxfp4_mxfp4(
      %arg0: tensor<128x64xi8, #blocked>,
      %arg1: tensor<64x128xi8, #blocked1>,
      %arg2: tensor<128x4xi8, #blocked2>,
      %arg3: tensor<128x4xi8, #blocked2>,
      %arg4: tensor<128x128x!tt.ptr<f32>, #blocked1>
      ) {
    // CHECK-NOT: arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK-NOT: arith.constant dense<127> : tensor<128x4xi8, #linear1>
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[C:.+]] = ttg.convert_layout {{.*}} : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #mma>
    // CHECK: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<128x64xi8, #blocked> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<64x128xi8, #blocked1> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : tensor<128x4xi8, #blocked2> -> tensor<128x4xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : tensor<128x4xi8, #blocked2> -> tensor<128x4xi8, #linear1>
    // CHECK: tt.dot_scaled %[[A]] scale %[[SCALE0]], %[[B]] scale %[[SCALE1]], %[[C]] lhs = e2m1 rhs = e2m1
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #blocked>, tensor<128x4xi8, #blocked2> * tensor<64x128xi8, #blocked1>, tensor<128x4xi8, #blocked2> -> tensor<128x128xf32, #blocked1>
    tt.store %arg4, %1 : tensor<128x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_mxfp4_fp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_mxfp4_fp4(
      %arg0: tensor<128x64xi8, #blocked>,
      %arg1: tensor<64x128xi8, #blocked1>,
      %arg2: tensor<128x4xi8, #blocked2>,
      %arg3: tensor<128x128x!tt.ptr<f32>, #blocked1>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[CST1:.+]] = arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : tensor<128x4xi8, #blocked2> -> tensor<128x4xi8, #linear1>
    // CHECK: tt.dot_scaled {{.*}} scale %[[SCALE0]], {{.*}} scale %[[CST1]], {{.*}} lhs = e2m1 rhs = e2m1
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #blocked>, tensor<128x4xi8, #blocked2> * tensor<64x128xi8, #blocked1> -> tensor<128x128xf32, #blocked1>
    tt.store %arg3, %1 : tensor<128x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_fp4_mxfp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_fp4_mxfp4(
      %arg0: tensor<128x64xi8, #blocked>,
      %arg1: tensor<64x128xi8, #blocked1>,
      %arg2: tensor<128x4xi8, #blocked2>,
      %arg3: tensor<128x128x!tt.ptr<f32>, #blocked1>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[CST0:.+]] = arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : tensor<128x4xi8, #blocked2> -> tensor<128x4xi8, #linear1>
    // CHECK: tt.dot_scaled {{.*}} scale %[[CST0]], {{.*}} scale %[[SCALE1]], {{.*}} lhs = e2m1 rhs = e2m1
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %1 = tt.dot_scaled %arg0, %arg1 scale %arg2, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #blocked> * tensor<64x128xi8, #blocked1>, tensor<128x4xi8, #blocked2> -> tensor<128x128xf32, #blocked1>
    tt.store %arg3, %1 : tensor<128x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
// #blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_fp4_fp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_fp4_fp4(
      %arg0: tensor<128x64xi8, #blocked>,
      %arg1: tensor<64x128xi8, #blocked1>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked1>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: tt.dot_scaled {{[^ ]+}}, {{[^ ]+}}, {{[^ ]+}} lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<128x128xf32, #mma>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %1 = tt.dot_scaled %arg0, %arg1, %cst lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<128x64xi8, #blocked> * tensor<64x128xi8, #blocked1> -> tensor<128x128xf32, #blocked1>
    tt.store %arg2, %1 : tensor<128x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 2], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[0, 0], [32, 0]], block = []}>
// CHECK{LITERAL}: #linear1 = #ttg.linear<{register = [[0, 2], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 1]], warp = [[32, 0], [0, 0]], block = []}>
// CHECK-LABEL: mfma_dot_scaled_mxfp8e4_mxfp8e4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_mxfp8e4_mxfp8e4(
      %arg0: tensor<128x128xf8E4M3FN, #blocked>,
      %arg1: tensor<128x128xf8E4M3FN, #blocked>,
      %arg2: tensor<128x4xi8, #blocked1>,
      %arg3: tensor<128x4xi8, #blocked1>,
      %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK-NOT: arith.constant dense<127> : tensor<128x4xi8, #linear1>
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[C:.+]] = ttg.convert_layout {{.*}} : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #mma>
    // CHECK: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : tensor<128x4xi8, #blocked1> -> tensor<128x4xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : tensor<128x4xi8, #blocked1> -> tensor<128x4xi8, #linear1>
    // CHECK: tt.dot_scaled %[[A]] scale %[[SCALE0]], %[[B]] scale %[[SCALE1]], %[[C]] lhs = e4m3 rhs = e4m3
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1 scale %arg3, %cst lhs = e4m3 rhs = e4m3 {fastMath = false} : tensor<128x128xf8E4M3FN, #blocked>, tensor<128x4xi8, #blocked1> * tensor<128x128xf8E4M3FN, #blocked>, tensor<128x4xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.store %arg4, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_fp8e4_mxfp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_fp8e4_mxfp4(
      %arg0: tensor<128x128xf8E4M3FN, #blocked>,
      %arg1: tensor<64x128xi8, #blocked>,
      %arg2: tensor<128x4xi8, #blocked1>,
      %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[CST0:.+]] = arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK: %[[SCALE1:.+]] = ttg.convert_layout {{.*}} : tensor<128x4xi8, #blocked1> -> tensor<128x4xi8, #linear1>
    // CHECK: tt.dot_scaled {{.*}} scale %[[CST0]], {{.*}} scale %[[SCALE1]], {{.*}} lhs = e4m3 rhs = e2m1
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot_scaled %arg0, %arg1 scale %arg2, %cst lhs = e4m3 rhs = e2m1 {fastMath = false} : tensor<128x128xf8E4M3FN, #blocked> * tensor<64x128xi8, #blocked>, tensor<128x4xi8, #blocked1> -> tensor<128x128xf32, #blocked>
    tt.store %arg3, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_mxfp4_fp8e5
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_mxfp4_fp8e5(
      %arg0: tensor<128x64xi8, #blocked>,
      %arg1: tensor<128x128xf8E5M2, #blocked>,
      %arg2: tensor<128x4xi8, #blocked1>,
      %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK: %[[CST1:.+]] = arith.constant dense<127> : tensor<128x4xi8, #linear>
    // CHECK: %[[SCALE0:.+]] = ttg.convert_layout {{.*}} : tensor<128x4xi8, #blocked1> -> tensor<128x4xi8, #linear1>
    // CHECK: tt.dot_scaled {{.*}} scale %[[SCALE0]], {{.*}} scale %[[CST1]], {{.*}} lhs = e2m1 rhs = e5m2
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot_scaled %arg0 scale %arg2, %arg1, %cst lhs = e2m1 rhs = e5m2 {fastMath = false} : tensor<128x64xi8, #blocked>, tensor<128x4xi8, #blocked1> * tensor<128x128xf8E5M2, #blocked> -> tensor<128x128xf32, #blocked>
    tt.store %arg3, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot_op_a = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_op_b = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK-LABEL: mfma_bf8_dot_to_dot_scaled
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_bf8_dot_to_dot_scaled(
      %arg0: tensor<128x64xf8E5M2, #dot_op_a>,
      %arg1: tensor<64x128xf8E5M2, #dot_op_b>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK-NOT: tt.dot {{.*}}, {{.*}}, {{.*}}
    // CHECK-DAG: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK-DAG: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    // CHECK: tt.dot_scaled %[[A]], %[[B]], {{.*}} lhs = e5m2 rhs = e5m2 {fastMath = false} : tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> * tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>> -> tensor<128x128xf32, #mma>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<128x64xf8E5M2, #dot_op_a> * tensor<64x128xf8E5M2, #dot_op_b> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot_op_a = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dot_op_b = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK-LABEL: mfma_fp16_dot_to_dot
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_fp16_dot_to_dot(
      %arg0: tensor<128x64xf16, #dot_op_a>,
      %arg1: tensor<64x128xf16, #dot_op_b>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK-NOT: tt.dot_scaled
    // CHECK-DAG: %[[A:.+]] = ttg.convert_layout {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    // CHECK-DAG: %[[B:.+]] = ttg.convert_layout {{.*}} : tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: tt.dot %[[A]], %[[B]], {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<128x128xf32, #mma>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<128x64xf16, #dot_op_a> * tensor<64x128xf16, #dot_op_b> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %1 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK{LITERAL}: #shared = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 4, order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_mxfp4_b_packed_mn
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_mxfp4_b_packed_mn(
      %a: tensor<128x128xf8E5M2, #blocked>,
      %b: tensor<128x64xi8, #blocked1>,
      %c: tensor<128x128xf32, #blocked>,
      %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    %b1 = ttg.convert_layout %b : tensor<128x64xi8, #blocked1> -> tensor<128x64xi8, #blocked>
    // CHECK: %[[ALLOCB:.+]] = ttg.local_alloc {{.*}} : (tensor<128x64xi8, #blocked>) -> !ttg.memdesc<128x64xi8, #shared, #smem>
    // CHECK: %[[B:.+]] = amdgpu.local_load_packed_tranposed  %[[ALLOCB]] : !ttg.memdesc<128x64xi8, #shared, #smem> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    // CHECK: tt.dot_scaled %{{.*}}, %[[B]], %{{.*}} lhs = e5m2 rhs = e2m1 {fastMath = false}
    %accumulator_52 = tt.dot_scaled %a, %b1, %c lhs = e5m2 rhs = e2m1 {fastMath = false, rhs_k_pack = false} : tensor<128x128xf8E5M2, #blocked> * tensor<128x64xi8, #blocked> -> tensor<128x128xf32, #blocked>
    tt.store %arg4, %accumulator_52 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK{LITERAL}: #shared = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 4, order = [0, 1]}>
// CHECK-LABEL: mfma_dot_scaled_mxfp4_a_packed_mn
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_mxfp4_a_packed_mn(
      %a: tensor<64x128xi8, #blocked>,
      %b: tensor<128x128xf8E5M2, #blocked1>,
      %c: tensor<128x128xf32, #blocked>,
      %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    %b1 = ttg.convert_layout %b : tensor<128x128xf8E5M2, #blocked1> -> tensor<128x128xf8E5M2, #blocked>
    // CHECK: %[[ALLOCA:.+]] = ttg.local_alloc {{.*}} : (tensor<64x128xi8, #blocked>) -> !ttg.memdesc<64x128xi8, #shared, #smem>
    // CHECK: %[[A:.+]] = amdgpu.local_load_packed_tranposed  %[[ALLOCA]] : !ttg.memdesc<64x128xi8, #shared, #smem> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK: tt.dot_scaled %[[A]], %{{.*}}, %{{.*}} lhs = e2m1 rhs = e5m2 {fastMath = false}
    %accumulator_52 = tt.dot_scaled %a, %b1, %c lhs = e2m1 rhs = e5m2 {fastMath = false, lhs_k_pack = false} : tensor<64x128xi8, #blocked> * tensor<128x128xf8E5M2, #blocked> -> tensor<128x128xf32, #blocked>
    tt.store %arg4, %accumulator_52 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK{LITERAL}: #shared = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 4, order = [0, 1]}>
// CHECK{LITERAL}: #shared1 = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 4, order = [1, 0]}>
// CHECK-LABEL: mfma_dot_scaled_mxfp4_ab_packed_mn
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_mxfp4_ab_packed_mn(
      %a: tensor<64x128xi8, #blocked>,
      %b: tensor<128x64xi8, #blocked1>,
      %c: tensor<128x128xf32, #blocked>,
      %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>
      ) {
    %b1 = ttg.convert_layout %b : tensor<128x64xi8, #blocked1> -> tensor<128x64xi8, #blocked>
    // CHECK: %[[ALLOCA:.+]] = ttg.local_alloc {{.*}} : (tensor<64x128xi8, #blocked>) -> !ttg.memdesc<64x128xi8, #shared, #smem>
    // CHECK: %[[A:.+]] = amdgpu.local_load_packed_tranposed  %[[ALLOCA]] : !ttg.memdesc<64x128xi8, #shared, #smem> -> tensor<128x64xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    // CHECK: %[[ALLOCB:.+]] = ttg.local_alloc {{.*}} : (tensor<128x64xi8, #blocked>) -> !ttg.memdesc<128x64xi8, #shared1, #smem>
    // CHECK: %[[B:.+]] = amdgpu.local_load_packed_tranposed  %[[ALLOCB]] : !ttg.memdesc<128x64xi8, #shared1, #smem> -> tensor<64x128xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>>
    // CHECK: tt.dot_scaled %[[A]], %[[B]], %{{.*}} lhs = e2m1 rhs = e2m1 {fastMath = false}
    %accumulator_52 = tt.dot_scaled %a, %b1, %c lhs = e2m1 rhs = e2m1 {fastMath = false, lhs_k_pack = false, rhs_k_pack = false} : tensor<64x128xi8, #blocked> * tensor<128x64xi8, #blocked> -> tensor<128x128xf32, #blocked>
    tt.store %arg4, %accumulator_52 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Checks that for fp8 * fp8 problems with a K < 64, we don't promote to use
// V_MFMA_SCALE_F32_*_F8F6F4 which requires shape 16x16x128 or 32x32x64.

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 64], warpsPerCTA = [4, 2], order = [1, 0]}>
// CHECK{LITERAL}: #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 8], instrShape = [16, 16, 32], isTransposed = true}>
// CHECK-LABEL: mfma_dot_small_k
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 8], instrShape = [16, 16, 32], isTransposed = true}>
// MFMA16-LABEL: mfma_dot_small_k
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_small_k(
      %arg0: tensor<16x32xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<32x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %init: tensor<16x256xf32, #blocked>,
      %arg4: tensor<16x256x!tt.ptr<f32>, #blocked>
      ) {
    // CHECK: tt.dot {{.*}} -> tensor<16x256xf32, #mma>
    // MFMA16: tt.dot {{.*}} -> tensor<16x256xf32, #mma>
    %1 = tt.dot %arg0, %arg1, %init : tensor<16x32xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<32x256xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x256xf32, #blocked>
    tt.store %arg4, %1 : tensor<16x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 2, 2, 1], threadsPerWarp = [1, 1, 4, 16, 1, 1, 1], warpsPerCTA = [4, 1, 1, 1, 1, 1, 1], order = [6, 5, 4, 3, 2, 1, 0]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 2, 1, 1, 2, 1, 1], threadsPerWarp = [1, 1, 16, 1, 1, 4, 1], warpsPerCTA = [4, 1, 1, 1, 1, 1, 1], order = [6, 1, 4, 2, 5, 3, 0]}>
#linear = #ttg.linear<{register = [[16, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]], warp = [[32, 0], [64, 0]], block = []}>

// MFMA16: [[$linear1:#.*]] = #ttg.linear<{register = {{\[\[}}0, 4{{]]}}, lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2{{]]}}, warp = {{\[\[}}0, 0], [0, 0{{]]}}, block = []}>
// MFMA16: [[$linear2:#.*]] = #ttg.linear<{register = {{\[\[}}0, 4], [16, 0{{]]}}, lane = {{\[\[}}1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2{{]]}}, warp = {{\[\[}}32, 0], [64, 0{{]]}}, block = []}>
// MFMA16: [[$mma:#.*]] = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 4], instrShape = [16, 16, 128], isTransposed = true, tilesPerWarp = [1, 2]}>
// MFMA16-LABEL: mfma_dot_scaled_fp8_mxfp4
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_fp8_mxfp4(
      %arg0: tensor<16x256xf8E4M3FN, #blocked6>,
      %arg1: tensor<4x256x!tt.ptr<i8>, #blocked5>,
      %arg2: tensor<128x128xi8, #blocked1>,
      %arg3: tensor<16x128x!tt.ptr<f32>, #blocked1>
      ) {
    // MFMA16: [[SCALE0:%.+]] = ttg.convert_layout {{.*}} : {{.*}} -> tensor<16x8xi8, [[$linear1]]>
    // MFMA16: [[SCALE1:%.+]] = ttg.convert_layout {{.*}} : {{.*}} -> tensor<128x8xi8, [[$linear2]]>
    // MFMA16: tt.dot_scaled {{.*}} scale [[SCALE0]], {{.*}} scale [[SCALE1]], {{.*}} -> tensor<16x128xf32, [[$mma]]>
    %cst0 = arith.constant dense<127> : tensor<16x8xi8, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #blocked1>
    %load = tt.load %arg1 : tensor<4x256x!tt.ptr<i8>, #blocked5>
    %reshape0 = tt.reshape %load : tensor<4x256xi8, #blocked5> -> tensor<4x1x4x16x2x2x1xi8, #blocked7>
    %trans = tt.trans %reshape0 {order = array<i32: 0, 5, 3, 1, 4, 2, 6>} : tensor<4x1x4x16x2x2x1xi8, #blocked7> -> tensor<4x2x16x1x2x4x1xi8, #blocked8>
    %reshape1 = tt.reshape %trans : tensor<4x2x16x1x2x4x1xi8, #blocked8> -> tensor<128x8xi8, #linear>
    %scale = ttg.convert_layout %reshape1 : tensor<128x8xi8, #linear> -> tensor<128x8xi8, #blocked>
    %1 = tt.dot_scaled %arg0 scale %cst0, %arg2 scale %scale, %cst1 lhs = e4m3 rhs = e2m1 {fastMath = true} : tensor<16x256xf8E4M3FN, #blocked6>, tensor<16x8xi8, #blocked> * tensor<128x128xi8, #blocked1>, tensor<128x8xi8, #blocked> -> tensor<16x128xf32, #blocked1>
    tt.store %arg3, %1 : tensor<16x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
