// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx950 matrix-instruction-size=0" | FileCheck %s --check-prefixes CHECK

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

// CHECK-LABEL: mfma_dot_scaled_bf16_fp8e4
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_bf16_fp8e4(
      %arg0: tensor<32x64xbf16, #blocked2>,
      %arg1: tensor<64x32xf8E4M3FN, #blocked>,
      %arg2: tensor<32x2xi8, #blocked1>,
      %arg3: tensor<32x32x!tt.ptr<f32>, #blocked>
    ) {
    // CHECK-NOT: tt.fp_to_fp
    // CHECK-NOT: tt.dot_scaled
    // CHECK: %[[A:.*]] = ttg.convert_layout %{{.*}} : tensor<32x64xbf16, #blocked{{.*}}> -> tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    // CHECK: %[[B:.+]] = ttg.convert_layout %{{.*}} : tensor<64x32xf8E4M3FN, #blocked{{.*}}> -> tensor<64x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: %[[S:.+]] = ttg.convert_layout %{{.*}} : tensor<32x2xi8, #blocked{{.*}}> -> tensor<32x2xi8, #blocked{{.*}}>
    // CHECK: %[[UB:.+]] = amdgpu.upcast_mxfp %[[B]], %[[S]] fp_type = e4m3 {fastMath = false} : tensor<64x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<32x2xi8, #blocked{{.*}}> -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: %{{.*}} = tt.dot %[[A]], %[[UB]], %{{.*}} : tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %1 = tt.dot_scaled %arg0, %arg1 scale %arg2, %cst lhs = bf16 rhs = e4m3 {fastMath = false} : tensor<32x64xbf16, #blocked2> * tensor<64x32xf8E4M3FN, #blocked>, tensor<32x2xi8, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.store %arg3, %1 : tensor<32x32x!tt.ptr<f32>, #blocked>
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
