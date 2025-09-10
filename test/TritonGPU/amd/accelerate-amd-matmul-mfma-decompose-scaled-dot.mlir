// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx950 matrix-instruction-size=0" -tritongpu-remove-layout-conversions | FileCheck %s --check-prefixes CHECK

// CHECK-LABEL: mfma_dot_scaled_bf16_fp8e4
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_bf16_fp8e4(
      %arg0: tensor<32x64x!tt.ptr<bf16>, #blocked2>,
      %arg1: tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>,
      %arg2: tensor<32x2x!tt.ptr<i8>, #blocked1>,
      %arg3: tensor<32x32x!tt.ptr<f32>, #blocked>
    ) {
    // CHECK: %[[CST:.*]] = arith.constant dense<7> : tensor<2x32xi16, #ttg.slice<{dim = 2, parent = #linear{{.*}}}>>
    // CHECK: %[[B:.*]] = ttg.convert_layout %{{.*}} : tensor<64x32xf8E4M3FN, #blocked{{.*}}> -> tensor<64x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: %[[S:.*]] = ttg.convert_layout %{{.*}} : tensor<32x2xi8, #blocked{{.*}}> -> tensor<32x2xi8, #linear{{.*}}>
    // CHECK: %[[TS:.*]] = tt.trans %[[S]] {order = array<i32: 1, 0>}
    // CHECK: %[[ES:.*]] = arith.extui %[[TS]]
    // CHECK: %[[SHS:.*]] = arith.shli %[[ES]], %[[CST]]
    // CHECK: %[[BS:.*]] = tt.bitcast %[[SHS]] : tensor<2x32xi16, #ttg.slice<{dim = 2, parent = #linear{{.*}}}>> -> tensor<2x32xbf16, #ttg.slice<{dim = 2, parent = #linear{{.*}}}>>
    // CHECK: %[[EPS:.*]] = tt.expand_dims %[[BS]] {axis = 2 : i32} : tensor<2x32xbf16, #ttg.slice<{dim = 2, parent = #linear{{.*}}}>> -> tensor<2x32x1xbf16, #linear{{.*}}>
    // CHECK: %[[BCS:.*]] = tt.broadcast %[[EPS]] : tensor<2x32x1xbf16, #linear{{.*}}> -> tensor<2x32x32xbf16, #linear{{.*}}>
    // CHECK: %[[TBCS:.*]] = tt.trans %[[BCS]] {order = array<i32: 0, 2, 1>} : tensor<2x32x32xbf16, #linear{{.*}}> -> tensor<2x32x32xbf16, #linear{{.*}}>
    // CHECK: %[[RTBCS:.*]] = tt.reshape %[[TBCS]] : tensor<2x32x32xbf16, #linear{{.*}}> -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: %[[UB:.*]] = amdgpu.scaled_upcast_fp8 %[[B]] scale %[[RTBCS]] : tensor<64x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: %[[SELECTEDB:.*]] = arith.select %{{.*}}, %{{.*}}, %[[UB]] : tensor<64x32xi1, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: %[[A:.*]] = ttg.convert_layout %{{.*}} : tensor<32x64xbf16, #blocked{{.*}}> -> tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    // CHECK: %{{.*}} = tt.dot %[[A]], %[[SELECTEDB]], %{{.*}} : tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %1 = tt.load %arg0 : tensor<32x64x!tt.ptr<bf16>, #blocked2>
    %2 = tt.load %arg1 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>
    %3 = tt.load %arg2 : tensor<32x2x!tt.ptr<i8>, #blocked1>
    %4 = tt.dot_scaled %1, %2 scale %3, %cst lhs = bf16 rhs = e4m3 {fastMath = false} : tensor<32x64xbf16, #blocked2> * tensor<64x32xf8E4M3FN, #blocked>, tensor<32x2xi8, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.store %arg3, %4 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: mfma_dot_scaled_bf16_fp8e4_fast_math
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_bf16_fp8e4_fast_math(
      %arg0: tensor<32x64x!tt.ptr<bf16>, #blocked2>,
      %arg1: tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>,
      %arg2: tensor<32x2x!tt.ptr<i8>, #blocked1>,
      %arg3: tensor<32x32x!tt.ptr<f32>, #blocked>
    ) {
    // CHECK: %[[UB:.*]] = amdgpu.scaled_upcast_fp8 %{{.*}} scale %{{.*}} : tensor<64x32xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>, tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: %{{.*}} = tt.dot %{{.*}}, %[[UB]], %{{.*}} : tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %1 = tt.load %arg0 : tensor<32x64x!tt.ptr<bf16>, #blocked2>
    %2 = tt.load %arg1 : tensor<64x32x!tt.ptr<f8E4M3FN>, #blocked>
    %3 = tt.load %arg2 : tensor<32x2x!tt.ptr<i8>, #blocked1>
    %4 = tt.dot_scaled %1, %2 scale %3, %cst lhs = bf16 rhs = e4m3 {fastMath = true} : tensor<32x64xbf16, #blocked2> * tensor<64x32xf8E4M3FN, #blocked>, tensor<32x2xi8, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.store %arg3, %4 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: mfma_dot_scaled_bf16_fp4
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_dot_scaled_bf16_fp4(
      %arg0: tensor<32x64x!tt.ptr<bf16>, #blocked2>,
      %arg1: tensor<32x32x!tt.ptr<i8>, #blocked>,
      %arg2: tensor<32x2x!tt.ptr<i8>, #blocked1>,
      %arg3: tensor<32x32x!tt.ptr<f32>, #blocked>
    ) {
    // CHECK: %[[CST:.*]] = arith.constant dense<7> : tensor<2x32xi16, #ttg.slice<{dim = 2, parent = #linear{{.*}}}>>
    // CHECK: %[[B:.*]] = ttg.convert_layout %{{.*}} : tensor<32x32xi8, #blocked{{.*}}> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    // CHECK: %[[S:.*]] = ttg.convert_layout %{{.*}} : tensor<32x2xi8, #blocked{{.*}}> -> tensor<32x2xi8, #linear{{.*}}>
    // CHECK: %[[TS:.*]] = tt.trans %[[S]] {order = array<i32: 1, 0>}
    // CHECK: %[[ES:.*]] = arith.extui %[[TS]]
    // CHECK: %[[SHS:.*]] = arith.shli %[[ES]], %[[CST]]
    // CHECK: %[[BS:.*]] = tt.bitcast %[[SHS]] : tensor<2x32xi16, #ttg.slice<{dim = 2, parent = #linear{{.*}}}>> -> tensor<2x32xbf16, #ttg.slice<{dim = 2, parent = #linear{{.*}}}>>
    // CHECK: %[[EPS:.*]] = tt.expand_dims %[[BS]] {axis = 2 : i32} : tensor<2x32xbf16, #ttg.slice<{dim = 2, parent = #linear{{.*}}}>> -> tensor<2x32x1xbf16, #linear{{.*}}>
    // CHECK: %[[BCS:.*]] = tt.broadcast %[[EPS]] : tensor<2x32x1xbf16, #linear{{.*}}> -> tensor<2x32x32xbf16, #linear{{.*}}>
    // CHECK: %[[TBCS:.*]] = tt.trans %[[BCS]] {order = array<i32: 0, 2, 1>} : tensor<2x32x32xbf16, #linear{{.*}}> -> tensor<2x32x32xbf16, #linear{{.*}}>
    // CHECK: %[[RTBCS:.*]] = tt.reshape %[[TBCS]] : tensor<2x32x32xbf16, #linear{{.*}}> -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: %[[UB:.*]] = amdgpu.scaled_upcast_fp4 %[[B]] scale %[[RTBCS]] {axis = 0 : i32} : tensor<32x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
    // CHECK: %[[A:.*]] = ttg.convert_layout %{{.*}} : tensor<32x64xbf16, #blocked{{.*}}> -> tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
    // CHECK: %{{.*}} = tt.dot %[[A]], %[[UB]], %{{.*}} : tensor<32x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<32x32xf32, #mma>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #blocked>
    %1 = tt.load %arg0 : tensor<32x64x!tt.ptr<bf16>, #blocked2>
    %2 = tt.load %arg1 : tensor<32x32x!tt.ptr<i8>, #blocked>
    %3 = tt.load %arg2 : tensor<32x2x!tt.ptr<i8>, #blocked1>
    %4 = tt.dot_scaled %1, %2 scale %3, %cst lhs = bf16 rhs = e2m1 {fastMath = true} : tensor<32x64xbf16, #blocked2> * tensor<32x32xi8, #blocked>, tensor<32x2xi8, #blocked1> -> tensor<32x32xf32, #blocked>
    tt.store %arg3, %4 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
