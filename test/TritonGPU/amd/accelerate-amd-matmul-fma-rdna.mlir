// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx1100 matrix-instruction-size=0" --verify-diagnostics | FileCheck %s

// An f32 dot is not supported by WMMA, so it stays blocked and is lowered via
// the FMA path. RebalanceBlockedFMA rewrites the layout
// (here threadsPerWarp = [32, 1], i.e. all 32 lanes along M) into a balanced
// one (threadsPerWarp = [2, 16]) to reduce the LDS read volume.

// CHECK-DAG: #[[ORIG1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[BAL1:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: rebalance_layout_fma_dot_f32(
  tt.func public @rebalance_layout_fma_dot_f32(
      %arg0: tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-DAG: %[[A:.+]] = ttg.convert_layout {{.*}} -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[BAL1]]}>>
    // CHECK-DAG: %[[B:.+]] = ttg.convert_layout {{.*}} -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #[[BAL1]]}>>
    // CHECK-DAG: %[[C:.+]] = ttg.convert_layout {{.*}} -> tensor<128x128xf32, #[[BAL1]]>
    // CHECK: %[[D:.+]] = tt.dot %[[A]], %[[B]], %[[C]]{{.*}} -> tensor<128x128xf32, #[[BAL1]]>
    // expected-remark @+1 {{Attempting to map dot operation to FMA intrinsic.}}
    %0 = tt.dot %arg0, %arg1, %cst : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    // CHECK: ttg.convert_layout %[[D]]{{.*}} -> tensor<128x128xf32, #[[ORIG1]]>
    tt.store %arg2, %0 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-DAG: #[[BAL2:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: already_balanced_layout_fma_dot_f32(
  tt.func public @already_balanced_layout_fma_dot_f32(
      %arg0: tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-NOT: ttg.convert_layout
    // CHECK: %{{.*}} = tt.dot {{.*}} -> tensor<128x128xf32, #[[BAL2]]>
    // expected-remark @+1 {{Attempting to map dot operation to FMA intrinsic.}}
    %0 = tt.dot %arg0, %arg1, %cst : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    // CHECK-NOT: ttg.convert_layout
    tt.store %arg2, %0 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-DAG: #[[ORIG3:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[BAL3:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: rebalance_layout_fma_dot_f32_non_square(
  tt.func public @rebalance_layout_fma_dot_f32_non_square(
      %arg0: tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x256xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<64x256x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x256xf32, #blocked>
    // CHECK-DAG: %[[A:.+]] = ttg.convert_layout {{.*}} -> tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[BAL3]]}>>
    // CHECK-DAG: %[[B:.+]] = ttg.convert_layout {{.*}} -> tensor<64x256xf32, #ttg.dot_op<{opIdx = 1, parent = #[[BAL3]]}>>
    // CHECK-DAG: %[[C:.+]] = ttg.convert_layout {{.*}} -> tensor<64x256xf32, #[[BAL3]]>
    // CHECK: %[[D:.+]] = tt.dot %[[A]], %[[B]], %[[C]]{{.*}} -> tensor<64x256xf32, #[[BAL3]]>
    // expected-remark @+1 {{Attempting to map dot operation to FMA intrinsic.}}
    %0 = tt.dot %arg0, %arg1, %cst : tensor<64x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x256xf32, #blocked>
    // CHECK: ttg.convert_layout %[[D]]{{.*}} -> tensor<64x256xf32, #[[ORIG3]]>
    tt.store %arg2, %0 : tensor<64x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK-DAG: #[[ORIG4:.+]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK-DAG: #[[BAL4:.+]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: rebalance_layout_fma_dot_f32_size_per_thread(
  tt.func public @rebalance_layout_fma_dot_f32_size_per_thread(
      %arg0: tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-DAG: %[[A:.+]] = ttg.convert_layout {{.*}} -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[BAL4]]}>>
    // CHECK-DAG: %[[B:.+]] = ttg.convert_layout {{.*}} -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #[[BAL4]]}>>
    // CHECK-DAG: %[[C:.+]] = ttg.convert_layout {{.*}} -> tensor<128x128xf32, #[[BAL4]]>
    // CHECK: %[[D:.+]] = tt.dot %[[A]], %[[B]], %[[C]]{{.*}} -> tensor<128x128xf32, #[[BAL4]]>
    // expected-remark @+1 {{Attempting to map dot operation to FMA intrinsic.}}
    %0 = tt.dot %arg0, %arg1, %cst : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    // CHECK: ttg.convert_layout %[[D]]{{.*}} -> tensor<128x128xf32, #[[ORIG4]]>
    tt.store %arg2, %0 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Same layout as the first test, but with order = [0, 1] instead of
// [1, 0]. Here M (dim 0) is the contiguous/fastest dimension, so the heuristic
// must take that into account and produce the exact transpose of the order = [1, 0].

// CHECK-DAG: #[[ORIG5:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK-DAG: #[[BAL5:.+]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: rebalance_layout_fma_dot_f32_transposed_order(
  tt.func public @rebalance_layout_fma_dot_f32_transposed_order(
      %arg0: tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // CHECK-DAG: %[[A:.+]] = ttg.convert_layout {{.*}} -> tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[BAL5]]}>>
    // CHECK-DAG: %[[B:.+]] = ttg.convert_layout {{.*}} -> tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #[[BAL5]]}>>
    // CHECK-DAG: %[[C:.+]] = ttg.convert_layout {{.*}} -> tensor<128x128xf32, #[[BAL5]]>
    // CHECK: %[[D:.+]] = tt.dot %[[A]], %[[B]], %[[C]]{{.*}} -> tensor<128x128xf32, #[[BAL5]]>
    // expected-remark @+1 {{Attempting to map dot operation to FMA intrinsic.}}
    %0 = tt.dot %arg0, %arg1, %cst : tensor<128x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    // CHECK: ttg.convert_layout %[[D]]{{.*}} -> tensor<128x128xf32, #[[ORIG5]]>
    tt.store %arg2, %0 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
