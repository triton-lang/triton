// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 matrix-instruction-size=0" | FileCheck %s

// The MFMA shape must fit the per-warp share of the tile, not just the tile itself.
// 32x64 outputs spread over 4 warps is a 512-output share, but v_mfma_*_32x32x* produces
// 1024 outputs on a single wavefront, so 32x32 cannot be distributed across the warps in
// M/N and 16x16 is selected instead.

// CHECK-DAG: #[[$MMA_CAP:.+]] = #ttg.amd_mfma<{{.*}}instrShape = [16, 16, 16]{{.*}}>
// CHECK-LABEL: mfma_share_limits_shape
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_share_limits_shape(
      %arg0: tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<32x64x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x64xf32, #blocked>
    // CHECK: tt.dot {{.*}} -> tensor<32x64xf32, #[[$MMA_CAP]]>
    %0 = tt.dot %arg0, %arg1, %cst : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf32, #blocked>
    tt.store %arg2, %0 : tensor<32x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// 64x64 outputs over 4 warps is a 1024-output share, which 32x32 fills exactly, so the
// larger shape is still selected. This pins the boundary rather than the direction.

// CHECK-DAG: #[[$MMA_KEEP:.+]] = #ttg.amd_mfma<{{.*}}instrShape = [32, 32, 8]{{.*}}>
// CHECK-LABEL: mfma_share_allows_32x32
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_share_allows_32x32(
      %arg0: tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<64x64x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    // CHECK: tt.dot {{.*}} -> tensor<64x64xf32, #[[$MMA_KEEP]]>
    %0 = tt.dot %arg0, %arg1, %cst : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x64xf32, #blocked>
    tt.store %arg2, %0 : tensor<64x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A floor, not just a cap. 16x16 is the smallest square MFMA, so the share test is not
// applied to it: 16x16 outputs over 4 warps is a 64-output share, which no square MFMA
// fits, and rejecting 16x16 here would leave no candidate at all and silently drop the
// dot off MFMA entirely. Upstream selects 16x16 for this shape and so must this pass.

// CHECK-DAG: #[[$MMA_FLOOR:.+]] = #ttg.amd_mfma<{{.*}}instrShape = [16, 16, 16]{{.*}}>
// CHECK-LABEL: mfma_share_is_floored_at_16
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_share_is_floored_at_16(
      %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<16x16x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    // CHECK: tt.dot {{.*}} -> tensor<16x16xf32, #[[$MMA_FLOOR]]>
    %0 = tt.dot %arg0, %arg1, %cst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x16xf32, #blocked>
    tt.store %arg2, %0 : tensor<16x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
