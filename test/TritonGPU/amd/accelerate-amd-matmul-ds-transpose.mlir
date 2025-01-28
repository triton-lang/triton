// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul='arch-generation-name=gfx950 matrix-instruction-size=16' | FileCheck %s --check-prefixes CHECK

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true, transA = false, transB = true}>
// CHECK-LABEL: ds_transpose_t_t
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @ds_transpose_t_t(
      %arg0: tensor<128x64x!tt.ptr<f16>, #blocked1>,
      %arg1: tensor<64x128x!tt.ptr<f16>, #blocked2>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.load %arg0 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %1 = tt.load %arg1 : tensor<64x128x!tt.ptr<f16>, #blocked2>
    %2 = ttg.convert_layout %0 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %3 = ttg.convert_layout %1 : tensor<64x128xf16, #blocked2> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %4 = tt.dot %2, %3, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %4 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true, transA = true, transB = true}>
// CHECK-LABEL: ds_transpose_n_t
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @ds_transpose_n_t(
      %arg0: tensor<128x64x!tt.ptr<f16>, #blocked1>,
      %arg1: tensor<64x128x!tt.ptr<f16>, #blocked2>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.load %arg0 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %1 = tt.load %arg1 : tensor<64x128x!tt.ptr<f16>, #blocked2>
    %2 = ttg.convert_layout %0 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %3 = ttg.convert_layout %1 : tensor<64x128xf16, #blocked2> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %4 = tt.dot %2, %3, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %4 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK: #mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true, transA = true, transB = false}>
// CHECK-LABEL: ds_transpose_n_n
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @ds_transpose_n_n(
      %arg0: tensor<128x64x!tt.ptr<f16>, #blocked1>,
      %arg1: tensor<64x128x!tt.ptr<f16>, #blocked2>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.load %arg0 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %1 = tt.load %arg1 : tensor<64x128x!tt.ptr<f16>, #blocked2>
    %2 = ttg.convert_layout %0 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %3 = ttg.convert_layout %1 : tensor<64x128xf16, #blocked2> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %4 = tt.dot %2, %3, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %4 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK: #mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true, transA = false, transB = false}>
// CHECK-LABEL: ds_transpose_t_n
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @ds_transpose_t_n(
      %arg0: tensor<128x64x!tt.ptr<f16>, #blocked1>,
      %arg1: tensor<64x128x!tt.ptr<f16>, #blocked2>,
      %arg2: tensor<128x128x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.load %arg0 : tensor<128x64x!tt.ptr<f16>, #blocked1>
    %1 = tt.load %arg1 : tensor<64x128x!tt.ptr<f16>, #blocked2>
    %2 = ttg.convert_layout %0 : tensor<128x64xf16, #blocked1> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %3 = ttg.convert_layout %1 : tensor<64x128xf16, #blocked2> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %4 = tt.dot %2, %3, %cst : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x128xf32, #blocked>
    tt.store %arg2, %4 : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
