// RUN: triton-opt %s -split-input-file -tritonamdgpu-in-thread-transpose | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK: [[threadrake_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [4, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK: [[load_ptr:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<64x256x!tt.ptr<f16>, [[threadrake_layout]]>
// CHECK: {{.*}} = tt.load [[load_ptr]] : tensor<64x256x!tt.ptr<f16>, [[threadrake_layout]]>
  tt.func public @threadRake_transpose_b(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x64x!tt.ptr<f16>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked1>
    %2 = tt.load %0 : tensor<256x64x!tt.ptr<f16>, #blocked>
    %3 = tt.load %1 : tensor<64x256x!tt.ptr<f16>, #blocked1>
    %4 = triton_gpu.convert_layout %2 : tensor<256x64xf16, #blocked> -> tensor<256x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %5 = triton_gpu.convert_layout %3 : tensor<64x256xf16, #blocked1> -> tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %6 = tt.dot %4, %5, %cst_0 : tensor<256x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [32, 32], isTransposed = true}>

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, triton_gpu.target = "hip:gfx942", "triton_gpu.threads-per-warp" = 64 : i32} {

// CHECK: [[threadrake_layout:#.*]] = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK: [[load_ptr:%.*]] = triton_gpu.convert_layout {{.*}} -> tensor<32x128x!tt.ptr<f16>, [[threadrake_layout]]>
// CHECK: {{.*}} = tt.load [[load_ptr]] : tensor<32x128x!tt.ptr<f16>, [[threadrake_layout]]>
  tt.func public @threadRake_transpose_b_no_change(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x32x!tt.ptr<f16>, #blocked>
    %1 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked1>
    %2 = tt.load %0 : tensor<256x32x!tt.ptr<f16>, #blocked>
    %3 = tt.load %1 : tensor<32x128x!tt.ptr<f16>, #blocked1>
    %4 = triton_gpu.convert_layout %2 : tensor<256x32xf16, #blocked> -> tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %5 = triton_gpu.convert_layout %3 : tensor<32x128xf16, #blocked1> -> tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
    %6 = tt.dot %4, %5, %cst_0 : tensor<256x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x128xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x128xf32, #mma>
    tt.return
  }
}