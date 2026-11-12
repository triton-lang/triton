// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-epilogue | FileCheck %s

// CHECK-LABEL: one_op_in_chain
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @one_op_in_chain(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = arith.truncf %1 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: store_dword_mfma32_small_n
// CHECK-NOT: tensor<32x8x!tt.ptr<f16>, #linear>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x8x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_mfma32_small_n(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x8xf32, #mma>
    %0 = ttg.convert_layout %cst : tensor<32x8xf32, #mma> -> tensor<32x8xf32, #blocked>
    %1 = arith.truncf %0 : tensor<32x8xf32, #blocked> to tensor<32x8xf16, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x8x!tt.ptr<f16>, #blocked>
    tt.store %2, %1 : tensor<32x8x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: two_ops_in_chain
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @two_ops_in_chain(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    %2 = math.exp2 %1 : tensor<32x32xf32, #blocked>
    %3 = arith.truncf %2 : tensor<32x32xf32, #blocked> to tensor<32x32xf16, #blocked>
    %4 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.store %4, %3 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[32, 0], [64, 0]], block = []}>
// CHECK-LABEL: store_dword_128x128
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128x!tt.ptr<f16>, #mma> -> tensor<128x128x!tt.ptr<f16>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<128x128x!tt.ptr<f16>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_128x128(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 128], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 8]], warp = [[0, 32], [0, 64], [32, 0]], block = []}>
// CHECK-LABEL: store_dword_256x256
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<256x256xf32, #mma> -> tensor<256x256xf32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<256x256x!tt.ptr<f16>, #mma> -> tensor<256x256x!tt.ptr<f16>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<256x256xf16, #mma> -> tensor<256x256xf16, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<256x256x!tt.ptr<f16>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_256x256(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<256x256xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<256x256xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<256x256xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<256x256xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<256x256xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<256x256xf32, #mma> -> tensor<256x256xf32, #blocked>
    %2 = arith.truncf %1 : tensor<256x256xf32, #blocked> to tensor<256x256xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<256x256x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<256x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// CHECK{LITERAL}: #linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 32], [0, 64], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16], [0, 8]], warp = [[16, 0], [32, 0]], block = []}>
// CHECK-LABEL: store_dword_16x16
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
// CHECK-DAG: %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128x!tt.ptr<f16>, #mma> -> tensor<128x128x!tt.ptr<f16>, #linear>
// CHECK-DAG: %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #linear>
// CHECK: tt.store %[[PTR]], %[[VAL]] : tensor<128x128x!tt.ptr<f16>, #linear>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_16x16(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// To validate if  warpsPerCTA is not expected, no linear layout will be created.
// CHECK-LABEL: store_dword_16x16
// CHECK-NOT: #linear
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_16x16(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %2 = arith.truncf %1 : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<128x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----
// To validate if N of the input shape is not expected, larger or equal 16X2, no linear layout will be created.
// CHECK-LABEL: store_dword_16x16
// CHECK-NOT: #linear
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16, 32], isTransposed = true}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @store_dword_16x16(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<16x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<16x16xf32, #mma>
    %1 = ttg.convert_layout %0 : tensor<16x16xf32, #mma> -> tensor<16x16xf32, #blocked>
    %2 = arith.truncf %1 : tensor<16x16xf32, #blocked> to tensor<16x16xf16, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<16x16x!tt.ptr<f16>, #blocked>
    tt.store %3, %2 : tensor<16x16x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// FMA dot layout cases.

// Both contiguous sizePerThread are identical: bypass is profitable.
// CHECK-LABEL: fma_blocked_equal_contig
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       %[[PTR:.+]] = ttg.convert_layout %{{.*}} : tensor<64x128x!tt.ptr<f32>, #blocked1> -> tensor<64x128x!tt.ptr<f32>, #blocked>
// CHECK:       tt.store %[[PTR]], %{{.*}} : tensor<64x128x!tt.ptr<f32>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_equal_contig(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<64x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %cst_1 = arith.constant dense<1.230000e+02> : tensor<16x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %0 = tt.dot %cst_0, %cst_1, %cst : tensor<64x16xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x128xf32, #blocked>
    %1 = ttg.convert_layout %0 : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %2, %1 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// Source sizePerThread is bigger than the store layout's: bypass is profitable.
// CHECK-LABEL: fma_blocked_wider_contig
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %{{.*}} : tensor<64x128x!tt.ptr<f32>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_wider_contig(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// Source sizePerThread is smaller than the store layout's: bypass is not profitable.
// CHECK-LABEL: fma_blocked_narrower_contig
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<64x128x!tt.ptr<f32>, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_narrower_contig(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// The two layouts disagree on the order, so bypass is not profitable.
// CHECK-LABEL: fma_blocked_order_mismatch
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<64x128x!tt.ptr<f32>, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_order_mismatch(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// The source keeps a single lane on the contiguous dimension, so consecutive
// lanes do not continue each other's run and the store would not be coalesced,
// even though the contiguous sizePerThread match.
// CHECK-LABEL: fma_blocked_lanes_not_contig
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<64x128x!tt.ptr<f32>, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_lanes_not_contig(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// The source sizePerThread is smaller than the store layout's, but both already
// fill the 128 bit store width, so neither needs more stores than the other and
// the bypass is still profitable.
// CHECK-LABEL: fma_blocked_narrower_but_store_width_bound
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #blocked1>
// CHECK:       tt.store %{{.*}}, %{{.*}} : tensor<64x128x!tt.ptr<f16>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [4, 8], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_narrower_but_store_width_bound(%arg0: !tt.ptr<f16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf16, #blocked>
    %0 = ttg.convert_layout %cst : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked1>
    tt.store %1, %0 : tensor<64x128x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

// -----

// The store dimension holds a single element, so the sizePerThread of 4 along it
// is register broadcast and both layouts map to the same linear layout. Storing
// in the source layout costs the same as in the store layout, so the bypass is
// accepted.
// CHECK-LABEL: fma_blocked_tensor_narrower_than_tile
// CHECK-NOT:   ttg.convert_layout %{{.*}} : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %{{.*}} : tensor<128x1x!tt.ptr<f32>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_blocked_tensor_narrower_than_tile(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x1xf32, #blocked>
    %0 = ttg.convert_layout %cst : tensor<128x1xf32, #blocked> -> tensor<128x1xf32, #blocked1>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked1>
    tt.store %1, %0 : tensor<128x1x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// A row reduction yields a slice layout replicated across the 16 lanes of each
// row, so each thread holds 32 values against 2 in the store layout. The
// contiguous runs match, but storing in the source layout would need more store
// instructions, so the bypass is not profitable.
// CHECK-LABEL: fma_slice_replicated_source
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<128x!tt.ptr<f32>, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx1101", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @fma_slice_replicated_source(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %0 = "tt.reduce"(%cst) <{axis = 1 : i32}> ({
    ^bb0(%a: f32, %b: f32):
      %m = arith.maxnumf %a, %b : f32
      tt.reduce.return %m : f32
    }) : (tensor<128x64xf32, #blocked>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %1 = ttg.convert_layout %0 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128xf32, #blocked1>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    tt.store %2, %1 : tensor<128x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

// Transposing an MFMA accumulator yields a linear layout, which is not an MMA
// layout. Its registers run along dim 1 but its lanes run along dim 0, so the
// store would not be coalesced and the bypass is not profitable.
// CHECK-LABEL: mfma_transposed_linear_source
// CHECK:       %[[VAL:.+]] = ttg.convert_layout %{{.*}} : tensor<128x64xf32, #linear> -> tensor<128x64xf32, #blocked>
// CHECK:       tt.store %{{.*}}, %[[VAL]] : tensor<128x64x!tt.ptr<f32>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [0, 4]], warp = [[32, 0], [64, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_transposed_linear_source(%arg0: !tt.ptr<f32>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %0 = tt.trans %cst {order = array<i32: 1, 0>} : tensor<64x128xf32, #mma> -> tensor<128x64xf32, #linear>
    %1 = ttg.convert_layout %0 : tensor<128x64xf32, #linear> -> tensor<128x64xf32, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x64x!tt.ptr<f32>, #blocked>
    tt.store %2, %1 : tensor<128x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----
// gfx1250 WMMA keeps the blocked store, so the tile goes out as whole lines.
// CHECK-LABEL: wmma_v3_keeps_blocked_store
// CHECK: ttg.convert_layout %{{.*}} : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<128x128x!tt.ptr<bf16>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_v3_keeps_blocked_store(%arg0: !tt.ptr<bf16>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = ttg.convert_layout %cst : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked>
    %1 = arith.truncf %0 : tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %2 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.store %2, %1 : tensor<128x128x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}
