// RUN: triton-opt %s -split-input-file -tritonamdgpu-optimize-epilogue | FileCheck %s

// CHECK-LABEL: one_op_in_chain
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
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

// CHECK-LABEL: two_ops_in_chain
// CHECK-NOT: ttg.convert_layout %{{.*}} : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
// CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x32x!tt.ptr<f16>, #mma>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [1, 1], instrShape = [32, 32], isTransposed = false}>
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
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
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
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [32, 32], isTransposed = true}>
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
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
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
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
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
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
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
