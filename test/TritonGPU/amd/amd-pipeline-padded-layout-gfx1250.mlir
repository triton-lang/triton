// RUN: triton-opt %s -split-input-file -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline -canonicalize | FileCheck %s

// Verify that the gfx1250 pipeline pass produces padded shared encodings
// for dot-operand loads, with correct padding values per dtype and access
// pattern (transposed vs non-transposed).

// ============================================================
// f16 GEMM: 64x64 tile, 4 warps, WMMA v3
//   opIdx=0 (non-transposed): pad = 128/16 = 8,  interval = K = 32
//   opIdx=1 (transposed):     pad = 2*128/16 = 16, interval = N = 64
// ============================================================
// CHECK: #shared = #ttg.padded_shared<[32:+8] {order = [1, 0], shape = [64, 32]}>
// CHECK: #shared1 = #ttg.padded_shared<[64:+16] {order = [1, 0], shape = [32, 64]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: padded_layout_f16
  tt.func @padded_layout_f16(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
    %2 = tt.broadcast %1 : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    %4 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x64x!tt.ptr<f16>, #blocked>
    %5 = tt.addptr %3, %2 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>

    %7 = scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<64x64xf32, #mma>)  : i32 {
      %9 = tt.load %5 : tensor<64x32x!tt.ptr<f16>, #blocked>
      %11 = ttg.convert_layout %9 : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %10 = tt.load %4 : tensor<32x64x!tt.ptr<f16>, #blocked>
      %12 = ttg.convert_layout %10 : tensor<32x64xf16, #blocked> -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %13 = tt.dot %11, %12, %arg4 : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<32x64xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<64x64xf32, #mma>
      scf.yield %13 : tensor<64x64xf32, #mma>
    }
    tt.return
  }
}

// -----

// ============================================================
// f8E4M3FN GEMM: 64x64 tile, 4 warps, WMMA v3
//   opIdx=0 (non-transposed): pad = 128/8 = 16,  interval = K = 64
//   opIdx=1 (transposed):     pad = 2*64/8 = 16, interval = N = 64
//   Both operands have same shape and padding → single shared encoding
// ============================================================
// CHECK: #shared = #ttg.padded_shared<[64:+16] {order = [1, 0], shape = [64, 64]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 64]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: padded_layout_f8
  tt.func @padded_layout_f8(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %2 = tt.broadcast %1 : tensor<1x64xi32, #blocked> -> tensor<64x64xi32, #blocked>
    %3 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>
    %4 = tt.splat %arg1 : !tt.ptr<f8E4M3FN> -> tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>
    %5 = tt.addptr %3, %2 : tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x64xi32, #blocked>
    %6 = tt.addptr %4, %2 : tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>, tensor<64x64xi32, #blocked>

    %7 = scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg4 = %cst) -> (tensor<64x64xf32, #mma>)  : i32 {
      %9 = tt.load %5 : tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>
      %11 = ttg.convert_layout %9 : tensor<64x64xf8E4M3FN, #blocked> -> tensor<64x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %10 = tt.load %6 : tensor<64x64x!tt.ptr<f8E4M3FN>, #blocked>
      %12 = ttg.convert_layout %10 : tensor<64x64xf8E4M3FN, #blocked> -> tensor<64x64xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %13 = tt.dot %11, %12, %arg4 : tensor<64x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<64x64xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<64x64xf32, #mma>
      scf.yield %13 : tensor<64x64xf32, #mma>
    }
    tt.return
  }
}
