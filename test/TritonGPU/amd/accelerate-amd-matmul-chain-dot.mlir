// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx942 matrix-instruction-size=16" | FileCheck %s --check-prefixes MFMA16,CHECK
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx942 matrix-instruction-size=32" | FileCheck %s --check-prefixes MFMA32,CHECK
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx950 matrix-instruction-size=32" | FileCheck %s --check-prefixes CHECK-GFX950
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx950 matrix-instruction-size=16" | FileCheck %s --check-prefixes CHECK-GFX950

// Check the warpsPerCTA parameter of #mma layout of the two dot's.
// The 1st dot always has warpsPerCTA = [4, 1].
// The warpsPerCTA for the 2nd dot depends on mfma instruction size and BLOCK_M size.


// BLOCK_M = 128
// warpsPerCTA = [4, 1] for mfma16 and mfma32
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
// CHECK-LABEL: mfma_chain_dot_BM128
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<128x16xf32, #mma>
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<128x128xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_BM128(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x16xf16, #dotOp1>,
      %v: tensor<16x128xf16, #dotOp1>,
      %o_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x16xf16, #dotOp1> -> tensor<128x16xf32, #blocked>
    %qk_f16 = arith.truncf %qk :  tensor<128x16xf32, #blocked> to tensor<128x16xf16, #blocked>
    %p = ttg.convert_layout %qk_f16 : tensor<128x16xf16, #blocked> -> tensor<128x16xf16, #dotOp0>
    %o = tt.dot %p, %v, %cst1 : tensor<128x16xf16, #dotOp0> * tensor<16x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----

// BLOCK_M = 64
// warpsPerCTA = [4, 1] for mfma16
// warpsPerCTA = [2, 2] for mfma32
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
// MFMA32{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
// CHECK-LABEL: mfma_chain_dot_BM64
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<64x16xf32, #mma>
// MFMA16: tt.dot {{.*}} : {{.*}} -> tensor<64x128xf32, #mma>
// MFMA32: tt.dot {{.*}} : {{.*}} -> tensor<64x128xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_BM64(
      %q: tensor<64x128xf16, #dotOp0>,
      %k: tensor<128x16xf16, #dotOp1>,
      %v: tensor<16x128xf16, #dotOp1>,
      %o_ptr: tensor<64x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<64x128xf16, #dotOp0> * tensor<128x16xf16, #dotOp1> -> tensor<64x16xf32, #blocked>
    %qk_f16 = arith.truncf %qk :  tensor<64x16xf32, #blocked> to tensor<64x16xf16, #blocked>
    %p = ttg.convert_layout %qk_f16 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #dotOp0>
    %o = tt.dot %p, %v, %cst1 : tensor<64x16xf16, #dotOp0> * tensor<16x128xf16, #dotOp1> -> tensor<64x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<64x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----

// BLOCK_M = 32
// warpsPerCTA = [2, 2] for mfma16
// warpsPerCTA = [1, 4] for mfma32
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
// MFMA16{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [32, 32, 8], isTransposed = true}>
// CHECK-LABEL: mfma_chain_dot_BM32
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<32x16xf32, #mma>
// MFMA16: tt.dot {{.*}} : {{.*}} -> tensor<32x128xf32, #mma1>
// MFMA32: tt.dot {{.*}} : {{.*}} -> tensor<32x128xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_BM32(
      %q: tensor<32x128xf16, #dotOp0>,
      %k: tensor<128x16xf16, #dotOp1>,
      %v: tensor<16x128xf16, #dotOp1>,
      %o_ptr: tensor<32x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<32x16xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<32x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<32x128xf16, #dotOp0> * tensor<128x16xf16, #dotOp1> -> tensor<32x16xf32, #blocked>
    %qk_f16 = arith.truncf %qk :  tensor<32x16xf32, #blocked> to tensor<32x16xf16, #blocked>
    %p = ttg.convert_layout %qk_f16 : tensor<32x16xf16, #blocked> -> tensor<32x16xf16, #dotOp0>
    %o = tt.dot %p, %v, %cst1 : tensor<32x16xf16, #dotOp0> * tensor<16x128xf16, #dotOp1> -> tensor<32x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<32x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----

// BLOCK_M = 16, only check mfma16 since it's too small for mfma32
// warpsPerCTA = [1, 4] for mfma16
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA16{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [16, 16, 16], isTransposed = true}>
// CHECK-LABEL: mfma_chain_dot_BM16
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<16x16xf32, #mma>
// MFMA16: tt.dot {{.*}} : {{.*}} -> tensor<16x128xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_BM16(
      %q: tensor<16x128xf16, #dotOp0>,
      %k: tensor<128x16xf16, #dotOp1>,
      %v: tensor<16x128xf16, #dotOp1>,
      %o_ptr: tensor<16x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<16x128xf16, #dotOp0> * tensor<128x16xf16, #dotOp1> -> tensor<16x16xf32, #blocked>
    %qk_f16 = arith.truncf %qk :  tensor<16x16xf32, #blocked> to tensor<16x16xf16, #blocked>
    %p = ttg.convert_layout %qk_f16 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #dotOp0>
    %o = tt.dot %p, %v, %cst1 : tensor<16x16xf16, #dotOp0> * tensor<16x128xf16, #dotOp1> -> tensor<16x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<16x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}


// -----

// Check kWidth of both operands of the 2nd dot. To avoid in-warp shuffle for
// the layout conversion from #mma to #dotOp, kWidth should be set to 4

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK-LABEL: mfma_chain_dot_kWidth_f16
// CHECK-GFX950: tt.dot {{.*}} : {{.*}} -> tensor<128x128xf32, #mma>
// CHECK-GFX950: tt.dot {{.*}} : tensor<128x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> {{.*}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_kWidth_f16(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x128xf16, #dotOp1>,
      %v: tensor<128x128xf16, #dotOp1>,
      %o_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    %qk_f16 = arith.truncf %qk :  tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    %p = ttg.convert_layout %qk_f16 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #dotOp0>
    %o = tt.dot %p, %v, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK-LABEL: mfma_chain_dot_kWidth_bf16
// CHECK-GFX950: tt.dot {{.*}} : {{.*}} -> tensor<128x128xf32, #mma>
// CHECK-GFX950: tt.dot {{.*}} : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> {{.*}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_chain_dot_kWidth_bf16(
      %q: tensor<128x128xbf16, #dotOp0>,
      %k: tensor<128x128xbf16, #dotOp1>,
      %v: tensor<128x128xbf16, #dotOp1>,
      %o_ptr: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<128x128xbf16, #dotOp0> * tensor<128x128xbf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    %qk_bf16 = arith.truncf %qk :  tensor<128x128xf32, #blocked> to tensor<128x128xbf16, #blocked>
    %p = ttg.convert_layout %qk_bf16 : tensor<128x128xbf16, #blocked> -> tensor<128x128xbf16, #dotOp0>
    %o = tt.dot %p, %v, %cst : tensor<128x128xbf16, #dotOp0> * tensor<128x128xbf16, #dotOp1> -> tensor<128x128xf32, #blocked>
    tt.store %o_ptr, %o : tensor<128x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1, 8], threadsPerWarp = [1, 32, 2], warpsPerCTA = [1, 4, 1], order = [2, 1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1, 1, 8], threadsPerWarp = [1, 4, 16, 1], warpsPerCTA = [1, 4, 1, 1], order = [3, 2, 1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 8, 1, 1], threadsPerWarp = [4, 1, 1, 16], warpsPerCTA = [4, 1, 1, 1], order = [1, 3, 0, 2]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 8, 1], threadsPerWarp = [1, 2, 32], warpsPerCTA = [1, 1, 4], order = [1, 2, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[0, 4], [0, 8], [1, 0], [2, 0], [4, 0], [8, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 16], [0, 32], [0, 64], [0, 128]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [16, 0], [32, 0], [64, 0], [128, 0]], lane = [[8, 0], [0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 32], [0, 64]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @_paged_attn_decode_v2_w_dot_kernel_reshape_noloop_qk(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg6: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg7: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg8: f32, %arg9: f32, %arg10: f32, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}, %arg18: i32 {tt.divisibility = 16 : i32}, %arg19: i32 {tt.divisibility = 16 : i32}, %arg20: i32 {tt.divisibility = 16 : i32}, %arg21: i32 {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32, %arg24: i32 {tt.divisibility = 16 : i32}, %arg25: i32 {tt.divisibility = 16 : i32}, %arg26: i32 {tt.divisibility = 16 : i32}, %arg27: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<16x256xf32, #blocked>
    %cst_0 = arith.constant dense<1.44269502> : tensor<16x256xf32, #blocked>
    %cst_1 = arith.constant dense<8> : tensor<16xi32, #blocked1>
    %c15_i32 = arith.constant 15 : i32
    %c8_i32 = arith.constant 8 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16x128xf32, #blocked2>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #blocked>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<16x128xbf16, #blocked3>
    %c256_i32 = arith.constant 256 : i32
    %cst_5 = arith.constant dense<0> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>>
    %cst_6 = arith.constant dense<0> : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>
    %cst_7 = arith.constant dense<8> : tensor<16x1xi32, #blocked3>
    %cst_8 = arith.constant dense<8> : tensor<16x1xi32, #blocked>
    %cst_9 = arith.constant dense<128> : tensor<1x128xi32, #blocked3>
    %cst_10 = arith.constant dense<8> : tensor<1x1x16x1xi32, #blocked5>
    %cst_11 = arith.constant dense<16> : tensor<16x1xi32, #linear>
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = tt.addptr %arg7, %0 : !tt.ptr<i32>, i32
    %4 = tt.load %3 : !tt.ptr<i32>
    %5 = arith.muli %2, %c256_i32 : i32
    %6 = arith.cmpi sge, %5, %4 : i32
    cf.cond_br %6, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    tt.return
  ^bb2:  // pred: ^bb0
    %7 = arith.addi %5, %c256_i32 : i32
    %8 = arith.minsi %7, %4 : i32
    %9 = arith.subi %8, %5 : i32
    %10 = arith.addi %9, %c15_i32 : i32
    %11 = arith.divsi %10, %c16_i32 : i32
    %12 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>>
    %13 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>
    %14 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked1>
    %15 = tt.splat %11 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>>
    %16 = tt.splat %11 : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>
    %17 = arith.cmpi slt, %12, %15 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>>
    %18 = arith.cmpi slt, %13, %16 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>
    %19 = arith.select %17, %12, %cst_5 : tensor<16xi1, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>>
    %20 = arith.select %18, %13, %cst_6 : tensor<16xi1, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>
    %21 = arith.muli %2, %c16_i32 : i32
    %22 = arith.muli %0, %arg27 : i32
    %23 = tt.addptr %arg6, %22 : !tt.ptr<i32>, i32
    %24 = tt.addptr %23, %21 : !tt.ptr<i32>, i32
    %25 = tt.splat %24 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>>
    %26 = tt.splat %24 : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>
    %27 = tt.addptr %25, %19 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>>
    %28 = tt.addptr %26, %20 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>, tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>
    %29 = tt.load %27 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>>
    %30 = tt.load %28 : tensor<16x!tt.ptr<i32>, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>
    %31 = arith.muli %0, %arg18 : i32
    %32 = arith.muli %1, %c8_i32 : i32
    %33 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>>
    %34 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %35 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %36 = tt.expand_dims %33 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked3}>> -> tensor<16x1xi32, #blocked3>
    %37 = tt.expand_dims %34 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<16x1xi32, #linear>
    %38 = tt.expand_dims %35 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %39 = tt.splat %32 : i32 -> tensor<16x1xi32, #blocked3>
    %40 = arith.addi %39, %36 : tensor<16x1xi32, #blocked3>
    %41 = tt.splat %arg19 : i32 -> tensor<16x1xi32, #blocked3>
    %42 = arith.muli %40, %41 : tensor<16x1xi32, #blocked3>
    %43 = tt.splat %31 : i32 -> tensor<16x1xi32, #blocked3>
    %44 = arith.addi %43, %42 : tensor<16x1xi32, #blocked3>
    %45 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %46 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>>
    %47 = tt.expand_dims %45 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x128xi32, #blocked3>
    %48 = tt.expand_dims %46 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>> -> tensor<1x128xi32, #ttg.slice<{dim = 2, parent = #blocked4}>>
    %49 = tt.broadcast %44 : tensor<16x1xi32, #blocked3> -> tensor<16x128xi32, #blocked3>
    %50 = tt.broadcast %47 : tensor<1x128xi32, #blocked3> -> tensor<16x128xi32, #blocked3>
    %51 = arith.addi %49, %50 : tensor<16x128xi32, #blocked3>
    %52 = arith.cmpi slt, %36, %cst_7 : tensor<16x1xi32, #blocked3>
    %53 = arith.cmpi slt, %38, %cst_8 : tensor<16x1xi32, #blocked>
    %54 = arith.cmpi slt, %47, %cst_9 : tensor<1x128xi32, #blocked3>
    %55 = tt.broadcast %52 : tensor<16x1xi1, #blocked3> -> tensor<16x128xi1, #blocked3>
    %56 = tt.broadcast %54 : tensor<1x128xi1, #blocked3> -> tensor<16x128xi1, #blocked3>
    %57 = arith.andi %55, %56 : tensor<16x128xi1, #blocked3>
    %58 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>, #blocked3>
    %59 = tt.addptr %58, %51 : tensor<16x128x!tt.ptr<bf16>, #blocked3>, tensor<16x128xi32, #blocked3>
    %60 = tt.load %59, %57, %cst_4 : tensor<16x128x!tt.ptr<bf16>, #blocked3>
    %61 = arith.extf %60 : tensor<16x128xbf16, #blocked3> to tensor<16x128xf32, #blocked3>
    %62 = tt.splat %arg8 : f32 -> tensor<16x128xf32, #blocked3>
    %63 = arith.mulf %61, %62 : tensor<16x128xf32, #blocked3>
    %64 = arith.truncf %63 : tensor<16x128xf32, #blocked3> to tensor<16x128xbf16, #blocked3>
    %65 = tt.expand_dims %29 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked4}>}>> -> tensor<16x1xi32, #ttg.slice<{dim = 2, parent = #blocked4}>>
    %66 = tt.expand_dims %30 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>> -> tensor<16x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>>
    %67 = tt.expand_dims %65 {axis = 2 : i32} : tensor<16x1xi32, #ttg.slice<{dim = 2, parent = #blocked4}>> -> tensor<16x1x1xi32, #blocked4>
    %68 = tt.expand_dims %66 {axis = 2 : i32} : tensor<16x1xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>> -> tensor<16x1x1xi32, #ttg.slice<{dim = 3, parent = #blocked5}>>
    %69 = tt.expand_dims %68 {axis = 3 : i32} : tensor<16x1x1xi32, #ttg.slice<{dim = 3, parent = #blocked5}>> -> tensor<16x1x1x1xi32, #blocked5>
    %70 = tt.splat %arg20 : i32 -> tensor<16x1x1x1xi32, #blocked5>
    %71 = arith.muli %69, %70 : tensor<16x1x1x1xi32, #blocked5>
    %72 = arith.muli %1, %arg21 : i32
    %73 = tt.splat %72 : i32 -> tensor<16x1x1x1xi32, #blocked5>
    %74 = arith.addi %71, %73 : tensor<16x1x1x1xi32, #blocked5>
    %75 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %76 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>
    %77 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked4}>}>>
    %78 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>>
    %79 = tt.expand_dims %75 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x16xi32, #linear>
    %80 = tt.expand_dims %76 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>> -> tensor<1x16xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>>
    %81 = tt.expand_dims %77 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #blocked4}>}>> -> tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %82 = tt.expand_dims %78 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>}>> -> tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>>
    %83 = tt.expand_dims %80 {axis = 2 : i32} : tensor<1x16xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>> -> tensor<1x16x1xi32, #ttg.slice<{dim = 3, parent = #blocked5}>>
    %84 = tt.expand_dims %83 {axis = 3 : i32} : tensor<1x16x1xi32, #ttg.slice<{dim = 3, parent = #blocked5}>> -> tensor<1x16x1x1xi32, #blocked5>
    %85 = tt.splat %arg22 : i32 -> tensor<1x16x1x1xi32, #blocked5>
    %86 = arith.muli %84, %85 : tensor<1x16x1x1xi32, #blocked5>
    %87 = tt.broadcast %74 : tensor<16x1x1x1xi32, #blocked5> -> tensor<16x16x1x1xi32, #blocked5>
    %88 = tt.broadcast %86 : tensor<1x16x1x1xi32, #blocked5> -> tensor<16x16x1x1xi32, #blocked5>
    %89 = arith.addi %87, %88 : tensor<16x16x1x1xi32, #blocked5>
    %90 = tt.expand_dims %81 {axis = 1 : i32} : tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<1x1x16xi32, #blocked4>
    %91 = tt.expand_dims %82 {axis = 1 : i32} : tensor<1x16xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 3, parent = #blocked5}>}>> -> tensor<1x1x16xi32, #ttg.slice<{dim = 3, parent = #blocked5}>>
    %92 = tt.expand_dims %91 {axis = 3 : i32} : tensor<1x1x16xi32, #ttg.slice<{dim = 3, parent = #blocked5}>> -> tensor<1x1x16x1xi32, #blocked5>
    %93 = arith.muli %92, %cst_10 : tensor<1x1x16x1xi32, #blocked5>
    %94 = tt.broadcast %89 : tensor<16x16x1x1xi32, #blocked5> -> tensor<16x16x16x1xi32, #blocked5>
    %95 = tt.broadcast %93 : tensor<1x1x16x1xi32, #blocked5> -> tensor<16x16x16x1xi32, #blocked5>
    %96 = arith.addi %94, %95 : tensor<16x16x16x1xi32, #blocked5>
    %97 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked5}>}>}>>
    %98 = tt.expand_dims %97 {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked5}>}>}>> -> tensor<1x8xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked5}>}>>
    %99 = tt.expand_dims %98 {axis = 1 : i32} : tensor<1x8xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked5}>}>> -> tensor<1x1x8xi32, #ttg.slice<{dim = 2, parent = #blocked5}>>
    %100 = tt.expand_dims %99 {axis = 2 : i32} : tensor<1x1x8xi32, #ttg.slice<{dim = 2, parent = #blocked5}>> -> tensor<1x1x1x8xi32, #blocked5>
    %101 = tt.broadcast %96 : tensor<16x16x16x1xi32, #blocked5> -> tensor<16x16x16x8xi32, #blocked5>
    %102 = tt.broadcast %100 : tensor<1x1x1x8xi32, #blocked5> -> tensor<16x16x16x8xi32, #blocked5>
    %103 = arith.addi %101, %102 : tensor<16x16x16x8xi32, #blocked5>
    %104 = tt.splat %21 : i32 -> tensor<16x1xi32, #linear>
    %105 = arith.addi %104, %37 : tensor<16x1xi32, #linear>
    %106 = arith.muli %105, %cst_11 : tensor<16x1xi32, #linear>
    %107 = tt.broadcast %106 : tensor<16x1xi32, #linear> -> tensor<16x16xi32, #linear>
    %108 = tt.broadcast %79 : tensor<1x16xi32, #linear> -> tensor<16x16xi32, #linear>
    %109 = arith.addi %107, %108 : tensor<16x16xi32, #linear>
    %110 = tt.splat %arg4 : !tt.ptr<bf16> -> tensor<16x16x16x8x!tt.ptr<bf16>, #blocked5>
    %111 = tt.addptr %110, %103 : tensor<16x16x16x8x!tt.ptr<bf16>, #blocked5>, tensor<16x16x16x8xi32, #blocked5>
    %112 = tt.load %111 : tensor<16x16x16x8x!tt.ptr<bf16>, #blocked5>
    %113 = tt.trans %112 {order = array<i32: 1, 3, 0, 2>} : tensor<16x16x16x8xbf16, #blocked5> -> tensor<16x8x16x16xbf16, #blocked6>
    %114 = tt.reshape %113 : tensor<16x8x16x16xbf16, #blocked6> -> tensor<128x256xbf16, #linear1>
    %115 = ttg.convert_layout %64 : tensor<16x128xbf16, #blocked3> -> tensor<16x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %116 = ttg.convert_layout %114 : tensor<128x256xbf16, #linear1> -> tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    %117 = tt.dot %115, %116, %cst_3, inputPrecision = tf32 : tensor<16x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<16x256xf32, #blocked>
    %118 = tt.reshape %109 : tensor<16x16xi32, #linear> -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %119 = tt.expand_dims %118 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %120 = tt.splat %4 : i32 -> tensor<1x256xi32, #blocked>
    %121 = arith.cmpi slt, %119, %120 : tensor<1x256xi32, #blocked>
    %122 = tt.broadcast %53 : tensor<16x1xi1, #blocked> -> tensor<16x256xi1, #blocked>
    %123 = tt.broadcast %121 : tensor<1x256xi1, #blocked> -> tensor<16x256xi1, #blocked>
    %124 = arith.andi %122, %123 : tensor<16x256xi1, #blocked>
    %125 = arith.select %124, %117, %cst : tensor<16x256xi1, #blocked>, tensor<16x256xf32, #blocked>
    %126 = "tt.reduce"(%125) <{axis = 1 : i32}> ({
    ^bb0(%arg28: f32, %arg29: f32):
      %192 = arith.maxnumf %arg28, %arg29 : f32
      tt.reduce.return %192 : f32
    }) : (tensor<16x256xf32, #blocked>) -> tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %127 = tt.expand_dims %126 {axis = 1 : i32} : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xf32, #blocked>
    %128 = tt.broadcast %127 : tensor<16x1xf32, #blocked> -> tensor<16x256xf32, #blocked>
    %129 = arith.subf %125, %128 : tensor<16x256xf32, #blocked>
    %130 = arith.mulf %129, %cst_0 : tensor<16x256xf32, #blocked>
    %131 = math.exp2 %130 : tensor<16x256xf32, #blocked>
    %132 = arith.truncf %131 : tensor<16x256xf32, #blocked> to tensor<16x256xbf16, #blocked>
    %133 = "tt.reduce"(%132) <{axis = 1 : i32}> ({
    ^bb0(%arg28: bf16, %arg29: bf16):
      %192 = arith.addf %arg28, %arg29 : bf16
      tt.reduce.return %192 : bf16
    }) : (tensor<16x256xbf16, #blocked>) -> tensor<16xbf16, #ttg.slice<{dim = 1, parent = #blocked}>>
    %134 = tt.splat %arg24 : i32 -> tensor<16x1x1xi32, #blocked4>
    %135 = arith.muli %67, %134 : tensor<16x1x1xi32, #blocked4>
    %136 = arith.muli %1, %arg25 : i32
    %137 = tt.splat %136 : i32 -> tensor<16x1x1xi32, #blocked4>
    %138 = arith.addi %135, %137 : tensor<16x1x1xi32, #blocked4>
    %139 = tt.expand_dims %48 {axis = 2 : i32} : tensor<1x128xi32, #ttg.slice<{dim = 2, parent = #blocked4}>> -> tensor<1x128x1xi32, #blocked4>
    %140 = tt.splat %arg26 : i32 -> tensor<1x128x1xi32, #blocked4>
    %141 = arith.muli %139, %140 : tensor<1x128x1xi32, #blocked4>
    %142 = tt.broadcast %138 : tensor<16x1x1xi32, #blocked4> -> tensor<16x128x1xi32, #blocked4>
    %143 = tt.broadcast %141 : tensor<1x128x1xi32, #blocked4> -> tensor<16x128x1xi32, #blocked4>
    %144 = arith.addi %142, %143 : tensor<16x128x1xi32, #blocked4>
    %145 = tt.broadcast %144 : tensor<16x128x1xi32, #blocked4> -> tensor<16x128x16xi32, #blocked4>
    %146 = tt.broadcast %90 : tensor<1x1x16xi32, #blocked4> -> tensor<16x128x16xi32, #blocked4>
    %147 = arith.addi %145, %146 : tensor<16x128x16xi32, #blocked4>
    %148 = tt.splat %arg5 : !tt.ptr<bf16> -> tensor<16x128x16x!tt.ptr<bf16>, #blocked4>
    %149 = tt.addptr %148, %147 : tensor<16x128x16x!tt.ptr<bf16>, #blocked4>, tensor<16x128x16xi32, #blocked4>
    %150 = tt.load %149 : tensor<16x128x16x!tt.ptr<bf16>, #blocked4>
    %151 = tt.trans %150 {order = array<i32: 0, 2, 1>} : tensor<16x128x16xbf16, #blocked4> -> tensor<16x16x128xbf16, #blocked7>
    %152 = tt.reshape %151 : tensor<16x16x128xbf16, #blocked7> -> tensor<256x128xbf16, #linear2>
    %153 = arith.muli %0, %arg11 : i32
    %154 = arith.muli %1, %arg12 : i32
    %155 = arith.addi %153, %154 : i32
    %156 = arith.muli %2, %arg13 : i32
    %157 = arith.addi %155, %156 : i32
    %158 = tt.splat %157 : i32 -> tensor<16xi32, #blocked1>
    %159 = arith.addi %158, %14 : tensor<16xi32, #blocked1>
    %160 = arith.cmpi slt, %14, %cst_1 : tensor<16xi32, #blocked1>
    %161 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked1>
    %162 = tt.addptr %161, %159 : tensor<16x!tt.ptr<f32>, #blocked1>, tensor<16xi32, #blocked1>
    %163 = ttg.convert_layout %126 : tensor<16xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16xf32, #blocked1>
    tt.store %162, %163, %160 : tensor<16x!tt.ptr<f32>, #blocked1>
    %164 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>, #blocked1>
    %165 = tt.addptr %164, %159 : tensor<16x!tt.ptr<f32>, #blocked1>, tensor<16xi32, #blocked1>
    %166 = ttg.convert_layout %133 : tensor<16xbf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16xbf16, #blocked1>
    %167 = arith.extf %166 : tensor<16xbf16, #blocked1> to tensor<16xf32, #blocked1>
    tt.store %165, %167, %160 : tensor<16x!tt.ptr<f32>, #blocked1>
    %168 = ttg.convert_layout %132 : tensor<16x256xbf16, #blocked> -> tensor<16x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>>
    %169 = ttg.convert_layout %152 : tensor<256x128xbf16, #linear2> -> tensor<256x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>>
    %170 = tt.dot %168, %169, %cst_2, inputPrecision = tf32 : tensor<16x256xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked2}>> * tensor<256x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked2}>> -> tensor<16x128xf32, #blocked2>
    %171 = tt.expand_dims %133 {axis = 1 : i32} : tensor<16xbf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xbf16, #blocked>
    %172 = arith.extf %171 : tensor<16x1xbf16, #blocked> to tensor<16x1xf32, #blocked>
    %173 = ttg.convert_layout %172 : tensor<16x1xf32, #blocked> -> tensor<16x1xf32, #blocked2>
    %174 = tt.broadcast %173 : tensor<16x1xf32, #blocked2> -> tensor<16x128xf32, #blocked2>
    %175 = arith.divf %170, %174 : tensor<16x128xf32, #blocked2>
    %176 = arith.muli %0, %arg14 : i32
    %177 = arith.muli %1, %arg15 : i32
    %178 = arith.addi %176, %177 : i32
    %179 = arith.muli %2, %arg16 : i32
    %180 = tt.splat %arg17 : i32 -> tensor<16x1xi32, #blocked3>
    %181 = arith.muli %36, %180 : tensor<16x1xi32, #blocked3>
    %182 = tt.splat %179 : i32 -> tensor<16x1xi32, #blocked3>
    %183 = arith.addi %182, %181 : tensor<16x1xi32, #blocked3>
    %184 = tt.broadcast %183 : tensor<16x1xi32, #blocked3> -> tensor<16x128xi32, #blocked3>
    %185 = arith.addi %184, %50 : tensor<16x128xi32, #blocked3>
    %186 = tt.splat %178 : i32 -> tensor<16x128xi32, #blocked3>
    %187 = arith.addi %186, %185 : tensor<16x128xi32, #blocked3>
    %188 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<16x128x!tt.ptr<bf16>, #blocked3>
    %189 = tt.addptr %188, %187 : tensor<16x128x!tt.ptr<bf16>, #blocked3>, tensor<16x128xi32, #blocked3>
    %190 = arith.truncf %175 : tensor<16x128xf32, #blocked2> to tensor<16x128xbf16, #blocked2>
    %191 = ttg.convert_layout %190 : tensor<16x128xbf16, #blocked2> -> tensor<16x128xbf16, #blocked3>
    tt.store %189, %191, %57 : tensor<16x128x!tt.ptr<bf16>, #blocked3>
    tt.return
  }
}
