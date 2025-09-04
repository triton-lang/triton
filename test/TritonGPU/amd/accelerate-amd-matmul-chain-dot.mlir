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
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
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
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
// MFMA32{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
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
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
// MFMA16{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [32, 32], isTransposed = true}>
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
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16], isTransposed = true}>
// MFMA16{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [16, 16], isTransposed = true}>
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
