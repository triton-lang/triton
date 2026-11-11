// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 matrix-instruction-size=16" | FileCheck %s --check-prefixes MFMA16,CHECK
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx942 matrix-instruction-size=32" | FileCheck %s --check-prefixes MFMA32,CHECK
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx950 matrix-instruction-size=32" | FileCheck %s --check-prefixes CHECK-GFX950
// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="gfx-arch=gfx950 matrix-instruction-size=16" | FileCheck %s --check-prefixes CHECK-GFX950

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

// Cross-iteration chain-dot: in pipelined FA the QK dot result is yielded and
// consumed as operand A of the PV dot on the next iteration.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK-GFX950-LABEL: cross_iter_chain_dot_fa
// CHECK-GFX950: tt.dot {{.*}} : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
// CHECK-GFX950: tt.dot {{.*}} -> tensor<128x64xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @cross_iter_chain_dot_fa(
      %q: tensor<128x128xf16, #dotOp0>,
      %k: tensor<128x64xf16, #dotOp1>,
      %v: tensor<64x128xf16, #dotOp1>,
      %acc_init: tensor<128x128xf32, #blocked>,
      %p_init: tensor<128x64xf16, #dotOp0>,
      %lb: index, %ub: index, %step: index) -> tensor<128x128xf32, #blocked> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #blocked>
    %result:2 = scf.for %iv = %lb to %ub step %step
        iter_args(%acc = %acc_init, %p_prev = %p_init) -> (tensor<128x128xf32, #blocked>, tensor<128x64xf16, #dotOp0>) {
      %pv = tt.dot %p_prev, %v, %acc : tensor<128x64xf16, #dotOp0> * tensor<64x128xf16, #dotOp1> -> tensor<128x128xf32, #blocked>
      %qk = tt.dot %q, %k, %cst : tensor<128x128xf16, #dotOp0> * tensor<128x64xf16, #dotOp1> -> tensor<128x64xf32, #blocked>
      %qk_f16 = arith.truncf %qk : tensor<128x64xf32, #blocked> to tensor<128x64xf16, #blocked>
      %p_next = ttg.convert_layout %qk_f16 : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #dotOp0>
      scf.yield %pv, %p_next : tensor<128x128xf32, #blocked>, tensor<128x64xf16, #dotOp0>
    }
    tt.return %result#0 : tensor<128x128xf32, #blocked>
  }
}

// -----

// Second dot inside an scf.if: same layouts as mfma_chain_dot_BM64.
// BLOCK_M = 64 because at 128 the generic layout is [4, 1] too.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
// MFMA32{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
// CHECK-LABEL: chain_dot_scf_if_BM64
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<64x16xf32, #mma>
// MFMA16: tt.dot {{.*}} : {{.*}} -> tensor<64x128xf32, #mma>
// MFMA32: tt.dot {{.*}} : {{.*}} -> tensor<64x128xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @chain_dot_scf_if_BM64(
      %q: tensor<64x128xf16, #dotOp0>,
      %k: tensor<128x16xf16, #dotOp1>,
      %v: tensor<16x128xf16, #dotOp1>,
      %cond: i1,
      %o_ptr: tensor<64x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<64x128xf16, #dotOp0> * tensor<128x16xf16, #dotOp1> -> tensor<64x16xf32, #blocked>
    %o = scf.if %cond -> (tensor<64x128xf32, #blocked>) {
      %qk_f16 = arith.truncf %qk : tensor<64x16xf32, #blocked> to tensor<64x16xf16, #blocked>
      %p = ttg.convert_layout %qk_f16 : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #dotOp0>
      %d = tt.dot %p, %v, %cst1 : tensor<64x16xf16, #dotOp0> * tensor<16x128xf16, #dotOp1> -> tensor<64x128xf32, #blocked>
      scf.yield %d : tensor<64x128xf32, #blocked>
    } else {
      scf.yield %cst1 : tensor<64x128xf32, #blocked>
    }
    tt.store %o_ptr, %o : tensor<64x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// First dot feeds only the scf.if condition, so this is not a chain. A false
// match would give the first dot [4, 1]; only mfma32 tells, as [2, 2] otherwise.
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// MFMA16{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [32, 32, 8], isTransposed = true}>
// MFMA16{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [16, 16, 16], isTransposed = true}>
// MFMA32{LITERAL}: #mma1 = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [32, 32, 8], isTransposed = true}>
// CHECK-LABEL: chain_dot_cond_only_BM64
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<64x16xf32, #mma>
// CHECK: tt.dot {{.*}} : {{.*}} -> tensor<64x128xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @chain_dot_cond_only_BM64(
      %q: tensor<64x128xf16, #dotOp0>,
      %k: tensor<128x16xf16, #dotOp1>,
      %a2: tensor<64x16xf16, #dotOp0>,
      %v: tensor<16x128xf16, #dotOp1>,
      %o_ptr: tensor<64x128x!tt.ptr<f32>, #blocked>) {
    %c0 = arith.constant 0.000000e+00 : f32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    %cst1 = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %qk = tt.dot %q, %k, %cst : tensor<64x128xf16, #dotOp0> * tensor<128x16xf16, #dotOp1> -> tensor<64x16xf32, #blocked>
    %r0 = "tt.reduce"(%qk) <{axis = 1 : i32}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %m = arith.maxnumf %arg0, %arg1 : f32
      tt.reduce.return %m : f32
    }) : (tensor<64x16xf32, #blocked>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r1 = "tt.reduce"(%r0) <{axis = 0 : i32}> ({
    ^bb0(%arg0: f32, %arg1: f32):
      %m = arith.maxnumf %arg0, %arg1 : f32
      tt.reduce.return %m : f32
    }) : (tensor<64xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> f32
    %cond = arith.cmpf ogt, %r1, %c0 : f32
    %o = scf.if %cond -> (tensor<64x128xf32, #blocked>) {
      %d = tt.dot %a2, %v, %cst1 : tensor<64x16xf16, #dotOp0> * tensor<16x128xf16, #dotOp1> -> tensor<64x128xf32, #blocked>
      scf.yield %d : tensor<64x128xf32, #blocked>
    } else {
      scf.yield %cst1 : tensor<64x128xf32, #blocked>
    }
    tt.store %o_ptr, %o : tensor<64x128x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
