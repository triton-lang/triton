// RUN: triton-opt %s -split-input-file -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline | FileCheck %s

// Positive case: FMA dot, B operand (opIdx = 1) whose
// global load is K-contiguous (order = [0, 1]). The dot reads from LDS strided
// in this orientation, so the shared buffer must be flipped to operand-major
// (N-contiguous, order = [1, 0]).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
#fma = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: #shared = {{.*}}order = [1, 0]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fma_operand_b_kcontig_is_flipped
  tt.func @fma_operand_b_kcontig_is_flipped(
                %argB: tensor<32x32x!tt.ptr<f32>, #blocked>,
                %argA: tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #fma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #fma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<32x32xf32, #fma>) : i32 {
      %b = tt.load %argB : tensor<32x32x!tt.ptr<f32>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>> -> tensor<32x32xf32, #fma>
      scf.yield %c : tensor<32x32xf32, #fma>
    }
    tt.return %result : tensor<32x32xf32, #fma>
  }
}

// -----

// Positive case: FMA dot, A operand (opIdx = 0) whose global load is
// K-contiguous (order = [1, 0]). Flip to operand-major (M-contiguous,
// order = [0, 1]).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
#fma = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: #shared = {{.*}}order = [0, 1]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fma_operand_a_kcontig_is_flipped
  tt.func @fma_operand_a_kcontig_is_flipped(
                %argA: tensor<32x32x!tt.ptr<f32>, #blocked>,
                %argB: tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #fma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #fma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<32x32xf32, #fma>) : i32 {
      %a = tt.load %argA : tensor<32x32x!tt.ptr<f32>, #blocked>
      %a_dot = ttg.convert_layout %a : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>>
      %c = tt.dot %a_dot, %argB, %acc : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>> -> tensor<32x32xf32, #fma>
      scf.yield %c : tensor<32x32xf32, #fma>
    }
    tt.return %result : tensor<32x32xf32, #fma>
  }
}

// -----

// Negative case: MFMA dot, the layout must be left untouched.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = true}>
// CHECK: #shared = {{.*}}order = [0, 1]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: mfma_operand_b_kcontig_is_not_flipped
  tt.func @mfma_operand_b_kcontig_is_not_flipped(
                %argB: tensor<32x32x!tt.ptr<f32>, #blocked>,
                %argA: tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<16x32xf32, #mma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<16x32xf32, #mma>) : i32 {
      %b = tt.load %argB : tensor<32x32x!tt.ptr<f32>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x32xf32, #mma>
      scf.yield %c : tensor<16x32xf32, #mma>
    }
    tt.return %result : tensor<16x32xf32, #mma>
  }
}

// -----

// No-op case: FMA dot, B operand (opIdx = 1) whose global load is already
// operand-major (N-contiguous, order = [1, 0]).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
#fma = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: #shared = {{.*}}order = [1, 0]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: fma_operand_b_already_operand_major
  tt.func @fma_operand_b_already_operand_major(
                %argB: tensor<32x32x!tt.ptr<f32>, #blocked>,
                %argA: tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<32x32xf32, #fma> {
    // CHECK: ttg.local_alloc {{.*}} #shared
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #fma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<32x32xf32, #fma>) : i32 {
      %b = tt.load %argB : tensor<32x32x!tt.ptr<f32>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>>
      %c = tt.dot %argA, %b_dot, %acc : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #fma}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #fma}>> -> tensor<32x32xf32, #fma>
      scf.yield %c : tensor<32x32xf32, #fma>
    }
    tt.return %result : tensor<32x32xf32, #fma>
  }
}
