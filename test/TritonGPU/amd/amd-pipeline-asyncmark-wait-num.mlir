// RUN: triton-opt %s -split-input-file -tritonamdgpu-schedule-loops="num_stages=3" -tritonamdgpu-pipeline="use_async_copy=1" -canonicalize | FileCheck %s

// On asyncmark targets (CDNA3/CDNA4) ttg.async_wait's `num` lowers straight to
// wait.asyncmark(N), so the pipeliner-authored num=0 ("wait for all") would
// serialize the SWP (PR #9883). Verify the pipeline pass runs updateWaits and
// rewrites the steady-state wait to a non-zero commit-group count.

#blocked = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 4], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_wait_num_stages3
  // Single load, num_stages=3 -> 1 commit allowed in flight.
  // CHECK: scf.for
  // CHECK:   ttg.async_wait %{{.*}} {num = 1 : i32}
  // CHECK:   scf.yield
  tt.func @async_wait_num_stages3(
      %arg0: tensor<16x32x!tt.ptr<f32>, #blocked> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>},
      %arg1: tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>,
      %lb: i32, %ub: i32, %step: i32) -> tensor<16x32xf32, #mma> {
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<16x32xf32, #mma>) : i32 {
      %a = tt.load %arg0 : tensor<16x32x!tt.ptr<f32>, #blocked>
      %a_dot = ttg.convert_layout %a : tensor<16x32xf32, #blocked> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %c = tt.dot %a_dot, %arg1, %acc : tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x32xf32, #mma>
      scf.yield %c : tensor<16x32xf32, #mma>
    }
    tt.return %result : tensor<16x32xf32, #mma>
  }
}

// -----

// Two loads (gemm-shaped) at num_stages=3: each iteration emits two commit
// groups, the steady-state multi-token wait must allow both in flight, so
// updateWaits should derive num=2.

#blockedA = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 4], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: gemm_two_loads_stages3
  // CHECK: scf.for
  // CHECK:   ttg.async_wait %{{[^,]+}}, %{{[^,]+}} {num = 2 : i32}
  // CHECK:   scf.yield
  tt.func @gemm_two_loads_stages3(
      %argA: tensor<16x32x!tt.ptr<f32>, #blockedA> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>},
      %argB: tensor<32x32x!tt.ptr<f32>, #blockedB> {tt.contiguity = dense<[2, 1]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>},
      %lb: i32, %ub: i32, %step: i32) -> tensor<16x32xf32, #mma> {
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<16x32xf32, #mma>) : i32 {
      %a = tt.load %argA : tensor<16x32x!tt.ptr<f32>, #blockedA>
      %b = tt.load %argB : tensor<32x32x!tt.ptr<f32>, #blockedB>
      %a_dot = ttg.convert_layout %a : tensor<16x32xf32, #blockedA> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %b_dot = ttg.convert_layout %b : tensor<32x32xf32, #blockedB> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %c = tt.dot %a_dot, %b_dot, %acc : tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x32xf32, #mma>
      scf.yield %c : tensor<16x32xf32, #mma>
    }
    tt.return %result : tensor<16x32xf32, #mma>
  }
}
