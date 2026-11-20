// RUN: triton-opt %s -split-input-file -tritonamdgpu-schedule-loops="num_stages=2" -tritonamdgpu-pipeline="use_async_copy=1" -canonicalize | FileCheck %s

#blocked1 = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #shared = {{.*}}vec = 1, {{.*}} order = [1, 0]
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 4], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy_shared_vec2_clamp_to_vec1
  tt.func @async_copy_shared_vec2_clamp_to_vec1(%arg0: tensor<16x32x!tt.ptr<f32>, #blocked1> {tt.contiguity = dense<[1, 2]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>},
                %arg1: tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<16x32xf32, #mma> {
    // CHECK: ttg.async_copy_global_to_local {{.*}} -> <16x32xf32, #shared, #smem, mutable>
    %cst = arith.constant dense<32> : tensor<16x32xi32, #blocked1>
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<16x32xf32, #mma>) : i32 {
      %a = tt.load %arg0 : tensor<16x32x!tt.ptr<f32>, #blocked1>
      %a_dot = ttg.convert_layout %a : tensor<16x32xf32, #blocked1> -> tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %c = tt.dot %a_dot, %arg1, %acc : tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x32xf32, #mma>
      scf.yield %c : tensor<16x32xf32, #mma>
    }
    tt.return %result : tensor<16x32xf32, #mma>
  }
}

// -----

// Test with #blocked layout (sizePerThread = [1, 1]) for the 32x32 load
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #shared = {{.*}} order = [1, 0]
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [16, 16, 4], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy_shared_layout_vec1_order
  tt.func @async_copy_shared_layout_vec1_order(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>},
                %arg1: tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>,
                %lb: i32, %ub: i32, %step: i32) -> tensor<16x32xf32, #mma> {
    // CHECK: ttg.async_copy_global_to_local {{.*}} -> <32x32xf32, #shared, #smem, mutable>
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<16x32xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<16x32xf32, #mma>) : i32 {
      %b = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %c = tt.dot %arg1, %b_dot, %acc : tensor<16x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x32xf32, #mma>
      scf.yield %c : tensor<16x32xf32, #mma>
    }
    tt.return %result : tensor<16x32xf32, #mma>
  }
}

// -----

// The CTA tile is large enough for one 32-bit write per lane even though the
// initial layout broadcasts lanes along dim0. CoalesceAsyncCopy can rewrite the
// source layout later, so the pipeliner should still select async copy.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy_large_cta_tile_with_lane_broadcast
  tt.func @async_copy_large_cta_tile_with_lane_broadcast(
              %arg0: tensor<32x16x!tt.ptr<f32>, #blocked> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>},
              %arg1: tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>,
              %lb: i32, %ub: i32, %step: i32) -> tensor<32x16xf32, #mma> {
    // CHECK: ttg.async_copy_global_to_local {{.*}} -> <32x16xf32
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<32x16xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<32x16xf32, #mma>) : i32 {
      %a = tt.load %arg0 : tensor<32x16x!tt.ptr<f32>, #blocked>
      %a_dot = ttg.convert_layout %a : tensor<32x16xf32, #blocked> -> tensor<32x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %c = tt.dot %a_dot, %arg1, %acc : tensor<32x16xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x16xf32, #mma>
      scf.yield %c : tensor<32x16xf32, #mma>
    }
    tt.return %result : tensor<32x16xf32, #mma>
  }
}

// -----

// For sizePerThrad=[1, 1] threadsPerWarp=[1, 64] order=[1, 0] and dim1=64 the registers will be contigious along dim0 which is the *non* contig dimension.
// Check that we correctly follow the memory order which will be the lane order instead of the register order.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: #shared = {{.*}} order = [1, 0]
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy_lanes_cover_contig_dim
  tt.func @async_copy_lanes_cover_contig_dim(
              %arg0: tensor<16x64x!tt.ptr<f32>, #blocked> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>},
              %arg1: tensor<64x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>,
              %lb: i32, %ub: i32, %step: i32) -> tensor<16x16xf32, #mma> {
    // CHECK: ttg.async_copy_global_to_local {{.*}} -> <16x64xf32, #shared, #smem, mutable>
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<16x16xf32, #mma>) : i32 {
      %a = tt.load %arg0 : tensor<16x64x!tt.ptr<f32>, #blocked>
      %a_dot = ttg.convert_layout %a : tensor<16x64xf32, #blocked> -> tensor<16x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %c = tt.dot %a_dot, %arg1, %acc : tensor<16x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x16xf32, #mma>
      scf.yield %c : tensor<16x16xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return %result : tensor<16x16xf32, #mma>
  }
}

// -----

// If sizePerThread is > 1 in the non contig dim we still need to choose the actual memory order.
#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 1], order = [1, 0]}>
// CHECK: #shared = {{.*}} order = [1, 0]
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [1, 1], instrShape = [16, 16, 4], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy_sizeperthread_in_noncontig_dim_not_vectorized
  tt.func @async_copy_sizeperthread_in_noncontig_dim_not_vectorized(
              %arg0: tensor<16x64x!tt.ptr<f32>, #blocked> {tt.contiguity = dense<[1, 1]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>},
              %arg1: tensor<64x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>,
              %lb: i32, %ub: i32, %step: i32) -> tensor<16x16xf32, #mma> {
    // CHECK: ttg.async_copy_global_to_local {{.*}} -> <16x64xf32, #shared, #smem, mutable>
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<16x16xf32, #mma>) : i32 {
      %a = tt.load %arg0 : tensor<16x64x!tt.ptr<f32>, #blocked>
      %a_dot = ttg.convert_layout %a : tensor<16x64xf32, #blocked> -> tensor<16x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %c = tt.dot %a_dot, %arg1, %acc : tensor<16x64xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<64x16xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<16x16xf32, #mma>
      scf.yield %c : tensor<16x16xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return %result : tensor<16x16xf32, #mma>
  }
}

// -----

// A preshuffled B operand stored as [K / 16, N * 16]: each 16-byte K slice of
// 16 consecutive N rows is contiguous, so the MFMA operand reads are already
// free of LDS bank conflicts. Keep LDS unswizzled so async copy does not need
// cross-lane address swizzling.
// CHECK: #shared = #ttg.swizzled_shared<{vec = 16, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 16], warpsPerCTA = [2, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [4, 0]], lane = [[0, 16], [0, 32], [0, 64], [0, 128], [1, 0], [2, 0]], warp = [[0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [4, 0, 0]], lane = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [1, 0, 0], [2, 0, 0]], warp = [[0, 0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 4, 0], [0, 8, 0], [4, 0, 0]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 0, 8], [1, 0, 0], [2, 0, 0]], warp = [[0, 0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1], instrShape = [16, 16, 128], isTransposed = true}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: preshuffled_weight_without_lds_bank_conflicts
  tt.func @preshuffled_weight_without_lds_bank_conflicts(
      %ptrs: tensor<8x256x!tt.ptr<i8>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>},
      %a: tensor<32x128xi8, #dotA>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x16xf32, #mma> {
    // CHECK: ttg.async_copy_global_to_local {{.*}} -> <8x256xi8, #shared, #smem, mutable>
    %zero = arith.constant dense<0.0> : tensor<32x16xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero) -> (tensor<32x16xf32, #mma>) : i32 {
      %raw = tt.load %ptrs : tensor<8x256x!tt.ptr<i8>, #blocked>
      %cvt = ttg.convert_layout %raw : tensor<8x256xi8, #blocked> -> tensor<8x256xi8, #linear>
      // [K / 16, N, 16] -> [K / 16, 16, N] -> [K, N]
      %r = tt.reshape %cvt : tensor<8x256xi8, #linear> -> tensor<8x16x16xi8, #linear1>
      %t = tt.trans %r {order = array<i32: 0, 2, 1>} : tensor<8x16x16xi8, #linear1> -> tensor<8x16x16xi8, #linear2>
      %b = tt.reshape %t : tensor<8x16x16xi8, #linear2> -> tensor<128x16xi8, #dotB>
      %dot = tt.dot_scaled %a, %b, %acc lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x128xi8, #dotA> * tensor<128x16xi8, #dotB> -> tensor<32x16xf32, #mma>
      scf.yield %dot : tensor<32x16xf32, #mma>
    }
    tt.return %result : tensor<32x16xf32, #mma>
  }
}

// -----

// A linear consumer layout alone does not justify an unswizzled LDS layout:
// lanes 0-15 read 16-byte K slices from 16 rows of this [N, K] tile, which
// conflict without swizzling.
// CHECK: #shared = #ttg.swizzled_shared<{vec = 16, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [2, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 16], [0, 32]], warp = [[0, 0]], block = []}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 1], instrShape = [16, 16, 128], isTransposed = true}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: linear_weight_with_lds_bank_conflicts
  tt.func @linear_weight_with_lds_bank_conflicts(
      %ptrs: tensor<16x128x!tt.ptr<i8>, #blocked> {tt.contiguity = dense<[1, 16]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>},
      %a: tensor<32x128xi8, #dotA>, %lb: i32, %ub: i32, %step: i32) -> tensor<32x16xf32, #mma> {
    // CHECK: ttg.async_copy_global_to_local {{.*}} -> <16x128xi8, #shared, #smem, mutable>
    %zero = arith.constant dense<0.0> : tensor<32x16xf32, #mma>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %zero) -> (tensor<32x16xf32, #mma>) : i32 {
      %raw = tt.load %ptrs : tensor<16x128x!tt.ptr<i8>, #blocked>
      %cvt = ttg.convert_layout %raw : tensor<16x128xi8, #blocked> -> tensor<16x128xi8, #linear>
      %b = tt.trans %cvt {order = array<i32: 1, 0>} : tensor<16x128xi8, #linear> -> tensor<128x16xi8, #dotB>
      %dot = tt.dot_scaled %a, %b, %acc lhs = e2m1 rhs = e2m1 {fastMath = false} : tensor<32x128xi8, #dotA> * tensor<128x16xi8, #dotB> -> tensor<32x16xf32, #mma>
      scf.yield %dot : tensor<32x16xf32, #mma>
    }
    tt.return %result : tensor<32x16xf32, #mma>
  }
}
