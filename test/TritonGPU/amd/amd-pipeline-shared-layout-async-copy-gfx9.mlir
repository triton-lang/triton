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

// On GFX9 the swizzle is realized as a lane shuffle. Lanes are 4 vec blocks
// apart here, so the maxPhase=8 XOR leaves the warp and must be clamped.
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 1, order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy_swizzle_clamped_to_warp
  tt.func @async_copy_swizzle_clamped_to_warp(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
              %arg1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>,
              %lb: i32, %ub: i32, %step: i32) -> tensor<128x16xf32, #mma> {
    // CHECK: ttg.async_copy_global_to_local {{.*}} -> <64x16xf16, #shared, #smem, mutable>
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.broadcast %0 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %4 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %5 = tt.addptr %3, %4 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<128x16xf32, #mma>) : i32 {
      %b = tt.load %5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %c = tt.dot %arg1, %b_dot, %acc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      scf.yield %c : tensor<128x16xf32, #mma>
    }
    tt.return %result : tensor<128x16xf32, #mma>
  }
}

// -----

// Same deduced swizzle as above, but each thread holds exactly one vec block
// so the shuffle stays inside the warp and maxPhase must be preserved.
#blocked = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: #shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32, 16], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_copy_swizzle_stays_in_warp
  tt.func @async_copy_swizzle_stays_in_warp(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
              %arg1: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>,
              %lb: i32, %ub: i32, %step: i32) -> tensor<128x16xf32, #mma> {
    // CHECK: ttg.async_copy_global_to_local {{.*}} -> <64x16xf16, #shared, #smem, mutable>
    %cst_acc = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #mma>
    %0 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<1x16x!tt.ptr<f16>, #blocked>
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %3 = tt.broadcast %0 : tensor<1x16x!tt.ptr<f16>, #blocked> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %4 = tt.broadcast %2 : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %5 = tt.addptr %3, %4 : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%acc = %cst_acc) -> (tensor<128x16xf32, #mma>) : i32 {
      %b = tt.load %5 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %b_dot = ttg.convert_layout %b : tensor<64x16xf16, #blocked> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %c = tt.dot %arg1, %b_dot, %acc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      scf.yield %c : tensor<128x16xf32, #mma>
    }
    tt.return %result : tensor<128x16xf32, #mma>
  }
}
