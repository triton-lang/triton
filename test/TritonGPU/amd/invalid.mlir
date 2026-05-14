// RUN: triton-opt --split-input-file %s --verify-diagnostics

// expected-error @+1 {{WMMA version must be in the [1, 3] range}}
#wmma = #ttg.amd_wmma<{version = 0, isTranspose = false, ctaLayout = {warp = [[0, 1], [1, 0]]}}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
    tt.func public @fn(%arg0: !tt.ptr<i32>) {
        %t = tt.splat %arg0 : !tt.ptr<i32,1> -> tensor<32x32x!tt.ptr<i32,1>, #wmma>
        tt.return
    }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [1, 0], [2, 0]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_output_encoding(%arg0: tensor<32x32xf16, #blocked>) {
// expected-error-re @+15 {{Expect output layout to be transposed per thread:{{.*}}- register=1 -> (1, 0){{.*}}register=2 -> (2, 0){{.*}}register=4 -> (0, 1){{.*}}register=8 -> (0, 2)}}
// Full expected layout is following:
// - register=1 -> (1, 0)
//   register=2 -> (2, 0)
//   register=4 -> (0, 1)
//   register=8 -> (0, 2)}}
// - lane=1 -> (0, 4)
//   lane=2 -> (0, 8)
//   lane=4 -> (0, 16)
//   lane=8 -> (4, 0)
//   lane=16 -> (8, 0)
//   lane=32 -> (16, 0)
// - warp is a size 1 dimension
// - block is a size 1 dimension
// where out dims are: [dim0 (size 32), dim1 (size 32)]
    %0 = amdg.in_thread_transpose %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#mfma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [4, 1], instrShape = [16, 16, 16], isTransposed = true}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_input_encoding(%arg0: tensor<32x32xf16, #mfma>) {
// expected-error @+1 {{Expect input tensor in Blocked encoding}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<32x32xf16, #mfma> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_shape(%arg0: tensor<64x64xf16, #blocked>) {
// expected-error @+1 {{Expect equal input and output shapes}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<64x64xf16, #blocked> -> tensor<32x32xf16, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [2, 0], [0, 1], [0, 2]], lane = [[0, 4], [0, 8], [0, 16], [4, 0], [8, 0], [16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_wrong_dtype(%arg0: tensor<32x32xf16, #blocked>) {
// expected-error @+1 {{Expect input and output tensor to have same dtype}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<32x32xf16, #blocked> -> tensor<32x32xf32, #linear>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4, 4], threadsPerWarp = [1, 8, 8], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#linear = #ttg.linear<{register = [[0, 1, 0], [0, 2, 0], [0, 0, 1], [0, 0, 2]], lane = [[0, 0, 4], [0, 0, 8], [0, 0, 16], [0, 4, 0], [0, 8, 0], [0, 16, 0]], warp = [], block = []}>
module attributes {"ttg.target" = "hip:gfx942", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @amd_in_thread_transpose_3d_shape(%arg0: tensor<2x32x32xf16, #blocked>) {
// expected-error @+1 {{Expect 2d tensor}}
    %0 = amdg.in_thread_transpose %arg0 : tensor<2x32x32xf16, #blocked> -> tensor<2x32x32xf16, #linear>
    tt.return
  }
}

// -----

#mma32 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @local_load_packed_tranposed_wrong_op_idx(%arg0: !ttg.memdesc<16x64xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<64x16xi8, #shared1, #smem, mutable>) {
// expected-error @+1 {{Order of dimensions don't match expected}}
    %1 = amdg.local_load_packed_tranposed %arg0 : !ttg.memdesc<16x64xi8, #shared, #smem, mutable> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 1, parent = #mma32, kWidth = 16}>>
    tt.return
  }

  tt.func @local_load_packed_tranposed_wrong_op_idx2(%arg0: !ttg.memdesc<64x16xi8, #shared, #smem, mutable>) {
// expected-error @+1 {{Input and output dimensions don't match after packing changes}}
    %1 = amdg.local_load_packed_tranposed %arg0 : !ttg.memdesc<64x16xi8, #shared, #smem, mutable> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma32, kWidth = 16}>>
    tt.return
  }
  //  CHECK-LABEL: ds_transpose_t_fp4_mfma16
  tt.func @local_load_packed_tranposed_wrong_shape(%arg0: !ttg.memdesc<8x128xi8, #shared, #smem, mutable>, %arg1: !ttg.memdesc<128x8xi8, #shared1, #smem, mutable>) {
// expected-error @+1 {{only works with DotOperandEncodingAttr dst encoding}}
    %1 = amdg.local_load_packed_tranposed %arg0 : !ttg.memdesc<8x128xi8, #shared, #smem, mutable> -> tensor<256x128xi32, #blocked>
    tt.return
  }

}

// -----

#wmma_v3 = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#wmma_v2 = #ttg.amd_wmma<{version = 2, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 32]}>
#wmma_diff_warp = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [0, 0]]}, instrShape = [16, 16, 32]}>
#wmma_diff_shape = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 64]}>
#wmma_diff_transpose = #ttg.amd_wmma<{version = 3, ctaLayout = {warp = [[0, 1], [1, 0]]}, instrShape = [16, 16, 64], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_dot_incompatible_versions(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_v2, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_v2, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }

  tt.func @wmma_dot_incompatible_warp_layouts(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_warp, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_warp, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }

  tt.func @wmma_dot_incomptible_shapes(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_shape, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_shape, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }

  tt.func @wmma_dot_incomptible_transpose(
              %arg0: tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>>,
              %arg1: tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_transpose, kWidth = 8}>>,
              %dst: tensor<16x16xf32, #wmma_v3>
  ) {
    // expected-error @+2 {{'tt.dot' op failed to infer returned types}}
    // expected-error @+1 {{Incompatible parent encoding}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_v3, kWidth = 8}>> * tensor<16x16xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_diff_transpose, kWidth = 8}>> -> tensor<16x16xf32, #wmma_v3>
    tt.return
  }
}

// -----

#wmma_acc = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, CGALayout = [[1, 0], [0, 1]], instrShape = [16, 16, 32]}>
#wmma_dim1 = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, CGALayout = [[1, 0], [0, 0]], instrShape = [16, 16, 32]}>
#wmma_dim2 = #ttg.amd_wmma<{version = 3, isTranspose = true, ctaLayout = {warp = [[0, 1], [1, 0]]}, CGALayout = [[0, 0], [0, 1]], instrShape = [16, 16, 32]}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wmma_invalid_cga_split_operand_0(
              %arg0: tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>>,
              %arg1: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>>,
              %dst: tensor<32x32xf32, #wmma_acc>
  ) {
    // expected-error @+1 {{Incompatible CGA layout for operand 0}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>> -> tensor<32x32xf32, #wmma_acc>
    tt.return
  }

  tt.func @wmma_invalid_cga_split_operand_1(
              %arg0: tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim1, kWidth = 8}>>,
              %arg1: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim1, kWidth = 8}>>,
              %dst: tensor<32x32xf32, #wmma_acc>
  ) {
    // expected-error @+1 {{Incompatible CGA layout for operand 1}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim1, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim1, kWidth = 8}>> -> tensor<32x32xf32, #wmma_acc>
    tt.return
  }

  tt.func @wmma_invalid_cga_split_accumulator(
              %arg0: tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>>,
              %arg1: tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>>,
              %dst: tensor<32x32xf32, #wmma_dim1>
  ) {
    // expected-error @+1 {{Accumulator CGA layout should not broadcast or have repeated rows}}
    %0 = tt.dot %arg0, %arg1, %dst : tensor<32x32xf16, #ttg.dot_op<{opIdx = 0, parent = #wmma_dim2, kWidth = 8}>> * tensor<32x32xf16, #ttg.dot_op<{opIdx = 1, parent = #wmma_dim2, kWidth = 8}>> -> tensor<32x32xf32, #wmma_dim1>
    tt.return
  }
}

// -----

#shared_32 = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [128, 64]}>
#shared_2_intervals = #ttg.padded_shared<[64:+4, 128:+4] {order = [1, 0], shape = [128, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @interval_not_matching_innermost_block_dimension(
    %tensorDesc: !tt.tensordesc<128x64xf16>,
    %memDesc: !ttg.memdesc<128x64xf16, #shared_32, #smem, mutable>
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{TDM store padding is only supported when padding interval equals the innermost block dimension}}
    amdg.async_tdm_copy_local_to_global %tensorDesc[%c0_i32, %c0_i32] from %memDesc: !ttg.memdesc<128x64xf16, #shared_32, #smem, mutable> -> !tt.tensordesc<128x64xf16>
    tt.return
  }

  tt.func public @tdm_store_two_padding_intervals(
    %tensorDesc: !tt.tensordesc<128x64xf16>,
    %memDesc: !ttg.memdesc<128x64xf16, #shared_2_intervals, #smem, mutable>
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{TDM store only supports single interval paddings}}
    amdg.async_tdm_copy_local_to_global %tensorDesc[%c0_i32, %c0_i32] from %memDesc: !ttg.memdesc<128x64xf16, #shared_2_intervals, #smem, mutable> -> !tt.tensordesc<128x64xf16>
    tt.return
  }
}

// -----

// Gather with an index layout that distributes values across lanes (invalid).
// parent blocked: threadsPerWarp = [32, 1] → lanes map to dim 0.
// slice dim 1 → 1D tensor where each lane holds a different value.
#blocked_lane_dist = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#slice_lane_dist = #ttg.slice<{dim = 1, parent = #blocked_lane_dist}>
#shared_gather = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_gather = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @tdm_gather_invalid_lane_distribution(
    %memDesc: !ttg.memdesc<32x128xf16, #shared_gather, #smem_gather, mutable>,
    %tensorDesc: !tt.tensordesc<32x128xf16>,
    %row_indices: tensor<32xi32, #slice_lane_dist>,
    %pred: i32
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{index layout distributes values across lanes}}
    %token = amdg.async_tdm_gather %tensorDesc[%row_indices, %c0_i32] to %memDesc, pred = %pred : tensor<32xi32, #slice_lane_dist>, !ttg.memdesc<32x128xf16, #shared_gather, #smem_gather, mutable> -> !tt.tensordesc<32x128xf16>
    tt.return
  }
}

// -----

// Scatter with padded shared layout where padding interval != innermost block dimension.
#shared_scatter_32 = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [8, 64]}>
// Scatter with two padding intervals (only single interval is supported).
#shared_scatter_2_intervals = #ttg.padded_shared<[64:+4, 128:+4] {order = [1, 0], shape = [8, 64]}>
#smem_scatter = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @scatter_interval_not_matching_innermost_block_dimension(
    %tensorDesc: !tt.tensordesc<8x64xf16>,
    %memDesc: !ttg.memdesc<8x64xf16, #shared_scatter_32, #smem_scatter, mutable>,
    %row_indices: tensor<8xi32>
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{TDM scatter padding is only supported when padding interval equals the innermost block dimension}}
    amdg.async_tdm_scatter %tensorDesc[%row_indices, %c0_i32] from %memDesc : tensor<8xi32>, !ttg.memdesc<8x64xf16, #shared_scatter_32, #smem_scatter, mutable> -> !tt.tensordesc<8x64xf16>
    tt.return
  }

  tt.func public @scatter_two_padding_intervals(
    %tensorDesc: !tt.tensordesc<8x64xf16>,
    %memDesc: !ttg.memdesc<8x64xf16, #shared_scatter_2_intervals, #smem_scatter, mutable>,
    %row_indices: tensor<8xi32>
  ) {
    %c0_i32 = arith.constant 0 : i32
    // expected-error @+1 {{TDM scatter only supports single interval paddings}}
    amdg.async_tdm_scatter %tensorDesc[%row_indices, %c0_i32] from %memDesc : tensor<8xi32>, !ttg.memdesc<8x64xf16, #shared_scatter_2_intervals, #smem_scatter, mutable> -> !tt.tensordesc<8x64xf16>
    tt.return
  }
}

// -----

// warp_used_hint validation tests
#shared_wb = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem_wb = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  // hint == 0 has no active warps; rejected.
  tt.func @warp_used_hint_zero(
    %tensorDesc: !tt.tensordesc<256x64xf16>,
    %memDesc: !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{warp_used_hint must have at least one bit set}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc[%c0, %c0] into %memDesc, pred = %pred {warp_used_hint = 0 : i32} : !tt.tensordesc<256x64xf16> -> !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>
    tt.return
  }

  // 0x69 (warps 0,3,5,6) is rejected: K=4 is a power of two but the
  // active set spans 3 warpId bit positions, not log2(K) = 2 -- a
  // non axis-aligned pattern is not supported.
  tt.func @warp_used_hint_non_axis_aligned(
    %tensorDesc: !tt.tensordesc<256x64xf16>,
    %memDesc: !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{is not axis-aligned}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc[%c0, %c0] into %memDesc, pred = %pred {warp_used_hint = 105 : i32} : !tt.tensordesc<256x64xf16> -> !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>
    tt.return
  }

  // popcount must be a power of two.  0x07 has K=3 -- rejected even
  // though warps 0..2 are otherwise contiguous.
  tt.func @warp_used_hint_non_pow2_k(
    %tensorDesc: !tt.tensordesc<256x64xf16>,
    %memDesc: !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{popcount(warp_used_hint) = 3 must be a power of two}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc[%c0, %c0] into %memDesc, pred = %pred {warp_used_hint = 7 : i32} : !tt.tensordesc<256x64xf16> -> !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>
    tt.return
  }

  // hint sets all 16 low bits but num_warps = 8 so bits 8..15 don't
  // correspond to any warp.  Reported by the bits-beyond check.
  tt.func @warp_used_hint_exceeds_num_warps(
    %tensorDesc: !tt.tensordesc<256x64xf16>,
    %memDesc: !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{warp_used_hint = 0xffff sets bits beyond num_warps = 8}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc[%c0, %c0] into %memDesc, pred = %pred {warp_used_hint = 65535 : i32} : !tt.tensordesc<256x64xf16> -> !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>
    tt.return
  }

  // Bits outside [0, num_warps) must be zero.  K=2 is otherwise valid,
  // but warp index 9 is not in [0, 8).
  tt.func @warp_used_hint_bits_beyond_num_warps(
    %tensorDesc: !tt.tensordesc<256x64xf16>,
    %memDesc: !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{sets bits beyond num_warps = 8}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc[%c0, %c0] into %memDesc, pred = %pred {warp_used_hint = 513 : i32} : !tt.tensordesc<256x64xf16> -> !ttg.memdesc<256x64xf16, #shared_wb, #smem_wb, mutable>
    tt.return
  }
}

// -----

#fp4_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#fp4_dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#fp4_scale_bad = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_upcast_fp4_incompatible_scale_encoding(%src: tensor<16x32xi8, #fp4_src>, %scale: tensor<16x64xbf16, #fp4_scale_bad>) {
    // expected-error @+1 {{scale and output encodings are not compatible}}
    %0 = amdg.scaled_upcast_fp4 %src scale %scale {axis = 1 : i32} : tensor<16x32xi8, #fp4_src>, tensor<16x64xbf16, #fp4_scale_bad> -> tensor<16x64xbf16, #fp4_dst>
    tt.return
  }
}

// -----

// Partitioned encoding requires K to be a multiple of numLogicalPieces
// (= numPartitions*numGroups = 4) so the hinted copy fits in a single
// TDM instruction.  Here K=2 < numLogicalPieces=4 is rejected.
#shared_inner_mi = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned_mi = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #shared_inner_mi}>
#smem_mi = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @warp_used_hint_partitioned_insufficient(
    %tensorDesc: !tt.tensordesc<128x16xf16>,
    %memDesc: !ttg.memdesc<128x16xf16, #partitioned_mi, #smem_mi, mutable>,
    %pred: i32
  ) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{warp_used_hint with a partitioned shared encoding must select K active warps}}
    %0 = amdg.async_tdm_copy_global_to_local %tensorDesc[%c0, %c0] into %memDesc, pred = %pred {warp_used_hint = 3 : i32} : !tt.tensordesc<128x16xf16> -> !ttg.memdesc<128x16xf16, #partitioned_mi, #smem_mi, mutable>
    tt.return
  }
}

// -----

#fp4_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#fp4_dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#fp4_dst_bad = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_upcast_fp4_incompatible_src_encoding(%src: tensor<16x32xi8, #fp4_src>, %scale: tensor<16x64xbf16, #fp4_dst_bad>) {
    // expected-error @+1 {{Src and Dst encodings are not compatible}}
    %0 = amdg.scaled_upcast_fp4 %src scale %scale {axis = 1 : i32} : tensor<16x32xi8, #fp4_src>, tensor<16x64xbf16, #fp4_dst_bad> -> tensor<16x64xbf16, #fp4_dst_bad>
    tt.return
  }
}

// -----

#fp4_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#fp4_dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_upcast_fp4_invalid_result_type(%src: tensor<16x32xi8, #fp4_src>, %scale: tensor<16x64xbf16, #fp4_dst>) {
    // expected-error @+1 {{must be ranked tensor of 16-bit float or bfloat16 type values}}
    %0 = amdg.scaled_upcast_fp4 %src scale %scale {axis = 1 : i32} : tensor<16x32xi8, #fp4_src>, tensor<16x64xbf16, #fp4_dst> -> tensor<16x64xf32, #fp4_dst>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @scaled_upcast_fp8_invalid_result_type(%src: tensor<16x64xf8E4M3FN, #blocked>, %scale: tensor<16x64xbf16, #blocked>) {
    // expected-error @+1 {{must be ranked tensor of 16-bit float or bfloat16 type values}}
    %0 = amdg.scaled_upcast_fp8 %src scale %scale : tensor<16x64xf8E4M3FN, #blocked>, tensor<16x64xbf16, #blocked> -> tensor<16x64xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @update_tensor_descriptor_wrong_offset_count(
    %desc: !tt.tensordesc<64x64xf16, #shared>, %dx: i32
  ) -> !tt.tensordesc<64x64xf16, #shared> {
    // expected-error @+1 {{expected 2 add_offsets to match descriptor rank, got 1}}
    %result = amdg.update_tensor_descriptor %desc add_offsets = [%dx] : !tt.tensordesc<64x64xf16, #shared>
    tt.return %result : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @update_tensor_descriptor_wrong_bounds_count(
    %desc: !tt.tensordesc<64x64xf16, #shared>, %m: i32, %n: i32, %k: i32
  ) -> !tt.tensordesc<64x64xf16, #shared> {
    // expected-error @+1 {{expected 2 set_bounds to match descriptor rank, got 3}}
    %result = amdg.update_tensor_descriptor %desc set_bounds = [%m, %n, %k] : !tt.tensordesc<64x64xf16, #shared>
    tt.return %result : !tt.tensordesc<64x64xf16, #shared>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @update_tensor_descriptor_no_kwargs(
    %desc: !tt.tensordesc<64x64xf16, #shared>
  ) -> !tt.tensordesc<64x64xf16, #shared> {
    // expected-error @+1 {{must provide at least one of add_offsets, set_bounds, dest, pred, or barrier}}
    %result = amdg.update_tensor_descriptor %desc : !tt.tensordesc<64x64xf16, #shared>
    tt.return %result : !tt.tensordesc<64x64xf16, #shared>
  }
}
