// RUN: triton-opt --split-input-file %s --verify-diagnostics

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [2, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#smem = #ttg.shared_memory
tt.func public @non_trivial_block(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{non-trivial block}}
    %a = ttg.memdesc_subslice %arg0 [0, 0] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<8x8xf32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @miss_encoding(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{,}}
    %a = ttg.memdesc_subslice %arg0 [0, 0] : !ttg.memdesc<8x16xf32> -> !ttg.memdesc<8x16xf16>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @miss_memory_space(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{,}}
    %a = ttg.memdesc_subslice %arg0 [0, 0] : !ttg.memdesc<8x16xf32, #shared> -> !ttg.memdesc<8x16xf16>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @subview_element_ty(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{element type}}
    %a = ttg.memdesc_subslice %arg0 [0, 0] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<8x16xf16, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @too_many_offsets(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{offsets}}
    %a = ttg.memdesc_subslice %arg0 [0, 0, 0] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<8x16xf32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @too_few_offsets(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    // expected-error @+1 {{offsets}}
    %a = ttg.memdesc_subslice %arg0 [0] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<8x16xf32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @result_rank_too_large(%arg0: !ttg.memdesc<3x8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result rank}}
    %a = ttg.memdesc_index %arg0[%zero] : !ttg.memdesc<3x8x16xf32, #shared, #smem> -> !ttg.memdesc<3x8x16xf32, #shared, #smem>
    tt.return
}
// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0]}>
#smem = #ttg.shared_memory
tt.func public @result_1d_to_1d(%arg0: !ttg.memdesc<8xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result rank}}
    %a = ttg.memdesc_index %arg0[%zero] : !ttg.memdesc<8xf32, #shared, #smem> -> !ttg.memdesc<2xf32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @subview_along_swizzling(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{swizzling pattern}}
    %a = ttg.memdesc_subslice %arg0 [0, 0] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<8x4xf32, #shared, #smem>
    tt.return
}


// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @subview_along_swizzling(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>, %index: i32) {
    // expected-error @+1 {{tile}}
    %a = ttg.memdesc_subslice %arg0 [2, 0] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<4x16xf32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @result_dim_too_large(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result shape}}
    %a = ttg.memdesc_index %arg0[%zero] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<32xf32, #shared, #smem>
    tt.return
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf32, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{element types of operands A and B must have same bit width}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf32, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{mismatching encoding between A and B operands}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32>) {
    // expected-error@+1 {{miss encoding of C operand}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32>
    tt.return
  }
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1], instrShape = [16, 8]}>
#dot_operand_a = #ttg.dot_op<{opIdx=0, parent=#mma0, kWidth=1}>
#dot_operand_b = #ttg.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"ttg.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{mismatching kWidth between A and B operands}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

tt.func @warp_specialize_no_holder() {
  // expected-error @below {{'ttg.warp_specialize' op expected to find only a `ttg.warp_specialize.partitions` op inside its second region}}
  "ttg.warp_specialize"() ({
    "ttg.warp_yield"() : () -> ()
  }, {
    "ttg.warp_yield"() : () -> ()
  }) {partitionNumWarps = array<i32>} : () -> ()
  tt.return
}

// -----

tt.func @warp_specialize_mismatch_partition_count() {
  // expected-error @below {{'ttg.warp_specialize' op has 0 partitions but `partitionNumWarps` has 1 elements}}
  "ttg.warp_specialize"() ({
    "ttg.warp_yield"() : () -> ()
  }, {
    "ttg.warp_specialize.partitions"() : () -> ()
  }) {partitionNumWarps = array<i32: 1>} : () -> ()
}

// -----

tt.func @not_power_of_2() {
  // expected-error @below {{'ttg.warp_specialize' op partition #0 number of warps (3) must be a power of 2}}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(3) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

// -----

tt.func @bad_argument_count() {
  // expected-error @below {{'ttg.warp_specialize' op partition region #0 has 1 arguments but expected 0}}
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0(%arg0: i32) num_warps(4) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

// -----

tt.func @bad_argument_type(%arg0: i32) {
  // expected-error @below {{'ttg.warp_specialize' op partition region #0 argument #0 has type 'i64' but corresponding capture has type 'i32'}}
  ttg.warp_specialize(%arg0)
  default {
    ttg.warp_yield
  }
  partition0(%arg1: i64) num_warps(4) {
    ttg.warp_return
  } : (i32) -> ()
  tt.return
}

// -----

tt.func @bad_default_yields(%arg0: i32) {
  ttg.warp_specialize()
  default {
    // expected-error @below {{'ttg.warp_yield' op has 0 operands but parent op expected 1}}
    ttg.warp_yield
  } : () -> i32
  tt.return
}

// -----

tt.func @bad_default_yields(%arg0: i32, %arg1: i64) {
  ttg.warp_specialize()
  default {
    // expected-error @below {{'ttg.warp_yield' op operand #0 has type 'i64' but parent op expected 'i32'}}
    ttg.warp_yield %arg1 : i64
  } : () -> i32
  tt.return
}

// -----

#blocked_4_warps = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @function_scope() attributes {"ttg.num-warps" = 8 : i32} {
  // expected-error @below {{Layout has 4 warps per CTA, but the context requires 8 warps per CTA}}
  tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_4_warps>
  tt.return
}

}

// -----

#blocked_1_warps = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @function_no_scope() {
  // expected-error @below {{Layout has 1 warps per CTA, but the context requires 4 warps per CTA}}
  tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_1_warps>
  tt.return
}

}

// -----

#blocked_8_warps = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @function_no_scope() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    // expected-error @below {{Layout has 8 warps per CTA, but the context requires 2 warps per CTA}}
    tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_8_warps>
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

#blocked_2_warps = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @function_no_scope() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    ttg.warp_return
  }
  partition1() num_warps(1) {
    // expected-error @below {{Layout has 2 warps per CTA, but the context requires 1 warps per CTA}}
    tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_2_warps>
    ttg.warp_return
  } : () -> ()
  tt.return
}

}

// -----

tt.func @illegal_ws_nest() {
  ttg.warp_specialize()
  default {
    // expected-error @below {{'ttg.warp_specialize' op cannot be nested inside another `ttg.warp_specialize` op}}
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    } : () -> ()
    ttg.warp_yield
  } : () -> ()
  tt.return
}

// -----

tt.func @invalid_start_ids() {
  // expected-error @below {{'ttg.warp_specialize' op has 1 warp group start IDs but expected 2}}
  ttg.warp_specialize() attributes {warpGroupStartIds = array<i32: 4>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    ttg.warp_return
  }
  partition1() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  tt.return
}

// -----

tt.func @partition_no_terminator() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  // expected-error @below {{region with at least 1 blocks}}
  partition0() num_warps(2) {
  } : () -> ()
  tt.return
}

// -----

tt.func @partition_no_terminator() {
  ttg.warp_specialize()
  default {
    ttg.warp_yield
  }
  partition0() num_warps(2) {
    // expected-error @below {{block with no terminator}}
    %c1_i32 = arith.constant 1 : i32
  } : () -> ()
  tt.return
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @async_copy_invalid_mask_type(%input: tensor<64x64x!tt.ptr<f16>, #blocked>,
    %view: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
    %invalid_mask: tensor<64x64xi32, #blocked> // expected-note {{prior use here}}
  ) {
    // expected-error @+1 {{expects different type than prior uses}}
    %token = ttg.async_copy_global_to_local %input, %view mask %invalid_mask
      : tensor<64x64x!tt.ptr<f16>, #blocked> -> <64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_copy_invalid_other_type(%input: tensor<64x64x!tt.ptr<f16>, #blocked>,
    %view: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
    %mask: tensor<64x64xi1, #blocked>,
    %invalid_other: tensor<64x64xf32, #blocked> // expected-note {{prior use here}}
  ) {
  // expected-error @+1 {{expects different type than prior uses}}
  %token = ttg.async_copy_global_to_local %input, %view mask %mask other %invalid_other : tensor<64x64x!tt.ptr<f16>, #blocked> -> <64x64xf16, #shared, #smem, mutable>
  tt.return
}
}

// -----

// expected-error @below {{parent layout must have at least rank >= 2}}
#slice = #ttg.slice<{dim = 0, parent = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>}>

// -----

// expected-error @below {{slice dim=2 must be less than the parent rank=2}}
#slice = #ttg.slice<{dim = 2, parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>}>

// -----

// expected-error @below {{rank 0 memdesc is not allowed}}
!memdesc = !ttg.memdesc<i64, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>
