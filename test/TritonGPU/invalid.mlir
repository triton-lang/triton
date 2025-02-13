// RUN: triton-opt --split-input-file %s --verify-diagnostics

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @miss_encoding(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{,}}
    %a = ttg.memdesc_subview %arg0[%zero, %zero] : !ttg.memdesc<8x16xf32> -> !ttg.memdesc<8x16xf16>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @miss_memory_space(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{,}}
    %a = ttg.memdesc_subview %arg0[%zero, %zero] : !ttg.memdesc<8x16xf32, #shared> -> !ttg.memdesc<8x16xf16>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @subview_element_ty(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{element type}}
    %a = ttg.memdesc_subview %arg0[%zero, %zero] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<8x16xf16, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @too_many_offsets(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{offsets}}
    %a = ttg.memdesc_subview %arg0[%zero, %zero, %zero] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<f32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @too_few_offsets(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{offsets}}
    %a = ttg.memdesc_subview %arg0[%zero] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<f32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @result_rank_too_large(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result rank}}
    %a = ttg.memdesc_subview %arg0[%zero, %zero] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<3x8x16xf32, #shared, #smem>
    tt.return
}

// -----

#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
tt.func public @result_dim_too_large(%arg0: !ttg.memdesc<8x16xf32, #shared, #smem>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result shape}}
    %a = ttg.memdesc_subview %arg0[%zero, %zero] : !ttg.memdesc<8x16xf32, #shared, #smem> -> !ttg.memdesc<32xf32, #shared, #smem>
    tt.return
}

// -----

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
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

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
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

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
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

#mma0 = #ttg.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
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

tt.func @bad_partition_terminator() {
  // expected-error @below {{'ttg.warp_specialize' op partition region #0 does not end with a `ttg.warp_return` op}}
  "ttg.warp_specialize"() ({
    "ttg.warp_yield"() : () -> ()
  }, {
    "ttg.warp_specialize.partitions"() ({
      "ttg.warp_yield"() : () -> ()
    }) : () -> ()
  }) {partitionNumWarps = array<i32: 1>} : () -> ()
  tt.return
}

// -----

tt.func @bad_default_terminator() {
  // expected-error @below {{'ttg.warp_specialize' op expected its default region to end with a `ttg.warp_yield` op}}
  ttg.warp_specialize()
  default {
    scf.yield
  } : () -> ()
  tt.return
}
