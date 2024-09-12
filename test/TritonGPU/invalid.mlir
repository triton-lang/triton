// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func public @subview_element_ty(%arg0: !tt.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{element type}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero] : !tt.memdesc<8x16xf32> -> !tt.memdesc<8x16xf16>
    tt.return
}

// -----

tt.func public @too_many_offsets(%arg0: !tt.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{offsets}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero, %zero] : !tt.memdesc<8x16xf32> -> !tt.memdesc<f32>
    tt.return
}

// -----

tt.func public @too_few_offsets(%arg0: !tt.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{offsets}}
    %a = triton_gpu.memdesc_subview %arg0[%zero] : !tt.memdesc<8x16xf32> -> !tt.memdesc<f32>
    tt.return
}

// -----

tt.func public @result_rank_too_large(%arg0: !tt.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result rank}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero] : !tt.memdesc<8x16xf32> -> !tt.memdesc<3x8x16xf32>
    tt.return
}

// -----

tt.func public @result_dim_too_large(%arg0: !tt.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result shape}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero] : !tt.memdesc<8x16xf32> -> !tt.memdesc<32xf32>
    tt.return
}

// -----

#mma0 = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf32, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{element types of operands A and B must have same bit width}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf32, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#mma0 = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=1}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{mismatching encoding between A and B operands}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#mma0 = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=2}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32>) {
    // expected-error@+1 {{miss encoding of C operand}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32>
    tt.return
  }
}

// -----

#mma0 = #triton_gpu.nvidia_mma<{versionMajor=2, warpsPerCTA=[1,1]}>
#dot_operand_a = #triton_gpu.dot_op<{opIdx=0, parent=#mma0, kWidth=1}>
#dot_operand_b = #triton_gpu.dot_op<{opIdx=1, parent=#mma0, kWidth=2}>
module attributes {"triton_gpu.num-warps" = 1 : i32} {
  tt.func @convert_dot(%A: tensor<16x16xf16, #dot_operand_a>, %B: tensor<16x16xf16, #dot_operand_b>, %C: tensor<16x16xf32, #mma0>) {
    // expected-error@+1 {{mismatching kWidth between A and B operands}}
    %D = tt.dot %A, %B, %C : tensor<16x16xf16, #dot_operand_a> * tensor<16x16xf16, #dot_operand_b> -> tensor<16x16xf32, #mma0>
    tt.return
  }
}

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 16 : i32, "triton_gpu.proton-slots" = 1 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<i32>) attributes {noinline = false} {
      %0 = "triton_gpu.proton_init"() : () -> !tt.ptr<i32>
      %1 = triton_gpu.local_alloc  : () -> !tt.memdesc<64xi32, #shared, #triton_gpu.shared_memory, mutable>
      // expected-error@+1 {{Proton slots must be greater than the number of warpgroups per CTA}}
      "triton_gpu.local_record"(%1, %0) <{isStart = true, regionId = 0 : i32}> : (!tt.memdesc<64xi32, #shared, #triton_gpu.shared_memory, mutable>, !tt.ptr<i32>) -> ()
      "triton_gpu.proton_finalize"(%1, %0, %arg0) : (!tt.memdesc<64xi32, #shared, #triton_gpu.shared_memory, mutable>, !tt.ptr<i32>, !tt.ptr<i32>) -> ()
      tt.return
  }
}

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], hasLeadingOffset = false}>
module attributes {"triton_gpu.num-warps" = 16 : i32} {
  tt.func public @add_kernel(%arg0: !tt.ptr<i32>) attributes {noinline = false} {
      %0 = "triton_gpu.proton_init"() : () -> !tt.ptr<i32>
      %1 = triton_gpu.local_alloc  : () -> !tt.memdesc<64xi32, #shared, #triton_gpu.shared_memory, mutable>
      // expected-error@+1 {{Intra-kernel profiling not enabled}}
      "triton_gpu.local_record"(%1, %0) <{isStart = true, regionId = 0 : i32}> : (!tt.memdesc<64xi32, #shared, #triton_gpu.shared_memory, mutable>, !tt.ptr<i32>) -> ()
      "triton_gpu.proton_finalize"(%1, %0, %arg0) : (!tt.memdesc<64xi32, #shared, #triton_gpu.shared_memory, mutable>, !tt.ptr<i32>, !tt.ptr<i32>) -> ()
      tt.return
  }
}
