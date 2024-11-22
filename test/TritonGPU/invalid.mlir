// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func public @subview_element_ty(%arg0: !triton_gpu.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{element type}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero] : !triton_gpu.memdesc<8x16xf32> -> !triton_gpu.memdesc<8x16xf16>
    tt.return
}

// -----

tt.func public @too_many_offsets(%arg0: !triton_gpu.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{offsets}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero, %zero] : !triton_gpu.memdesc<8x16xf32> -> !triton_gpu.memdesc<f32>
    tt.return
}

// -----

tt.func public @too_few_offsets(%arg0: !triton_gpu.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{offsets}}
    %a = triton_gpu.memdesc_subview %arg0[%zero] : !triton_gpu.memdesc<8x16xf32> -> !triton_gpu.memdesc<f32>
    tt.return
}

// -----

tt.func public @result_rank_too_large(%arg0: !triton_gpu.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result rank}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero] : !triton_gpu.memdesc<8x16xf32> -> !triton_gpu.memdesc<3x8x16xf32>
    tt.return
}

// -----

tt.func public @result_dim_too_large(%arg0: !triton_gpu.memdesc<8x16xf32>) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{result shape}}
    %a = triton_gpu.memdesc_subview %arg0[%zero, %zero] : !triton_gpu.memdesc<8x16xf32> -> !triton_gpu.memdesc<32xf32>
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
