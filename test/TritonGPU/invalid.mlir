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
