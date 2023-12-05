// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func public @reshape_different_num_elements(%arg0: tensor<32x128xf16>) {
    // expected-error @+1 {{number of src and dst elements of reshape must be the same}}
    %a = tt.reshape %arg0 {allow_reorder = false} : tensor<32x128xf16> -> tensor<64x32xf16>
    tt.return
}

// -----

// Valid inline asm; no errors.
tt.func public @fn(%arg0: tensor<128xi32>, %arg1: tensor<128xf32>) {
    // No return type
    tt.elementwise_inline_asm "" {constraints = "", packed_element = 0 : i32, pure = false} %arg0 : tensor<128xi32>
    // Different element types (but same shape).
    %a = tt.elementwise_inline_asm "" {constraints = "", packed_element = 0 : i32, pure = false} %arg0 : tensor<128xi32> -> tensor<128xf32>
    tt.elementwise_inline_asm "" {constraints = "", packed_element = 0 : i32, pure = false} %arg0, %arg1 : tensor<128xi32>, tensor<128xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<128xi32>) {
    // No error
    tt.elementwise_inline_asm "" {constraints = "", packed_element = 0 : i32, pure = false} %arg0 : tensor<128xi32>

    // expected-error @+1 {{same shape}}
    %0 = tt.elementwise_inline_asm ""
         {constraints = "", packed_element = 0 : i32, pure = false}
         %arg0 : tensor<128xi32> -> tensor<256xi32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<128xi32>, %arg1: tensor<256xi32>) {
    // expected-error @+1 {{same shape}}
    tt.elementwise_inline_asm ""
         {constraints = "", packed_element = 0 : i32, pure = false}
         %arg0, %arg1 : tensor<128xi32>, tensor<256xi32>
    tt.return
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
tt.func public @fn(%arg0: tensor<128xi32, #blocked>, %arg1: tensor<128xi32, #blocked1>) {
    // expected-error @+1 {{same encoding}}
    tt.elementwise_inline_asm ""
         {constraints = "", packed_element = 0 : i32, pure = false}
         %arg0, %arg1 : tensor<128xi32, #blocked>, tensor<128xi32, #blocked1>
    tt.return
}
}  // end module

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
tt.func public @fn(%arg0: tensor<128xi32, #blocked>) {
    // expected-error @+1 {{same encoding}}
    tt.elementwise_inline_asm ""
         {constraints = "", packed_element = 0 : i32, pure = false}
         %arg0 : tensor<128xi32, #blocked> -> tensor<128xi32, #blocked1>
    tt.return
}
}  // end module

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
tt.func public @fn() {
    // expected-error @+1 {{same encoding}}
    tt.elementwise_inline_asm ""
         {constraints = "", packed_element = 0 : i32, pure = false}
         -> tensor<128xi32, #blocked>, tensor<128xi32, #blocked1>
    tt.return
}
}  // end module
