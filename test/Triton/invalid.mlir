// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func public @reshape_different_num_elements(%arg0: tensor<32x128xf16>) {
    // expected-error @+1 {{number of src and dst elements of reshape must be the same}}
    %a = tt.reshape %arg0 {allow_reorder = false} : tensor<32x128xf16> -> tensor<64x32xf16>
    tt.return
}

// -----

// expected-note @+1 {{prior use}}
tt.func public @fn(%arg0: tensor<32xf32>, %arg1: tensor<33xf32>) {
    // expected-error @+1 {{different type}}
    %a = tt.experimental_interleave %arg0, %arg1 : tensor<32xf32> -> tensor<64xf32>
    tt.return
}

// -----

// expected-note @+1 {{prior use}}
tt.func public @fn(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf16>) {
    // expected-error @+1 {{different type}}
    %a = tt.experimental_interleave %arg0, %arg1 : tensor<32x32xf32> -> tensor<64x64xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) {
    // expected-error @+1 {{last dimension}}
    %a = tt.experimental_interleave %arg0, %arg1 : tensor<32xf32> -> tensor<128xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) {
    // expected-error @+1 {{shape}}
    %a = tt.experimental_interleave %arg0, %arg1 : tensor<32x32xf32> -> tensor<64x64xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<f32>, %arg1: tensor<f32>) {
    // expected-error @+1 {{at least 1D}}
    %a = tt.experimental_interleave %arg0, %arg1 : tensor<f32> -> tensor<f32>
    tt.return
}

// -----

tt.func public @fn(%arg0: f32, %arg1: f32) {
    // expected-error @+1 {{tensor}}
    %a = tt.experimental_interleave %arg0, %arg1 : f32 -> f32
    tt.return
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<32xf32, #blocked>) {
    // expected-error @+1 {{encoding}}
    %a = tt.experimental_interleave %arg0, %arg0 : tensor<32xf32, #blocked> -> tensor<64xf32>
    tt.return
}
}  // end module

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<32xf32>) {
    // expected-error @+1 {{encoding}}
    %a = tt.experimental_interleave %arg0, %arg0 : tensor<32xf32> -> tensor<64xf32, #shared>
    tt.return
}
}  // end module

// -----

#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<32xf32, #shared>) {
    // expected-error @+1 {{encoding}}
    %a = tt.experimental_interleave %arg0, %arg0 : tensor<32xf32, #shared> -> tensor<64xf32, #blocked>
    tt.return
}
}  // end module

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<32xf32, #blocked>) {
    // expected-error @+1 {{encoding}}
    %a = tt.experimental_interleave %arg0, %arg0 : tensor<32xf32, #blocked> -> tensor<64xf32, #blocked>
    tt.return
}
}  // end module

// -----

// Bad order; should be [1,0]
#blocked  = #triton_gpu.blocked<{sizePerThread = [1,1], threadsPerWarp = [1,32], warpsPerCTA = [1,1], order = [0,1], CTAsPerCGA = [1,1], CTASplitNum = [1,1], CTAOrder = [0,1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1,2], threadsPerWarp = [1,32], warpsPerCTA = [1,1], order = [0,1], CTAsPerCGA = [1,1], CTASplitNum = [1,1], CTAOrder = [0,1]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<32x32xf32, #blocked>) {
    // expected-error @+1 {{order}}
    %a = tt.experimental_interleave %arg0, %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x64xf32, #blocked1>
    tt.return
}
}  // end module
