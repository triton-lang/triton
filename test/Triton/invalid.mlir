// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func public @fn(%arg0: tensor<128xf32>) {
    // expected-error @+1 {{packed_element}}
    %a = tt.elementwise_inline_asm ""
      {constraints = "=r,r", packed_element=3:i32, pure=true} %arg0 : tensor<128xf32> -> tensor<128xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<128xf32>, %arg1: tensor<64xf32>) {
    // expected-error @+1 {{same shape}}
    %a = tt.elementwise_inline_asm ""
      {constraints = "=r,r,r", packed_element=1:i32, pure=true}
      %arg0, %arg1: tensor<128xf32>, tensor<64xf32> -> tensor<128xf32>
    tt.return
}
// -----

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

// -----

// Valid ops.
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32>) {
    %a = tt.trans %arg0 {order = array<i32: 0, 1, 2>} : (tensor<16x32x64xf32>) -> tensor<16x32x64xf32>
    %b = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : (tensor<16x32x64xf32>) -> tensor<32x16x64xf32>
    tt.return
}
}  // end module

// -----

// Valid op with blocked encoding.
#blocked  = #triton_gpu.blocked<{sizePerThread = [1,2,3,4], threadsPerWarp = [2,4,2,2], warpsPerCTA = [4,2,4,2], order = [3,2,1,0], CTAsPerCGA = [1,2,2,2], CTASplitNum = [1,2,4,8], CTAOrder = [3,2,1,0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [2,4,3,1], threadsPerWarp = [4,2,2,2], warpsPerCTA = [2,2,4,4], order = [1,2,0,3], CTAsPerCGA = [2,2,2,1], CTASplitNum = [2,8,4,1], CTAOrder = [1,2,0,3]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [2,4,4], warpsPerCTA = [2,4,8], order = [0,1,2], CTAsPerCGA = [1,2,4], CTASplitNum = [1,2,4], CTAOrder = [0,1,2]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [2,1,4], threadsPerWarp = [4,2,4], warpsPerCTA = [4,2,8], order = [1,0,2], CTAsPerCGA = [2,1,4], CTASplitNum = [2,1,4], CTAOrder = [1,0,2]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 8 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x4x8x16xf32, #blocked>, %arg1: tensor<16x32x64xf32, #blocked2>) {
    %a = tt.trans %arg0 {order = array<i32: 1, 3, 2, 0>} : (tensor<2x4x8x16xf32, #blocked>) -> tensor<4x16x8x2xf32, #blocked1>
    %b = tt.trans %arg1 {order = array<i32: 1, 0, 2>} : (tensor<16x32x64xf32, #blocked2>) -> tensor<32x16x64xf32, #blocked3>
    tt.return
}
}  // end module

// -----

// Valid op with shared encoding.
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [3, 2, 1, 0]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 2, 0, 3]}>
#shared2 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 2], CTASplitNum = [2, 4], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared3 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [2, 1], CTASplitNum = [4, 2], CTAOrder = [1, 0], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 8 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x4x8x16xf32, #shared>, %arg1: tensor<16x32xf32, #shared2>) {
    %a = tt.trans %arg0 {order = array<i32: 1, 3, 2, 0>} : (tensor<2x4x8x16xf32, #shared>) -> tensor<4x16x8x2xf32, #shared1>
    %b = tt.trans %arg1 {order = array<i32: 1, 0>} : (tensor<16x32xf32, #shared2>) -> tensor<32x16xf32, #shared3>
    tt.return
}
}  // end module

// -----

// Invalid blocked encoding.
#blocked  = #triton_gpu.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [2,4,4], warpsPerCTA = [2,4,8], order = [0,1,2], CTAsPerCGA = [1,2,4], CTASplitNum = [1,2,4], CTAOrder = [0,1,2]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [4,2,4], warpsPerCTA = [4,2,8], order = [1,0,2], CTAsPerCGA = [2,1,4], CTASplitNum = [2,1,4], CTAOrder = [1,0,2]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 8 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32, #blocked>) {
    // expected-error @+1 {{type}}
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : (tensor<16x32x64xf32, #blocked>) -> tensor<32x16x64xf32, #blocked1>
    tt.return
}
}  // end module

// -----

// Invalid shared encoding.
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1, 2]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 0, 1]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 8 : i32, "triton_gpu.num-warps" = 64 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32, #shared>) {
    // expected-error @+1 {{type}}
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : (tensor<16x32x64xf32, #shared>) -> tensor<32x16x64xf32, #shared1>
    tt.return
}
}  // end module

// -----

module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order}}
    %a = tt.trans %arg0 {order = array<i32: 0>} : (tensor<16x32xf32>) -> tensor<32x16xf32>
    tt.return
}
}  // end module

// -----

module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order}}
    %a = tt.trans %arg0 {order = array<i32: 2, 1, 0>} : (tensor<16x32xf32>) -> tensor<32x16xf32>
    tt.return
}
}  // end module

// -----

module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order must be a permutation}}
    %a = tt.trans %arg0 {order = array<i32: 0, 0>} : (tensor<16x32xf32>) -> tensor<32x16xf32>
    tt.return
}
}  // end module
