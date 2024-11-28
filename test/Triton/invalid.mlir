// RUN: triton-opt --split-input-file %s --verify-diagnostics

tt.func @fn(%v: i32) {
  %b = tt.splat %v : i32 -> tensor<128xi32>
  // expected-error @+1 {{rank of source must be same as rank of result}}
  %c = tt.broadcast %b : tensor<128xi32> -> tensor<128x32xi32>
  tt.return
}

// -----

tt.func @fn(%v: i32) {
  %b = tt.splat %v : i32 -> tensor<2x32xi32>
  // expected-error @+1 {{Different dimensions at index 0 between source and result.  Broadcast requires the source dimension to be 1.}}
  %c = tt.broadcast %b : tensor<2x32xi32> -> tensor<128x32xi32>
  tt.return
}

// -----

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
    %a = tt.reshape %arg0 : tensor<32x128xf16> -> tensor<64x32xf16>
    tt.return
}

// -----

// expected-note @+1 {{prior use}}
tt.func public @fn(%arg0: tensor<32xf32>, %arg1: tensor<33xf32>) {
    // expected-error @+1 {{expects different type}}
    %a = tt.join %arg0, %arg1 : tensor<32xf32> -> tensor<32x2xf32>
    tt.return
}

// -----

// expected-note @+1 {{prior use}}
tt.func public @fn(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf16>) {
    // expected-error @+1 {{expects different type}}
    %a = tt.join %arg0, %arg1 : tensor<32x32xf32> -> tensor<32x32x2xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<32xf32>, %arg1: tensor<32xf32>) {
    // expected-error @+2 {{op failed to infer returned types}}
    // expected-error @+1 {{incompatible with return type}}
    %a = tt.join %arg0, %arg1 : tensor<32xf32> -> tensor<64xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) {
    // expected-error @+2 {{op failed to infer returned types}}
    // expected-error @+1 {{incompatible with return type}}
    %a = tt.join %arg0, %arg1 : tensor<32x32xf32> -> tensor<32x64xf32>
    tt.return
}

// -----

// This one is OK
tt.func public @fn(%arg0: tensor<f32>, %arg1: tensor<f32>) {
    %a = tt.join %arg0, %arg1 : tensor<f32> -> tensor<2xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: f32, %arg1: f32) {
    // expected-error @+1 {{kind of type}}
    %a = tt.join %arg0, %arg1 : f32 -> tensor<2xf32>
    tt.return
}

// -----

tt.func public @fn(%v: tensor<4x128xf64>) {
    // expected-error @+1 {{operand types and result types}}
    %a = "tt.reduce" (%v) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      tt.reduce.return %add : f32
    }) {axis = 0 : i32}  : (tensor<4x128xf64>) -> tensor<128xf32>
    tt.return
}

// -----

tt.func @reduce_different_input_shapes(%arg0: tensor<32x32x64xf32>, %arg1: tensor<16x32x64xf32>) -> (tensor<32x64xf32>, tensor<16x64xf32>) {
    // expected-error @below {{op requires the same shape for all operands}}
    %0:2 = "tt.reduce" (%arg0, %arg1) <{axis = 1 : i32}> ({
    ^bb0(%acc0: f32, %acc1: f32, %cur0: f32, %cur1: f32):
      %1 = arith.addf %acc0, %cur0 : f32
      %2 = arith.addf %acc1, %cur1 : f32
      tt.reduce.return %1, %2 : f32, f32
    }) : (tensor<32x32x64xf32>, tensor<16x32x64xf32>) -> (tensor<32x64xf32>, tensor<16x64xf32>)
    tt.return %0#0, %0#1 : tensor<32x64xf32>, tensor<16x64xf32>
}

// -----

tt.func public @fn(%v: tensor<4x128xf32>) {
    // expected-error @+1 {{requires the same shape}}
    %a = "tt.scan" (%v) ({
    ^bb0(%arg0: f32, %arg1: f32):
      %add = arith.addf %arg0, %arg1 : f32
      tt.scan.return %add : f32
    }) {axis = 0 : i32, reverse = false}  : (tensor<4x128xf32>) -> tensor<128xf32>
    tt.return
}

// -----

tt.func public @fn(%v1: tensor<4x128xf32>, %v2: tensor<4x128xi64>) {
    // expected-error @+1 {{operand types and result types}}
    %a, %b = "tt.scan" (%v1, %v2) ({
    ^bb0(%arg0: f32, %arg1: i32, %arg2: f32, %arg3: i32):
      %add = arith.addf %arg0, %arg2 : f32
      tt.scan.return %add, %arg1 : f32, i32
    }) {axis = 0 : i32, reverse = false}  : (tensor<4x128xf32>, tensor<4x128xi64>) -> (tensor<4x128xi64>, tensor<4x128xf32>)
    tt.return
}

// -----

tt.func public @fn(%v1: tensor<4x128xf32>, %v2: tensor<4x128xi64>) {
    // expected-error @+1 {{operand types and result types}}
    %a, %b = "tt.reduce" (%v1, %v2) ({
    ^bb0(%arg0: f32, %arg1: i32, %arg2: f32, %arg3: i32):
      %add = arith.addf %arg0, %arg2 : f32
      tt.reduce.return %add, %arg1 : f32, i32
    }) {axis = 0 : i32}  : (tensor<4x128xf32>, tensor<4x128xi64>) -> (tensor<128xi64>, tensor<128xf32>)
    tt.return
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<32xf32, #blocked>) {
    // expected-error @+2 {{op failed to infer returned types}}
    // expected-error @+1 {{incompatible with return type}}
    %a = tt.join %arg0, %arg0 : tensor<32xf32, #blocked> -> tensor<32x2xf32>
    tt.return
}
}  // end module

// -----

// Bad order; should be [1,0]
#blocked  = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1,2], threadsPerWarp = [32,1], warpsPerCTA = [1,1], order = [0,1]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<32xf32, #blocked>) {
    // expected-error @+2 {{order}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a = tt.join %arg0, %arg0 : tensor<32xf32, #blocked> -> tensor<32x2xf32, #blocked1>
    tt.return
}
}  // end module

// -----

tt.func public @fn(%arg0: tensor<32xf32>) {
    // expected-error @+2 {{last dimension}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<32xf32> -> tensor<16xf32>
    tt.return
}

// -----

tt.func public @fn(%arg0: tensor<32x2xf32>) {
    // expected-error @+2 {{op inferred type}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<32x2xf32> -> tensor<32xf16>
    tt.return
}

// -----

tt.func public @fn(%arg0: f32) {
    // expected-error @+1 {{invalid kind of type}}
    %a, %b = tt.split %arg0 : f32 -> f16
    tt.return
}
// -----

tt.func public @fn(%arg0: tensor<2xf32>) {
    %a, %b = tt.split %arg0 : tensor<2xf32> -> tensor<f32> // OK
    tt.return
}

// -----

#blocked  = #ttg.blocked<{sizePerThread = [1,1,2], threadsPerWarp = [1,32,1], warpsPerCTA = [1,1,1], order = [2,0,1]}>
// Bad order, should be [1,0].
#blocked1 = #ttg.blocked<{sizePerThread = [1,1], threadsPerWarp = [1,32], warpsPerCTA = [1,1], order = [1,0]}>

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x2x2xf32, #blocked>) {
    // expected-error @+2 {{op inferred type}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<2x2x2xf32, #blocked> -> tensor<2x2xf32, #blocked1>
    tt.return
}
}  // end module

// -----

#blocked  = #ttg.blocked<{sizePerThread = [1,1,2], threadsPerWarp = [1,32,1], warpsPerCTA = [1,1,1], order = [2,0,1]}>
// bad sizePerThread; should be [1,1].
#blocked1 = #ttg.blocked<{sizePerThread = [1,2], threadsPerWarp = [1,32], warpsPerCTA = [1,1], order = [0,1]}>

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x2x2xf32, #blocked>) {
    // expected-error @+2 {{op inferred type}}
    // expected-error @+1 {{op failed to infer returned types}}
    %a, %b = tt.split %arg0 : tensor<2x2x2xf32, #blocked> -> tensor<2x2xf32, #blocked1>
    tt.return
}
}  // end module

// -----

// Valid ops.
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32>) {
    %a = tt.trans %arg0 {order = array<i32: 0, 1, 2>} : tensor<16x32x64xf32> -> tensor<16x32x64xf32>
    %b = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32> -> tensor<32x16x64xf32>
    tt.return
}
}  // end module

// -----

// Valid op with blocked encoding.
#blocked  = #ttg.blocked<{sizePerThread = [1,2,3,4], threadsPerWarp = [2,4,2,2], warpsPerCTA = [4,2,4,2], order = [3,2,1,0], CTAsPerCGA = [1,2,2,2], CTASplitNum = [1,2,4,8], CTAOrder = [3,2,1,0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [2,4,3,1], threadsPerWarp = [4,2,2,2], warpsPerCTA = [2,2,4,4], order = [1,2,0,3], CTAsPerCGA = [2,2,2,1], CTASplitNum = [2,8,4,1], CTAOrder = [1,2,0,3]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [2,4,4], warpsPerCTA = [2,4,8], order = [0,1,2], CTAsPerCGA = [1,2,4], CTASplitNum = [1,2,4], CTAOrder = [0,1,2]}>
#blocked3 = #ttg.blocked<{sizePerThread = [2,1,4], threadsPerWarp = [4,2,4], warpsPerCTA = [4,2,8], order = [1,0,2], CTAsPerCGA = [2,1,4], CTASplitNum = [2,1,4], CTAOrder = [1,0,2]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<2x4x8x16xf32, #blocked>, %arg1: tensor<16x32x64xf32, #blocked2>) {
    %a = tt.trans %arg0 {order = array<i32: 1, 3, 2, 0>} : tensor<2x4x8x16xf32, #blocked> -> tensor<4x16x8x2xf32, #blocked1>
    %b = tt.trans %arg1 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #blocked2> -> tensor<32x16x64xf32, #blocked3>
    tt.return
}
}  // end module

// -----

// Valid op with shared encoding.
#shared = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [3, 2, 1, 0]}>
#shared1 = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 2, 0, 3]}>
#shared2 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 2], CTASplitNum = [2, 4], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared3 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [2, 1], CTASplitNum = [4, 2], CTAOrder = [1, 0], hasLeadingOffset = true}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: !ttg.memdesc<2x4x8x16xf32, #shared>, %arg1: !ttg.memdesc<16x32xf32, #shared2>) {
    %a = ttg.memdesc_trans %arg0 {order = array<i32: 1, 3, 2, 0>} : !ttg.memdesc<2x4x8x16xf32, #shared> -> !ttg.memdesc<4x16x8x2xf32, #shared1>
    %b = ttg.memdesc_trans %arg1 {order = array<i32: 1, 0>} : !ttg.memdesc<16x32xf32, #shared2> -> !ttg.memdesc<32x16xf32, #shared3>
    tt.return
}
}  // end module

// -----

// Invalid blocked encoding.
#blocked  = #ttg.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [2,4,4], warpsPerCTA = [2,4,8], order = [0,1,2], CTAsPerCGA = [1,2,4], CTASplitNum = [1,2,4], CTAOrder = [0,1,2]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1,2,4], threadsPerWarp = [4,2,4], warpsPerCTA = [4,2,8], order = [1,0,2], CTAsPerCGA = [2,1,4], CTASplitNum = [2,1,4], CTAOrder = [1,0,2]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32, #blocked>) {
    // expected-error @+1 {{type}}
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #blocked> -> tensor<32x16x64xf32, #blocked1>
    tt.return
}
}  // end module

// -----

// Invalid shared encoding.
#shared = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1, 2]}>
#shared1 = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 0, 1]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32, #shared>) {
    // expected-error @+1 {{type}}
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #shared> -> tensor<32x16x64xf32, #shared1>
    tt.return
}
}  // end module

// -----

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order}}
    %a = tt.trans %arg0 {order = array<i32: 0>} : tensor<16x32xf32> -> tensor<32x16xf32>
    tt.return
}
}  // end module

// -----

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order}}
    %a = tt.trans %arg0 {order = array<i32: 2, 1, 0>} : tensor<16x32xf32> -> tensor<32x16xf32>
    tt.return
}
}  // end module

// -----

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32xf32>) {
    // expected-error @+1 {{order must be a permutation}}
    %a = tt.trans %arg0 {order = array<i32: 0, 0>} : tensor<16x32xf32> -> tensor<32x16xf32>
    tt.return
}
}  // end module

// -----

// Invalid tensor with shared encoding.
#shared = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1, 2]}>
#shared1 = #ttg.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 0, 1]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 64 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @fn(%arg0: tensor<16x32x64xf32, #shared>) {
    // expected-error @+1 {{has an invalid layout: Shared layout is not allowed on tensor type.}}
    %a = tt.trans %arg0 {order = array<i32: 1, 0, 2>} : tensor<16x32x64xf32, #shared> -> tensor<32x16x64xf32, #shared1>
    tt.return
}
}  // end module

// -----

tt.func @gather_op(%arg0: tensor<128x16xf32>, %arg1: tensor<512x4xi32>) {
  // expected-error @below {{indices and output shapes must match}}
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128x16xf32>, tensor<512x4xi32>) -> tensor<512xf32>
  tt.return
}

// -----

#blocked  = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func @gather_op(%arg0: tensor<128x16xf32>, %arg1: tensor<512x4xi32, #blocked>) {
  // expected-error @below {{indices and output encodings must match}}
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128x16xf32>, tensor<512x4xi32, #blocked>) -> tensor<512x4xf32, #blocked1>
  tt.return
}
}

// -----

tt.func @gather_op(%arg0: tensor<128x16xf16>, %arg1: tensor<512x4xi32>) {
  // expected-error @below {{input and output element types must match}}
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128x16xf16>, tensor<512x4xi32>) -> tensor<512x4xf32>
  tt.return
}

// -----

tt.func @gather_op(%arg0: tensor<128xf32>, %arg1: tensor<512x4xi32>) {
  // expected-error @below {{input and indices ranks must match}}
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128xf32>, tensor<512x4xi32>) -> tensor<512x4xf32>
  tt.return
}

// -----

tt.func @gather_op(%arg0: tensor<128x16xf32>, %arg1: tensor<512x32xi32>) {
  // expected-error @below {{indices dimension 1 must match the corresponding input dimension}}
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<128x16xf32>, tensor<512x32xi32>) -> tensor<512x32xf32>
  tt.return
}
// -----

tt.func @gather_op(%arg0: tensor<128x16xf32>, %arg1: tensor<512x4xi32>) {
  // expected-error @below {{gather dimension must be less than the input rank}}
  %0 = tt.gather %arg0[%arg1] {axis = 3 : i32} : (tensor<128x16xf32>, tensor<512x4xi32>) -> tensor<512x4xf32>
  tt.return
}
