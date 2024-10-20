// RUN: triton-opt %s -split-input-file -canonicalize | FileCheck %s

// CHECK-LABEL: dead_load
tt.func @dead_load(%ptr: tensor<32x128x!tt.ptr<f16>>) {
  %mask = arith.constant dense<true> : tensor<32x128xi1>
  %other = arith.constant dense<0.00e+00> : tensor<32x128xf16>
  // CHECK-NOT: tt.load {{.*}}isVolatile = false
  //     CHECK: tt.load {{.*}}isVolatile = true
  %a = tt.load %ptr, %mask, %other : tensor<32x128x!tt.ptr<f16>>
  %b = tt.load %ptr, %mask, %other {isVolatile = true} : tensor<32x128x!tt.ptr<f16>>
  tt.return
}

// CHECK-LABEL: make_range
tt.func @make_range() -> (tensor<128x1xi32>, tensor<1xi32>) {
  // CHECK-DAG: %[[c:.*]] = arith.constant dense<0> : tensor<128x1xi32>
  %a = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
  %b = tt.expand_dims %a {axis = 1 : i32} : tensor<1xi32> -> tensor<1x1xi32>
  %c = tt.broadcast %b : tensor<1x1xi32> -> tensor<128x1xi32>

  // CHECK-DAG: %[[d:.*]] = arith.constant dense<1> : tensor<1xi32>
  %d = tt.make_range {end = 2 : i32, start = 1 : i32} : tensor<1xi32>

  // CHECK-DAG: tt.return %[[c]], %[[d]] : tensor<128x1xi32>, tensor<1xi32>
  tt.return %c, %d : tensor<128x1xi32>, tensor<1xi32>
}

// CHECK-LABEL: fold_advance
tt.func @fold_advance(%arg: !tt.ptr<tensor<64x64xf16>>) -> (!tt.ptr<tensor<64x64xf16>>) {
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.advance %arg, [%c0_i32, %c0_i32] : <tensor<64x64xf16>>
  // CHECK-NOT: tt.advance
  //     CHECK: tt.return %arg
  tt.return %0 : !tt.ptr<tensor<64x64xf16>>
}


// -----

#blocked0 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#sliced0 = #triton_gpu.slice<{dim = 1, parent = #blocked0}>

// CHECK-LABEL: fn
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
tt.func @fn(%arg0: tensor<1xf32, #sliced0>) -> (tensor<32x1xf32, #blocked0>){
  // CHECK: %[[a:.*]] = tt.expand_dims
  // CHECK: tt.broadcast %[[a]]
  %a = tt.broadcast %arg0 : tensor<1xf32, #sliced0> -> tensor<32xf32, #sliced0>
  %b = tt.expand_dims %a {axis = 1 : i32} : tensor<32xf32, #sliced0> -> tensor<32x1xf32, #blocked0>
  tt.return %b : tensor<32x1xf32, #blocked0>
}
}  // end module

// -----

// CHECK-LABEL: tt.func @reduce(
// CHECK-SAME:                  %[[ARG0:.*]]: tensor<2x1x16xf32>,
// CHECK-SAME:                  %[[ARG1:.*]]: tensor<2x1x16xf16>
tt.func @reduce(%arg0: tensor<2x1x16xf32>, %arg1: tensor<2x1x16xf16>) -> (tensor<2x16xf32>, tensor<2x16xf16>) {
  // CHECK: %[[VAL0:.*]] = tt.reshape %[[ARG0]] allow_reorder : tensor<2x1x16xf32> -> tensor<2x16xf32>
  // CHECK: %[[VAL1:.*]] = tt.reshape %[[ARG1]] allow_reorder : tensor<2x1x16xf16> -> tensor<2x16xf16>
  %0:2 = "tt.reduce"(%arg0, %arg1) <{axis=1 : i32}> ({
  ^bb0(%acc0: f32, %acc1: f16, %curr0: f32, %curr1: f16):
    %1 = arith.addf %acc0, %curr0 : f32
    %2 = arith.mulf %acc1, %curr1 : f16
    tt.reduce.return %1, %2 : f32, f16
  }) : (tensor<2x1x16xf32>, tensor<2x1x16xf16>) -> (tensor<2x16xf32>, tensor<2x16xf16>)
  // CHECK: tt.return %[[VAL0]], %[[VAL1]] : tensor<2x16xf32>, tensor<2x16xf16>
  tt.return %0#0, %0#1 : tensor<2x16xf32>, tensor<2x16xf16>
}
