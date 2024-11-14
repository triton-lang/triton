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

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  tt.func @fp_to_fp_pos_zero_fold() -> tensor<32x128xf8E4M3FNUZ, #blocked> {
    // CHECK-LABEL: fp_to_fp_pos_zero_fold
    // CHECK-NEXT: %[[cst_folded:.+]] = arith.constant dense<0.000000e+00> : tensor<32x128xf8E4M3FNUZ, #blocked>
    // CHECK-NEXT: tt.return %[[cst_folded]]
    %cst = arith.constant dense<0.00e+00> : tensor<32x128xf32, #blocked>
    %cst_converted = tt.fp_to_fp %cst, rounding = rtne : tensor<32x128xf32, #blocked> -> tensor<32x128xf8E4M3FNUZ, #blocked>
    tt.return %cst_converted : tensor<32x128xf8E4M3FNUZ, #blocked>
  }
}  // end module

// -----

module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  tt.func @fp_to_fp_pos_zero_fold_scalar() -> f8E4M3FNUZ {
    // CHECK-LABEL: fp_to_fp_pos_zero_fold_scalar
    // CHECK-NEXT: %[[cst_folded:.+]] = arith.constant 0.000000e+00 : f8E4M3FNUZ
    // CHECK-NEXT: tt.return %[[cst_folded]]
    %cst = arith.constant 0.00e+00 : f32
    %cst_converted = tt.fp_to_fp %cst, rounding = rtne : f32 -> f8E4M3FNUZ
    tt.return %cst_converted : f8E4M3FNUZ
  }
}  // end module

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  tt.func @fp_to_fp_neg_zero_fold() -> tensor<32x128xf8E4M3FN, #blocked> {
    // CHECK-LABEL: fp_to_fp_neg_zero_fold
    // CHECK-NEXT: %[[cst_folded:.+]] = arith.constant dense<-0.000000e+00> : tensor<32x128xf8E4M3FN, #blocked>
    // CHECK-NEXT: tt.return %[[cst_folded]]
    %cst = arith.constant dense<-0.00e+00> : tensor<32x128xf32, #blocked>
    %cst_converted = tt.fp_to_fp %cst, rounding = rtne : tensor<32x128xf32, #blocked> -> tensor<32x128xf8E4M3FN, #blocked>
    tt.return %cst_converted : tensor<32x128xf8E4M3FN, #blocked>
  }
}  // end module

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  tt.func @fp_to_fp_neg_zero_fold() -> tensor<32x128xf8E4M3FNUZ, #blocked> {
    // CHECK-LABEL: fp_to_fp_neg_zero_fold
    // We fold to the positive zero here given by definition f8E4M3FNUZ does not have negative zero encoding.
    // CHECK-NEXT: %[[cst_folded:.+]] = arith.constant dense<0.000000e+00> : tensor<32x128xf8E4M3FNUZ, #blocked>
    // CHECK-NEXT: tt.return %[[cst_folded]]
    %cst = arith.constant dense<-0.00e+00> : tensor<32x128xf32, #blocked>
    %cst_converted = tt.fp_to_fp %cst, rounding = rtne : tensor<32x128xf32, #blocked> -> tensor<32x128xf8E4M3FNUZ, #blocked>
    tt.return %cst_converted : tensor<32x128xf8E4M3FNUZ, #blocked>
  }
}  // end module

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  tt.func @fold_fp_to_fp_non_zero_nofold() -> tensor<32x128xf8E4M3FNUZ, #blocked> {
    // CHECK-LABEL: fold_fp_to_fp_non_zero_nofold
    // CHECK-NEXT: %[[cst:.+]] = arith.constant dense<0xFF800000> : tensor<32x128xf32, #blocked>
    // CHECK-NEXT: %[[cst_cvt:.+]] = tt.fp_to_fp %[[cst]]
    // CHECK-NEXT: tt.return %[[cst_cvt]]
    %cst = arith.constant dense<0xFF800000> : tensor<32x128xf32, #blocked>
    %cst_converted = tt.fp_to_fp %cst, rounding = rtne : tensor<32x128xf32, #blocked> -> tensor<32x128xf8E4M3FNUZ, #blocked>
    tt.return %cst_converted : tensor<32x128xf8E4M3FNUZ, #blocked>
  }
}  // end module

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32} {
  tt.func @fold_fp_to_fp_non_constant_nofold(%arg0: tensor<32x128xf32, #blocked>) -> tensor<32x128xf8E4M3FNUZ, #blocked> {
    // CHECK-LABEL: fold_fp_to_fp_non_constant_nofold
    // CHECK-NEXT: %[[arg_cvt:.+]] = tt.fp_to_fp %arg0
    // CHECK-NEXT: tt.return %[[arg_cvt]]
    %cst_converted = tt.fp_to_fp %arg0, rounding = rtne : tensor<32x128xf32, #blocked> -> tensor<32x128xf8E4M3FNUZ, #blocked>
    tt.return %cst_converted : tensor<32x128xf8E4M3FNUZ, #blocked>
  }
}  // end module
