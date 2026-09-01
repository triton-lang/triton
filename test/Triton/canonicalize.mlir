// RUN: triton-opt %s -split-input-file -canonicalize | FileCheck %s
// RUN: triton-opt %s -split-input-file -gluon-canonicalize | FileCheck %s --check-prefix=GLUON

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

// -----

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

// -----

// CHECK-LABEL: fold_addptr
tt.func @fold_addptr(%arg: tensor<64x64x!tt.ptr<f16>>) -> (tensor<64x64x!tt.ptr<f16>>) {
  // CHECK-NOT: tt.addptr
  // CHECK-NOT: arith.constant
  //     CHECK: tt.return %arg
  %c0_i32 = arith.constant dense<0> : tensor<64x64xi32>
  %0 = tt.addptr %arg, %c0_i32 : tensor<64x64x!tt.ptr<f16>>, tensor<64x64xi32>
  tt.return %0 : tensor<64x64x!tt.ptr<f16>>
}

// -----

// CHECK-LABEL: fold_addptr_scalar
tt.func @fold_addptr_scalar(%arg: !tt.ptr<f16>) -> (!tt.ptr<f16>) {
  // CHECK-NOT: tt.addptr
  // CHECK-NOT: arith.constant
  //     CHECK: tt.return %arg
  %c0_i32 = arith.constant 0 : i32
  %0 = tt.addptr %arg, %c0_i32 : !tt.ptr<f16>, i32
  tt.return %0 : !tt.ptr<f16>
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#sliced0 = #ttg.slice<{dim = 1, parent = #blocked0}>

// CHECK-LABEL: fn
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
tt.func @fn(%arg0: tensor<1xf32, #sliced0>) -> (tensor<32x1xf32, #blocked0>){
  // CHECK: %[[a:.*]] = tt.expand_dims
  // CHECK: tt.broadcast %[[a]]
  %a = tt.broadcast %arg0 : tensor<1xf32, #sliced0> -> tensor<32xf32, #sliced0>
  %b = tt.expand_dims %a {axis = 1 : i32} : tensor<32xf32, #sliced0> -> tensor<32x1xf32, #blocked0>
  tt.return %b : tensor<32x1xf32, #blocked0>
}
}  // end module

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
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

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
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

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
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

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
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

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
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

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  tt.func @fold_fp_to_fp_non_constant_nofold(%arg0: tensor<32x128xf32, #blocked>) -> tensor<32x128xf8E4M3FNUZ, #blocked> {
    // CHECK-LABEL: fold_fp_to_fp_non_constant_nofold
    // CHECK-NEXT: %[[arg_cvt:.+]] = tt.fp_to_fp %arg0
    // CHECK-NEXT: tt.return %[[arg_cvt]]
    %cst_converted = tt.fp_to_fp %arg0, rounding = rtne : tensor<32x128xf32, #blocked> -> tensor<32x128xf8E4M3FNUZ, #blocked>
    tt.return %cst_converted : tensor<32x128xf8E4M3FNUZ, #blocked>
  }
}  // end module

// -----

// CHECK-LABEL: @fold_broadcast_constant_pattern
tt.func @fold_broadcast_constant_pattern(%cst : f32) -> tensor<8x2xf32> {
    // CHECK: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<8x2xf32>
    %const = arith.constant dense<1.0> : tensor<8x1xf32>
    %bst_out = tt.broadcast %const : tensor<8x1xf32> -> tensor<8x2xf32>

    // CHECK-NEXT: tt.return %[[cst]] : tensor<8x2xf32>
    tt.return %bst_out : tensor<8x2xf32>
}

// -----

// CHECK-LABEL: @fold_transpose_constant
tt.func @fold_transpose_constant() -> tensor<128x16xf32> {
    // CHECK: %[[cst:.*]] = arith.constant dense<1.000000e+00> : tensor<128x16xf32>
    %cst = arith.constant dense<1.0> : tensor<16x128xf32>
    %r = tt.trans %cst {order = array<i32: 1, 0>} : tensor<16x128xf32> -> tensor<128x16xf32>
    // CHECK-NEXT: tt.return %[[cst]] : tensor<128x16xf32>
    tt.return %r : tensor<128x16xf32>
}
// -----

// CHECK-LABEL: @canonicalize_int_to_ptr_of_ptr_to_int
// Test: int_to_ptr(ptr_to_int(ptr)) -> ptr (round-trip elimination)
tt.func @canonicalize_int_to_ptr_of_ptr_to_int(%ptr: tensor<64x!tt.ptr<f32>>) -> tensor<64x!tt.ptr<f32>> {
  // CHECK-NOT: tt.ptr_to_int
  // CHECK-NOT: tt.int_to_ptr
  // CHECK-NOT: tt.bitcast
  // CHECK: tt.return %{{.*}} : tensor<64x!tt.ptr<f32>>
  %int = tt.ptr_to_int %ptr : tensor<64x!tt.ptr<f32>> -> tensor<64xi64>
  %result = tt.int_to_ptr %int : tensor<64xi64> -> tensor<64x!tt.ptr<f32>>
  tt.return %result : tensor<64x!tt.ptr<f32>>
}

// -----

// CHECK-LABEL: @canonicalize_int_to_ptr_of_ptr_to_int_with_different_ptr_type
tt.func @canonicalize_int_to_ptr_of_ptr_to_int_with_different_ptr_type(%ptr: !tt.ptr<f32>) -> !tt.ptr<f16> {
  // CHECK-NOT: tt.ptr_to_int
  // CHECK-NOT: tt.int_to_ptr
  // CHECK: %[[RESULT:.*]] = tt.bitcast %{{.*}} : !tt.ptr<f32> -> !tt.ptr<f16>
  %int = tt.ptr_to_int %ptr : !tt.ptr<f32> -> i64
  %result = tt.int_to_ptr %int : i64 -> !tt.ptr<f16>
  // CHECK-NEXT: tt.return %[[RESULT]] : !tt.ptr<f16>
  tt.return %result : !tt.ptr<f16>
}

// -----

// CHECK-LABEL: @canonicalize_int_to_ptr_with_constant_offset_f32
// Test: int_to_ptr(addi(ptr_to_int(ptr), constant)) -> addptr(ptr, element_offset)
// For f32 (4 bytes): 16 bytes = 4 elements
tt.func @canonicalize_int_to_ptr_with_constant_offset_f32(%base: tensor<128x!tt.ptr<f32>>) -> tensor<128x!tt.ptr<f32>> {
  // CHECK: %[[OFFSET:.*]] = arith.constant dense<4> : tensor<128xi64>
  // CHECK-NEXT: %[[RESULT:.*]] = tt.addptr %{{.*}}, %[[OFFSET]] : tensor<128x!tt.ptr<f32>>, tensor<128xi64>
  %byte_offset = arith.constant dense<16> : tensor<128xi64>
  %ptr_as_int = tt.ptr_to_int %base : tensor<128x!tt.ptr<f32>> -> tensor<128xi64>
  %offset_ptr_int = arith.addi %ptr_as_int, %byte_offset : tensor<128xi64>
  %result = tt.int_to_ptr %offset_ptr_int : tensor<128xi64> -> tensor<128x!tt.ptr<f32>>
  // CHECK-NEXT: tt.return %[[RESULT]] : tensor<128x!tt.ptr<f32>>
  tt.return %result : tensor<128x!tt.ptr<f32>>
}

// -----

// CHECK-LABEL: @canonicalize_int_to_ptr_with_constant_offset_f16
// Test: For f16 (2 bytes): 32 bytes = 16 elements
tt.func @canonicalize_int_to_ptr_with_constant_offset_f16(%base: tensor<1024x!tt.ptr<f16>>) -> tensor<1024x!tt.ptr<f16>> {
  // CHECK: %[[OFFSET:.*]] = arith.constant dense<16> : tensor<1024xi64>
  // CHECK-NEXT: %[[RESULT:.*]] = tt.addptr %{{.*}}, %[[OFFSET]] : tensor<1024x!tt.ptr<f16>>, tensor<1024xi64>
  %byte_offset = arith.constant dense<32> : tensor<1024xi64>
  %ptr_as_int = tt.ptr_to_int %base : tensor<1024x!tt.ptr<f16>> -> tensor<1024xi64>
  %offset_ptr_int = arith.addi %ptr_as_int, %byte_offset : tensor<1024xi64>
  %result = tt.int_to_ptr %offset_ptr_int : tensor<1024xi64> -> tensor<1024x!tt.ptr<f16>>
  // CHECK-NEXT: tt.return %[[RESULT]] : tensor<1024x!tt.ptr<f16>>
  tt.return %result : tensor<1024x!tt.ptr<f16>>
}

// -----

// CHECK-LABEL: @no_canonicalize_non_constant_offset
// Test: Non-constant offsets should not be canonicalized
tt.func @no_canonicalize_non_constant_offset(%base: tensor<128x!tt.ptr<f32>>, %offset: tensor<128xi64>) -> tensor<128x!tt.ptr<f32>> {
  // CHECK: tt.ptr_to_int
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: tt.int_to_ptr
  %ptr_as_int = tt.ptr_to_int %base : tensor<128x!tt.ptr<f32>> -> tensor<128xi64>
  %offset_ptr_int = arith.addi %ptr_as_int, %offset : tensor<128xi64>
  %result = tt.int_to_ptr %offset_ptr_int : tensor<128xi64> -> tensor<128x!tt.ptr<f32>>
  tt.return %result : tensor<128x!tt.ptr<f32>>
}

// -----

// CHECK-LABEL: @no_canonicalize_indivisible_offset
// Test: Offset not divisible by element size should not be canonicalized
tt.func @no_canonicalize_indivisible_offset(%base: tensor<128x!tt.ptr<f32>>) -> tensor<128x!tt.ptr<f32>> {
  // 7 bytes is not divisible by 4 (size of f32)
  // CHECK: tt.ptr_to_int
  // CHECK-NEXT: arith.addi
  // CHECK-NEXT: tt.int_to_ptr
  %byte_offset = arith.constant dense<7> : tensor<128xi64>
  %ptr_as_int = tt.ptr_to_int %base : tensor<128x!tt.ptr<f32>> -> tensor<128xi64>
  %offset_ptr_int = arith.addi %ptr_as_int, %byte_offset : tensor<128xi64>
  %result = tt.int_to_ptr %offset_ptr_int : tensor<128xi64> -> tensor<128x!tt.ptr<f32>>
  tt.return %result : tensor<128x!tt.ptr<f32>>
}

// -----

// CHECK-LABEL: @one_hot_reduce_identities
// CHECK-NOT: "tt.reduce"
// CHECK: tt.gather %arg0
// CHECK: tt.unsplat
// CHECK: tt.gather %arg1
// CHECK: tt.unsplat
// CHECK: tt.gather %arg2
// CHECK: tt.unsplat
// CHECK-NOT: "tt.reduce"
// CHECK: tt.return
tt.func @one_hot_reduce_identities(%a: tensor<8xi8>, %b: tensor<8xi16>, %c: tensor<8xi64>) -> (i8, i16, i64) {
  %offsets = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %k = arith.constant dense<3> : tensor<8xi32>
  %mask = arith.cmpi eq, %k, %offsets : tensor<8xi32>
  %zero = arith.constant dense<0> : tensor<8xi8>
  %ones = arith.constant dense<-1> : tensor<8xi16>
  %one = arith.constant dense<1> : tensor<8xi64>
  %sa = arith.select %mask, %a, %zero : tensor<8xi1>, tensor<8xi8>
  %sb = arith.select %mask, %b, %ones : tensor<8xi1>, tensor<8xi16>
  %sc = arith.select %mask, %c, %one : tensor<8xi1>, tensor<8xi64>
  %ra = "tt.reduce"(%sa) <{axis = 0 : i32}> ({
  ^bb0(%x: i8, %y: i8):
    %v = arith.addi %x, %y : i8
    tt.reduce.return %v : i8
  }) : (tensor<8xi8>) -> i8
  %rb = "tt.reduce"(%sb) <{axis = 0 : i32}> ({
  ^bb0(%x: i16, %y: i16):
    %v = arith.andi %x, %y : i16
    tt.reduce.return %v : i16
  }) : (tensor<8xi16>) -> i16
  %rc = "tt.reduce"(%sc) <{axis = 0 : i32}> ({
  ^bb0(%x: i64, %y: i64):
    %v = arith.muli %x, %y : i64
    tt.reduce.return %v : i64
  }) : (tensor<8xi64>) -> i64
  tt.return %ra, %rb, %rc : i8, i16, i64
}

// -----

// CHECK-LABEL: @one_hot_reduce_runtime_rows
// CHECK-NOT: "tt.reduce"
// CHECK: arith.andi {{.*}} : tensor<4x1xi32>
// CHECK: tt.gather %arg0{{.*}} -> tensor<4x1xi32>
// CHECK: "tt.reduce"
// CHECK: }) : (tensor<4x1xi32>) -> tensor<4xi32>
// CHECK-NOT: "tt.reduce"
// CHECK: tt.return
tt.func @one_hot_reduce_runtime_rows(%values: tensor<4x8xi32>, %row_indices: tensor<4xi32>) -> tensor<4xi32> {
  %cols = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %cols2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %offsets = tt.broadcast %cols2d : tensor<1x8xi32> -> tensor<4x8xi32>
  %rows2d = tt.expand_dims %row_indices {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %rows = tt.broadcast %rows2d : tensor<4x1xi32> -> tensor<4x8xi32>
  %seven = arith.constant dense<7> : tensor<4x8xi32>
  %indices = arith.andi %rows, %seven : tensor<4x8xi32>
  %mask = arith.cmpi eq, %indices, %offsets : tensor<4x8xi32>
  %zero = arith.constant dense<0> : tensor<4x8xi32>
  %selected = arith.select %mask, %values, %zero : tensor<4x8xi1>, tensor<4x8xi32>
  %result = "tt.reduce"(%selected) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %v = arith.ori %b, %a : i32
    tt.reduce.return %v : i32
  }) : (tensor<4x8xi32>) -> tensor<4xi32>
  tt.return %result : tensor<4xi32>
}

// -----

// CHECK-LABEL: @one_hot_scan_shifted
// CHECK: arith.subi
// CHECK: arith.cmpi ult
// CHECK: arith.select
// CHECK: tt.gather %arg0
// CHECK: arith.cmpi uge
// CHECK: arith.select
// CHECK: tt.gather %arg0
// CHECK: arith.cmpi ule
// CHECK-NOT: "tt.scan"
// CHECK: tt.return
tt.func @one_hot_scan_shifted(%values: tensor<8xi16>, %k: i32) -> (tensor<8xi16>, tensor<8xi16>) {
  %offsets = tt.make_range {start = 8 : i32, end = 16 : i32} : tensor<8xi32>
  %indices = tt.splat %k : i32 -> tensor<8xi32>
  %mask = arith.cmpi ne, %indices, %offsets : tensor<8xi32>
  %ones = arith.constant dense<-1> : tensor<8xi16>
  %selected = arith.select %mask, %ones, %values : tensor<8xi1>, tensor<8xi16>
  %forward = "tt.scan"(%selected) <{axis = 0 : i32, reverse = false}> ({
  ^bb0(%a: i16, %b: i16):
    %v = arith.andi %a, %b : i16
    tt.scan.return %v : i16
  }) : (tensor<8xi16>) -> tensor<8xi16>
  %reverse = "tt.scan"(%selected) <{axis = 0 : i32, reverse = true}> ({
  ^bb0(%a: i16, %b: i16):
    %v = arith.andi %a, %b : i16
    tt.scan.return %v : i16
  }) : (tensor<8xi16>) -> tensor<8xi16>
  tt.return %forward, %reverse : tensor<8xi16>, tensor<8xi16>
}

// -----

// CHECK-LABEL: @one_hot_constant_mask
// CHECK: tt.gather %arg0
// CHECK: arith.select
// CHECK: "tt.reduce"
// CHECK: }) : (tensor<4x1xi32>) -> tensor<4xi32>
// CHECK: tt.gather %arg0
// CHECK: arith.select
// CHECK-NOT: "tt.reduce"
// CHECK-NOT: "tt.scan"
// CHECK: tt.return
tt.func @one_hot_constant_mask(%values: tensor<4x4xi32>) -> (tensor<4xi32>, tensor<4x4xi32>) {
  %mask = arith.constant dense<[[true, false, false, false], [false, false, false, false], [false, false, true, false], [false, false, false, true]]> : tensor<4x4xi1>
  %zero = arith.constant dense<0> : tensor<4x4xi32>
  %selected = arith.select %mask, %values, %zero : tensor<4x4xi1>, tensor<4x4xi32>
  %reduced = "tt.reduce"(%selected) <{axis = 1 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %v = arith.addi %a, %b : i32
    tt.reduce.return %v : i32
  }) : (tensor<4x4xi32>) -> tensor<4xi32>
  %scanned = "tt.scan"(%selected) <{axis = 1 : i32, reverse = true}> ({
  ^bb0(%a: i32, %b: i32):
    %v = arith.addi %a, %b : i32
    tt.scan.return %v : i32
  }) : (tensor<4x4xi32>) -> tensor<4x4xi32>
  tt.return %reduced, %scanned : tensor<4xi32>, tensor<4x4xi32>
}

// -----

#one_hot_layout = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#one_hot_cols = #ttg.slice<{dim = 0, parent = #one_hot_layout}>
#one_hot_rows = #ttg.slice<{dim = 1, parent = #one_hot_layout}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @one_hot_reduce_gluon_layout
  // CHECK: tt.gather %arg0
  // CHECK: "tt.reduce"
  // CHECK: }) : (tensor<1x32xi32,
  // CHECK-NOT: "tt.reduce"
  // CHECK: tt.return
  // GLUON-LABEL: @one_hot_reduce_gluon_layout
  // GLUON: tt.gather %arg0
  // GLUON: "tt.reduce"
  // GLUON: }) : (tensor<1x32xi32,
  // GLUON-NOT: "tt.reduce"
  // GLUON: tt.return
  tt.func @one_hot_reduce_gluon_layout(%values: tensor<8x32xi32, #one_hot_layout>, %k: tensor<32xi32, #one_hot_cols>) -> tensor<32xi32, #one_hot_cols> {
    %row = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32, #one_hot_rows>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<8xi32, #one_hot_rows> -> tensor<8x1xi32, #one_hot_layout>
    %offsets = tt.broadcast %row2d : tensor<8x1xi32, #one_hot_layout> -> tensor<8x32xi32, #one_hot_layout>
    %k2d = tt.expand_dims %k {axis = 0 : i32} : tensor<32xi32, #one_hot_cols> -> tensor<1x32xi32, #one_hot_layout>
    %indices = tt.broadcast %k2d : tensor<1x32xi32, #one_hot_layout> -> tensor<8x32xi32, #one_hot_layout>
    %mask = arith.cmpi eq, %offsets, %indices : tensor<8x32xi32, #one_hot_layout>
    %minimum = arith.constant dense<-2147483648> : tensor<8x32xi32, #one_hot_layout>
    %selected = arith.select %mask, %values, %minimum : tensor<8x32xi1, #one_hot_layout>, tensor<8x32xi32, #one_hot_layout>
    %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %v = arith.maxsi %a, %b : i32
      tt.reduce.return %v : i32
    }) : (tensor<8x32xi32, #one_hot_layout>) -> tensor<32xi32, #one_hot_cols>
    tt.return %result : tensor<32xi32, #one_hot_cols>
  }
}

// -----

// CHECK-LABEL: @one_hot_unsafe_combiners
// CHECK: "tt.reduce"
// CHECK: "tt.reduce"
// CHECK: tt.assert
// CHECK: "tt.reduce"
// CHECK: "tt.reduce"
// CHECK-NOT: tt.gather
// CHECK: tt.return
tt.func @one_hot_unsafe_combiners(%values: tensor<8xi32>, %floats: tensor<8xf32>, %key: i32) -> (i32, i32, f32, i32) {
  %offsets = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %k = arith.constant dense<3> : tensor<8xi32>
  %mask = arith.cmpi eq, %offsets, %k : tensor<8xi32>
  %bad_identity = arith.constant dense<2> : tensor<8xi32>
  %zero = arith.constant dense<0> : tensor<8xi32>
  %fzero = arith.constant dense<0.0> : tensor<8xf32>
  %bad = arith.select %mask, %values, %bad_identity : tensor<8xi1>, tensor<8xi32>
  %good = arith.select %mask, %values, %zero : tensor<8xi1>, tensor<8xi32>
  %fp = arith.select %mask, %floats, %fzero : tensor<8xi1>, tensor<8xf32>
  %a = "tt.reduce"(%bad) <{axis = 0 : i32}> ({
  ^bb0(%x: i32, %y: i32):
    %v = arith.addi %x, %y : i32
    tt.reduce.return %v : i32
  }) : (tensor<8xi32>) -> i32
  %b = "tt.reduce"(%good) <{axis = 0 : i32}> ({
  ^bb0(%x: i32, %y: i32):
    %valid = arith.cmpi ne, %x, %y : i32
    tt.assert %valid, "side effect must remain" : i1
    %v = arith.xori %x, %y : i32
    tt.reduce.return %v : i32
  }) : (tensor<8xi32>) -> i32
  %c = "tt.reduce"(%fp) <{axis = 0 : i32}> ({
  ^bb0(%x: f32, %y: f32):
    %v = arith.addf %x, %y : f32
    tt.reduce.return %v : f32
  }) : (tensor<8xf32>) -> f32
  %scalar_zero = arith.constant 0 : i32
  %flag = arith.cmpi eq, %key, %scalar_zero : i32
  %uniform = arith.select %flag, %values, %zero : tensor<8xi32>
  %d = "tt.reduce"(%uniform) <{axis = 0 : i32}> ({
  ^bb0(%x: i32, %y: i32):
    %v = arith.addi %x, %y : i32
    tt.reduce.return %v : i32
  }) : (tensor<8xi32>) -> i32
  tt.return %a, %b, %c, %d : i32, i32, f32, i32
}

// -----

// CHECK-LABEL: @one_hot_multiple_survivors
// CHECK-NOT: tt.gather
// CHECK: "tt.reduce"
// CHECK: tt.return
tt.func @one_hot_multiple_survivors(%values: tensor<8xi32>) -> i32 {
  %mask = arith.constant dense<[true, false, false, false, false, true, false, false]> : tensor<8xi1>
  %zero = arith.constant dense<0> : tensor<8xi32>
  %selected = arith.select %mask, %values, %zero : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %v = arith.xori %a, %b : i32
    tt.reduce.return %v : i32
  }) : (tensor<8xi32>) -> i32
  tt.return %result : i32
}

// -----

#allocated = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.shared" = 16 : i32} {
  // A singleton gather across four warps needs different scratch than the
  // allocated reduction. The post-allocation Gluon cleanup must leave it alone.
  // CHECK-LABEL: @one_hot_after_allocation
  // CHECK: tt.gather
  // CHECK-NOT: "tt.reduce"
  // CHECK: tt.return
  // GLUON-LABEL: @one_hot_after_allocation
  // GLUON-NOT: tt.gather
  // GLUON: "tt.reduce"
  // GLUON: tt.return
  tt.func @one_hot_after_allocation(%values: tensor<128xi32, #allocated>) -> i32 {
    %offsets = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #allocated>
    %index = arith.constant dense<33> : tensor<128xi32, #allocated>
    %mask = arith.cmpi eq, %offsets, %index : tensor<128xi32, #allocated>
    %zero = arith.constant dense<0> : tensor<128xi32, #allocated>
    %selected = arith.select %mask, %values, %zero : tensor<128xi1, #allocated>, tensor<128xi32, #allocated>
    %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %v = arith.xori %a, %b : i32
      tt.reduce.return %v : i32
    }) : (tensor<128xi32, #allocated>) -> i32
    tt.return %result : i32
  }
}

// -----

// CHECK-LABEL: @one_hot_packed_indices
// CHECK-NOT: tt.gather
// CHECK: "tt.reduce"
// CHECK: tt.return
tt.func @one_hot_packed_indices(%values: tensor<8xi32>) -> i32 {
  %offsets = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %zero = arith.constant dense<0> : tensor<8xi32>
  %indices = tt.elementwise_inline_asm "mov.b32 $0, 0; mov.b32 $1, 1;" {constraints = "=r,=r,r,r", packed_element = 2 : i32, pure = true} %zero : tensor<8xi32> -> tensor<8xi32>
  %mask = arith.cmpi eq, %offsets, %indices : tensor<8xi32>
  %selected = arith.select %mask, %values, %zero : tensor<8xi1>, tensor<8xi32>
  %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
  ^bb0(%a: i32, %b: i32):
    %v = arith.xori %a, %b : i32
    tt.reduce.return %v : i32
  }) : (tensor<8xi32>) -> i32
  tt.return %result : i32
}

// -----

#lanes = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // The assembly result is explicitly tensor-valued, with one logical element
  // per lane. Uniform input operands do not make its output axis-invariant.
  // CHECK-LABEL: @one_hot_opaque_tensor_indices
  // CHECK-NOT: tt.gather
  // CHECK: "tt.reduce"
  // CHECK: }) : (tensor<32xi32,
  // CHECK: tt.return
  tt.func @one_hot_opaque_tensor_indices(%values: tensor<32xi32, #lanes>) -> i32 {
    %offsets = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #lanes>
    %zero = arith.constant dense<0> : tensor<32xi32, #lanes>
    %indices = tt.elementwise_inline_asm "mov.u32 $0, %laneid;" {constraints = "=r,r", packed_element = 1 : i32, pure = true} %zero : tensor<32xi32, #lanes> -> tensor<32xi32, #lanes>
    %mask = arith.cmpi eq, %offsets, %indices : tensor<32xi32, #lanes>
    %selected = arith.select %mask, %values, %zero : tensor<32xi1, #lanes>, tensor<32xi32, #lanes>
    %result = "tt.reduce"(%selected) <{axis = 0 : i32}> ({
    ^bb0(%a: i32, %b: i32):
      %v = arith.xori %a, %b : i32
      tt.reduce.return %v : i32
    }) : (tensor<32xi32, #lanes>) -> i32
    tt.return %result : i32
  }
}
