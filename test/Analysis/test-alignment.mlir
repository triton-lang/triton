// RUN: triton-opt %s -test-print-alignment -split-input-file 2>&1 | FileCheck %s

func @permute_2d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) {
  // CHECK: Contiguity: [1, 1] ; Divisibility: [1, 1] ; Constancy: [1, 1]
  %cst = arith.constant dense<true> : tensor<128x128xi1>
  // CHECK-NEXT: Contiguity: [1, 1] ; Divisibility: [1, 1] ; Constancy: [1, 1]
  %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
  // CHECK-NEXT: Contiguity: [128] ; Divisibility: [65536] ; Constancy: [1]
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: Contiguity: [128] ; Divisibility: [65536] ; Constancy: [1]
  %1 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK-NEXT: Contiguity: [128, 1] ; Divisibility: [65536, 1] ; Constancy: [1, 1]
  %2 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<128xi32>) -> tensor<128x1xi32>
  // CHECK-NEXT: Contiguity: [1, 1] ; Divisibility: [16, 16] ; Constancy: [128, 1]
  %3 = tt.splat %arg1 : (i32) -> tensor<128x1xi32>
  // CHECK-NEXT: Contiguity: [1, 1] ; Divisibility: [1048576, 16] ; Constancy: [1, 1]
  %4 = arith.muli %2, %3 : tensor<128x1xi32>
  // CHECK-NEXT: Contiguity: [1, 1] ; Divisibility: [16, 16] ; Constancy: [128, 1]
  %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>>
  // CHECK-NEXT: Contiguity: [1, 1] ; Divisibility: [16, 16] ; Constancy: [1, 1]
  %6 = tt.addptr %5, %4 : tensor<128x1x!tt.ptr<f32>>
  // CHECK-NEXT: Contiguity: [1, 128] ; Divisibility: [1, 65536] ; Constancy: [1, 1]
  %7 = tt.expand_dims %1 {axis = 0 : i32}: (tensor<128xi32>) -> tensor<1x128xi32>
  // CHECK-NEXT: Contiguity: [1, 1] ; Divisibility: [16, 16] ; Constancy: [1, 128]
  %8 = tt.broadcast %6 : (tensor<128x1x!tt.ptr<f32>>) -> tensor<128x128x!tt.ptr<f32>>
  // CHECK-NEXT: Contiguity: [1, 128] ; Divisibility: [1, 65536] ; Constancy: [128, 1]
  %9 = tt.broadcast %7 : (tensor<1x128xi32>) -> tensor<128x128xi32>
  // CHECK-NEXT: Contiguity: [1, 128] ; Divisibility: [1, 16] ; Constancy: [1, 1]
  %10 = tt.addptr %8, %9 : tensor<128x128x!tt.ptr<f32>>
  // CHECK-NEXT: Contiguity: [128, 1] ; Divisibility: [65536, 1] ; Constancy: [1, 1]
  %11 = tt.expand_dims %0 {axis = 1 : i32}: (tensor<128xi32>) -> tensor<128x1xi32>
  // CHECK-NEXT: Contiguity: [1, 1] ; Divisibility: [16, 16] ; Constancy: [128, 1]
  %12 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<128x1x!tt.ptr<f32>>
  // CHECK-NEXT: Contiguity: [128, 1] ; Divisibility: [16, 1] ; Constancy: [1, 1]
  %13 = tt.addptr %12, %11 : tensor<128x1x!tt.ptr<f32>>
  // CHECK-NEXT: Contiguity: [1, 128] ; Divisibility: [1, 65536] ; Constancy: [1, 1]
  %14 = tt.expand_dims %1 {axis = 0 : i32} : (tensor<128xi32>) -> tensor<1x128xi32>
  // CHECK-NEXT: Contiguity: [1, 1] ; Divisibility: [16, 16] ; Constancy: [1, 128]
  %15 = tt.splat %arg3 : (i32) -> tensor<1x128xi32>
  // CHECK-NEXT: Contiguity: [1, 1] ; Divisibility: [16, 1048576] ; Constancy: [1, 1]
  %16 = arith.muli %14, %15 : tensor<1x128xi32>
  // CHECK-NEXT: Contiguity: [128, 1] ; Divisibility: [16, 1] ; Constancy: [1, 128]
  %17 = tt.broadcast %13 : (tensor<128x1x!tt.ptr<f32>>) -> tensor<128x128x!tt.ptr<f32>>
  // CHECK-NEXT: Contiguity: [1, 1] ; Divisibility: [16, 1048576] ; Constancy: [128, 1]
  %18 = tt.broadcast %16 : (tensor<1x128xi32>) -> tensor<128x128xi32>
  // CHECK-NEXT: Contiguity: [128, 1] ; Divisibility: [16, 1] ; Constancy: [1, 1]
  %19 = tt.addptr %17, %18 : tensor<128x128x!tt.ptr<f32>>
  // CHECK-NEXT: Contiguity: [1, 1] ; Divisibility: [1, 1] ; Constancy: [1, 1]
  %20 = tt.load %10, %cst, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf32>
  tt.store %19, %20, %cst : tensor<128x128xf32>
  return
}

// -----

module {

// This is a tiny test for verifying StoreOp-related alignment, It simply store a constant to a buffer.
func @store_constant_align(%addr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n: i32 {tt.divisibility = 16 : i32}) {
  // CHECK: Contiguity: [1] ; Divisibility: [1] ; Constancy: [1]
  %pid = tt.get_program_id {axis = 0 : i32} : i32
  // CHECK-NEXT: Contiguity: [1] ; Divisibility: [128] ; Constancy: [1]
  %c128_i32 = arith.constant 128 : i32
  // CHECK-NEXT: Contiguity: [1] ; Divisibility: [128] ; Constancy: [1]
  %1 = arith.muli %pid, %c128_i32 : i32
  // CHECK-NEXT: Contiguity: [128] ; Divisibility: [65536] ; Constancy: [1]
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
 // CHECK-NEXT: Contiguity: [1] ; Divisibility: [128] ; Constancy: [128]
  %3 = tt.splat %1 : (i32) -> tensor<128xi32>
 // CHECK-NEXT: Contiguity: [128] ; Divisibility: [128] ; Constancy: [1]
  %4 = arith.addi %3, %2 : tensor<128xi32>
  // CHECK-NEXT: Contiguity: [1] ; Divisibility: [16] ; Constancy: [128]
  %5 = tt.splat %addr : (!tt.ptr<f32>) -> tensor<128x!tt.ptr<f32>>
  // CHECK-NEXT: Contiguity: [128] ; Divisibility: [16] ; Constancy: [1]
  %6 = tt.addptr %5, %4 : tensor<128x!tt.ptr<f32>>
  // CHECK-NEXT: Contiguity: [1] ; Divisibility: [16] ; Constancy: [128]
  %9 = tt.splat %n : (i32) -> tensor<128xi32>
  // CHECK-NEXT: Contiguity: [1] ; Divisibility: [128] ; Constancy: [16]
  %mask = arith.cmpi slt, %4, %9 : tensor<128xi32>
  // CHECK-NEXT: Contiguity: [1] ; Divisibility: [1] ; Constancy: [1]
  %cst = arith.constant dense<0.0> : tensor<128xf32>
  tt.store %5, %cst, %mask : tensor<128xf32>
  return
}

}

// -----

// This IR is dumped from vecadd test.
// Note, the hint {tt.divisibility = 16 : i32} for %n_elements affects the alignment of mask.
func @vecadd_mask_align_16(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32 {tt.divisibility = 16 : i32}) {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = tt.splat %1 : (i32) -> tensor<64xi32>
  %4 = arith.addi %3, %2 : tensor<64xi32>
  %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>
  %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>
  %9 = tt.splat %n_elements : (i32) -> tensor<64xi32>
  // CHECK: Contiguity: [1] ; Divisibility: [64] ; Constancy: [16] ( %{{.*}} = arith.cmpi slt, %{{.*}}, %{{.*}} : tensor<64xi32> )
  %mask = arith.cmpi slt, %4, %9 : tensor<64xi32>
  %11 = tt.load %6, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %12 = tt.load %8, %mask {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %13 = arith.addf %11, %12 : tensor<64xf32>
  %14 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  // CHECK: Contiguity: [64] ; Divisibility: [16] ; Constancy: [1] ( %{{.*}} = tt.addptr %{{.*}}, %{{.*}} : tensor<64x!tt.ptr<f32>> )
  %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>
  tt.store %15, %13, %mask : tensor<64xf32>
  return
}

// -----

// This IR is dumped from vecadd test.
// Note, there is no divisibility hint for %n_elements, Triton should assume its divisibility to be 1 by default.
func @vecadd_mask_align_1(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %n_elements: i32) {
  %c64_i32 = arith.constant 64 : i32
  %0 = tt.get_program_id {axis = 0 : i32} : i32
  %1 = arith.muli %0, %c64_i32 : i32
  %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %3 = tt.splat %1 : (i32) -> tensor<64xi32>
  %4 = arith.addi %3, %2 : tensor<64xi32>
  %5 = tt.splat %arg0 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %6 = tt.addptr %5, %4 : tensor<64x!tt.ptr<f32>>
  %7 = tt.splat %arg1 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %8 = tt.addptr %7, %4 : tensor<64x!tt.ptr<f32>>
  %9 = tt.splat %n_elements : (i32) -> tensor<64xi32>
  // CHECK: Contiguity: [1] ; Divisibility: [64] ; Constancy: [1] ( %{{.*}} = arith.cmpi slt, %{{.*}}, %{{.*}} : tensor<64xi32> )
  %10 = arith.cmpi slt, %4, %9 : tensor<64xi32>
  %11 = tt.load %6, %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %12 = tt.load %8, %10 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64xf32>
  %13 = arith.addf %11, %12 : tensor<64xf32>
  %14 = tt.splat %arg2 : (!tt.ptr<f32>) -> tensor<64x!tt.ptr<f32>>
  %15 = tt.addptr %14, %4 : tensor<64x!tt.ptr<f32>>
  tt.store %15, %13, %10 : tensor<64xf32>
  return
}
