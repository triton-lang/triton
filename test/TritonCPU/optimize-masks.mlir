// RUN: triton-opt %s -split-input-file -triton-cpu-optimize-masks -canonicalize | FileCheck %s

// Convert strided masked loads to scalar loads.

// CHECK-LABEL: @remove_masks_in_for_loop
// CHECK:       %[[VAL:.+]] = vector.load {{.+}} : memref<16xf32>, vector<16xf32>
// CHECK:       vector.store %[[VAL]], {{.+}} : memref<16xf32>, vector<16xf32>

module {
  tt.func public @remove_masks_in_for_loop(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<16xf32>
    %0 = arith.addi %arg2, %c15_i32 : i32
    %1 = arith.divsi %0, %c16_i32 : i32
    %2 = vector.splat %arg2 : vector<16xi32>
    scf.for %arg3 = %c0_i32 to %1 step %c1_i32  : i32 {
      %3 = arith.muli %arg3, %c16_i32 : i32
      %4 = vector.splat %3 : vector<16xi32>
      %5 = arith.addi %4, %cst : vector<16xi32>
      %6 = arith.cmpi slt, %5, %2 : vector<16xi32>
      %7 = tt.addptr %arg0, %3 : !tt.ptr<f32>, i32
      %8 = triton_cpu.ptr_to_memref %7 : <f32> -> memref<16xf32>
      %9 = vector.maskedload %8[%c0], %6, %cst_0 : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
      %10 = tt.addptr %arg1, %3 : !tt.ptr<f32>, i32
      %11 = triton_cpu.ptr_to_memref %10 : <f32> -> memref<16xf32>
      vector.maskedstore %11[%c0], %6, %9 : memref<16xf32>, vector<16xi1>, vector<16xf32>
    }
    tt.return
  }
}

// -----

// Replace masked load with a regular load and optimize out arith.select.

// CHECK-LABEL: @optimize_select
// CHECK:       vector.load
// CHECK-NEXT:  arith.addf
// CHECK-NEXT:  arith.addf
// CHECK-NEXT:  scf.yield

module {
  tt.func public @optimize_select(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
    %cst_1 = arith.constant dense<1.000000e+00> : vector<16xf32>
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : vector<16xf32>
    %0 = vector.splat %arg2 : vector<16xi32>
    %1 = scf.for %arg3 = %c0_i32 to %arg2 step %c16_i32 iter_args(%arg4 = %cst_2) -> (vector<16xf32>)  : i32 {
      %3 = vector.splat %arg3 : vector<16xi32>
      %4 = arith.addi %3, %cst_0 : vector<16xi32>
      %5 = arith.cmpi slt, %4, %0 : vector<16xi32>
      %6 = tt.addptr %arg0, %arg3 : !tt.ptr<f32>, i32
      %7 = triton_cpu.ptr_to_memref %6 : <f32> -> memref<16xf32>
      %8 = vector.maskedload %7[%c0], %5, %cst_2 : memref<16xf32>, vector<16xi1>, vector<16xf32> into vector<16xf32>
      %9 = arith.addf %8, %cst_1 : vector<16xf32>
      %10 = arith.select %5, %9, %cst_2 : vector<16xi1>, vector<16xf32>
      %11 = arith.addf %arg4, %10 : vector<16xf32>
      scf.yield %11 : vector<16xf32>
    }
    %2 = vector.multi_reduction <add>, %1, %cst [0] : vector<16xf32> to f32
    tt.store %arg1, %2 : !tt.ptr<f32>
    tt.return
  }
}

// -----

// Regression test for the infinite optimization loop bug.

module {
  tt.func public @remove_masks_in_for_loop(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: i32 {tt.divisibility = 16 : i32}) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : vector<16xi32>
    %c15_i32 = arith.constant 15 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : vector<16xf32>
    %0 = arith.addi %arg1, %c15_i32 : i32
    %1 = arith.divsi %0, %c16_i32 : i32
    tt.store %arg0, %1 : !tt.ptr<i32>
    tt.return
  }
}
