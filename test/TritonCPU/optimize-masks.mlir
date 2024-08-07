// RUN: triton-opt %s -split-input-file -triton-cpu-optimize-masks | FileCheck %s

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
