// RUN: triton-opt %s -allow-unregistered-dialect -split-input-file -tritonamdgpu-canonicalize-pointers="enable-large-tensor-ptr-canon=false" -canonicalize -verify-diagnostics | FileCheck %s

// this case is copied from amd-canonicalize-pointers-no-large-tensor.mlir. With
// enable-large-tensor-ptr-canon=false, the input is not changed at all.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion1(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.splat %1 : i32 -> tensor<1024xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.addptr %3, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %5 = tt.load %4 : tensor<1024x!tt.ptr<f32>>
    tt.return %5 : tensor<1024xf32>
  }
}

// CHECK-LABEL:   tt.func @conversion1
// CHECK: %[[ADDPTR:.*]] = tt.addptr
// CHECK:                = tt.load %[[ADDPTR]]

// -----

// Test that when a promotable pointer (pointer_range=32) merges with a
// non-promotable pointer (no pointer_range) at an arith.select, the pass
// does NOT decompose either pointer into a fat pointer. Both pointers should
// remain as-is and flow through the select unchanged.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @mixed_pointer_range_select(
      %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
      %arg2: i32 {tt.divisibility = 16 : i32},
      %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %c0_i32 = arith.constant 0 : i32
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = arith.cmpi sgt, %arg2, %c0_i32 : i32
    %6 = arith.select %5, %arg0, %arg1 : !tt.ptr<f16>
    %7 = tt.splat %6 : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<f16>>, tensor<512xi32>
    %9 = tt.load %8 : tensor<512x!tt.ptr<f16>>
    %10 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<512x!tt.ptr<f16>>
    %11 = tt.addptr %10, %4 : tensor<512x!tt.ptr<f16>>, tensor<512xi32>
    tt.store %11, %9 : tensor<512x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK-LABEL:   tt.func @mixed_pointer_range_select
// CHECK-SAME:      %[[ARG0:.*]]: !tt.ptr<f16> {{.*}}, %[[ARG1:.*]]: !tt.ptr<f16> {{.*}}, %[[ARG2:.*]]: i32 {{.*}}, %[[ARG3:.*]]: !tt.ptr<f16>
// The arith.select on scalar pointers must remain — no fat pointer decomposition.
// CHECK:           %[[CMP:.*]] = arith.cmpi sgt, %[[ARG2]]
// CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]] : !tt.ptr<f16>
// The selected pointer flows through splat → addptr → load unchanged.
// CHECK:           %[[SPLAT:.*]] = tt.splat %[[SEL]]
// CHECK:           %[[ADDPTR:.*]] = tt.addptr %[[SPLAT]]
// CHECK:           %[[LOAD:.*]] = tt.load %[[ADDPTR]]
// CHECK:           tt.store
