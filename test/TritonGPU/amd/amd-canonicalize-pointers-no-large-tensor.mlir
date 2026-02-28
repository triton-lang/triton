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

// ---
// Verify that a scalar select no longer crashes
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: _scalar_select
  tt.func public @_scalar_select(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg4: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c9_i32 = arith.constant 9 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_program_id z : i32
    %3 = tt.addptr %arg3, %0 : !tt.ptr<i32>, i32
    %4 = tt.load %3 : !tt.ptr<i32>
    %5 = arith.addi %1, %4 : i32
    %6 = arith.addi %0, %c1_i32 : i32
    %7 = tt.addptr %arg3, %6 : !tt.ptr<i32>, i32
    %8 = tt.load %7 : !tt.ptr<i32>
    %9 = arith.cmpi sge, %2, %c9_i32 : i32
    %10 = tt.addptr %arg0, %5 : !tt.ptr<bf16>, i32
    %11 = arith.muli %5, %arg8 : i32
    %12 = arith.muli %2, %arg9 : i32
    %13 = arith.addi %11, %12 : i32
    %14 = tt.addptr %arg1, %13 : !tt.ptr<bf16>, i32
    %15 = tt.addptr %arg4, %0 : !tt.ptr<i32>, i32
    %16 = tt.load %15 : !tt.ptr<i32>
    %17 = tt.addptr %arg5, %0 : !tt.ptr<i32>, i32
    %18 = tt.load %17 : !tt.ptr<i32>
    %19 = arith.addi %16, %18 : i32
    %20 = arith.subi %8, %5 : i32
    %21 = arith.subi %19, %20 : i32
    %22 = arith.subi %2, %c9_i32 : i32
    %23 = arith.muli %22, %arg7 : i32
    %24 = arith.muli %21, %arg6 : i32
    %25 = arith.addi %23, %24 : i32
    %26 = tt.addptr %arg2, %25 : !tt.ptr<bf16>, i32
    // CHECK-COUNT-2: tt.addptr
    // CHECK: arith.select
    %27 = arith.select %9, %26, %14 : !tt.ptr<bf16>
    %28 = tt.load %10 : !tt.ptr<bf16>
    tt.store %27, %28 : !tt.ptr<bf16>
    tt.return
  }
}
