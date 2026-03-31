// RUN: triton-opt %s -split-input-file -tritonamdgpu-canonicalize-pointers -canonicalize | FileCheck %s

// CHECK-LABEL: tt.func @scf_if_different_bases
// CHECK: [[BASE:%.*]] = arith.select %arg2, %arg0, %arg1 : !tt.ptr<f32>
// CHECK: [[OFFSET:%.*]] = arith.select %arg2, %c16_i32, %c32_i32 : i32
// CHECK: [[PTR:%.*]] = tt.addptr [[BASE]], [[OFFSET]]
// CHECK: tt.load [[PTR]]
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @scf_if_different_bases(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                  %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                  %arg2: i1) -> f32 {
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %0 = scf.if %arg2 -> (!tt.ptr<f32>) {
      %2 = tt.addptr %arg0, %c16_i32 : !tt.ptr<f32>, i32
      scf.yield %2 : !tt.ptr<f32>
    } else {
      %2 = tt.addptr %arg1, %c32_i32 : !tt.ptr<f32>, i32
      scf.yield %2 : !tt.ptr<f32>
    }
    %1 = tt.load %0 : !tt.ptr<f32>
    tt.return %1 : f32
  }
}

// -----

// CHECK-LABEL: tt.func @select_different_bases
// CHECK: [[BASE:%.*]] = arith.select %arg2, %arg0, %arg1 : !tt.ptr<f32>
// CHECK: [[OFFSET:%.*]] = arith.select %arg2, %c16_i32, %c32_i32 : i32
// CHECK: [[PTR:%.*]] = tt.addptr [[BASE]], [[OFFSET]]
// CHECK: tt.load [[PTR]]
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @select_different_bases(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                  %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                  %arg2: i1) -> f32 {
    %c16_i32 = arith.constant 16 : i32
    %c32_i32 = arith.constant 32 : i32
    %2 = tt.addptr %arg0, %c16_i32 : !tt.ptr<f32>, i32
    %3 = tt.addptr %arg1, %c32_i32 : !tt.ptr<f32>, i32
    %4 = arith.select %arg2, %2, %3 : !tt.ptr<f32>
    %5 = tt.load %4 : !tt.ptr<f32>
    tt.return %5 : f32
  }
}
