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

// -----

// A select between two different base pointers under a tensor (per-element)
// condition cannot use a uniform scalar base, so both arms are materialized
// into tensor pointers before the select.
// CHECK-LABEL: tt.func @select_tensor_cond_different_bases
// CHECK: [[TRUE:%.*]] = tt.addptr {{.*}} : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
// CHECK: [[FALSE:%.*]] = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK: [[SEL:%.*]] = arith.select {{.*}}, [[TRUE]], [[FALSE]] : tensor<1024xi1>, tensor<1024x!tt.ptr<f32>>
// CHECK: tt.load [[SEL]]
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @select_tensor_cond_different_bases(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                              %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                              %arg2: i32) -> tensor<1024xf32> {
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %1 = tt.splat %arg2 : i32 -> tensor<1024xi32>
    %2 = arith.cmpi eq, %0, %1 : tensor<1024xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.addptr %3, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %6 = arith.select %2, %4, %5 : tensor<1024xi1>, tensor<1024x!tt.ptr<f32>>
    %7 = tt.load %6 : tensor<1024x!tt.ptr<f32>>
    tt.return %7 : tensor<1024xf32>
  }
}

// -----

// A tensor-condition select sharing the same base keeps the uniform base and
// only selects the offsets (no pointer select).
// CHECK-LABEL: tt.func @select_tensor_cond_same_base
// CHECK: [[OFF:%.*]] = arith.select {{.*}} : tensor<1024xi1>, tensor<1024xi32>
// CHECK: [[BASE:%.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
// CHECK: [[PTR:%.*]] = tt.addptr [[BASE]], [[OFF]]
// CHECK: tt.load [[PTR]]
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @select_tensor_cond_same_base(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                        %arg1: i32) -> tensor<1024xf32> {
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %1 = tt.splat %arg1 : i32 -> tensor<1024xi32>
    %2 = arith.cmpi eq, %0, %1 : tensor<1024xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.addptr %3, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %5 = tt.splat %c16_i32 : i32 -> tensor<1024xi32>
    %6 = tt.addptr %3, %5 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %7 = arith.select %2, %4, %6 : tensor<1024xi1>, tensor<1024x!tt.ptr<f32>>
    %8 = tt.load %7 : tensor<1024x!tt.ptr<f32>>
    tt.return %8 : tensor<1024xf32>
  }
}
