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

// -----

// Chained arith.select over scalar bases under tensor conditions. The first
// select materializes a tensor pointer via createTensorPointer (size 1), the
// next select then sees an asymmetric pair: a size-2 (base, offset) from the
// fresh splat and a size-1 materialized pointer from the previous select.
// The asymmetric branch of ConvertArithSelectOp must route the size-2 side
// through createTensorPointer too; a raw tt.addptr with scalar base and
// tensor result type would fail TT_AddPtrOp's TypesMatchWith verifier.
// CHECK-LABEL: tt.func @chained_select_uniform_bases
// CHECK: tt.load
// CHECK-NOT: tt.addptr {{.*}}: !tt.ptr<bf16>, tensor
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @chained_select_uniform_bases(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                        %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                        %arg2: !tt.ptr<bf16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32}) -> tensor<4xbf16> {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %c1_i32 : i32 -> tensor<4xi32>
    %2 = tt.splat %c2_i32 : i32 -> tensor<4xi32>
    %cmp1 = arith.cmpi eq, %0, %1 : tensor<4xi32>
    %cmp2 = arith.cmpi eq, %0, %2 : tensor<4xi32>
    %splat0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>>
    %splat1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>>
    %splat2 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<4x!tt.ptr<bf16>>
    %sel1 = arith.select %cmp1, %splat1, %splat0 : tensor<4xi1>, tensor<4x!tt.ptr<bf16>>
    %sel2 = arith.select %cmp2, %splat2, %sel1 : tensor<4xi1>, tensor<4x!tt.ptr<bf16>>
    %val = tt.load %sel2 : tensor<4x!tt.ptr<bf16>>
    tt.return %val : tensor<4xbf16>
  }
}

// -----

// Mixed promotability at a scalar arith.select merge point (issue #9859).
// %arg0 has no tt.pointer_range so it stays as a single value; %arg1 has
// pointer_range=32 so it decomposes into a (base, offset) pair. The
// asymmetric branch of ConvertArithSelectOp must materialize the size-2 arm
// so a scalar arith.select over materialized pointers can be produced.
// CHECK-LABEL: tt.func @scf_select_mixed_promotable
// CHECK: tt.load
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @scf_select_mixed_promotable(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
                                       %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32, tt.pointer_range = 32 : i32},
                                       %arg2: i32) -> tensor<512xf16> {
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
    tt.return %9 : tensor<512xf16>
  }
}
