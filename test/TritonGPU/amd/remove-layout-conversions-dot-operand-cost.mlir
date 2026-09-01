// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -cse | FileCheck %s

// hoistConvertDotOperand still moves a cheap elementwise chain into the
// inferred dot-operand layout, with the convert sitting next to the load.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx942"} {
  // CHECK-LABEL: @dot_operand_hoist_cheap_elementwise
  tt.func @dot_operand_hoist_cheap_elementwise(
      %pa: tensor<128x64x!tt.ptr<f16>, #blocked>,
      %b: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>,
      %c: tensor<128x128xf32, #mma>, %n: i32) -> tensor<128x128xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %out = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c) -> (tensor<128x128xf32, #mma>) : i32 {
      %a = tt.load %pa : tensor<128x64x!tt.ptr<f16>, #blocked>
      %m = arith.mulf %a, %a : tensor<128x64xf16, #blocked>
      %ac = ttg.convert_layout %m : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %r = tt.dot %ac, %b, %acc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
      scf.yield %r : tensor<128x128xf32, #mma>
    }
    tt.return %out : tensor<128x128xf32, #mma>
  }
}

// CHECK: %[[LOAD:.*]] = tt.load
// CHECK-NEXT: %[[CVT:.*]] = ttg.convert_layout %[[LOAD]]
// CHECK-NEXT: %[[MUL:.*]] = arith.mulf %[[CVT]], %[[CVT]]
// CHECK-NEXT: tt.dot %[[MUL]]

// -----

// Two same-sized load leaves would each need a convert. That is more expensive
// than keeping the original convert after the add, so the hoist is skipped.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx942"} {
  // CHECK-LABEL: @dot_operand_skip_two_load_leaves
  tt.func @dot_operand_skip_two_load_leaves(
      %pa1: tensor<128x64x!tt.ptr<f16>, #blocked>,
      %pa2: tensor<128x64x!tt.ptr<f16>, #blocked>,
      %b: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>,
      %c: tensor<128x128xf32, #mma>, %n: i32) -> tensor<128x128xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %out = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c) -> (tensor<128x128xf32, #mma>) : i32 {
      %a1 = tt.load %pa1 : tensor<128x64x!tt.ptr<f16>, #blocked>
      %a2 = tt.load %pa2 : tensor<128x64x!tt.ptr<f16>, #blocked>
      %add = arith.addf %a1, %a2 : tensor<128x64xf16, #blocked>
      %ac = ttg.convert_layout %add : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %r = tt.dot %ac, %b, %acc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
      scf.yield %r : tensor<128x128xf32, #mma>
    }
    tt.return %out : tensor<128x128xf32, #mma>
  }
}

// CHECK: %[[LOAD0:.*]] = tt.load
// CHECK-NEXT: %[[LOAD1:.*]] = tt.load
// CHECK-NEXT: %[[ADD:.*]] = arith.addf %[[LOAD0]], %[[LOAD1]]
// CHECK-NEXT: ttg.convert_layout %[[ADD]]
// CHECK-NEXT: tt.dot

// -----

// Expensive math that is also stored must stay in the original layout. Hoisting
// would rematerialize log/tanh in the dot-operand layout while keeping the
// stored copy, which the cost model rejects.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [4, 1], instrShape = [32, 32, 8], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32, ttg.target = "hip:gfx942"} {
  // CHECK-LABEL: @dot_operand_skip_expensive_math_multi_use
  tt.func @dot_operand_skip_expensive_math_multi_use(
      %pa: tensor<128x64x!tt.ptr<f16>, #blocked>,
      %ps: tensor<128x64x!tt.ptr<f16>, #blocked>,
      %b: tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>,
      %c: tensor<128x128xf32, #mma>, %n: i32) -> tensor<128x128xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %out = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %c) -> (tensor<128x128xf32, #mma>) : i32 {
      %a = tt.load %pa : tensor<128x64x!tt.ptr<f16>, #blocked>
      %e0 = math.log %a : tensor<128x64xf16, #blocked>
      %e1 = math.tanh %e0 : tensor<128x64xf16, #blocked>
      %m = arith.mulf %e1, %e1 : tensor<128x64xf16, #blocked>
      %ac = ttg.convert_layout %m : tensor<128x64xf16, #blocked> -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %r = tt.dot %ac, %b, %acc : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<64x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
      tt.store %ps, %m : tensor<128x64x!tt.ptr<f16>, #blocked>
      scf.yield %r : tensor<128x128xf32, #mma>
    }
    tt.return %out : tensor<128x128xf32, #mma>
  }
}

// CHECK: tt.load
// CHECK-NEXT: math.log
// CHECK-NEXT: math.tanh
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: ttg.convert_layout
// CHECK-NEXT: tt.dot
// CHECK-NEXT: tt.store
