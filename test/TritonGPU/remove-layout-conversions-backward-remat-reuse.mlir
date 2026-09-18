// RUN: triton-opt %s -tritongpu-remove-layout-conversions | FileCheck %s

// CHECK-LABEL: @backward_remat_reuse
// CHECK-COUNT-2: tt.broadcast
// CHECK-COUNT-1: ttg.convert_layout
// CHECK-NOT: ttg.convert_layout

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @backward_remat_reuse(%arg0: !tt.ptr<f32>, %arg1: i32, %arg2: i32) -> tensor<64x2xf32, #blocked> {
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<64x2xf32, #blocked>
    %cst_1 = arith.constant dense<1> : tensor<64x2xi32, #blocked>
    %0 = tt.splat %arg1 : i32 -> tensor<64x1xi32, #blocked1>
    %1:2 = scf.for %arg3 = %arg2 to %arg1 step %arg1 iter_args(%arg4 = %cst_0, %arg5 = %cst_0) -> (tensor<64x2xf32, #blocked>, tensor<64x2xf32, #blocked>)  : i32 {
      %2 = tt.broadcast %0 : tensor<64x1xi32, #blocked1> -> tensor<64x2xi32, #blocked1>
      %3 = ttg.convert_layout %2 : tensor<64x2xi32, #blocked1> -> tensor<64x2xi32, #blocked>
      %4 = arith.cmpi slt, %3, %cst_1 : tensor<64x2xi32, #blocked>
      %5 = tt.broadcast %0 : tensor<64x1xi32, #blocked1> -> tensor<64x2xi32, #blocked1>
      %6 = ttg.convert_layout %5 : tensor<64x2xi32, #blocked1> -> tensor<64x2xi32, #blocked>
      %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x2x!tt.ptr<f32>, #blocked>
      %8 = tt.addptr %7, %6 : tensor<64x2x!tt.ptr<f32>, #blocked>, tensor<64x2xi32, #blocked>
      %9 = ttg.convert_layout %8 : tensor<64x2x!tt.ptr<f32>, #blocked> -> tensor<64x2x!tt.ptr<f32>, #blocked1>
      %10 = tt.load %9 : tensor<64x2x!tt.ptr<f32>, #blocked1>
      %11 = ttg.convert_layout %10 : tensor<64x2xf32, #blocked1> -> tensor<64x2xf32, #blocked>
      %12 = arith.select %4, %cst_0, %arg5 : tensor<64x2xi1, #blocked>, tensor<64x2xf32, #blocked>
      %13 = arith.mulf %11, %arg5 : tensor<64x2xf32, #blocked>
      %14 = arith.select %4, %13, %arg4 : tensor<64x2xi1, #blocked>, tensor<64x2xf32, #blocked>
      scf.yield %14, %12 : tensor<64x2xf32, #blocked>, tensor<64x2xf32, #blocked>
    }
    tt.return %1#0 : tensor<64x2xf32, #blocked>
  }
}
