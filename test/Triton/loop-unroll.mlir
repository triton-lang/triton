// RUN: triton-opt %s -triton-loop-unroll | FileCheck %s

module {
  tt.func @add_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %c256_i32 = arith.constant 256 : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %cst = arith.constant 0.000000e+00 : f32
    %9 = tt.splat %cst : f32 -> tensor<256xf32>
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: scf.for
    // CHECK-COUNT-4: tt.load
    %10:3 = scf.for %arg5 = %c0_i32 to %arg3 step %c32_i32 iter_args(%arg6 = %9, %arg7 = %6, %arg8 = %8) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>)  : i32 {
      %13 = tt.load %arg7 : tensor<256x!tt.ptr<f32>>
      %14 = tt.load %arg8 : tensor<256x!tt.ptr<f32>>
      %15 = arith.addf %13, %14 : tensor<256xf32>
      %16 = arith.addf %arg6, %15 : tensor<256xf32>
      %17 = tt.splat %arg4 : i32 -> tensor<256xi32>
      %18 = tt.addptr %arg7, %17 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %19 = tt.splat %arg4 : i32 -> tensor<256xi32>
      %20 = tt.addptr %arg8, %19 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      scf.yield %16, %18, %20 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>, tensor<256x!tt.ptr<f32>>
    } {tt.unrolled_iteration = 2 : i32}
    %11 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %12 = tt.addptr %11, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %12, %10#0 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
