// RUN: triton-opt %s -split-input-file -tritonamdgpu-canonicalize-pointers --tritonamdgpu-convert-buffer-ops='arch-generation-name=gfx940' | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @forOp(%arg0: !tt.ptr<f32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %5:3 = scf.for %arg2 = %c0 to %c128 step %c1 iter_args(%arg3 = %3, %arg4 = %4, %arg5 = %arg1) -> (!tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>) {
      %12 = tt.addptr %arg3, %1 : !tt.ptr<f32>, i32
      %13 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
      %14 = arith.addi %13, %arg4 : tensor<1024xi64>
      %15 = tt.splat %12 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      %16 = tt.addptr %15, %14 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
      %17 = tt.load %16 : tensor<1024x!tt.ptr<f32>>
      %18 = arith.addf %17, %arg5 : tensor<1024xf32>
      scf.yield %12, %14, %18 : !tt.ptr<f32>, tensor<1024xi64>, tensor<1024xf32>
    }
    %6 = tt.addptr %5#0, %1 : !tt.ptr<f32>, i32
    %7 = arith.extsi %2 : tensor<1024xi32> to tensor<1024xi64>
    %8 = arith.addi %7, %5#1 : tensor<1024xi64>
    %9 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %10 = tt.addptr %9, %8 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi64>
    %11 = tt.load %10 : tensor<1024x!tt.ptr<f32>>
    tt.return %11 : tensor<1024xf32>
  }
}
