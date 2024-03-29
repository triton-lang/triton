// RUN: TRITON_ENABLE_ARG_REMAT=1 triton-opt %s -split-input-file -tritongpu-remove-layout-conversions 2>&1 | FileCheck %s
// CHECK-LABEL: remove_convert
// CHECK-NOT: triton_gpu.convert_layout [[REGISTER:%[0-9]+]]#[[IDX:[0-9]+]] : tensor<1x512xf32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @remove_convert(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}) -> tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x512xf16, #blocked>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<1x512xf32, #blocked>
    %c512_i32 = arith.constant 512 : i32
    %c1152_i32 = arith.constant 1152 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<1x512xf32, #blocked>
    %cst_3 = arith.constant dense<1152> : tensor<1x512xi32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.cmpi slt, %0, %arg5 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked1>
    %3 = triton_gpu.convert_layout %2 : tensor<512xi32, #blocked1> -> tensor<512xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<512xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x512xi32, #blocked2>
    %5 = triton_gpu.convert_layout %4 : tensor<1x512xi32, #blocked2> -> tensor<1x512xi32, #blocked>
    %6 = arith.muli %0, %c1152_i32 : i32
    %7 = tt.splat %6 : i32 -> tensor<1x512xi32, #blocked>
    %8 = tt.splat %arg0 : !tt.ptr<f16, 1> -> tensor<1x512x!tt.ptr<f16, 1>, #blocked>
    %9 = tt.splat %1 : i1 -> tensor<1x512xi1, #blocked>
    %10 = tt.splat %arg1 : !tt.ptr<f16, 1> -> tensor<1x512x!tt.ptr<f16, 1>, #blocked>
    %11:3 = scf.for %arg7 = %c0_i32 to %c1152_i32 step %c512_i32 iter_args(%arg8 = %cst_2, %arg9 = %cst_2, %arg10 = %cst_2) -> (tensor<1x512xf32, #blocked>, tensor<1x512xf32, #blocked>, tensor<1x512xf32, #blocked>)  : i32 {
      %13 = tt.splat %arg7 : i32 -> tensor<1x512xi32, #blocked>
      %14 = arith.addi %13, %5 : tensor<1x512xi32, #blocked>
      %15 = arith.cmpi slt, %14, %cst_3 : tensor<1x512xi32, #blocked>
      %16 = arith.addi %14, %7 : tensor<1x512xi32, #blocked>
      %17 = tt.addptr %8, %16 : tensor<1x512x!tt.ptr<f16, 1>, #blocked>, tensor<1x512xi32, #blocked>
      %18 = arith.andi %15, %9 : tensor<1x512xi1, #blocked>
      %19 = triton_gpu.convert_layout %17 : tensor<1x512x!tt.ptr<f16, 1>, #blocked> -> tensor<1x512x!tt.ptr<f16, 1>, #blocked3>
      %20 = triton_gpu.convert_layout %18 : tensor<1x512xi1, #blocked> -> tensor<1x512xi1, #blocked3>
      %21 = triton_gpu.convert_layout %cst : tensor<1x512xf16, #blocked> -> tensor<1x512xf16, #blocked3>
      %22 = tt.load %19, %20, %21 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<1x512xf16, #blocked3>
      %23 = triton_gpu.convert_layout %22 : tensor<1x512xf16, #blocked3> -> tensor<1x512xf16, #blocked>
      %24 = arith.extf %23 : tensor<1x512xf16, #blocked> to tensor<1x512xf32, #blocked>
      %25 = tt.addptr %10, %14 : tensor<1x512x!tt.ptr<f16, 1>, #blocked>, tensor<1x512xi32, #blocked>
      %26 = triton_gpu.convert_layout %25 : tensor<1x512x!tt.ptr<f16, 1>, #blocked> -> tensor<1x512x!tt.ptr<f16, 1>, #blocked3>
      %27 = triton_gpu.convert_layout %15 : tensor<1x512xi1, #blocked> -> tensor<1x512xi1, #blocked3>
      %28 = triton_gpu.convert_layout %cst : tensor<1x512xf16, #blocked> -> tensor<1x512xf16, #blocked3>
      %29 = tt.load %26, %27, %28 {cache = 1 : i32, evict = 3 : i32, isVolatile = false} : tensor<1x512xf16, #blocked3>
      %30 = triton_gpu.convert_layout %29 : tensor<1x512xf16, #blocked3> -> tensor<1x512xf16, #blocked>
      %31 = arith.extf %30 : tensor<1x512xf16, #blocked> to tensor<1x512xf32, #blocked>
      %32 = arith.addf %24, %31 : tensor<1x512xf32, #blocked>
      %33 = arith.subf %32, %arg8 : tensor<1x512xf32, #blocked>
      %34 = arith.addf %arg10, %cst_1 : tensor<1x512xf32, #blocked>
      %35 = arith.divf %33, %34 : tensor<1x512xf32, #blocked>
      %36 = arith.addf %arg8, %35 : tensor<1x512xf32, #blocked>
      %37 = arith.subf %32, %36 : tensor<1x512xf32, #blocked>
      %38 = arith.mulf %33, %37 : tensor<1x512xf32, #blocked>
      %39 = arith.addf %arg9, %38 : tensor<1x512xf32, #blocked>
      %40 = arith.select %18, %36, %arg8 : tensor<1x512xi1, #blocked>, tensor<1x512xf32, #blocked>
      %41 = arith.select %18, %39, %arg9 : tensor<1x512xi1, #blocked>, tensor<1x512xf32, #blocked>
      %42 = arith.select %18, %34, %arg10 : tensor<1x512xi1, #blocked>, tensor<1x512xf32, #blocked>
      scf.yield %40, %41, %42 : tensor<1x512xf32, #blocked>, tensor<1x512xf32, #blocked>, tensor<1x512xf32, #blocked>
    }
    %12:3 = "tt.reduce"(%11#0, %11#1, %11#2) <{axis = 1 : i32}> ({
    ^bb0(%arg7: f32, %arg8: f32, %arg9: f32, %arg10: f32, %arg11: f32, %arg12: f32):
      %13 = arith.subf %arg10, %arg7 : f32
      %14 = arith.addf %arg9, %arg12 : f32
      %15 = arith.cmpf oeq, %14, %cst_0 : f32
      %16 = arith.divf %arg12, %14 : f32
      %17 = arith.select %15, %cst_0, %16 : f32
      %18 = arith.mulf %13, %17 : f32
      %19 = arith.addf %arg7, %18 : f32
      %20 = arith.addf %arg8, %arg11 : f32
      %21 = arith.mulf %13, %13 : f32
      %22 = arith.mulf %21, %arg9 : f32
      %23 = arith.mulf %22, %17 : f32
      %24 = arith.addf %20, %23 : f32
      tt.reduce.return %19, %24, %14 : f32, f32, f32
    }) : (tensor<1x512xf32, #blocked>, tensor<1x512xf32, #blocked>, tensor<1x512xf32, #blocked>) -> (tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>, tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>)
    tt.return %12#1 : tensor<1xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
  }
}
