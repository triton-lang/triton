// RUN: triton-opt %s -split-input-file --tritonamdgpu-stream-pipeline | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #triton_gpu.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [32, 32], isTransposed = false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, triton_gpu.target = "hip:gfx90a", "triton_gpu.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @matmul_kernel
  // CHECK-SAME: %[[arg0:.*]]: !tt.ptr<f32> {{.*}}, %[[A:.*]]: !tt.ptr<f32> {{.*}}, %[[B:.*]]: !tt.ptr<f32> {{.*}}, %[[C:.*]]: !tt.ptr<f32> {{.*}}, %[[arg4:.*]]: i32, %[[arg5:.*]]: i32 {{.*}}, %[[arg6:.*]] {{.*}}, %[[arg7:.*]] {{.*}}, %[[arg8:.*]] {{.*}})
  //  %[[arg5:.*]]: i32
  tt.func public @matmul_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32, %arg8: i32) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<128x1xi32, #blocked>
    %cst_0 = arith.constant dense<16> : tensor<128x16xi32, #blocked1>
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst_1 = arith.constant dense<1> : tensor<128xi32, #blocked2>
    %c128_i32 = arith.constant 128 : i32
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x16xf32, #blocked1>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<16x256xf32, #blocked3>
    // CHECK: %[[C16:.*]] = arith.constant 16 : i32
    // CHECK: %[[C1:.*]] = arith.constant 1 : i32
    // CHECK: %[[C15:.*]] = arith.constant 15 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c15_i32 = arith.constant 15 : i32
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %cst_5 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #mma>
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg4, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    %5 = arith.muli %4, %c256_i32 : i32
    %6 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %7 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %8 = tt.splat %5 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %9 = tt.splat %5 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %10 = arith.addi %8, %6 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %11 = arith.addi %9, %7 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %12 = tt.splat %arg4 : i32 -> tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %13 = arith.remsi %10, %12 : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>>
    %14 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %15 = tt.expand_dims %14 {axis = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi32, #blocked1>
    %16 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>, #blocked1>
    %17 = tt.addptr %16, %15 : tensor<1x16x!tt.ptr<f32>, #blocked1>, tensor<1x16xi32, #blocked1>
    %18 = tt.broadcast %17 : tensor<1x16x!tt.ptr<f32>, #blocked1> -> tensor<128x16x!tt.ptr<f32>, #blocked1>
    %19 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %20 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>
    %21 = tt.expand_dims %19 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<16x1xi32, #blocked3>
    %22 = tt.expand_dims %20 {axis = 1 : i32} : tensor<16xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> -> tensor<16x1xi32, #blocked3>
    %23 = tt.splat %arg7 : i32 -> tensor<16x1xi32, #blocked3>
    %24 = arith.muli %21, %23 : tensor<16x1xi32, #blocked3>
    %25 = tt.expand_dims %13 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x256xi32, #blocked3>
    %26 = tt.broadcast %24 : tensor<16x1xi32, #blocked3> -> tensor<16x256xi32, #blocked3>
    %27 = tt.broadcast %25 : tensor<1x256xi32, #blocked3> -> tensor<16x256xi32, #blocked3>
    %28 = arith.addi %26, %27 : tensor<16x256xi32, #blocked3>
    %29 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x256x!tt.ptr<f32>, #blocked3>
    %30 = tt.addptr %29, %28 : tensor<16x256x!tt.ptr<f32>, #blocked3>, tensor<16x256xi32, #blocked3>
    %31 = arith.addi %arg5, %c15_i32 : i32
    %32 = arith.divsi %31, %c16_i32 : i32
    %33 = arith.cmpi eq, %4, %c0_i32 : i32
    %34 = arith.muli %arg7, %c16_i32 : i32
    %35 = tt.splat %34 : i32 -> tensor<16x256xi32, #blocked3>
    // CHECK: %[[T0:.*]] = arith.addi %[[arg5]], %[[C15]]
    // CHECK: %[[UB:.*]] = arith.divsi %[[T0]], %[[C16]]
    // CHECK: %[[UB1:.*]] = arith.subi %[[UB]], %[[C1]] : i32
    %36:4 = scf.for %arg9 = %c0_i32 to %32 step %c1_i32 iter_args(%arg10 = %cst_5, %arg11 = %cst_4, %arg12 = %18, %arg13 = %30) -> (tensor<128x256xf32, #mma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>, tensor<128x16x!tt.ptr<f32>, #blocked1>, tensor<16x256x!tt.ptr<f32>, #blocked3>)  : i32 {
      %61 = arith.muli %arg9, %c16_i32 : i32
      %62 = arith.subi %arg5, %61 : i32
      %63 = tt.splat %62 : i32 -> tensor<1x16xi32, #blocked1>
      %64 = arith.cmpi slt, %15, %63 : tensor<1x16xi32, #blocked1>
      %65 = tt.broadcast %64 : tensor<1x16xi1, #blocked1> -> tensor<128x16xi1, #blocked1>
      %66 = tt.load %arg12, %65, %cst_2 : tensor<128x16x!tt.ptr<f32>, #blocked1>
      %67 = tt.splat %62 : i32 -> tensor<16x1xi32, #blocked3>
      %68 = arith.cmpi slt, %22, %67 : tensor<16x1xi32, #blocked3>
      %69 = tt.broadcast %68 : tensor<16x1xi1, #blocked3> -> tensor<16x256xi1, #blocked3>
      %70 = tt.load %arg13, %69, %cst_3 : tensor<16x256x!tt.ptr<f32>, #blocked3>
      %71 = triton_gpu.convert_layout %66 : tensor<128x16xf32, #blocked1> -> tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
      %72 = triton_gpu.convert_layout %70 : tensor<16x256xf32, #blocked3> -> tensor<16x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %73 = tt.dot %71, %72, %arg10 : tensor<128x16xf32, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<16x256xf32, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<128x256xf32, #mma>
      %74 = scf.if %33 -> (tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) {
        %77 = "tt.reduce"(%66) <{axis = 1 : i32}> ({
        ^bb0(%arg14: f32, %arg15: f32):
          %79 = arith.addf %arg14, %arg15 : f32
          tt.reduce.return %79 : f32
        }) : (tensor<128x16xf32, #blocked1>) -> tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
        %78 = arith.addf %arg11, %77 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
        scf.yield %78 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      } else {
        scf.yield %arg11 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
      }
      %75 = tt.addptr %arg12, %cst_0 : tensor<128x16x!tt.ptr<f32>, #blocked1>, tensor<128x16xi32, #blocked1>
      %76 = tt.addptr %arg13, %35 : tensor<16x256x!tt.ptr<f32>, #blocked3>, tensor<16x256xi32, #blocked3>
      scf.yield %73, %74, %75, %76 : tensor<128x256xf32, #mma>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>, tensor<128x16x!tt.ptr<f32>, #blocked1>, tensor<16x256x!tt.ptr<f32>, #blocked3>
    }
    // CHECK: arith.muli %[[UB1]], %[[C16]]
    %37 = arith.cmpi eq, %4, %c0_i32 : i32
    scf.if %37 {
      %61 = arith.muli %3, %c128_i32 : i32
      %62 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2>
      %63 = tt.splat %61 : i32 -> tensor<128xi32, #blocked2>
      %64 = arith.addi %63, %62 : tensor<128xi32, #blocked2>
      %65 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked2>
      %66 = tt.addptr %65, %64 : tensor<128x!tt.ptr<f32>, #blocked2>, tensor<128xi32, #blocked2>
      %67 = arith.cmpi slt, %64, %cst_1 : tensor<128xi32, #blocked2>
      %68 = triton_gpu.convert_layout %36#1 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>> -> tensor<128xf32, #blocked2>
      tt.store %66, %68, %67 : tensor<128x!tt.ptr<f32>, #blocked2>
    }
    %38 = arith.truncf %36#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma>
    %39 = arith.muli %3, %c128_i32 : i32
    %40 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %41 = tt.splat %39 : i32 -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %42 = arith.addi %41, %40 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %43 = tt.expand_dims %42 {axis = 1 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %44 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked>
    %45 = arith.muli %44, %43 : tensor<128x1xi32, #blocked>
    %46 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>, #blocked>
    %47 = tt.addptr %46, %45 : tensor<128x1x!tt.ptr<f32>, #blocked>, tensor<128x1xi32, #blocked>
    %48 = tt.expand_dims %11 {axis = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %49 = tt.broadcast %47 : tensor<128x1x!tt.ptr<f32>, #blocked> -> tensor<128x256x!tt.ptr<f32>, #blocked>
    %50 = tt.broadcast %48 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %51 = tt.addptr %49, %50 : tensor<128x256x!tt.ptr<f32>, #blocked>, tensor<128x256xi32, #blocked>
    %52 = arith.cmpi slt, %43, %cst : tensor<128x1xi32, #blocked>
    %53 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked>
    %54 = arith.cmpi slt, %48, %53 : tensor<1x256xi32, #blocked>
    %55 = tt.broadcast %52 : tensor<128x1xi1, #blocked> -> tensor<128x256xi1, #blocked>
    %56 = tt.broadcast %54 : tensor<1x256xi1, #blocked> -> tensor<128x256xi1, #blocked>
    %57 = arith.andi %55, %56 : tensor<128x256xi1, #blocked>
    %58 = arith.extf %38 : tensor<128x256xf16, #mma> to tensor<128x256xf32, #mma>
    %59 = triton_gpu.convert_layout %51 : tensor<128x256x!tt.ptr<f32>, #blocked> -> tensor<128x256x!tt.ptr<f32>, #mma>
    %60 = triton_gpu.convert_layout %57 : tensor<128x256xi1, #blocked> -> tensor<128x256xi1, #mma>
    tt.store %59, %58, %60 : tensor<128x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}

