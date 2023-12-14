// RUN: triton-opt %s -split-input-file -tritongpu-rewrite-tensor-pointer -canonicalize -tritongpu-pipeline=compute-capability=90 -canonicalize | FileCheck %s


#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#mma1 = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 8 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
// CHECK-LABEL: two_dependent_dot
  tt.func public @two_dependent_dot(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32} , %arg3: f32 , %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32} , %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32} , %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg11: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg12: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg13: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg14: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg15: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg16: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg17: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg18: i32 , %arg19: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg20: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} , %arg21: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<0xFF800000> : tensor<128x64xf32, #mma>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst_1 = arith.constant dense<0xFF800000> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma1>
    %c1_i32 = arith.constant 1 : i32
    %cst_4 = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %1, %arg7 : i32
    %3 = arith.divsi %2, %arg8 : i32
    %4 = arith.extsi %arg21 : i32 to i64
    %5 = arith.extsi %arg11 : i32 to i64
    %6 = tt.make_tensor_ptr %arg1, [%c128_i64, %4], [%c1_i64, %5], [%c0_i32, %3] {order = array<i32: 0, 1>} : <tensor<128x64xf16, #blocked>, 1>
    %7 = arith.extsi %arg14 : i32 to i64
    %8 = tt.make_tensor_ptr %arg2, [%4, %c128_i64], [%7, %c1_i64], [%3, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x128xf16, #blocked1>, 1>
    %9 = arith.muli %0, %c128_i32 : i32
    %10 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %11 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked3>
    %13 = tt.splat %9 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %14 = tt.splat %9 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %15 = tt.splat %9 : (i32) -> tensor<128xi32, #blocked3>
    %16 = arith.addi %13, %10 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %17 = arith.addi %14, %11 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
    %18 = arith.addi %15, %12 : tensor<128xi32, #blocked3>
    %19 = arith.mulf %arg3, %cst_4 : f32
    %20 = tt.addptr %arg0, %2 : !tt.ptr<f16, 1>, i32
    %21 = tt.expand_dims %16 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi32, #blocked2>
    %22 = tt.expand_dims %17 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #mma}>>) -> tensor<128x1xi32, #mma>
    %23 = tt.splat %arg8 : (i32) -> tensor<128x1xi32, #blocked2>
    %24 = arith.muli %21, %23 : tensor<128x1xi32, #blocked2>
    %25 = tt.splat %20 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked2>
    %26 = tt.addptr %25, %24 : tensor<128x1x!tt.ptr<f16, 1>, #blocked2>, tensor<128x1xi32, #blocked2>
    %27 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %28 = tt.expand_dims %27 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi32, #blocked2>
    %29 = tt.broadcast %26 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked2>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
    %30 = tt.broadcast %28 : (tensor<1x128xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    %31 = tt.addptr %29, %30 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi32, #blocked2>
    %32 = tt.load %31 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x128xf16, #blocked2>
    %33 = tt.splat %19 : (f32) -> tensor<128x128xf32, #blocked2>
    %34 = arith.extf %32 : tensor<128x128xf16, #blocked2> to tensor<128x128xf32, #blocked2>
    %35 = arith.mulf %34, %33 : tensor<128x128xf32, #blocked2>
    %36 = arith.truncf %35 : tensor<128x128xf32, #blocked2> to tensor<128x128xf16, #blocked2>
    %37 = arith.addi %0, %c1_i32 : i32
    %38 = arith.muli %37, %c128_i32 : i32
    %42:5 = scf.for %arg22 = %c0_i32 to %38 step %c64_i32 iter_args(%arg23 = %cst_3, %arg24 = %cst_2, %arg25 = %cst_1, %arg26 = %6, %arg27 = %8) -> (tensor<128x128xf32, #mma1>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>)  : i32 {
      %59 = tt.load %arg26 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xf16, #blocked>, 1> -> tensor<128x64xf16, #blocked4>
      %60 = tt.load %arg27 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x128xf16, #blocked1>, 1> -> tensor<64x128xf16, #blocked2>
      %66 = triton_gpu.convert_layout %36 : (tensor<128x128xf16, #blocked2>) -> tensor<128x128xf16, #shared>
      %67 = triton_gpu.convert_layout %59 : (tensor<128x64xf16, #blocked4>) -> tensor<128x64xf16, #shared1>
      %68 = tt.dot %66, %67, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x128xf16, #shared> * tensor<128x64xf16, #shared1> -> tensor<128x64xf32, #mma>
      %81 = arith.truncf %68 : tensor<128x64xf32, #mma> to tensor<128x64xf16, #mma>
      %82 = triton_gpu.convert_layout %60 : (tensor<64x128xf16, #blocked2>) -> tensor<64x128xf16, #shared>
      %83 = triton_gpu.convert_layout %81 : (tensor<128x64xf16, #mma>) -> tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>>
      // CHECK: triton_nvidia_gpu.dot_async
      // CHECK-NOT: triton_nvidia_gpu.dot_wait
      // CHECK: scf.yield
      %84 = tt.dot %83, %82, %arg23 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma}>> * tensor<64x128xf16, #shared> -> tensor<128x128xf32, #mma1>
      %85 = arith.mulf %arg24, %arg25 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %87 = arith.addf %85, %arg25 : tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>
      %88 = tt.advance %arg26, [%c0_i32, %c64_i32] : <tensor<128x64xf16, #blocked>, 1>
      %89 = tt.advance %arg27, [%c64_i32, %c0_i32] : <tensor<64x128xf16, #blocked1>, 1>
      scf.yield %84, %87, %arg25, %88, %89 : tensor<128x128xf32, #mma1>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, tensor<128xf32, #triton_gpu.slice<{dim = 1, parent = #mma}>>, !tt.ptr<tensor<128x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>
    }
    %54 = arith.addi %3, %9 : i32
    %55 = arith.extsi %arg17 : i32 to i64
    %56 = tt.make_tensor_ptr %arg5, [%4, %c128_i64], [%55, %c1_i64], [%54, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x128xf16, #blocked>, 1>
    %57 = arith.truncf %42 : tensor<128x128xf32, #mma1> to tensor<128x128xf16, #mma1>
    %58 = triton_gpu.convert_layout %57 : (tensor<128x128xf16, #mma1>) -> tensor<128x128xf16, #blocked2>
    tt.store %56, %58 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x128xf16, #blocked>, 1>, tensor<128x128xf16, #blocked2>
    tt.return
  }
}
