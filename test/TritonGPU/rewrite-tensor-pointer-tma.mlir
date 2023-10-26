// RUN: ENABLE_TMA=1 triton-opt %s -split-input-file -tritongpu-rewrite-tensor-pointer=compute-capability=90 | FileCheck %s
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked5 = #triton_gpu.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked6 = #triton_gpu.blocked<{sizePerThread = [16, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked7 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked8 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_0d1d2d3de4de5de6de7c8c9de10de11c(%arg0: !tt.ptr<f8E5M2, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg4: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %c127_i32 = arith.constant 127 : i32 
    %c64_i32 = arith.constant 64 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf16, #blocked> 
    %c1_i64 = arith.constant 1 : i64 
    %c128_i32 = arith.constant 128 : i32 
    %c8_i32 = arith.constant 8 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.addi %arg4, %c127_i32 : i32 
    %2 = arith.divsi %1, %c128_i32 : i32 
    %3 = arith.addi %arg3, %c127_i32 : i32 
    %4 = arith.divsi %3, %c128_i32 : i32 
    %5 = arith.muli %2, %c8_i32 : i32 
    %6 = arith.divsi %0, %5 : i32 
    %7 = arith.muli %6, %c8_i32 : i32 
    %8 = arith.subi %4, %7 : i32 
    %9 = arith.minsi %8, %c8_i32 : i32 
    %10 = arith.remsi %0, %9 : i32 
    %11 = arith.addi %7, %10 : i32 
    %12 = arith.remsi %0, %5 : i32 
    %13 = arith.divsi %12, %9 : i32 
    %14 = arith.muli %11, %c128_i32 : i32 
    %15 = arith.muli %13, %c128_i32 : i32 
    %16 = arith.extsi %arg3 : i32 to i64 
    %17 = arith.extsi %arg5 : i32 to i64 
    %18 = arith.extsi %arg6 : i32 to i64 
    // CHECK: tt.make_tensor_ptr
    %19 = tt.make_tensor_ptr %arg0, [%16, %17], [%18, %c1_i64], [%14, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xf8E5M2, #blocked>, 1> 
    %20 = arith.extsi %arg4 : i32 to i64 
    %21 = arith.extsi %arg7 : i32 to i64 
    // CHECK: tt.make_tensor_ptr
    %22 = tt.make_tensor_ptr %arg1, [%17, %20], [%c1_i64, %21], [%c0_i32, %15] {order = array<i32: 0, 1>} : <tensor<64x128xf8E5M2, #blocked1>, 1> 
    %23 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked2> 
    %24 = tt.splat %14 : (i32) -> tensor<128xi32, #blocked2> 
    %25 = arith.addi %24, %23 : tensor<128xi32, #blocked2> 
    %26 = tt.splat %15 : (i32) -> tensor<128xi32, #blocked2> 
    %27 = arith.addi %26, %23 : tensor<128xi32, #blocked2> 
    %28 = triton_gpu.convert_layout %25 : (tensor<128xi32, #blocked2>) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> 
    %29 = tt.expand_dims %28 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>) -> tensor<128x1xi32, #blocked3>
    %30 = triton_gpu.convert_layout %29 : (tensor<128x1xi32, #blocked3>) -> tensor<128x1xi32, #blocked> 
    %31 = tt.splat %arg8 : (i32) -> tensor<128x1xi32, #blocked> 
    %32 = arith.muli %30, %31 : tensor<128x1xi32, #blocked> 
    %33 = tt.splat %arg2 : (!tt.ptr<f16, 1>) -> tensor<128x1x!tt.ptr<f16, 1>, #blocked> 
    %34 = tt.addptr %33, %32 : tensor<128x1x!tt.ptr<f16, 1>, #blocked>, tensor<128x1xi32, #blocked> 
    %35 = triton_gpu.convert_layout %27 : (tensor<128xi32, #blocked2>) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>> 
    %36 = tt.expand_dims %35 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x128xi32, #blocked4> 
    %37 = tt.broadcast %34 : (tensor<128x1x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked> 
    %38 = tt.broadcast %36 : (tensor<1x128xi32, #blocked4>) -> tensor<128x128xi32, #blocked4> 
    %39 = triton_gpu.convert_layout %38 : (tensor<128x128xi32, #blocked4>) -> tensor<128x128xi32, #blocked> 
    %40 = tt.addptr %37, %39 : tensor<128x128x!tt.ptr<f16, 1>, #blocked>, tensor<128x128xi32, #blocked> 
    %41 = tt.splat %arg3 : (i32) -> tensor<128xi32, #blocked2> 
    %42 = arith.cmpi slt, %25, %41 : tensor<128xi32, #blocked2> 
    %43 = triton_gpu.convert_layout %42 : (tensor<128xi1, #blocked2>) -> tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked3}>> 
    %44 = tt.expand_dims %43 {axis = 1 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked3}>>) -> tensor<128x1xi1, #blocked3> 
    %45 = triton_gpu.convert_layout %44 : (tensor<128x1xi1, #blocked3>) -> tensor<128x1xi1, #blocked> 
    %46 = tt.splat %arg4 : (i32) -> tensor<128xi32, #blocked2> 
    %47 = arith.cmpi slt, %27, %46 : tensor<128xi32, #blocked2> 
    %48 = triton_gpu.convert_layout %47 : (tensor<128xi1, #blocked2>) -> tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked4}>> 
    %49 = tt.expand_dims %48 {axis = 0 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x128xi1, #blocked4> 
    %50 = tt.broadcast %45 : (tensor<128x1xi1, #blocked>) -> tensor<128x128xi1, #blocked> 
    %51 = tt.broadcast %49 : (tensor<1x128xi1, #blocked4>) -> tensor<128x128xi1, #blocked4> 
    %52 = triton_gpu.convert_layout %51 : (tensor<128x128xi1, #blocked4>) -> tensor<128x128xi1, #blocked> 
    %53 = arith.andi %50, %52 : tensor<128x128xi1, #blocked>
    %54:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg10 = %cst, %arg11 = %19, %arg12 = %22) -> (tensor<128x128xf16, #blocked>, !tt.ptr<tensor<128x64xf8E5M2, #blocked>, 1>, !tt.ptr<tensor<64x128xf8E5M2, #blocked1>, 1>)  : i32 {
      %58 = tt.load %arg11 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xf8E5M2, #blocked>, 1> -> tensor<128x64xf8E5M2, #blocked5>
      %59 = triton_gpu.convert_layout %58 : (tensor<128x64xf8E5M2, #blocked5>) -> tensor<128x64xf8E5M2, #blocked> 
      %60 = tt.load %arg12 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x128xf8E5M2, #blocked1>, 1> -> tensor<64x128xf8E5M2, #blocked6>
      %61 = triton_gpu.convert_layout %60 : (tensor<64x128xf8E5M2, #blocked6>) -> tensor<64x128xf8E5M2, #blocked1> 
      %62 = triton_gpu.convert_layout %59 : (tensor<128x64xf8E5M2, #blocked>) -> tensor<128x64xf8E5M2, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked7}>> 
      %63 = triton_gpu.convert_layout %61 : (tensor<64x128xf8E5M2, #blocked1>) -> tensor<64x128xf8E5M2, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked7}>>
      %64 = triton_gpu.convert_layout %arg10 : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #blocked7> 
      %65 = tt.dot %62, %63, %64 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xf8E5M2, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked7}>> * tensor<64x128xf8E5M2, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked7}>> -> tensor<128x128xf16, #blocked7> 
      %66 = triton_gpu.convert_layout %65 : (tensor<128x128xf16, #blocked7>) -> tensor<128x128xf16, #blocked> 
      %67 = tt.advance %arg11, [%c0_i32, %c64_i32] : <tensor<128x64xf8E5M2, #blocked>, 1> 
      %68 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x128xf8E5M2, #blocked1>, 1> 
      scf.yield %66, %67, %68 : tensor<128x128xf16, #blocked>, !tt.ptr<tensor<128x64xf8E5M2, #blocked>, 1>, !tt.ptr<tensor<64x128xf8E5M2, #blocked1>, 1> 
    } 
    %55 = triton_gpu.convert_layout %40 : (tensor<128x128x!tt.ptr<f16, 1>, #blocked>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked8> 
    %56 = triton_gpu.convert_layout %54#0 : (tensor<128x128xf16, #blocked>) -> tensor<128x128xf16, #blocked8>
    %57 = triton_gpu.convert_layout %53 : (tensor<128x128xi1, #blocked>) -> tensor<128x128xi1, #blocked8>
    tt.store %55, %56, %57 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked8>
    tt.return
  }
}