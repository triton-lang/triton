// RUN: triton-opt -split-input-file -triton-nvidia-gpu-ws-feasibility-checking='compute-capability=90' %s 2>&1 | FileCheck %s

// Check if all opereations are labeled with appropriate attributes.
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
// CHECK: "triton_gpu.enable-warp-specialization" = 1 : i32
// CHECK-LABEL @simple_gemm
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  tt.func public @simple_gemm(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<32> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<32> : tensor<128x32xi32, #blocked1>
    %c31_i32 = arith.constant 31 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.addi %arg3, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.addi %arg4, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %0, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.cmpi slt, %9, %c8_i32 : i32
    %11 = arith.select %10, %9, %c8_i32 : i32
    %12 = arith.remsi %0, %11 : i32
    %13 = arith.addi %8, %12 : i32
    %14 = arith.remsi %0, %6 : i32
    %15 = arith.divsi %14, %11 : i32
    %16 = arith.muli %13, %c128_i32 : i32
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %23 = tt.splat %16 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %24 = tt.splat %16 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %25 = tt.splat %16 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %26 = arith.addi %23, %17 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %27 = arith.addi %24, %19 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %28 = arith.addi %25, %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %29 = arith.muli %15, %c128_i32 : i32
    %30 = tt.splat %29 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %31 = tt.splat %29 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %32 = tt.splat %29 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %33 = arith.addi %30, %18 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %34 = arith.addi %31, %20 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %35 = arith.addi %32, %22 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %36 = tt.splat %arg3 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %37 = tt.splat %arg3 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %38 = arith.remsi %26, %36 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %39 = tt.splat %arg4 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %40 = tt.splat %arg4 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %41 = arith.remsi %33, %39 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %42 = arith.muli %1, %c32_i32 : i32
    %43 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %44 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %45 = tt.splat %42 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %46 = tt.splat %42 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %47 = arith.addi %45, %43 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %48 = arith.addi %46, %44 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %49 = tt.expand_dims %38 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    %50 = tt.splat %arg6 : (i32) -> tensor<128x1xi32, #blocked1>
    %51 = arith.muli %49, %50 : tensor<128x1xi32, #blocked1>
    %52 = tt.expand_dims %47 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x32xi32, #blocked1>
    %53 = tt.broadcast %51 : (tensor<128x1xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    %54 = tt.broadcast %52 : (tensor<1x32xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    %55 = arith.addi %53, %54 : tensor<128x32xi32, #blocked1>
    %56 = tt.splat %arg0 : (!tt.ptr<f16, 1>) -> tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
    %57 = tt.addptr %56, %55 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    %58 = tt.expand_dims %48 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %59 = tt.expand_dims %41 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi32, #blocked>
    %60 = tt.splat %arg7 : (i32) -> tensor<1x128xi32, #blocked>
    %61 = arith.muli %59, %60 : tensor<1x128xi32, #blocked>
    %62 = tt.broadcast %58 : (tensor<32x1xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    %63 = tt.broadcast %61 : (tensor<1x128xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    %64 = arith.addi %62, %63 : tensor<32x128xi32, #blocked>
    %65 = tt.splat %arg1 : (!tt.ptr<f16, 1>) -> tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    %66 = tt.addptr %65, %64 : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
    %67 = arith.addi %arg5, %c31_i32 : i32
    %68 = arith.divsi %67, %c32_i32 : i32
    %69 = arith.index_cast %68 : i32 to index
    %70:3 = scf.for %arg9 = %c0 to %69 step %c1 iter_args(%arg10 = %cst, %arg11 = %57, %arg12 = %66) -> (tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>) {
      %89 = tt.load %arg11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
      %90 = tt.load %arg12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
      %91 = triton_gpu.convert_layout %89 : (tensor<128x32xf16, #blocked1>) -> tensor<128x32xf16, #shared>
      %92 = triton_gpu.convert_layout %90 : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
      %93 = tt.dot %91, %92, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #shared> * tensor<32x128xf16, #shared1> -> tensor<128x128xf32, #mma>
      %94 = tt.addptr %arg11, %cst_1 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
      %95 = tt.addptr %arg12, %cst_0 : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
      scf.yield %93, %94, %95 : tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    }
    %71 = arith.truncf %70#0 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %72 = tt.expand_dims %27 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi32, #blocked2>
    %73 = tt.splat %arg8 : (i32) -> tensor<128x1xi32, #blocked2>
    %74 = arith.muli %72, %73 : tensor<128x1xi32, #blocked2>
    %75 = tt.expand_dims %34 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi32, #blocked2>
    %76 = tt.broadcast %74 : (tensor<128x1xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    %77 = tt.broadcast %75 : (tensor<1x128xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    %78 = arith.addi %76, %77 : tensor<128x128xi32, #blocked2>
    %79 = tt.splat %arg2 : (!tt.ptr<f16, 1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
    %80 = tt.addptr %79, %78 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi32, #blocked2>
    %81 = arith.cmpi "slt", %28, %37 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %82 = tt.expand_dims %81 {axis = 1 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi1, #blocked2>
    %83 = arith.cmpi "slt", %35, %40 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %84 = tt.expand_dims %83 {axis = 0 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi1, #blocked2>
    %85 = tt.broadcast %82 : (tensor<128x1xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    %86 = tt.broadcast %84 : (tensor<1x128xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    %87 = arith.andi %85, %86 : tensor<128x128xi1, #blocked2>
    %88 = triton_gpu.convert_layout %71 : (tensor<128x128xf16, #mma>) -> tensor<128x128xf16, #blocked2>
    tt.store %80, %88, %87 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked2>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: "triton_gpu.enable-warp-specialization" = 1 : i32
  // CHECK-LABEL: @nested_for_gemm
  tt.func public @nested_for_gemm(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<32> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<32> : tensor<128x32xi32, #blocked1>
    %c31_i32 = arith.constant 31 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.addi %arg3, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.addi %arg4, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %0, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.cmpi slt, %9, %c8_i32 : i32
    %11 = arith.select %10, %9, %c8_i32 : i32
    %12 = arith.remsi %0, %11 : i32
    %13 = arith.addi %8, %12 : i32
    %14 = arith.remsi %0, %6 : i32
    %15 = arith.divsi %14, %11 : i32
    %16 = arith.muli %13, %c128_i32 : i32
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %23 = tt.splat %16 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %24 = tt.splat %16 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %25 = tt.splat %16 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %26 = arith.addi %23, %17 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %27 = arith.addi %24, %19 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %28 = arith.addi %25, %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %29 = arith.muli %15, %c128_i32 : i32
    %30 = tt.splat %29 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %31 = tt.splat %29 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %32 = tt.splat %29 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %33 = arith.addi %30, %18 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %34 = arith.addi %31, %20 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %35 = arith.addi %32, %22 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %36 = tt.splat %arg3 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %37 = tt.splat %arg3 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %38 = arith.remsi %26, %36 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %39 = tt.splat %arg4 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %40 = tt.splat %arg4 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %41 = arith.remsi %33, %39 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %42 = arith.muli %1, %c32_i32 : i32
    %43 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %44 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %45 = tt.splat %42 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %46 = tt.splat %42 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %47 = arith.addi %45, %43 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %48 = arith.addi %46, %44 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %49 = tt.expand_dims %38 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    %50 = tt.splat %arg6 : (i32) -> tensor<128x1xi32, #blocked1>
    %51 = arith.muli %49, %50 : tensor<128x1xi32, #blocked1>
    %52 = tt.expand_dims %47 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x32xi32, #blocked1>
    %53 = tt.broadcast %51 : (tensor<128x1xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    %54 = tt.broadcast %52 : (tensor<1x32xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    %55 = arith.addi %53, %54 : tensor<128x32xi32, #blocked1>
    %56 = tt.splat %arg0 : (!tt.ptr<f16, 1>) -> tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
    %57 = tt.addptr %56, %55 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    %58 = tt.expand_dims %48 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %59 = tt.expand_dims %41 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi32, #blocked>
    %60 = tt.splat %arg7 : (i32) -> tensor<1x128xi32, #blocked>
    %61 = arith.muli %59, %60 : tensor<1x128xi32, #blocked>
    %62 = tt.broadcast %58 : (tensor<32x1xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    %63 = tt.broadcast %61 : (tensor<1x128xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    %64 = arith.addi %62, %63 : tensor<32x128xi32, #blocked>
    %65 = tt.splat %arg1 : (!tt.ptr<f16, 1>) -> tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    %66 = tt.addptr %65, %64 : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
    %67 = arith.addi %arg5, %c31_i32 : i32
    %68 = arith.divsi %67, %c32_i32 : i32
    %69 = arith.index_cast %68 : i32 to index
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x32xf16, #shared>
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<32x128xf16, #shared1>
    %70:3 = scf.for %arg9 = %c0 to %69 step %c1 iter_args(%arg10 = %cst, %arg11 = %57, %arg12 = %66) -> (tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>) {
      %96:2 = scf.for %arg13 = %c0 to %69 step %c1 iter_args(%arg14 = %cst_2, %arg15 = %cst_3) -> (tensor<128x32xf16, #shared>, tensor<32x128xf16, #shared1>) {
        %89 = tt.load %arg11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
        %90 = tt.load %arg12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
        %91 = triton_gpu.convert_layout %89 : (tensor<128x32xf16, #blocked1>) -> tensor<128x32xf16, #shared>
        %92 = triton_gpu.convert_layout %90 : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
        scf.yield %91, %92 : tensor<128x32xf16, #shared>, tensor<32x128xf16, #shared1>
      }
      %93 = tt.dot %96#0, %96#1, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #shared> * tensor<32x128xf16, #shared1> -> tensor<128x128xf32, #mma>
      %94 = tt.addptr %arg11, %cst_1 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
      %95 = tt.addptr %arg12, %cst_0 : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
      scf.yield %93, %94, %95 : tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    }
    %71 = arith.truncf %70#0 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %72 = tt.expand_dims %27 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi32, #blocked2>
    %73 = tt.splat %arg8 : (i32) -> tensor<128x1xi32, #blocked2>
    %74 = arith.muli %72, %73 : tensor<128x1xi32, #blocked2>
    %75 = tt.expand_dims %34 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi32, #blocked2>
    %76 = tt.broadcast %74 : (tensor<128x1xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    %77 = tt.broadcast %75 : (tensor<1x128xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    %78 = arith.addi %76, %77 : tensor<128x128xi32, #blocked2>
    %79 = tt.splat %arg2 : (!tt.ptr<f16, 1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
    %80 = tt.addptr %79, %78 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi32, #blocked2>
    %81 = arith.cmpi "slt", %28, %37 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %82 = tt.expand_dims %81 {axis = 1 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi1, #blocked2>
    %83 = arith.cmpi "slt", %35, %40 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %84 = tt.expand_dims %83 {axis = 0 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi1, #blocked2>
    %85 = tt.broadcast %82 : (tensor<128x1xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    %86 = tt.broadcast %84 : (tensor<1x128xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    %87 = arith.andi %85, %86 : tensor<128x128xi1, #blocked2>
    %88 = triton_gpu.convert_layout %71 : (tensor<128x128xf16, #mma>) -> tensor<128x128xf16, #blocked2>
    tt.store %80, %88, %87 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked2>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: "triton_gpu.enable-warp-specialization" = 1 : i32
  // CHECK-LABEL: @if_in_for_gemm
  tt.func public @if_in_for_gemm(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %cst_0 = arith.constant dense<32> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<32> : tensor<128x32xi32, #blocked1>
    %c31_i32 = arith.constant 31 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.addi %arg3, %c127_i32 : i32
    %3 = arith.divsi %2, %c128_i32 : i32
    %4 = arith.addi %arg4, %c127_i32 : i32
    %5 = arith.divsi %4, %c128_i32 : i32
    %6 = arith.muli %5, %c8_i32 : i32
    %7 = arith.divsi %0, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %3, %8 : i32
    %10 = arith.cmpi slt, %9, %c8_i32 : i32
    %11 = arith.select %10, %9, %c8_i32 : i32
    %12 = arith.remsi %0, %11 : i32
    %13 = arith.addi %8, %12 : i32
    %14 = arith.remsi %0, %6 : i32
    %15 = arith.divsi %14, %11 : i32
    %16 = arith.muli %13, %c128_i32 : i32
    %17 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %19 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %21 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %22 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %23 = tt.splat %16 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %24 = tt.splat %16 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %25 = tt.splat %16 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %26 = arith.addi %23, %17 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %27 = arith.addi %24, %19 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %28 = arith.addi %25, %21 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %29 = arith.muli %15, %c128_i32 : i32
    %30 = tt.splat %29 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %31 = tt.splat %29 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %32 = tt.splat %29 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %33 = arith.addi %30, %18 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %34 = arith.addi %31, %20 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %35 = arith.addi %32, %22 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %36 = tt.splat %arg3 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %37 = tt.splat %arg3 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %38 = arith.remsi %26, %36 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %39 = tt.splat %arg4 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %40 = tt.splat %arg4 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %41 = arith.remsi %33, %39 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %42 = arith.muli %1, %c32_i32 : i32
    %43 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %44 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %45 = tt.splat %42 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %46 = tt.splat %42 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %47 = arith.addi %45, %43 : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %48 = arith.addi %46, %44 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %49 = tt.expand_dims %38 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    %50 = tt.splat %arg6 : (i32) -> tensor<128x1xi32, #blocked1>
    %51 = arith.muli %49, %50 : tensor<128x1xi32, #blocked1>
    %52 = tt.expand_dims %47 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x32xi32, #blocked1>
    %53 = tt.broadcast %51 : (tensor<128x1xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    %54 = tt.broadcast %52 : (tensor<1x32xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    %55 = arith.addi %53, %54 : tensor<128x32xi32, #blocked1>
    %56 = tt.splat %arg0 : (!tt.ptr<f16, 1>) -> tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
    %57 = tt.addptr %56, %55 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    %58 = tt.expand_dims %48 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %59 = tt.expand_dims %41 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi32, #blocked>
    %60 = tt.splat %arg7 : (i32) -> tensor<1x128xi32, #blocked>
    %61 = arith.muli %59, %60 : tensor<1x128xi32, #blocked>
    %62 = tt.broadcast %58 : (tensor<32x1xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    %63 = tt.broadcast %61 : (tensor<1x128xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    %64 = arith.addi %62, %63 : tensor<32x128xi32, #blocked>
    %65 = tt.splat %arg1 : (!tt.ptr<f16, 1>) -> tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    %66 = tt.addptr %65, %64 : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
    %67 = arith.addi %arg5, %c31_i32 : i32
    %68 = arith.divsi %67, %c32_i32 : i32
    %69 = arith.index_cast %68 : i32 to index
    %70:3 = scf.for %arg9 = %c0 to %69 step %c1 iter_args(%arg10 = %cst, %arg11 = %57, %arg12 = %66) -> (tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>) {
      %arg9_i32 = arith.index_cast %arg9 : index to i32
      %96 = arith.cmpi ne, %arg9_i32, %c31_i32 : i32
      %89 = scf.if %96 -> (tensor<128x32xf16, #blocked1>) {
        %r0_0 = arith.select %96, %c31_i32, %c127_i32 : i32
        %r0_1 = tt.splat %r0_0 : (i32) -> tensor<128x32xi32, #blocked1>
        %new_addr = tt.addptr %arg11, %r0_1 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
        %new_89 = tt.load %new_addr {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
        scf.yield %new_89 : tensor<128x32xf16, #blocked1>
      } else {
        %new_89 = tt.load %arg11 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
        scf.yield %new_89 : tensor<128x32xf16, #blocked1>
      }
      %90 = tt.load %arg12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
      %91 = triton_gpu.convert_layout %89 : (tensor<128x32xf16, #blocked1>) -> tensor<128x32xf16, #shared>
      %92 = triton_gpu.convert_layout %90 : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
      %93 = tt.dot %91, %92, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<128x32xf16, #shared> * tensor<32x128xf16, #shared1> -> tensor<128x128xf32, #mma>
      %base_94 = scf.if %96 -> (tensor<128x32x!tt.ptr<f16, 1>, #blocked1>) {
        %r1_0 = arith.select %96, %c31_i32, %c127_i32 : i32
        %r1_1 = tt.splat %r1_0 : (i32) -> tensor<128x32xi32, #blocked1>
        %98 = tt.addptr %arg11, %r1_1 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
        scf.yield %98 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
      } else {
        %98 = tt.addptr %arg11, %cst_1 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
        scf.yield %98 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
      }
      %94 = tt.addptr %base_94, %cst_1 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
      %95 = tt.addptr %arg12, %cst_0 : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
      scf.yield %93, %94, %95 : tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    }
    %71 = arith.truncf %70#0 : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    %72 = tt.expand_dims %27 {axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi32, #blocked2>
    %73 = tt.splat %arg8 : (i32) -> tensor<128x1xi32, #blocked2>
    %74 = arith.muli %72, %73 : tensor<128x1xi32, #blocked2>
    %75 = tt.expand_dims %34 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi32, #blocked2>
    %76 = tt.broadcast %74 : (tensor<128x1xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    %77 = tt.broadcast %75 : (tensor<1x128xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    %78 = arith.addi %76, %77 : tensor<128x128xi32, #blocked2>
    %79 = tt.splat %arg2 : (!tt.ptr<f16, 1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
    %80 = tt.addptr %79, %78 : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi32, #blocked2>
    %81 = arith.cmpi "slt", %28, %37 : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %82 = tt.expand_dims %81 {axis = 1 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi1, #blocked2>
    %83 = arith.cmpi "slt", %35, %40 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %84 = tt.expand_dims %83 {axis = 0 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi1, #blocked2>
    %85 = tt.broadcast %82 : (tensor<128x1xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    %86 = tt.broadcast %84 : (tensor<1x128xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    %87 = arith.andi %85, %86 : tensor<128x128xi1, #blocked2>
    %88 = triton_gpu.convert_layout %71 : (tensor<128x128xf16, #mma>) -> tensor<128x128xf16, #blocked2>
    tt.store %80, %88, %87 {cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked2>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: "triton_gpu.enable-warp-specialization" = 1 : i32
  // CHECK-LABEL: @tma_warp_specialized_matmul
  tt.func public @tma_warp_specialized_matmul(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c63_i32 = arith.constant 63 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg4, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    %5 = arith.muli %3, %c64_i32 : i32
    %6 = arith.muli %4, %c64_i32 : i32
    %7 = arith.extsi %arg3 : i32 to i64
    %8 = arith.extsi %arg5 : i32 to i64
    %9 = arith.extsi %arg6 : i32 to i64
    %10 = tt.make_tensor_ptr %arg0, [%7, %8], [%9, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #blocked>, 1>
    %11 = arith.extsi %arg4 : i32 to i64
    %12 = arith.extsi %arg7 : i32 to i64
    %13 = tt.make_tensor_ptr %arg1, [%8, %11], [%c1_i64, %12], [%c0_i32, %6] {order = array<i32: 0, 1>} : <tensor<16x64xf16, #blocked1>, 1>
    %14:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c16_i32 iter_args(%arg10 = %cst, %arg11 = %10, %arg12 = %13) -> (tensor<64x64xf32, #mma>, !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>)  : i32 {
      %46 = tt.load %arg11 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x16xf16, #blocked>, 1> -> tensor<64x16xf16, #blocked2>
      %47 = tt.load %arg12 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x64xf16, #blocked1>, 1> -> tensor<16x64xf16, #blocked3>
      %48 = triton_gpu.convert_layout %46 : (tensor<64x16xf16, #blocked2>) -> tensor<64x16xf16, #shared>
      %49 = triton_gpu.convert_layout %47 : (tensor<16x64xf16, #blocked3>) -> tensor<16x64xf16, #shared1>
      %50 = tt.dot %48, %49, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x16xf16, #shared> * tensor<16x64xf16, #shared1> -> tensor<64x64xf32, #mma>
      %51 = tt.advance %arg11, [%c0_i32, %c16_i32] : <tensor<64x16xf16, #blocked>, 1>
      %52 = tt.advance %arg12, [%c16_i32, %c0_i32] : <tensor<16x64xf16, #blocked1>, 1>
      scf.yield %50, %51, %52 : tensor<64x64xf32, #mma>, !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>
    }
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %19 = tt.splat %5 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %20 = tt.splat %5 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %21 = arith.addi %16, %19 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %22 = arith.addi %18, %20 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %23 = tt.splat %6 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %24 = tt.splat %6 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %25 = arith.addi %15, %23 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %26 = arith.addi %17, %24 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %27 = tt.expand_dims %21 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<64x1xi32, #blocked4>
    %28 = tt.splat %arg8 : (i32) -> tensor<64x1xi32, #blocked4>
    %29 = arith.muli %27, %28 : tensor<64x1xi32, #blocked4>
    %30 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked4>
    %31 = tt.addptr %30, %29 : tensor<64x1x!tt.ptr<f32, 1>, #blocked4>, tensor<64x1xi32, #blocked4>
    %32 = tt.expand_dims %25 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x64xi32, #blocked4>
    %33 = tt.broadcast %31 : (tensor<64x1x!tt.ptr<f32, 1>, #blocked4>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked4>
    %34 = tt.broadcast %32 : (tensor<1x64xi32, #blocked4>) -> tensor<64x64xi32, #blocked4>
    %35 = tt.addptr %33, %34 : tensor<64x64x!tt.ptr<f32, 1>, #blocked4>, tensor<64x64xi32, #blocked4>
    %36 = tt.splat %arg3 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %37 = arith.cmpi "slt", %22, %36 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %38 = tt.expand_dims %37 {axis = 1 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<64x1xi1, #blocked4>
    %39 = tt.splat %arg4 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %40 = arith.cmpi "slt", %26, %39 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %41 = tt.expand_dims %40 {axis = 0 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x64xi1, #blocked4>
    %42 = tt.broadcast %38 : (tensor<64x1xi1, #blocked4>) -> tensor<64x64xi1, #blocked4>
    %43 = tt.broadcast %41 : (tensor<1x64xi1, #blocked4>) -> tensor<64x64xi1, #blocked4>
    %44 = arith.andi %42, %43 : tensor<64x64xi1, #blocked4>
    %45 = triton_gpu.convert_layout %14#0 : (tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #blocked4>
    tt.store %35, %45, %44 {cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf32, #blocked4>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: "triton_gpu.enable-warp-specialization" = 1 : i32
  // CHECK-LABEL: @store_after_load
  tt.func public @store_after_load(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c63_i32 = arith.constant 63 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg4, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    %5 = arith.muli %3, %c64_i32 : i32
    %6 = arith.muli %4, %c64_i32 : i32
    %7 = arith.extsi %arg3 : i32 to i64
    %8 = arith.extsi %arg5 : i32 to i64
    %9 = arith.extsi %arg6 : i32 to i64
    %10 = tt.make_tensor_ptr %arg0, [%7, %8], [%9, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #blocked>, 1>
    %11 = arith.extsi %arg4 : i32 to i64
    %12 = arith.extsi %arg7 : i32 to i64
    %13 = tt.make_tensor_ptr %arg1, [%8, %11], [%c1_i64, %12], [%c0_i32, %6] {order = array<i32: 0, 1>} : <tensor<16x64xf16, #blocked1>, 1>
    %14:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c16_i32 iter_args(%arg10 = %cst, %arg11 = %10, %arg12 = %13) -> (tensor<64x64xf32, #mma>, !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>)  : i32 {
      %46 = tt.load %arg11 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x16xf16, #blocked>, 1> -> tensor<64x16xf16, #blocked2>
      %47 = tt.load %arg12 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x64xf16, #blocked1>, 1> -> tensor<16x64xf16, #blocked3>
      %48 = triton_gpu.convert_layout %46 : (tensor<64x16xf16, #blocked2>) -> tensor<64x16xf16, #shared>
      %49 = triton_gpu.convert_layout %47 : (tensor<16x64xf16, #blocked3>) -> tensor<16x64xf16, #shared1>
      %50 = tt.dot %48, %49, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x16xf16, #shared> * tensor<16x64xf16, #shared1> -> tensor<64x64xf32, #mma>
      %51 = tt.advance %arg11, [%c0_i32, %c16_i32] : <tensor<64x16xf16, #blocked>, 1>
      %52 = tt.advance %arg12, [%c16_i32, %c0_i32] : <tensor<16x64xf16, #blocked1>, 1>
      scf.yield %50, %51, %52 : tensor<64x64xf32, #mma>, !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>
    }
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %19 = tt.splat %5 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %20 = tt.splat %5 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %21 = arith.addi %16, %19 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %22 = arith.addi %18, %20 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %23 = tt.splat %6 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %24 = tt.splat %6 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %25 = arith.addi %15, %23 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %26 = arith.addi %17, %24 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %27 = tt.expand_dims %21 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<64x1xi32, #blocked4>
    %28 = tt.splat %arg8 : (i32) -> tensor<64x1xi32, #blocked4>
    %29 = arith.muli %27, %28 : tensor<64x1xi32, #blocked4>
    %30 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked4>
    %31 = tt.addptr %30, %29 : tensor<64x1x!tt.ptr<f32, 1>, #blocked4>, tensor<64x1xi32, #blocked4>
    %32 = tt.expand_dims %25 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x64xi32, #blocked4>
    %33 = tt.broadcast %31 : (tensor<64x1x!tt.ptr<f32, 1>, #blocked4>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked4>
    %34 = tt.broadcast %32 : (tensor<1x64xi32, #blocked4>) -> tensor<64x64xi32, #blocked4>
    %35 = tt.addptr %33, %34 : tensor<64x64x!tt.ptr<f32, 1>, #blocked4>, tensor<64x64xi32, #blocked4>
    %36 = tt.splat %arg3 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %37 = arith.cmpi "slt", %22, %36 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %38 = tt.expand_dims %37 {axis = 1 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<64x1xi1, #blocked4>
    %39 = tt.splat %arg4 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %40 = arith.cmpi "slt", %26, %39 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %41 = tt.expand_dims %40 {axis = 0 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x64xi1, #blocked4>
    %42 = tt.broadcast %38 : (tensor<64x1xi1, #blocked4>) -> tensor<64x64xi1, #blocked4>
    %43 = tt.broadcast %41 : (tensor<1x64xi1, #blocked4>) -> tensor<64x64xi1, #blocked4>
    %44 = arith.andi %42, %43 : tensor<64x64xi1, #blocked4>
    %45 = triton_gpu.convert_layout %14#0 : (tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #blocked4>
    %46 = tt.load %35, %44 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, #blocked4>
    %47 = arith.addf %45, %46 : tensor<64x64xf32, #blocked4>
    tt.store %35, %47, %44 {cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf32, #blocked4>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: "triton_gpu.enable-warp-specialization" = 0 : i32
  // CHECK-LABEL: @load_after_store
  tt.func public @load_after_store(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c63_i32 = arith.constant 63 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg4, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    %5 = arith.muli %3, %c64_i32 : i32
    %6 = arith.muli %4, %c64_i32 : i32
    %7 = arith.extsi %arg3 : i32 to i64
    %8 = arith.extsi %arg5 : i32 to i64
    %9 = arith.extsi %arg6 : i32 to i64
    %10 = tt.make_tensor_ptr %arg0, [%7, %8], [%9, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #blocked>, 1>
    %11 = arith.extsi %arg4 : i32 to i64
    %12 = arith.extsi %arg7 : i32 to i64
    %13 = tt.make_tensor_ptr %arg1, [%8, %11], [%c1_i64, %12], [%c0_i32, %6] {order = array<i32: 0, 1>} : <tensor<16x64xf16, #blocked1>, 1>
    %14:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c16_i32 iter_args(%arg10 = %cst, %arg11 = %10, %arg12 = %13) -> (tensor<64x64xf32, #mma>, !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>)  : i32 {
      %46 = tt.load %arg11 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x16xf16, #blocked>, 1> -> tensor<64x16xf16, #blocked2>
      %47 = tt.load %arg12 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x64xf16, #blocked1>, 1> -> tensor<16x64xf16, #blocked3>
      %48 = triton_gpu.convert_layout %46 : (tensor<64x16xf16, #blocked2>) -> tensor<64x16xf16, #shared>
      %49 = triton_gpu.convert_layout %47 : (tensor<16x64xf16, #blocked3>) -> tensor<16x64xf16, #shared1>
      %50 = tt.dot %48, %49, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x16xf16, #shared> * tensor<16x64xf16, #shared1> -> tensor<64x64xf32, #mma>
      %51 = tt.advance %arg11, [%c0_i32, %c16_i32] : <tensor<64x16xf16, #blocked>, 1>
      %52 = tt.advance %arg12, [%c16_i32, %c0_i32] : <tensor<16x64xf16, #blocked1>, 1>
      scf.yield %50, %51, %52 : tensor<64x64xf32, #mma>, !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>
    }
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %19 = tt.splat %5 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %20 = tt.splat %5 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %21 = arith.addi %16, %19 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %22 = arith.addi %18, %20 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %23 = tt.splat %6 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %24 = tt.splat %6 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %25 = arith.addi %15, %23 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %26 = arith.addi %17, %24 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %27 = tt.expand_dims %21 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<64x1xi32, #blocked4>
    %28 = tt.splat %arg8 : (i32) -> tensor<64x1xi32, #blocked4>
    %29 = arith.muli %27, %28 : tensor<64x1xi32, #blocked4>
    %30 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked4>
    %31 = tt.addptr %30, %29 : tensor<64x1x!tt.ptr<f32, 1>, #blocked4>, tensor<64x1xi32, #blocked4>
    %32 = tt.expand_dims %25 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x64xi32, #blocked4>
    %33 = tt.broadcast %31 : (tensor<64x1x!tt.ptr<f32, 1>, #blocked4>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked4>
    %34 = tt.broadcast %32 : (tensor<1x64xi32, #blocked4>) -> tensor<64x64xi32, #blocked4>
    %35 = tt.addptr %33, %34 : tensor<64x64x!tt.ptr<f32, 1>, #blocked4>, tensor<64x64xi32, #blocked4>
    %36 = tt.splat %arg3 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %37 = arith.cmpi "slt", %22, %36 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %38 = tt.expand_dims %37 {axis = 1 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<64x1xi1, #blocked4>
    %39 = tt.splat %arg4 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %40 = arith.cmpi "slt", %26, %39 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %41 = tt.expand_dims %40 {axis = 0 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x64xi1, #blocked4>
    %42 = tt.broadcast %38 : (tensor<64x1xi1, #blocked4>) -> tensor<64x64xi1, #blocked4>
    %43 = tt.broadcast %41 : (tensor<1x64xi1, #blocked4>) -> tensor<64x64xi1, #blocked4>
    %44 = arith.andi %42, %43 : tensor<64x64xi1, #blocked4>
    %45 = triton_gpu.convert_layout %14#0 : (tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #blocked4>
    %46 = tt.load %35, %44 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, #blocked4>
    %47 = arith.addf %45, %46 : tensor<64x64xf32, #blocked4>
    tt.store %35, %47, %44 {cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf32, #blocked4>
    %48 = tt.load %35, %44 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, #blocked4>
    %49 = arith.addf %45, %48 : tensor<64x64xf32, #blocked4>
    tt.store %35, %49, %44 {cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf32, #blocked4>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: "triton_gpu.enable-warp-specialization" = 0 : i32
  // CHECK-LABEL: @global_bar
  tt.func public @global_bar(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg100: !tt.ptr<i32, 1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c63_i32 = arith.constant 63 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg4, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.divsi %0, %2 : i32
    %4 = arith.remsi %0, %2 : i32
    %5 = arith.muli %3, %c64_i32 : i32
    %6 = arith.muli %4, %c64_i32 : i32
    %7 = arith.extsi %arg3 : i32 to i64
    %8 = arith.extsi %arg5 : i32 to i64
    %9 = arith.extsi %arg6 : i32 to i64
    %10 = tt.make_tensor_ptr %arg0, [%7, %8], [%9, %c1_i64], [%5, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #blocked>, 1>
    %11 = arith.extsi %arg4 : i32 to i64
    %12 = arith.extsi %arg7 : i32 to i64
    %13 = tt.make_tensor_ptr %arg1, [%8, %11], [%c1_i64, %12], [%c0_i32, %6] {order = array<i32: 0, 1>} : <tensor<16x64xf16, #blocked1>, 1>
    %14:3 = scf.for %arg9 = %c0_i32 to %arg5 step %c16_i32 iter_args(%arg10 = %cst, %arg11 = %10, %arg12 = %13) -> (tensor<64x64xf32, #mma>, !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>)  : i32 {
      %46 = tt.load %arg11 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x16xf16, #blocked>, 1> -> tensor<64x16xf16, #blocked2>
      %47 = tt.load %arg12 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x64xf16, #blocked1>, 1> -> tensor<16x64xf16, #blocked3>
      %48 = triton_gpu.convert_layout %46 : (tensor<64x16xf16, #blocked2>) -> tensor<64x16xf16, #shared>
      %49 = triton_gpu.convert_layout %47 : (tensor<16x64xf16, #blocked3>) -> tensor<16x64xf16, #shared1>
      %50 = tt.dot %48, %49, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x16xf16, #shared> * tensor<16x64xf16, #shared1> -> tensor<64x64xf32, #mma>
      %51 = tt.advance %arg11, [%c0_i32, %c16_i32] : <tensor<64x16xf16, #blocked>, 1>
      %52 = tt.advance %arg12, [%c16_i32, %c0_i32] : <tensor<16x64xf16, #blocked1>, 1>
      scf.yield %50, %51, %52 : tensor<64x64xf32, #mma>, !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>
    }
    %15 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %19 = tt.splat %5 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %20 = tt.splat %5 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %21 = arith.addi %16, %19 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %22 = arith.addi %18, %20 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %23 = tt.splat %6 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %24 = tt.splat %6 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %25 = arith.addi %15, %23 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %26 = arith.addi %17, %24 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %27 = tt.expand_dims %21 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<64x1xi32, #blocked4>
    %28 = tt.splat %arg8 : (i32) -> tensor<64x1xi32, #blocked4>
    %29 = arith.muli %27, %28 : tensor<64x1xi32, #blocked4>
    %30 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked4>
    %31 = tt.addptr %30, %29 : tensor<64x1x!tt.ptr<f32, 1>, #blocked4>, tensor<64x1xi32, #blocked4>
    %32 = tt.expand_dims %25 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x64xi32, #blocked4>
    %33 = tt.broadcast %31 : (tensor<64x1x!tt.ptr<f32, 1>, #blocked4>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked4>
    %34 = tt.broadcast %32 : (tensor<1x64xi32, #blocked4>) -> tensor<64x64xi32, #blocked4>
    %35 = tt.addptr %33, %34 : tensor<64x64x!tt.ptr<f32, 1>, #blocked4>, tensor<64x64xi32, #blocked4>
    %36 = tt.splat %arg3 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %37 = arith.cmpi "slt", %22, %36 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>
    %38 = tt.expand_dims %37 {axis = 1 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked4}>>) -> tensor<64x1xi1, #blocked4>
    %39 = tt.splat %arg4 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %40 = arith.cmpi "slt", %26, %39 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>
    %41 = tt.expand_dims %40 {axis = 0 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked4}>>) -> tensor<1x64xi1, #blocked4>
    %42 = tt.broadcast %38 : (tensor<64x1xi1, #blocked4>) -> tensor<64x64xi1, #blocked4>
    %43 = tt.broadcast %41 : (tensor<1x64xi1, #blocked4>) -> tensor<64x64xi1, #blocked4>
    %44 = arith.andi %42, %43 : tensor<64x64xi1, #blocked4>
    %45 = triton_gpu.convert_layout %14#0 : (tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #blocked4>
    "tt.atomic_cas"(%arg100, %c0_i32, %c1_i32) {sem = 1 : i32, scope = 1 : i32}: (!tt.ptr<i32, 1>, i32, i32) -> i32
    %46 = tt.load %35, %44 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x64xf32, #blocked4>
    %47 = arith.addf %45, %46 : tensor<64x64xf32, #blocked4>
    tt.store %35, %47, %44 {cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf32, #blocked4>
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: "triton_gpu.enable-warp-specialization" = 1 : i32
  // CHECK-LABEL: @store_in_nested_for
  tt.func public @store_in_nested_for(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c63_i32 = arith.constant 63 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c114_i32 = arith.constant 114 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.addi %arg4, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.divsi %0, %4 : i32
    %9 = arith.remsi %0, %4 : i32
    %10 = arith.muli %8, %c256_i32 : i32
    %11 = arith.muli %9, %c128_i32 : i32
    %12 = arith.extsi %arg3 : i32 to i64
    %13 = arith.extsi %arg5 : i32 to i64
    %14 = arith.extsi %arg6 : i32 to i64
    %15 = tt.make_tensor_ptr %arg0, [%12, %13], [%14, %c1_i64], [%10, %c0_i32] {order = array<i32: 1, 0>} : <tensor<256x64xf16, #blocked>, 1>
    %16 = arith.extsi %arg4 : i32 to i64
    %17 = arith.extsi %arg7 : i32 to i64
    %18 = tt.make_tensor_ptr %arg1, [%13, %16], [%c1_i64, %17], [%c0_i32, %11] {order = array<i32: 0, 1>} : <tensor<64x128xf16, #blocked1>, 1>
    %19 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %20 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %21 = tt.splat %arg8 : (i32) -> tensor<256x1xi32, #blocked2>
    %22 = tt.splat %arg2 : (!tt.ptr<f32, 1>) -> tensor<256x1x!tt.ptr<f32, 1>, #blocked2>
    %23:4 = scf.for %arg9 = %0 to %7 step %c114_i32 iter_args(%arg10 = %15, %arg11 = %18, %arg12 = %8, %arg13 = %9) -> (!tt.ptr<tensor<256x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>, i32, i32)  : i32 {
      %24 = arith.divsi %arg9, %4 : i32
      %25 = arith.remsi %arg9, %4 : i32
      %26 = arith.cmpi "sge", %arg9, %c114_i32 : i32
      %27:2 = scf.if %26 -> (!tt.ptr<tensor<256x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>) {
        %43 = arith.subi %24, %arg12 : i32
        %44 = arith.muli %43, %c256_i32 : i32
        %45 = arith.subi %c0_i32, %6 : i32
        %46 = arith.muli %45, %c64_i32 : i32
        %47 = tt.advance %arg10, [%44, %46] : <tensor<256x64xf16, #blocked>, 1>
        %48 = arith.subi %25, %arg13 : i32
        %49 = arith.muli %48, %c128_i32 : i32
        %50 = tt.advance %arg11, [%46, %49] : <tensor<64x128xf16, #blocked1>, 1>
        scf.yield %47, %50 : !tt.ptr<tensor<256x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>
      } else {
        scf.yield %arg10, %arg11 : !tt.ptr<tensor<256x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>
      }
      %28:3 = scf.for %arg14 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg15 = %cst, %arg16 = %27#0, %arg17 = %27#1) -> (tensor<256x128xf32, #mma>, !tt.ptr<tensor<256x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>)  : i32 {
        %43 = tt.load %arg16 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<256x64xf16, #blocked>, 1> -> tensor<256x64xf16, #blocked3>
        %44 = tt.load %arg17 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x128xf16, #blocked1>, 1> -> tensor<64x128xf16, #blocked4>
        %45 = triton_gpu.convert_layout %43 : (tensor<256x64xf16, #blocked3>) -> tensor<256x64xf16, #shared>
        %46 = triton_gpu.convert_layout %44 : (tensor<64x128xf16, #blocked4>) -> tensor<64x128xf16, #shared1>
        %47 = tt.dot %45, %46, %arg15 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x64xf16, #shared> * tensor<64x128xf16, #shared1> -> tensor<256x128xf32, #mma>
        %48 = tt.advance %arg16, [%c0_i32, %c64_i32] : <tensor<256x64xf16, #blocked>, 1>
        %49 = tt.advance %arg17, [%c64_i32, %c0_i32] : <tensor<64x128xf16, #blocked1>, 1>
        scf.yield %47, %48, %49 : tensor<256x128xf32, #mma>, !tt.ptr<tensor<256x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>
      }
      %29 = arith.muli %24, %c256_i32 : i32
      %30 = tt.splat %29 : (i32) -> tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %31 = arith.addi %19, %30 : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %32 = arith.muli %25, %c128_i32 : i32
      %33 = tt.splat %32 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %34 = arith.addi %20, %33 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %35 = tt.expand_dims %31 {axis = 1 : i32} : (tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<256x1xi32, #blocked2>
      %36 = arith.muli %35, %21 : tensor<256x1xi32, #blocked2>
      %37 = tt.addptr %22, %36 : tensor<256x1x!tt.ptr<f32, 1>, #blocked2>, tensor<256x1xi32, #blocked2>
      %38 = tt.expand_dims %34 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi32, #blocked2>
      %39 = tt.broadcast %37 : (tensor<256x1x!tt.ptr<f32, 1>, #blocked2>) -> tensor<256x128x!tt.ptr<f32, 1>, #blocked2>
      %40 = tt.broadcast %38 : (tensor<1x128xi32, #blocked2>) -> tensor<256x128xi32, #blocked2>
      %41 = tt.addptr %39, %40 : tensor<256x128x!tt.ptr<f32, 1>, #blocked2>, tensor<256x128xi32, #blocked2>
      %42 = triton_gpu.convert_layout %28#0 : (tensor<256x128xf32, #mma>) -> tensor<256x128xf32, #blocked2>
      tt.store %41, %42 {cache = 1 : i32, evict = 1 : i32} : tensor<256x128xf32, #blocked2>
      scf.yield %28#1, %28#2, %24, %25 : !tt.ptr<tensor<256x64xf16, #blocked>, 1>, !tt.ptr<tensor<64x128xf16, #blocked1>, 1>, i32, i32
    }
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: "triton_gpu.enable-warp-specialization" = 1 : i32
  // CHECK-LABEL: @matched_load_type
  tt.func public @matched_load_type(
      %arg0: !tt.ptr<tensor<256x64xf16, #blocked>, 1>,
      %arg1: !tt.ptr<tensor<64x128xf16, #blocked1>, 1>,
      %arg2: tensor<256x128x!tt.ptr<f32, 1>, #blocked2>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    scf.for %iv = %c0 to %c10 step %c1 iter_args() -> () {
      %a = tt.load %arg0 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<256x64xf16, #blocked>, 1> -> tensor<256x64xf16, #blocked3>
      %b = tt.load %arg1 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x128xf16, #blocked1>, 1> -> tensor<64x128xf16, #blocked4>
      %shm_a = triton_gpu.convert_layout %a : (tensor<256x64xf16, #blocked3>) -> tensor<256x64xf16, #shared>
      %shm_b = triton_gpu.convert_layout %b : (tensor<64x128xf16, #blocked4>) -> tensor<64x128xf16, #shared1>
      %d = tt.dot %shm_a, %shm_b, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x64xf16, #shared> * tensor<64x128xf16, #shared1> -> tensor<256x128xf32, #mma>
      %out = triton_gpu.convert_layout %d : (tensor<256x128xf32, #mma>) -> tensor<256x128xf32, #blocked2>
      tt.store %arg2, %out {cache = 1 : i32, evict = 1 : i32} : tensor<256x128xf32, #blocked2>
    }
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: "triton_gpu.enable-warp-specialization" = 0 : i32
  // CHECK-LABEL: @mismatch_load_type
  tt.func public @mismatch_load_type(
      %arg0: !tt.ptr<tensor<256x64xf16, #blocked>, 1>,
      %arg1: tensor<64x128x!tt.ptr<f16, 1>, #blocked4>,
      %arg2: tensor<256x128x!tt.ptr<f32, 1>, #blocked2>
  ) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x128xf32, #mma>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    scf.for %iv = %c0 to %c10 step %c1 iter_args() -> () {
      %a = tt.load %arg0 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<256x64xf16, #blocked>, 1> -> tensor<256x64xf16, #blocked3>
      %b = tt.load %arg1 {boundaryCheck = array<i32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x128x!tt.ptr<f16, 1>, #blocked4> -> tensor<64x128xf16, #blocked4>
      %shm_a = triton_gpu.convert_layout %a : (tensor<256x64xf16, #blocked3>) -> tensor<256x64xf16, #shared>
      %shm_b = triton_gpu.convert_layout %b : (tensor<64x128xf16, #blocked4>) -> tensor<64x128xf16, #shared1>
      %d = tt.dot %shm_a, %shm_b, %cst {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<256x64xf16, #shared> * tensor<64x128xf16, #shared1> -> tensor<256x128xf32, #mma>
      %out = triton_gpu.convert_layout %d : (tensor<256x128xf32, #mma>) -> tensor<256x128xf32, #blocked2>
      tt.store %arg2, %out {cache = 1 : i32, evict = 1 : i32} : tensor<256x128xf32, #blocked2>
    }
    tt.return
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 64, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 4, maxPhase = 2, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {
  // CHECK: "triton_gpu.enable-warp-specialization" = 1 : i32
  // CHECK-LABEL: @epilogue_with_reduce
  tt.func public @epilogue_with_reduce(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg6: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg7: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg8: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg9: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}, %arg10: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %c15_i32 = arith.constant 15 : i32
    %c63_i32 = arith.constant 63 : i32
    %c132_i32 = arith.constant 132 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg6, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg5, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.muli %4, %2 : i32
    %6 = arith.muli %2, %c8_i32 : i32
    %7 = arith.divsi %0, %6 : i32
    %8 = arith.muli %7, %c8_i32 : i32
    %9 = arith.subi %4, %8 : i32
    %10 = arith.cmpi "slt", %9, %c8_i32 : i32
    %11 = arith.select %10, %9, %c8_i32 : i32
    %12 = arith.remsi %0, %6 : i32
    %13 = arith.remsi %12, %11 : i32
    %14 = arith.addi %8, %13 : i32
    %15 = arith.divsi %12, %11 : i32
    %16 = arith.muli %14, %c64_i32 : i32
    %17 = arith.muli %15, %c64_i32 : i32
    %18 = arith.extsi %arg5 : i32 to i64
    %19 = arith.extsi %arg7 : i32 to i64
    %20 = arith.extsi %arg8 : i32 to i64
    %21 = tt.make_tensor_ptr %arg0, [%18, %19], [%20, %c1_i64], [%16, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x16xf16, #blocked>, 1>
    %22 = arith.extsi %arg6 : i32 to i64
    %23 = arith.extsi %arg9 : i32 to i64
    %24 = tt.make_tensor_ptr %arg1, [%19, %22], [%23, %c1_i64], [%c0_i32, %17] {order = array<i32: 1, 0>} : <tensor<16x64xf16, #blocked1>, 1>
    %25 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %26 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %27 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %28 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %29 = tt.splat %arg10 : (i32) -> tensor<64x1xi32, #blocked2>
    %30 = tt.splat %arg4 : (!tt.ptr<f32, 1>) -> tensor<64x1x!tt.ptr<f32, 1>, #blocked2>
    %31 = tt.splat %arg5 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %32 = tt.splat %arg6 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %33 = arith.addi %arg7, %c15_i32 : i32
    %34 = arith.divsi %33, %c16_i32 : i32
    %35 = arith.subi %c0_i32, %34 : i32
    %36 = arith.muli %35, %c16_i32 : i32
    %37:4 = scf.for %arg11 = %0 to %5 step %c132_i32 iter_args(%arg12 = %21, %arg13 = %24, %arg14 = %14, %arg15 = %15) -> (!tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>, i32, i32)  : i32 {
      %38 = arith.divsi %arg11, %6 : i32
      %39 = arith.muli %38, %c8_i32 : i32
      %40 = arith.subi %4, %39 : i32
      %41 = arith.cmpi "slt", %40, %c8_i32 : i32
      %42 = arith.select %41, %40, %c8_i32 : i32
      %43 = arith.remsi %arg11, %6 : i32
      %44 = arith.remsi %43, %42 : i32
      %45 = arith.addi %39, %44 : i32
      %46 = arith.divsi %43, %42 : i32
      %47 = arith.muli %45, %c64_i32 : i32
      %48 = arith.muli %46, %c64_i32 : i32
      %49 = tt.splat %47 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %50 = tt.splat %47 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %51 = arith.addi %49, %26 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %52 = arith.addi %50, %28 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %53 = tt.splat %48 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %54 = tt.splat %48 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %55 = arith.addi %53, %25 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %56 = arith.addi %54, %27 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %57 = tt.expand_dims %51 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi32, #blocked2>
      %58 = arith.muli %57, %29 : tensor<64x1xi32, #blocked2>
      %59 = tt.addptr %30, %58 : tensor<64x1x!tt.ptr<f32, 1>, #blocked2>, tensor<64x1xi32, #blocked2>
      %60 = tt.expand_dims %55 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x64xi32, #blocked2>
      %61 = tt.broadcast %59 : (tensor<64x1x!tt.ptr<f32, 1>, #blocked2>) -> tensor<64x64x!tt.ptr<f32, 1>, #blocked2>
      %62 = tt.broadcast %60 : (tensor<1x64xi32, #blocked2>) -> tensor<64x64xi32, #blocked2>
      %63 = tt.addptr %61, %62 : tensor<64x64x!tt.ptr<f32, 1>, #blocked2>, tensor<64x64xi32, #blocked2>
      %64 = arith.cmpi "slt", %52, %31 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %65 = tt.expand_dims %64 {axis = 1 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xi1, #blocked2>
      %66 = arith.cmpi "slt", %56, %32 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
      %67 = tt.expand_dims %66 {axis = 0 : i32} : (tensor<64xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x64xi1, #blocked2>
      %68 = tt.broadcast %65 : (tensor<64x1xi1, #blocked2>) -> tensor<64x64xi1, #blocked2>
      %69 = tt.broadcast %67 : (tensor<1x64xi1, #blocked2>) -> tensor<64x64xi1, #blocked2>
      %70 = arith.andi %68, %69 : tensor<64x64xi1, #blocked2>
      %71 = arith.subi %45, %arg14 : i32
      %72 = arith.muli %71, %c64_i32 : i32
      %73 = tt.advance %arg12, [%72, %c0_i32] : <tensor<64x16xf16, #blocked>, 1>
      %74 = arith.subi %46, %arg15 : i32
      %75 = arith.muli %74, %c64_i32 : i32
      %76 = tt.advance %arg13, [%c0_i32, %75] : <tensor<16x64xf16, #blocked1>, 1>
      %77:3 = scf.for %arg16 = %c0_i32 to %arg7 step %c16_i32 iter_args(%arg17 = %cst, %arg18 = %73, %arg19 = %76) -> (tensor<64x64xf32, #mma>, !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>)  : i32 {
        %91 = tt.load %arg18 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<64x16xf16, #blocked>, 1> -> tensor<64x16xf16, #blocked3>
        %92 = tt.load %arg19 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<16x64xf16, #blocked1>, 1> -> tensor<16x64xf16, #blocked4>
        %93 = triton_gpu.convert_layout %91 : (tensor<64x16xf16, #blocked3>) -> tensor<64x16xf16, #shared>
        %94 = triton_gpu.convert_layout %92 : (tensor<16x64xf16, #blocked4>) -> tensor<16x64xf16, #shared1>
        %95 = tt.dot %93, %94, %arg17 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x16xf16, #shared> * tensor<16x64xf16, #shared1> -> tensor<64x64xf32, #mma>
        %96 = tt.advance %arg18, [%c0_i32, %c16_i32] : <tensor<64x16xf16, #blocked>, 1>
        %97 = tt.advance %arg19, [%c16_i32, %c0_i32] : <tensor<16x64xf16, #blocked1>, 1>
        scf.yield %95, %96, %97 : tensor<64x64xf32, #mma>, !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>
      }
      %78 = triton_gpu.convert_layout %77#0 : (tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #blocked2>
      %79 = triton_gpu.convert_layout %77#0 : (tensor<64x64xf32, #mma>) -> tensor<64x64xf32, #blocked2>
      %80 = tt.advance %77#1, [%c0_i32, %36] : <tensor<64x16xf16, #blocked>, 1>
      %81 = tt.advance %77#2, [%36, %c0_i32] : <tensor<16x64xf16, #blocked1>, 1>
      %82 = "tt.reduce"(%78) ({
      ^bb0(%arg16: f32, %arg17: f32):
        %91 = arith.cmpf "ogt", %arg16, %arg17 : f32
        %92 = arith.select %91, %arg16, %arg17 : f32
        tt.reduce.return %92 : f32
      }) {axis = 1 : i32} : (tensor<64x64xf32, #blocked2>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %83 = tt.expand_dims %82 {axis = 1 : i32} : (tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xf32, #blocked2>
      %84 = tt.broadcast %83 : (tensor<64x1xf32, #blocked2>) -> tensor<64x64xf32, #blocked2>
      %85 = arith.subf %79, %84 : tensor<64x64xf32, #blocked2>
      %86 = math.exp %85 : tensor<64x64xf32, #blocked2>
      %87 = "tt.reduce"(%86) ({
      ^bb0(%arg16: f32, %arg17: f32):
        %91 = arith.addf %arg16, %arg17 : f32
        tt.reduce.return %91 : f32
      }) {axis = 1 : i32} : (tensor<64x64xf32, #blocked2>) -> tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
      %88 = tt.expand_dims %87 {axis = 1 : i32} : (tensor<64xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<64x1xf32, #blocked2>
      %89 = tt.broadcast %88 : (tensor<64x1xf32, #blocked2>) -> tensor<64x64xf32, #blocked2>
      %90 = arith.divf %86, %89 : tensor<64x64xf32, #blocked2>
      tt.store %63, %90, %70 {cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf32, #blocked2>
      scf.yield %80, %81, %45, %46 : !tt.ptr<tensor<64x16xf16, #blocked>, 1>, !tt.ptr<tensor<16x64xf16, #blocked1>, 1>, i32, i32
    }
    tt.return
  }
}
