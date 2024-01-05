// RUN: triton-opt -split-input-file -triton-nvidia-gpu-ws-decomposing='compute-capability=90' %s 2>&1 | FileCheck %s
// XFAIL: *
// TODO: change this test to not check for a fixed IR.

// Check if all operations are labeled with appropriate attributes.
#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.enable-warp-specialization" = 1 : i32} {
  // CHECK-LABEL: @simple_gemm
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

    // CHECK-NEXT:     %cst = arith.constant {async_agent = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // CHECK-NEXT:     %cst_0 = arith.constant {async_agent = dense<0> : vector<1xi32>} dense<32> : tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %cst_1 = arith.constant {async_agent = dense<0> : vector<1xi32>} dense<32> : tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %c31_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 31 : i32
    // CHECK-NEXT:     %c127_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 127 : i32
    // CHECK-NEXT:     %c1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 1 : index
    // CHECK-NEXT:     %c0 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : index
    // CHECK-NEXT:     %c32_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 32 : i32
    // CHECK-NEXT:     %c128_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 128 : i32
    // CHECK-NEXT:     %c8_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 8 : i32
    // CHECK-NEXT:     %0 = tt.get_program_id {async_agent = dense<[0, 1]> : vector<2xi32>, axis = 0 : i32} : i32
    // CHECK-NEXT:     %1 = tt.get_program_id {async_agent = dense<0> : vector<1xi32>, axis = 1 : i32} : i32
    // CHECK-NEXT:     %2 = arith.addi %arg3, %c127_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %3 = arith.divsi %2, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %4 = arith.addi %arg4, %c127_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %5 = arith.divsi %4, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %6 = arith.muli %5, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %7 = arith.divsi %0, %6 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %8 = arith.muli %7, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %9 = arith.subi %3, %8 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %10 = arith.cmpi slt, %9, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %11 = arith.select %10, %9, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %12 = arith.remsi %0, %11 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %13 = arith.addi %8, %12 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %14 = arith.remsi %0, %6 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %15 = arith.divsi %14, %11 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %16 = arith.muli %13, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %17 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %18 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %19 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %20 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %21 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %22 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %23 = tt.splat %16 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %24 = tt.splat %16 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %25 = tt.splat %16 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %26 = arith.addi %23, %17 {async_agent = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %27 = arith.addi %24, %19 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %28 = arith.addi %25, %21 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %29 = arith.muli %15, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %30 = tt.splat %29 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %31 = tt.splat %29 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %32 = tt.splat %29 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %33 = arith.addi %30, %18 {async_agent = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %34 = arith.addi %31, %20 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %35 = arith.addi %32, %22 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %36 = tt.splat %arg3 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %37 = tt.splat %arg3 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %38 = arith.remsi %26, %36 {async_agent = dense<0> : vector<1xi32>, tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %39 = tt.splat %arg4 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %40 = tt.splat %arg4 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %41 = arith.remsi %33, %39 {async_agent = dense<0> : vector<1xi32>, tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %42 = arith.muli %1, %c32_i32 {async_agent = dense<0> : vector<1xi32>} : i32
    // CHECK-NEXT:     %43 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    // CHECK-NEXT:     %44 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT:     %45 = tt.splat %42 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    // CHECK-NEXT:     %46 = tt.splat %42 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT:     %47 = arith.addi %45, %43 {async_agent = dense<0> : vector<1xi32>} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    // CHECK-NEXT:     %48 = arith.addi %46, %44 {async_agent = dense<0> : vector<1xi32>} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT:     %49 = tt.expand_dims %38 {async_agent = dense<0> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    // CHECK-NEXT:     %50 = tt.splat %arg6 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128x1xi32, #blocked1>
    // CHECK-NEXT:     %51 = arith.muli %49, %50 {async_agent = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked1>
    // CHECK-NEXT:     %52 = tt.expand_dims %47 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x32xi32, #blocked1>
    // CHECK-NEXT:     %53 = tt.broadcast %51 {async_agent = dense<0> : vector<1xi32>} : (tensor<128x1xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %54 = tt.broadcast %52 {async_agent = dense<0> : vector<1xi32>} : (tensor<1x32xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %55 = arith.addi %53, %54 {async_agent = dense<0> : vector<1xi32>} : tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %56 = tt.splat %arg0 {async_agent = dense<0> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
    // CHECK-NEXT:     %57 = tt.addptr %56, %55 {async_agent = dense<0> : vector<1xi32>} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %58 = tt.expand_dims %48 {async_agent = dense<0> : vector<1xi32>, axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    // CHECK-NEXT:     %59 = tt.expand_dims %41 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi32, #blocked>
    // CHECK-NEXT:     %60 = tt.splat %arg7 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<1x128xi32, #blocked>
    // CHECK-NEXT:     %61 = arith.muli %59, %60 {async_agent = dense<0> : vector<1xi32>} : tensor<1x128xi32, #blocked>
    // CHECK-NEXT:     %62 = tt.broadcast %58 {async_agent = dense<0> : vector<1xi32>} : (tensor<32x1xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %63 = tt.broadcast %61 {async_agent = dense<0> : vector<1xi32>} : (tensor<1x128xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %64 = arith.addi %62, %63 {async_agent = dense<0> : vector<1xi32>} : tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %65 = tt.splat %arg1 {async_agent = dense<0> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    // CHECK-NEXT:     %66 = tt.addptr %65, %64 {async_agent = dense<0> : vector<1xi32>} : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %67 = arith.addi %arg5, %c31_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %68 = arith.divsi %67, %c32_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %69 = arith.index_cast %68 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32 to index
    // CHECK-NEXT:     %70:3 = scf.for %arg9 = %c0 to %69 step %c1 iter_args(%arg10 = %cst, %arg11 = %57, %arg12 = %66) -> (tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>) {
    // CHECK-NEXT:       %89 = tt.load %arg11 {async_agent = dense<0> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
    // CHECK-NEXT:       %90 = tt.load %arg12 {async_agent = dense<0> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
    // CHECK-NEXT:       %91 = triton_gpu.convert_layout %89 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x32xf16, #blocked1>) -> tensor<128x32xf16, #shared>
    // CHECK-NEXT:       %92 = triton_gpu.convert_layout %90 {async_agent = dense<1> : vector<1xi32>} : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
    // CHECK-NEXT:       %93 = tt.dot %91, %92, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<128x32xf16, #shared> * tensor<32x128xf16, #shared1> -> tensor<128x128xf32, #mma>
    // CHECK-NEXT:       %94 = tt.addptr %arg11, %cst_1 {async_agent = dense<0> : vector<1xi32>} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:       %95 = tt.addptr %arg12, %cst_0 {async_agent = dense<0> : vector<1xi32>} : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
    // CHECK-NEXT:       scf.yield {async_agent = dense<[0, 1]> : vector<2xi32>} %93, %94, %95 : tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    // CHECK-NEXT:     } {async_agent = dense<[0, 1]> : vector<2xi32>}
    // CHECK-NEXT:     %71 = arith.truncf %70#0 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    // CHECK-NEXT:     %72 = tt.expand_dims %27 {async_agent = dense<1> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi32, #blocked2>
    // CHECK-NEXT:     %73 = tt.splat %arg8 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128x1xi32, #blocked2>
    // CHECK-NEXT:     %74 = arith.muli %72, %73 {async_agent = dense<1> : vector<1xi32>} : tensor<128x1xi32, #blocked2>
    // CHECK-NEXT:     %75 = tt.expand_dims %34 {async_agent = dense<1> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi32, #blocked2>
    // CHECK-NEXT:     %76 = tt.broadcast %74 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x1xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %77 = tt.broadcast %75 {async_agent = dense<1> : vector<1xi32>} : (tensor<1x128xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %78 = arith.addi %76, %77 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %79 = tt.splat %arg2 {async_agent = dense<1> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
    // CHECK-NEXT:     %80 = tt.addptr %79, %78 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %81 = arith.cmpi "slt", %28, %37 {async_agent = dense<1> : vector<1xi32>, predicate = 2 : i64} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %82 = tt.expand_dims %81 {async_agent = dense<1> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi1, #blocked2>
    // CHECK-NEXT:     %83 = arith.cmpi "slt", %35, %40 {async_agent = dense<1> : vector<1xi32>, predicate = 2 : i64} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %84 = tt.expand_dims %83 {async_agent = dense<1> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi1, #blocked2>
    // CHECK-NEXT:     %85 = tt.broadcast %82 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x1xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    // CHECK-NEXT:     %86 = tt.broadcast %84 {async_agent = dense<1> : vector<1xi32>} : (tensor<1x128xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    // CHECK-NEXT:     %87 = arith.andi %85, %86 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xi1, #blocked2>
    // CHECK-NEXT:     %88 = triton_gpu.convert_layout %71 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x128xf16, #mma>) -> tensor<128x128xf16, #blocked2>
    // CHECK-NEXT:     tt.store %80, %88, %87 {async_agent = dense<1> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked2>
  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.enable-warp-specialization" = 1 : i32} {
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

    // CHECK-NEXT:     %cst = arith.constant {async_agent = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // CHECK-NEXT:     %cst_0 = arith.constant {async_agent = dense<0> : vector<1xi32>} dense<32> : tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %cst_1 = arith.constant {async_agent = dense<0> : vector<1xi32>} dense<32> : tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %c31_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 31 : i32
    // CHECK-NEXT:     %c127_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 127 : i32
    // CHECK-NEXT:     %c1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 1 : index
    // CHECK-NEXT:     %c0 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : index
    // CHECK-NEXT:     %c32_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 32 : i32
    // CHECK-NEXT:     %c128_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 128 : i32
    // CHECK-NEXT:     %c8_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 8 : i32
    // CHECK-NEXT:     %0 = tt.get_program_id {async_agent = dense<[0, 1]> : vector<2xi32>, axis = 0 : i32} : i32
    // CHECK-NEXT:     %1 = tt.get_program_id {async_agent = dense<0> : vector<1xi32>, axis = 1 : i32} : i32
    // CHECK-NEXT:     %2 = arith.addi %arg3, %c127_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %3 = arith.divsi %2, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %4 = arith.addi %arg4, %c127_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %5 = arith.divsi %4, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %6 = arith.muli %5, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %7 = arith.divsi %0, %6 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %8 = arith.muli %7, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %9 = arith.subi %3, %8 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %10 = arith.cmpi slt, %9, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %11 = arith.select %10, %9, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %12 = arith.remsi %0, %11 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %13 = arith.addi %8, %12 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %14 = arith.remsi %0, %6 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %15 = arith.divsi %14, %11 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %16 = arith.muli %13, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %17 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %18 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %19 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %20 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %21 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %22 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %23 = tt.splat %16 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %24 = tt.splat %16 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %25 = tt.splat %16 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %26 = arith.addi %23, %17 {async_agent = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %27 = arith.addi %24, %19 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %28 = arith.addi %25, %21 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %29 = arith.muli %15, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %30 = tt.splat %29 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %31 = tt.splat %29 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %32 = tt.splat %29 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %33 = arith.addi %30, %18 {async_agent = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %34 = arith.addi %31, %20 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %35 = arith.addi %32, %22 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %36 = tt.splat %arg3 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %37 = tt.splat %arg3 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %38 = arith.remsi %26, %36 {async_agent = dense<0> : vector<1xi32>, tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %39 = tt.splat %arg4 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %40 = tt.splat %arg4 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %41 = arith.remsi %33, %39 {async_agent = dense<0> : vector<1xi32>, tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %42 = arith.muli %1, %c32_i32 {async_agent = dense<0> : vector<1xi32>} : i32
    // CHECK-NEXT:     %43 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    // CHECK-NEXT:     %44 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT:     %45 = tt.splat %42 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    // CHECK-NEXT:     %46 = tt.splat %42 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT:     %47 = arith.addi %45, %43 {async_agent = dense<0> : vector<1xi32>} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    // CHECK-NEXT:     %48 = arith.addi %46, %44 {async_agent = dense<0> : vector<1xi32>} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT:     %49 = tt.expand_dims %38 {async_agent = dense<0> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    // CHECK-NEXT:     %50 = tt.splat %arg6 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128x1xi32, #blocked1>
    // CHECK-NEXT:     %51 = arith.muli %49, %50 {async_agent = dense<0> : vector<1xi32>} : tensor<128x1xi32, #blocked1>
    // CHECK-NEXT:     %52 = tt.expand_dims %47 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x32xi32, #blocked1>
    // CHECK-NEXT:     %53 = tt.broadcast %51 {async_agent = dense<0> : vector<1xi32>} : (tensor<128x1xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %54 = tt.broadcast %52 {async_agent = dense<0> : vector<1xi32>} : (tensor<1x32xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %55 = arith.addi %53, %54 {async_agent = dense<0> : vector<1xi32>} : tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %56 = tt.splat %arg0 {async_agent = dense<0> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
    // CHECK-NEXT:     %57 = tt.addptr %56, %55 {async_agent = dense<0> : vector<1xi32>} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %58 = tt.expand_dims %48 {async_agent = dense<0> : vector<1xi32>, axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    // CHECK-NEXT:     %59 = tt.expand_dims %41 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi32, #blocked>
    // CHECK-NEXT:     %60 = tt.splat %arg7 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<1x128xi32, #blocked>
    // CHECK-NEXT:     %61 = arith.muli %59, %60 {async_agent = dense<0> : vector<1xi32>} : tensor<1x128xi32, #blocked>
    // CHECK-NEXT:     %62 = tt.broadcast %58 {async_agent = dense<0> : vector<1xi32>} : (tensor<32x1xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %63 = tt.broadcast %61 {async_agent = dense<0> : vector<1xi32>} : (tensor<1x128xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %64 = arith.addi %62, %63 {async_agent = dense<0> : vector<1xi32>} : tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %65 = tt.splat %arg1 {async_agent = dense<0> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    // CHECK-NEXT:     %66 = tt.addptr %65, %64 {async_agent = dense<0> : vector<1xi32>} : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %67 = arith.addi %arg5, %c31_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %68 = arith.divsi %67, %c32_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %69 = arith.index_cast %68 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32 to index
    // CHECK-NEXT:     %cst_2 = arith.constant {async_agent = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<128x32xf16, #shared>
    // CHECK-NEXT:     %cst_3 = arith.constant {async_agent = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<32x128xf16, #shared1>
    // CHECK-NEXT:     %70:3 = scf.for %arg9 = %c0 to %69 step %c1 iter_args(%arg10 = %cst, %arg11 = %57, %arg12 = %66) -> (tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>) {
    // CHECK-NEXT:       %89:2 = scf.for %arg13 = %c0 to %69 step %c1 iter_args(%arg14 = %cst_2, %arg15 = %cst_3) -> (tensor<128x32xf16, #shared>, tensor<32x128xf16, #shared1>) {
    // CHECK-NEXT:         %93 = tt.load %arg11 {async_agent = dense<0> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
    // CHECK-NEXT:         %94 = tt.load %arg12 {async_agent = dense<0> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
    // CHECK-NEXT:         %95 = triton_gpu.convert_layout %93 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x32xf16, #blocked1>) -> tensor<128x32xf16, #shared>
    // CHECK-NEXT:         %96 = triton_gpu.convert_layout %94 {async_agent = dense<1> : vector<1xi32>} : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
    // CHECK-NEXT:         scf.yield {async_agent = dense<[0, 1]> : vector<2xi32>} %95, %96 : tensor<128x32xf16, #shared>, tensor<32x128xf16, #shared1>
    // CHECK-NEXT:       } {async_agent = dense<[0, 1]> : vector<2xi32>}
    // CHECK-NEXT:       %90 = tt.dot %89#0, %89#1, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<128x32xf16, #shared> * tensor<32x128xf16, #shared1> -> tensor<128x128xf32, #mma>
    // CHECK-NEXT:       %91 = tt.addptr %arg11, %cst_1 {async_agent = dense<0> : vector<1xi32>} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:       %92 = tt.addptr %arg12, %cst_0 {async_agent = dense<0> : vector<1xi32>} : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
    // CHECK-NEXT:       scf.yield {async_agent = dense<[0, 1]> : vector<2xi32>} %90, %91, %92 : tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    // CHECK-NEXT:     } {async_agent = dense<[0, 1]> : vector<2xi32>}
    // CHECK-NEXT:     %71 = arith.truncf %70#0 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    // CHECK-NEXT:     %72 = tt.expand_dims %27 {async_agent = dense<1> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi32, #blocked2>
    // CHECK-NEXT:     %73 = tt.splat %arg8 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128x1xi32, #blocked2>
    // CHECK-NEXT:     %74 = arith.muli %72, %73 {async_agent = dense<1> : vector<1xi32>} : tensor<128x1xi32, #blocked2>
    // CHECK-NEXT:     %75 = tt.expand_dims %34 {async_agent = dense<1> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi32, #blocked2>
    // CHECK-NEXT:     %76 = tt.broadcast %74 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x1xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %77 = tt.broadcast %75 {async_agent = dense<1> : vector<1xi32>} : (tensor<1x128xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %78 = arith.addi %76, %77 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %79 = tt.splat %arg2 {async_agent = dense<1> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
    // CHECK-NEXT:     %80 = tt.addptr %79, %78 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %81 = arith.cmpi "slt", %28, %37 {async_agent = dense<1> : vector<1xi32>, predicate = 2 : i64} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %82 = tt.expand_dims %81 {async_agent = dense<1> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi1, #blocked2>
    // CHECK-NEXT:     %83 = arith.cmpi "slt", %35, %40 {async_agent = dense<1> : vector<1xi32>, predicate = 2 : i64} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %84 = tt.expand_dims %83 {async_agent = dense<1> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi1, #blocked2>
    // CHECK-NEXT:     %85 = tt.broadcast %82 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x1xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    // CHECK-NEXT:     %86 = tt.broadcast %84 {async_agent = dense<1> : vector<1xi32>} : (tensor<1x128xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    // CHECK-NEXT:     %87 = arith.andi %85, %86 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xi1, #blocked2>
    // CHECK-NEXT:     %88 = triton_gpu.convert_layout %71 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x128xf16, #mma>) -> tensor<128x128xf16, #blocked2>
    // CHECK-NEXT:     tt.store %80, %88, %87 {async_agent = dense<1> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked2>

  }
}

// -----

#blocked = #triton_gpu.blocked<{sizePerThread = [8, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 4], order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#mma = #triton_gpu.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 128, 16]}>
#shared = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
#shared1 = #triton_gpu.shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = true}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.enable-warp-specialization" = 1 : i32} {
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
    // CHECK-NEXT:     %cst = arith.constant {async_agent = dense<1> : vector<1xi32>} dense<0.000000e+00> : tensor<128x128xf32, #mma>
    // CHECK-NEXT:     %cst_0 = arith.constant {async_agent = dense<0> : vector<1xi32>} dense<32> : tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %cst_1 = arith.constant {async_agent = dense<1> : vector<1xi32>} dense<32> : tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %c31_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 31 : i32
    // CHECK-NEXT:     %c127_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 127 : i32
    // CHECK-NEXT:     %c1 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 1 : index
    // CHECK-NEXT:     %c0 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 0 : index
    // CHECK-NEXT:     %c32_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 32 : i32
    // CHECK-NEXT:     %c128_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 128 : i32
    // CHECK-NEXT:     %c8_i32 = arith.constant {async_agent = dense<[0, 1]> : vector<2xi32>} 8 : i32
    // CHECK-NEXT:     %0 = tt.get_program_id {async_agent = dense<[0, 1]> : vector<2xi32>, axis = 0 : i32} : i32
    // CHECK-NEXT:     %1 = tt.get_program_id {async_agent = dense<[0, 1]> : vector<2xi32>, axis = 1 : i32} : i32
    // CHECK-NEXT:     %2 = arith.addi %arg3, %c127_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %3 = arith.divsi %2, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %4 = arith.addi %arg4, %c127_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %5 = arith.divsi %4, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %6 = arith.muli %5, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %7 = arith.divsi %0, %6 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %8 = arith.muli %7, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %9 = arith.subi %3, %8 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %10 = arith.cmpi slt, %9, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %11 = arith.select %10, %9, %c8_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %12 = arith.remsi %0, %11 {async_agent = dense<1> : vector<1xi32>} : i32
    // CHECK-NEXT:     %13 = arith.addi %8, %12 {async_agent = dense<1> : vector<1xi32>} : i32
    // CHECK-NEXT:     %14 = arith.remsi %0, %6 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %15 = arith.divsi %14, %11 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %16 = arith.muli %13, %c128_i32 {async_agent = dense<1> : vector<1xi32>} : i32
    // CHECK-NEXT:     %17 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %18 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %19 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %20 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %21 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %22 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %23 = tt.splat %16 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %24 = tt.splat %16 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %25 = tt.splat %16 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %26 = arith.addi %23, %17 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %27 = arith.addi %24, %19 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %28 = arith.addi %25, %21 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %29 = arith.muli %15, %c128_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %30 = tt.splat %29 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %31 = tt.splat %29 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %32 = tt.splat %29 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %33 = arith.addi %30, %18 {async_agent = dense<0> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %34 = arith.addi %31, %20 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %35 = arith.addi %32, %22 {async_agent = dense<1> : vector<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %36 = tt.splat %arg3 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %37 = tt.splat %arg3 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %38 = arith.remsi %26, %36 {async_agent = dense<1> : vector<1xi32>, tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    // CHECK-NEXT:     %39 = tt.splat %arg4 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %40 = tt.splat %arg4 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %41 = arith.remsi %33, %39 {async_agent = dense<0> : vector<1xi32>, tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    // CHECK-NEXT:     %42 = arith.muli %1, %c32_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %43 = tt.make_range {async_agent = dense<1> : vector<1xi32>, end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    // CHECK-NEXT:     %44 = tt.make_range {async_agent = dense<0> : vector<1xi32>, end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT:     %45 = tt.splat %42 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    // CHECK-NEXT:     %46 = tt.splat %42 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT:     %47 = arith.addi %45, %43 {async_agent = dense<1> : vector<1xi32>} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    // CHECK-NEXT:     %48 = arith.addi %46, %44 {async_agent = dense<0> : vector<1xi32>} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT:     %49 = tt.expand_dims %38 {async_agent = dense<1> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<128x1xi32, #blocked1>
    // CHECK-NEXT:     %50 = tt.splat %arg6 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128x1xi32, #blocked1>
    // CHECK-NEXT:     %51 = arith.muli %49, %50 {async_agent = dense<1> : vector<1xi32>} : tensor<128x1xi32, #blocked1>
    // CHECK-NEXT:     %52 = tt.expand_dims %47 {async_agent = dense<1> : vector<1xi32>, axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x32xi32, #blocked1>
    // CHECK-NEXT:     %53 = tt.broadcast %51 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x1xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %54 = tt.broadcast %52 {async_agent = dense<1> : vector<1xi32>} : (tensor<1x32xi32, #blocked1>) -> tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %55 = arith.addi %53, %54 {async_agent = dense<1> : vector<1xi32>} : tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %56 = tt.splat %arg0 {async_agent = dense<1> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
    // CHECK-NEXT:     %57 = tt.addptr %56, %55 {async_agent = dense<1> : vector<1xi32>} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:     %58 = tt.expand_dims %48 {async_agent = dense<0> : vector<1xi32>, axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    // CHECK-NEXT:     %59 = tt.expand_dims %41 {async_agent = dense<0> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi32, #blocked>
    // CHECK-NEXT:     %60 = tt.splat %arg7 {async_agent = dense<0> : vector<1xi32>} : (i32) -> tensor<1x128xi32, #blocked>
    // CHECK-NEXT:     %61 = arith.muli %59, %60 {async_agent = dense<0> : vector<1xi32>} : tensor<1x128xi32, #blocked>
    // CHECK-NEXT:     %62 = tt.broadcast %58 {async_agent = dense<0> : vector<1xi32>} : (tensor<32x1xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %63 = tt.broadcast %61 {async_agent = dense<0> : vector<1xi32>} : (tensor<1x128xi32, #blocked>) -> tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %64 = arith.addi %62, %63 {async_agent = dense<0> : vector<1xi32>} : tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %65 = tt.splat %arg1 {async_agent = dense<0> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    // CHECK-NEXT:     %66 = tt.addptr %65, %64 {async_agent = dense<0> : vector<1xi32>} : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
    // CHECK-NEXT:     %67 = arith.addi %arg5, %c31_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %68 = arith.divsi %67, %c32_i32 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32
    // CHECK-NEXT:     %69 = arith.index_cast %68 {async_agent = dense<[0, 1]> : vector<2xi32>} : i32 to index
    // CHECK-NEXT:     %70:3 = scf.for %arg9 = %c0 to %69 step %c1 iter_args(%arg10 = %cst, %arg11 = %57, %arg12 = %66) -> (tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>) {
    // CHECK-NEXT:       %89 = arith.index_cast %arg9 {async_agent = dense<1> : vector<1xi32>} : index to i32
    // CHECK-NEXT:       %90 = arith.cmpi ne, %89, %c31_i32 {async_agent = dense<1> : vector<1xi32>} : i32
    // CHECK-NEXT:       %91 = scf.if %90 -> (tensor<128x32xf16, #blocked1>) {
    // CHECK-NEXT:         %99 = arith.select %90, %c31_i32, %c127_i32 {async_agent = dense<1> : vector<1xi32>} : i32
    // CHECK-NEXT:         %100 = tt.splat %99 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:         %101 = tt.addptr %arg11, %100 {async_agent = dense<1> : vector<1xi32>} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:         %102 = tt.load %101 {async_agent = dense<1> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
    // CHECK-NEXT:         scf.yield {async_agent = dense<1> : vector<1xi32>} %102 : tensor<128x32xf16, #blocked1>
    // CHECK-NEXT:       } else {
    // CHECK-NEXT:         %99 = tt.load %arg11 {async_agent = dense<1> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128x32xf16, #blocked1>
    // CHECK-NEXT:         scf.yield {async_agent = dense<1> : vector<1xi32>} %99 : tensor<128x32xf16, #blocked1>
    // CHECK-NEXT:       } {async_agent = dense<1> : vector<1xi32>}
    // CHECK-NEXT:       %92 = tt.load %arg12 {async_agent = dense<0> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf16, #blocked>
    // CHECK-NEXT:       %93 = triton_gpu.convert_layout %91 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x32xf16, #blocked1>) -> tensor<128x32xf16, #shared>
    // CHECK-NEXT:       %94 = triton_gpu.convert_layout %92 {async_agent = dense<1> : vector<1xi32>} : (tensor<32x128xf16, #blocked>) -> tensor<32x128xf16, #shared1>
    // CHECK-NEXT:       %95 = tt.dot %93, %94, %arg10 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, async_agent = dense<1> : vector<1xi32>} : tensor<128x32xf16, #shared> * tensor<32x128xf16, #shared1> -> tensor<128x128xf32, #mma>
    // CHECK-NEXT:       %96 = scf.if %90 -> (tensor<128x32x!tt.ptr<f16, 1>, #blocked1>) {
    // CHECK-NEXT:         %99 = arith.select %90, %c31_i32, %c127_i32 {async_agent = dense<1> : vector<1xi32>} : i32
    // CHECK-NEXT:         %100 = tt.splat %99 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:         %101 = tt.addptr %arg11, %100 {async_agent = dense<1> : vector<1xi32>} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:         scf.yield {async_agent = dense<1> : vector<1xi32>} %101 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
    // CHECK-NEXT:       } else {
    // CHECK-NEXT:         %99 = tt.addptr %arg11, %cst_1 {async_agent = dense<1> : vector<1xi32>} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:         scf.yield {async_agent = dense<1> : vector<1xi32>} %99 : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>
    // CHECK-NEXT:       } {async_agent = dense<1> : vector<1xi32>}
    // CHECK-NEXT:       %97 = tt.addptr %96, %cst_1 {async_agent = dense<1> : vector<1xi32>} : tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<128x32xi32, #blocked1>
    // CHECK-NEXT:       %98 = tt.addptr %arg12, %cst_0 {async_agent = dense<0> : vector<1xi32>} : tensor<32x128x!tt.ptr<f16, 1>, #blocked>, tensor<32x128xi32, #blocked>
    // CHECK-NEXT:       scf.yield {async_agent = dense<[0, 1]> : vector<2xi32>} %95, %97, %98 : tensor<128x128xf32, #mma>, tensor<128x32x!tt.ptr<f16, 1>, #blocked1>, tensor<32x128x!tt.ptr<f16, 1>, #blocked>
    // CHECK-NEXT:     } {async_agent = dense<[0, 1]> : vector<2xi32>}
    // CHECK-NEXT:     %71 = arith.truncf %70#0 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xf32, #mma> to tensor<128x128xf16, #mma>
    // CHECK-NEXT:     %72 = tt.expand_dims %27 {async_agent = dense<1> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi32, #blocked2>
    // CHECK-NEXT:     %73 = tt.splat %arg8 {async_agent = dense<1> : vector<1xi32>} : (i32) -> tensor<128x1xi32, #blocked2>
    // CHECK-NEXT:     %74 = arith.muli %72, %73 {async_agent = dense<1> : vector<1xi32>} : tensor<128x1xi32, #blocked2>
    // CHECK-NEXT:     %75 = tt.expand_dims %34 {async_agent = dense<1> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi32, #blocked2>
    // CHECK-NEXT:     %76 = tt.broadcast %74 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x1xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %77 = tt.broadcast %75 {async_agent = dense<1> : vector<1xi32>} : (tensor<1x128xi32, #blocked2>) -> tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %78 = arith.addi %76, %77 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %79 = tt.splat %arg2 {async_agent = dense<1> : vector<1xi32>} : (!tt.ptr<f16, 1>) -> tensor<128x128x!tt.ptr<f16, 1>, #blocked2>
    // CHECK-NEXT:     %80 = tt.addptr %79, %78 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128x!tt.ptr<f16, 1>, #blocked2>, tensor<128x128xi32, #blocked2>
    // CHECK-NEXT:     %81 = arith.cmpi "slt", %28, %37 {async_agent = dense<1> : vector<1xi32>, predicate = 2 : i64} : tensor<128xi32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    // CHECK-NEXT:     %82 = tt.expand_dims %81 {async_agent = dense<1> : vector<1xi32>, axis = 1 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<128x1xi1, #blocked2>
    // CHECK-NEXT:     %83 = arith.cmpi "slt", %35, %40 {async_agent = dense<1> : vector<1xi32>, predicate = 2 : i64} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    // CHECK-NEXT:     %84 = tt.expand_dims %83 {async_agent = dense<1> : vector<1xi32>, axis = 0 : i32} : (tensor<128xi1, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x128xi1, #blocked2>
    // CHECK-NEXT:     %85 = tt.broadcast %82 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x1xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    // CHECK-NEXT:     %86 = tt.broadcast %84 {async_agent = dense<1> : vector<1xi32>} : (tensor<1x128xi1, #blocked2>) -> tensor<128x128xi1, #blocked2>
    // CHECK-NEXT:     %87 = arith.andi %85, %86 {async_agent = dense<1> : vector<1xi32>} : tensor<128x128xi1, #blocked2>
    // CHECK-NEXT:     %88 = triton_gpu.convert_layout %71 {async_agent = dense<1> : vector<1xi32>} : (tensor<128x128xf16, #mma>) -> tensor<128x128xf16, #blocked2>
    // CHECK-NEXT:     tt.store %80, %88, %87 {async_agent = dense<1> : vector<1xi32>, cache = 1 : i32, evict = 1 : i32} : tensor<128x128xf16, #blocked2>
  }
}
