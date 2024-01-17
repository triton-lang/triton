// RUN: triton-opt --convert-triton-gpu-to-llvm=target=rocdl %s | FileCheck %s

// CHECK: module attributes {{.*}}, triton_gpu.shared = 9216 : i32
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared = #triton_gpu.shared<{vec = 4, perPhase = 2, maxPhase = 8, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#mfma = #triton_gpu.mfma<{versionMajor = 2, warpsPerCTA = [2, 2], instrShape = [32,32], isTransposed=false}>
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 64 : i32} {
  tt.func public @matmul_kernel_0d1d2d3d4d5d6d7c8d9c10d11c(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mfma>
    %cst_0 = arith.constant dense<32> : tensor<64x32xi32, #blocked>
    %c31_i32 = arith.constant 31 : i32
    %c63_i32 = arith.constant 63 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.addi %arg4, %c63_i32 : i32
    %4 = arith.divsi %3, %c64_i32 : i32
    %5 = arith.muli %4, %c4_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c4_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.cmpi "slt", %8, %c4_i32: i32
    %10 = arith.select %9, %8, %c4_i32 : i32
    %11 = arith.remsi %0, %10 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.remsi %0, %5 : i32
    %14 = arith.divsi %13, %10 : i32
    %15 = arith.muli %12, %c64_i32 : i32
    %16 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %17 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %18 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %19 = tt.splat %15 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %20 = tt.splat %15 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %21 = arith.addi %19, %16 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %22 = arith.addi %20, %18 : tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %23 = arith.muli %14, %c64_i32 : i32
    %24 = tt.splat %23 : (i32) -> tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %25 = arith.addi %24, %17 : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %26 = tt.expand_dims %21 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<64x1xi32, #blocked>
    %27 = tt.expand_dims %22 {axis = 1 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<64x1xi32, #blocked1>
    %28 = tt.splat %arg6 : (i32) -> tensor<64x1xi32, #blocked>
    %29 = arith.muli %26, %28 : tensor<64x1xi32, #blocked>
    %30 = tt.splat %arg0 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>, #blocked>
    %31 = tt.addptr %30, %29 : tensor<64x1x!tt.ptr<f16>, #blocked>, tensor<64x1xi32, #blocked>
    %32 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %33 = tt.expand_dims %32 {axis = 0 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x32xi32, #blocked>
    %34 = tt.broadcast %31 : (tensor<64x1x!tt.ptr<f16>, #blocked>) -> tensor<64x32x!tt.ptr<f16>, #blocked>
    %35 = tt.broadcast %33 : (tensor<1x32xi32, #blocked>) -> tensor<64x32xi32, #blocked>
    %36 = tt.addptr %34, %35 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
    %37 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %38 = tt.expand_dims %37 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<32x1xi32, #blocked1>
    %39 = tt.splat %arg7 : (i32) -> tensor<32x1xi32, #blocked1>
    %40 = arith.muli %38, %39 : tensor<32x1xi32, #blocked1>
    %41 = tt.splat %arg1 : (!tt.ptr<f16>) -> tensor<32x1x!tt.ptr<f16>, #blocked1>
    %42 = tt.addptr %41, %40 : tensor<32x1x!tt.ptr<f16>, #blocked1>, tensor<32x1xi32, #blocked1>
    %43 = tt.expand_dims %25 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x64xi32, #blocked1>
    %44 = tt.broadcast %42 : (tensor<32x1x!tt.ptr<f16>, #blocked1>) -> tensor<32x64x!tt.ptr<f16>, #blocked1>
    %45 = tt.broadcast %43 : (tensor<1x64xi32, #blocked1>) -> tensor<32x64xi32, #blocked1>
    %46 = tt.addptr %44, %45 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
    %47 = arith.addi %arg5, %c31_i32 : i32
    %48 = arith.divsi %47, %c32_i32 : i32
    %49 = arith.muli %arg7, %c32_i32 : i32
    %50 = tt.splat %49 : (i32) -> tensor<32x64xi32, #blocked1>
    %51 = tt.load %36 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x32xf16, #blocked>
    %52 = triton_gpu.convert_layout %51 : (tensor<64x32xf16, #blocked>) -> tensor<64x32xf16, #shared>
    %53 = tt.load %46 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked1>
    %54 = triton_gpu.convert_layout %53 : (tensor<32x64xf16, #blocked1>) -> tensor<32x64xf16, #shared1>
    %55 = tt.addptr %36, %cst_0 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
    %56 = tt.addptr %46, %50 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
    %57 = arith.subi %48, %c1_i32 : i32
    cf.br ^bb1(%c0_i32, %cst, %52, %54, %55, %56 : i32, tensor<64x64xf32, #mfma>, tensor<64x32xf16, #shared>, tensor<32x64xf16, #shared1>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>)
  ^bb1(%58: i32, %59: tensor<64x64xf32, #mfma>, %60: tensor<64x32xf16, #shared>, %61: tensor<32x64xf16, #shared1>, %62: tensor<64x32x!tt.ptr<f16>, #blocked>, %63: tensor<32x64x!tt.ptr<f16>, #blocked1>):  // 2 preds: ^bb0, ^bb2
    %64 = arith.cmpi slt, %58, %57 : i32
    cf.cond_br %64, ^bb2, ^bb3
  ^bb2:  // pred: ^bb1
    %65 = tt.load %62 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<64x32xf16, #blocked>
    %66 = tt.load %63 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked1>
    %67 = triton_gpu.convert_layout %60 : (tensor<64x32xf16, #shared>) -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
    %68 = triton_gpu.convert_layout %61 : (tensor<32x64xf16, #shared1>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
    %69 = tt.dot %67, %68, %59 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<64x64xf32, #mfma>
    %70 = tt.addptr %62, %cst_0 : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
    %71 = tt.addptr %63, %50 : tensor<32x64x!tt.ptr<f16>, #blocked1>, tensor<32x64xi32, #blocked1>
    %72 = triton_gpu.convert_layout %65 : (tensor<64x32xf16, #blocked>) -> tensor<64x32xf16, #shared>
    %73 = triton_gpu.convert_layout %66 : (tensor<32x64xf16, #blocked1>) -> tensor<32x64xf16, #shared1>
    %74 = arith.addi %58, %c1_i32 : i32
    cf.br ^bb1(%74, %69, %72, %73, %70, %71 : i32, tensor<64x64xf32, #mfma>, tensor<64x32xf16, #shared>, tensor<32x64xf16, #shared1>, tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<32x64x!tt.ptr<f16>, #blocked1>)
  ^bb3:  // pred: ^bb1
    %75 = triton_gpu.convert_layout %60 : (tensor<64x32xf16, #shared>) -> tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>>
    %76 = triton_gpu.convert_layout %61 : (tensor<32x64xf16, #shared1>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>>
    %77 = tt.dot %75, %76, %59 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32, transA = false, transB = false} : tensor<64x32xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mfma, kWidth = 8}>> * tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mfma, kWidth = 8}>> -> tensor<64x64xf32, #mfma>
    %78 = arith.truncf %77 : tensor<64x64xf32, #mfma> to tensor<64x64xf16, #mfma>
    %79 = tt.splat %arg8 : (i32) -> tensor<64x1xi32, #blocked1>
    %80 = arith.muli %79, %27 : tensor<64x1xi32, #blocked1>
    %81 = tt.splat %arg2 : (!tt.ptr<f16>) -> tensor<64x1x!tt.ptr<f16>, #blocked1>
    %82 = tt.addptr %81, %80 : tensor<64x1x!tt.ptr<f16>, #blocked1>, tensor<64x1xi32, #blocked1>
    %83 = tt.broadcast %82 : (tensor<64x1x!tt.ptr<f16>, #blocked1>) -> tensor<64x64x!tt.ptr<f16>, #blocked1>
    %84 = tt.broadcast %43 : (tensor<1x64xi32, #blocked1>) -> tensor<64x64xi32, #blocked1>
    %85 = tt.addptr %83, %84 : tensor<64x64x!tt.ptr<f16>, #blocked1>, tensor<64x64xi32, #blocked1>
    %86 = tt.splat %arg3 : (i32) -> tensor<64x1xi32, #blocked1>
    %87 = arith.cmpi "slt", %27, %86 : tensor<64x1xi32, #blocked1>
    %88 = tt.splat %arg4 : (i32) -> tensor<1x64xi32, #blocked1>
    %89 = arith.cmpi "slt", %43, %88 : tensor<1x64xi32, #blocked1>
    %90 = tt.broadcast %87 : (tensor<64x1xi1, #blocked1>) -> tensor<64x64xi1, #blocked1>
    %91 = tt.broadcast %89 : (tensor<1x64xi1, #blocked1>) -> tensor<64x64xi1, #blocked1>
    %92 = arith.andi %90, %91 : tensor<64x64xi1, #blocked1>
    %93 = triton_gpu.convert_layout %78 : (tensor<64x64xf16, #mfma>) -> tensor<64x64xf16, #blocked1>
    tt.store %85, %93, %92 {cache = 1 : i32, evict = 1 : i32} : tensor<64x64xf16, #blocked1>
    tt.return
  }
}
