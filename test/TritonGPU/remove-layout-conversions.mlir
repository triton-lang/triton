// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

// CHECK-LABEL: hoist_convert_above_extf_and_remat
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked3 = #triton_gpu.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#blocked4 = #triton_gpu.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
#mma = #triton_gpu.mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0], instrShape = [16, 8]}>
#shared = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
#shared1 = #triton_gpu.shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1], hasLeadingOffset = false}>
module attributes {"triton_gpu.compute-capability" = 80 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @hoist_convert_above_extf_and_remat(%arg0: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32},
                                                    %arg1: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32},
                                                    %arg2: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32},
                                                    %arg3: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32},
                                                    %arg4: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32},
                                                    %arg5: !tt.ptr<f16, 1> {tt.divisibility = 16 : i32},
                                                    %arg6: !tt.ptr<f16, 1>)  attributes {noinline = false} {
    %cst = arith.constant dense<256> : tensor<32x1xi32, #blocked>
    %cst_0 = arith.constant dense<256> : tensor<32x1xi32, #blocked1>
    %cst_1 = arith.constant dense<256> : tensor<256x1xi32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_2 = arith.constant dense<1.000000e-03> : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %cst_3 = arith.constant dense<2.560000e+02> : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %cst_4 = arith.constant dense<0.000000e+00> : tensor<32x256xf32, #blocked3>
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %4 = tt.splat %1 : (i32) -> tensor<32x1xi32, #blocked>
    %5 = arith.addi %4, %3 : tensor<32x1xi32, #blocked>
    %6 = arith.muli %5, %cst : tensor<32x1xi32, #blocked>
    %7 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %9 = tt.expand_dims %7 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x64xi32, #blocked>
    %10 = tt.expand_dims %8 {axis = 0 : i32} : (tensor<64xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x64xi32, #blocked>
    %11 = tt.broadcast %9 : (tensor<1x64xi32, #blocked>) -> tensor<32x64xi32, #blocked>
    %12 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32} : (tensor<256xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<256x1xi32, #blocked>
    %14 = arith.muli %13, %cst_1 : tensor<256x1xi32, #blocked>
    %15 = tt.broadcast %10 : (tensor<1x64xi32, #blocked>) -> tensor<256x64xi32, #blocked>
    %16 = tt.splat %arg0 : (!tt.ptr<f16, 1>) -> tensor<32x64x!tt.ptr<f16, 1>, #blocked>
    %17 = tt.splat %arg1 : (!tt.ptr<f16, 1>) -> tensor<256x64x!tt.ptr<f16, 1>, #blocked>
    %18 = scf.for %arg7 = %c0_i32 to %c256_i32 step %c64_i32 iter_args(%arg8 = %cst_4) -> (tensor<32x256xf32, #blocked3>)  : i32 {
      %76 = tt.splat %arg7 : (i32) -> tensor<32x1xi32, #blocked>
      %77 = arith.addi %6, %76 : tensor<32x1xi32, #blocked>
      %78 = tt.broadcast %77 : (tensor<32x1xi32, #blocked>) -> tensor<32x64xi32, #blocked>
      %79 = arith.addi %78, %11 : tensor<32x64xi32, #blocked>
      %80 = tt.splat %arg7 : (i32) -> tensor<256x1xi32, #blocked>
      %81 = arith.addi %14, %80 : tensor<256x1xi32, #blocked>
      %82 = tt.broadcast %81 : (tensor<256x1xi32, #blocked>) -> tensor<256x64xi32, #blocked>
      %83 = arith.addi %82, %15 : tensor<256x64xi32, #blocked>
      %84 = tt.addptr %16, %79 : tensor<32x64x!tt.ptr<f16, 1>, #blocked>, tensor<32x64xi32, #blocked>
      %85 = tt.load %84 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x64xf16, #blocked>
      %86 = tt.addptr %17, %83 : tensor<256x64x!tt.ptr<f16, 1>, #blocked>, tensor<256x64xi32, #blocked>
      %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<256x64xf16, #blocked>
      %88 = triton_gpu.convert_layout %87 : (tensor<256x64xf16, #blocked>) -> tensor<256x64xf16, #shared>
      %89 = tt.trans %88 : (tensor<256x64xf16, #shared>) -> tensor<64x256xf16, #shared1>
      %90 = triton_gpu.convert_layout %85 : (tensor<32x64xf16, #blocked>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked3}>>
      %91 = triton_gpu.convert_layout %89 : (tensor<64x256xf16, #shared1>) -> tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked3}>>
      %92 = triton_gpu.convert_layout %arg8 : (tensor<32x256xf32, #blocked3>) -> tensor<32x256xf32, #mma>
      %93 = triton_gpu.convert_layout %90 : (tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked3}>>) -> tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %94 = triton_gpu.convert_layout %91 : (tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked3}>>) -> tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %95 = tt.dot %93, %94, %92 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<32x64xf16, #triton_gpu.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x256xf16, #triton_gpu.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x256xf32, #mma>
      %96 = triton_gpu.convert_layout %95 : (tensor<32x256xf32, #mma>) -> tensor<32x256xf32, #blocked3>
      scf.yield %96 : tensor<32x256xf32, #blocked3>
    }
    %19 = arith.truncf %18 : tensor<32x256xf32, #blocked3> to tensor<32x256xf16, #blocked3>
    %20 = triton_gpu.convert_layout %19 : (tensor<32x256xf16, #blocked3>) -> tensor<32x256xf16, #blocked2>
    %21 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked4>
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked4>
    %23 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>
    %24 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>
    %25 = tt.expand_dims %23 {axis = 0 : i32} : (tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked2}>>) -> tensor<1x256xi32, #blocked2>
    %26 = tt.expand_dims %24 {axis = 0 : i32} : (tensor<256xi32, #triton_gpu.slice<{dim = 0, parent = #blocked1}>>) -> tensor<1x256xi32, #blocked1>
    %27 = tt.splat %arg2 : (!tt.ptr<f16, 1>) -> tensor<1x256x!tt.ptr<f16, 1>, #blocked2>
    %28 = tt.addptr %27, %25 : tensor<1x256x!tt.ptr<f16, 1>, #blocked2>, tensor<1x256xi32, #blocked2>
    %29 = tt.load %28 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<1x256xf16, #blocked2>
    %30 = tt.broadcast %29 : (tensor<1x256xf16, #blocked2>) -> tensor<32x256xf16, #blocked2>
    %31 = arith.addf %20, %30 : tensor<32x256xf16, #blocked2>
    %42 = arith.extf %31 : tensor<32x256xf16, #blocked2> to tensor<32x256xf32, #blocked2>
    // CHECK:  %[[cvt:.*]] = triton_gpu.convert_layout %{{.*}} : (tensor<32x256xf16, #mma>)
    // CHECK:  arith.extf %[[cvt]]
    // CHECK:  arith.extf %[[cvt]]
    // CHECK:  arith.extf %[[cvt]]
    // CHECK:  arith.extf %[[cvt]]
    %43 = "tt.reduce"(%42) <{axis = 1 : i32}> ({
    ^bb0(%arg7: f32 loc(unknown), %arg8: f32 loc(unknown)):
      %76 = arith.addf %arg7, %arg8 : f32
      tt.reduce.return %76 : f32
    }) : (tensor<32x256xf32, #blocked2>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %44 = arith.divf %43, %cst_3 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %45 = arith.mulf %42, %42 : tensor<32x256xf32, #blocked2>
    %46 = "tt.reduce"(%45) <{axis = 1 : i32}> ({
    ^bb0(%arg7: f32 loc(unknown), %arg8: f32 loc(unknown)):
      %76 = arith.addf %arg7, %arg8 : f32
      tt.reduce.return %76 : f32
    }) : (tensor<32x256xf32, #blocked2>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %47 = arith.divf %46, %cst_3 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %48 = arith.mulf %44, %44 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %49 = arith.subf %47, %48 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %50 = math.sqrt %49 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %51 = arith.addf %50, %cst_2 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>
    %52 = tt.expand_dims %44 {axis = 1 : i32} : (tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<32x1xf32, #blocked2>
    %53 = tt.expand_dims %51 {axis = 1 : i32} : (tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked2}>>) -> tensor<32x1xf32, #blocked2>
    %54 = tt.broadcast %52 : (tensor<32x1xf32, #blocked2>) -> tensor<32x256xf32, #blocked2>
    %55 = arith.subf %42, %54 : tensor<32x256xf32, #blocked2>
    %56 = tt.broadcast %53 : (tensor<32x1xf32, #blocked2>) -> tensor<32x256xf32, #blocked2>
    %57 = arith.divf %55, %56 : tensor<32x256xf32, #blocked2>
    %64 = arith.truncf %57 : tensor<32x256xf32, #blocked2> to tensor<32x256xf16, #blocked2>
    %65 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>
    %66 = tt.expand_dims %65 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked1}>>) -> tensor<32x1xi32, #blocked1>
    %67 = arith.muli %66, %cst_0 : tensor<32x1xi32, #blocked1>
    %68 = tt.splat %1 : (i32) -> tensor<32x1xi32, #blocked1>
    %69 = arith.addi %68, %67 : tensor<32x1xi32, #blocked1>
    %70 = tt.broadcast %69 : (tensor<32x1xi32, #blocked1>) -> tensor<32x256xi32, #blocked1>
    %71 = tt.broadcast %26 : (tensor<1x256xi32, #blocked1>) -> tensor<32x256xi32, #blocked1>
    %72 = arith.addi %70, %71 : tensor<32x256xi32, #blocked1>
    %73 = tt.splat %arg5 : (!tt.ptr<f16, 1>) -> tensor<32x256x!tt.ptr<f16, 1>, #blocked1>
    %74 = tt.addptr %73, %72 : tensor<32x256x!tt.ptr<f16, 1>, #blocked1>, tensor<32x256xi32, #blocked1>
    %75 = triton_gpu.convert_layout %64 : (tensor<32x256xf16, #blocked2>) -> tensor<32x256xf16, #blocked1>
    tt.store %74, %75 {cache = 1 : i32, evict = 1 : i32} : tensor<32x256xf16, #blocked1>
    tt.return
  }
}