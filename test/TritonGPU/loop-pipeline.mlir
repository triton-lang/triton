// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 -canonicalize | FileCheck %s

// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 1]}>
#BL = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 1]}>
#ALs0 = #triton_gpu.slice<{parent=#AL, dim=0}>
#BLs0 = #triton_gpu.slice<{parent=#BL, dim=0}>
#BLs1 = #triton_gpu.slice<{parent=#BL, dim=1}>
#C = #triton_gpu.mma<{versionMajor = 2, warpsPerCTA = [4, 1], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [1, 0]}>
#A = #triton_gpu.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #triton_gpu.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

tt.func @lut_temp(%77: tensor<16x16xi64, #BL> {tt.divisibility=16: i32, tt.constancy=16: i32},
                   %76: index,
                   %48: tensor<16x16x!tt.ptr<f16>, #AL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %49: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=2 : i32},
                   %75: tensor<16x!tt.ptr<i64>, #BLs1>,
                   %78: tensor<16x16xi32, #AL> {tt.constancy=16: i32, tt.divisibility=16: i32},
                   %60: tensor<16x16x!tt.ptr<f16>, #BL> {tt.divisibility=16: i32, tt.contiguity=16 : i32}) -> tensor<16x16xf32, #C>{
  %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #C>
  %c4_i32 = arith.constant 4 : i32
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c0_i64 = arith.constant 0 : i64
  %c1_i32 = arith.constant 1 : i32
  %c1_i32_splat = tt.splat %c1_i32 : (i32) -> tensor<16xi32, #BLs1>
  %1 = tt.load %48 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #AL>
  %79:3 = scf.for %arg18 = %c0 to %76 step %c1 iter_args(%arg19 = %cst, %arg20 = %49, %arg21 = %75) -> (tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x!tt.ptr<i64>, #BLs1>) {
    %2 = tt.load %arg21 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16xi64, #BLs1>
    %92 = tt.addptr %arg21, %c1_i32_splat : tensor<16x!tt.ptr<i64>, #BLs1>, tensor<16xi32, #BLs1>
    %84 = tt.expand_dims %2 {axis=1: i32}: (tensor<16xi64, #BLs1>) -> tensor<16x1xi64, #BL>
    %850 = tt.broadcast %84 : (tensor<16x1xi64, #BL>) -> tensor<16x16xi64, #BL>
    %85 = arith.muli %77, %850 : tensor<16x16xi64, #BL>
    %86 = tt.addptr %arg20, %85 : tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x16xi64, #BL>
    %87 = tt.load %86 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<16x16xf16, #BL>
    %88 = triton_gpu.convert_layout %1 : (tensor<16x16xf16, #AL>) -> tensor<16x16xf16, #A>
    %89 = triton_gpu.convert_layout %87 : (tensor<16x16xf16, #BL>) -> tensor<16x16xf16, #B>
    %90 = tt.dot %88, %89, %arg19 {allowTF32 = true, maxNumImpreciseAcc = 0 : i32} : tensor<16x16xf16, #A> * tensor<16x16xf16, #B> -> tensor<16x16xf32, #C>
    scf.yield %90, %86, %92 : tensor<16x16xf32, #C>, tensor<16x16x!tt.ptr<f16>, #BL>, tensor<16x!tt.ptr<i64>, #BLs1>
  }
  tt.return %79#0 : tensor<16x16xf32, #C>
}
