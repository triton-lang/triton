// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions -cse | FileCheck %s

// Check that we can hoist ttg.convert_layout ops that eventually feed into dot
// for decomposed mxfp emulation for AMD GPUs.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 16], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked5 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [16, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 64, 1], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>
#blocked7 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 64], warpsPerCTA = [2, 1, 2], order = [1, 2, 0]}>
#linear = #ttg.linear<{register = [[1, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], warp = [[0, 64], [2, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [64, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], warp = [[0, 64], [32, 0]], block = []}>
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @fp8_mxfp4_matmul_decompose
  tt.func public @fp8_mxfp4_matmul_decompose(%59: i32, %71: tensor<128x128x!tt.ptr<f32>, #blocked4>, %47: tensor<128x128x!tt.ptr<f8E5M2>, #blocked3>, %57: tensor<64x128x!tt.ptr<i8>, #blocked3>, %37: tensor<128x4x!tt.ptr<i8>, #blocked2>, %61: tensor<64x128xi32, #blocked3>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0x7FC0> : tensor<128x128xbf16, #linear>
    %cst_0 = arith.constant dense<-1> : tensor<4x128xi8, #blocked>
    %cst_1 = arith.constant dense<7> : tensor<4x128xi16, #blocked>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_3 = arith.constant dense<4> : tensor<128x4xi32, #blocked2>
    %cst_4 = arith.constant dense<128> : tensor<128x128xi32, #blocked3>
    //     CHECK: scf.for
    //     CHECK:   tt.load
    //     CHECK:   ttg.convert_layout
    //     CHECK:   tt.load
    //     CHECK:   ttg.convert_layout
    //     CHECK:   tt.load
    //     CHECK:   ttg.convert_layout
    // CHECK-NOT:   ttg.convert_layout
    //     CHECK:   scf.yield
    %62:4 = scf.for %arg11 = %c0_i32 to %59 step %c1_i32 iter_args(%arg12 = %cst_2, %arg13 = %47, %arg14 = %57, %arg15 = %37) -> (tensor<128x128xf32, #blocked1>, tensor<128x128x!tt.ptr<f8E5M2>, #blocked3>, tensor<64x128x!tt.ptr<i8>, #blocked3>, tensor<128x4x!tt.ptr<i8>, #blocked2>)  : i32 {
      %80 = tt.load %arg13 : tensor<128x128x!tt.ptr<f8E5M2>, #blocked3>
      %81 = ttg.convert_layout %80 : tensor<128x128xf8E5M2, #blocked3> -> tensor<128x128xf8E5M2, #blocked1>
      %82 = tt.load %arg14 : tensor<64x128x!tt.ptr<i8>, #blocked3>
      %83 = ttg.convert_layout %82 : tensor<64x128xi8, #blocked3> -> tensor<64x128xi8, #blocked1>
      %84 = tt.load %arg15 : tensor<128x4x!tt.ptr<i8>, #blocked2>
      %85 = ttg.convert_layout %84 : tensor<128x4xi8, #blocked2> -> tensor<128x4xi8, #blocked5>
      %86 = tt.fp_to_fp %81 : tensor<128x128xf8E5M2, #blocked1> -> tensor<128x128xbf16, #blocked1>
      %87 = ttg.convert_layout %86 : tensor<128x128xbf16, #blocked1> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
      %88 = ttg.fp4_to_fp %83 {axis = 0 : i32} : tensor<64x128xi8, #blocked1> -> tensor<128x128xbf16, #linear>
      %89 = tt.trans %85 {order = array<i32: 1, 0>} : tensor<128x4xi8, #blocked5> -> tensor<4x128xi8, #blocked>
      %90 = arith.extui %89 : tensor<4x128xi8, #blocked> to tensor<4x128xi16, #blocked>
      %91 = arith.shli %90, %cst_1 : tensor<4x128xi16, #blocked>
      %92 = tt.bitcast %91 : tensor<4x128xi16, #blocked> -> tensor<4x128xbf16, #blocked>
      %93 = ttg.convert_layout %92 : tensor<4x128xbf16, #blocked> -> tensor<4x128xbf16, #ttg.slice<{dim = 2, parent = #blocked6}>>
      %94 = tt.expand_dims %93 {axis = 2 : i32} : tensor<4x128xbf16, #ttg.slice<{dim = 2, parent = #blocked6}>> -> tensor<4x128x1xbf16, #blocked6>
      %95 = tt.broadcast %94 : tensor<4x128x1xbf16, #blocked6> -> tensor<4x128x32xbf16, #blocked6>
      %96 = tt.trans %95 {order = array<i32: 0, 2, 1>} : tensor<4x128x32xbf16, #blocked6> -> tensor<4x32x128xbf16, #blocked7>
      %97 = tt.reshape %96 : tensor<4x32x128xbf16, #blocked7> -> tensor<128x128xbf16, #linear1>
      %98 = ttg.convert_layout %97 : tensor<128x128xbf16, #linear1> -> tensor<128x128xbf16, #linear>
      %99 = arith.mulf %88, %98 : tensor<128x128xbf16, #linear>
      %100 = arith.cmpi eq, %89, %cst_0 : tensor<4x128xi8, #blocked>
      %101 = ttg.convert_layout %100 : tensor<4x128xi1, #blocked> -> tensor<4x128xi1, #ttg.slice<{dim = 2, parent = #blocked6}>>
      %102 = tt.expand_dims %101 {axis = 2 : i32} : tensor<4x128xi1, #ttg.slice<{dim = 2, parent = #blocked6}>> -> tensor<4x128x1xi1, #blocked6>
      %103 = tt.broadcast %102 : tensor<4x128x1xi1, #blocked6> -> tensor<4x128x32xi1, #blocked6>
      %104 = tt.trans %103 {order = array<i32: 0, 2, 1>} : tensor<4x128x32xi1, #blocked6> -> tensor<4x32x128xi1, #blocked7>
      %105 = tt.reshape %104 : tensor<4x32x128xi1, #blocked7> -> tensor<128x128xi1, #linear1>
      %106 = ttg.convert_layout %105 : tensor<128x128xi1, #linear1> -> tensor<128x128xi1, #linear>
      %107 = arith.select %106, %cst, %99 : tensor<128x128xi1, #linear>, tensor<128x128xbf16, #linear>
      %108 = ttg.convert_layout %107 : tensor<128x128xbf16, #linear> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
      %109 = ttg.convert_layout %arg12 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #mma>
      %110 = ttg.convert_layout %87 : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
      %111 = ttg.convert_layout %108 : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      %112 = tt.dot %110, %111, %109 : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<128x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<128x128xf32, #mma>
      %113 = ttg.convert_layout %112 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked1>
      %114 = ttg.convert_layout %113 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked1>
      %115 = tt.addptr %arg13, %cst_4 : tensor<128x128x!tt.ptr<f8E5M2>, #blocked3>, tensor<128x128xi32, #blocked3>
      %116 = tt.addptr %arg14, %61 : tensor<64x128x!tt.ptr<i8>, #blocked3>, tensor<64x128xi32, #blocked3>
      %117 = tt.addptr %arg15, %cst_3 : tensor<128x4x!tt.ptr<i8>, #blocked2>, tensor<128x4xi32, #blocked2>
      scf.yield %114, %115, %116, %117 : tensor<128x128xf32, #blocked1>, tensor<128x128x!tt.ptr<f8E5M2>, #blocked3>, tensor<64x128x!tt.ptr<i8>, #blocked3>, tensor<128x4x!tt.ptr<i8>, #blocked2>
    } {tt.num_stages = 2 : i32}
    %79 = ttg.convert_layout %62#0 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked4>
    tt.store %71, %79 : tensor<128x128x!tt.ptr<f32>, #blocked4>
    tt.return
  }
}
