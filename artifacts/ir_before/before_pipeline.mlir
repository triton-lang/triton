#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @gemm_scatter_kernel(%arg0: index, %arg1: index, %arg2: index, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #mma> {
    %cst = arith.constant dense<4.000000e+00> : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %cst_0 = arith.constant dense<4> : tensor<32x128xi32, #blocked>
    %cst_1 = arith.constant dense<4> : tensor<128x32xi32, #blocked1>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0 = tt.splat %arg3 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #blocked1>
    %1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %2 = tt.expand_dims %1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x32xi32, #blocked1>
    %3 = tt.broadcast %2 : tensor<1x32xi32, #blocked1> -> tensor<128x32xi32, #blocked1>
    %4 = tt.addptr %0, %3 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
    %5 = tt.splat %arg4 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #blocked>
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %8 = tt.broadcast %7 : tensor<1x128xi32, #blocked> -> tensor<32x128xi32, #blocked>
    %9 = tt.addptr %5, %8 : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %10:3 = scf.for %arg6 = %arg0 to %arg1 step %arg2 iter_args(%arg7 = %4, %arg8 = %9, %arg9 = %cst_2) -> (tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>) {
      %13 = tt.load %arg7 : tensor<128x32x!tt.ptr<f16>, #blocked1>
      %14 = ttg.convert_layout %13 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x32xf16, #blocked1> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %15 = tt.load %arg8 : tensor<32x128x!tt.ptr<f16>, #blocked>
      %16 = ttg.convert_layout %15 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<32x128xf16, #blocked> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %17 = arith.mulf %16, %cst {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %18 = tt.dot %14, %17, %arg9 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x128xf32, #mma>
      %19 = tt.addptr %arg7, %cst_1 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<128x32xi32, #blocked1>
      %20 = tt.addptr %arg8, %cst_0 {loop.cluster = 2 : i32, loop.stage = 1 : i32} : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
      scf.yield %19, %20, %18 : tensor<128x32x!tt.ptr<f16>, #blocked1>, tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<128x128xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    %11 = ttg.convert_layout %10#2 : tensor<128x128xf32, #mma> -> tensor<128x128xf32, #blocked1>
    %12 = tt.splat %arg5 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #blocked1>
    tt.store %12, %11 : tensor<128x128x!tt.ptr<f32>, #blocked1>
    tt.return %10#2 : tensor<128x128xf32, #mma>
  }
}

