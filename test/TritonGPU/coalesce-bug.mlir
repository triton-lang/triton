// RUN: triton-opt %s -tritongpu-coalesce

#blocked = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 70 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 1 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @triton_softmax_clone_impl(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant dense<127> : tensor<128xi32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked>
    %cst_1 = arith.constant dense<127> : tensor<128xi64, #blocked>
    %cst_2 = arith.constant dense<0> : tensor<128xi64, #blocked>
    %c127_i64 = arith.constant 127 : i64
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.muli %1, %c127_i64 : i64
    %3 = tt.addptr %arg0, %2 : !tt.ptr<f32, 1>, i64
    %4 = tt.splat %3 : (!tt.ptr<f32, 1>) -> tensor<128x!tt.ptr<f32, 1>, #blocked>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %6 = arith.extsi %5 : tensor<128xi32, #blocked> to tensor<128xi64, #blocked>
    %7 = tt.addptr %4, %6 : tensor<128x!tt.ptr<f32, 1>, #blocked>, tensor<128xi64, #blocked>
    %8 = arith.cmpi sge, %6, %cst_2 : tensor<128xi64, #blocked>
    %9 = arith.cmpi slt, %6, %cst_1 : tensor<128xi64, #blocked>
    %10 = arith.andi %8, %9 : tensor<128xi1, #blocked>
    %11 = tt.load %7, %10, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128xf32, #blocked>
    %12 = tt.addptr %arg1, %c0_i64 : !tt.ptr<f32, 1>, i64
    %13 = tt.splat %12 : (!tt.ptr<f32, 1>) -> tensor<128x!tt.ptr<f32, 1>, #blocked>
    %14 = tt.addptr %13, %6 : tensor<128x!tt.ptr<f32, 1>, #blocked>, tensor<128xi64, #blocked>
    %15 = tt.load %14, %10, %cst_0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<128xf32, #blocked>
    %16 = arith.mulf %11, %15 : tensor<128xf32, #blocked>
    %17 = arith.cmpi slt, %5, %cst : tensor<128xi32, #blocked>
    %18 = arith.select %17, %16, %cst_0 : tensor<128xi1, #blocked>, tensor<128xf32, #blocked>
    %19 = "tt.reduce"(%18) <{axis = 0 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      %25 = arith.addf %arg3, %arg4 : f32
      tt.reduce.return %25 : f32
    }) : (tensor<128xf32, #blocked>) -> f32
    %20 = tt.splat %19 : (f32) -> tensor<128xf32, #blocked>
    %21 = arith.mulf %16, %20 : tensor<128xf32, #blocked>
    %22 = tt.addptr %arg2, %2 : !tt.ptr<f32, 1>, i64
    %23 = tt.splat %22 : (!tt.ptr<f32, 1>) -> tensor<128x!tt.ptr<f32, 1>, #blocked>
    %24 = tt.addptr %23, %6 : tensor<128x!tt.ptr<f32, 1>, #blocked>, tensor<128xi64, #blocked>
    tt.store %24, %21, %10 {cache = 1 : i32, evict = 1 : i32} : tensor<128xf32, #blocked>
    tt.return
  }
}
