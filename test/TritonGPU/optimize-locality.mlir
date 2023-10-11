#blocked = #triton_gpu.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CTAsPerCGA = [1, 1], CTASplitNum = [1, 1], CTAOrder = [0, 1]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CTAsPerCGA = [1], CTASplitNum = [1], CTAOrder = [0]}>
module attributes {"triton_gpu.compute-capability" = 90 : i32, "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_0d1d2de(%arg0: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32, 1> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 8 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c127_i32 = arith.constant 127 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = tt.get_num_programs {axis = 1 : i32} : i32
    %3 = arith.muli %0, %c32_i32 : i32
    %4 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %5 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked1>
    %6 = tt.splat %3 : (i32) -> tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.splat %3 : (i32) -> tensor<32xi32, #blocked1>
    %8 = arith.addi %6, %4 : tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    %9 = arith.addi %7, %5 : tensor<32xi32, #blocked1>
    %10 = arith.addi %arg2, %c127_i32 : i32
    %11 = arith.divsi %10, %c128_i32 : i32
    %12 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
    %13 = tt.expand_dims %8 {axis = 1 : i32} : (tensor<32xi32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32x1xi32, #blocked>
    %14 = tt.splat %arg2 : (i32) -> tensor<32x1xi32, #blocked>
    %15 = arith.muli %13, %14 : tensor<32x1xi32, #blocked>
    %16 = tt.splat %arg0 : (!tt.ptr<f32, 1>) -> tensor<32x1x!tt.ptr<f32, 1>, #blocked>
    %17 = tt.addptr %16, %15 : tensor<32x1x!tt.ptr<f32, 1>, #blocked>, tensor<32x1xi32, #blocked>
    %18 = tt.broadcast %17 : (tensor<32x1x!tt.ptr<f32, 1>, #blocked>) -> tensor<32x128x!tt.ptr<f32, 1>, #blocked>
    %19 = scf.for %arg3 = %1 to %11 step %2 iter_args(%arg4 = %cst) -> (tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>)  : i32 {
      %27 = arith.muli %arg3, %c128_i32 : i32
      %28 = tt.splat %27 : (i32) -> tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %29 = arith.addi %28, %12 : tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %30 = tt.expand_dims %29 {axis = 0 : i32} : (tensor<128xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<1x128xi32, #blocked>
      %31 = tt.broadcast %30 : (tensor<1x128xi32, #blocked>) -> tensor<32x128xi32, #blocked>
      %32 = tt.addptr %18, %31 : tensor<32x128x!tt.ptr<f32, 1>, #blocked>, tensor<32x128xi32, #blocked>
      %33 = tt.load %32 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<32x128xf32, #blocked>
      %34 = "tt.reduce"(%33) <{axis = 1 : i32}> ({
      ^bb0(%arg5: f32, %arg6: f32):
        %36 = arith.addf %arg5, %arg6 : f32
        tt.reduce.return %36 : f32
      }) : (tensor<32x128xf32, #blocked>) -> tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      %35 = arith.addf %arg4, %34 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
      scf.yield %35 : tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>
    }
    %20 = tt.splat %2 : (i32) -> tensor<32xi32, #blocked1>
    %21 = arith.muli %9, %20 : tensor<32xi32, #blocked1>
    %22 = tt.splat %arg1 : (!tt.ptr<f32, 1>) -> tensor<32x!tt.ptr<f32, 1>, #blocked1>
    %23 = tt.addptr %22, %21 : tensor<32x!tt.ptr<f32, 1>, #blocked1>, tensor<32xi32, #blocked1>
    %24 = tt.splat %1 : (i32) -> tensor<32xi32, #blocked1>
    %25 = tt.addptr %23, %24 : tensor<32x!tt.ptr<f32, 1>, #blocked1>, tensor<32xi32, #blocked1>
    %26 = triton_gpu.convert_layout %19 : (tensor<32xf32, #triton_gpu.slice<{dim = 1, parent = #blocked}>>) -> tensor<32xf32, #blocked1>
    tt.store %25, %26 {cache = 1 : i32, evict = 1 : i32} : tensor<32xf32, #blocked1>
    tt.return
  }
}
