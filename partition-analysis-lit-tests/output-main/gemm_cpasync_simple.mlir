#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"nvws.warp-specialized" = true, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm_cpasync_simple(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant {ttg.partitions = [1 : i32]} false
    %cst = arith.constant {ttg.partitions = [0 : i32]} dense<128> : tensor<128x128xi32, #blocked>
    %c256_i32 = arith.constant {ttg.partitions = [0 : i32, 2 : i32]} 256 : i32
    %c128_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32, 2 : i32]} 128 : i32
    %c0_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32]} 0 : i32
    %c1_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32]} 1 : i32
    %c127_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32, 2 : i32]} 127 : i32
    %true = arith.constant {ttg.partitions = [1 : i32]} true
    %0 = tt.get_program_id x {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %1 = arith.addi %arg4, %c127_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %2 = arith.divsi %1, %c128_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %3 = arith.remsi %0, %2 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %4 = arith.divsi %0, %2 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %5 = arith.muli %3, %c128_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %6 = tt.make_range {end = 128 : i32, start = 0 : i32, ttg.partitions = [0 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %7 = tt.make_range {end = 128 : i32, start = 0 : i32, ttg.partitions = [0 : i32, 2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %8 = tt.splat %5 {ttg.partitions = [0 : i32]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %9 = tt.splat %5 {ttg.partitions = [2 : i32]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %10 = arith.addi %8, %6 {ttg.partitions = [0 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %11 = arith.addi %9, %7 {ttg.partitions = [2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %12 = tt.splat %arg4 {ttg.partitions = [0 : i32]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = arith.remsi %10, %12 {ttg.partitions = [0 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %14 = arith.muli %4, %c256_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %15 = tt.make_range {end = 256 : i32, start = 0 : i32, ttg.partitions = [0 : i32, 2 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %16 = tt.make_range {end = 256 : i32, start = 0 : i32, ttg.partitions = [2 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %17 = tt.splat %14 {ttg.partitions = [0 : i32, 2 : i32]} : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %18 = tt.splat %14 {ttg.partitions = [2 : i32]} : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %19 = arith.addi %17, %15 {ttg.partitions = [0 : i32, 2 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %20 = arith.addi %18, %16 {ttg.partitions = [2 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %21 = tt.splat %arg5 {ttg.partitions = [0 : i32]} : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %22 = tt.splat %arg5 {ttg.partitions = [2 : i32]} : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %23 = arith.remsi %19, %21 {ttg.partitions = [0 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %24 = arith.remsi %20, %22 {ttg.partitions = [2 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %25 = tt.expand_dims %13 {axis = 1 : i32, ttg.partitions = [0 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %26 = tt.splat %arg7 {ttg.partitions = [0 : i32]} : i32 -> tensor<128x1xi32, #blocked>
    %27 = arith.muli %25, %26 {ttg.partitions = [0 : i32]} : tensor<128x1xi32, #blocked>
    %28 = tt.make_range {end = 128 : i32, start = 0 : i32, ttg.partitions = [0 : i32]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %29 = tt.expand_dims %28 {axis = 0 : i32, ttg.partitions = [0 : i32]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %30 = tt.broadcast %27 {ttg.partitions = [0 : i32]} : tensor<128x1xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %31 = tt.broadcast %29 {ttg.partitions = [0 : i32]} : tensor<1x128xi32, #blocked> -> tensor<128x128xi32, #blocked>
    %32 = arith.addi %30, %31 {ttg.partitions = [0 : i32]} : tensor<128x128xi32, #blocked>
    %33 = tt.splat %arg0 {ttg.partitions = [0 : i32]} : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked>
    %34 = tt.addptr %33, %32 {ttg.partitions = [0 : i32]} : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %35 = tt.expand_dims %7 {axis = 1 : i32, ttg.partitions = [0 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %36 = tt.splat %arg8 {ttg.partitions = [0 : i32]} : i32 -> tensor<128x1xi32, #blocked1>
    %37 = arith.muli %35, %36 {ttg.partitions = [0 : i32]} : tensor<128x1xi32, #blocked1>
    %38 = tt.expand_dims %23 {axis = 0 : i32, ttg.partitions = [0 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %39 = tt.expand_dims %24 {axis = 0 : i32, ttg.partitions = [2 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
    %40 = tt.broadcast %37 {ttg.partitions = [0 : i32]} : tensor<128x1xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %41 = tt.broadcast %38 {ttg.partitions = [0 : i32]} : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %42 = arith.addi %40, %41 {ttg.partitions = [0 : i32]} : tensor<128x256xi32, #blocked1>
    %43 = tt.splat %arg1 {ttg.partitions = [0 : i32]} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
    %44 = tt.addptr %43, %42 {ttg.partitions = [0 : i32]} : tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<128x256xi32, #blocked1>
    %45 = arith.addi %arg6, %c127_i32 {ttg.partitions = [0 : i32, 1 : i32]} : i32
    %46 = arith.divsi %45, %c128_i32 {ttg.partitions = [0 : i32, 1 : i32]} : i32
    %result, %token = ttng.tmem_alloc {ttg.partitions = [1 : i32, 2 : i32]} : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %47:4 = scf.for %arg10 = %c0_i32 to %46 step %c1_i32 iter_args(%arg11 = %34, %arg12 = %44, %arg13 = %false, %arg14 = %token) -> (tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x256x!tt.ptr<f16>, #blocked1>, i1, !ttg.async.token)  : i32 {
      %65 = tt.load %arg11 {ttg.partitions = [0 : i32]} : tensor<128x128x!tt.ptr<f16>, #blocked>
      %66 = ttg.local_alloc %65 {ttg.partitions = [0 : i32]} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %67 = tt.load %arg12 {ttg.partitions = [0 : i32]} : tensor<128x256x!tt.ptr<f16>, #blocked1>
      %68 = ttg.local_alloc %67 {ttg.partitions = [0 : i32]} : (tensor<128x256xf16, #blocked1>) -> !ttg.memdesc<128x256xf16, #shared, #smem>
      %69 = ttng.tc_gen5_mma %66, %68, %result[%arg14], %arg13, %true {ttg.partitions = [1 : i32]} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x256xf16, #shared, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %70 = tt.addptr %arg11, %cst {ttg.partitions = [0 : i32]} : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
      %71 = arith.muli %arg8, %c128_i32 {ttg.partitions = [0 : i32]} : i32
      %72 = tt.splat %71 {ttg.partitions = [0 : i32]} : i32 -> tensor<128x256xi32, #blocked1>
      %73 = tt.addptr %arg12, %72 {ttg.partitions = [0 : i32]} : tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<128x256xi32, #blocked1>
      scf.yield %70, %73, %true, %69 : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x256x!tt.ptr<f16>, #blocked1>, i1, !ttg.async.token
    } {ttg.partitions = [0 : i32, 1 : i32], ttg.partitions.0 = [0 : i32], ttg.partitions.1 = [0 : i32], ttg.partitions.2 = [1 : i32], ttg.partitions.3 = [1 : i32]}
    %result_0, %token_1 = ttng.tmem_load %result[%47#3] {ttg.partitions = [2 : i32]} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked3>
    %48 = arith.truncf %result_0 {ttg.partitions = [2 : i32]} : tensor<128x256xf32, #blocked3> to tensor<128x256xf16, #blocked3>
    %49 = ttg.convert_layout %48 {ttg.partitions = [2 : i32]} : tensor<128x256xf16, #blocked3> -> tensor<128x256xf16, #blocked1>
    %50 = tt.splat %arg2 {ttg.partitions = [2 : i32]} : !tt.ptr<f16> -> tensor<1x256x!tt.ptr<f16>, #blocked2>
    %51 = tt.addptr %50, %39 {ttg.partitions = [2 : i32]} : tensor<1x256x!tt.ptr<f16>, #blocked2>, tensor<1x256xi32, #blocked2>
    %52 = tt.load %51 {ttg.partitions = [2 : i32]} : tensor<1x256x!tt.ptr<f16>, #blocked2>
    %53 = tt.expand_dims %11 {axis = 1 : i32, ttg.partitions = [2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1>
    %54 = tt.splat %arg9 {ttg.partitions = [2 : i32]} : i32 -> tensor<128x1xi32, #blocked1>
    %55 = arith.muli %54, %53 {ttg.partitions = [2 : i32]} : tensor<128x1xi32, #blocked1>
    %56 = tt.splat %arg3 {ttg.partitions = [2 : i32]} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked1>
    %57 = tt.addptr %56, %55 {ttg.partitions = [2 : i32]} : tensor<128x1x!tt.ptr<f16>, #blocked1>, tensor<128x1xi32, #blocked1>
    %58 = tt.expand_dims %19 {axis = 0 : i32, ttg.partitions = [2 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x256xi32, #blocked1>
    %59 = tt.broadcast %57 {ttg.partitions = [2 : i32]} : tensor<128x1x!tt.ptr<f16>, #blocked1> -> tensor<128x256x!tt.ptr<f16>, #blocked1>
    %60 = tt.broadcast %58 {ttg.partitions = [2 : i32]} : tensor<1x256xi32, #blocked1> -> tensor<128x256xi32, #blocked1>
    %61 = tt.addptr %59, %60 {ttg.partitions = [2 : i32]} : tensor<128x256x!tt.ptr<f16>, #blocked1>, tensor<128x256xi32, #blocked1>
    %62 = ttg.convert_layout %52 {ttg.partitions = [2 : i32]} : tensor<1x256xf16, #blocked2> -> tensor<1x256xf16, #blocked1>
    %63 = tt.broadcast %62 {ttg.partitions = [2 : i32]} : tensor<1x256xf16, #blocked1> -> tensor<128x256xf16, #blocked1>
    %64 = arith.addf %49, %63 {ttg.partitions = [2 : i32]} : tensor<128x256xf16, #blocked1>
    tt.store %61, %64 {ttg.partitions = [2 : i32]} : tensor<128x256x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}

