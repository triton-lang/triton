#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {nvws.group.epilogue = {num_warps = 4 : i32, reg_count = 248 : i32, start_warp = 0 : i32}, nvws.group.mma = {num_warps = 1 : i32, reg_count = 24 : i32, start_warp = 8 : i32}, nvws.group.tma_load = {num_warps = 4 : i32, reg_count = 248 : i32, start_warp = 4 : i32}, "nvws.warp-specialized" = true, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm_cpasync_mixed(%arg0: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant {groups = [@nvws.group.mma]} false
    %true = arith.constant {groups = [@nvws.group.mma]} true
    %c128_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} 128 : i32
    %c256_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} 256 : i32
    %c0_i32 = arith.constant {groups = [@nvws.group.mma, @nvws.group.tma_load]} 0 : i32
    %c1_i32 = arith.constant {groups = [@nvws.group.mma, @nvws.group.tma_load]} 1 : i32
    %c127_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} 127 : i32
    %0 = tt.get_program_id x {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
    %1 = arith.addi %arg7, %c127_i32 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
    %2 = arith.divsi %1, %c128_i32 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
    %3 = arith.remsi %0, %2 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
    %4 = arith.divsi %0, %2 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
    %5 = arith.muli %3, %c128_i32 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
    %6 = arith.muli %4, %c256_i32 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
    %7 = tt.make_range {end = 256 : i32, groups = [@nvws.group.epilogue, @nvws.group.tma_load], start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %8 = tt.splat %6 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %9 = arith.addi %8, %7 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %10 = tt.splat %arg8 {groups = [@nvws.group.tma_load]} : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %11 = arith.remsi %9, %10 {groups = [@nvws.group.tma_load]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %12 = tt.make_range {end = 128 : i32, groups = [@nvws.group.epilogue, @nvws.group.tma_load], start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %13 = tt.expand_dims %12 {axis = 1 : i32, groups = [@nvws.group.tma_load]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %14 = tt.splat %arg10 {groups = [@nvws.group.tma_load]} : i32 -> tensor<128x1xi32, #blocked>
    %15 = arith.muli %13, %14 {groups = [@nvws.group.tma_load]} : tensor<128x1xi32, #blocked>
    %16 = tt.expand_dims %11 {axis = 0 : i32, groups = [@nvws.group.tma_load]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %17 = tt.broadcast %15 {groups = [@nvws.group.tma_load]} : tensor<128x1xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %18 = tt.broadcast %16 {groups = [@nvws.group.tma_load]} : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %19 = arith.addi %17, %18 {groups = [@nvws.group.tma_load]} : tensor<128x256xi32, #blocked>
    %20 = tt.splat %arg5 {groups = [@nvws.group.tma_load]} : !tt.ptr<f16> -> tensor<128x256x!tt.ptr<f16>, #blocked>
    %21 = tt.addptr %20, %19 {groups = [@nvws.group.tma_load]} : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
    %22 = arith.addi %arg9, %c127_i32 {groups = [@nvws.group.mma, @nvws.group.tma_load]} : i32
    %23 = arith.divsi %22, %c128_i32 {groups = [@nvws.group.mma, @nvws.group.tma_load]} : i32
    %result, %token = ttng.tmem_alloc {groups = [@nvws.group.epilogue, @nvws.group.mma]} : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %24:4 = scf.for %arg12 = %c0_i32 to %23 step %c1_i32 iter_args(%arg13 = %c0_i32, %arg14 = %21, %arg15 = %false, %arg16 = %token) -> (i32, tensor<128x256x!tt.ptr<f16>, #blocked>, i1, !ttg.async.token)  : i32 {
      %38 = tt.descriptor_load %arg0[%5, %arg13] {groups = [@nvws.group.tma_load]} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      %39 = ttg.local_alloc %38 {groups = [@nvws.group.tma_load]} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %40 = tt.load %arg14 {groups = [@nvws.group.tma_load]} : tensor<128x256x!tt.ptr<f16>, #blocked>
      %41 = ttg.local_alloc %40 {groups = [@nvws.group.tma_load]} : (tensor<128x256xf16, #blocked>) -> !ttg.memdesc<128x256xf16, #shared, #smem>
      %42 = ttng.tc_gen5_mma %39, %41, %result[%arg16], %arg15, %true {groups = [@nvws.group.mma]} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x256xf16, #shared, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %43 = arith.muli %arg10, %c128_i32 {groups = [@nvws.group.tma_load]} : i32
      %44 = tt.splat %43 {groups = [@nvws.group.tma_load]} : i32 -> tensor<128x256xi32, #blocked>
      %45 = tt.addptr %arg14, %44 {groups = [@nvws.group.tma_load]} : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
      %46 = arith.addi %arg13, %c128_i32 {groups = [@nvws.group.tma_load]} : i32
      scf.yield %46, %45, %true, %42 : i32, tensor<128x256x!tt.ptr<f16>, #blocked>, i1, !ttg.async.token
    } {groups = [@nvws.group.mma, @nvws.group.tma_load], groups.0 = [@nvws.group.tma_load], groups.1 = [@nvws.group.tma_load], groups.2 = [@nvws.group.mma], groups.3 = [@nvws.group.mma]}
    %result_0, %token_1 = ttng.tmem_load %result[%24#3] {groups = [@nvws.group.epilogue]} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked2>
    %25 = tt.splat %5 {groups = [@nvws.group.epilogue]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %26 = arith.addi %25, %12 {groups = [@nvws.group.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %27 = tt.expand_dims %26 {axis = 1 : i32, groups = [@nvws.group.epilogue]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<128x1xi32, #blocked>
    %28 = tt.splat %arg11 {groups = [@nvws.group.epilogue]} : i32 -> tensor<128x1xi32, #blocked>
    %29 = arith.muli %28, %27 {groups = [@nvws.group.epilogue]} : tensor<128x1xi32, #blocked>
    %30 = tt.splat %arg6 {groups = [@nvws.group.epilogue]} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked>
    %31 = tt.addptr %30, %29 {groups = [@nvws.group.epilogue]} : tensor<128x1x!tt.ptr<f16>, #blocked>, tensor<128x1xi32, #blocked>
    %32 = tt.expand_dims %9 {axis = 0 : i32, groups = [@nvws.group.epilogue]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %33 = tt.broadcast %31 {groups = [@nvws.group.epilogue]} : tensor<128x1x!tt.ptr<f16>, #blocked> -> tensor<128x256x!tt.ptr<f16>, #blocked>
    %34 = tt.broadcast %32 {groups = [@nvws.group.epilogue]} : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %35 = tt.addptr %33, %34 {groups = [@nvws.group.epilogue]} : tensor<128x256x!tt.ptr<f16>, #blocked>, tensor<128x256xi32, #blocked>
    %36 = arith.truncf %result_0 {groups = [@nvws.group.epilogue]} : tensor<128x256xf32, #blocked2> to tensor<128x256xf16, #blocked2>
    %37 = ttg.convert_layout %36 {groups = [@nvws.group.epilogue]} : tensor<128x256xf16, #blocked2> -> tensor<128x256xf16, #blocked>
    tt.store %35, %37 {groups = [@nvws.group.epilogue]} : tensor<128x256x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

