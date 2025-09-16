#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"nvws.warp-specialized" = true, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm_fp16(%arg0: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<256x128xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant {ttg.partitions = [1 : i32]} false
    %true = arith.constant {ttg.partitions = [1 : i32]} true
    %c128_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32, 2 : i32]} 128 : i32
    %c256_i32 = arith.constant {ttg.partitions = [0 : i32, 2 : i32]} 256 : i32
    %c0_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32]} 0 : i32
    %c1_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32]} 1 : i32
    %c127_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32, 2 : i32]} 127 : i32
    %0 = tt.get_program_id x {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %1 = arith.addi %arg11, %c127_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %2 = arith.divsi %1, %c128_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %3 = arith.remsi %0, %2 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %4 = arith.divsi %0, %2 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %5 = arith.muli %3, %c128_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %6 = arith.muli %4, %c256_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %7 = arith.addi %arg13, %c127_i32 {ttg.partitions = [0 : i32, 1 : i32]} : i32
    %8 = arith.divsi %7, %c128_i32 {ttg.partitions = [0 : i32, 1 : i32]} : i32
    %result, %token = ttng.tmem_alloc {ttg.partitions = [1 : i32, 2 : i32]} : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %9:3 = scf.for %arg15 = %c0_i32 to %8 step %c1_i32 iter_args(%arg16 = %c0_i32, %arg17 = %false, %arg18 = %token) -> (i32, i1, !ttg.async.token)  : i32 {
      %27 = tt.descriptor_load %arg0[%5, %arg16] {ttg.partitions = [0 : i32]} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked>
      %28 = ttg.local_alloc %27 {ttg.partitions = [0 : i32]} : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %29 = tt.descriptor_load %arg5[%6, %arg16] {ttg.partitions = [0 : i32]} : !tt.tensordesc<tensor<256x128xf16, #shared>> -> tensor<256x128xf16, #blocked>
      %30 = ttg.local_alloc %29 {ttg.partitions = [0 : i32]} : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared, #smem>
      %31 = ttg.memdesc_trans %30 {order = array<i32: 1, 0>, ttg.partitions = [1 : i32]} : !ttg.memdesc<256x128xf16, #shared, #smem> -> !ttg.memdesc<128x256xf16, #shared1, #smem>
      %32 = ttng.tc_gen5_mma %28, %31, %result[%arg18], %arg17, %true {ttg.partitions = [1 : i32]} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x256xf16, #shared1, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %33 = arith.addi %arg16, %c128_i32 {ttg.partitions = [0 : i32]} : i32
      scf.yield %33, %true, %32 : i32, i1, !ttg.async.token
    } {ttg.partitions = [0 : i32, 1 : i32], ttg.partitions.0 = [0 : i32], ttg.partitions.1 = [1 : i32], ttg.partitions.2 = [1 : i32]}
    %result_0, %token_1 = ttng.tmem_load %result[%9#2] {ttg.partitions = [2 : i32]} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked1>
    %10 = arith.truncf %result_0 {ttg.partitions = [2 : i32]} : tensor<128x256xf32, #blocked1> to tensor<128x256xf16, #blocked1>
    %11 = tt.make_range {end = 128 : i32, start = 0 : i32, ttg.partitions = [2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %12 = tt.splat %5 {ttg.partitions = [2 : i32]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %13 = arith.addi %12, %11 {ttg.partitions = [2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %14 = tt.make_range {end = 256 : i32, start = 0 : i32, ttg.partitions = [2 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %15 = tt.splat %6 {ttg.partitions = [2 : i32]} : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %16 = arith.addi %15, %14 {ttg.partitions = [2 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %17 = tt.expand_dims %13 {axis = 1 : i32, ttg.partitions = [2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %18 = tt.splat %arg14 {ttg.partitions = [2 : i32]} : i32 -> tensor<128x1xi32, #blocked2>
    %19 = arith.muli %18, %17 {ttg.partitions = [2 : i32]} : tensor<128x1xi32, #blocked2>
    %20 = tt.splat %arg10 {ttg.partitions = [2 : i32]} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
    %21 = tt.addptr %20, %19 {ttg.partitions = [2 : i32]} : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
    %22 = tt.expand_dims %16 {axis = 0 : i32, ttg.partitions = [2 : i32]} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2>
    %23 = tt.broadcast %21 {ttg.partitions = [2 : i32]} : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2>
    %24 = tt.broadcast %22 {ttg.partitions = [2 : i32]} : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2>
    %25 = tt.addptr %23, %24 {ttg.partitions = [2 : i32]} : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2>
    %26 = ttg.convert_layout %10 {ttg.partitions = [2 : i32]} : tensor<128x256xf16, #blocked1> -> tensor<128x256xf16, #blocked2>
    tt.store %25, %26 {ttg.partitions = [2 : i32]} : tensor<128x256x!tt.ptr<f16>, #blocked2>
    tt.return
  }
}

