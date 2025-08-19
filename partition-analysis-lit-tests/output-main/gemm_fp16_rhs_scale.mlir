#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"nvws.warp-specialized" = true, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_scale_rhs_kernel(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg11: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}, %arg15: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant {ttg.partitions = [1 : i32]} false
    %true = arith.constant {ttg.partitions = [1 : i32]} true
    %c128_i32 = arith.constant {ttg.partitions = [0 : i32, 2 : i32]} 128 : i32
    %c0_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32]} 0 : i32
    %c64_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32]} 64 : i32
    %cst = arith.constant {ttg.partitions = [1 : i32]} dense<64> : tensor<1x64xi32, #blocked>
    %c1_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32]} 1 : i32
    %c127_i32 = arith.constant {ttg.partitions = [0 : i32, 2 : i32]} 127 : i32
    %c63_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32]} 63 : i32
    %0 = tt.get_program_id x {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %1 = arith.addi %arg12, %c127_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %2 = arith.divsi %1, %c128_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %3 = arith.remsi %0, %2 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %4 = arith.divsi %0, %2 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %5 = arith.muli %3, %c128_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %6 = arith.muli %4, %c128_i32 {ttg.partitions = [0 : i32, 2 : i32]} : i32
    %7 = tt.make_range {end = 64 : i32, start = 0 : i32, ttg.partitions = [1 : i32]} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %8 = tt.expand_dims %7 {axis = 0 : i32, ttg.partitions = [1 : i32]} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x64xi32, #blocked>
    %9 = tt.splat %arg10 {ttg.partitions = [1 : i32]} : !tt.ptr<f16> -> tensor<1x64x!tt.ptr<f16>, #blocked>
    %10 = tt.addptr %9, %8 {ttg.partitions = [1 : i32]} : tensor<1x64x!tt.ptr<f16>, #blocked>, tensor<1x64xi32, #blocked>
    %11 = arith.addi %arg14, %c63_i32 {ttg.partitions = [0 : i32, 1 : i32]} : i32
    %12 = arith.divsi %11, %c64_i32 {ttg.partitions = [0 : i32, 1 : i32]} : i32
    %result, %token = ttng.tmem_alloc {ttg.partitions = [1 : i32, 2 : i32]} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %13:4 = scf.for %arg16 = %c0_i32 to %12 step %c1_i32 iter_args(%arg17 = %c0_i32, %arg18 = %10, %arg19 = %false, %arg20 = %token) -> (i32, tensor<1x64x!tt.ptr<f16>, #blocked>, i1, !ttg.async.token)  : i32 {
      %31 = tt.descriptor_load %arg0[%5, %arg17] {ttg.partitions = [0 : i32]} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
      %32 = ttg.local_alloc %31 {ttg.partitions = [0 : i32]} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %33 = tt.descriptor_load %arg5[%6, %arg17] {ttg.partitions = [0 : i32]} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked>
      %34 = tt.load %arg18 {ttg.partitions = [1 : i32]} : tensor<1x64x!tt.ptr<f16>, #blocked>
      %35 = tt.broadcast %34 {ttg.partitions = [1 : i32]} : tensor<1x64xf16, #blocked> -> tensor<128x64xf16, #blocked>
      %36 = arith.mulf %33, %35 {ttg.partitions = [1 : i32]} : tensor<128x64xf16, #blocked>
      %37 = ttg.local_alloc %36 {ttg.partitions = [1 : i32]} : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %38 = ttg.memdesc_trans %37 {order = array<i32: 1, 0>, ttg.partitions = [1 : i32]} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %39 = ttng.tc_gen5_mma %32, %38, %result[%arg20], %arg19, %true {ttg.partitions = [1 : i32]} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %40 = arith.addi %arg17, %c64_i32 {ttg.partitions = [0 : i32]} : i32
      %41 = tt.addptr %arg18, %cst {ttg.partitions = [1 : i32]} : tensor<1x64x!tt.ptr<f16>, #blocked>, tensor<1x64xi32, #blocked>
      scf.yield %40, %41, %true, %39 : i32, tensor<1x64x!tt.ptr<f16>, #blocked>, i1, !ttg.async.token
    } {ttg.partitions = [0 : i32, 1 : i32], ttg.partitions.0 = [0 : i32], ttg.partitions.1 = [1 : i32], ttg.partitions.2 = [1 : i32], ttg.partitions.3 = [1 : i32]}
    %result_0, %token_1 = ttng.tmem_load %result[%13#3] {ttg.partitions = [2 : i32]} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked1>
    %14 = arith.truncf %result_0 {ttg.partitions = [2 : i32]} : tensor<128x128xf32, #blocked1> to tensor<128x128xf16, #blocked1>
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32, ttg.partitions = [2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %16 = tt.make_range {end = 128 : i32, start = 0 : i32, ttg.partitions = [2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %17 = tt.splat %5 {ttg.partitions = [2 : i32]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %18 = arith.addi %17, %15 {ttg.partitions = [2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>>
    %19 = tt.splat %6 {ttg.partitions = [2 : i32]} : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %20 = arith.addi %19, %16 {ttg.partitions = [2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>>
    %21 = tt.expand_dims %18 {axis = 1 : i32, ttg.partitions = [2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2>
    %22 = tt.splat %arg15 {ttg.partitions = [2 : i32]} : i32 -> tensor<128x1xi32, #blocked2>
    %23 = arith.muli %22, %21 {ttg.partitions = [2 : i32]} : tensor<128x1xi32, #blocked2>
    %24 = tt.splat %arg11 {ttg.partitions = [2 : i32]} : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2>
    %25 = tt.addptr %24, %23 {ttg.partitions = [2 : i32]} : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2>
    %26 = tt.expand_dims %20 {axis = 0 : i32, ttg.partitions = [2 : i32]} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x128xi32, #blocked2>
    %27 = tt.broadcast %25 {ttg.partitions = [2 : i32]} : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x128x!tt.ptr<f16>, #blocked2>
    %28 = tt.broadcast %26 {ttg.partitions = [2 : i32]} : tensor<1x128xi32, #blocked2> -> tensor<128x128xi32, #blocked2>
    %29 = tt.addptr %27, %28 {ttg.partitions = [2 : i32]} : tensor<128x128x!tt.ptr<f16>, #blocked2>, tensor<128x128xi32, #blocked2>
    %30 = ttg.convert_layout %14 {ttg.partitions = [2 : i32]} : tensor<128x128xf16, #blocked1> -> tensor<128x128xf16, #blocked2>
    tt.store %29, %30 {ttg.partitions = [2 : i32]} : tensor<128x128x!tt.ptr<f16>, #blocked2>
    tt.return
  }
}

