#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 2], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 128]], warp = [[32, 0], [64, 0], [16, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"nvws.warp-specialized" = true, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttg.use-ttg-ws" = true} {
  tt.func public @fmha_fp8(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg3: !tt.tensordesc<tensor<128x128xf8E5M2, #shared>>, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: !tt.tensordesc<tensor<256x128xf8E5M2, #shared>>, %arg9: i32, %arg10: i32, %arg11: i64, %arg12: i64, %arg13: !tt.tensordesc<tensor<128x256xf8E5M2, #shared>>, %arg14: i32, %arg15: i32, %arg16: i64, %arg17: i64, %arg18: !tt.tensordesc<tensor<128x128xf8E5M2, #shared>>, %arg19: i32, %arg20: i32, %arg21: i64, %arg22: i64, %arg23: !tt.tensordesc<tensor<128xf32, #shared1>>, %arg24: i32, %arg25: i64, %arg26: f32, %arg27: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg28: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg29: i32 {tt.divisibility = 16 : i32}, %arg30: i32 {tt.divisibility = 16 : i32}, %arg31: i32 {tt.divisibility = 16 : i32}, %arg32: i32 {tt.divisibility = 16 : i32}, %arg33: i32 {tt.divisibility = 16 : i32}, %arg34: i32 {tt.divisibility = 16 : i32}, %arg35: i32, %arg36: i32, %arg37: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant {ttg.partitions = [0 : i32]} false
    %true = arith.constant {ttg.partitions = [0 : i32, 1 : i32, 3 : i32]} true
    %c128_i32 = arith.constant {ttg.partitions = [2 : i32, 4 : i32]} 128 : i32
    %c0_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32]} 0 : i32
    %cst = arith.constant {ttg.partitions = [4 : i32]} 1.44269502 : f32
    %c256_i32 = arith.constant {ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32]} 256 : i32
    %cst_0 = arith.constant {ttg.partitions = [4 : i32]} dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_1 = arith.constant {ttg.partitions = [4 : i32]} dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cst_2 = arith.constant {ttg.partitions = [1 : i32, 3 : i32]} dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x {ttg.partitions = [2 : i32, 4 : i32]} : i32
    %1 = tt.get_program_id y {ttg.partitions = [2 : i32, 4 : i32]} : i32
    %2 = arith.muli %0, %c128_i32 {ttg.partitions = [2 : i32, 4 : i32]} : i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32, ttg.partitions = [4 : i32]} : tensor<128xi32, #blocked1>
    %4 = tt.splat %2 {ttg.partitions = [4 : i32]} : i32 -> tensor<128xi32, #blocked1>
    %5 = arith.addi %4, %3 {ttg.partitions = [4 : i32]} : tensor<128xi32, #blocked1>
    %6 = arith.mulf %arg26, %cst {ttg.partitions = [4 : i32]} : f32
    %7 = arith.muli %1, %arg37 {ttg.partitions = [2 : i32, 4 : i32]} : i32
    %8 = arith.addi %7, %2 {ttg.partitions = [2 : i32, 4 : i32]} : i32
    %9 = tt.descriptor_load %arg3[%8, %c0_i32] {ttg.partitions = [2 : i32]} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>> -> tensor<128x128xf8E5M2, #blocked2>
    %10 = ttg.local_alloc %9 {ttg.partitions = [2 : i32]} : (tensor<128x128xf8E5M2, #blocked2>) -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem>
    %result, %token = ttng.tmem_alloc {ttg.partitions = [0 : i32, 4 : i32]} : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_3, %token_4 = ttng.tmem_alloc {ttg.partitions = [1 : i32, 3 : i32, 4 : i32]} : () -> (!ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %11 = ttng.tmem_store %cst_2, %result_3[%token_4], %true {ttg.partitions = [1 : i32]} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
    %12:5 = scf.for %arg38 = %c0_i32 to %arg37 step %c256_i32 iter_args(%arg39 = %cst_1, %arg40 = %cst_0, %arg41 = %7, %arg42 = %token, %arg43 = %11) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, i32, !ttg.async.token, !ttg.async.token)  : i32 {
      %25 = tt.descriptor_load %arg8[%arg41, %c0_i32] {ttg.partitions = [2 : i32]} : !tt.tensordesc<tensor<256x128xf8E5M2, #shared>> -> tensor<256x128xf8E5M2, #blocked2>
      %26 = ttg.local_alloc %25 {ttg.partitions = [2 : i32]} : (tensor<256x128xf8E5M2, #blocked2>) -> !ttg.memdesc<256x128xf8E5M2, #shared, #smem>
      %27 = ttg.memdesc_trans %26 {order = array<i32: 1, 0>, ttg.partitions = [0 : i32]} : !ttg.memdesc<256x128xf8E5M2, #shared, #smem> -> !ttg.memdesc<128x256xf8E5M2, #shared2, #smem>
      %28 = ttng.tc_gen5_mma %10, %27, %result[%arg42], %false, %true {ttg.partitions = [0 : i32]} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem>, !ttg.memdesc<128x256xf8E5M2, #shared2, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_7, %token_8 = ttng.tmem_load %result[%28] {ttg.partitions = [4 : i32]} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #linear>
      %29 = "tt.reduce"(%result_7) <{axis = 1 : i32}> ({
      ^bb0(%arg44: f32, %arg45: f32):
        %56 = arith.maxnumf %arg44, %arg45 {ttg.partitions = [4 : i32]} : f32
        tt.reduce.return %56 {ttg.partitions = [4 : i32]} : f32
      }) {ttg.partitions = [4 : i32]} : (tensor<128x256xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %30 = tt.splat %6 {ttg.partitions = [4 : i32]} : f32 -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %31 = arith.mulf %29, %30 {ttg.partitions = [4 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %32 = arith.maxnumf %arg39, %31 {ttg.partitions = [4 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %33 = tt.splat %6 {ttg.partitions = [4 : i32]} : f32 -> tensor<128x256xf32, #linear>
      %34 = arith.mulf %result_7, %33 {ttg.partitions = [4 : i32]} : tensor<128x256xf32, #linear>
      %35 = tt.expand_dims %32 {axis = 1 : i32, ttg.partitions = [4 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %36 = tt.broadcast %35 {ttg.partitions = [4 : i32]} : tensor<128x1xf32, #linear> -> tensor<128x256xf32, #linear>
      %37 = arith.subf %34, %36 {ttg.partitions = [4 : i32]} : tensor<128x256xf32, #linear>
      %38 = arith.subf %arg39, %32 {ttg.partitions = [4 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %39 = math.exp2 %38 {ttg.partitions = [4 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %40 = math.exp2 %37 {ttg.partitions = [4 : i32]} : tensor<128x256xf32, #linear>
      %41 = "tt.reduce"(%40) <{axis = 1 : i32}> ({
      ^bb0(%arg44: f32, %arg45: f32):
        %56 = arith.addf %arg44, %arg45 {ttg.partitions = [4 : i32]} : f32
        tt.reduce.return %56 {ttg.partitions = [4 : i32]} : f32
      }) {ttg.partitions = [4 : i32]} : (tensor<128x256xf32, #linear>) -> tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %42 = tt.fp_to_fp %40 {ttg.partitions = [4 : i32]}, rounding = rtne : tensor<128x256xf32, #linear> -> tensor<128x256xf8E5M2, #linear>
      %result_9 = ttng.tmem_alloc %42 {ttg.partitions = [4 : i32]} : (tensor<128x256xf8E5M2, #linear>) -> !ttg.memdesc<128x256xf8E5M2, #tmem, #ttng.tensor_memory>
      %43 = arith.mulf %arg40, %39 {ttg.partitions = [4 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %44 = arith.addf %43, %41 {ttg.partitions = [4 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
      %45 = tt.expand_dims %39 {axis = 1 : i32, ttg.partitions = [1 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
      %46 = ttg.convert_layout %45 {ttg.partitions = [1 : i32]} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #blocked>
      %47 = tt.broadcast %46 {ttg.partitions = [1 : i32]} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
      %result_10, %token_11 = ttng.tmem_load %result_3[%arg43] {ttg.partitions = [1 : i32]} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %48 = arith.mulf %result_10, %47 {ttg.partitions = [1 : i32]} : tensor<128x128xf32, #blocked>
      %49 = arith.muli %1, %c128_i32 {ttg.partitions = [2 : i32]} : i32
      %50 = tt.descriptor_load %arg13[%49, %arg38] {ttg.partitions = [2 : i32]} : !tt.tensordesc<tensor<128x256xf8E5M2, #shared>> -> tensor<128x256xf8E5M2, #blocked3>
      %51 = ttg.local_alloc %50 {ttg.partitions = [2 : i32]} : (tensor<128x256xf8E5M2, #blocked3>) -> !ttg.memdesc<128x256xf8E5M2, #shared, #smem>
      %52 = ttg.memdesc_trans %51 {order = array<i32: 1, 0>, ttg.partitions = [3 : i32]} : !ttg.memdesc<128x256xf8E5M2, #shared, #smem> -> !ttg.memdesc<256x128xf8E5M2, #shared2, #smem>
      %53 = ttng.tmem_store %48, %result_3[%token_11], %true {ttg.partitions = [1 : i32]} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %54 = ttng.tc_gen5_mma %result_9, %52, %result_3[%53], %true, %true {ttg.partitions = [3 : i32]} : !ttg.memdesc<128x256xf8E5M2, #tmem, #ttng.tensor_memory>, !ttg.memdesc<256x128xf8E5M2, #shared2, #smem>, !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable>
      %55 = arith.addi %arg41, %c256_i32 {ttg.partitions = [2 : i32]} : i32
      scf.yield %32, %44, %55, %token_8, %54 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>, i32, !ttg.async.token, !ttg.async.token
    } {tt.divisibility_arg1 = dense<256> : tensor<1xi32>, ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32, 4 : i32], ttg.partitions.0 = [4 : i32], ttg.partitions.1 = [4 : i32], ttg.partitions.2 = [2 : i32], ttg.partitions.3 = [4 : i32], ttg.partitions.4 = [3 : i32]}
    %result_5, %token_6 = ttng.tmem_load %result_3[%12#4] {ttg.partitions = [4 : i32]} : !ttg.memdesc<128x128xf32, #tmem1, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    %13 = math.log2 %12#1 {ttg.partitions = [4 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %14 = arith.addf %12#0, %13 {ttg.partitions = [4 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>>
    %15 = tt.expand_dims %12#1 {axis = 1 : i32, ttg.partitions = [4 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xf32, #linear>
    %16 = ttg.convert_layout %15 {ttg.partitions = [4 : i32]} : tensor<128x1xf32, #linear> -> tensor<128x1xf32, #blocked>
    %17 = tt.broadcast %16 {ttg.partitions = [4 : i32]} : tensor<128x1xf32, #blocked> -> tensor<128x128xf32, #blocked>
    %18 = arith.divf %result_5, %17 {ttg.partitions = [4 : i32]} : tensor<128x128xf32, #blocked>
    %19 = tt.addptr %arg27, %7 {ttg.partitions = [4 : i32]} : !tt.ptr<f32>, i32
    %20 = tt.splat %19 {ttg.partitions = [4 : i32]} : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked1>
    %21 = tt.addptr %20, %5 {ttg.partitions = [4 : i32]} : tensor<128x!tt.ptr<f32>, #blocked1>, tensor<128xi32, #blocked1>
    %22 = ttg.convert_layout %14 {ttg.partitions = [4 : i32]} : tensor<128xf32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128xf32, #blocked1>
    tt.store %21, %22 {ttg.partitions = [4 : i32]} : tensor<128x!tt.ptr<f32>, #blocked1>
    %23 = tt.fp_to_fp %18 {ttg.partitions = [4 : i32]}, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E5M2, #blocked>
    %24 = ttg.convert_layout %23 {ttg.partitions = [4 : i32]} : tensor<128x128xf8E5M2, #blocked> -> tensor<128x128xf8E5M2, #blocked2>
    tt.descriptor_store %arg18[%8, %c0_i32], %24 {ttg.partitions = [4 : i32]} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>>, tensor<128x128xf8E5M2, #blocked2>
    tt.return
  }
}

