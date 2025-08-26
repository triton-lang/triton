#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
module attributes {nvws.group.mma0 = {num_warps = 1 : i32, reg_count = 24 : i32, start_warp = 12 : i32}, nvws.group.mma1 = {num_warps = 1 : i32, reg_count = 24 : i32, start_warp = 13 : i32}, nvws.group.simt0 = {num_warps = 4 : i32, reg_count = 160 : i32, start_warp = 0 : i32}, nvws.group.simt1 = {num_warps = 4 : i32, reg_count = 160 : i32, start_warp = 4 : i32}, nvws.group.simt2 = {num_warps = 4 : i32, reg_count = 160 : i32, start_warp = 8 : i32}, nvws.group.tma_load = {num_warps = 1 : i32, reg_count = 24 : i32, start_warp = 14 : i32}, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg2: !tt.tensordesc<tensor<64x64xf16, #shared>>, %arg3: f32, %arg4: i32) {
    %true = arith.constant {groups = [@nvws.group.mma0, @nvws.group.mma1, @nvws.group.simt1, @nvws.group.simt2]} true
    %false = arith.constant {groups = [@nvws.group.mma0]} false
    %c0_i32 = arith.constant {groups = [@nvws.group.mma0, @nvws.group.mma1, @nvws.group.simt0, @nvws.group.simt1, @nvws.group.simt2, @nvws.group.tma_load]} 0 : i32
    %c64_i32 = arith.constant {groups = [@nvws.group.mma0, @nvws.group.mma1, @nvws.group.simt0, @nvws.group.simt1, @nvws.group.simt2, @nvws.group.tma_load]} 64 : i32
    %cst = arith.constant {groups = [@nvws.group.simt0]} dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant {groups = [@nvws.group.mma1, @nvws.group.simt1]} dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant {groups = [@nvws.group.simt1, @nvws.group.simt2]} dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %result, %token = ttng.tmem_alloc {groups = [@nvws.group.mma0, @nvws.group.simt0]} : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_2, %token_3 = ttng.tmem_alloc {groups = [@nvws.group.mma1, @nvws.group.simt1, @nvws.group.simt2]} : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst_0, %result_2[%token_3], %true {groups = [@nvws.group.simt1]} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %1:5 = scf.for %arg5 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg6 = %cst_1, %arg7 = %cst, %arg8 = %cst_1, %arg9 = %token, %arg10 = %0) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      %2 = tt.descriptor_load %arg1[%arg5, %c0_i32] {groups = [@nvws.group.tma_load]} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %3 = ttg.local_alloc %2 {groups = [@nvws.group.tma_load]} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %4 = ttg.memdesc_trans %3 {groups = [@nvws.group.mma0], order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      %5 = ttng.tc_gen5_mma %arg0, %4, %result[%arg9], %false, %true {groups = [@nvws.group.mma0]} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %result_6, %token_7 = ttng.tmem_load %result[%5] {groups = [@nvws.group.simt0]} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      %6 = "compute_row_max"(%result_6, %arg3) {groups = [@nvws.group.simt0]} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %7 = "sub_row_max"(%result_6, %6, %arg3) {groups = [@nvws.group.simt0]} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %8 = math.exp2 %7 {groups = [@nvws.group.simt0]} : tensor<256x64xf32, #blocked>
      %9 = arith.subf %arg7, %6 {groups = [@nvws.group.simt0]} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %10 = math.exp2 %9 {groups = [@nvws.group.simt0]} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %11 = "tt.reduce"(%8) <{axis = 1 : i32}> ({
      ^bb0(%arg11: f32, %arg12: f32):
        %26 = arith.addf %arg11, %arg12 {groups = [@nvws.group.simt1]} : f32
        tt.reduce.return %26 {groups = [@nvws.group.simt1]} : f32
      }) {groups = [@nvws.group.simt1]} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %12 = arith.mulf %arg6, %10 {groups = [@nvws.group.simt1]} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %13 = arith.addf %12, %11 {groups = [@nvws.group.simt1]} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = tt.expand_dims %10 {axis = 1 : i32, groups = [@nvws.group.simt2]} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %15 = tt.broadcast %14 {groups = [@nvws.group.simt2]} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>
      %result_8, %token_9 = ttng.tmem_load %result_2[%arg10] {groups = [@nvws.group.simt2]} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
      %16 = arith.mulf %result_8, %15 {groups = [@nvws.group.simt2]} : tensor<256x64xf32, #blocked>
      %17 = arith.addf %8, %8 {groups = [@nvws.group.simt2]} : tensor<256x64xf32, #blocked>
      %18 = arith.addf %result_8, %17 {groups = [@nvws.group.simt2]} : tensor<256x64xf32, #blocked>
      %19 = "sum"(%18) {groups = [@nvws.group.simt2]} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %20 = arith.addf %arg8, %19 {groups = [@nvws.group.simt2]} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %21 = tt.descriptor_load %arg2[%arg5, %c0_i32] {groups = [@nvws.group.tma_load]} : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #blocked1>
      %22 = ttg.local_alloc %21 {groups = [@nvws.group.tma_load]} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %23 = arith.truncf %8 {groups = [@nvws.group.simt0]} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      %result_10 = ttng.tmem_alloc %23 {groups = [@nvws.group.simt0]} : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>
      %24 = ttng.tmem_store %16, %result_2[%token_9], %true {groups = [@nvws.group.simt2]} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      %25 = ttng.tc_gen5_mma %result_10, %22, %result_2[%24], %true, %true {groups = [@nvws.group.mma1]} : !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %13, %6, %20, %token_7, %25 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {groups = [@nvws.group.mma0, @nvws.group.mma1, @nvws.group.simt0, @nvws.group.simt1, @nvws.group.simt2, @nvws.group.tma_load], groups.0 = [@nvws.group.simt1], groups.1 = [@nvws.group.simt0], groups.2 = [@nvws.group.simt2], groups.3 = [@nvws.group.simt0], groups.4 = [@nvws.group.mma1], tt.warp_specialize}
    %result_4, %token_5 = ttng.tmem_load %result_2[%1#4] {groups = [@nvws.group.simt1]} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    "use"(%1#0, %result_4, %1#1) {groups = [@nvws.group.simt1]} : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }
}

