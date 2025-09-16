#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {nvws.group.epilogue = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.group.mma = {num_warps = 1 : i32, start_warp = 4 : i32}, nvws.group.tma_load = {num_warps = 1 : i32, start_warp = 5 : i32}, "nvws.warp-specialized" = true, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm_persistent_flattened_fp16(%arg0: !tt.tensordesc<tensor<256x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<256x128xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<256x256xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant {groups = [@nvws.group.mma]} false
    %true = arith.constant {groups = [@nvws.group.mma]} true
    %c84_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} 84 : i32
    %c1_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} 1 : i32
    %c-1_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} -1 : i32
    %c0_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} 0 : i32
    %c8_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} 8 : i32
    %c256_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} 256 : i32
    %c128_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} 128 : i32
    %c255_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} 255 : i32
    %c127_i32 = arith.constant {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} 127 : i32
    %0 = tt.get_program_id x {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %1 = arith.addi %arg15, %c255_i32 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %2 = arith.divsi %1, %c256_i32 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %3 = arith.addi %arg16, %c255_i32 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %4 = arith.divsi %3, %c256_i32 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %5 = arith.addi %arg17, %c127_i32 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %6 = arith.divsi %5, %c128_i32 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %7 = arith.muli %2, %4 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %8 = arith.divsi %7, %c84_i32 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %9 = arith.remsi %7, %c84_i32 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %10 = arith.cmpi slt, %0, %9 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %11 = scf.if %10 -> (i32) {
      %16 = arith.addi %8, %c1_i32 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
      scf.yield %16 : i32
    } else {
      scf.yield %8 : i32
    } {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load], groups.0 = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]}
    %12 = arith.subi %0, %c84_i32 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
    %13 = arith.muli %4, %c8_i32 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
    %14 = arith.muli %6, %11 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
    %result, %token = ttng.tmem_alloc {groups = [@nvws.group.epilogue, @nvws.group.mma]} : () -> (!ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %15:7 = scf.for %arg18 = %c0_i32 to %14 step %c1_i32 iter_args(%arg19 = %c-1_i32, %arg20 = %12, %arg21 = %c0_i32, %arg22 = %c0_i32, %arg23 = %12, %arg24 = %false, %arg25 = %token) -> (i32, i32, i32, i32, i32, i1, !ttg.async.token)  : i32 {
      %16 = arith.subi %6, %c1_i32 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
      %17 = arith.cmpi eq, %arg19, %16 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
      %18 = arith.addi %arg19, %c1_i32 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
      %19 = arith.select %17, %c0_i32, %18 {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load]} : i32
      %20 = arith.cmpi eq, %19, %c0_i32 {groups = [@nvws.group.tma_load]} : i32
      %21:3 = scf.if %20 -> (i32, i32, i32) {
        %32 = arith.addi %arg20, %c84_i32 {groups = [@nvws.group.tma_load]} : i32
        %33 = arith.divsi %32, %13 {groups = [@nvws.group.tma_load]} : i32
        %34 = arith.muli %33, %c8_i32 {groups = [@nvws.group.tma_load]} : i32
        %35 = arith.subi %2, %34 {groups = [@nvws.group.tma_load]} : i32
        %36 = arith.minsi %35, %c8_i32 {groups = [@nvws.group.tma_load]} : i32
        %37 = arith.remsi %32, %36 {groups = [@nvws.group.tma_load]} : i32
        %38 = arith.addi %34, %37 {groups = [@nvws.group.tma_load]} : i32
        %39 = arith.remsi %32, %13 {groups = [@nvws.group.tma_load]} : i32
        %40 = arith.divsi %39, %36 {groups = [@nvws.group.tma_load]} : i32
        %41 = arith.muli %38, %c256_i32 {groups = [@nvws.group.tma_load]} : i32
        %42 = arith.muli %40, %c256_i32 {groups = [@nvws.group.tma_load]} : i32
        scf.yield %32, %41, %42 : i32, i32, i32
      } else {
        scf.yield %arg20, %arg21, %arg22 : i32, i32, i32
      } {groups = [@nvws.group.tma_load], groups.0 = [@nvws.group.tma_load], groups.1 = [@nvws.group.tma_load], groups.2 = [@nvws.group.tma_load]}
      %22 = arith.muli %19, %c128_i32 {groups = [@nvws.group.tma_load]} : i32
      %23 = tt.descriptor_load %arg0[%21#1, %22] {groups = [@nvws.group.tma_load]} : !tt.tensordesc<tensor<256x128xf16, #shared>> -> tensor<256x128xf16, #blocked>
      %24 = ttg.local_alloc %23 {groups = [@nvws.group.tma_load]} : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared, #smem>
      %25 = tt.descriptor_load %arg5[%21#2, %22] {groups = [@nvws.group.tma_load]} : !tt.tensordesc<tensor<256x128xf16, #shared>> -> tensor<256x128xf16, #blocked>
      %26 = ttg.local_alloc %25 {groups = [@nvws.group.tma_load]} : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared, #smem>
      %27 = ttg.memdesc_trans %26 {groups = [@nvws.group.mma], order = array<i32: 1, 0>} : !ttg.memdesc<256x128xf16, #shared, #smem> -> !ttg.memdesc<128x256xf16, #shared1, #smem>
      %28 = ttng.tc_gen5_mma %24, %27, %result[%arg25], %arg24, %true {groups = [@nvws.group.mma]} : !ttg.memdesc<256x128xf16, #shared, #smem>, !ttg.memdesc<128x256xf16, #shared1, #smem>, !ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %29 = arith.cmpi eq, %19, %16 {groups = [@nvws.group.epilogue, @nvws.group.mma]} : i32
      %30 = arith.cmpi ne, %19, %16 {groups = [@nvws.group.mma]} : i32
      %31:2 = scf.if %29 -> (i32, !ttg.async.token) {
        %32 = arith.addi %arg23, %c84_i32 {groups = [@nvws.group.epilogue]} : i32
        %33 = arith.divsi %32, %13 {groups = [@nvws.group.epilogue]} : i32
        %34 = arith.muli %33, %c8_i32 {groups = [@nvws.group.epilogue]} : i32
        %35 = arith.subi %2, %34 {groups = [@nvws.group.epilogue]} : i32
        %36 = arith.minsi %35, %c8_i32 {groups = [@nvws.group.epilogue]} : i32
        %37 = arith.remsi %32, %36 {groups = [@nvws.group.epilogue]} : i32
        %38 = arith.addi %34, %37 {groups = [@nvws.group.epilogue]} : i32
        %39 = arith.remsi %32, %13 {groups = [@nvws.group.epilogue]} : i32
        %40 = arith.divsi %39, %36 {groups = [@nvws.group.epilogue]} : i32
        %41 = arith.muli %38, %c256_i32 {groups = [@nvws.group.epilogue]} : i32
        %42 = arith.muli %40, %c256_i32 {groups = [@nvws.group.epilogue]} : i32
        %result_0, %token_1 = ttng.tmem_load %result[%28] {groups = [@nvws.group.epilogue]} : !ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x256xf32, #blocked1>
        %43 = arith.truncf %result_0 {groups = [@nvws.group.epilogue]} : tensor<256x256xf32, #blocked1> to tensor<256x256xf16, #blocked1>
        %44 = ttg.convert_layout %43 {groups = [@nvws.group.epilogue]} : tensor<256x256xf16, #blocked1> -> tensor<256x256xf16, #blocked>
        tt.descriptor_store %arg10[%41, %42], %44 {groups = [@nvws.group.epilogue]} : !tt.tensordesc<tensor<256x256xf16, #shared>>, tensor<256x256xf16, #blocked>
        scf.yield %32, %token_1 : i32, !ttg.async.token
      } else {
        scf.yield %arg23, %28 : i32, !ttg.async.token
      } {groups = [@nvws.group.epilogue, @nvws.group.mma], groups.0 = [@nvws.group.epilogue], groups.1 = [@nvws.group.mma]}
      scf.yield %19, %21#0, %21#1, %21#2, %31#0, %30, %31#1 : i32, i32, i32, i32, i32, i1, !ttg.async.token
    } {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load], groups.0 = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load], groups.1 = [@nvws.group.tma_load], groups.2 = [@nvws.group.tma_load], groups.3 = [@nvws.group.tma_load], groups.4 = [@nvws.group.epilogue], groups.5 = [@nvws.group.mma], groups.6 = [@nvws.group.mma]}
    tt.return
  }
}

