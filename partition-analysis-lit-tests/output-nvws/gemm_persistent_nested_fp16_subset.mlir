#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
module attributes {"nvws.warp-specialized" = true, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm_persistent_nested_fp16(%arg0: !tt.tensordesc<tensor<256x128xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<256x128xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<256x256xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1_i32 = arith.constant 1 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg15, %c255_i32 : i32
    %2 = arith.divsi %1, %c256_i32 : i32
    %3 = arith.addi %arg16, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.addi %arg17, %c127_i32 : i32
    %6 = arith.divsi %5, %c128_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.muli %4, %c8_i32 : i32
    %9 = tt.get_num_programs x : i32
    scf.for %arg18 = %0 to %7 step %9  : i32 {
      %10 = arith.divsi %arg18, %8 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
      %11 = arith.muli %10, %c8_i32 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
      %12 = arith.subi %2, %11 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
      %13 = arith.minsi %12, %c8_i32 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
      %14 = arith.remsi %arg18, %13 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
      %15 = arith.addi %11, %14 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
      %16 = arith.remsi %arg18, %8 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
      %17 = arith.divsi %16, %13 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
      %18 = arith.muli %15, %c256_i32 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
      %19 = arith.muli %17, %c256_i32 {groups = [@nvws.group.epilogue, @nvws.group.tma_load]} : i32
      %result, %token = ttng.tmem_alloc {groups = [@nvws.group.epilogue, @nvws.group.mma]} : () -> (!ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %20:3 = scf.for %arg19 = %c0_i32 to %6 step %c1_i32 iter_args(%arg20 = %c0_i32, %arg21 = %false, %arg22 = %token) -> (i32, i1, !ttg.async.token)  : i32 {
        %23 = tt.descriptor_load %arg0[%18, %arg20] {groups = [@nvws.group.tma_load]} : !tt.tensordesc<tensor<256x128xf16, #shared>> -> tensor<256x128xf16, #blocked>
        %24 = ttg.local_alloc %23 {groups = [@nvws.group.tma_load]} : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared, #smem>
        %25 = tt.descriptor_load %arg5[%19, %arg20] {groups = [@nvws.group.tma_load]} : !tt.tensordesc<tensor<256x128xf16, #shared>> -> tensor<256x128xf16, #blocked>
        %26 = ttg.local_alloc %25 {groups = [@nvws.group.tma_load]} : (tensor<256x128xf16, #blocked>) -> !ttg.memdesc<256x128xf16, #shared, #smem>
        %27 = ttg.memdesc_trans %26 {groups = [@nvws.group.mma], order = array<i32: 1, 0>} : !ttg.memdesc<256x128xf16, #shared, #smem> -> !ttg.memdesc<128x256xf16, #shared1, #smem>
        %28 = ttng.tc_gen5_mma %24, %27, %result[%arg22], %arg21, %true {groups = [@nvws.group.mma]} : !ttg.memdesc<256x128xf16, #shared, #smem>, !ttg.memdesc<128x256xf16, #shared1, #smem>, !ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable>
        %29 = arith.addi %arg20, %c128_i32 {groups = [@nvws.group.tma_load]} : i32
        scf.yield %29, %true, %28 : i32, i1, !ttg.async.token
      } {groups = [@nvws.group.mma, @nvws.group.tma_load], groups.0 = [@nvws.group.tma_load], groups.1 = [@nvws.group.mma], groups.2 = [@nvws.group.mma]}
      %result_0, %token_1 = ttng.tmem_load %result[%20#2] {groups = [@nvws.group.epilogue]} : !ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x256xf32, #blocked1>
      %21 = arith.truncf %result_0 {groups = [@nvws.group.epilogue]} : tensor<256x256xf32, #blocked1> to tensor<256x256xf16, #blocked1>
      %22 = ttg.convert_layout %21 {groups = [@nvws.group.epilogue]} : tensor<256x256xf16, #blocked1> -> tensor<256x256xf16, #blocked>
      tt.descriptor_store %arg10[%18, %19], %22 {groups = [@nvws.group.epilogue]} : !tt.tensordesc<tensor<256x256xf16, #shared>>, tensor<256x256xf16, #blocked>
    } {groups = [@nvws.group.epilogue, @nvws.group.mma, @nvws.group.tma_load], nvws.group.epilogue = {num_warps = 4 : i32, start_warp = 0 : i32}, nvws.group.mma = {num_warps = 1 : i32, start_warp = 4 : i32}, nvws.group.tma_load = {num_warps = 1 : i32, start_warp = 5 : i32}, "nvws.warp-specialize"}
    tt.return
  }
}

