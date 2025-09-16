#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
module attributes {"nvws.warp-specialized" = true, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm_persistent_nested_fp8_epilogue_subtile(%arg0: !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>> loc(unknown), %arg1: i32 loc(unknown), %arg2: i32 loc(unknown), %arg3: i64 loc(unknown), %arg4: i64 loc(unknown), %arg5: !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>> loc(unknown), %arg6: i32 loc(unknown), %arg7: i32 loc(unknown), %arg8: i64 loc(unknown), %arg9: i64 loc(unknown), %arg10: !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>> loc(unknown), %arg11: i32 loc(unknown), %arg12: i32 loc(unknown), %arg13: i64 loc(unknown), %arg14: i64 loc(unknown), %arg15: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg16: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg17: i32 {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
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
      %10 = arith.divsi %arg18, %8 : i32
      %11 = arith.muli %10, %c8_i32 : i32
      %12 = arith.subi %2, %11 : i32
      %13 = arith.minsi %12, %c8_i32 : i32
      %14 = arith.remsi %arg18, %13 : i32
      %15 = arith.addi %11, %14 : i32
      %16 = arith.remsi %arg18, %8 : i32
      %17 = arith.divsi %16, %13 : i32
      %18 = arith.muli %15, %c256_i32 : i32
      %19 = arith.muli %17, %c256_i32 : i32
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %20:3 = scf.for %arg19 = %c0_i32 to %6 step %c1_i32 iter_args(%arg20 = %c0_i32, %arg21 = %false, %arg22 = %token) -> (i32, i1, !ttg.async.token)  : i32 {
        %28 = tt.descriptor_load %arg0[%18, %arg20] : !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>> -> tensor<256x128xf8E4M3FN, #blocked>
        %29 = ttg.local_alloc %28 : (tensor<256x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem>
        %30 = tt.descriptor_load %arg5[%19, %arg20] : !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>> -> tensor<256x128xf8E4M3FN, #blocked>
        %31 = ttg.local_alloc %30 : (tensor<256x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem>
        %32 = ttg.memdesc_trans %31 {order = array<i32: 1, 0>} : !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem>
        %33 = ttng.tc_gen5_mma %29, %32, %result[%arg22], %arg21, %true : !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable>
        %34 = arith.addi %arg20, %c128_i32 : i32
        scf.yield %34, %true, %33 : i32, i1, !ttg.async.token
      }
      %result_0, %token_1 = ttng.tmem_load %result[%20#2] : !ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x256xf32, #blocked1>
      %21 = tt.reshape %result_0 : tensor<256x256xf32, #blocked1> -> tensor<256x2x128xf32, #blocked2>
      %22 = tt.trans %21 {order = array<i32: 0, 2, 1>} : tensor<256x2x128xf32, #blocked2> -> tensor<256x128x2xf32, #blocked3>
      %outLHS, %outRHS = tt.split %22 : tensor<256x128x2xf32, #blocked3> -> tensor<256x128xf32, #blocked4>
      %23 = tt.fp_to_fp %outLHS, rounding = rtne : tensor<256x128xf32, #blocked4> -> tensor<256x128xf8E4M3FN, #blocked4>
      %24 = ttg.convert_layout %23 : tensor<256x128xf8E4M3FN, #blocked4> -> tensor<256x128xf8E4M3FN, #blocked>
      tt.descriptor_store %arg10[%18, %19], %24 : !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>>, tensor<256x128xf8E4M3FN, #blocked>
      %25 = tt.fp_to_fp %outRHS, rounding = rtne : tensor<256x128xf32, #blocked4> -> tensor<256x128xf8E4M3FN, #blocked4>
      %26 = ttg.convert_layout %25 : tensor<256x128xf8E4M3FN, #blocked4> -> tensor<256x128xf8E4M3FN, #blocked>
      %27 = arith.addi %19, %c128_i32 : i32
      tt.descriptor_store %arg10[%18, %27], %26 : !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>>, tensor<256x128xf8E4M3FN, #blocked>
    }
    tt.return
  }
}