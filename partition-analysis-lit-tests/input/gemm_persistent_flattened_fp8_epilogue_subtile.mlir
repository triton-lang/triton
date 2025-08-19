#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 2, 128], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 2, 1]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 128, 2], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
module attributes {"nvws.warp-specialized" = true, "ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm_persistent_flattened_fp8_epilogue_subtile(%arg0: !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>> loc(unknown), %arg1: i32 loc(unknown), %arg2: i32 loc(unknown), %arg3: i64 loc(unknown), %arg4: i64 loc(unknown), %arg5: !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>> loc(unknown), %arg6: i32 loc(unknown), %arg7: i32 loc(unknown), %arg8: i64 loc(unknown), %arg9: i64 loc(unknown), %arg10: !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>> loc(unknown), %arg11: i32 loc(unknown), %arg12: i32 loc(unknown), %arg13: i64 loc(unknown), %arg14: i64 loc(unknown), %arg15: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg16: i32 {tt.divisibility = 16 : i32} loc(unknown), %arg17: i32 {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c84_i32 = arith.constant 84 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
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
    %8 = arith.divsi %7, %c84_i32 : i32
    %9 = arith.remsi %7, %c84_i32 : i32
    %10 = arith.cmpi slt, %0, %9 : i32
    %11 = scf.if %10 -> (i32) {
      %16 = arith.addi %8, %c1_i32 : i32
      scf.yield %16 : i32
    } else {
      scf.yield %8 : i32
    }
    %12 = arith.subi %0, %c84_i32 : i32
    %13 = arith.muli %4, %c8_i32 : i32
    %14 = arith.muli %6, %11 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %15:7 = scf.for %arg18 = %c0_i32 to %14 step %c1_i32 iter_args(%arg19 = %c-1_i32, %arg20 = %12, %arg21 = %c0_i32, %arg22 = %c0_i32, %arg23 = %12, %arg24 = %false, %arg25 = %token) -> (i32, i32, i32, i32, i32, i1, !ttg.async.token)  : i32 {
      %16 = arith.subi %6, %c1_i32 : i32
      %17 = arith.cmpi eq, %arg19, %16 : i32
      %18 = arith.addi %arg19, %c1_i32 : i32
      %19 = arith.select %17, %c0_i32, %18 : i32
      %20 = arith.cmpi eq, %19, %c0_i32 : i32
      %21:3 = scf.if %20 -> (i32, i32, i32) {
        %32 = arith.addi %arg20, %c84_i32 : i32
        %33 = arith.divsi %32, %13 : i32
        %34 = arith.muli %33, %c8_i32 : i32
        %35 = arith.subi %2, %34 : i32
        %36 = arith.minsi %35, %c8_i32 : i32
        %37 = arith.remsi %32, %36 : i32
        %38 = arith.addi %34, %37 : i32
        %39 = arith.remsi %32, %13 : i32
        %40 = arith.divsi %39, %36 : i32
        %41 = arith.muli %38, %c256_i32 : i32
        %42 = arith.muli %40, %c256_i32 : i32
        scf.yield %32, %41, %42 : i32, i32, i32
      } else {
        scf.yield %arg20, %arg21, %arg22 : i32, i32, i32
      }
      %22 = arith.muli %19, %c128_i32 : i32
      %23 = tt.descriptor_load %arg0[%21#1, %22] : !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>> -> tensor<256x128xf8E4M3FN, #blocked>
      %24 = ttg.local_alloc %23 : (tensor<256x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem>
      %25 = tt.descriptor_load %arg5[%21#2, %22] : !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>> -> tensor<256x128xf8E4M3FN, #blocked>
      %26 = ttg.local_alloc %25 : (tensor<256x128xf8E4M3FN, #blocked>) -> !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem>
      %27 = ttg.memdesc_trans %26 {order = array<i32: 1, 0>} : !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem>
      %28 = ttng.tc_gen5_mma %24, %27, %result[%arg25], %arg24, %true : !ttg.memdesc<256x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x256xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %29 = arith.cmpi eq, %19, %16 : i32
      %30 = arith.cmpi ne, %19, %16 : i32
      %31:2 = scf.if %29 -> (i32, !ttg.async.token) {
        %32 = arith.addi %arg23, %c84_i32 : i32
        %33 = arith.divsi %32, %13 : i32
        %34 = arith.muli %33, %c8_i32 : i32
        %35 = arith.subi %2, %34 : i32
        %36 = arith.minsi %35, %c8_i32 : i32
        %37 = arith.remsi %32, %36 : i32
        %38 = arith.addi %34, %37 : i32
        %39 = arith.remsi %32, %13 : i32
        %40 = arith.divsi %39, %36 : i32
        %41 = arith.muli %38, %c256_i32 : i32
        %42 = arith.muli %40, %c256_i32 : i32
        %result_0, %token_1 = ttng.tmem_load %result[%28] : !ttg.memdesc<256x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x256xf32, #blocked1>
        %43 = tt.reshape %result_0 : tensor<256x256xf32, #blocked1> -> tensor<256x2x128xf32, #blocked2>
        %44 = tt.trans %43 {order = array<i32: 0, 2, 1>} : tensor<256x2x128xf32, #blocked2> -> tensor<256x128x2xf32, #blocked3>
        %outLHS, %outRHS = tt.split %44 : tensor<256x128x2xf32, #blocked3> -> tensor<256x128xf32, #blocked4>
        %45 = tt.fp_to_fp %outLHS, rounding = rtne : tensor<256x128xf32, #blocked4> -> tensor<256x128xf8E4M3FN, #blocked4>
        %46 = ttg.convert_layout %45 : tensor<256x128xf8E4M3FN, #blocked4> -> tensor<256x128xf8E4M3FN, #blocked>
        tt.descriptor_store %arg10[%41, %42], %46 : !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>>, tensor<256x128xf8E4M3FN, #blocked>
        %47 = tt.fp_to_fp %outRHS, rounding = rtne : tensor<256x128xf32, #blocked4> -> tensor<256x128xf8E4M3FN, #blocked4>
        %48 = ttg.convert_layout %47 : tensor<256x128xf8E4M3FN, #blocked4> -> tensor<256x128xf8E4M3FN, #blocked>
        %49 = arith.addi %42, %c128_i32 : i32
        tt.descriptor_store %arg10[%41, %49], %48 : !tt.tensordesc<tensor<256x128xf8E4M3FN, #shared>>, tensor<256x128xf8E4M3FN, #blocked>
        scf.yield %32, %token_1 : i32, !ttg.async.token
      } else {
        scf.yield %arg23, %28 : i32, !ttg.async.token
      }
      scf.yield %19, %21#0, %21#1, %21#2, %31#0, %30, %31#1 : i32, i32, i32, i32, i32, i1, !ttg.async.token
    }
    tt.return
  }
}

