// RUN: triton-opt -split-input-file %s --tritongpu-hoist-tmem-alloc --tritongpu-partition-analysis -allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: gemm_persistent_nested

#blocked = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm_persistent_nested(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<256x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x256xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-COUNT-2: ttg.partitions = [0 : i32]
    %false = arith.constant false
    %true = arith.constant true
    // CHECK-NEXT: ttg.partitions = [1 : i32, 2 : i32, 3 : i32]
    %c8_i32 = arith.constant 8 : i32
    // CHECK-COUNT-2: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    // CHECK-COUNT-3: ttg.partitions = [0 : i32, 1 : i32, 2 : i32]
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK-COUNT-2: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    // CHECK-NEXT: ttg.partitions = [0 : i32, 1 : i32, 2 : i32]
    %c63_i32 = arith.constant 63 : i32
    // CHECK-NEXT: ttg.partitions = [0 : i32]
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    // CHECK-COUNT-5: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg15, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg16, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    // CHECK-COUNT-2: ttg.partitions = [0 : i32, 1 : i32, 2 : i32]
    %5 = arith.addi %arg17, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    // CHECK-NEXT: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    %7 = arith.muli %2, %4 : i32
    // CHECK-NEXT: ttg.partitions = [1 : i32, 2 : i32, 3 : i32]
    %8 = arith.muli %4, %c8_i32 : i32
    // CHECK-NEXT: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    %9 = tt.get_num_programs x : i32
    scf.for %arg18 = %0 to %7 step %9  : i32 {
      // CHECK-COUNT-4: ttg.partitions = [1 : i32, 2 : i32, 3 : i32]
      %10 = arith.divsi %arg18, %8 : i32
      %11 = arith.muli %10, %c8_i32 : i32
      %12 = arith.subi %2, %11 : i32
      %13 = arith.minsi %12, %c8_i32 : i32
      // CHECK-COUNT-2: ttg.partitions = [1 : i32, 3 : i32]
      %14 = arith.remsi %arg18, %13 : i32
      %15 = arith.addi %11, %14 : i32
      // CHECK-COUNT-2: ttg.partitions = [2 : i32, 3 : i32]
      %16 = arith.remsi %arg18, %8 : i32
      %17 = arith.divsi %16, %13 : i32
      // CHECK-NEXT: ttg.partitions = [1 : i32, 3 : i32]
      %18 = arith.muli %15, %c128_i32 : i32
      // CHECK-NEXT: ttg.partitions = [2 : i32, 3 : i32]
      %19 = arith.muli %17, %c256_i32 : i32
      // CHECK-NEXT: ttg.partitions = [0 : i32, 3 : i32]
      %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK-NEXT: ttg.partitions = [0 : i32]
      %20 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %21:3 = scf.for %arg19 = %c0_i32 to %6 step %c1_i32 iter_args(%arg20 = %c0_i32, %arg21 = %false, %arg22 = %20) -> (i32, i1, !ttg.async.token)  : i32 {
        // CHECK-COUNT-2: ttg.partitions = [1 : i32]
        %24 = tt.descriptor_load %arg0[%18, %arg20] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
        %25 = ttg.local_alloc %24 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
        // CHECK-COUNT-2: ttg.partitions = [2 : i32]
        %26 = tt.descriptor_load %arg5[%19, %arg20] : !tt.tensordesc<tensor<256x64xf16, #shared>> -> tensor<256x64xf16, #blocked1>
        %27 = ttg.local_alloc %26 : (tensor<256x64xf16, #blocked1>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
        // CHECK-COUNT-2: ttg.partitions = [0 : i32]
        %28 = ttg.memdesc_trans %27 {order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem> -> !ttg.memdesc<64x256xf16, #shared1, #smem>
        %29 = ttng.tc_gen5_mma %25, %28, %result[%arg22], %arg21, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x256xf16, #shared1, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
        // CHECK-NEXT: ttg.partitions = [1 : i32, 2 : i32]
        %30 = arith.addi %arg20, %c64_i32 : i32
        scf.yield %30, %true, %29 : i32, i1, !ttg.async.token
      // CHECK: ttg.partitions = [0 : i32, 1 : i32, 2 : i32], ttg.partitions.0 = [1 : i32, 2 : i32], ttg.partitions.1 = [0 : i32], ttg.partitions.2 = [0 : i32]
      }
      // CHECK-COUNT-4: ttg.partitions = [3 : i32]
      %result_0, %token_1 = ttng.tmem_load %result[%21#2] : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
      %22 = arith.truncf %result_0 : tensor<128x256xf32, #blocked> to tensor<128x256xf16, #blocked>
      %23 = ttg.convert_layout %22 : tensor<128x256xf16, #blocked> -> tensor<128x256xf16, #blocked2>
      tt.descriptor_store %arg10[%18, %19], %23 : !tt.tensordesc<tensor<128x256xf16, #shared>>, tensor<128x256xf16, #blocked2>
    // CHECK: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    }
    tt.return
  }
}

// -----

// CHECK-LABEL: gemm_persistent_flattened

#blocked = #ttg.blocked<{sizePerThread = [1, 256], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-stages" = 3 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @gemm_persistent_flattened(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<tensor<256x64xf16, #shared>>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<tensor<128x256xf16, #shared>>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: i32 {tt.divisibility = 16 : i32}, %arg16: i32 {tt.divisibility = 16 : i32}, %arg17: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-COUNT-2: ttg.partitions = [0 : i32]
    %false = arith.constant false
    %true = arith.constant true
    // CHECK-COUNT-4: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    %c84_i32 = arith.constant 84 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK-NEXT: ttg.partitions = [1 : i32, 2 : i32, 3 : i32]
    %c8_i32 = arith.constant 8 : i32
    // CHECK-COUNT-6: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c64_i32 = arith.constant 64 : i32
    %c127_i32 = arith.constant 127 : i32
    %c255_i32 = arith.constant 255 : i32
    %c63_i32 = arith.constant 63 : i32
    // CHECK-NEXT: ttg.partitions = [0 : i32]
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    // CHECK-COUNT-11: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg15, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg16, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.addi %arg17, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %2, %4 : i32
    %8 = arith.divsi %7, %c84_i32 : i32
    %9 = arith.remsi %7, %c84_i32 : i32
    %10 = arith.cmpi slt, %0, %9 : i32
    %11 = scf.if %10 -> (i32) {
      // CHECK: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
      %17 = arith.addi %8, %c1_i32 : i32
      scf.yield %17 : i32
    } else {
      scf.yield %8 : i32
    // CHECK: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32], ttg.partitions.0 = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    }
    // CHECK-COUNT-2: ttg.partitions = [1 : i32, 2 : i32, 3 : i32]
    %12 = arith.subi %0, %c84_i32 : i32
    %13 = arith.muli %4, %c8_i32 : i32
    // CHECK-NEXT: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
    %14 = arith.muli %6, %11 : i32
    // CHECK-NEXT: ttg.partitions = [0 : i32, 3 : i32]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK-NEXT: ttg.partitions = [0 : i32]
    %15 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %16:7 = scf.for %arg18 = %c0_i32 to %14 step %c1_i32 iter_args(%arg19 = %c-1_i32, %arg20 = %12, %arg21 = %c0_i32, %arg22 = %c0_i32, %arg23 = %12, %arg24 = %false, %arg25 = %15) -> (i32, i32, i32, i32, i32, i1, !ttg.async.token)  : i32 {
      // CHECK-COUNT-4: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32]
      %17 = arith.subi %6, %c1_i32 : i32
      %18 = arith.cmpi eq, %arg19, %17 : i32
      %19 = arith.addi %arg19, %c1_i32 : i32
      %20 = arith.select %18, %c0_i32, %19 : i32
      // CHECK-NEXT: ttg.partitions = [1 : i32, 2 : i32]
      %21 = arith.cmpi eq, %20, %c0_i32 : i32
      %22:3 = scf.if %21 -> (i32, i32, i32) {
        // CHECK-COUNT-5: ttg.partitions = [1 : i32, 2 : i32]
        %33 = arith.addi %arg20, %c84_i32 : i32
        %34 = arith.divsi %33, %13 : i32
        %35 = arith.muli %34, %c8_i32 : i32
        %36 = arith.subi %2, %35 : i32
        %37 = arith.minsi %36, %c8_i32 : i32
        // CHECK-COUNT-2: ttg.partitions = [1 : i32]
        %38 = arith.remsi %33, %37 : i32
        %39 = arith.addi %35, %38 : i32
        // CHECK-COUNT-2: ttg.partitions = [2 : i32]
        %40 = arith.remsi %33, %13 : i32
        %41 = arith.divsi %40, %37 : i32
        // CHECK-NEXT: ttg.partitions = [1 : i32]
        %42 = arith.muli %39, %c128_i32 : i32
        // CHECK-NEXT: ttg.partitions = [2 : i32]
        %43 = arith.muli %41, %c256_i32 : i32
        scf.yield %33, %42, %43 : i32, i32, i32
      } else {
        scf.yield %arg20, %arg21, %arg22 : i32, i32, i32
      // CHECK: ttg.partitions = [1 : i32, 2 : i32], ttg.partitions.0 = [1 : i32, 2 : i32], ttg.partitions.1 = [1 : i32], ttg.partitions.2 = [2 : i32]
      }
      // CHECK-NEXT: ttg.partitions = [1 : i32, 2 : i32]
      %23 = arith.muli %20, %c64_i32 : i32
      // CHECK-COUNT-2: ttg.partitions = [1 : i32]
      %24 = tt.descriptor_load %arg0[%22#1, %23] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %25 = ttg.local_alloc %24 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // CHECK-COUNT-2: ttg.partitions = [2 : i32]
      %26 = tt.descriptor_load %arg5[%22#2, %23] : !tt.tensordesc<tensor<256x64xf16, #shared>> -> tensor<256x64xf16, #blocked1>
      %27 = ttg.local_alloc %26 : (tensor<256x64xf16, #blocked1>) -> !ttg.memdesc<256x64xf16, #shared, #smem>
      // CHECK-COUNT-2: ttg.partitions = [0 : i32]
      %28 = ttg.memdesc_trans %27 {order = array<i32: 1, 0>} : !ttg.memdesc<256x64xf16, #shared, #smem> -> !ttg.memdesc<64x256xf16, #shared1, #smem>
      %29 = ttng.tc_gen5_mma %25, %28, %result[%arg25], %arg24, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x256xf16, #shared1, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: ttg.partitions = [0 : i32, 3 : i32]
      %30 = arith.cmpi eq, %20, %17 : i32
      // CHECK-NEXT: ttg.partitions = [0 : i32]
      %31 = arith.cmpi ne, %20, %17 : i32
      %32:2 = scf.if %30 -> (i32, !ttg.async.token) {
        // CHECK-COUNT-15: ttg.partitions = [3 : i32]
        %33 = arith.addi %arg23, %c84_i32 : i32
        %34 = arith.divsi %33, %13 : i32
        %35 = arith.muli %34, %c8_i32 : i32
        %36 = arith.subi %2, %35 : i32
        %37 = arith.minsi %36, %c8_i32 : i32
        %38 = arith.remsi %33, %37 : i32
        %39 = arith.addi %35, %38 : i32
        %40 = arith.remsi %33, %13 : i32
        %41 = arith.divsi %40, %37 : i32
        %42 = arith.muli %39, %c128_i32 : i32
        %43 = arith.muli %41, %c256_i32 : i32
        %result_0, %token_1 = ttng.tmem_load %result[%29] : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
        %44 = arith.truncf %result_0 : tensor<128x256xf32, #blocked> to tensor<128x256xf16, #blocked>
        %45 = ttg.convert_layout %44 : tensor<128x256xf16, #blocked> -> tensor<128x256xf16, #blocked2>
        tt.descriptor_store %arg10[%42, %43], %45 : !tt.tensordesc<tensor<128x256xf16, #shared>>, tensor<128x256xf16, #blocked2>
        scf.yield %33, %token_1 : i32, !ttg.async.token
      } else {
        scf.yield %arg23, %29 : i32, !ttg.async.token
      // CHECK: ttg.partitions = [0 : i32, 3 : i32], ttg.partitions.0 = [3 : i32], ttg.partitions.1 = [0 : i32]
      }
      scf.yield %20, %22#0, %22#1, %22#2, %32#0, %31, %32#1 : i32, i32, i32, i32, i32, i1, !ttg.async.token
    // CHECK: ttg.partitions = [0 : i32, 1 : i32, 2 : i32, 3 : i32], ttg.partitions.0 = [0 : i32, 1 : i32, 2 : i32, 3 : i32], ttg.partitions.1 = [1 : i32, 2 : i32], ttg.partitions.2 = [1 : i32], ttg.partitions.3 = [2 : i32], ttg.partitions.4 = [3 : i32], ttg.partitions.5 = [0 : i32], ttg.partitions.6 = [0 : i32]
    }
    tt.return
  }
}
