// RUN: triton-opt %s --split-input-file --tritongpu-hoist-tmem-alloc --tritongpu-partition-scheduling -allow-unregistered-dialect | FileCheck %s

// CHECK: matmul_nested_persistent_ws_kernel
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_nested_persistent_ws_kernel(%a_desc_0: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %b_desc_1: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %c_desc_2: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %start_pid = tt.get_program_id x : i32
    %num_pid_m_3 = arith.divsi %M, %c128_i32 : i32
    %num_pid_n_4 = arith.divsi %N, %c128_i32 : i32
    %k_tiles_5 = arith.divsi %K, %c128_i32 : i32
    %num_tiles = arith.muli %num_pid_m_3, %num_pid_n_4 : i32
    %num_pid_in_group = arith.muli %num_pid_n_4, %c8_i32 : i32
    scf.for %tile_id = %start_pid to %num_tiles step %c148_i32  : i32 {
      %group_id = arith.divsi %tile_id, %num_pid_in_group : i32
      %first_pid_m = arith.muli %group_id, %c8_i32 : i32
      %group_size_m = arith.subi %num_pid_m_3, %first_pid_m : i32
      %group_size_m_6 = arith.minsi %group_size_m, %c8_i32 : i32
      %pid_m = arith.remsi %tile_id, %group_size_m_6 : i32
      %pid_m_7 = arith.addi %first_pid_m, %pid_m : i32
      %pid_n = arith.remsi %tile_id, %num_pid_in_group : i32
      %pid_n_8 = arith.divsi %pid_n, %group_size_m_6 : i32
      %off_am = arith.muli %pid_m_7, %c128_i32 : i32
      %off_bn = arith.muli %pid_n_8, %c128_i32 : i32
      %accumulator, %accumulator_9 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %accumulator_10 = ttng.tmem_store %cst, %accumulator[%accumulator_9], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %accumulator_11:2 = scf.for %accumulator_15 = %c0_i32 to %k_tiles_5 step %c1_i32 iter_args(%arg11 = %false, %accumulator_16 = %accumulator_10) -> (i1, !ttg.async.token)  : i32 {
        %off_k = arith.muli %accumulator_15, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
        %a = tt.descriptor_load %a_desc_0[%off_am, %off_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
        %a_17 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %b = tt.descriptor_load %b_desc_1[%off_bn, %off_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
        %accumulator_18 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %accumulator_19 = ttg.memdesc_trans %accumulator_18 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
        %accumulator_20 = ttng.tc_gen5_mma %a_17, %accumulator_19, %accumulator[%accumulator_16], %arg11, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %accumulator_20 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32}
      %accumulator_12, %accumulator_13 = ttng.tmem_load %accumulator[%accumulator_11#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %c = tt.fp_to_fp %accumulator_12, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %c_14 = ttg.convert_layout %c : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
      tt.descriptor_store %c_desc_2[%off_am, %off_bn], %c_14 : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, tensor<128x128xf8E4M3FN, #blocked1>
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    tt.return
  }
}