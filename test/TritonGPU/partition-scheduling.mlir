// RUN: triton-opt %s --split-input-file --tritongpu-hoist-tmem-alloc --tritongpu-partition-scheduling -allow-unregistered-dialect | FileCheck %s

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: matmul_nested_persistent_ws_kernel
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
    // CHECK: ttng.tmem_alloc {{.*}} : (tensor<128x128xf32, #blocked>)
    // CHECK: scf.for
    // CHECK-Not: tmem_alloc
    scf.for %tile_id = %start_pid to %num_tiles step %c148_i32  : i32 {
      // CHECK-COUNT-10: {ttg.partition = array<i32: 0, 2>}
      // CHECK-COUNT-3: {ttg.partition = array<i32: 1, 2>}
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
      // a random upper bound dependent on the outer tile id
      %ub = arith.addi %k_tiles_5, %tile_id : i32
      %assume = arith.cmpi sgt, %ub, %c0_i32 : i32
      llvm.intr.assume %assume : i1
      // CHECK: scf.for
      %accumulator_11:2 = scf.for %accumulator_15 = %c0_i32 to %ub step %c1_i32 iter_args(%arg11 = %false, %accumulator_16 = %accumulator_10) -> (i1, !ttg.async.token)  : i32 {
	// CHECK: arith.muli {{.*}}ttg.partition = array<i32: 2>}
        %off_k = arith.muli %accumulator_15, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
	// CHECK: tt.descriptor_load {{.*}}ttg.partition = array<i32: 2>}
        %a = tt.descriptor_load %a_desc_0[%off_am, %off_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
        %a_17 = ttg.local_alloc %a {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %b = tt.descriptor_load %b_desc_1[%off_bn, %off_k] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
        %accumulator_18 = ttg.local_alloc %b {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %accumulator_19 = ttg.memdesc_trans %accumulator_18 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
        // CHECK: ttng.tc_gen5_mma {{.*}}ttg.partition = array<i32: 1>}
        %accumulator_20 = ttng.tc_gen5_mma %a_17, %accumulator_19, %accumulator[%accumulator_16], %arg11, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield %true, %accumulator_20 : i1, !ttg.async.token
      // CHECK: } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1>]}
      } {tt.scheduled_max_stage = 2 : i32}
      // CHECK-COUNT-4: {ttg.partition = array<i32: 0>}
      %accumulator_12, %accumulator_13 = ttng.tmem_load %accumulator[%accumulator_11#1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %c = tt.fp_to_fp %accumulator_12, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %c_14 = ttg.convert_layout %c : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
      tt.descriptor_store %c_desc_2[%off_am, %off_bn], %c_14 : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, tensor<128x128xf8E4M3FN, #blocked1>
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    tt.return
  }

  // CHECK-LABEL: matmul_nested_persistent_ws_kernel_no_hoist
  tt.func public @matmul_nested_persistent_ws_kernel_no_hoist(%a_desc_0: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %b_desc_1: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %c_desc_2: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    // CHECK: scf.for
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
      // CHECK: tmem_alloc {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NOT: tmem_store
      %accumulator, %accumulator_9 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %accumulator_10 = ttng.tmem_store %cst, %accumulator[%accumulator_9], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // a random upper bound dependent on the outer tile id. There is no assume op asserting that the inner loop executs at least once.
      %ub = arith.addi %k_tiles_5, %tile_id : i32
      // CHECK: scf.for
      %accumulator_11:2 = scf.for %accumulator_15 = %c0_i32 to %ub step %c1_i32 iter_args(%arg11 = %false, %accumulator_16 = %accumulator_10) -> (i1, !ttg.async.token)  : i32 {
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

// -----

// CHECK-LABEL: attention_persistent_inner_loop_kernel
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @attention_persistent_inner_loop_kernel(%desc_q: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_q_0: i32, %desc_q_1: i32, %desc_q_2: i64, %desc_q_3: i64, %desc_k: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_k_4: i32, %desc_k_5: i32, %desc_k_6: i64, %desc_k_7: i64, %desc_v: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_v_8: i32, %desc_v_9: i32, %desc_v_10: i64, %desc_v_11: i64, %desc_acc: !tt.tensordesc<tensor<128x128xf16, #shared>>, %desc_acc_12: i32, %desc_acc_13: i32, %desc_acc_14: i64, %desc_acc_15: i64, %l_i_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %m_i_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %qk_scale: f32) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_16 = arith.constant dense<0xFF800000> : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_17 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %prog_id = tt.get_program_id x : i32
    %num_sm = tt.get_num_programs x : i32
    %num_tiles = arith.divsi %M, %c128_i32 : i32
    %tiles_per_sm = arith.divsi %num_tiles, %num_sm : i32
    // CHECK: scf.for
    %tile_idx = scf.for %_ = %c0_i32 to %tiles_per_sm step %c1_i32 iter_args(%tile_idx_20 = %prog_id) -> (i32)  : i32 {
      %off_m = arith.muli %tile_idx_20, %c128_i32 : i32
      %q = tt.descriptor_load %desc_q[%off_m, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
      %q_21 = ttg.local_alloc %q : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      // CHECK: ttng.tmem_alloc {ttg.partition = array<i32: 0, 1>}
      %qk_22, %qk_23 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      // CHECK: ttng.tmem_alloc {{.*}} {ttg.partition = array<i32: 3>}
      // CHECK-NOT: tmem_store
      %acc, %acc_24 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %acc_25 = ttng.tmem_store %cst_17, %acc[%acc_24], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: scf.for
      %acc_26:4 = scf.for %acc_30 = %c0_i32 to %N step %c128_i32 iter_args(%arg28 = %cst_16, %arg29 = %cst, %qk_31 = %qk_23, %acc_32 = %acc_25) -> (tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
        %k = tt.descriptor_load %desc_k[%acc_30, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
        %k_33 = ttg.local_alloc %k : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %k_34 = ttg.memdesc_trans %k_33 {order = array<i32: 1, 0>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared1, #smem>
        %qk_35 = ttng.tc_gen5_mma %q_21, %k_34, %qk_22[%qk_31], %false, %true : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %qk_36, %qk_37 = ttng.tmem_load %qk_22[%qk_35] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>

        %acc_47, %p, %next_l_i, %row_max = "softmax_work"(%qk_36, %arg29, %arg28) : (tensor<128x128xf32, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> (tensor<128x128xf32, #blocked>, tensor<128x128xf16, #blocked>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>)

        %acc_48, %acc_49 = ttng.tmem_load %acc[%acc_32] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        %acc_50 = arith.mulf %acc_48, %acc_47 : tensor<128x128xf32, #blocked>
        %p_53 = ttg.local_alloc %p : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
        %acc_54 = ttng.tmem_store %acc_50, %acc[%acc_49], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        %v = tt.descriptor_load %desc_v[%acc_30, %c0_i32] : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
        %v_51 = ttg.local_alloc %v : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared, #smem>

        %acc_55 = ttng.tc_gen5_mma %p_53, %v_51, %acc[%acc_54], %true, %true : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

        scf.yield %row_max, %next_l_i, %qk_37, %acc_55 : tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<128xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
      // CHECK: } {ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 3>]}
      }
      // CHECK: arith.addi {{.*}}, {{.*}} {ttg.partition = array<i32: 2>}
      %tile_idx_29 = arith.addi %tile_idx_20, %num_sm : i32
      scf.yield %tile_idx_29 : i32
    } {tt.num_stages = 3 : i32, tt.warp_specialize}
    tt.return
  }
}
