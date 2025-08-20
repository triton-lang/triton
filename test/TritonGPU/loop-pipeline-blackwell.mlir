// RUN: triton-opt %s -split-input-file -tritongpu-hoist-tmem-alloc -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -triton-nvidia-gpu-remove-tmem-tokens -canonicalize | FileCheck %s --check-prefixes=CHECK

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @chained_dot_scaled_acc
  // CHECK-DAG: %[[C0_F:.+]] = arith.constant dense<0.000000e+00>
  // CHECK-DAG: %[[TRUE:.+]] = arith.constant true
  // CHECK-DAG: %[[C0:.+]] = arith.constant 0 : i32
  // CHECK-DAG: %[[C1:.+]] = arith.constant 1 : i32
  // CHECK-DAG: %[[C2:.+]] = arith.constant 2 : i32
  // CHECK: %[[TMEM_BUF:.+]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32
  // CHECK: ttng.tmem_store %[[C0_F]], %[[TMEM_BUF]], %[[TRUE]]
  // CHECK: %[[BAR_BUF:.+]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: %[[ACC1:.+]] = ttng.tmem_load %[[TMEM_BUF]]
  // CHECK: %[[ACC2:.+]] = arith.mulf %[[ACC1]]
  // CHECK: ttng.tmem_store %[[ACC2]], %[[TMEM_BUF]]
  // CHECK: %[[BAR_SLICE:.+]] = ttg.memdesc_index %[[BAR_BUF]]{{\[}}%[[C0]]{{\]}}
  // CHECK: ttng.tc_gen5_mma %[[A_OP:.*]], %[[B_OP:.*]], %[[TMEM_BUF]], {{.*}}, %[[BAR_SLICE]]
  // CHECK: scf.for {{.*}} iter_args(%[[PHASE:.+]] = %[[C0]], %[[BAR_IDX:.+]] = %[[C1]], {{.*}}, %[[BAR_PREV:.*]] = %[[BAR_SLICE]], %[[PHASE_PREV:.+]] = %[[C0]], %[[A_DEP:.+]] = %[[A_OP]], %[[B_DEP:.+]] = %[[B_OP]]
  // CHECK:   ttng.wait_barrier %[[BAR_PREV]], %[[PHASE_PREV]] deps %[[A_DEP]], %[[B_DEP]]
  // CHECK:   %[[ACC1:.+]] = ttng.tmem_load %[[TMEM_BUF]]
  // CHECK:   %[[ACC2:.+]] = arith.mulf %[[ACC1]]
  // CHECK:   ttng.tmem_store %[[ACC2]], %[[TMEM_BUF]]
  // CHECK:   %[[BAR_SLICE:.+]] = ttg.memdesc_index %[[BAR_BUF]]{{\[}}%[[BAR_IDX]]{{\]}}
  // CHECK:   ttng.tc_gen5_mma %[[A_OP:.*]], %[[B_OP:.*]], %[[TMEM_BUF]], %[[TRUE]], {{.*}}, %[[BAR_SLICE]]
  // CHECK:   %[[PHASE_NEG:.+]] = arith.xori %[[PHASE]], %[[C1]]
  // CHECK:   %[[BAR_IDX_P1:.+]] = arith.addi %[[BAR_IDX]], %[[C1]]
  // CHECK:   %[[BAR_IDX_CMP:.+]] = arith.cmpi sge, %[[BAR_IDX_P1]], %[[C2]]
  // CHECK:   %[[BAR_IDX_NEXT:.+]] = arith.select %[[BAR_IDX_CMP]], %[[C0]], %[[BAR_IDX_P1]]
  // CHECK:   %[[PHASE_NEXT:.+]] = arith.select %[[BAR_IDX_CMP]], %[[PHASE_NEG]], %[[PHASE]]
  // CHECK:   scf.yield %[[PHASE_NEXT]], %[[BAR_IDX_NEXT]], {{.*}}, %[[BAR_SLICE]], %[[PHASE]], %[[A_OP]], %[[B_OP]]
  // CHECK: ttg.local_dealloc %[[BAR_BUF]]
  // CHECK: ttng.tmem_load %[[TMEM_BUF]]
  tt.func public @chained_dot_scaled_acc(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}, %arg3: i32) -> tensor<128x128xf16, #blocked> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %sacc = arith.mulf %acc, %cst2 : tensor<128x128xf32, #blocked>
      %acc_tm, %acc_tok = ttng.tmem_alloc %sacc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %acc_res : tensor<128x128xf32, #blocked>
    }
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @chained_scale_after_dot
  // CHECK: ttng.tmem_alloc
  // CHECK: scf.for
  // CHECK:   ttng.tc_gen5_mma
  // CHECK:   ttng.tmem_load
  // CHECK:   arith.mulf
  // CHECK:   ttng.tmem_store
  tt.func public @chained_scale_after_dot(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %B_ptr: tensor<128x128x!tt.ptr<f16>, #blocked1>, %arg3: i32) -> tensor<128x128xf16, #blocked> {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst2 = arith.constant dense<2.000000e+00> : tensor<128x128xf32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %res = scf.for %i = %c0_i32 to %arg3 step %c1_i32 iter_args(%acc = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %A = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %A_sh = ttg.local_alloc %A : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %B = tt.load %B_ptr : tensor<128x128x!tt.ptr<f16>, #blocked1>
      %B_sh = ttg.local_alloc %B : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>
      %acc_tm, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %A_sh, %B_sh, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %acc_res, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %sacc = arith.mulf %acc_res, %cst2 : tensor<128x128xf32, #blocked>
      scf.yield %sacc : tensor<128x128xf32, #blocked>
    }
    %res_f16 = arith.truncf %res : tensor<128x128xf32, #blocked> to tensor<128x128xf16, #blocked>
    tt.return %res_f16 : tensor<128x128xf16, #blocked>
  }
}

// -----
// 4 warps
// matmul: 128x32 @ 32x128 -> 128x128
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #ttg.slice<{parent=#AL, dim=0}>
#BLs0 = #ttg.slice<{parent=#BL, dim=0}>
#BLs1 = #ttg.slice<{parent=#BL, dim=1}>
#C = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#A = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#B = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @matmul_loop_cast_load(%lb : index, %ub : index, %step : index,
                    %A : !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32},
                    %B : !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #C> {
// CHECK-LABEL: tt.func @matmul_loop_cast_load
// CHECK: scf.for
// CHECK: ttg.local_load
// CHECK: tt.fp_to_fp
// CHECK: ttng.wait_barrier
// CHECK: ttg.local_store
// CHECK: ttg.memdesc_trans
// CHECK: ttng.tc_gen5_mma {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}
// CHECK: ttg.async_copy_global_to_local
    %a_ptr_splat = tt.splat %A : !tt.ptr<f8E4M3FN> -> tensor<128x32x!tt.ptr<f8E4M3FN>, #AL>
    %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
    %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<32xi32, #ALs0> -> tensor<1x32xi32, #AL>
    %a_offs = tt.broadcast %a_tmp1 : tensor<1x32xi32, #AL> -> tensor<128x32xi32, #AL>
    %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f8E4M3FN>, #AL>, tensor<128x32xi32, #AL>

    %b_ptr_splat = tt.splat %B : !tt.ptr<f8E4M3FN> -> tensor<128x32x!tt.ptr<f8E4M3FN>, #BL>
    %b_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #BLs0>
    %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<32xi32, #BLs0> -> tensor<1x32xi32, #BL>
    %b_offs = tt.broadcast %b_tmp1 : tensor<1x32xi32, #BL> -> tensor<128x32xi32, #BL>
    %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<128x32x!tt.ptr<f8E4M3FN>, #BL>, tensor<128x32xi32, #BL>

    %true = arith.constant true
    %b_mask = arith.constant dense<true> : tensor<128x32xi1, #BL>
    %b_other = arith.constant dense<0.00e+00> : tensor<128x32xf8E4M3FN, #BL>
    %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

    %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
    %b_off = arith.constant dense<4> : tensor<128x32xi32, #BL>

    %loop:3 = scf.for %iv = %lb to %ub step %step iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init) -> (tensor<128x32x!tt.ptr<f8E4M3FN>, #AL>, tensor<128x32x!tt.ptr<f8E4M3FN>, #BL>, tensor<128x128xf32, #C>) {
      %a___ = tt.load %a_ptr : tensor<128x32x!tt.ptr<f8E4M3FN>, #AL>
      %a__ = tt.fp_to_fp %a___ : tensor<128x32xf8E4M3FN, #AL> -> tensor<128x32xf16, #AL>
      %a_ = ttg.convert_layout %a__ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>
      %b___ = tt.load %b_ptr, %b_mask, %b_other : tensor<128x32x!tt.ptr<f8E4M3FN>, #BL>
      %b__ = tt.fp_to_fp %b___ : tensor<128x32xf8E4M3FN, #BL> -> tensor<128x32xf16, #BL>
      %b_ = ttg.convert_layout %b__ : tensor<128x32xf16, #BL> -> tensor<128x32xf16, #B>

      %a = ttg.local_alloc %a_ {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #A>) -> !ttg.memdesc<128x32xf16, #shared, #smem>
      %b = ttg.local_alloc %b_ {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<128x32xf16, #B>) -> !ttg.memdesc<128x32xf16, #shared, #smem>
      %bt = ttg.memdesc_trans %b {loop.cluster = 1 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x32xf16, #shared, #smem> -> !ttg.memdesc<32x128xf16, #shared1, #smem>
      %acc_tm, %acc_tok = ttng.tmem_alloc %prev_c : (tensor<128x128xf32, #C>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %mma_tok = ttng.tc_gen5_mma %a, %bt, %acc_tm[%acc_tok], %true, %true : !ttg.memdesc<128x32xf16, #shared, #smem>, !ttg.memdesc<32x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %c, %load_tok = ttng.tmem_load %acc_tm[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #C>

      %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f8E4M3FN>, #AL>, tensor<128x32xi32, #AL>
      %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<128x32x!tt.ptr<f8E4M3FN>, #BL>, tensor<128x32xi32, #BL>
      scf.yield %next_a_ptr, %next_b_ptr, %c : tensor<128x32x!tt.ptr<f8E4M3FN>, #AL>, tensor<128x32x!tt.ptr<f8E4M3FN>, #BL>, tensor<128x128xf32, #C>
    }
    tt.return %loop#2: tensor<128x128xf32, #C>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#nvmma_64 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#nvmma_128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @pipelined_gather
// CHECK-SAME: [[LHS_DESC:%arg[0-9]+]]:
// CHECK-SAME: [[RHS_DESC:%arg[0-9]+]]:
// CHECK-SAME: [[LHS_X:%arg[0-9]+]]:
// CHECK-SAME: [[RHS_X:%arg[0-9]+]]:
tt.func private @pipelined_gather(
    %lhs_desc: !tt.tensordesc<tensor<1x128xbf16, #nvmma_128>>,
    %rhs_desc: !tt.tensordesc<tensor<1x32xbf16, #nvmma_64>>,
    %lhs_x_offsets: tensor<32xi32, #blocked1>,
    %rhs_x_offsets: tensor<128xi32, #blocked1>) -> tensor<32x32xf32, #blocked> {
  %c0_i32 = arith.constant 0 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1024_i32 = arith.constant 1024 : i32

  %c0 = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>

  // CHECK: [[LHS_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xbf16,
  // CHECK: [[RHS_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xbf16,
  // CHECK: [[BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64,

  // CHECK-COUNT-2: ttng.init_barrier

  // CHECK: [[BAR0:%.*]] = ttg.memdesc_index [[BARS]]{{\[}}%c0_i32{{\]}}
  // CHECK: ttng.barrier_expect [[BAR0]], 16384
  // CHECK: [[LHS_BUF0:%.*]] = ttg.memdesc_index [[LHS_BUFS]]{{\[}}%c0_i32{{\]}}
  // CHECK: ttng.async_tma_gather [[LHS_DESC]][[[LHS_X]], %c0_i32] [[LHS_BUF0]], [[BAR0]], %true
  // CHECK: [[RHS_BUF0:%.*]] = ttg.memdesc_index [[RHS_BUFS]]{{\[}}%c0_i32{{\]}}
  // CHECK: ttng.async_tma_gather [[RHS_DESC]][[[RHS_X]], %c0_i32] [[RHS_BUF0]], [[BAR0]], %true

  // CHECK: [[BAR1:%.*]] = ttg.memdesc_index [[BARS]]{{\[}}%c1_i32{{\]}}
  // CHECK: ttng.barrier_expect [[BAR1]], 16384
  // CHECK: [[LHS_BUF1:%.*]] = ttg.memdesc_index [[LHS_BUFS]]{{\[}}%c1_i32{{\]}}
  // CHECK: ttng.async_tma_gather [[LHS_DESC]][[[LHS_X]], %c128_i32] [[LHS_BUF1]], [[BAR1]], %true
  // CHECK: [[RHS_BUF1:%.*]] = ttg.memdesc_index [[RHS_BUFS]]{{\[}}%c1_i32{{\]}}
  // CHECK: ttng.async_tma_gather [[RHS_DESC]][[[RHS_X]], %c128_i32] [[RHS_BUF1]], [[BAR1]], %true

  // CHECK: scf.for
  %out = scf.for %y = %c0_i32 to %c1024_i32 step %c128_i32 iter_args(%acc = %c0) -> (tensor<32x32xf32, #mma>)  : i32 {
    // CHECK: ttng.wait_barrier
    // CHECK: [[RHS_VIEW:%.*]] = ttg.memdesc_index [[RHS_BUFS]]
    // CHECK: [[RHS:%.*]] = ttg.local_load [[RHS_VIEW]]
    // CHECK: [[LHS_VIEW:%.*]] = ttg.memdesc_index [[LHS_BUFS]]
    // CHECK: [[LHS:%.*]] = ttg.local_load [[LHS_VIEW]]
    // CHECK: tt.dot [[LHS]], [[RHS]]
    %lhs = tt.descriptor_gather %lhs_desc[%lhs_x_offsets, %y] : (!tt.tensordesc<tensor<1x128xbf16, #nvmma_128>>, tensor<32xi32, #blocked1>, i32) -> tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %rhs = tt.descriptor_gather %rhs_desc[%rhs_x_offsets, %y] : (!tt.tensordesc<tensor<1x32xbf16, #nvmma_64>>, tensor<128xi32, #blocked1>, i32) -> tensor<128x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %next = tt.dot %lhs, %rhs, %acc : tensor<32x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> *
                                      tensor<128x32xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
                                   -> tensor<32x32xf32, #mma>


    // CHECK-COUNT-2: async_tma_gather
    scf.yield %next : tensor<32x32xf32, #mma>
  }
  %out_cvt = ttg.convert_layout %out : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
  tt.return %out_cvt : tensor<32x32xf32, #blocked>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 4, 8, 1, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 1, 2, 3, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[32, 0], [64, 0], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#scales = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @block_scale_mxfp_matmul(%lb : index, %ub : index, %step : index, %arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #blocked4> {
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x128x256xf8E5M2
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x256x128xf8E5M2
    // Do not multibuffer the scale loads, as we cannot pipeline the mma due to tmem.cp not being used
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x1x2x32x4x4xi8
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x1x2x32x4x4xi8

    %true = arith.constant true
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked4>
    %incr_A = arith.constant dense<4> : tensor<128x256xi32, #blocked>
    %incr_B = arith.constant dense<4> : tensor<256x128xi32, #blocked1>
    %incr_scale = arith.constant dense<4> : tensor<1x2x32x4x4xi32, #blocked2>

    %arg0_splat = tt.splat %arg0: !tt.ptr<f8E5M2> -> tensor<128x256x!tt.ptr<f8E5M2>, #blocked>
    %arg1_splat = tt.splat %arg1: !tt.ptr<f8E5M2> -> tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>
    %arg3_splat = tt.splat %arg3: !tt.ptr<i8> -> tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %arg4_splat = tt.splat %arg4: !tt.ptr<i8> -> tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>

    %76 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %77 = tt.expand_dims %76 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %79 = tt.broadcast %77 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %arg0_init = tt.addptr %arg0_splat, %79 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256xi32, #blocked>

    %83 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %84 = tt.expand_dims %83 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %88 = tt.broadcast %84 : tensor<1x128xi32, #blocked1> -> tensor<256x128xi32, #blocked1>
    %arg1_init = tt.addptr %arg1_splat, %88 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x128xi32, #blocked1>

    %44 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>}>>
    %46 = tt.expand_dims %44 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>}>> -> tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>>
    %48 = tt.expand_dims %46 {axis = 1 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>> -> tensor<1x1x4xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>>
    %50 = tt.expand_dims %48 {axis = 2 : i32} : tensor<1x1x4xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>> -> tensor<1x1x1x4xi32, #ttg.slice<{dim = 3, parent = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>}>>
    %56 = tt.expand_dims %50 {axis = 3 : i32} : tensor<1x1x1x4xi32, #ttg.slice<{dim = 3, parent = #blocked2}>> -> tensor<1x1x1x1x4xi32, #blocked2>
    %57 = tt.broadcast %56 : tensor<1x1x1x1x4xi32, #blocked2> -> tensor<1x2x32x4x4xi32, #blocked2>

    %arg3_init = tt.addptr %arg3_splat, %57 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>
    %arg4_init = tt.addptr %arg4_splat, %57 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>

    %99:5 = scf.for %iv = %lb to %ub step %step iter_args(%arg15 = %cst_1, %arg16 = %arg0_init, %arg17 = %arg1_init, %arg18 = %arg3_init, %arg19 = %arg4_init) -> (tensor<128x128xf32, #blocked4>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>) {
      %117 = tt.load %arg16 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>
      %118 = ttg.local_alloc %117 : (tensor<128x256xf8E5M2, #blocked>) -> !ttg.memdesc<128x256xf8E5M2, #shared, #ttg.shared_memory>
      %119 = tt.load %arg17 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>
      %120 = ttg.local_alloc %119 : (tensor<256x128xf8E5M2, #blocked1>) -> !ttg.memdesc<256x128xf8E5M2, #shared, #ttg.shared_memory>
      %121 = tt.load %arg18 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
      %122 = tt.load %arg19 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>

      %137 = ttg.local_alloc %121 : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
      %138 = ttg.local_load %137 : !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem> -> tensor<1x2x32x4x4xi8, #blocked2>
      %123 = tt.trans %138 {order = array<i32: 0, 3, 2, 1, 4>} : tensor<1x2x32x4x4xi8, #blocked2> -> tensor<1x4x32x2x4xi8, #blocked3>
      %124 = tt.reshape %123 : tensor<1x4x32x2x4xi8, #blocked3> -> tensor<128x8xi8, #linear>

      %139 = ttg.local_alloc %122 : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
      %140 = ttg.local_load %139 : !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem> -> tensor<1x2x32x4x4xi8, #blocked2>
      %125 = tt.trans %140 {order = array<i32: 0, 3, 2, 1, 4>} : tensor<1x2x32x4x4xi8, #blocked2> -> tensor<1x4x32x2x4xi8, #blocked3>
      %126 = tt.reshape %125 : tensor<1x4x32x2x4xi8, #blocked3> -> tensor<128x8xi8, #linear>

      %127, %acc_tok = ttng.tmem_alloc %arg15 : (tensor<128x128xf32, #blocked4>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %128 = ttg.convert_layout %124 : tensor<128x8xi8, #linear> -> tensor<128x8xi8, #scales>
      %129 = ttg.convert_layout %126 : tensor<128x8xi8, #linear> -> tensor<128x8xi8, #scales>
      %130 = ttng.tmem_alloc %128 : (tensor<128x8xi8, #scales>) -> !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>
      %131 = ttng.tmem_alloc %129 : (tensor<128x8xi8, #scales>) -> !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>
      %mma_tok = ttng.tc_gen5_mma_scaled %118, %120, %127[%acc_tok], %130, %131, %true, %true lhs = e5m2 rhs = e5m2 : !ttg.memdesc<128x256xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<256x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>
      %132, %load_tok = ttng.tmem_load %127[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked4>

      %133 = tt.addptr %arg16, %incr_A : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256xi32, #blocked>
      %134 = tt.addptr %arg17, %incr_B : tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x128xi32, #blocked1>
      %135 = tt.addptr %arg18, %incr_scale : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>
      %136 = tt.addptr %arg19, %incr_scale : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>
      scf.yield %132, %133, %134, %135, %136 : tensor<128x128xf32, #blocked4>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    } {tt.num_stages = 3 : i32}
     tt.return %99#0 : tensor<128x128xf32, #blocked4>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[32, 0], [64, 0], [1, 0], [2, 0], [4, 0]], warp = [[8, 0], [16, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @block_scale_mxfp_matmul_tmem_copy(%lb : index, %ub : index, %step : index, %arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<128x128xf32, #blocked4> {
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x128x256xf8E5M2
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x256x128xf8E5M2
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x1x2x32x4x4xi8
    // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x1x2x32x4x4xi8
    %false = arith.constant false
    %true = arith.constant true
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked4>
    %incr_A = arith.constant dense<4> : tensor<128x256xi32, #blocked>
    %incr_B = arith.constant dense<4> : tensor<256x128xi32, #blocked1>
    %incr_scale = arith.constant dense<4> : tensor<1x2x32x4x4xi32, #blocked2>

    %arg0_splat = tt.splat %arg0: !tt.ptr<f8E5M2> -> tensor<128x256x!tt.ptr<f8E5M2>, #blocked>
    %arg1_splat = tt.splat %arg1: !tt.ptr<f8E5M2> -> tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>
    %arg3_splat = tt.splat %arg3: !tt.ptr<i8> -> tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
    %arg4_splat = tt.splat %arg4: !tt.ptr<i8> -> tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>

    %76 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %77 = tt.expand_dims %76 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked>
    %79 = tt.broadcast %77 : tensor<1x256xi32, #blocked> -> tensor<128x256xi32, #blocked>
    %arg0_init = tt.addptr %arg0_splat, %79 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256xi32, #blocked>

    %83 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %84 = tt.expand_dims %83 {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x128xi32, #blocked1>
    %88 = tt.broadcast %84 : tensor<1x128xi32, #blocked1> -> tensor<256x128xi32, #blocked1>
    %arg1_init = tt.addptr %arg1_splat, %88 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x128xi32, #blocked1>

    %44 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>}>>
    %46 = tt.expand_dims %44 {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>}>> -> tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>>
    %48 = tt.expand_dims %46 {axis = 1 : i32} : tensor<1x4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>}>> -> tensor<1x1x4xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>>
    %50 = tt.expand_dims %48 {axis = 2 : i32} : tensor<1x1x4xi32, #ttg.slice<{dim = 2, parent = #ttg.slice<{dim = 3, parent = #blocked2}>}>> -> tensor<1x1x1x4xi32, #ttg.slice<{dim = 3, parent = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 4], threadsPerWarp = [1, 1, 8, 4, 1], warpsPerCTA = [1, 1, 4, 1, 1], order = [4, 3, 2, 1, 0]}>}>>
    %56 = tt.expand_dims %50 {axis = 3 : i32} : tensor<1x1x1x4xi32, #ttg.slice<{dim = 3, parent = #blocked2}>> -> tensor<1x1x1x1x4xi32, #blocked2>
    %57 = tt.broadcast %56 : tensor<1x1x1x1x4xi32, #blocked2> -> tensor<1x2x32x4x4xi32, #blocked2>

    %arg3_init = tt.addptr %arg3_splat, %57 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>
    %arg4_init = tt.addptr %arg4_splat, %57 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>

    %99:6 = scf.for %iv = %lb to %ub step %step iter_args(%arg15 = %cst_1, %arg16 = %arg0_init, %arg17 = %arg1_init, %arg18 = %arg3_init, %arg19 = %arg4_init, %init_flag=%false) -> (tensor<128x128xf32, #blocked4>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, i1) {
      %117 = tt.load %arg16 : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>
      %118 = ttg.local_alloc %117 : (tensor<128x256xf8E5M2, #blocked>) -> !ttg.memdesc<128x256xf8E5M2, #shared, #ttg.shared_memory>
      %119 = tt.load %arg17 : tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>
      %120 = ttg.local_alloc %119 : (tensor<256x128xf8E5M2, #blocked1>) -> !ttg.memdesc<256x128xf8E5M2, #shared, #ttg.shared_memory>
      %121 = tt.load %arg18 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>
      %122 = tt.load %arg19 : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>

      %137 = ttg.local_alloc %121 : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
      %139 = ttg.local_alloc %122 : (tensor<1x2x32x4x4xi8, #blocked2>) -> !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>

      %127, %acc_tok = ttng.tmem_alloc %arg15 : (tensor<128x128xf32, #blocked4>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

      // CHECK: tc_gen5_mma_scaled %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %true, %{{.*}}
      %mma_tok = ttng.tc_gen5_mma_scaled %118, %120, %127[%acc_tok], %137, %139, %init_flag, %true lhs = e5m2 rhs = e5m2 : !ttg.memdesc<128x256xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<256x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>, !ttg.memdesc<1x2x32x4x4xi8, #shared1, #smem>
      %132, %load_tok = ttng.tmem_load %127[%mma_tok] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked4>

      %133 = tt.addptr %arg16, %incr_A : tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<128x256xi32, #blocked>
      %134 = tt.addptr %arg17, %incr_B : tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<256x128xi32, #blocked1>
      %135 = tt.addptr %arg18, %incr_scale : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>
      %136 = tt.addptr %arg19, %incr_scale : tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4xi32, #blocked2>
      scf.yield %132, %133, %134, %135, %136, %true : tensor<128x128xf32, #blocked4>, tensor<128x256x!tt.ptr<f8E5M2>, #blocked>, tensor<256x128x!tt.ptr<f8E5M2>, #blocked1>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, tensor<1x2x32x4x4x!tt.ptr<i8>, #blocked2>, i1
    } {tt.num_stages = 3 : i32}
     tt.return %99#0 : tensor<128x128xf32, #blocked4>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#load_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#scales = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_T = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>

#smem = #ttg.shared_memory
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem_lhs = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @load_into_async_mma
tt.func public @load_into_async_mma(
  %lhs_ptrs: tensor<128x64x!tt.ptr<f8E4M3FN>, #load_blocked>,
  %scale_ptrs: tensor<128x8x!tt.ptr<i8>, #load_blocked>,
  %tmem: !ttg.memdesc<128x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
  %barrier: !ttg.memdesc<1xi64, #shared, #smem, mutable>,
  %rhs_shared: !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>,
  %n_tiles: i32
) {
  %true = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32

  %cst = arith.constant dense<0> : tensor<64x8xi8, #scales>
  %rhs_scales = ttng.tmem_alloc %cst : (tensor<64x8xi8, #scales>) -> !ttg.memdesc<64x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>

  // CHECK-COUNT-6: ttg.async_copy_global_to_local
  scf.for %i = %c0_i32 to %n_tiles step %c64_i32 : i32 {
    %lhs_offs = tt.splat %i : i32 -> tensor<128x64xi32, #load_blocked>
    %lhs_ptrs_i = tt.addptr %lhs_ptrs, %lhs_offs {tt.divisibility = dense<16> : tensor<128x64xi32>, tt.contiguity = dense<32> : tensor<128x64xi32>, tt.constancy = dense<1> : tensor<128x64xi32>} : tensor<128x64x!tt.ptr<f8E4M3FN>, #load_blocked>, tensor<128x64xi32, #load_blocked>
    %lhs = tt.load %lhs_ptrs_i : tensor<128x64x!tt.ptr<f8E4M3FN>, #load_blocked>
    %lhs_shared = ttg.local_alloc %lhs : (tensor<128x64xf8E4M3FN, #load_blocked>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem>

    %scales_offs = tt.splat %i : i32 -> tensor<128x8xi32, #load_blocked>
    %scales_ptrs_i = tt.addptr %scale_ptrs, %scales_offs {tt.divisibility = dense<16> : tensor<128x8xi32>, tt.contiguity = dense<32> : tensor<128x8xi32>, tt.constancy = dense<1> : tensor<128x8xi32>} : tensor<128x8x!tt.ptr<i8>, #load_blocked>, tensor<128x8xi32, #load_blocked>
    %scales = tt.load %scales_ptrs_i : tensor<128x8x!tt.ptr<i8>, #load_blocked>
    %scales_cvt = ttg.convert_layout %scales : tensor<128x8xi8, #load_blocked> -> tensor<128x8xi8, #scales>
    %scales_tmem = ttng.tmem_alloc %scales_cvt : (tensor<128x8xi8, #scales>) -> !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>

    ttng.tc_gen5_mma_scaled %lhs_shared, %rhs_shared, %tmem, %scales_tmem, %rhs_scales, %true, %true lhs = e4m3 rhs = e4m3, %barrier[%true] {is_async} :
      !ttg.memdesc<128x64xf8E4M3FN, #shared, #smem>,
      !ttg.memdesc<64x64xf8E4M3FN, #shared, #smem>,
      !ttg.memdesc<128x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>,
      !ttg.memdesc<64x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>,
      !ttg.memdesc<1xi64, #shared, #smem, mutable>
  }

  tt.return
}

}
