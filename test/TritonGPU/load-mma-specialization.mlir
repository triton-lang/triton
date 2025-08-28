// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritongpu-hoist-tmem-alloc | FileCheck %s --check-prefix=TMEM --check-prefix=FUNC
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -verify-diagnostics --tritongpu-hoist-tmem-alloc -tritongpu-partition-scheduling -tritongpu-load-mma-specialization -sccp -int-range-optimizations -canonicalize -cse -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -verify-diagnostics --tritongpu-hoist-tmem-alloc -tritongpu-automatic-warp-specialization | FileCheck %s --check-prefix=AWS --check-prefix=FUNC

#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#oper_layout_trans = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
// CHECK-DAG: [[SHARED:#.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_trans = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#nvmma_smem = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#scales = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
// CHECK-DAG: [[ACC_TMEM:#.*]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

#lhs_layout = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#lhs_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>

#fp4_padded_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// FUNC-LABEL: @warp_specialize_tma_matmul

// TMEM: ttng.tmem_alloc
// TMEM: scf.for

// AWS: ttg.warp_specialize
// AWS: num_warps(1)
// AWS: num_warps(2)
// AWS-NOT: num_warps(

// CHECK: @warp_specialize_tma_matmul
// CHECK-SAME: [[K_TILES:%arg[0-9]+]]
// CHECK-SAME: [[OFF_M:%arg[0-9]+]]
// CHECK-SAME: [[OFF_N:%arg[0-9]+]]
// CHECK-SAME: [[A_DESC:%arg[0-9]+]]
// CHECK-SAME: [[B_DESC:%arg[0-9]+]]
tt.func @warp_specialize_tma_matmul(
  %k_tiles: i32,
  %off_m: i32,
  %off_n: i32,
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>
) {
  // CHECK-DAG: [[TRUE:%.*]] = arith.constant true
  %true = arith.constant true
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : i32
  %c0_i32 = arith.constant 0 : i32
  // CHECK-DAG: [[C1:%.*]] = arith.constant 1 : i32
  %c1_i32 = arith.constant 1 : i32

  // CHECK-DAG: [[BLOCK_K:%.*]] = arith.constant 64 : i32
  %BLOCK_K = arith.constant 64 : i32
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant dense<0.0
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  // CHECK-DAG: [[C2:%.*]] = arith.constant 2 : i32

  // CHECK:      [[ACC_BUFS:%.*]], [[ACC_TOK:.*]] = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32, [[ACC_TMEM]], #ttng.tensor_memory, mutable>
  // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}[[C0]]{{\]}}
  // CHECK-NEXT: [[INIT_TOK:%.*]] = ttng.tmem_store [[ZERO]], [[ACC_BUF]][[[ACC_TOK]]]

  // CHECK-NEXT: [[A_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, [[SHARED]]
  // CHECK-NEXT: [[B_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, [[SHARED]]

  // CHECK-NEXT: [[READY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK-NEXT: [[READY_MBAR0:%.*]] = ttg.memdesc_index [[READY_MBARS]]{{\[}}[[C0]]{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[READY_MBAR0]], 1
  // CHECK-NEXT: [[READY_MBAR1:%.*]] = ttg.memdesc_index [[READY_MBARS]]{{\[}}[[C1]]{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[READY_MBAR1]], 1

  // CHECK-NEXT: [[OPER_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK-NEXT: [[OPER_MBAR0:%.*]] = ttg.memdesc_index [[OPER_MBARS]]{{\[}}[[C0]]{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[OPER_MBAR0]], 1
  // CHECK-NEXT: [[OPER_MBAR1:%.*]] = ttg.memdesc_index [[OPER_MBARS]]{{\[}}[[C1]]{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[OPER_MBAR1]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[READY_MBAR0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[READY_MBAR1]], 1

  // CHECK-NEXT: [[LAST_ITER:%.*]] = arith.subi [[K_TILES]], [[C1]]

  // CHECK-NEXT: [[DONE_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
  // CHECK-NEXT: [[DONE_MBAR0:%.*]] = ttg.memdesc_index [[DONE_MBAR]]{{\[}}[[C0]]{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[DONE_MBAR0]], 1

  // CHECK-NEXT: [[LAST:%.*]]:3 = scf.for [[K:%arg[0-9]+]] = [[C0]] to [[K_TILES]] step [[C1]]
  // CHECK-SAME: [[TOK:%arg[0-9]+]] = [[INIT_TOK]]
  // CHECK-SAME: [[IDX:%arg[0-9]+]] = [[C0]]
  // CHECK-SAME: [[PHASE:%arg[0-9]+]] = [[C0]]
  // CHECK-SAME: -> (!ttg.async.token, i32, i32)
  %result = scf.for %k = %c0_i32 to %k_tiles step %c1_i32
      iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK-NEXT: [[OFF_K:%.*]] = arith.muli [[K]], [[BLOCK_K]]
    %off_k = arith.muli %k, %BLOCK_K : i32

    // CHECK-NEXT: [[READY_MBAR:%.*]] = ttg.memdesc_index [[READY_MBARS]]{{\[}}[[IDX]]{{\]}}
    // CHECK-NEXT: ttng.wait_barrier [[READY_MBAR]], [[PHASE]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[OPER_MBAR:%.*]] = ttg.memdesc_index [[OPER_MBARS]]{{\[}}[[IDX]]{{\]}}
    // CHECK-NEXT: ttng.barrier_expect [[OPER_MBAR]], 32768 {ttg.partition = 2 : i32}

    // CHECK-NEXT: [[A_BUF:%.*]] = ttg.memdesc_index [[A_BUFS]]{{\[}}[[IDX]]{{\]}}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local [[A_DESC]][[[OFF_M]], [[OFF_K]]] [[A_BUF]], [[OPER_MBAR]], [[TRUE]] {ttg.partition = 2 : i32}
    %a_reg = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    // CHECK-NEXT: [[B_BUF:%.*]] = ttg.memdesc_index [[B_BUFS]]{{\[}}[[IDX]]{{\]}}
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local [[B_DESC]][[[OFF_N]], [[OFF_K]]] [[B_BUF]], [[OPER_MBAR]], [[TRUE]] {ttg.partition = 2 : i32}
    %b_reg = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>

    %a_shared = ttg.local_alloc %a_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    // CHECK-NEXT: [[B_T:%.*]] = ttg.memdesc_trans [[B_BUF]] {order = array<i32: 1, 0>, ttg.partition = 1 : i32}
    // CHECK-NEXT: ttng.wait_barrier [[OPER_MBAR]], [[PHASE]] {ttg.partition = 1 : i32}
    %b_T_shared = ttg.memdesc_trans %b_shared {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared_trans, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK-NEXT: [[IS_LAST:%.*]] = arith.cmpi eq, [[K]], [[LAST_ITER]]
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma [[A_BUF]], [[B_T]], [[ACC_BUF]][], [[TRUE]], [[TRUE]], [[READY_MBAR]][%true], [[DONE_MBAR0]][[[IS_LAST]]] {is_async, ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_T_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared_trans, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>

    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: [[IDX_INCR:%.*]] = arith.addi [[IDX]], [[C1]]
    // CHECK-NEXT: [[PHASE_INCR:%.*]] = arith.xori [[PHASE]], [[C1]]
    // CHECK-NEXT: [[ROLLOVER:%.*]] = arith.cmpi eq, [[IDX_INCR]], [[C2]]
    // CHECK-NEXT: [[IDX_NEXT:%.*]] = arith.select [[ROLLOVER]], [[C0]], [[IDX_INCR]]
    // CHECK-NEXT: [[PHASE_NEXT:%.*]] = arith.select [[ROLLOVER]], [[PHASE_INCR]], [[PHASE]]

    // CHECK-NEXT: yield %{{[0-9]+}}, [[IDX_NEXT]], [[PHASE_NEXT]]
    scf.yield %c : tensor<128x128xf32, #acc_layout>

  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  // CHECK-NEXT: ttng.wait_barrier [[DONE_MBAR0]], %c0_i32
  // CHECK-NEXT: ttng.inval_barrier [[DONE_MBAR0]]
  // CHECK-NEXT: ttg.local_dealloc [[DONE_MBAR]]

  // CHECK-NEXT: ttng.inval_barrier [[OPER_MBAR0]]
  // CHECK-NEXT: ttng.inval_barrier [[OPER_MBAR1]]
  // CHECK-NEXT: ttg.local_dealloc [[OPER_MBARS]]

  // CHECK-NEXT: ttng.inval_barrier [[READY_MBAR0]]
  // CHECK-NEXT: ttng.inval_barrier [[READY_MBAR1]]
  // CHECK-NEXT: ttg.local_dealloc [[READY_MBARS]]

  // CHECK-NEXT: ttg.local_dealloc [[B_BUFS]]
  // CHECK-NEXT: ttg.local_dealloc [[A_BUFS]]

  // CHECK-NEXT: [[RESULT:%.*]], [[RESULT_TOK:%.*]] = ttng.tmem_load [[ACC_BUF]][[[LAST]]#0]
  // CHECK-NEXT: "use"([[RESULT]])
  "use"(%result) : (tensor<128x128xf32, #acc_layout>) -> ()
  tt.return
}
// FUNC-LABEL: @unsupported_load
// TMEM: ttng.tmem_alloc
// TMEM: scf.for

// CHECK-LABEL: @unsupported_load
tt.func @unsupported_load() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant dense<0.0
  // CHECK-DAG: [[C0:%.*]] = arith.constant 0 : i32
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK: [[ACC_ALLOC:%.*]], %{{.*}} = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32
  // CHECK-NEXT: [[ACC:%.*]] = ttg.memdesc_index [[ACC_ALLOC]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: tmem_store [[ZERO]], [[ACC]]

  // CHECK-NEXT: [[DONE_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
  // CHECK-NEXT: [[DONE_MBAR0:%.*]] = ttg.memdesc_index [[DONE_MBAR]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[DONE_MBAR0]], 1

  // CHECK-NEXT: scf.for
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK-NEXT: get_ptrs
    %a_ptrs, %b_ptrs = "get_ptrs"(%k) : (i32) -> (tensor<128x64x!tt.ptr<f16>, #oper_layout>, tensor<64x128x!tt.ptr<f16>, #oper_layout>)
    %a = tt.load %a_ptrs : tensor<128x64x!tt.ptr<f16>, #oper_layout>
    %b = tt.load %b_ptrs : tensor<64x128x!tt.ptr<f16>, #oper_layout>

    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: [[IS_LAST:%.*]] = arith.cmpi eq, %{{.*}}, %c31_i32
    // CHECK-NEXT: ttng.tc_gen5_mma %{{.*}}, [[ACC]][], %true, %true, [[DONE_MBAR0]][[[IS_LAST]]] {is_async, ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    scf.yield %c : tensor<128x128xf32, #acc_layout>
  // CHECK: ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 1 : i32
  } {tt.warp_specialize}

  // CHECK-NEXT: ttng.wait_barrier [[DONE_MBAR0]], %c0_i32
  // CHECK-NEXT: ttng.inval_barrier [[DONE_MBAR0]]
  // CHECK-NEXT: ttg.local_dealloc [[DONE_MBAR]]

  tt.return
}

// FUNC-LABEL: @cant_pipeline_mma
// TMEM: ttng.tmem_alloc
// TMEM: scf.for

// CHECK-LABEL: @cant_pipeline_mma
tt.func @cant_pipeline_mma(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<3x{{.*}}xf16,
  // CHECK-COUNT-3: ttng.arrive_barrier
  // CHECK-NOT: ttng.arrive_barrier

  // CHECK: scf.for
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 : i32 {
    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)
    %a = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

    %c_tmem, %c_tok = ttng.tmem_alloc %zero : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
  } {tt.warp_specialize}

  tt.return
}

// FUNC-LABEL: @invalid_acc_reset
// TMEM: ttng.tmem_alloc
// TMEM: scf.for

// CHECK-LABEL: @invalid_acc_reset
tt.func @invalid_acc_reset(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<3x{{.*}}xf16,
  // CHECK-COUNT-3: ttng.arrive_barrier
  // CHECK-NOT: ttng.arrive_barrier

  // CHECK: scf.for
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)
    %a = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

    %c_tmem, %c_tok = ttng.tmem_alloc %zero : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    scf.yield %c : tensor<128x128xf32, #acc_layout>
  } {tt.warp_specialize}

  tt.return
}

// FUNC-LABEL: @matmul_tma_acc_with_unconditional_user

// TMEM: ttng.tmem_alloc
// TMEM: scf.for

// AWS: ttg.warp_specialize
// AWS: num_warps(4)
// AWS: num_warps(2)
// AWS-NOT: num_warps(

// CHECK-LABEL: @matmul_tma_acc_with_unconditional_user
// CHECK-SAME: [[A_DESC:%arg[0-9]+]]
// CHECK-SAME: [[B_DESC:%arg[0-9]+]]
tt.func @matmul_tma_acc_with_unconditional_user(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant dense<0.0
  // CHECK-DAG: [[ACC_RESET:%.*]] = arith.constant dense<1.0
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %acc_reset = arith.constant dense<1.0> : tensor<128x128xf32, #acc_layout>
  // CHECK-DAG: [[K_TILES:%.*]] = arith.constant 32 : i32
  %k_tiles = arith.constant 32 : i32

  // CHECK:      [[ACC_BUFS:%.*]], [[ACC_TOK:%.*]] = ttng.tmem_alloc : () -> (!ttg.memdesc<2x128x128xf32
  // CHECK-NEXT: [[ACC_BUF0:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: [[INIT_TOK:%.*]] = ttng.tmem_store [[ZERO]], [[ACC_BUF0]][[[ACC_TOK]]]

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64

  // CHECK:      [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK-NEXT: [[ACC_READY_BUF0:%.*]] = ttg.memdesc_index [[ACC_READY_BUFS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF0]], 1
  // CHECK-NEXT: [[ACC_READY_BUF1:%.*]] = ttg.memdesc_index [[ACC_READY_BUFS]]{{\[}}%c1_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF1]], 1

  // CHECK-NEXT: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK-NEXT: [[ACC_EMPTY_BUF0:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[ACC_EMPTY_BUF0]], 1
  // CHECK-NEXT: [[ACC_EMPTY_BUF1:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}%c1_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[ACC_EMPTY_BUF1]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[ACC_EMPTY_BUF0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[ACC_EMPTY_BUF1]], 1

  // CHECK-NEXT: {{[0-9]+}}:4 = scf.for [[K:%arg[0-9]+]] = %c0_i32 to [[K_TILES]] step %c1_i32
  // CHECK-SAME: [[LOAD_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LOAD_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK-NEXT: [[OFFS:%.*]]:3 = "get_offsets"([[K]])
    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)

    // CHECK: ttng.wait_barrier
    // CHECK: ttng.barrier_expect
    // CHECK-COUNT-2: ttng.async_tma_copy_global_to_local
    %a = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    // CHECK: ttng.wait_barrier
    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}

    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_index [[ACC_READY_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], %true, %true, {{.*}}, [[CUR_ACC_READY_BAR]][%true] {is_async, ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>

    // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BAR]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[C:%.*]], [[LOAD_TOK:%.*]] = ttng.tmem_load [[ACC_BUF]][] {ttg.partition = 0 : i32}
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}
    // CHECK-NEXT: ttng.arrive_barrier [[CUR_ACC_EMPTY_BAR]], 1 {ttg.partition = 0 : i32}
    "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()

    // CHECK-NEXT: [[ACC_INDEX_INCR:%.*]] = arith.addi [[ACC_INDEX]], %c1_i32
    // CHECK-NEXT: [[ACC_PHASE_INCR:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_ROLLVER:%.*]] = arith.cmpi eq, [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[ACC_NEXT_INDEX:%.*]] = arith.select [[ACC_ROLLVER]], %c0_i32, [[ACC_INDEX_INCR]]
    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.select [[ACC_ROLLVER]], [[ACC_PHASE_INCR]], [[ACC_PHASE]]

    // CHECK-NEXT: "acc_user"([[C]]) {ttg.partition = 0 : i32}

    // CHECK-NEXT: [[NEXT_ACC_BUF:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}[[ACC_NEXT_INDEX]]{{\]}}
    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}[[ACC_NEXT_INDEX]]{{\]}}
    // CHECK-NEXT: ttng.wait_barrier [[NEXT_ACC_EMPTY_BAR]], [[ACC_NEXT_PHASE]], %true {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[STORE_TOK:%.*]] = ttng.tmem_store [[ACC_RESET]], [[NEXT_ACC_BUF]][], %true {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_NEXT_INDEX]], [[ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32]
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  tt.return
}

// FUNC-LABEL: @matmul_tma_acc_with_conditional_user

// TMEM: ttng.tmem_alloc
// TMEM: scf.for

// AWS: ttg.warp_specialize
// AWS: num_warps(4)
// AWS: num_warps(2)
// AWS-NOT: num_warps(

// CHECK-LABEL: @matmul_tma_acc_with_conditional_user
// CHECK-SAME: [[A_DESC:%arg[0-9]+]]
// CHECK-SAME: [[B_DESC:%arg[0-9]+]]
tt.func @matmul_tma_acc_with_conditional_user(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  // CHECK-DAG: [[ACC_RESET:%.*]] = arith.constant dense<1.0
  %acc_reset = arith.constant dense<1.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK: [[ACC_BUFS:%.*]], [[ACC_TOK:%.*]] = ttng.tmem_alloc
  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64

  // CHECK: [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64

  // CHECK:      {{[0-9]+}}:4 = scf.for [[K:%arg[0-9]+]]
  // CHECK-SAME: [[LOAD_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LOAD_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK-NEXT: [[OFFS:%.*]]:3 = "get_offsets"([[K]])
    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)

    // CHECK: ttng.wait_barrier
    // CHECK: ttng.barrier_expect
    // CHECK-COUNT-2: ttng.async_tma_copy_global_to_local
    %a = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    // CHECK: ttng.wait_barrier
    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}
    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_index [[ACC_READY_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}
    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = arith.cmpi
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], %true, %true, {{.*}}, [[CUR_ACC_READY_BAR]][[[DO_EPILOGUE]]] {is_async, ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    %do_epilogue = arith.cmpi eq, %k, %c0_i32 : i32

    // CHECK-NEXT: scf.if [[DO_EPILOGUE]]
    scf.if %do_epilogue {
      // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BAR]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: [[C:%.*]], [[USER_TOK:%.*]] = ttng.tmem_load [[ACC_BUF]][]
      // CHECK-NEXT: "acc_user"([[C]])
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
      // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}
      // CHECK-NEXT: ttng.arrive_barrier [[CUR_ACC_EMPTY_BAR]], 1 {ttg.partition = 0 : i32}
    // CHECK-NEXT: }
    }

    // CHECK-NEXT: [[ACC_INDEX_INCR:%.*]] = arith.addi [[ACC_INDEX]], %c1_i32
    // CHECK-NEXT: [[ACC_PHASE_INCR:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_ROLLVER:%.*]] = arith.cmpi eq, [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[ACC_NEXT_INDEX:%.*]] = arith.select [[ACC_ROLLVER]], %c0_i32, [[ACC_INDEX_INCR]]
    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.select [[ACC_ROLLVER]], [[ACC_PHASE_INCR]], [[ACC_PHASE]]
    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_INDEX:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_INDEX]], [[ACC_INDEX]]
    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_PHASE:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_PHASE]], [[ACC_PHASE]]

    // CHECK-NEXT: [[ACC_NEXT_BUF:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}[[EPILOGUE_ACC_NEXT_INDEX]]{{\]}}
    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}[[EPILOGUE_ACC_NEXT_INDEX]]{{\]}}
    // CHECK-NEXT: ttng.wait_barrier [[NEXT_ACC_EMPTY_BAR]], [[EPILOGUE_ACC_NEXT_PHASE]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: ttng.tmem_store [[ACC_RESET]], [[ACC_NEXT_BUF]][], %true {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield %{{[0-9]+}}, %{{[0-9]+}}, [[EPILOGUE_ACC_NEXT_INDEX]], [[EPILOGUE_ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32]
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  tt.return
}

// FUNC-LABEL: @matmul_tma_acc_with_conditional_def

// TMEM: ttng.tmem_alloc
// TMEM: scf.for

// AWS: ttg.warp_specialize
// AWS: num_warps(4)
// AWS: num_warps(2)
// AWS-NOT: num_warps(

// CHECK-LABEL: @matmul_tma_acc_with_conditional_def
// CHECK-SAME: [[A_DESC:%arg[0-9]+]]
// CHECK-SAME: [[B_DESC:%arg[0-9]+]]
tt.func @matmul_tma_acc_with_conditional_def(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  // CHECK: [[ZERO:%.*]] = arith.constant dense<0.0
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK: [[ACC_BUFS:%.*]], [[ACC_TOK:%.*]] = ttng.tmem_alloc
  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64

  // CHECK: [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64

  // CHECK:      {{[0-9]+}}:4 = scf.for [[K:%arg[0-9]+]]
  // CHECK-SAME: [[LOAD_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LOAD_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {

    // CHECK-NEXT: [[OFFS:%.*]]:3 = "get_offsets"([[K]])
    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)

    // CHECK: ttng.wait_barrier
    // CHECK: ttng.barrier_expect
    // CHECK-COUNT-2: ttng.async_tma_copy_global_to_local
    %a = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    // CHECK: ttng.wait_barrier
    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}

    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_index [[ACC_READY_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], %true, %true, {{.*}}, [[CUR_ACC_READY_BAR]][%true] {is_async, ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = arith.cmpi
    %do_epilogue = arith.cmpi eq, %k, %c0_i32 : i32
    %acc_reset = arith.select %do_epilogue, %zero, %c : tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BAR]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[C:%.*]], [[LOAD_TOK:%.*]] = ttng.tmem_load [[ACC_BUF]][] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}
    // CHECK-NEXT: ttng.arrive_barrier [[CUR_ACC_EMPTY_BAR]], 1 {ttg.partition = 0 : i32}

    // CHECK-NEXT: [[ACC_INDEX_INCR:%.*]] = arith.addi [[ACC_INDEX]], %c1_i32
    // CHECK-NEXT: [[ACC_PHASE_INCR:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_ROLLVER:%.*]] = arith.cmpi eq, [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[ACC_NEXT_INDEX:%.*]] = arith.select [[ACC_ROLLVER]], %c0_i32, [[ACC_INDEX_INCR]]
    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.select [[ACC_ROLLVER]], [[ACC_PHASE_INCR]], [[ACC_PHASE]]

    // CHECK-NEXT: "acc_user"([[C]])
    "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()

    // CHECK-NEXT: [[NEXT_ACC_BUF:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}[[ACC_NEXT_INDEX]]{{\]}}
    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}[[ACC_NEXT_INDEX]]{{\]}}
    // CHECK-NEXT: ttng.wait_barrier [[NEXT_ACC_EMPTY_BAR]], [[ACC_NEXT_PHASE]], %true {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[STORE_TOK:%.*]] = ttng.tmem_store [[ZERO]], [[NEXT_ACC_BUF]][], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield {{.*}} [[ACC_NEXT_INDEX]], [[ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32]
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  tt.return
}

// FUNC-LABEL: @matmul_tma_acc_with_conditional_def_and_use

// TMEM: ttng.tmem_alloc
// TMEM: scf.for

// AWS: ttg.warp_specialize
// AWS: num_warps(4)
// AWS: num_warps(2)
// AWS-NOT: num_warps(

// CHECK-LABEL: @matmul_tma_acc_with_conditional_def_and_use
// CHECK-SAME: [[A_DESC:%arg[0-9]+]]
// CHECK-SAME: [[B_DESC:%arg[0-9]+]]
tt.func @matmul_tma_acc_with_conditional_def_and_use(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  // CHECK: [[ZERO:%.*]] = arith.constant dense<0.0
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK: [[ACC_BUFS:%.*]], [[ACC_TOK:%.*]] = ttng.tmem_alloc
  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64

  // CHECK: [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64

  // CHECK:      {{[0-9]+}}:4 = scf.for [[K:%arg[0-9]+]]
  // CHECK-SAME: [[LOAD_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LOAD_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK-NEXT: [[OFFS:%.*]]:3 = "get_offsets"([[K]])
    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)

    // CHECK: ttng.wait_barrier
    // CHECK: ttng.barrier_expect
    // CHECK-COUNT-2: ttng.async_tma_copy_global_to_local
    %a = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    // CHECK: ttng.wait_barrier
    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}

    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_index [[ACC_READY_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}
    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = arith.cmpi
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], %true, %true, {{.*}}, [[CUR_ACC_READY_BAR]][[[DO_EPILOGUE]]] {is_async, ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    %do_epilogue = arith.cmpi eq, %k, %c0_i32 : i32
    %acc_reset = arith.select %do_epilogue, %zero, %c : tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: scf.if [[DO_EPILOGUE]]
    scf.if %do_epilogue {
      // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BAR]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: [[C:%.*]], [[USER_TOK:%.*]] = ttng.tmem_load [[ACC_BUF]][]
      // CHECK-NEXT: "acc_user"([[C]])
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
      // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}
      // CHECK-NEXT: ttng.arrive_barrier [[CUR_ACC_EMPTY_BAR]], 1 {ttg.partition = 0 : i32}
    // CHECK-NEXT: }
    }

    // CHECK-NEXT: [[ACC_INDEX_INCR:%.*]] = arith.addi [[ACC_INDEX]], %c1_i32
    // CHECK-NEXT: [[ACC_PHASE_INCR:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_ROLLVER:%.*]] = arith.cmpi eq, [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[ACC_NEXT_INDEX:%.*]] = arith.select [[ACC_ROLLVER]], %c0_i32, [[ACC_INDEX_INCR]]
    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.select [[ACC_ROLLVER]], [[ACC_PHASE_INCR]], [[ACC_PHASE]]
    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_INDEX:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_INDEX]], [[ACC_INDEX]]
    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_PHASE:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_PHASE]], [[ACC_PHASE]]

    // CHECK-NEXT: [[NEXT_ACC_BUF:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}[[EPILOGUE_ACC_NEXT_INDEX]]{{\]}}
    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}[[EPILOGUE_ACC_NEXT_INDEX]]{{\]}}
    // CHECK-NEXT: ttng.wait_barrier [[NEXT_ACC_EMPTY_BAR]], [[EPILOGUE_ACC_NEXT_PHASE]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[STORE_TOK:%.*]] = ttng.tmem_store [[ZERO]], [[NEXT_ACC_BUF]][], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield {{.*}} [[EPILOGUE_ACC_NEXT_INDEX]], [[EPILOGUE_ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32]
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  tt.return
}

// FUNC-LABEL: @matmul_tma_acc_with_conditional_def_and_use_no_multibuf_flag

// TMEM: ttng.tmem_alloc
// TMEM: scf.for

// AWS: ttg.warp_specialize
// AWS: num_warps(1)
// AWS: num_warps(2)
// AWS-NOT: num_warps(

// CHECK-LABEL: @matmul_tma_acc_with_conditional_def_and_use_no_multibuf_flag
// CHECK-SAME: [[A_DESC:%arg[0-9]+]]
// CHECK-SAME: [[B_DESC:%arg[0-9]+]]
tt.func @matmul_tma_acc_with_conditional_def_and_use_no_multibuf_flag(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %false = arith.constant false
  // CHECK: [[ZERO:%.*]] = arith.constant dense<0.0
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK: [[ACC_BUFS:%.*]], [[ACC_TOK:%.*]] = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32,
  // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: [[INIT_TOK:%.*]] = ttng.tmem_store [[ZERO]], [[ACC_BUF]][[[ACC_TOK]]], %true

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64

  // CHECK:      [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
  // CHECK-NEXT: [[ACC_READY_BUF0:%.*]] = ttg.memdesc_index [[ACC_READY_BUFS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF0]], 1

  // CHECK-NEXT: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
  // CHECK-NEXT: [[ACC_EMPTY_BUF0:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[ACC_EMPTY_BUF0]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[ACC_EMPTY_BUF0]], 1

  // CHECK-NEXT: {{[0-9]+}}:4 = scf.for [[K:%arg[0-9]+]]
  // CHECK-SAME: [[FLAG:%arg[0-9]+]] = %true
  // CHECK-SAME: [[LOAD_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LOAD_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero, %flag = %true) -> (tensor<128x128xf32, #acc_layout>, i1) : i32 {
    // CHECK-NEXT: [[OFFS:%.*]]:3 = "get_offsets"([[K]])
    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)

    // CHECK: ttng.wait_barrier
    // CHECK: ttng.barrier_expect
    // CHECK-COUNT-2: ttng.async_tma_copy_global_to_local
    %a = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    // CHECK: ttng.wait_barrier
    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = arith.cmpi eq, [[K:%.*]], %c0_i32 : i32
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], [[FLAG]], %true, {{.*}}, [[ACC_READY_BUF0]][[[DO_EPILOGUE]]] {is_async, ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %flag, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    %do_epilogue = arith.cmpi eq, %k, %c0_i32 : i32
    // CHECK-NEXT: [[NEXT_FLAG:%.*]] = arith.cmpi ne, [[K]], %c0_i32

    %use_acc = arith.select %do_epilogue, %false, %true : i1

    // CHECK-NEXT: scf.if [[DO_EPILOGUE]]
    scf.if %do_epilogue {
      // CHECK-NEXT: "some_op"()
      "some_op"() : () -> ()
      // CHECK-NEXT: ttng.wait_barrier [[ACC_READY_BUF0]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: [[C:%.*]], [[USER_TOK:%.*]] = ttng.tmem_load [[ACC_BUF]][]
      // CHECK-NEXT: ttng.arrive_barrier [[ACC_EMPTY_BUF0]], 1 {ttg.partition = 0 : i32}
      // CHECK-NEXT: "acc_user"([[C]])
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
    // CHECK-NEXT: }
    }

    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_PHASE:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_PHASE]], [[ACC_PHASE]]
    // CHECK-NEXT: ttng.wait_barrier [[ACC_EMPTY_BUF0]], [[EPILOGUE_ACC_NEXT_PHASE]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield [[NEXT_FLAG]], %{{[0-9]+}}, %{{[0-9]+}}, [[EPILOGUE_ACC_NEXT_PHASE]]
    scf.yield %c, %use_acc : tensor<128x128xf32, #acc_layout>, i1
  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32]
  } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32}

  tt.return
}

// FUNC-LABEL: @matmul_scaled_rhs_scales_tma
// CHECK-LABEL: @matmul_scaled_rhs_scales_tma
tt.func @matmul_scaled_rhs_scales_tma(
  %k_tiles: i32,
  %off_m: i32,
  %off_n: i32,
  %a_desc: !tt.tensordesc<tensor<128x64xf8E4M3FN, #nvmma_smem>>,
  %b_desc: !tt.tensordesc<tensor<128x64xf8E4M3FN, #nvmma_smem>>,
  %b_scale_desc: !tt.tensordesc<tensor<128x8xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>>>
) {
  %true = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %BLOCK_K = arith.constant 64 : i32
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  %a_scales_const = arith.constant dense<127> : tensor<128x8xi8, #scales>
  %a_scales_tmem = ttng.tmem_alloc %a_scales_const : (tensor<128x8xi8, #scales>) -> !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,
  // CHECK-NOT: ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,

  // CHECK: [[LAST_ITER:%.*]] = arith.subi %{{.*}}, %c1_i32

  %result = scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    %off_k = arith.muli %k, %BLOCK_K : i32

    // CHECK: ttng.wait_barrier
    // CHECK-COUNT-3: async_tma_copy_global_to_local {{.*}} {ttg.partition = 2 : i32}
    %a_reg = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf8E4M3FN, #nvmma_smem>> -> tensor<128x64xf8E4M3FN, #oper_layout>
    %b_reg = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<128x64xf8E4M3FN, #nvmma_smem>> -> tensor<128x64xf8E4M3FN, #oper_layout>
    %b_scales_reg = tt.descriptor_load %b_scale_desc[%off_m, %c0_i32] : !tt.tensordesc<tensor<128x8xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>>> -> tensor<128x8xi8, #scales>

    %a_sh = ttg.local_alloc %a_reg : (tensor<128x64xf8E4M3FN, #oper_layout>) -> !ttg.memdesc<128x64xf8E4M3FN, #nvmma_smem, #smem>
    %b_sh_raw = ttg.local_alloc %b_reg : (tensor<128x64xf8E4M3FN, #oper_layout>) -> !ttg.memdesc<128x64xf8E4M3FN, #nvmma_smem, #smem>
    // CHECK-NEXT: memdesc_trans {{.*}} ttg.partition = 1 : i32
    %b_sh = ttg.memdesc_trans %b_sh_raw {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf8E4M3FN, #nvmma_smem, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>, #smem>

    // CHECK-NEXT: wait_barrier {{.*}} {ttg.partition = 1 : i32}

    %b_scales_tmem = ttng.tmem_alloc %b_scales_reg : (tensor<128x8xi8, #scales>) -> !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>

    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK-NEXT: [[IS_LAST:%.*]] = arith.cmpi eq, %arg6, [[LAST_ITER]]
    // CHECK-NEXT: tc_gen5_mma_scaled {{.*}} {is_async, ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma_scaled %a_sh, %b_sh, %c_tmem[%c_tok], %a_scales_tmem, %b_scales_tmem, %true, %true lhs = e4m3 rhs = e4m3 : !ttg.memdesc<128x64xf8E4M3FN, #nvmma_smem, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>

    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    scf.yield %c : tensor<128x128xf32, #acc_layout>
  } {tt.warp_specialize}

  tt.return
}

// CHECK-LABEL: @warp_specialize_only_rhs_is_loaded
tt.func @warp_specialize_only_rhs_is_loaded(
  %k_tiles: i32,
  %off_m: i32,
  %off_n: i32,
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>
) {
  %true = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  %BLOCK_K = arith.constant 64 : i32
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  %a_reg = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
  %a_shared = ttg.local_alloc %a_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

  // CHECK-COUNT-1: ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16
  // CHECK-NOT: ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16

  // CHECK: scf.for
  %result = scf.for %k = %c0_i32 to %k_tiles step %c1_i32
      iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    %off_k = arith.muli %k, %BLOCK_K : i32

    // CHECK: wait_barrier
    // CHECK: barrier_expect %{{[0-9]+}}, 16384
    // CHECK: async_tma_copy_global_to_local
    %b_reg = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b_shared = ttg.local_alloc %b_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    // CHECK-NEXT: memdesc_trans
    // CHECK-NEXT: wait_barrier
    %b_T_shared = ttg.memdesc_trans %b_shared {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared_trans, #smem>

    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_T_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared_trans, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    scf.yield %c : tensor<128x128xf32, #acc_layout>

  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  "use"(%result) : (tensor<128x128xf32, #acc_layout>) -> ()
  tt.return
}

// CHECK-LABEL: @user_partition_has_cycle
tt.func @user_partition_has_cycle(
  %k_tiles: i32,
  %off_m: i32,
  %off_n: i32,
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  %BLOCK_K = arith.constant 64 : i32
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  %a_reg = tt.descriptor_load %a_desc[%c0_i32, %c0_i32] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
  %a_shared = ttg.local_alloc %a_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

  // CHECK: scf.for
  // CHECK-SAME: [[PRODUCT:%arg[0-9]+]] = %cst
  %result = scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%product = %zero) -> (tensor<128x128xf32, #acc_layout>) : i32 {
    %off_k = arith.muli %k, %BLOCK_K : i32

    %b_reg = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b_shared = ttg.local_alloc %b_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_T_shared = ttg.memdesc_trans %b_shared {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared_trans, #smem>

    %c_tmem, %c_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_T_shared, %c_tmem[%c_tok], %false, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared_trans, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    // CHECK: [[TIMES_TWO:%.*]] = arith.addf [[PRODUCT]], [[PRODUCT]] {ttg.partition = 0 : i32}
    %times_two = arith.addf %product, %product : tensor<128x128xf32, #acc_layout>
    // CHECK: [[C:%.*]], %{{.*}} = ttng.tmem_load {{.*}} {ttg.partition = 0 : i32}
    // CHECK: arrive_barrier
    // CHECK: [[NEXT_PRODUCT:%.*]] = arith.mulf [[TIMES_TWO]], [[C]] {ttg.partition = 0 : i32}
    %next_product = arith.mulf %times_two, %c : tensor<128x128xf32, #acc_layout>

    // CHECK: yield [[NEXT_PRODUCT]]
    scf.yield %next_product : tensor<128x128xf32, #acc_layout>
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  "use"(%result) : (tensor<128x128xf32, #acc_layout>) -> ()

  tt.return
}

// CHECK-LABEL: @matmul_tma_acc_with_conditional_def_and_use_flag
// CHECK-SAME: [[A_DESC:%arg[0-9]+]]
// CHECK-SAME: [[B_DESC:%arg[0-9]+]]
tt.func @matmul_tma_acc_with_conditional_def_and_use_flag(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %false = arith.constant false
  // CHECK: [[ZERO:%.*]] = arith.constant dense<0.0
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK: [[ACC_BUFS:%.*]], [[ACC_TOK:%.*]] = ttng.tmem_alloc : () -> (!ttg.memdesc<2x128x128xf32,
  // CHECK-NEXT: [[ACC_BUF0:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.tmem_store [[ZERO]], [[ACC_BUF0]]

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<4x{{.*}}xf16,
  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<4x1xi64
  // CHECK-COUNT-4: ttng.arrive_barrier

  // CHECK:      [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK-NEXT: [[ACC_READY_BUF0:%.*]] = ttg.memdesc_index [[ACC_READY_BUFS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF0]], 1
  // CHECK-NEXT: [[ACC_READY_BUF1:%.*]] = ttg.memdesc_index [[ACC_READY_BUFS]]{{\[}}%c1_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF1]], 1

  // CHECK-NEXT: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK-NEXT: [[ACC_EMPTY_BUF0:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[ACC_EMPTY_BUF0]], 1
  // CHECK-NEXT: [[ACC_EMPTY_BUF1:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}%c1_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[ACC_EMPTY_BUF1]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[ACC_EMPTY_BUF0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[ACC_EMPTY_BUF1]], 1

  // CHECK-NEXT: {{[0-9]+}}:5 = scf.for [[K:%arg[0-9]+]]
  // CHECK-SAME: [[FLAG:%arg[0-9]+]] = %true
  // CHECK-SAME: [[LOAD_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LOAD_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero, %flag = %true) -> (tensor<128x128xf32, #acc_layout>, i1) : i32 {
    // CHECK-NEXT: [[OFFS:%.*]]:3 = "get_offsets"([[K]])
    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)

    // CHECK: ttng.wait_barrier
    // CHECK: ttng.barrier_expect
    // CHECK-COUNT-2: ttng.async_tma_copy_global_to_local
    %a = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    // CHECK: ttng.wait_barrier
    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_index [[ACC_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}
    // CHECK-NEXT: [[CUR_ACC_READY_BUF:%.*]] = ttg.memdesc_index [[ACC_READY_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}

    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = arith.cmpi eq, [[K:%.*]], %c0_i32
    // CHECK-NEXT: ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], [[FLAG]], %true, {{.*}}, [[CUR_ACC_READY_BUF]][[[DO_EPILOGUE]]] {is_async, ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %flag, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    %do_epilogue = arith.cmpi eq, %k, %c0_i32 : i32
    // CHECK-NEXT: [[NEXT_FLAG:%.*]] = arith.cmpi ne, [[K]], %c0_i32

    %use_acc = arith.select %do_epilogue, %false, %true : i1

    // CHECK-NEXT: scf.if [[DO_EPILOGUE]]
    scf.if %do_epilogue {
      // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BUF]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: "some_op"()
      "some_op"() : () -> ()
      // CHECK-NEXT: [[C:%.*]], [[USER_TOK:%.*]] = ttng.tmem_load [[ACC_BUF]][]
      // CHECK-NEXT: "acc_user"([[C]])
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
      // CHECK-NEXT: [[CUR_ACC_EMPTY_BUF:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}[[ACC_INDEX]]{{\]}}
      // CHECK-NEXT: ttng.arrive_barrier [[CUR_ACC_EMPTY_BUF]], 1 {ttg.partition = 0 : i32}
    // CHECK-NEXT: }
    }

    // CHECK-NEXT: [[ACC_INDEX_INCR:%.*]] = arith.addi [[ACC_INDEX]], %c1_i32
    // CHECK-NEXT: [[ACC_PHASE_INCR:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_ROLLVER:%.*]] = arith.cmpi eq, [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[ACC_NEXT_INDEX:%.*]] = arith.select [[ACC_ROLLVER]], %c0_i32, [[ACC_INDEX_INCR]]
    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.select [[ACC_ROLLVER]], [[ACC_PHASE_INCR]], [[ACC_PHASE]]
    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_INDEX:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_INDEX]], [[ACC_INDEX]]
    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_PHASE:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_PHASE]], [[ACC_PHASE]]

    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BUF:%.*]] = ttg.memdesc_index [[ACC_EMPTY_BUFS]]{{\[}}[[EPILOGUE_ACC_NEXT_INDEX]]{{\]}}
    // CHECK-NEXT: ttng.wait_barrier [[NEXT_ACC_EMPTY_BUF]], [[EPILOGUE_ACC_NEXT_PHASE]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield [[NEXT_FLAG]], %{{[0-9]+}}, %{{[0-9]+}}, [[EPILOGUE_ACC_NEXT_INDEX]], [[EPILOGUE_ACC_NEXT_PHASE]]
    scf.yield %c, %use_acc : tensor<128x128xf32, #acc_layout>, i1
  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32]
  } {tt.warp_specialize, tt.num_stages = 4 : i32}

  tt.return
}

// CHECK-LABEL: @specialize_load_only
tt.func @specialize_load_only(%desc: !tt.tensordesc<tensor<128x64xf16, #shared>>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK: local_alloc : () -> !ttg.memdesc<3x128x64xf16,
  scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
    // CHECK: wait_barrier {{.*}} {ttg.partition = 0 : i32}
    // CHECK-NEXT: local_load {{.*}} {ttg.partition = 0 : i32}
    // CHECK-NEXT: fence_async_shared {{.*}}partition = 0
    // CHECK-NEXT: arrive_barrier {{.*}} {ttg.partition = 0 : i32}
    %val = tt.descriptor_load %desc[%i, %i] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    "use"(%val) : (tensor<128x64xf16, #oper_layout>) -> ()
  } {tt.warp_specialize}
  tt.return
}

// CHECK-LABEL: @fp4_padded_load
tt.func @fp4_padded_load(%desc: !tt.tensordesc<tensor<1x256x64xui8, #fp4_padded_shared>>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  // CHECK: scf.for [[I:%arg[0-9]+]]
  scf.for %i = %c0_i32 to %ub step %c1_i32 : i32 {
    // CHECK: [[IDX:%.*]] = arith.muli [[I]], %c2_i32 : i32
    // CHECK: async_tma_copy_global_to_local %arg{{[0-9]+}}[[[I]], [[IDX]]]
    %val = tt.descriptor_load %desc[%i, %i] : !tt.tensordesc<tensor<1x256x64xui8, #fp4_padded_shared>> -> tensor<256x64xi8, #oper_layout>
    "use"(%val) : (tensor<256x64xi8, #oper_layout>) -> ()
  } {tt.warp_specialize}
  tt.return
}

// CHECK-LABEL: @specialize_mma_only
tt.func @specialize_mma_only(%rhs_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>, %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>, %ub: i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  // CHECK-COUNT-2: local_alloc : () -> !ttg.memdesc<3x1xi64,

  // CHECK:      [[EMPTY_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // CHECK-NEXT: [[EMPTY_BAR0:%.*]] = ttg.memdesc_index [[EMPTY_BARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[EMPTY_BAR0]], 1

  // CHECK-NEXT: [[READY_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // CHECK-NEXT: [[READY_BAR0:%.*]] = ttg.memdesc_index [[READY_BARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[READY_BAR0]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[READY_BAR0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[EMPTY_BAR0]], 1

  // CHECK-NEXT: [[OPERAND:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, {{.*}}, mutable

  // CHECK-NEXT: scf.for
  %out = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK: wait_barrier
    // CHECK: barrier_expect %{{[0-9]+}}, 16384
    // CHECK: async_tma_copy_global_to_local
    %loaded = tt.descriptor_load %rhs_desc[%i, %i] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    // CHECK: wait_barrier [[READY_BAR0]]
    // CHECK-NEXT: [[LOADED:%.*]], %{{.*}} = ttng.tmem_load [[ACC_TMEM:%.*]][]
    // CHECK: wait_barrier
    // CHECK-NEXT: local_load
    // CHECK-NEXT: fence_async_shared {{.*}}partition = 0
    // CHECK-NEXT: arrive_barrier
    // CHECK-NEXT: [[RESULTS:%.*]]:2 = "some_producer"
    %rhs_reg, %next_acc = "some_producer"(%loaded, %acc) : (tensor<64x128xf16, #oper_layout>, tensor<128x128xf32, #acc_layout>) -> (tensor<128x64xf16, #oper_layout>, tensor<128x128xf32, #acc_layout>)
    // CHECK-NEXT: local_store [[RESULTS]]#0, [[OPERAND]]{{.*}}partition = 0
    // CHECK-NEXT: fence_async_shared {{.*}}partition = 0
    // CHECK-NEXT: [[RHS_T:%.*]] = ttg.memdesc_trans [[OPERAND]] {{.*}}, mutable
    // CHECK-NEXT: tmem_store [[RESULTS]]#1, [[ACC_TMEM]]{{.*}}partition = 0
    // CHECK-NEXT: arrive_barrier [[EMPTY_BAR0]]{{.*}}partition = 0
    %rhs = ttg.local_alloc %rhs_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %rhs_T = ttg.memdesc_trans %rhs {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared_trans, #smem>
    %acc_tmem, %acc_tok = ttng.tmem_alloc %next_acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: wait_barrier [[EMPTY_BAR0]]{{.*}}partition = 1
    // CHECK-NEXT: ttng.tc_gen5_mma %arg1, [[RHS_T]], {{.*}} [[READY_BAR0]][%true] {{.*}}partition = 1
    %mma_tok = ttng.tc_gen5_mma %lhs, %rhs_T, %acc_tmem[%acc_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared_trans, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %acc_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    scf.yield %c : tensor<128x128xf32, #acc_layout>
  } {tt.warp_specialize, tt.num_stages = 3 : i32}
  "use"(%out) : (tensor<128x128xf32, #acc_layout>) -> ()
  tt.return
}

// CHECK-LABEL: @load_scale_mma_user
tt.func @load_scale_mma_user(
  %lhs: !ttg.memdesc<128x64xf16, #shared, #smem>,
  %rhs: !ttg.memdesc<64x128xf16, #shared, #smem>,
  %scales_desc: !tt.tensordesc<tensor<8x128xi8, #shared>>,
  %b_scales: !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>,
  %ub: i32
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  // CHECK: scf.for
  %out = scf.for %i = %c0_i32 to %ub step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK: wait_barrier [[EMPTY_BAR:%.*]], %{{.*}}partition = 2
    // CHECK: barrier_expect [[SCALES_BAR:%.*]], 1024 {{.*}}partition = 2
    // CHECK: async_tma_copy_global_to_local {{.*}}partition = 2
    %scales_result = tt.descriptor_load %scales_desc[%i, %i] : !tt.tensordesc<tensor<8x128xi8, #shared>> -> tensor<8x128xi8, #oper_layout>
    %scales_shared = ttg.local_alloc %scales_result : (tensor<8x128xi8, #oper_layout>) -> !ttg.memdesc<8x128xi8, #shared, #smem>
    // CHECK: wait_barrier [[SCALES_BAR]]{{.*}}partition = 0
    // CHECK-NEXT: [[SCALES_REG:%.*]] = ttg.local_load {{.*}}partition = 0
    // CHECK-NEXT: arrive_barrier [[EMPTY_BAR]]{{.*}}partition = 0
    %scales_reg = ttg.local_load %scales_shared : !ttg.memdesc<8x128xi8, #shared, #smem> -> tensor<8x128xi8, #oper_layout>
    // CHECK-NEXT: [[SCALES_TRANS:%.*]] = tt.trans [[SCALES_REG]] {{.*}}partition = 0
    %scales_T = tt.trans %scales_reg {order = array<i32: 1, 0>} : tensor<8x128xi8, #oper_layout> -> tensor<128x8xi8, #oper_layout_trans>
    %scales_cvt = ttg.convert_layout %scales_T : tensor<128x8xi8, #oper_layout_trans> -> tensor<128x8xi8, #scales>
    // CHECK-NEXT: wait_barrier [[SCALES_TMEM_BAR:%.*]], %arg{{[0-9]+}} {{.*}}partition = 0
    // CHECK-NEXT: tmem_store [[SCALES_TRANS]], [[SCALES_TMEM:%.*]], %true {{.*}}partition = 0
    %scales_tmem = ttng.tmem_alloc %scales_cvt : (tensor<128x8xi8, #scales>) -> !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>
    // CHECK-NEXT: arrive_barrier [[SCALES_READY_BAR:%.*]], 1 {{.*}}partition = 0

    // CHECK: wait_barrier [[USER_DONE:%.*]], %arg{{[0-9]+}}, %true {{.*}}partition = 1
    // CHECK: wait_barrier [[SCALES_READY_BAR]]{{.*}}partition = 1
    %acc_tmem, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK-NEXT: tc_gen5_mma_scaled {{.*}} [[SCALES_TMEM]]{{.*}} [[USER_BAR:%.*]][%true], [[SCALES_TMEM_BAR]][%true] {{.*}}partition = 1
    %mma_tok = ttng.tc_gen5_mma_scaled %lhs, %rhs, %acc_tmem[%acc_tok], %scales_tmem, %b_scales, %true, %true lhs = e4m3 rhs = e4m3 : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>

    // CHECK: wait_barrier [[USER_BAR]]{{.*}}partition = 0
    // CHECK-NEXT: tmem_load
    %c, %load_tok = ttng.tmem_load %acc_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    // CHECK: arrive_barrier [[USER_DONE]]{{.*}}partition = 0

    "user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()

    scf.yield %c : tensor<128x128xf32, #acc_layout>
  } {tt.warp_specialize, tt.num_stages = 3 : i32}
  "use"(%out) : (tensor<128x128xf32, #acc_layout>) -> ()
  tt.return
}

// CHECK-LABEL: @store_mma_load
tt.func @store_mma_load(
  %ub: i32,
  %lhs_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %rhs: !ttg.memdesc<64x128xf16, #shared, #smem>
) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %true = arith.constant true

  // CHECK: [[LHS_EMPTY_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64,
  // CHECK: [[LHS_EMPTY_BAR0:%.*]] = ttg.memdesc_index [[LHS_EMPTY_BARS]]{{\[}}%c0_i32{{\]}}
  // CHECK: [[LHS_EMPTY_BAR1:%.*]] = ttg.memdesc_index [[LHS_EMPTY_BARS]]{{\[}}%c1_i32{{\]}}
  // CHECK: [[LHS_READY_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64,
  // CHECK: arrive_barrier [[LHS_EMPTY_BAR0]]
  // CHECK: arrive_barrier [[LHS_EMPTY_BAR1]]
  // CHECK-NOT: arrive_barrier

  // CHECK: [[MMA_ENTRY_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // CHECK: [[MMA_ENTRY_BAR:%.*]] = ttg.memdesc_index [[MMA_ENTRY_BARS]]{{\[}}%c0_i32{{\]}}
  // CHECK: [[MMA_EXIT_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
  // CHECK: [[MMA_EXIT_BAR:%.*]] = ttg.memdesc_index [[MMA_EXIT_BARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NOT: arrive_barrier

  // CHECK: [[LHS_SHARED:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16,

  // CHECK: scf.for
  scf.for %i = %c0 to %ub step %c1 : i32 {
    // CHECK-NEXT: [[LOAD_EMPTY_BAR:%.*]] = ttg.memdesc_index [[LHS_EMPTY_BARS]]
    // CHECK-NEXT: wait_barrier [[LOAD_EMPTY_BAR]]{{.*}}partition = 2
    // CHECK-NEXT: [[LOAD_READY_BAR:%.*]] = ttg.memdesc_index [[LHS_READY_BARS]]
    // CHECK-NEXT: barrier_expect [[LOAD_READY_BAR]]{{.*}}partition = 2
    // CHECK-NEXT: [[LOAD_BUF:%.*]] = ttg.memdesc_index
    // CHECK-NEXT: async_tma_copy_global_to_local{{.*}}partition = 2
    %lhs = tt.descriptor_load %lhs_desc[%i, %i] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>

    // CHECK-NEXT: wait_barrier [[LOAD_READY_BAR]], {{.*}}partition = 0
    // CHECK-NEXT: [[LHS:%.*]] = ttg.local_load [[LOAD_BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: fence_async_shared {{.*}}partition = 0
    // CHECK-NEXT: arrive_barrier [[LOAD_EMPTY_BAR]], {{.*}}partition = 0
    // CHECK-NEXT: [[LHS_OP:%.*]] = arith.addf [[LHS]], [[LHS]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: local_store [[LHS_OP]], [[LHS_SHARED]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: fence_async_shared {bCluster = false, ttg.partition = 0 : i32}
    %lhs_op = arith.addf %lhs, %lhs : tensor<128x64xf16, #oper_layout>
    %lhs_shared = ttg.local_alloc %lhs_op : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

    // CHECK-NEXT: [[ACC:%.*]] = "make_acc"()
    %acc = "make_acc"() : () -> tensor<128x128xf32, #acc_layout>
    // CHECK-NEXT: [[ACC_TMEM:%.*]] = ttg.memdesc_index
    // CHECK-NEXT: tmem_store [[ACC]], [[ACC_TMEM]][], %true {{.*}}partition = 0
    // CHECK-NEXT: arrive_barrier [[MMA_ENTRY_BAR]], {{.*}}partition = 0
    %acc_tmem, %acc_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK-NEXT: wait_barrier [[MMA_ENTRY_BAR]], {{.*}}partition = 1
    // CHECK-NEXT: tc_gen5_mma {{.*}} [[MMA_EXIT_BAR]][%true]
    %mma_tok = ttng.tc_gen5_mma %lhs_shared, %rhs, %acc_tmem[%acc_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>

    // CHECK-NEXT: wait_barrier [[MMA_EXIT_BAR]], {{.*}}partition = 0
    // CHECK-NEXT: [[ACC_VALUE:%.*]], [[LOAD_TOK:%.*]] = ttng.tmem_load [[ACC_TMEM]][]
    %acc_value, %load_tok = ttng.tmem_load %acc_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    // CHECK-NEXT: arith.xori
    // CHECK-NEXT: "use"([[ACC_VALUE]])
    "use"(%acc_value) : (tensor<128x128xf32, #acc_layout>) -> ()
  } {tt.warp_specialize, tt.num_stages = 2 : i32, tt.disallow_acc_multi_buffer}
  tt.return
}

// CHECK-LABEL: @local_alloc_into_mma
tt.func @local_alloc_into_mma(
  %ub: i32,
  %lhs_reg: tensor<128x64xf16, #oper_layout>,
  %rhs_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %acc, %acc_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
  %true = arith.constant true
  // CHECK: [[LHS_SHARED:%.*]] = ttg.local_alloc %arg1 : (tensor<128x64xf16, {{.*}}>) -> !ttg.memdesc<128x64xf16,
  // CHECK: scf.for
  scf.for %i = %c0 to %ub step %c1 iter_args(%tok = %acc_tok) -> !ttg.async.token : i32 {
    // CHECK: barrier_expect [[LOAD_READY_BAR:%.*]], 16384 {ttg.partition = 2 : i32}
    %lhs_shared = ttg.local_alloc %lhs_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %rhs_reg = tt.descriptor_load %rhs_desc[%i, %i] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    // CHECK: wait_barrier [[LOAD_READY_BAR]], {{.*}}partition = 0
    // CHECK-NEXT: [[RHS_REG:%.*]] = ttg.local_load {{.*}}partition = 0
    // CHECK-NEXT: fence_async_shared {{.*}}partition = 0
    // CHECK-NEXT: arrive_barrier
    // CHECK-NEXT: [[RHS_REG_MOD:%.*]] = arith.addf [[RHS_REG]], [[RHS_REG]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: wait_barrier [[MMA_OPER_BAR:%.*]], %arg{{.*}}partition = 0
    // CHECK-NEXT: local_store [[RHS_REG_MOD]], [[RHS_SHARED:%.*]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: fence_async_shared {bCluster = false, ttg.partition = 0 : i32}
    // CHECK-NEXT: arrive_barrier [[MMA_READY_BAR:%.*]], 1 {{.*}}partition = 0
    %rhs_reg_mod = arith.addf %rhs_reg, %rhs_reg : tensor<64x128xf16, #oper_layout>
    %rhs_shared = ttg.local_alloc %rhs_reg_mod : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
    // CHECK: wait_barrier [[MMA_READY_BAR]], {{.*}}partition = 1
    // CHECK-NEXT: tc_gen5_mma [[LHS_SHARED]], [[RHS_SHARED]], {{.*}} [[MMA_OPER_BAR]][%true] {{.*}}partition = 1
    %mma_tok = ttng.tc_gen5_mma %lhs_shared, %rhs_shared, %acc[%acc_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    scf.yield %mma_tok : !ttg.async.token
  } {tt.warp_specialize, tt.num_stages = 2 : i32}
  tt.return
}

// CHECK-LABEL: @shmem_sink_iterator_invalidation
// CHECK-SAME: [[A_DESC:%arg[0-9]+]]: !tt.tensordesc
// CHECK-SAME: [[B_DESC:%arg[0-9]+]]: !tt.tensordesc
tt.func @shmem_sink_iterator_invalidation(
  %k_tiles: i32,
  %off_m: i32,
  %off_n: i32,
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>
) {
  %true = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32

  %BLOCK_K = arith.constant 64 : i32
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  %result = scf.for %k = %c0_i32 to %k_tiles step %c1_i32
      iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    %off_k = arith.muli %k, %BLOCK_K : i32

    // CHECK: async_tma_copy_global_to_local [[B_DESC]]
    %b_reg = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    // CHECK: wait_barrier [[B_EMPTY:%[0-9]+]]
    // CHECK: async_tma_copy_global_to_local [[A_DESC]][{{.*}}] [[B_DEST:%[0-9]+]], [[B_BAR:%[0-9]+]]
    %a_reg = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>

    %a_shared = ttg.local_alloc %a_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    // CHECK: wait_barrier [[B_BAR]]
    // CHECK-NEXT: [[B:%.*]] = ttg.local_load [[B_DEST]]
    // CHECK-NEXT: arrive_barrier [[B_EMPTY]]
    // CHECK-NEXT: memdesc_trans
    %a = ttg.local_load %a_shared : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #lhs_layout>
    %b_shared = ttg.local_alloc %b_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_T_shared = ttg.memdesc_trans %b_shared {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared_trans, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %a_tmem = ttng.tmem_alloc %a : (tensor<128x64xf16, #lhs_layout>) -> !ttg.memdesc<128x64xf16, #lhs_tmem, #ttng.tensor_memory>
    %mma_tok = ttng.tc_gen5_mma %a_tmem, %b_T_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #lhs_tmem, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared_trans, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>

    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    scf.yield %c : tensor<128x128xf32, #acc_layout>

  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  "use"(%result) : (tensor<128x128xf32, #acc_layout>) -> ()
  tt.return
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#load_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_T = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>

#smem = #ttg.shared_memory
#tmem_acc = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = true>
#tmem_lhs = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// CHECK-LABEL: @attention_forward
// CHECK-SAME: [[Q_SHARED:%arg[0-9]+]]
// CHECK-SAME: [[K_DESC:%arg[0-9]+]]
// CHECK-SAME: [[V_DESC:%arg[0-9]+]]
// CHECK-SAME: [[QK_SCALE:%arg[0-9]+]]
// CHECK-SAME: [[N_TILES:%arg[0-9]+]]
tt.func public @attention_forward(
  %Q_shared: !ttg.memdesc<256x64xf16, #shared, #smem>,
  %K_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>,
  %V_desc: !tt.tensordesc<tensor<64x64xf16, #shared>>,
  %qk_scale: f32,
  %n_tiles: i32
) {
  %true = arith.constant true
  %false = arith.constant false
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32

  %neg_inf = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  %zero = arith.constant dense<0.0> : tensor<256x64xf32, #blocked>
  %one = arith.constant dense<1.0> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

  // CHECK-DAG: [[NEG_INF:%.*]] = arith.constant dense<0xFF800000>
  // CHECK-DAG: [[ZERO:%.*]] = arith.constant dense<0.0
  // CHECK-DAG: [[ONE:%.*]] = arith.constant dense<1.0

  // CHECK:      [[QK_TMEM:%.*]], [[PV_TOK:%.*]] = ttng.tmem_alloc : () -> (!ttg.memdesc<2x256x64xf32,

  // CHECK-NEXT: [[PV_TMEM:%.*]], [[QK_TOK:%.*]] = ttng.tmem_alloc : () -> (!ttg.memdesc<1x256x64xf32,
  // CHECK-NEXT: [[PV_0:%.*]] = ttg.memdesc_index [[PV_TMEM]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.tmem_store [[ZERO]], [[PV_0]]

  // CHECK-NEXT: [[K_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x64x64xf16,

  // CHECK-NEXT: [[K_EMPTY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
  // CHECK-NEXT: [[K_EMPTY_BAR0:%.*]] = ttg.memdesc_index [[K_EMPTY_MBARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[K_EMPTY_BAR0]], 1
  // CHECK-NEXT: [[K_EMPTY_BAR1:%.*]] = ttg.memdesc_index [[K_EMPTY_MBARS]]{{\[}}%c1_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[K_EMPTY_BAR1]], 1
  // CHECK-NEXT: [[K_EMPTY_BAR2:%.*]] = ttg.memdesc_index [[K_EMPTY_MBARS]]{{\[}}%c2_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[K_EMPTY_BAR2]], 1

  // CHECK-NEXT: [[K_READY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
  // CHECK-NEXT: [[K_READY_BAR0:%.*]] = ttg.memdesc_index [[K_READY_MBARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[K_READY_BAR0]], 1
  // CHECK-NEXT: [[K_READY_BAR1:%.*]] = ttg.memdesc_index [[K_READY_MBARS]]{{\[}}%c1_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[K_READY_BAR1]], 1
  // CHECK-NEXT: [[K_READY_BAR2:%.*]] = ttg.memdesc_index [[K_READY_MBARS]]{{\[}}%c2_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[K_READY_BAR2]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[K_EMPTY_BAR0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[K_EMPTY_BAR1]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[K_EMPTY_BAR2]], 1

  // CHECK-NEXT: [[V_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x64x64xf16,

  // CHECK-NEXT: [[V_EMPTY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
  // CHECK-NEXT: [[V_EMPTY_BAR0:%.*]] = ttg.memdesc_index [[V_EMPTY_MBARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[V_EMPTY_BAR0]], 1
  // CHECK-NEXT: [[V_EMPTY_BAR1:%.*]] = ttg.memdesc_index [[V_EMPTY_MBARS]]{{\[}}%c1_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[V_EMPTY_BAR1]], 1
  // CHECK-NEXT: [[V_EMPTY_BAR2:%.*]] = ttg.memdesc_index [[V_EMPTY_MBARS]]{{\[}}%c2_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[V_EMPTY_BAR2]], 1

  // CHECK-NEXT: [[V_READY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
  // CHECK-NEXT: [[V_READY_BAR0:%.*]] = ttg.memdesc_index [[V_READY_MBARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[V_READY_BAR0]], 1
  // CHECK-NEXT: [[V_READY_BAR1:%.*]] = ttg.memdesc_index [[V_READY_MBARS]]{{\[}}%c1_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[V_READY_BAR1]], 1
  // CHECK-NEXT: [[V_READY_BAR2:%.*]] = ttg.memdesc_index [[V_READY_MBARS]]{{\[}}%c2_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[V_READY_BAR2]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[V_EMPTY_BAR0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[V_EMPTY_BAR1]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[V_EMPTY_BAR2]], 1

  // CHECK-NEXT: [[QK_READY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK-NEXT: [[QK_READY_BAR0:%.*]] = ttg.memdesc_index [[QK_READY_MBARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[QK_READY_BAR0]], 1
  // CHECK-NEXT: [[QK_READY_BAR1:%.*]] = ttg.memdesc_index [[QK_READY_MBARS]]{{\[}}%c1_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[QK_READY_BAR1]], 1

  // CHECK-NEXT: [[QK_EMPTY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK-NEXT: [[QK_EMPTY_BAR0:%.*]] = ttg.memdesc_index [[QK_EMPTY_MBARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[QK_EMPTY_BAR0]], 1
  // CHECK-NEXT: [[QK_EMPTY_BAR1:%.*]] = ttg.memdesc_index [[QK_EMPTY_MBARS]]{{\[}}%c1_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[QK_EMPTY_BAR1]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[QK_EMPTY_BAR0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[QK_EMPTY_BAR1]], 1

  // CHECK-NEXT: [[PV_EMPTY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
  // CHECK-NEXT: [[PV_EMPTY_BAR0:%.*]] = ttg.memdesc_index [[PV_EMPTY_MBARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[PV_EMPTY_BAR0]], 1

  // CHECK-NEXT: [[PV_READY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
  // CHECK-NEXT: [[PV_READY_BAR0:%.*]] = ttg.memdesc_index [[PV_READY_MBARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[PV_READY_BAR0]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[PV_READY_BAR0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[PV_EMPTY_BAR0]], 1

  // CHECK-NEXT: [[P_BUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<256x64xf16,

  // CHECK-NEXT: [[P_EMPTY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
  // CHECK-NEXT: [[P_EMPTY_BAR0:%.*]] = ttg.memdesc_index [[P_EMPTY_MBARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[P_EMPTY_BAR0]], 1

  // CHECK-NEXT: [[P_READY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
  // CHECK-NEXT: [[P_READY_BAR0:%.*]] = ttg.memdesc_index [[P_READY_MBARS]]{{\[}}%c0_i32{{\]}}
  // CHECK-NEXT: ttng.init_barrier [[P_READY_BAR0]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[P_EMPTY_BAR0]], 1

  // CHECK-NEXT: [[OUTS:%.*]]:11 = scf.for [[I:%.*]] = %c0_i32 to [[N_TILES]] step %c64_i32 iter_args(
  // CHECK-SAME: [[L_I:%arg[0-9]+]] = [[ONE]],
  // CHECK-SAME: [[M_I:%arg[0-9]+]] = [[NEG_INF]],
  // CHECK-SAME: {{%arg[0-9]+}}
  // CHECK-SAME: [[K_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[K_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[V_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[V_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[QK_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[QK_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[PV_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[P_PHASE:%arg[0-9]+]] = %c0_i32
  %loop_outs:3 = scf.for %i = %c0_i32 to %n_tiles step %c64_i32 iter_args(
    %l_i = %one,
    %acc = %zero,
    %m_i = %neg_inf
  ) -> (
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>,
    tensor<256x64xf32, #blocked>,
    tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  ) : i32 {

    // CHECK-NEXT: [[K_EMPTY_BAR:%.*]] = ttg.memdesc_index [[K_EMPTY_MBARS]]{{\[}}[[K_INDEX]]{{\]}}
    // CHECK-NEXT: wait_barrier [[K_EMPTY_BAR]], [[K_PHASE]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[K_READY_BAR:%.*]] = ttg.memdesc_index [[K_READY_MBARS]]{{\[}}[[K_INDEX]]{{\]}}
    // CHECK-NEXT: barrier_expect [[K_READY_BAR]], 8192 {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[K_BUF:%.*]] = ttg.memdesc_index [[K_BUFS]]{{\[}}[[K_INDEX]]{{\]}}
    // CHECK-NEXT: async_tma_copy_global_to_local [[K_DESC]][[[I]], %c0_i32] [[K_BUF]], [[K_READY_BAR]], %true {ttg.partition = 2 : i32}
    %K = tt.descriptor_load %K_desc[%i, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #load_blocked>
    %K_shared = ttg.local_alloc %K : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>

    // CHECK-NEXT: [[K_TRANS:%.*]] = ttg.memdesc_trans [[K_BUF]] {order = array<i32: 1, 0>, ttg.partition = 1 : i32}
    %K_trans = ttg.memdesc_trans %K_shared {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared_T, #smem>
    // CHECK-NEXT: wait_barrier [[K_READY_BAR]], [[K_PHASE]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[QK_BUF:%.*]] = ttg.memdesc_index [[QK_TMEM]]{{\[}}[[QK_INDEX]]{{\]}}
    // CHECK-NEXT: [[QK_EMPTY_BAR:%.*]] = ttg.memdesc_index [[QK_EMPTY_MBARS]]{{\[}}[[QK_INDEX]]{{\]}}
    // CHECK-NEXT: wait_barrier [[QK_EMPTY_BAR]], [[QK_PHASE]], %true {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[QK_READY_BAR:%.*]] = ttg.memdesc_index [[QK_READY_MBARS]]{{\[}}[[QK_INDEX]]{{\]}}
    // CHECK-NEXT: tc_gen5_mma [[Q_SHARED]], [[K_TRANS]], [[QK_BUF]][], %false, %true, [[K_EMPTY_BAR]][%true], [[QK_READY_BAR]][%true] {is_async, ttg.partition = 1 : i32}
    %QK_tmem, %QK_tok = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %QK_mma_tok = ttng.tc_gen5_mma %Q_shared, %K_trans, %QK_tmem[%QK_tok], %false, %true : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared_T, #smem>, !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>

    // CHECK-NEXT: wait_barrier [[QK_READY_BAR]], [[QK_PHASE]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[QK:%.*]], [[QK_LOAD_TOK:%.*]] = ttng.tmem_load [[QK_BUF]][] {ttg.partition = 0 : i32}
    %QK, %QK_load_tok = ttng.tmem_load %QK_tmem[%QK_mma_tok] : !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    // CHECK-NEXT: arrive_barrier [[QK_EMPTY_BAR]], 1 {ttg.partition = 0 : i32}

    // CHECK-NEXT: [[QK_INDEX_INCR:%.*]] = arith.addi [[QK_INDEX]], %c1_i32
    // CHECK-NEXT: [[QK_PHASE_INCR:%.*]] = arith.xori [[QK_PHASE]], %c1_i32
    // CHECK-NEXT: [[QK_ROLLVER:%.*]] = arith.cmpi eq, [[QK_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[QK_NEXT_INDEX:%.*]] = arith.select [[QK_ROLLVER]], %c0_i32, [[QK_INDEX_INCR]]
    // CHECK-NEXT: [[QK_NEXT_PHASE:%.*]] = arith.select [[QK_ROLLVER]], [[QK_PHASE_INCR]], [[QK_PHASE]]

    // CHECK-NEXT: [[ROW_MAX:%.*]] = "compute_row_max"([[QK]], [[QK_SCALE]]) {ttg.partition = 0 : i32}
    %row_max = "compute_row_max"(%QK, %qk_scale) : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT: [[QK_ADJ:%.*]] = "sub_row_max"([[QK]], [[ROW_MAX]], [[QK_SCALE]]) {ttg.partition = 0 : i32}
    %QK_adj = "sub_row_max"(%QK, %row_max, %qk_scale) : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
    // CHECK-NEXT: [[SOFTMAX:%.*]] = math.exp2 [[QK_ADJ]] {ttg.partition = 0 : i32}
    %softmax = math.exp2 %QK_adj : tensor<256x64xf32, #blocked>

    // CHECK-NEXT: [[DIFF_CORR:%.*]] = arith.subf [[M_I]], [[ROW_MAX]] {ttg.partition = 3 : i32}
    // CHECK-NEXT: [[DIFF_SOFT:%.*]] = arith.subf [[M_I]], [[ROW_MAX]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[ALPHA_CORR:%.*]] = math.exp2 [[DIFF_CORR]] {ttg.partition = 3 : i32}
    // CHECK-NEXT: [[ALPHA_SOFT:%.*]] = math.exp2 [[DIFF_SOFT]] {ttg.partition = 0 : i32}
    %diff = arith.subf %m_i, %row_max : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %alpha = math.exp2 %diff : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    // CHECK-NEXT: [[L_IJ:%.*]] = "tt.reduce"([[SOFTMAX]])
    %l_ij = "tt.reduce"(%softmax) <{axis = 1 : i32}> ({
    ^bb0(%arg29: f32, %arg30: f32):
      %68 = arith.addf %arg29, %arg30 : f32
      // CHECK: tt.reduce.return
      tt.reduce.return %68 : f32
    // CHECK-NEXT: {ttg.partition = 0 : i32}
    }) : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT: [[L_I_SCALED:%.*]] = arith.mulf [[L_I]], [[ALPHA_SOFT]] {ttg.partition = 0 : i32}
    %l_i_scaled = arith.mulf %l_i, %alpha : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK-NEXT: [[NEXT_L_I:%.*]] = arith.addf [[L_I_SCALED]], [[L_IJ]] {ttg.partition = 0 : i32}
    %next_l_i = arith.addf %l_i_scaled, %l_ij : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>

    // CHECK-NEXT: [[ALPHA_0:%.*]] = tt.expand_dims [[ALPHA_CORR]] {axis = 1 : i32, ttg.partition = 3 : i32}
    %alpha_0 = tt.expand_dims %alpha {axis = 1 : i32} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
    // CHECK-NEXT: [[ALPHA_1:%.*]] = tt.broadcast [[ALPHA_0]] {ttg.partition = 3 : i32}
    %alpha_1 = tt.broadcast %alpha_0 : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>

    // CHECK-NEXT: wait_barrier [[PV_READY_BAR0]], [[PV_PHASE]] {ttg.partition = 3 : i32}
    // CHECK-NEXT: [[PV:%.*]], [[PV_TOK:%.*]] = ttng.tmem_load [[PV_0]][] {ttg.partition = 3 : i32}
    // CHECK-NEXT: [[NEXT_PV_PHASE:%.*]] = arith.xori [[PV_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_CORRECTED:%.*]] = arith.mulf [[PV]], [[ALPHA_1]] {ttg.partition = 3 : i32}
    %acc_corrected = arith.mulf %acc, %alpha_1 : tensor<256x64xf32, #blocked>

    // CHECK-NEXT: [[V_EMPTY_BAR:%.*]] = ttg.memdesc_index [[V_EMPTY_MBARS]]{{\[}}[[V_INDEX]]{{\]}}
    // CHECK-NEXT: wait_barrier [[V_EMPTY_BAR]], [[V_PHASE]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[V_READY_BAR:%.*]] = ttg.memdesc_index [[V_READY_MBARS]]{{\[}}[[V_INDEX]]{{\]}}
    // CHECK-NEXT: barrier_expect [[V_READY_BAR]], 8192 {ttg.partition = 2 : i32}
    // CHECK-NEXT: [[V_BUF:%.*]] = ttg.memdesc_index [[V_BUFS]]{{\[}}[[V_INDEX]]{{\]}}
    // CHECK-NEXT: async_tma_copy_global_to_local [[V_DESC]][[[I]], %c0_i32] [[V_BUF]], [[V_READY_BAR]], %true {ttg.partition = 2 : i32}
    %V = tt.descriptor_load %V_desc[%i, %c0_i32] : !tt.tensordesc<tensor<64x64xf16, #shared>> -> tensor<64x64xf16, #load_blocked>
    %V_shared = ttg.local_alloc %V : (tensor<64x64xf16, #load_blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>

    // CHECK-NEXT: [[P:%.*]] = arith.truncf [[SOFTMAX]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: wait_barrier [[P_EMPTY_BAR0]], [[P_PHASE]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: tmem_store [[P]], [[P_BUF]], %true {ttg.partition = 0 : i32}
    // CHECK-NEXT: arrive_barrier [[P_READY_BAR0]], 1 {ttg.partition = 0 : i32}
    %P = arith.truncf %softmax : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>

    // CHECK-NEXT: tmem_store [[ACC_CORRECTED]], [[PV_0]][], %true {ttg.partition = 3 : i32}
    // CHECK-NEXT: arrive_barrier [[PV_EMPTY_BAR0]], 1 {ttg.partition = 3 : i32}

    // CHECK-NEXT: wait_barrier [[V_READY_BAR]], [[V_PHASE]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: wait_barrier [[PV_EMPTY_BAR0]], [[NEXT_PV_PHASE]], %true {ttg.partition = 1 : i32}
    // CHECK-NEXT: wait_barrier [[P_READY_BAR0]], [[P_PHASE]] {ttg.partition = 1 : i32}
    %P_tmem = ttng.tmem_alloc %P : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem_lhs, #ttng.tensor_memory>
    %acc_tmem, %acc_tok = ttng.tmem_alloc %acc_corrected : (tensor<256x64xf32, #blocked>) -> (!ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK-NEXT: tc_gen5_mma [[P_BUF]], [[V_BUF]], [[PV_0]][], %true, %true, [[V_EMPTY_BAR]][%true], [[PV_READY_BAR0]][%true], [[P_EMPTY_BAR0]][%true] {is_async, ttg.partition = 1 : i32}
    %PV_mma_tok = ttng.tc_gen5_mma %P_tmem, %V_shared, %acc_tmem[%acc_tok], %true, %true : !ttg.memdesc<256x64xf16, #tmem_lhs, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable>
    %O, %O_tok = ttng.tmem_load %acc_tmem[%PV_mma_tok] : !ttg.memdesc<256x64xf32, #tmem_acc, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>

    // CHECK-NEXT: [[K_INDEX_INCR:%.*]] = arith.addi [[K_INDEX]], %c1_i32
    // CHECK-NEXT: [[K_PHASE_INCR:%.*]] = arith.xori [[K_PHASE]], %c1_i32
    // CHECK-NEXT: [[K_ROLLVER:%.*]] = arith.cmpi eq, [[K_INDEX_INCR]], %c3_i32
    // CHECK-NEXT: [[K_NEXT_INDEX:%.*]] = arith.select [[K_ROLLVER]], %c0_i32, [[K_INDEX_INCR]]
    // CHECK-NEXT: [[K_NEXT_PHASE:%.*]] = arith.select [[K_ROLLVER]], [[K_PHASE_INCR]], [[K_PHASE]]

    // CHECK-NEXT: [[V_INDEX_INCR:%.*]] = arith.addi [[V_INDEX]], %c1_i32
    // CHECK-NEXT: [[V_PHASE_INCR:%.*]] = arith.xori [[V_PHASE]], %c1_i32
    // CHECK-NEXT: [[V_ROLLVER:%.*]] = arith.cmpi eq, [[V_INDEX_INCR]], %c3_i32
    // CHECK-NEXT: [[V_NEXT_INDEX:%.*]] = arith.select [[V_ROLLVER]], %c0_i32, [[V_INDEX_INCR]]
    // CHECK-NEXT: [[V_NEXT_PHASE:%.*]] = arith.select [[V_ROLLVER]], [[V_PHASE_INCR]], [[V_PHASE]]

    // CHECK-NEXT: [[NEXT_P_PHASE:%.*]] = arith.xori [[P_PHASE]], %c1_i32

    // CHECK-NEXT: yield [[NEXT_L_I]], [[ROW_MAX]], %{{[0-9]+}}, [[K_NEXT_INDEX]], [[K_NEXT_PHASE]], [[V_NEXT_INDEX]], [[V_NEXT_PHASE]], [[QK_NEXT_INDEX]], [[QK_NEXT_PHASE]], [[NEXT_PV_PHASE]], [[NEXT_P_PHASE]]

    scf.yield %next_l_i, %O, %row_max : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32
  } {tt.warp_specialize}

  // CHECK-NEXT: wait_barrier [[PV_READY_BAR0]], [[OUTS]]#9

  "use"(%loop_outs#0, %loop_outs#1, %loop_outs#2) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()

  tt.return
}

}
