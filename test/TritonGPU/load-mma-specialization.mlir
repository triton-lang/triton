// RUN: triton-opt %s -allow-unregistered-dialect -tritongpu-hoist-tmem-alloc | FileCheck %s --check-prefix=TMEM --check-prefix=FUNC
// RUN: triton-opt %s -allow-unregistered-dialect -verify-diagnostics --tritongpu-hoist-tmem-alloc -tritongpu-load-mma-specialization -int-range-optimizations -canonicalize -cse -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect -verify-diagnostics --tritongpu-hoist-tmem-alloc -tritongpu-automatic-warp-specialization | FileCheck %s --check-prefix=AWS --check-prefix=FUNC

#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK-DAG: [[SHARED:#.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_trans = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-DAG: [[ACC_TMEM:#.*]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

#fp4_padded_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true, CTAsPerCGA = [1, 1, 1], CTASplitNum = [1, 1, 1], CTAOrder = [2, 1, 0]}>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// FUNC-LABEL: @warp_specialize_tma_matmul

// TMEM: ttng.tmem_alloc
// TMEM: scf.for

// AWS: ttg.warp_specialize
// AWS: num_warps(1)
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
  // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[C0]], [[C0]], [[C0]]]
  // CHECK-NEXT: [[INIT_TOK:%.*]] = ttng.tmem_store [[ZERO]], [[ACC_BUF]][[[ACC_TOK]]]

  // CHECK-NEXT: [[A_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, [[SHARED]]
  // CHECK-NEXT: [[B_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, [[SHARED]]

  // CHECK-NEXT: [[READY_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[READY_MBAR0:%.*]] = ttg.memdesc_subview [[READY_MBARS]][[[C0]]]
  // CHECK-NEXT: ttng.init_barrier [[READY_MBAR0]], 1
  // CHECK-NEXT: [[READY_MBAR1:%.*]] = ttg.memdesc_subview [[READY_MBARS]][[[C1]]]
  // CHECK-NEXT: ttng.init_barrier [[READY_MBAR1]], 1

  // CHECK-NEXT: [[OPER_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[OPER_MBAR0:%.*]] = ttg.memdesc_subview [[OPER_MBARS]][[[C0]]]
  // CHECK-NEXT: ttng.init_barrier [[OPER_MBAR0]], 1
  // CHECK-NEXT: [[OPER_MBAR1:%.*]] = ttg.memdesc_subview [[OPER_MBARS]][[[C1]]]
  // CHECK-NEXT: ttng.init_barrier [[OPER_MBAR1]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[READY_MBAR0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[READY_MBAR1]], 1

  // CHECK-NEXT: [[DONE_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64
  // CHECK-NEXT: [[DONE_MBAR0:%.*]] = ttg.memdesc_subview [[DONE_MBAR]][[[C0]]]
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

    // CHECK-NEXT: [[READY_MBAR:%.*]] = ttg.memdesc_subview [[READY_MBARS]][[[IDX]]]
    // CHECK-NEXT: ttng.wait_barrier [[READY_MBAR]], [[PHASE]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[OPER_MBAR:%.*]] = ttg.memdesc_subview [[OPER_MBARS]][[[IDX]]]
    // CHECK-NEXT: ttng.barrier_expect [[OPER_MBAR]], 32768 {ttg.partition = 0 : i32}

    // CHECK-NEXT: [[A_BUF:%.*]] = ttg.memdesc_subview [[A_BUFS]][[[IDX]], [[C0]], [[C0]]]
    // CHECK-NEXT: [[A_DESC_PTR:%.*]] = ttng.tensor_desc_to_tma_ptr [[A_DESC]]
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local [[A_DESC_PTR]][[[OFF_M]], [[OFF_K]]] [[A_BUF]], [[OPER_MBAR]], [[TRUE]] {ttg.partition = 0 : i32}
    %a_reg = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    // CHECK-NEXT: [[B_BUF:%.*]] = ttg.memdesc_subview [[B_BUFS]][[[IDX]], [[C0]], [[C0]]]
    // CHECK-NEXT: [[B_DESC_PTR:%.*]] = ttng.tensor_desc_to_tma_ptr [[B_DESC]]
    // CHECK-NEXT: ttng.async_tma_copy_global_to_local [[B_DESC_PTR]][[[OFF_N]], [[OFF_K]]] [[B_BUF]], [[OPER_MBAR]], [[TRUE]] {ttg.partition = 0 : i32}
    %b_reg = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>

    %a_shared = ttg.local_alloc %a_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b_reg : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    // CHECK-NEXT: [[B_T:%.*]] = ttg.memdesc_trans [[B_BUF]] {order = array<i32: 1, 0>, ttg.partition = 1 : i32}
    // CHECK-NEXT: ttng.wait_barrier [[OPER_MBAR]], [[PHASE]] {ttg.partition = 1 : i32}
    %b_T_shared = ttg.memdesc_trans %b_shared {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared_trans, #smem>
    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK-NEXT: [[LAST_ITER:%.*]] = arith.subi [[K_TILES]], [[C1]]
    // CHECK-NEXT: [[IS_LAST:%.*]] = arith.cmpi eq, [[K]], [[LAST_ITER]]
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma [[A_BUF]], [[B_T]], [[ACC_BUF]][], [[TRUE]], [[TRUE]], [[READY_MBAR]][%true], [[DONE_MBAR0]][[[IS_LAST]]] {ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_T_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared_trans, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>

    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: [[IDX_INCR:%.*]] = arith.addi [[IDX]], [[C1]]
    // CHECK-NEXT: [[PHASE_INCR:%.*]] = arith.xori [[PHASE]], [[C1]]
    // CHECK-NEXT: [[ROLLOVER:%.*]] = arith.cmpi eq, [[IDX_INCR]], [[C2]]
    // CHECK-NEXT: [[IDX_NEXT:%.*]] = arith.select [[ROLLOVER]], [[C0]], [[IDX_INCR]]
    // CHECK-NEXT: [[PHASE_NEXT:%.*]] = arith.select [[ROLLOVER]], [[PHASE_INCR]], [[PHASE]]

    // CHECK-NEXT: yield %{{[0-9]+}}, [[IDX_NEXT]], [[PHASE_NEXT]]
    scf.yield %c : tensor<128x128xf32, #acc_layout>

  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 1 : i32]
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
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK: [[ACC_ALLOC:%.*]], %{{.*}} = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32
  // CHECK-NEXT: [[ACC:%.*]] = ttg.memdesc_subview [[ACC_ALLOC]][%c0_i32
  // CHECK-NEXT: tmem_store [[ZERO]], [[ACC]]

  // CHECK-NEXT: [[DONE_MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64
  // CHECK-NEXT: [[DONE_MBAR0:%.*]] = ttg.memdesc_subview [[DONE_MBAR]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[DONE_MBAR0]], 1

  // CHECK-NEXT: scf.for
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK-NEXT: get_ptrs
    %a_ptrs, %b_ptrs = "get_ptrs"(%k) : (i32) -> (tensor<128x64x!tt.ptr<f16>, #oper_layout>, tensor<64x128x!tt.ptr<f16>, #oper_layout>)
    // CHECK-NEXT: tt.load
    %a = tt.load %a_ptrs : tensor<128x64x!tt.ptr<f16>, #oper_layout>
    // CHECK-NEXT: tt.load
    %b = tt.load %b_ptrs : tensor<64x128x!tt.ptr<f16>, #oper_layout>

    // CHECK-NEXT: ttg.local_alloc
    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    // CHECK-NEXT: ttg.local_alloc
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK-NEXT: [[IS_LAST:%.*]] = arith.cmpi eq, %{{.*}}, %c31_i32
    // CHECK-NEXT: ttng.tc_gen5_mma %{{.*}}, [[ACC]][], %true, %true, [[DONE_MBAR0]][[[IS_LAST]]] {ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    scf.yield %c : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 1 : i32]
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
// AWS: num_warps(4)
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
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  // CHECK-DAG: [[K_TILES:%.*]] = arith.constant 32 : i32
  %k_tiles = arith.constant 32 : i32

  // CHECK:      [[ACC_BUFS:%.*]], [[ACC_TOK:%.*]] = ttng.tmem_alloc : () -> (!ttg.memdesc<2x128x128xf32
  // CHECK-NEXT: [[ACC_BUF0:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][%c0_i32, %c0_i32, %c0_i32]
  // CHECK-NEXT: [[INIT_TOK:%.*]] = ttng.tmem_store [[ZERO]], [[ACC_BUF0]][[[ACC_TOK]]]

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK:      [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[ACC_READY_BUF0:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF0]], 1
  // CHECK-NEXT: [[ACC_READY_BUF1:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][%c1_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF1]], 1

  // CHECK-NEXT: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[ACC_EMPTY_BUF0:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_EMPTY_BUF0]], 1
  // CHECK-NEXT: [[ACC_EMPTY_BUF1:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][%c1_i32]
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

    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[ACC_INDEX]], %c0_i32, %c0_i32]

    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][[[ACC_INDEX]]]
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], %true, %true, {{.*}}, [[CUR_ACC_READY_BAR]][%true] {ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>

    // CHECK-NEXT: [[ACC_RESET:%.*]] = "acc_reset"
    %acc_reset = "acc_reset"() : () -> tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BAR]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[C:%.*]], [[LOAD_TOK:%.*]] = ttng.tmem_load [[ACC_BUF]][] {ttg.partition = 0 : i32}
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[ACC_INDEX]]]
    // CHECK-NEXT: ttng.arrive_barrier [[CUR_ACC_EMPTY_BAR]], 1 {ttg.partition = 0 : i32}
    "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()

    // CHECK-NEXT: [[ACC_INDEX_INCR:%.*]] = arith.addi [[ACC_INDEX]], %c1_i32
    // CHECK-NEXT: [[ACC_PHASE_INCR:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_ROLLVER:%.*]] = arith.cmpi eq, [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[ACC_NEXT_INDEX:%.*]] = arith.select [[ACC_ROLLVER]], %c0_i32, [[ACC_INDEX_INCR]]
    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.select [[ACC_ROLLVER]], [[ACC_PHASE_INCR]], [[ACC_PHASE]]

    // CHECK-NEXT: "acc_user"([[C]]) {ttg.partition = 0 : i32}

    // CHECK-NEXT: [[NEXT_ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[ACC_NEXT_INDEX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[ACC_NEXT_INDEX]]]
    // CHECK-NEXT: ttng.wait_barrier [[NEXT_ACC_EMPTY_BAR]], [[ACC_NEXT_PHASE]], %true {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[STORE_TOK:%.*]] = ttng.tmem_store [[ACC_RESET]], [[NEXT_ACC_BUF]][], %true {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_NEXT_INDEX]], [[ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [2 : i32, 1 : i32, 0 : i32]
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  tt.return
}

// FUNC-LABEL: @matmul_tma_acc_with_conditional_user

// TMEM: ttng.tmem_alloc
// TMEM: scf.for

// AWS: ttg.warp_specialize
// AWS: num_warps(4)
// AWS: num_warps(4)
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
  %k_tiles = arith.constant 32 : i32

  // CHECK: [[ACC_BUFS:%.*]], [[ACC_TOK:%.*]] = ttng.tmem_alloc
  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK: [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64

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

    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[ACC_INDEX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][[[ACC_INDEX]]]
    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = arith.cmpi
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], %true, %true, {{.*}}, [[CUR_ACC_READY_BAR]][[[DO_EPILOGUE]]] {ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: [[ACC_RESET:%.*]] = "acc_reset"
    %acc_reset = "acc_reset"() : () -> tensor<128x128xf32, #acc_layout>
    %do_epilogue = arith.cmpi eq, %k, %c0_i32 : i32

    // CHECK-NEXT: scf.if [[DO_EPILOGUE]]
    scf.if %do_epilogue {
      // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BAR]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: [[C:%.*]], [[USER_TOK:%.*]] = ttng.tmem_load [[ACC_BUF]][]
      // CHECK-NEXT: "acc_user"([[C]])
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
      // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[ACC_INDEX]]]
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

    // CHECK-NEXT: [[ACC_NEXT_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[EPILOGUE_ACC_NEXT_INDEX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[EPILOGUE_ACC_NEXT_INDEX]]]
    // CHECK-NEXT: ttng.wait_barrier [[NEXT_ACC_EMPTY_BAR]], [[EPILOGUE_ACC_NEXT_PHASE]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: ttng.tmem_store [[ACC_RESET]], [[ACC_NEXT_BUF]][], %true {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield %{{[0-9]+}}, %{{[0-9]+}}, [[EPILOGUE_ACC_NEXT_INDEX]], [[EPILOGUE_ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [2 : i32, 1 : i32, 0 : i32]
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
  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK: [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64

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

    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[ACC_INDEX]], %c0_i32, %c0_i32]

    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][[[ACC_INDEX]]]
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], %true, %true, {{.*}}, [[CUR_ACC_READY_BAR]][%true] {ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = arith.cmpi
    %do_epilogue = arith.cmpi eq, %k, %c0_i32 : i32
    %acc_reset = arith.select %do_epilogue, %zero, %c : tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BAR]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[C:%.*]], [[LOAD_TOK:%.*]] = ttng.tmem_load [[ACC_BUF]][] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[ACC_INDEX]]]
    // CHECK-NEXT: ttng.arrive_barrier [[CUR_ACC_EMPTY_BAR]], 1 {ttg.partition = 0 : i32}

    // CHECK-NEXT: [[ACC_INDEX_INCR:%.*]] = arith.addi [[ACC_INDEX]], %c1_i32
    // CHECK-NEXT: [[ACC_PHASE_INCR:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_ROLLVER:%.*]] = arith.cmpi eq, [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[ACC_NEXT_INDEX:%.*]] = arith.select [[ACC_ROLLVER]], %c0_i32, [[ACC_INDEX_INCR]]
    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.select [[ACC_ROLLVER]], [[ACC_PHASE_INCR]], [[ACC_PHASE]]

    // CHECK-NEXT: "acc_user"([[C]])
    "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()

    // CHECK-NEXT: [[NEXT_ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[ACC_NEXT_INDEX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[ACC_NEXT_INDEX]]]
    // CHECK-NEXT: ttng.wait_barrier [[NEXT_ACC_EMPTY_BAR]], [[ACC_NEXT_PHASE]], %true {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[STORE_TOK:%.*]] = ttng.tmem_store [[ZERO]], [[NEXT_ACC_BUF]][], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield {{.*}} [[ACC_NEXT_INDEX]], [[ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [2 : i32, 1 : i32, 0 : i32]
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
  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK: [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64

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

    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[ACC_INDEX]], %c0_i32, %c0_i32]

    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][[[ACC_INDEX]]]
    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = arith.cmpi
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], %true, %true, {{.*}}, [[CUR_ACC_READY_BAR]][[[DO_EPILOGUE]]] {ttg.partition = 1 : i32}
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
      // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[ACC_INDEX]]]
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

    // CHECK-NEXT: [[NEXT_ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[EPILOGUE_ACC_NEXT_INDEX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[EPILOGUE_ACC_NEXT_INDEX]]]
    // CHECK-NEXT: ttng.wait_barrier [[NEXT_ACC_EMPTY_BAR]], [[EPILOGUE_ACC_NEXT_PHASE]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[STORE_TOK:%.*]] = ttng.tmem_store [[ZERO]], [[NEXT_ACC_BUF]][], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield {{.*}} [[EPILOGUE_ACC_NEXT_INDEX]], [[EPILOGUE_ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [2 : i32, 1 : i32, 0 : i32]
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
  // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][%c0_i32, %c0_i32, %c0_i32]
  // CHECK-NEXT: [[INIT_TOK:%.*]] = ttng.tmem_store [[ZERO]], [[ACC_BUF]][[[ACC_TOK]]], %true

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK:      [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64
  // CHECK-NEXT: [[ACC_READY_BUF0:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF0]], 1

  // CHECK-NEXT: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64
  // CHECK-NEXT: [[ACC_EMPTY_BUF0:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][%c0_i32]
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

    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = arith.cmpi eq, [[K:%.*]], %c0_i32
    // CHECK-NEXT: [[MMA_TOK:%.*]] = ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], [[FLAG]], %true, {{.*}}, [[ACC_READY_BUF0]][[[DO_EPILOGUE]]] {ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem[%c_tok], %flag, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %c_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    %do_epilogue = arith.cmpi eq, %k, %c0_i32 : i32
    // CHECK-NEXT: [[NEXT_FLAG:%.*]] = arith.cmpi ne, [[K]], %c0_i32

    %use_acc = arith.select %do_epilogue, %false, %true : i1

    // CHECK-NEXT: scf.if [[DO_EPILOGUE]]
    scf.if %do_epilogue {
      // CHECK-NEXT: ttng.wait_barrier [[ACC_READY_BUF0]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: "some_op"()
      "some_op"() : () -> ()
      // CHECK-NEXT: [[C:%.*]], [[USER_TOK:%.*]] = ttng.tmem_load [[ACC_BUF]][]
      // CHECK-NEXT: "acc_user"([[C]])
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
      // CHECK-NEXT: ttng.arrive_barrier [[ACC_EMPTY_BUF0]], 1 {ttg.partition = 0 : i32}
    // CHECK-NEXT: }
    }

    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_PHASE:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_PHASE]], [[ACC_PHASE]]
    // CHECK-NEXT: ttng.wait_barrier [[ACC_EMPTY_BUF0]], [[EPILOGUE_ACC_NEXT_PHASE]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield [[NEXT_FLAG]], %{{[0-9]+}}, %{{[0-9]+}}, [[EPILOGUE_ACC_NEXT_PHASE]]
    scf.yield %c, %use_acc : tensor<128x128xf32, #acc_layout>, i1
  // CHECK-NEXT: tt.scheduled_max_stage = 2 : i32
  // CHECK-SAME: ttg.partition.stages = [2 : i32, 1 : i32, 0 : i32]
  } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32}

  tt.return
}

// FUNC-LABEL: @matmul_scaled_rhs_scales_tma
// CHECK-LABEL: @matmul_scaled_rhs_scales_tma
tt.func @matmul_scaled_rhs_scales_tma(
  %k_tiles: i32,
  %off_m: i32,
  %off_n: i32,
  %a_desc: !tt.tensordesc<tensor<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>>>,
  %b_desc: !tt.tensordesc<tensor<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>>>,
  %b_scale_desc: !tt.tensordesc<tensor<128x8xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>>>
) {
  %true = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %BLOCK_K = arith.constant 64 : i32
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>

  %a_scales_const = arith.constant dense<127> : tensor<128x8xi8, #oper_layout>
  %a_scales_tmem = ttng.tmem_alloc %a_scales_const : (tensor<128x8xi8, #oper_layout>) -> !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<3xi64,
  // CHECK-NOT: ttg.local_alloc : () -> !ttg.memdesc<3xi64,

  %result = scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    %off_k = arith.muli %k, %BLOCK_K : i32

    // CHECK: ttng.wait_barrier
    // CHECK-COUNT-3: async_tma_copy_global_to_local {{.*}} {ttg.partition = 0 : i32}
    %a_reg = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>>> -> tensor<128x64xf8E4M3FN, #oper_layout>
    %b_reg = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>>> -> tensor<128x64xf8E4M3FN, #oper_layout>
    %b_scales_reg = tt.descriptor_load %b_scale_desc[%off_m, %c0_i32] : !tt.tensordesc<tensor<128x8xi8, #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>>> -> tensor<128x8xi8, #oper_layout>

    %a_sh = ttg.local_alloc %a_reg : (tensor<128x64xf8E4M3FN, #oper_layout>) -> !ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>, #smem>
    %b_sh_raw = ttg.local_alloc %b_reg : (tensor<128x64xf8E4M3FN, #oper_layout>) -> !ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>, #smem>
    // CHECK-NEXT: memdesc_trans {{.*}} ttg.partition = 1 : i32
    %b_sh = ttg.memdesc_trans %b_sh_raw {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>, #smem>

    // CHECK-NEXT: wait_barrier {{.*}} {ttg.partition = 1 : i32}

    %b_scales_tmem = ttng.tmem_alloc %b_scales_reg : (tensor<128x8xi8, #oper_layout>) -> !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>

    %c_tmem, %c_tok = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)

    // CHECK-NEXT: [[LAST_ITER:%.*]] = arith.subi %{{.*}}, %c1_i32
    // CHECK-NEXT: [[IS_LAST:%.*]] = arith.cmpi eq, %arg6, [[LAST_ITER]]
    // CHECK-NEXT: tc_gen5_mma_scaled {{.*}} {ttg.partition = 1 : i32}
    %mma_tok = ttng.tc_gen5_mma_scaled %a_sh, %b_sh, %c_tmem[%c_tok], %a_scales_tmem, %b_scales_tmem, %true, %true lhs = e4m3 rhs = e4m3 : !ttg.memdesc<128x64xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>

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
  // CHECK-NEXT: [[ACC_BUF0:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][%c0_i32, %c0_i32, %c0_i32]
  // CHECK-NEXT: ttng.tmem_store [[ZERO]], [[ACC_BUF0]]

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<4x{{.*}}xf16,
  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<4xi64
  // CHECK-COUNT-4: ttng.arrive_barrier

  // CHECK:      [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[ACC_READY_BUF0:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF0]], 1
  // CHECK-NEXT: [[ACC_READY_BUF1:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][%c1_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF1]], 1

  // CHECK-NEXT: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[ACC_EMPTY_BUF0:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_EMPTY_BUF0]], 1
  // CHECK-NEXT: [[ACC_EMPTY_BUF1:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][%c1_i32]
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

    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[ACC_INDEX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: [[CUR_ACC_READY_BUF:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][[[ACC_INDEX]]]

    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = arith.cmpi eq, [[K:%.*]], %c0_i32
    // CHECK-NEXT: ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]][], [[FLAG]], %true, {{.*}}, [[CUR_ACC_READY_BUF]][[[DO_EPILOGUE]]] {ttg.partition = 1 : i32}
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
      // CHECK-NEXT: [[CUR_ACC_EMPTY_BUF:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[ACC_INDEX]]]
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

    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BUF:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[EPILOGUE_ACC_NEXT_INDEX]]]
    // CHECK-NEXT: ttng.wait_barrier [[NEXT_ACC_EMPTY_BUF]], [[EPILOGUE_ACC_NEXT_PHASE]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}

    // CHECK: arith.addi
    // CHECK-NOT: arith.addi

    // CHECK: scf.yield [[NEXT_FLAG]], %{{[0-9]+}}, %{{[0-9]+}}, [[EPILOGUE_ACC_NEXT_INDEX]], [[EPILOGUE_ACC_NEXT_PHASE]]
    scf.yield %c, %use_acc : tensor<128x128xf32, #acc_layout>, i1
  // CHECK-NEXT: tt.scheduled_max_stage = 4 : i32
  // CHECK-SAME: ttg.partition.stages = [2 : i32, 1 : i32, 0 : i32]
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
    // CHECK: async_tma_copy_global_to_local %{{[0-9]+}}[[[I]], [[IDX]]]
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

  // CHECK-COUNT-2: local_alloc : () -> !ttg.memdesc<3xi64,

  // CHECK:      [[EMPTY_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64,
  // CHECK-NEXT: [[EMPTY_BAR0:%.*]] = ttg.memdesc_subview [[EMPTY_BARS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[EMPTY_BAR0]], 1

  // CHECK-NEXT: [[READY_BARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64,
  // CHECK-NEXT: [[READY_BAR0:%.*]] = ttg.memdesc_subview [[READY_BARS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[READY_BAR0]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[READY_BAR0]], 1
  // CHECK-NEXT: ttng.arrive_barrier [[EMPTY_BAR0]], 1

  // CHECK-NEXT: [[OPERAND:%.*]] = ttg.local_alloc {{.*}} : () -> !ttg.memdesc<64x128xf16,

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
    // CHECK-NEXT: arrive_barrier
    // CHECK-NEXT: [[RESULTS:%.*]]:2 = "some_producer"
    %rhs_reg, %next_acc = "some_producer"(%loaded, %acc) : (tensor<64x128xf16, #oper_layout>, tensor<128x128xf32, #acc_layout>) -> (tensor<64x128xf16, #oper_layout>, tensor<128x128xf32, #acc_layout>)
    // CHECK-NEXT: local_store [[RESULTS]]#0, [[OPERAND]]
    // CHECK-NEXT: tmem_store [[RESULTS]]#1, [[ACC_TMEM]]
    // CHECK-NEXT: arrive_barrier [[EMPTY_BAR0]]
    %rhs = ttg.local_alloc %rhs_reg : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

    %acc_tmem, %acc_tok = ttng.tmem_alloc %next_acc : (tensor<128x128xf32, #acc_layout>) -> (!ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: wait_barrier [[EMPTY_BAR0]]
    // CHECK-NEXT: ttng.tc_gen5_mma {{.*}} [[READY_BAR0]][%true]
    %mma_tok = ttng.tc_gen5_mma %lhs, %rhs, %acc_tmem[%acc_tok], %true, %true : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    %c, %load_tok = ttng.tmem_load %acc_tmem[%mma_tok] : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    scf.yield %c : tensor<128x128xf32, #acc_layout>
  } {tt.warp_specialize, tt.num_stages = 3 : i32}
  "use"(%out) : (tensor<128x128xf32, #acc_layout>) -> ()
  tt.return
}

}
