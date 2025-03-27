// RUN: triton-opt %s -allow-unregistered-dialect -verify-diagnostics -tritongpu-load-mma-specialization -int-range-optimizations -canonicalize -cse -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -allow-unregistered-dialect -verify-diagnostics -tritongpu-automatic-warp-specialization | FileCheck %s --check-prefix=AWS

#acc_layout = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#oper_layout = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK-DAG: [[SHARED:#.*]] = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_trans = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-DAG: [[ACC_TMEM:#.*]] = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#acc_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

// AWS-LABEL: @warp_specialize_tma_matmul
// AWS: ttg.warp_specialize

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

  // CHECK:      [[A_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, [[SHARED]]
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

  // CHECK-NEXT: [[MMA_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[MMA_MBAR0:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][[[C0]]]
  // CHECK-NEXT: ttng.init_barrier [[MMA_MBAR0]], 1
  // CHECK-NEXT: [[MMA_MBAR1:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][[[C1]]]
  // CHECK-NEXT: ttng.init_barrier [[MMA_MBAR1]], 1

  // CHECK-NEXT: [[ACC_BUFS:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, [[ACC_TMEM]], #ttng.tensor_memory, mutable>
  // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[C0]], [[C0]], [[C0]]]
  // CHECK-NEXT: ttng.tmem_store [[ZERO]], [[ACC_BUF]]

  // CHECK-NEXT: {{[0-9]+}}:4 = scf.for [[K:%arg[0-9]+]] = [[C0]] to [[K_TILES]] step [[C1]]
  // CHECK-SAME: [[IDX:%arg[0-9]+]] = [[C0]]
  // CHECK-SAME: [[PHASE:%arg[0-9]+]] = [[C0]]
  // CHECK-SAME: [[MMA_IDX:%arg[0-9]+]] = [[C0]]
  // CHECK-SAME: [[MMA_PHASE:%arg[0-9]+]] = [[C0]]
  // CHECK-SAME: -> (i32, i32, i32, i32)
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
    // CHECK-NEXT: ttng.wait_barrier [[OPER_MBAR]], [[PHASE]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: [[B_T:%.*]] = ttg.memdesc_trans [[B_BUF]] {order = array<i32: 1, 0>, ttg.partition = 1 : i32}
    %b_T_shared = ttg.memdesc_trans %b_shared {order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared_trans, #smem>
    %c_tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    // CHECK-NEXT: [[MMA_MBAR:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][[[MMA_IDX]]]
    // CHECK-NEXT: ttng.tc_gen5_mma [[A_BUF]], [[B_T]], [[ACC_BUF]], [[TRUE]], [[TRUE]], [[MMA_MBAR]] {ttg.partition = 1 : i32}
    // CHECK-NEXT: ttng.wait_barrier [[MMA_MBAR]], [[MMA_PHASE]] {ttg.partition = 2 : i32}
    // CHECK-NEXT: ttng.arrive_barrier [[READY_MBAR]], 1 {ttg.partition = 2 : i32}
    ttng.tc_gen5_mma %a_shared, %b_T_shared, %c_tmem, %true, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared_trans, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()

    %c = ttng.tmem_load %c_tmem : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: [[IDX_INCR:%.*]] = arith.addi [[IDX]], [[C1]]
    // CHECK-NEXT: [[PHASE_INCR:%.*]] = arith.xori [[PHASE]], [[C1]]
    // CHECK-NEXT: [[ROLLOVER:%.*]] = arith.cmpi eq, [[IDX_INCR]], [[C2]]
    // CHECK-NEXT: [[IDX_NEXT:%.*]] = arith.select [[ROLLOVER]], [[C0]], [[IDX_INCR]]
    // CHECK-NEXT: [[PHASE_NEXT:%.*]] = arith.select [[ROLLOVER]], [[PHASE_INCR]], [[PHASE]]

    // CHECK-NEXT: [[MMA_IDX_INCR:%.*]] = arith.addi [[MMA_IDX]], [[C1]]
    // CHECK-NEXT: [[MMA_PHASE_INCR:%.*]] = arith.xori [[MMA_PHASE]], [[C1]]
    // CHECK-NEXT: [[MMA_ROLLOVER:%.*]] = arith.cmpi eq, [[MMA_IDX_INCR]], [[C2]]
    // CHECK-NEXT: [[MMA_IDX_NEXT:%.*]] = arith.select [[MMA_ROLLOVER]], [[C0]], [[MMA_IDX_INCR]]
    // CHECK-NEXT: [[MMA_PHASE_NEXT:%.*]] = arith.select [[MMA_ROLLOVER]], [[MMA_PHASE_INCR]], [[MMA_PHASE]]

    // CHECK-NEXT: yield [[IDX_NEXT]], [[PHASE_NEXT]], [[MMA_IDX_NEXT]], [[MMA_PHASE_NEXT]]
    scf.yield %c : tensor<128x128xf32, #acc_layout>

  // CHECK-NEXT: ttg.partition.stages = [0 : i32, 2 : i32, 3 : i32]
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  // CHECK-NEXT: [[RESULT:%.*]] = ttng.tmem_load [[ACC_BUF]]
  // CHECK-NEXT: ttng.inval_barrier [[MMA_MBAR0]]
  // CHECK-NEXT: ttng.inval_barrier [[MMA_MBAR1]]
  // CHECK-NEXT: ttg.local_dealloc [[MMA_MBARS]]

  // CHECK-NEXT: ttng.inval_barrier [[OPER_MBAR0]]
  // CHECK-NEXT: ttng.inval_barrier [[OPER_MBAR1]]
  // CHECK-NEXT: ttg.local_dealloc [[OPER_MBARS]]

  // CHECK-NEXT: ttng.inval_barrier [[READY_MBAR0]]
  // CHECK-NEXT: ttng.inval_barrier [[READY_MBAR1]]
  // CHECK-NEXT: ttg.local_dealloc [[READY_MBARS]]

  // CHECK-NEXT: ttg.local_dealloc [[B_BUFS]]
  // CHECK-NEXT: ttg.local_dealloc [[A_BUFS]]

  // CHECK-NEXT: "use"([[RESULT]])
  "use"(%result) : (tensor<128x128xf32, #acc_layout>) -> ()
  tt.return
}

// AWS-LABEL: @unsupported_multiple_dot_ops
// CHECK: @unsupported_multiple_dot_ops
tt.func @unsupported_multiple_dot_ops() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // expected-warning @below {{failed to warp specialize: more than one `tt.dot` found in the loop}}
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc0 = %zero, %acc1 = %zero) -> (tensor<128x128xf32, #acc_layout>, tensor<128x128xf32, #acc_layout>) : i32 {
    %a, %b = "load"() : () -> (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>)

    %c0 = ttng.tmem_alloc %acc0 : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %c0, %true, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %cnext0 = ttng.tmem_load %c0 : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    %c1 = ttng.tmem_alloc %acc0 : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a, %b, %c1, %true, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %cnext1 = ttng.tmem_load %c1 : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    scf.yield %cnext0, %cnext1 : tensor<128x128xf32, #acc_layout>, tensor<128x128xf32, #acc_layout>
  } {tt.warp_specialize}

  tt.return
}

// AWS-LABEL: @unsupported_load
// CHECK: @unsupported_load
tt.func @unsupported_load() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // expected-warning @below {{failed to warp specialize: could not find TMA loads for `tt.dot` operands}}
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    %a_ptrs, %b_ptrs = "get_ptrs"(%k) : (i32) -> (tensor<128x64x!tt.ptr<f16>, #oper_layout>, tensor<64x128x!tt.ptr<f16>, #oper_layout>)
    %a = tt.load %a_ptrs : tensor<128x64x!tt.ptr<f16>, #oper_layout>
    %b = tt.load %b_ptrs : tensor<64x128x!tt.ptr<f16>, #oper_layout>

    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

    %c_tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem, %true, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %c = ttng.tmem_load %c_tmem : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>

    scf.yield %c : tensor<128x128xf32, #acc_layout>
  } {tt.warp_specialize}

  tt.return
}

// AWS-LABEL: @cant_pipeline_mma
// CHECK: @cant_pipeline_mma
tt.func @cant_pipeline_mma(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // expected-warning @below {{failed to warp specialize: could not determine if the MMA op can be pipelined}}
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 : i32 {
    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)
    %a = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

    %c_tmem = ttng.tmem_alloc %zero : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem, %true, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
  } {tt.warp_specialize}

  tt.return
}

// AWS-LABEL: @invalid_acc_reset
// CHECK: @invalid_acc_reset
tt.func @invalid_acc_reset(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // expected-warning @below {{failed to warp specialize: accumulator reset does not occur after the `tt.dot`}}
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    %off_m, %off_n, %off_k = "get_offsets"(%k) : (i32) -> (i32, i32, i32)
    %a = tt.descriptor_load %a_desc[%off_m, %off_k] : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #oper_layout>
    %b = tt.descriptor_load %b_desc[%off_n, %off_k] : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #oper_layout>

    %a_shared = ttg.local_alloc %a : (tensor<128x64xf16, #oper_layout>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %b_shared = ttg.local_alloc %b : (tensor<64x128xf16, #oper_layout>) -> !ttg.memdesc<64x128xf16, #shared, #smem>

    %c_tmem = ttng.tmem_alloc %zero : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem, %true, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %c = ttng.tmem_load %c_tmem : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    scf.yield %c : tensor<128x128xf32, #acc_layout>
  } {tt.warp_specialize}

  tt.return
}

// AWS-LABEL: @matmul_tma_acc_with_unconditional_user
// AWS: ttg.warp_specialize

// CHECK: @matmul_tma_acc_with_unconditional_user
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

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK:      [[MMA_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[MMA_MBAR0:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[MMA_MBAR0]], 1
  // CHECK-NEXT: [[MMA_MBAR1:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][%c1_i32]
  // CHECK-NEXT: ttng.init_barrier [[MMA_MBAR1]], 1

  // CHECK-NEXT: [[ACC_BUFS:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, [[ACC_TMEM]], #ttng.tensor_memory, mutable>
  // CHECK-NEXT: [[ACC_BUF0:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][%c0_i32, %c0_i32, %c0_i32]
  // CHECK-NEXT: ttng.tmem_store [[ZERO]], [[ACC_BUF0]]

  // CHECK-NEXT: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[ACC_EMPTY_BUF0:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_EMPTY_BUF0]], 1
  // CHECK-NEXT: [[ACC_EMPTY_BUF1:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][%c1_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_EMPTY_BUF1]], 1

  // CHECK-NEXT: [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[ACC_READY_BUF0:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF0]], 1
  // CHECK-NEXT: [[ACC_READY_BUF1:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][%c1_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF1]], 1

  // CHECK-NEXT: ttng.arrive_barrier [[ACC_EMPTY_BUF0]], 1

  // CHECK-NEXT: {{[0-9]+}}:6 = scf.for [[K:%arg[0-9]+]] = %c0_i32 to [[K_TILES]] step %c1_i32
  // CHECK-SAME: [[LOAD_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LOAD_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[MMA_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[MMA_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[ACC_INDEX]]]
    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][[[ACC_INDEX]]]

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
    %c_tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>

    // CHECK-NEXT: [[MMA_MBAR:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][[[MMA_INDEX]]]
    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[ACC_INDEX]], %c0_i32, %c0_i32]

    // CHECK-NEXT: ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]], %true, %true, [[MMA_MBAR]] {ttg.partition = 1 : i32}
    ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem, %true, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    // CHECK-NEXT: ttng.wait_barrier [[MMA_MBAR]], [[MMA_PHASE]] {ttg.partition = 3 : i32}
    // CHECK-NEXT: ttng.arrive_barrier {{.*}} {ttg.partition = 3 : i32}

    // CHECK-NEXT: ttng.arrive_barrier [[CUR_ACC_READY_BAR]], 1, %true {ttg.partition = 3 : i32}
    // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_EMPTY_BAR]], [[ACC_PHASE]], %true {ttg.partition = 1 : i32}

    // CHECK-NEXT: [[ACC_RESET:%.*]] = "acc_reset"
    %acc_reset = "acc_reset"() : () -> tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: [[ACC_INDEX_INCR:%.*]] = arith.addi [[ACC_INDEX]], %c1_i32
    // CHECK-NEXT: [[NEXT_ACC_INDEX:%.*]] = arith.remui [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[NEXT_ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[NEXT_ACC_INDEX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: ttng.tmem_store [[ACC_RESET]], [[NEXT_ACC_BUF]], %true {ttg.partition = 1 : i32}

    // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BAR]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[C:%.*]] = ttng.tmem_load [[ACC_BUF]] {ttg.partition = 0 : i32}
    %c = ttng.tmem_load %c_tmem : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[NEXT_ACC_INDEX]]]
    // CHECK-NEXT: ttng.arrive_barrier [[NEXT_ACC_EMPTY_BAR]], 1 {ttg.partition = 0 : i32}
    // CHECK-NEXT: "acc_user"([[C]])
    "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()

    // CHECK: arith.addi

    // CHECK:      [[MMA_INDEX_INCR:%.*]] = arith.addi [[MMA_INDEX]], %c1_i32
    // CHECK-NEXT: [[MMA_PHASE_INCR:%.*]] = arith.xori [[MMA_PHASE]], %c1_i32
    // CHECK-NEXT: [[MMA_ROLLVER:%.*]] = arith.cmpi eq, [[MMA_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[MMA_NEXT_INDEX:%.*]] = arith.select [[MMA_ROLLVER]], %c0_i32, [[MMA_INDEX_INCR]]
    // CHECK-NEXT: [[MMA_NEXT_PHASE:%.*]] = arith.select [[MMA_ROLLVER]], [[MMA_PHASE_INCR]], [[MMA_PHASE]]

    // CHECK-NEXT: [[ACC_PHASE_INCR:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_ROLLVER:%.*]] = arith.cmpi eq, [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[ACC_NEXT_INDEX:%.*]] = arith.select [[ACC_ROLLVER]], %c0_i32, [[ACC_INDEX_INCR]]
    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.select [[ACC_ROLLVER]], [[ACC_PHASE_INCR]], [[ACC_PHASE]]

    // CHECK-NEXT: scf.yield %{{[0-9]+}}, %{{[0-9]+}}, [[MMA_NEXT_INDEX]], [[MMA_NEXT_PHASE]], [[ACC_NEXT_INDEX]], [[ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [4 : i32, 2 : i32, 0 : i32, 4 : i32]
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  tt.return
}

// AWS-LABEL: @matmul_tma_acc_with_conditional_user
// AWS: ttg.warp_specialize

// CHECK: @matmul_tma_acc_with_conditional_user
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

  // CHECK-COUNT-3: ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK: [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK:      {{[0-9]+}}:6 = scf.for [[K:%arg[0-9]+]]
  // CHECK-SAME: [[LOAD_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LOAD_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[MMA_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[MMA_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[ACC_INDEX]]]
    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][[[ACC_INDEX]]]

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
    %c_tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>

    // CHECK-NEXT: [[MMA_MBAR:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][[[MMA_INDEX]]]
    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[ACC_INDEX]], %c0_i32, %c0_i32]

    // CHECK-NEXT: ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]], %true, %true, [[MMA_MBAR]] {ttg.partition = 1 : i32}
    ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem, %true, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %c = ttng.tmem_load %c_tmem : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    // CHECK-NEXT: ttng.wait_barrier [[MMA_MBAR]], [[MMA_PHASE]] {ttg.partition = 3 : i32}
    // CHECK-NEXT: ttng.arrive_barrier {{.*}} {ttg.partition = 3 : i32}

    // CHECK-NEXT: [[ACC_RESET:%.*]] = "acc_reset"
    %acc_reset = "acc_reset"() : () -> tensor<128x128xf32, #acc_layout>
    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = "epilogue_cond"([[K]])
    %do_epilogue = "epilogue_cond"(%k) : (i32) -> i1

    // CHECK-NEXT: ttng.arrive_barrier [[CUR_ACC_READY_BAR]], 1, [[DO_EPILOGUE]] {ttg.partition = 3 : i32}
    // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_EMPTY_BAR]], [[ACC_PHASE]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}

    // CHECK-NEXT: [[ACC_INDEX_INCR:%.*]] = arith.addi [[ACC_INDEX]], %c1_i32
    // CHECK-NEXT: [[NEXT_ACC_INDEX:%.*]] = arith.remui [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[NEXT_ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[NEXT_ACC_INDEX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: ttng.tmem_store [[ACC_RESET]], [[NEXT_ACC_BUF]], %true {ttg.partition = 1 : i32}

    // CHECK-NEXT: scf.if [[DO_EPILOGUE]]
    scf.if %do_epilogue {
      // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BAR]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: [[C:%.*]] = ttng.tmem_load [[ACC_BUF]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[NEXT_ACC_INDEX]]]
      // CHECK-NEXT: ttng.arrive_barrier [[NEXT_ACC_EMPTY_BAR]], 1 {ttg.partition = 0 : i32}
      // CHECK-NEXT: "acc_user"([[C]])
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
    // CHECK-NEXT: } {ttg.partition = 0 : i32}
    }

    // CHECK-COUNT-2: arith.addi

    // CHECK:      [[ACC_PHASE_INCR:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_ROLLVER:%.*]] = arith.cmpi eq, [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[ACC_NEXT_INDEX:%.*]] = arith.select [[ACC_ROLLVER]], %c0_i32, [[ACC_INDEX_INCR]]
    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.select [[ACC_ROLLVER]], [[ACC_PHASE_INCR]], [[ACC_PHASE]]

    // CHECK-NEXT: scf.yield {{.*}} [[ACC_NEXT_INDEX]], [[ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [4 : i32, 2 : i32, 0 : i32, 4 : i32]
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  tt.return
}

// AWS-LABEL: @matmul_tma_acc_with_conditional_def
// AWS: ttg.warp_specialize

// CHECK: @matmul_tma_acc_with_conditional_def
// CHECK-SAME: [[A_DESC:%arg[0-9]+]]
// CHECK-SAME: [[B_DESC:%arg[0-9]+]]
tt.func @matmul_tma_acc_with_conditional_def(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK-COUNT-3: ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK: [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK:      {{[0-9]+}}:6 = scf.for [[K:%arg[0-9]+]]
  // CHECK-SAME: [[LOAD_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LOAD_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[MMA_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[MMA_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[ACC_INDEX]]]
    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][[[ACC_INDEX]]]

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
    %c_tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>

    // CHECK-NEXT: [[MMA_MBAR:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][[[MMA_INDEX]]]
    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[ACC_INDEX]], %c0_i32, %c0_i32]

    // CHECK-NEXT: ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]], %true, %true, [[MMA_MBAR]] {ttg.partition = 1 : i32}
    ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem, %true, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %c = ttng.tmem_load %c_tmem : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    // CHECK-NEXT: ttng.wait_barrier [[MMA_MBAR]], [[MMA_PHASE]] {ttg.partition = 3 : i32}
    // CHECK-NEXT: ttng.arrive_barrier {{.*}} {ttg.partition = 3 : i32}

    // CHECK-NEXT: ttng.arrive_barrier [[CUR_ACC_READY_BAR]], 1, %true {ttg.partition = 3 : i32}
    // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_EMPTY_BAR]], [[ACC_PHASE]], %true {ttg.partition = 1 : i32}

    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = "epilogue_cond"([[K]])
    %do_epilogue = "epilogue_cond"(%k) : (i32) -> i1
    %acc_reset = arith.select %do_epilogue, %zero, %c : tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: [[ACC_INDEX_INCR:%.*]] = arith.addi [[ACC_INDEX]], %c1_i32
    // CHECK-NEXT: [[NEXT_ACC_INDEX:%.*]] = arith.remui [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[NEXT_ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[NEXT_ACC_INDEX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: ttng.tmem_store [[ZERO]], [[NEXT_ACC_BUF]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}

    // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BAR]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[C:%.*]] = ttng.tmem_load [[ACC_BUF]] {ttg.partition = 0 : i32}
    // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[NEXT_ACC_INDEX]]]
    // CHECK-NEXT: ttng.arrive_barrier [[NEXT_ACC_EMPTY_BAR]], 1 {ttg.partition = 0 : i32}
    // CHECK-NEXT: "acc_user"([[C]])
    "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()

    // CHECK-COUNT-2: arith.addi

    // CHECK:      [[ACC_PHASE_INCR:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_ROLLVER:%.*]] = arith.cmpi eq, [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[ACC_NEXT_INDEX:%.*]] = arith.select [[ACC_ROLLVER]], %c0_i32, [[ACC_INDEX_INCR]]
    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.select [[ACC_ROLLVER]], [[ACC_PHASE_INCR]], [[ACC_PHASE]]

    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_INDEX:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_INDEX]], [[ACC_INDEX]]
    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_PHASE:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_PHASE]], [[ACC_PHASE]]

    // CHECK-NEXT: scf.yield {{.*}} [[EPILOGUE_ACC_NEXT_INDEX]], [[EPILOGUE_ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [4 : i32, 2 : i32, 0 : i32, 4 : i32]
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  tt.return
}

// AWS-LABEL: @matmul_tma_acc_with_conditional_def_and_use
// AWS: ttg.warp_specialize

// CHECK: @matmul_tma_acc_with_conditional_def_and_use
// CHECK-SAME: [[A_DESC:%arg[0-9]+]]
// CHECK-SAME: [[B_DESC:%arg[0-9]+]]
tt.func @matmul_tma_acc_with_conditional_def_and_use(
  %a_desc: !tt.tensordesc<tensor<128x64xf16, #shared>>,
  %b_desc: !tt.tensordesc<tensor<64x128xf16, #shared>>
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %true = arith.constant true
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK-COUNT-3: ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK: [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK:      {{[0-9]+}}:6 = scf.for [[K:%arg[0-9]+]]
  // CHECK-SAME: [[LOAD_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LOAD_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[MMA_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[MMA_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[ACC_PHASE:%arg[0-9]+]] = %c0_i32
  scf.for %k = %c0_i32 to %k_tiles step %c1_i32 iter_args(%acc = %zero) -> tensor<128x128xf32, #acc_layout> : i32 {
    // CHECK-NEXT: [[CUR_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[ACC_INDEX]]]
    // CHECK-NEXT: [[CUR_ACC_READY_BAR:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][[[ACC_INDEX]]]

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
    %c_tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>

    // CHECK-NEXT: [[MMA_MBAR:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][[[MMA_INDEX]]]
    // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[ACC_INDEX]], %c0_i32, %c0_i32]

    // CHECK-NEXT: ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]], %true, %true, [[MMA_MBAR]] {ttg.partition = 1 : i32}
    ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem, %true, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %c = ttng.tmem_load %c_tmem : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    // CHECK-NEXT: ttng.wait_barrier [[MMA_MBAR]], [[MMA_PHASE]] {ttg.partition = 3 : i32}
    // CHECK-NEXT: ttng.arrive_barrier {{.*}} {ttg.partition = 3 : i32}

    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = "epilogue_cond"([[K]])
    %do_epilogue = "epilogue_cond"(%k) : (i32) -> i1
    // CHECK-NEXT: ttng.arrive_barrier [[CUR_ACC_READY_BAR]], 1, [[DO_EPILOGUE]] {ttg.partition = 3 : i32}
    // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_EMPTY_BAR]], [[ACC_PHASE]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}
    %acc_reset = arith.select %do_epilogue, %zero, %c : tensor<128x128xf32, #acc_layout>

    // CHECK-NEXT: [[ACC_INDEX_INCR:%.*]] = arith.addi [[ACC_INDEX]], %c1_i32
    // CHECK-NEXT: [[NEXT_ACC_INDEX:%.*]] = arith.remui [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[NEXT_ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][[[NEXT_ACC_INDEX]], %c0_i32, %c0_i32]
    // CHECK-NEXT: ttng.tmem_store [[ZERO]], [[NEXT_ACC_BUF]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}

    // CHECK-NEXT: scf.if [[DO_EPILOGUE]]
    scf.if %do_epilogue {
      // CHECK-NEXT: ttng.wait_barrier [[CUR_ACC_READY_BAR]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: [[C:%.*]] = ttng.tmem_load [[ACC_BUF]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: [[NEXT_ACC_EMPTY_BAR:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][[[NEXT_ACC_INDEX]]]
      // CHECK-NEXT: ttng.arrive_barrier [[NEXT_ACC_EMPTY_BAR]], 1 {ttg.partition = 0 : i32}
      // CHECK-NEXT: "acc_user"([[C]])
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
    // CHECK-NEXT: } {ttg.partition = 0 : i32}
    }

    // CHECK-COUNT-2: arith.addi

    // CHECK:      [[ACC_PHASE_INCR:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[ACC_ROLLVER:%.*]] = arith.cmpi eq, [[ACC_INDEX_INCR]], %c2_i32
    // CHECK-NEXT: [[ACC_NEXT_INDEX:%.*]] = arith.select [[ACC_ROLLVER]], %c0_i32, [[ACC_INDEX_INCR]]
    // CHECK-NEXT: [[ACC_NEXT_PHASE:%.*]] = arith.select [[ACC_ROLLVER]], [[ACC_PHASE_INCR]], [[ACC_PHASE]]

    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_INDEX:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_INDEX]], [[ACC_INDEX]]
    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_PHASE:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_PHASE]], [[ACC_PHASE]]

    // CHECK-NEXT: scf.yield {{.*}} [[EPILOGUE_ACC_NEXT_INDEX]], [[EPILOGUE_ACC_NEXT_PHASE]]
    scf.yield %acc_reset : tensor<128x128xf32, #acc_layout>
  // CHECK-NEXT: ttg.partition.stages = [4 : i32, 2 : i32, 0 : i32, 4 : i32]
  } {tt.warp_specialize, tt.num_stages = 2 : i32}

  tt.return
}

// AWS-LABEL: @matmul_tma_acc_with_conditional_def_and_use_no_multibuf
// AWS: ttg.warp_specialize

// CHECK: @matmul_tma_acc_with_conditional_def_and_use_no_multibuf
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
  %zero = arith.constant dense<0.0> : tensor<128x128xf32, #acc_layout>
  %k_tiles = arith.constant 32 : i32

  // CHECK-COUNT-2: ttg.local_alloc : () -> !ttg.memdesc<2xi64

  // CHECK:      [[MMA_MBARS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2xi64
  // CHECK-NEXT: [[MMA_MBAR0:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[MMA_MBAR0]], 1
  // CHECK-NEXT: [[MMA_MBAR1:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][%c1_i32]
  // CHECK-NEXT: ttng.init_barrier [[MMA_MBAR1]], 1

  // CHECK-NEXT: [[ACC_BUFS:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, [[ACC_TMEM]], #ttng.tensor_memory, mutable>
  // CHECK-NEXT: [[ACC_BUF:%.*]] = ttg.memdesc_subview [[ACC_BUFS]][%c0_i32, %c0_i32, %c0_i32]
  // CHECK-NEXT: ttng.tmem_store [[ZERO]], [[ACC_BUF]], %true

  // CHECK-NEXT: [[ACC_EMPTY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64
  // CHECK-NEXT: [[ACC_EMPTY_BUF0:%.*]] = ttg.memdesc_subview [[ACC_EMPTY_BUFS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_EMPTY_BUF0]], 1

  // CHECK-NEXT: [[ACC_READY_BUFS:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64
  // CHECK-NEXT: [[ACC_READY_BUF0:%.*]] = ttg.memdesc_subview [[ACC_READY_BUFS]][%c0_i32]
  // CHECK-NEXT: ttng.init_barrier [[ACC_READY_BUF0]], 1

  // CHECK-NEXT: {{[0-9]+}}:6 = scf.for [[K:%arg[0-9]+]]
  // CHECK-SAME: [[FLAG:%arg[0-9]+]] = %true
  // CHECK-SAME: [[LOAD_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[LOAD_PHASE:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[MMA_INDEX:%arg[0-9]+]] = %c0_i32
  // CHECK-SAME: [[MMA_PHASE:%arg[0-9]+]] = %c0_i32
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
    %c_tmem = ttng.tmem_alloc %acc : (tensor<128x128xf32, #acc_layout>) -> !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>

    // CHECK-NEXT: [[MMA_MBAR:%.*]] = ttg.memdesc_subview [[MMA_MBARS]][[[MMA_INDEX]]]
    // CHECK-NEXT: ttng.tc_gen5_mma %{{[0-9]+}}, %{{[0-9]+}}, [[ACC_BUF]], [[FLAG]], %true, [[MMA_MBAR]] {ttg.partition = 1 : i32}
    ttng.tc_gen5_mma %a_shared, %b_shared, %c_tmem, %flag, %true : (!ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable>, i1, i1) -> ()
    %c = ttng.tmem_load %c_tmem : !ttg.memdesc<128x128xf32, #acc_tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #acc_layout>
    // CHECK-NEXT: ttng.wait_barrier [[MMA_MBAR]], [[MMA_PHASE]] {ttg.partition = 3 : i32}
    // CHECK-NEXT: ttng.arrive_barrier {{.*}} {ttg.partition = 3 : i32}

    // CHECK-NEXT: [[DO_EPILOGUE:%.*]] = "epilogue_cond"([[K]])
    %do_epilogue = "epilogue_cond"(%k) : (i32) -> i1
    // CHECK-NEXT: [[NEXT_FLAG:%.*]] = arith.xori [[DO_EPILOGUE]], %true

    // CHECK-NEXT: ttng.arrive_barrier [[ACC_READY_BUF0]], 1, [[DO_EPILOGUE]] {ttg.partition = 3 : i32}
    // CHECK-NEXT: ttng.wait_barrier [[ACC_EMPTY_BUF0]], [[ACC_PHASE]], [[DO_EPILOGUE]] {ttg.partition = 1 : i32}
    %use_acc = arith.select %do_epilogue, %false, %true : i1

    // CHECK-NEXT: scf.if [[DO_EPILOGUE]]
    scf.if %do_epilogue {
      // CHECK-NEXT: ttng.wait_barrier [[ACC_READY_BUF0]], [[ACC_PHASE]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: [[C:%.*]] = ttng.tmem_load [[ACC_BUF]] {ttg.partition = 0 : i32}
      // CHECK-NEXT: ttng.arrive_barrier [[ACC_EMPTY_BUF0]], 1 {ttg.partition = 0 : i32}
      // CHECK-NEXT: "acc_user"([[C]])
      "acc_user"(%c) : (tensor<128x128xf32, #acc_layout>) -> ()
    // CHECK-NEXT: } {ttg.partition = 0 : i32}
    }

    // CHECK: arith.addi

    // CHECK:      [[ACC_NEXT_PHASE:%.*]] = arith.xori [[ACC_PHASE]], %c1_i32
    // CHECK-NEXT: [[EPILOGUE_ACC_NEXT_PHASE:%.*]] = arith.select [[DO_EPILOGUE]], [[ACC_NEXT_PHASE]], [[ACC_PHASE]]

    // CHECK-NEXT: scf.yield [[NEXT_FLAG]], %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}}, [[EPILOGUE_ACC_NEXT_PHASE]]
    scf.yield %c, %use_acc : tensor<128x128xf32, #acc_layout>, i1
  // CHECK-NEXT: tt.scheduled_max_stage = 2 : i32
  // CHECK-SAME: ttg.partition.stages = [3 : i32, 2 : i32, 0 : i32, 3 : i32]
  } {tt.warp_specialize, tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32}

  tt.return
}

}
