// RUN: triton-opt %s -tritongpu-prefetch -canonicalize | FileCheck %s --dump-input-context=50

#A_RING = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#B_RING = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#C_RING = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#A_RING_OP = #ttg.dot_op<{opIdx = 0, parent = #C_RING, kWidth = 2}>
#B_RING_OP = #ttg.dot_op<{opIdx = 1, parent = #C_RING, kWidth = 2}>
#smem = #ttg.shared_memory

// CHECK-LABEL: tt.func @prefetch_pipelined_loop_args
// CHECK-DAG: %[[INIT_WAIT:.+]] = ttg.async_wait %arg3, %arg4 {num = 4 : i32}
// CHECK-DAG: %[[A0_SMEM:.+]] = ttg.memdesc_subslice %[[A0:.+]][0, 0]
// CHECK-DAG: %[[A0_PREFETCH:.+]] = ttg.local_load %[[A0_SMEM]] token %[[INIT_WAIT]]
// CHECK-DAG: %[[B0_SMEM:.+]] = ttg.memdesc_subslice %[[B0:.+]][0, 0]
// CHECK-DAG: %[[B0_PREFETCH:.+]] = ttg.local_load %[[B0_SMEM]] token %[[INIT_WAIT]]
// CHECK: %[[LOOP:.+]]:7 = scf.for {{.+}} iter_args(%[[IDX_ARG:.+]] = %[[C0:.+]], %[[A_ARG:.+]] = %[[A0]], %[[B_ARG:.+]] = %[[B0]], %[[WAIT_ARG:.+]] = %[[INIT_WAIT]], %[[ACC_ARG:.+]] = %{{.+}}, %[[A_PREFETCH_ARG:.+]] = %[[A0_PREFETCH]], %[[B_PREFETCH_ARG:.+]] = %[[B0_PREFETCH]])
// CHECK: %[[WAIT_NEXT:.+]] = ttg.async_wait %arg3, %arg4 {num = 4 : i32}
// CHECK-DAG: %[[A1_SMEM:.+]] = ttg.memdesc_subslice %[[A_ARG]][0, 16]
// CHECK-DAG: %[[A1:.+]] = ttg.local_load %[[A1_SMEM]] token %[[WAIT_ARG]]
// CHECK-DAG: %[[B1_SMEM:.+]] = ttg.memdesc_subslice %[[B_ARG]][16, 0]
// CHECK-DAG: %[[B1:.+]] = ttg.local_load %[[B1_SMEM]] token %[[WAIT_ARG]]
// CHECK: %{{.+}} = tt.dot %[[A_PREFETCH_ARG]], %[[B_PREFETCH_ARG]], %[[ACC_ARG]]
// CHECK-DAG: %[[NEXT_A_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[0, 0]
// CHECK-DAG: %[[NEXT_A_PREFETCH:.+]] = ttg.local_load %[[NEXT_A_SMEM]] token %[[WAIT_NEXT]]
// CHECK-DAG: %[[NEXT_B_SMEM:.+]] = ttg.memdesc_subslice %{{.+}}[0, 0]
// CHECK-DAG: %[[NEXT_B_PREFETCH:.+]] = ttg.local_load %[[NEXT_B_SMEM]] token %[[WAIT_NEXT]]
module attributes { "ttg.num-warps" = 4 : i32 } {
tt.func @prefetch_pipelined_loop_args(%lb : index, %ub : index, %step : index, %tok0 : !ttg.async.token, %tok1 : !ttg.async.token) -> tensor<128x128xf32, #C_RING> {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c3_i32 = arith.constant 3 : i32
  %cst = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C_RING>
  %a = ttg.local_alloc : () -> !ttg.memdesc<3x128x32xf16, #A_RING, #smem, mutable>
  %b = ttg.local_alloc : () -> !ttg.memdesc<3x32x128xf16, #B_RING, #smem, mutable>
  %wait0 = ttg.async_wait %tok0, %tok1 {num = 4 : i32}
  %a0 = ttg.memdesc_index %a[%c0_i32] : !ttg.memdesc<3x128x32xf16, #A_RING, #smem, mutable> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
  %b0 = ttg.memdesc_index %b[%c0_i32] : !ttg.memdesc<3x32x128xf16, #B_RING, #smem, mutable> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
  %loop:7 = scf.for %iv = %lb to %ub step %step iter_args(%idx = %c0_i32, %a_view = %a0, %b_view = %b0, %wait = %wait0, %tok_a = %tok0, %tok_b = %tok1, %acc = %cst) -> (i32, !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>, !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token, tensor<128x128xf32, #C_RING>) {
    %a_val = ttg.local_load %a_view token %wait : !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable> -> tensor<128x32xf16, #A_RING_OP>
    %b_val = ttg.local_load %b_view token %wait : !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable> -> tensor<32x128xf16, #B_RING_OP>
    %idx_p1 = arith.addi %idx, %c1_i32 : i32
    %idx_cmp = arith.cmpi sge, %idx_p1, %c3_i32 : i32
    %idx_next = arith.select %idx_cmp, %c0_i32, %idx_p1 : i32
    %wait_next = ttg.async_wait %tok_a, %tok_b {num = 4 : i32}
    %a_next = ttg.memdesc_index %a[%idx_next] : !ttg.memdesc<3x128x32xf16, #A_RING, #smem, mutable> -> !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>
    %b_next = ttg.memdesc_index %b[%idx_next] : !ttg.memdesc<3x32x128xf16, #B_RING, #smem, mutable> -> !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>
    %acc_next = tt.dot %a_val, %b_val, %acc : tensor<128x32xf16, #A_RING_OP> * tensor<32x128xf16, #B_RING_OP> -> tensor<128x128xf32, #C_RING>
    scf.yield %idx_next, %a_next, %b_next, %wait_next, %tok_a, %tok_b, %acc_next : i32, !ttg.memdesc<128x32xf16, #A_RING, #smem, mutable>, !ttg.memdesc<32x128xf16, #B_RING, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token, tensor<128x128xf32, #C_RING>
  }
  tt.return %loop#6 : tensor<128x128xf32, #C_RING>
}
}  // end module
