// RUN: triton-opt %s -split-input-file --triton-nvidia-mma-lowering | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: gen5_mma_scaled_shmem_to_tmem
  tt.func public @gen5_mma_scaled_shmem_to_tmem(
    %A_sh: !ttg.memdesc<128x256xf8E5M2, #shared, #ttg.shared_memory>,
    %B_sh: !ttg.memdesc<256x64xf8E5M2, #shared, #ttg.shared_memory>,
    %C_tmem: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>,
    %A_scale_sh: !ttg.memdesc<128x8xi8, #shared1, #smem>,
    %B_scale_sh: !ttg.memdesc<64x8xi8, #shared1, #smem>,
    %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {

    %true = arith.constant true
    // Verify that the scale in tmem has the shape of (LHS) BlockM x BlockK / 32, (RHS) BlockN x BlockK / 32
    // CHECK: %[[A_SC_TMEM:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_copy {{.*}}, %[[A_SC_TMEM]]
    // CHECK: %[[B_SC_TMEM:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<64x8xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_copy {{.*}}, %[[B_SC_TMEM]]
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}, %[[A_SC_TMEM]], %[[B_SC_TMEM]]
    ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %C_tmem, %A_scale_sh, %B_scale_sh, %true, %true lhs = e5m2 rhs = e5m2, %barrier[%true] {is_async} : !ttg.memdesc<128x256xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<256x64xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #shared1, #smem>, !ttg.memdesc<64x8xi8, #shared1, #smem>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#sharedT = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: gen5_mma_scaled_shmem_to_tmem
  tt.func public @gen5_mma_scaled_shmem_to_tmem(
    %A_sh: !ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory>,
    %B_sh: !ttg.memdesc<256x64xi8, #sharedT, #ttg.shared_memory>,
    %C_tmem: !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>,
    %A_scale_sh: !ttg.memdesc<128x8xf8E4M3FN, #shared1, #smem>,
    %B_scale_sh: !ttg.memdesc<64x8xf8E4M3FN, #shared1, #smem>,
    %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>) {

    %true = arith.constant true
    // Verify that the scale in tmem has the shape of (LHS) BlockM x BlockK / 32, (RHS) BlockN x BlockK / 32
    // CHECK: %[[A_SC_TMEM:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_copy {{.*}}, %[[A_SC_TMEM]]
    // CHECK: %[[B_SC_TMEM:.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<64x8xf8E4M3FN, #tmem_scales, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_copy {{.*}}, %[[B_SC_TMEM]]
    // CHECK: ttng.tc_gen5_mma_scaled {{.*}}, %[[A_SC_TMEM]], %[[B_SC_TMEM]]
    ttng.tc_gen5_mma_scaled %A_sh, %B_sh, %C_tmem, %A_scale_sh, %B_scale_sh, %true, %true lhs = e2m1 rhs = e2m1, %barrier[%true] {is_async} : !ttg.memdesc<128x256xi8, #shared, #ttg.shared_memory>, !ttg.memdesc<256x64xi8, #sharedT, #ttg.shared_memory>, !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<64x8xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tcgen5_with_commit
  tt.func @tcgen5_with_commit(
    // CHECK: [[BARRIER1:%.*]]: !ttg.memdesc<1xi64, #shared
    %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
    // CHECK: [[BARRIER_PRED:%.*]]: i1,
    %barrierPred: i1,
    // CHECK: [[A_SMEM:%.*]]: !ttg.memdesc<128x128xf8E5M2
    %a: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
    %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
    %c: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %barrier2 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #shared2, #smem, mutable>
    %c0_i32 = arith.constant 0 : i32
    // CHECK: [[TRUE:%.*]] = arith.constant true
    // CHECK: [[BARRIER_SLICE:%.*]] = ttg.memdesc_index
    // CHECK: ttng.tc_gen5_mma {{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}, [[BARRIER1]][[[BARRIER_PRED]]], [[BARRIER_SLICE]][[[TRUE]]]
    %accUse = arith.constant false
    %pred = arith.constant true
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred {is_async} :
       !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %barrier, %barrierPred : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    %barrier_slice = ttg.memdesc_index %barrier2[%c0_i32] : !ttg.memdesc<2x1xi64, #shared2, #smem, mutable> -> !ttg.memdesc<1xi64, #shared2, #smem, mutable>
    ttng.tc_gen5_commit %barrier_slice : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>

    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred {is_async} :
       !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>

    %random_pred = arith.cmpi eq, %barrierPred, %pred : i1
    scf.if %random_pred {
      ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred {is_async} :
       !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    }
    // This commit should not be merged into any of two mma ops above
    // CHECK: tc_gen5_commit
    ttng.tc_gen5_commit %barrier, %barrierPred : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>

    // The mma predicate is not a constant true. The commit op should not be merged
    // CHECK: tc_gen5_commit
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %random_pred {is_async} :
       !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %barrier : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>

    // There is an impure op between mma and commit ops. Do not allow merging in such cases.
    // CHECK: tc_gen5_commit
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred {is_async} :
       !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.wait_barrier %barrier, %c0_i32 : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    ttng.tc_gen5_commit %barrier : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>

    tt.return
  }
}
