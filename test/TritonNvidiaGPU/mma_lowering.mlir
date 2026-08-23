// RUN: triton-opt %s -split-input-file --triton-nvidia-mma-lowering | FileCheck %s
// RUN: triton-opt %s -split-input-file --triton-nvidia-normalize-mma-k --triton-nvidia-mma-lowering --canonicalize --cse | FileCheck %s --check-prefix=K96

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

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tcgen5_no_matching_commit_descs
  tt.func @tcgen5_no_matching_commit_descs(
    %barrier: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
    %barrierPred: i1,
    %a: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
    %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
    %c: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %accUse = arith.constant false
    %pred = arith.constant true
    // CHECK: ttng.tc_gen5_mma %arg2, %arg3, %arg4, %false, %true {is_async}
    // CHECK: ttng.tc_gen5_commit %arg0, %arg1 descs %arg2, %arg3
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred {is_async} :
       !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %barrier, %barrierPred descs %a, %b : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>
    tt.return
  }

  // CHECK-LABEL: tcgen5_stop_at_mismatched_commit_descs
  tt.func @tcgen5_stop_at_mismatched_commit_descs(
    %barrier1: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
    %barrier2: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
    %barrier3: !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>,
    %barrierPred: i1,
    %a: !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
    %b: !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
    %c: !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>) {
    %accUse = arith.constant false
    %pred = arith.constant true
    // CHECK: ttng.tc_gen5_mma %arg4, %arg5, %arg6, %false, %true, %arg0[%arg3] {is_async, multicast}
    // CHECK: ttng.tc_gen5_commit %arg1, %arg3
    // CHECK: ttng.tc_gen5_commit %arg2, %arg3 descs %arg4, %arg5
    ttng.tc_gen5_mma %a, %b, %c, %accUse, %pred {is_async, multicast} :
       !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>,
       !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    ttng.tc_gen5_commit %barrier1, %barrierPred descs %a, %b : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>
    ttng.tc_gen5_commit %barrier2, %barrierPred : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>
    ttng.tc_gen5_commit %barrier3, %barrierPred descs %a, %b : !ttg.memdesc<1xi64, #shared2, #ttg.shared_memory, mutable>, !ttg.memdesc<128x128xf8E5M2, #shared, #ttg.shared_memory>, !ttg.memdesc<128x256xf8E5M2, #shared1, #ttg.shared_memory>
    tt.return
  }
}

// -----

#ka = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CGALayout = [[1, 0]]}>
#kb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, CGALayout = [[0, 1]]}>
#kd = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
#ks = #ttng.tensor_memory_scales_encoding<CGALayout = [[1, 0]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true} {
  // CHECK-LABEL: @native_k96_ring
  // CHECK: ttng.tc_gen5_mma_scaled
  // K96-LABEL: @native_k96_ring
  // K96: ttng.tc_gen5_mma_scaled {{.*}}a_next
  // K96-SAME: b_next
  // K96-SAME: a_scale_offset = 6
  // K96-SAME: b_scale_offset = 9
  // K96-SAME: instruction_k = 96
  // K96-SAME: k_base_offsets = array<i32: 0, 0, 0, 0>
  // K96-SAME: k_range = array<i32: 192, 288>
  tt.func @native_k96_ring(%index: i32, %next_index: i32, %d: !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, %sa: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, %sb: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    %ring = ttg.local_alloc : () -> !ttg.memdesc<6x256x128xi8, #ka, #ttg.shared_memory, mutable>
    %a = ttg.memdesc_index %ring[%index] : !ttg.memdesc<6x256x128xi8, #ka, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable>
    %an = ttg.memdesc_index %ring[%next_index] : !ttg.memdesc<6x256x128xi8, #ka, #ttg.shared_memory, mutable> -> !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>
    %bn = ttg.local_alloc : () -> !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>
    ttng.tc_gen5_mma_scaled %a, %b, %d, %sa, %sb, %true, %true lhs = e2m1 rhs = e2m1 a_next %an : !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable> b_next %bn : !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable> {two_ctas, is_async, instruction_k = 96 : i32, scale_block_size = 32 : i32, k_range = array<i32: 192, 288>, a_scale_offset = 6 : i32, b_scale_offset = 9 : i32} : !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable>, !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>, !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#ka = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CGALayout = [[1, 0]]}>
#kb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, CGALayout = [[0, 1]]}>
#kd = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
#ks = #ttng.tensor_memory_scales_encoding<CGALayout = [[1, 0]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true, ttng.enable_fp4_k96 = 1 : i32} {
  // CHECK-LABEL: @automatic_unknown_origin
  // CHECK: ttng.tc_gen5_mma_scaled
  // K96-LABEL: @automatic_unknown_origin
  // K96-NOT: k_base_offsets
  // K96: tt.return
  tt.func @automatic_unknown_origin(%a: !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable>, %b: !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>, %d: !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, %sa: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, %sb: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>) {
    %true = arith.constant true
    ttng.tc_gen5_mma_scaled %a, %b, %d, %sa, %sb, %true, %true lhs = e2m1 rhs = e2m1 {two_ctas, is_async} : !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable>, !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>, !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----


#ka = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CGALayout = [[1, 0]]}>
#kb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, CGALayout = [[0, 1]]}>
#kd = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
#ks = #ttng.tensor_memory_scales_encoding<CGALayout = [[1, 0]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true, ttng.enable_fp4_k96 = 1 : i32} {
  // CHECK-LABEL: @automatic_owned
  // CHECK: ttng.tc_gen5_mma_scaled
  // K96-LABEL: @automatic_owned
  // K96: k_base_offsets = array<i32: 0, 0, 0, 0>
  // K96: tt.return
  tt.func @automatic_owned(%d: !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, %sa: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, %sb: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>
    %true = arith.constant true
    ttng.tc_gen5_mma_scaled %a, %b, %d, %sa, %sb, %true, %true lhs = e2m1 rhs = e2m1 {two_ctas, is_async} : !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable>, !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>, !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----


#ka = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CGALayout = [[1, 0]]}>
#kb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, CGALayout = [[0, 1]]}>
#kd = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
#ks = #ttng.tensor_memory_scales_encoding<CGALayout = [[1, 0]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true, ttng.enable_fp4_k96 = 1 : i32} {
  // CHECK-LABEL: @automatic_other_arch
  // CHECK: ttng.tc_gen5_mma_scaled
  // K96-LABEL: @automatic_other_arch
  // K96-NOT: k_base_offsets
  // K96: tt.return
  tt.func @automatic_other_arch(%d: !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, %sa: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, %sb: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>
    %true = arith.constant true
    ttng.tc_gen5_mma_scaled %a, %b, %d, %sa, %sb, %true, %true lhs = e2m1 rhs = e2m1 {two_ctas, is_async} : !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable>, !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>, !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----


#ka = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CGALayout = [[1, 0]]}>
#kb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, CGALayout = [[0, 1]]}>
#kd = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
#ks = #ttng.tensor_memory_scales_encoding<CGALayout = [[1, 0]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true, ttng.enable_fp4_k96 = 1 : i32} {
  // CHECK-LABEL: @automatic_tmem_lhs
  // CHECK: ttng.tc_gen5_mma_scaled
  // K96-LABEL: @automatic_tmem_lhs
  // K96-NOT: k_base_offsets
  // K96: tt.return
  tt.func @automatic_tmem_lhs(%d: !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, %sa: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, %sb: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>) {
    %a = ttng.tmem_alloc : () -> !ttg.memdesc<256x128xi8, #kd, #ttng.tensor_memory, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>
    %true = arith.constant true
    ttng.tc_gen5_mma_scaled %a, %b, %d, %sa, %sb, %true, %true lhs = e2m1 rhs = e2m1 {two_ctas, is_async} : !ttg.memdesc<256x128xi8, #kd, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>, !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----



#ka = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, CGALayout = [[1, 0]]}>
#kb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8, CGALayout = [[0, 1]]}>
#kd = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1, CGALayout = [[1, 0]], twoCTAs = true>
#ks = #ttng.tensor_memory_scales_encoding<CGALayout = [[1, 0]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:103", "ttg.threads-per-warp" = 32 : i32, "ttng.two-ctas" = true, ttng.enable_fp4_k96 = 0 : i32} {
  // CHECK-LABEL: @automatic_disabled
  // CHECK: ttng.tc_gen5_mma_scaled
  // K96-LABEL: @automatic_disabled
  // K96-NOT: k_base_offsets
  // K96: tt.return
  tt.func @automatic_disabled(%d: !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, %sa: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, %sb: !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>) {
    %a = ttg.local_alloc : () -> !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>
    %true = arith.constant true
    ttng.tc_gen5_mma_scaled %a, %b, %d, %sa, %sb, %true, %true lhs = e2m1 rhs = e2m1 {two_ctas, is_async} : !ttg.memdesc<256x128xi8, #ka, #ttg.shared_memory, mutable>, !ttg.memdesc<128x256xi8, #kb, #ttg.shared_memory, mutable>, !ttg.memdesc<256x256xf32, #kd, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>, !ttg.memdesc<256x16xi8, #ks, #ttng.tensor_memory, mutable>
    tt.return
  }
}
