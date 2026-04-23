// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-semaphore | FileCheck %s --implicit-check-not=nvws.semaphore

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared_desc = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
!elt = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @basic
  tt.func @basic() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cm1_i32 = arith.constant -1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // Mbarrier allocation (2 slots for 2-buffer semaphore)
    // CHECK: [[MBAR:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
    // Per-slice barrier init
    // CHECK: [[SLICE0:%.*]] = ttg.memdesc_index [[MBAR]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE0]], 1
    // CHECK: [[SLICE1:%.*]] = ttg.memdesc_index [[MBAR]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE1]], 1
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>

    // Acquire: full pipeline computes the phase bit before wait_barrier.
    // CHECK: [[MBAR_ACQ:%.*]] = ttg.memdesc_index [[MBAR]][%{{.*}}] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: ttng.wait_barrier [[MBAR_ACQ]], %{{.*}} {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // Buffer indexed by stage
    // CHECK: [[BUF_VIEW:%.*]] = ttg.memdesc_index %{{.*}}[%{{.*}}] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: ttg.local_load [[BUF_VIEW]] {ttg.partition = array<i32: 0>}
    // CHECK: ttg.local_store {{%.*}}, [[BUF_VIEW]] {ttg.partition = array<i32: 0>}
    // Release: arrive on same mbar slice
    // CHECK: [[MBAR_REL:%.*]] = ttg.memdesc_index [[MBAR]][%{{.*}}] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: ttng.arrive_barrier [[MBAR_REL]], 1 {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    %tok = nvws.semaphore.acquire %sem {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem, %tok {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %v = ttg.local_load %view {ttg.partition = array<i32: 0>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
    ttg.local_store %v, %view {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

    // Cleanup: per-slice invalidation
    // CHECK: [[INV0:%.*]] = ttg.memdesc_index [[MBAR]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[INV0]]
    // CHECK: [[INV1:%.*]] = ttg.memdesc_index [[MBAR]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[INV1]]
    // CHECK: ttg.local_dealloc [[MBAR]]
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tma_load
  tt.func @tma_load(%desc: !tt.tensordesc<128x64xf16, #shared_desc>) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>
    // CHECK: [[MBAR_TMA:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK-COUNT-3: ttng.init_barrier %{{.*}}, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>]>

    // Acquire: wait on mbar, then TMA load with barrier_expect
    // CHECK: [[MBAR_TMA_ACQ:%.*]] = ttg.memdesc_index [[MBAR_TMA]][%{{.*}}] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.wait_barrier [[MBAR_TMA_ACQ]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: [[BUF_TMA:%.*]] = ttg.memdesc_index %{{.*}}[%{{.*}}] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: [[MBAR_TMA_EXP:%.*]] = ttg.memdesc_index [[MBAR_TMA]][%{{.*}}] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.barrier_expect [[MBAR_TMA_EXP]], 4096 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] [[BUF_TMA]], [[MBAR_TMA_EXP]], {{%.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    %tok = nvws.semaphore.acquire %sem {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>]> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem, %tok {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_desc, #smem, mutable>
    nvws.descriptor_load %desc[%c0_i32, %c0_i32] 4096 %view {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared_desc>, i32, i32, !ttg.memdesc<128x64xf16, #shared_desc, #smem, mutable>
    // Cleanup
    // CHECK-COUNT-3: ttng.inval_barrier %{{.*}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttg.local_dealloc [[MBAR_TMA]]
    nvws.semaphore.release %sem, %tok [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x128x64xf16, #shared_desc, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tma_gather
  tt.func @tma_gather(%desc: !tt.tensordesc<1x128xf16, #shared_desc>) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %xoffs = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : tensor<8xi32>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>
    // CHECK: [[MBAR_GAT:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK-COUNT-3: ttng.init_barrier %{{.*}}, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>]>

    // CHECK: [[MBAR_GAT_ACQ:%.*]] = ttg.memdesc_index [[MBAR_GAT]][%{{.*}}] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.wait_barrier [[MBAR_GAT_ACQ]], {{%.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: [[BUF_GAT:%.*]] = ttg.memdesc_index %{{.*}}[%{{.*}}] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: [[MBAR_GAT_EXP:%.*]] = ttg.memdesc_index [[MBAR_GAT]][%{{.*}}] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.barrier_expect [[MBAR_GAT_EXP]], 2048 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.async_tma_gather %arg0[{{%.*}}, %c0_i32] [[BUF_GAT]], [[MBAR_GAT_EXP]], {{%.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    %tok = nvws.semaphore.acquire %sem {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>]> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem, %tok {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<8x128xf16, #shared_desc, #smem, mutable>
    nvws.descriptor_gather %desc[%xoffs, %c0_i32] 2048 %view {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<1x128xf16, #shared_desc>, tensor<8xi32>, i32, !ttg.memdesc<8x128xf16, #shared_desc, #smem, mutable>
    // CHECK-COUNT-3: ttng.inval_barrier %{{.*}} : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // CHECK: ttg.local_dealloc [[MBAR_GAT]]
    nvws.semaphore.release %sem, %tok [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x8x128xf16, #shared_desc, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tc5mma_commit
  tt.func @tc5mma_commit() {
    %c0_i32 = arith.constant 0 : i32
    %c0_phase = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR_MMA:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[SLICE_MMA:%.*]] = ttg.memdesc_index [[MBAR_MMA]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE_MMA]], 1
    %sem = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>

    // CHECK: [[MBAR_MMA_ACQ:%.*]] = ttg.memdesc_index [[MBAR_MMA]][%{{.*}}] {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    // CHECK: ttng.wait_barrier [[MBAR_MMA_ACQ]], {{%.*}} {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    // tc5mma release: tc_gen5_commit on the mbar
    // CHECK: [[MBAR_MMA_REL:%.*]] = ttg.memdesc_index [[MBAR_MMA]][%{{.*}}] {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    // CHECK: ttng.tc_gen5_commit [[MBAR_MMA_REL]] {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    %tok = nvws.semaphore.acquire %sem {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    nvws.semaphore.release %sem, %tok [#nvws.async_op<tc5mma>] {loop.cluster = 4 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    // CHECK: [[INV_MMA:%.*]] = ttg.memdesc_index [[MBAR_MMA]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[INV_MMA]]
    // CHECK: ttg.local_dealloc [[MBAR_MMA]]
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @wgmma_pending_count
  // Tests that async_ops=[wgmma, none] is supported by pending-count init
  // and lowers to two arrives.
  tt.func @wgmma_pending_count() {
    %c0_i32 = arith.constant 0 : i32
    %c0_phase = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR_WGMMA:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[WGMMA_SLICE:%.*]] = ttg.memdesc_index [[MBAR_WGMMA]][{{%.*}}]
    // CHECK: ttng.init_barrier [[WGMMA_SLICE]], 2
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %c0_phase] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK-COUNT-2: ttng.arrive_barrier {{%.*}}, 1 {ttg.partition = array<i32: 0>}
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<wgmma>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tmem_copy_pending_count
  // Tests that async_ops=[tmem_copy, none] is supported by pending-count init
  // and lowers to tc_gen5_commit + arrive_barrier.
  tt.func @tmem_copy_pending_count() {
    %c0_i32 = arith.constant 0 : i32
    %c0_phase = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR_TMEM:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[TMEM_SLICE:%.*]] = ttg.memdesc_index [[MBAR_TMEM]][{{%.*}}]
    // CHECK: ttng.init_barrier [[TMEM_SLICE]], 2
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %c0_phase] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    // CHECK: ttng.tc_gen5_commit {{%.*}} {ttg.partition = array<i32: 0>}
    // CHECK: ttng.arrive_barrier {{%.*}}, 1 {ttg.partition = array<i32: 0>}
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<tmem_copy>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @fence_needed
  // Tests that async_op<none> inserts fence + arrive_barrier,
  // while async_op<tc5mma> uses tc_gen5_commit (no fence)
  tt.func @fence_needed() {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    // Two separate mbar allocs (one per semaphore)
    // CHECK: [[MBAR_GEN:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[SLICE_GEN:%.*]] = ttg.memdesc_index [[MBAR_GEN]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE_GEN]], 1
    %sem_generic = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[MBAR_TC5:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: [[SLICE_TC5:%.*]] = ttg.memdesc_index [[MBAR_TC5]][{{%.*}}]
    // CHECK: ttng.init_barrier [[SLICE_TC5]], 1
    %sem_other = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>

    // tc5mma semaphore: wait then tc_gen5_commit (no fence)
    // CHECK: ttng.wait_barrier {{%.*}}, {{%.*}} {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
    // CHECK: ttng.tc_gen5_commit {{%.*}} {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
    %tok_other = nvws.semaphore.acquire %sem_other {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    nvws.semaphore.release %sem_other, %tok_other [#nvws.async_op<tc5mma>] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

    // Generic semaphore: wait then fence + arrive_barrier
    // CHECK: ttng.wait_barrier {{%.*}}, {{%.*}} {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: ttng.fence_async_shared {bCluster = false, loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK: [[MBAR_GEN_REL:%.*]] = ttg.memdesc_index [[MBAR_GEN]][%{{.*}}] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttng.arrive_barrier [[MBAR_GEN_REL]], 1 {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
    %tok_generic = nvws.semaphore.acquire %sem_generic {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    nvws.semaphore.release %sem_generic, %tok_generic [#nvws.async_op<none>] {loop.cluster = 5 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

    ttg.local_dealloc %buf : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @tmem_scales_passthrough
  // Tests that tensor_memory_scales buffers pass through without memdesc_index
  tt.func @tmem_scales_passthrough(%arg0: !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %sem = nvws.semaphore.create %arg0 true : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>]>
    %tok = nvws.semaphore.acquire %sem[%c0_i32, %c1_i32] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>]> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem[%c0_i32], %tok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>]>, !ttg.async.token -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    // CHECK: "use_scale_load"(%arg0) : (!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>) -> ()
    // CHECK-NOT: ttg.memdesc_index {{.*}}#ttng.tensor_memory_scales_encoding
    "use_scale_load"(%view) : (!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>) -> ()
    nvws.semaphore.release %sem[%c0_i32], %tok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>]>, !ttg.async.token
    tt.return
  }

  // CHECK-LABEL: @two_consumers
  // Tests 1 producer + 2 consumers with 3-buffer semaphore pair
  // Verifies barrier init counts, stage-indexed mbar, and per-slice cleanup
  tt.func @two_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cm1_i32 = arith.constant -1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    // EMPTY semaphore: init_barrier with count=2 (2 consumers must arrive)
    // CHECK: [[EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[ES0:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.init_barrier [[ES0]], 2
    // CHECK: [[ES1:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.init_barrier [[ES1]], 2
    // CHECK: [[ES2:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.init_barrier [[ES2]], 2
    %sem_empty = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    // FULL semaphore: init_barrier with count=1 (1 producer must arrive)
    // CHECK: [[FULL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[FS0:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.init_barrier [[FS0]], 1
    // CHECK: [[FS1:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.init_barrier [[FS1]], 1
    // CHECK: [[FS2:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.init_barrier [[FS2]], 1
    %sem_full = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    // CHECK: scf.for
    scf.for %i = %arg0 to %arg1 step %arg2 : i32 {
      %val = "op_a"() {ttg.partition = array<i32: 0>} : () -> !elt
      // Producer: wait EMPTY[stage], store, arrive FULL[stage]
      // CHECK: "op_a"
      // CHECK: addi
      // CHECK: cmpi
      // CHECK: [[STAGE2:%.*]] = arith.select
      // CHECK: shli
      // CHECK: xori
      // CHECK: shrui
      // CHECK: [[PHASE2_P:%.*]] = arith.andi
      // CHECK-NEXT: [[EMPTY_MBAR:%.*]] = ttg.memdesc_index [[EMPTY]][[[STAGE2]]] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.wait_barrier [[EMPTY_MBAR]], [[PHASE2_P]] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      // CHECK: [[PBUF:%.*]] = ttg.memdesc_index %{{.*}}[[[STAGE2]]] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{%.*}}, [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[FULL_MBAR_P:%.*]] = ttg.memdesc_index [[FULL]][[[STAGE2]]] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.arrive_barrier [[FULL_MBAR_P]], 1 {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      %ptok = nvws.semaphore.acquire %sem_empty[%c0_i32, %c1_i32] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %pbuf = nvws.semaphore.buffer %sem_empty[%c0_i32], %ptok {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %val, %pbuf {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      nvws.semaphore.release %sem_full[%c0_i32], %ptok [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      // Consumer 1: wait FULL[stage], load, arrive EMPTY[stage]
      // CHECK: shli
      // CHECK: xori
      // CHECK: shrui
      // CHECK: [[PHASE2_C1:%.*]] = arith.andi
      // CHECK-NEXT: [[FULL_MBAR_C1:%.*]] = ttg.memdesc_index [[FULL]][[[STAGE2]]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.wait_barrier [[FULL_MBAR_C1]], [[PHASE2_C1]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_load {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[EMPTY_MBAR_C1:%.*]] = ttg.memdesc_index [[EMPTY]][[[STAGE2]]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTY_MBAR_C1]], 1 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>}
      %gtok1 = nvws.semaphore.acquire %sem_full[%c0_i32, %c1_i32] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %gbuf1 = nvws.semaphore.buffer %sem_full[%c0_i32], %gtok1 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %v1 = ttg.local_load %gbuf1 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
      nvws.semaphore.release %sem_empty[%c0_i32], %gtok1 [#nvws.async_op<none>] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%v1) {ttg.partition = array<i32: 1>} : (!elt) -> ()

      // Consumer 2: wait FULL[stage], load, arrive EMPTY[stage]
      // CHECK: shli
      // CHECK: xori
      // CHECK: shrui
      // CHECK: [[PHASE2_C2:%.*]] = arith.andi
      // CHECK-NEXT: [[FULL_MBAR_C2:%.*]] = ttg.memdesc_index [[FULL]][[[STAGE2]]] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttng.wait_barrier [[FULL_MBAR_C2]], [[PHASE2_C2]] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_load {{%.*}} {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[EMPTY_MBAR_C2:%.*]] = ttg.memdesc_index [[EMPTY]][[[STAGE2]]] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTY_MBAR_C2]], 1 {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>}
      %gtok2 = nvws.semaphore.acquire %sem_full[%c0_i32, %c1_i32] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %gbuf2 = nvws.semaphore.buffer %sem_full[%c0_i32], %gtok2 {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %v2 = ttg.local_load %gbuf2 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
      nvws.semaphore.release %sem_empty[%c0_i32], %gtok2 [#nvws.async_op<none>] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%v2) {ttg.partition = array<i32: 2>} : (!elt) -> ()
      "op_d"(%v2) {ttg.partition = array<i32: 2>} : (!elt) -> ()
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    // Cleanup: per-slice inval for EMPTY (3 slices)
    // CHECK: [[EI0:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.inval_barrier [[EI0]]
    // CHECK: [[EI1:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.inval_barrier [[EI1]]
    // CHECK: [[EI2:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.inval_barrier [[EI2]]
    // CHECK: ttg.local_dealloc [[EMPTY]]
    // Cleanup: per-slice inval for FULL (3 slices)
    // CHECK: [[FI0:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.inval_barrier [[FI0]]
    // CHECK: [[FI1:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.inval_barrier [[FI1]]
    // CHECK: [[FI2:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.inval_barrier [[FI2]]
    // CHECK: ttg.local_dealloc [[FULL]]
    ttg.local_dealloc %buf : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @three_consumers
  // Tests 1 producer + 3 consumers: init_barrier count=3 for EMPTY
  tt.func @three_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    // EMPTY: init_barrier with count=3 (3 consumers)
    // CHECK: [[EMPTY3:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[ES:%.*]] = ttg.memdesc_index [[EMPTY3]]
    // CHECK: ttng.init_barrier [[ES]], 3
    %sem_empty = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    // FULL: init_barrier with count=1 (1 producer)
    // CHECK: [[FULL3:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[FS:%.*]] = ttg.memdesc_index [[FULL3]]
    // CHECK: ttng.init_barrier [[FS]], 1
    %sem_full = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    scf.for %i = %arg0 to %arg1 step %arg2 : i32 {
      %val = "op_a"() {ttg.partition = array<i32: 0>} : () -> !elt
      // Producer
      // CHECK: "op_a"
      // CHECK: addi
      // CHECK: cmpi
      // CHECK: [[STAGE3:%.*]] = arith.select
      // CHECK: xori
      // CHECK: cmpi
      // CHECK: [[PHASE3_P:%.*]] = arith.select
      // CHECK-NEXT: [[EMPTY3_WAIT:%.*]] = ttg.memdesc_index [[EMPTY3]][[[STAGE3]]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.wait_barrier [[EMPTY3_WAIT]], [[PHASE3_P]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{%.*}}, {{%.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[FULL3_P:%.*]] = ttg.memdesc_index [[FULL3]][[[STAGE3]]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.arrive_barrier [[FULL3_P]], 1 {ttg.partition = array<i32: 0>}
      %ptok = nvws.semaphore.acquire %sem_empty[%c0_i32, %c1_i32] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %pbuf = nvws.semaphore.buffer %sem_empty[%c0_i32], %ptok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      ttg.local_store %val, %pbuf {ttg.partition = array<i32: 0>} : !elt -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      nvws.semaphore.release %sem_full[%c0_i32], %ptok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // Consumer 1
      // CHECK: xori
      // CHECK: cmpi
      // CHECK: [[PHASE3_C1:%.*]] = arith.select
      // CHECK-NEXT: [[FULL3_C1:%.*]] = ttg.memdesc_index [[FULL3]][[[STAGE3]]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.wait_barrier [[FULL3_C1]], [[PHASE3_C1]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_load {{%.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[EMPTY3_C1:%.*]] = ttg.memdesc_index [[EMPTY3]][[[STAGE3]]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTY3_C1]], 1 {ttg.partition = array<i32: 1>}
      %g1 = nvws.semaphore.acquire %sem_full[%c0_i32, %c1_i32] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %b1 = nvws.semaphore.buffer %sem_full[%c0_i32], %g1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %v1 = ttg.local_load %b1 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
      nvws.semaphore.release %sem_empty[%c0_i32], %g1 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%v1) {ttg.partition = array<i32: 1>} : (!elt) -> ()
      // Consumer 2
      // CHECK: xori
      // CHECK: cmpi
      // CHECK: [[PHASE3_C2:%.*]] = arith.select
      // CHECK-NEXT: [[FULL3_C2:%.*]] = ttg.memdesc_index [[FULL3]][[[STAGE3]]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttng.wait_barrier [[FULL3_C2]], [[PHASE3_C2]] {ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_load {{%.*}} {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[EMPTY3_C2:%.*]] = ttg.memdesc_index [[EMPTY3]][[[STAGE3]]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTY3_C2]], 1 {ttg.partition = array<i32: 2>}
      %g2 = nvws.semaphore.acquire %sem_full[%c0_i32, %c1_i32] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %b2 = nvws.semaphore.buffer %sem_full[%c0_i32], %g2 {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %v2 = ttg.local_load %b2 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
      nvws.semaphore.release %sem_empty[%c0_i32], %g2 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%v2) {ttg.partition = array<i32: 2>} : (!elt) -> ()
      // Consumer 3
      // CHECK: xori
      // CHECK: cmpi
      // CHECK: [[PHASE3_C3:%.*]] = arith.select
      // CHECK-NEXT: [[FULL3_C3:%.*]] = ttg.memdesc_index [[FULL3]][[[STAGE3]]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: ttng.wait_barrier [[FULL3_C3]], [[PHASE3_C3]] {ttg.partition = array<i32: 3>}
      // CHECK: ttg.local_load {{%.*}} {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: [[EMPTY3_C3:%.*]] = ttg.memdesc_index [[EMPTY3]][[[STAGE3]]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTY3_C3]], 1 {ttg.partition = array<i32: 3>}
      %g3 = nvws.semaphore.acquire %sem_full[%c0_i32, %c1_i32] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
      %b3 = nvws.semaphore.buffer %sem_full[%c0_i32], %g3 {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
      %v3 = ttg.local_load %b3 {ttg.partition = array<i32: 3>} : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> !elt
      nvws.semaphore.release %sem_empty[%c0_i32], %g3 [#nvws.async_op<none>] {ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_e"(%v3) {ttg.partition = array<i32: 3>} : (!elt) -> ()
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32, 3 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2, 3>}
    // Cleanup: 3 inval per mbar alloc
    // CHECK: [[E3I0:%.*]] = ttg.memdesc_index [[EMPTY3]]
    // CHECK: ttng.inval_barrier [[E3I0]]
    // CHECK: [[E3I1:%.*]] = ttg.memdesc_index [[EMPTY3]]
    // CHECK: ttng.inval_barrier [[E3I1]]
    // CHECK: [[E3I2:%.*]] = ttg.memdesc_index [[EMPTY3]]
    // CHECK: ttng.inval_barrier [[E3I2]]
    // CHECK: ttg.local_dealloc [[EMPTY3]]
    // CHECK: [[F3I0:%.*]] = ttg.memdesc_index [[FULL3]]
    // CHECK: ttng.inval_barrier [[F3I0]]
    // CHECK: [[F3I1:%.*]] = ttg.memdesc_index [[FULL3]]
    // CHECK: ttng.inval_barrier [[F3I1]]
    // CHECK: [[F3I2:%.*]] = ttg.memdesc_index [[FULL3]]
    // CHECK: ttng.inval_barrier [[F3I2]]
    // CHECK: ttg.local_dealloc [[FULL3]]
    ttg.local_dealloc %buf : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked4 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared4 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem4 = #ttg.shared_memory
!elt4 = tensor<1xi32, #blocked4>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @reuse_argument
  tt.func @reuse_argument(%arg0: i32, %arg1: i32, %arg2: i32) {
    %cst = arith.constant dense<1> : !elt4
    // Buffer alloc (depth=1, no multi-buffering since no TMA producer)
    // CHECK: [[BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32,
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>
    // EMPTY mbar: init_barrier with pending count = 2 (two consumer releases)
    // CHECK: [[MBAR_E:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
    // CHECK: [[ESLICE:%.*]] = ttg.memdesc_index [[MBAR_E]]
    // CHECK: ttng.init_barrier [[ESLICE]], 2
    // FULL mbar: init_barrier with pending count = 1 (one producer release)
    // CHECK: [[MBAR_F:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
    // CHECK: [[FSLICE:%.*]] = ttg.memdesc_index [[MBAR_F]]
    // CHECK: ttng.init_barrier [[FSLICE]], 1
    %empty = nvws.semaphore.create %alloc true : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>]>
    %full = nvws.semaphore.create %alloc false : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>]>
    // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (tensor<1xi32, #blocked{{.*}}>, i32, i32, i32, i32)
    scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg5 = %cst) -> (!elt4) : i32 {
      // Producer (partition 0): wait EMPTY, local_store, arrive FULL
      // CHECK: addi
      // CHECK: cmpi
      // CHECK: [[STAGE_RA:%.*]] = arith.select
      // CHECK: xori
      // CHECK: cmpi
      // CHECK: [[PHASE_RA_P:%.*]] = arith.select
      // CHECK: [[EMPTYBAR1:%.*]] = ttg.memdesc_index [[MBAR_E]][[[STAGE_RA]]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.wait_barrier [[EMPTYBAR1]], [[PHASE_RA_P]] {ttg.partition = array<i32: 0>}
      // CHECK: [[PBUF:%.*]] = ttg.memdesc_index [[BUF]][[[STAGE_RA]]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{.*}}, [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[FULLBAR1:%.*]] = ttg.memdesc_index [[MBAR_F]][[[STAGE_RA]]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.arrive_barrier [[FULLBAR1]], 1 {ttg.partition = array<i32: 0>}
      // CHECK: "op_a"
      %ptok = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>]> -> !ttg.async.token
      %pbuf = nvws.semaphore.buffer %empty, %ptok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared4, #smem4, mutable>
      ttg.local_store %arg5, %pbuf {ttg.partition = array<i32: 0>} : !elt4 -> !ttg.memdesc<1xi32, #shared4, #smem4, mutable>
      nvws.semaphore.release %full, %ptok [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>]>, !ttg.async.token
      %5 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !elt4
      // Consumer 1 (partition 1): wait FULL, local_load, arrive EMPTY
      // CHECK: xori
      // CHECK: cmpi
      // CHECK: [[PHASE_RA_C1:%.*]] = arith.select
      // CHECK: [[FULLMBAR1:%.*]] = ttg.memdesc_index [[MBAR_F]][[[STAGE_RA]]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR1]], [[PHASE_RA_C1]] {ttg.partition = array<i32: 1>}
      // CHECK: [[CBUF1:%.*]] = ttg.memdesc_index [[BUF]][[[STAGE_RA]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_load [[CBUF1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[EMPTYMBAR1:%.*]] = ttg.memdesc_index [[MBAR_E]][[[STAGE_RA]]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR1]], 1 {ttg.partition = array<i32: 1>}
      // CHECK: "op_d"
      %gtok1 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>]> -> !ttg.async.token
      %gbuf1 = nvws.semaphore.buffer %full, %gtok1 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared4, #smem4, mutable>
      %8 = ttg.local_load %gbuf1 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared4, #smem4, mutable> -> !elt4
      nvws.semaphore.release %empty, %gtok1 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>]>, !ttg.async.token
      "op_d"(%8) {ttg.partition = array<i32: 1>} : (!elt4) -> ()
      // Consumer 2 (partition 2): wait FULL, local_load, arrive EMPTY
      // CHECK: xori
      // CHECK: cmpi
      // CHECK: [[PHASE_RA_C2:%.*]] = arith.select
      // CHECK: [[FULLMBAR2:%.*]] = ttg.memdesc_index [[MBAR_F]][[[STAGE_RA]]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR2]], [[PHASE_RA_C2]] {ttg.partition = array<i32: 2>}
      // CHECK: [[CBUF2:%.*]] = ttg.memdesc_index [[BUF]][[[STAGE_RA]]] {ttg.partition = array<i32: 2>}
      // CHECK: ttg.local_load [[CBUF2]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[EMPTYMBAR2:%.*]] = ttg.memdesc_index [[MBAR_E]][[[STAGE_RA]]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR2]], 1 {ttg.partition = array<i32: 2>}
      // CHECK: "op_d"
      %gtok2 = nvws.semaphore.acquire %full {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>]> -> !ttg.async.token
      %gbuf2 = nvws.semaphore.buffer %full, %gtok2 {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared4, #smem4, mutable>
      %11 = ttg.local_load %gbuf2 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared4, #smem4, mutable> -> !elt4
      nvws.semaphore.release %empty, %gtok2 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>]>, !ttg.async.token
      "op_d"(%11) {ttg.partition = array<i32: 2>} : (!elt4) -> ()
      scf.yield %5 : !elt4
    } {ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], tt.warp_specialize}
    // Cleanup
    // CHECK: ttng.inval_barrier
    // CHECK: ttg.local_dealloc [[MBAR_E]]
    // CHECK: ttng.inval_barrier
    // CHECK: ttg.local_dealloc [[MBAR_F]]
    // CHECK: ttg.local_dealloc [[BUF]]
    ttg.local_dealloc %alloc : !ttg.memdesc<1x1xi32, #shared4, #smem4, mutable>
    tt.return
  }
}

// -----

#shared_ws_port = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_ws_port_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem_ws_port = #ttg.shared_memory
#tmem_ws_port = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @warp_specialize_tma_matmul
  tt.func @warp_specialize_tma_matmul(%desc_a: !tt.tensordesc<128x64xf16, #shared_ws_port>, %desc_b: !tt.tensordesc<128x64xf16, #shared_ws_port>, %lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem_ws_port, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %empty_a = nvws.semaphore.create %alloc_a true : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>
    %full_a = nvws.semaphore.create %alloc_a false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>
    %empty_b = nvws.semaphore.create %alloc_b true : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>
    %full_b = nvws.semaphore.create %alloc_b false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>
    // CHECK: [[BUF_A:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16,
    // CHECK: [[BUF_B:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16,
    // CHECK: [[TMA_EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,
    // CHECK: [[TES0:%.*]] = ttg.memdesc_index [[TMA_EMPTY]]
    // CHECK: ttng.init_barrier [[TES0]], 1
    // CHECK: [[TES1:%.*]] = ttg.memdesc_index [[TMA_EMPTY]]
    // CHECK: ttng.init_barrier [[TES1]], 1
    // CHECK: [[TES2:%.*]] = ttg.memdesc_index [[TMA_EMPTY]]
    // CHECK: ttng.init_barrier [[TES2]], 1
    // CHECK: [[TMA_FULL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,
    // CHECK: [[TFS0:%.*]] = ttg.memdesc_index [[TMA_FULL]]
    // CHECK: ttng.init_barrier [[TFS0]], 1
    // CHECK: [[TFS1:%.*]] = ttg.memdesc_index [[TMA_FULL]]
    // CHECK: ttng.init_barrier [[TFS1]], 1
    // CHECK: [[TFS2:%.*]] = ttg.memdesc_index [[TMA_FULL]]
    // CHECK: ttng.init_barrier [[TFS2]], 1
    %0 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %token) -> (!ttg.async.token) : i32 {
      // CHECK: addi
      // CHECK: cmpi
      // CHECK: [[WS_STAGE:%.*]] = arith.select
      // CHECK: xori
      // CHECK: cmpi
      // CHECK: [[WS_PHASE_P:%.*]] = arith.select
      // CHECK: [[TMA_EMPTY_WAIT:%.*]] = ttg.memdesc_index [[TMA_EMPTY]][[[WS_STAGE]]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.wait_barrier [[TMA_EMPTY_WAIT]], [[WS_PHASE_P]] {ttg.partition = array<i32: 0>}
      // CHECK: [[BUF_B_SLICE:%.*]] = ttg.memdesc_index [[BUF_B]][[[WS_STAGE]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[BUF_A_SLICE:%.*]] = ttg.memdesc_index [[BUF_A]][[[WS_STAGE]]] {ttg.partition = array<i32: 0>}
      // CHECK: [[TMA_FULL_SLICE:%.*]] = ttg.memdesc_index [[TMA_FULL]][[[WS_STAGE]]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.barrier_expect [[TMA_FULL_SLICE]], 32768 {ttg.partition = array<i32: 0>}
      // CHECK: ttng.async_tma_copy_global_to_local %arg0[{{.*}}] [[BUF_A_SLICE]], [[TMA_FULL_SLICE]], {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK: ttng.async_tma_copy_global_to_local %arg1[{{.*}}] [[BUF_B_SLICE]], [[TMA_FULL_SLICE]], {{.*}} {ttg.partition = array<i32: 0>}
      %tok_a = nvws.semaphore.acquire %empty_a {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]> -> !ttg.async.token
      %buf_a = nvws.semaphore.buffer %empty_a, %tok_a {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_ws_port, #smem_ws_port, mutable>
      nvws.descriptor_load %desc_a[%iv, %iv] 16384 %buf_a {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared_ws_port>, i32, i32, !ttg.memdesc<128x64xf16, #shared_ws_port, #smem_ws_port, mutable>
      nvws.semaphore.release %full_a, %tok_a [#nvws.async_op<tma_load>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>, !ttg.async.token
      %tok_b = nvws.semaphore.acquire %empty_b {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]> -> !ttg.async.token
      %buf_b = nvws.semaphore.buffer %empty_b, %tok_b {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_ws_port, #smem_ws_port, mutable>
      nvws.descriptor_load %desc_b[%iv, %iv] 16384 %buf_b {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared_ws_port>, i32, i32, !ttg.memdesc<128x64xf16, #shared_ws_port, #smem_ws_port, mutable>
      nvws.semaphore.release %full_b, %tok_b [#nvws.async_op<tma_load>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>, !ttg.async.token
      // CHECK: xori
      // CHECK: cmpi
      // CHECK: [[WS_PHASE_C:%.*]] = arith.select
      // CHECK: [[TMA_FULL_WAIT:%.*]] = ttg.memdesc_index [[TMA_FULL]][[[WS_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.wait_barrier [[TMA_FULL_WAIT]], [[WS_PHASE_C]] {ttg.partition = array<i32: 1>}
      // CHECK: [[BUF_B_SLICE_C:%.*]] = ttg.memdesc_index [[BUF_B]][[[WS_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[BUF_A_SLICE_C:%.*]] = ttg.memdesc_index [[BUF_A]][[[WS_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: [[BUF_B_TRANS:%.*]] = ttg.memdesc_trans [[BUF_B_SLICE_C]] {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma [[BUF_A_SLICE_C]], [[BUF_B_TRANS]], {{.*}} {is_async, ttg.partition = array<i32: 1>}
      // CHECK: [[TMA_EMPTY_COMMIT:%.*]] = ttg.memdesc_index [[TMA_EMPTY]][[[WS_STAGE]]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_commit [[TMA_EMPTY_COMMIT]] {ttg.partition = array<i32: 1>}
      %tok_ca = nvws.semaphore.acquire %full_a {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]> -> !ttg.async.token
      %cbuf_a = nvws.semaphore.buffer %full_a, %tok_ca {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_ws_port, #smem_ws_port, mutable>
      %tok_cb = nvws.semaphore.acquire %full_b {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]> -> !ttg.async.token
      %cbuf_b = nvws.semaphore.buffer %full_b, %tok_cb {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_ws_port, #smem_ws_port, mutable>
      %rhs = ttg.memdesc_trans %cbuf_b {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared_ws_port, #smem_ws_port, mutable> -> !ttg.memdesc<64x128xf16, #shared_ws_port_t, #smem_ws_port, mutable>
      %mma = ttng.tc_gen5_mma %cbuf_a, %rhs, %result[%acc], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared_ws_port, #smem_ws_port, mutable>, !ttg.memdesc<64x128xf16, #shared_ws_port_t, #smem_ws_port, mutable>, !ttg.memdesc<128x128xf32, #tmem_ws_port, #ttng.tensor_memory, mutable>
      nvws.semaphore.release %empty_a, %tok_ca [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>, !ttg.async.token
      nvws.semaphore.release %empty_b, %tok_cb [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_ws_port, #smem_ws_port, mutable>]>, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %mma : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], tt.warp_specialize}
    tt.return
  }
}

// -----

#blocked_mixed = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared_mixed = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem_mixed = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @load_used_as_reg_and_smem
  tt.func @load_used_as_reg_and_smem(%arg0: !tt.tensordesc<128x64xf16, #shared_mixed>, %arg1: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: ttng.init_barrier {{.*}}, 2
    // CHECK: ttng.init_barrier {{.*}}, 1
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>
    %empty = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>
    %full = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32 : i32 {
      %ptok = nvws.semaphore.acquire %empty {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]> -> !ttg.async.token
      %pbuf = nvws.semaphore.buffer %empty, %ptok {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_mixed, #smem_mixed, mutable>
      // CHECK: ttng.barrier_expect {{.*}}, 16384 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      nvws.descriptor_load %arg0[%arg2, %arg2] 16384 %pbuf {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared_mixed>, i32, i32, !ttg.memdesc<128x64xf16, #shared_mixed, #smem_mixed, mutable>
      nvws.semaphore.release %full, %ptok [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>, !ttg.async.token
      %gtok0 = nvws.semaphore.acquire %full {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]> -> !ttg.async.token
      %gbuf0 = nvws.semaphore.buffer %full, %gtok0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_mixed, #smem_mixed, mutable>
      %val = ttg.local_load %gbuf0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared_mixed, #smem_mixed, mutable> -> tensor<128x64xf16, #blocked_mixed>
      "use1"(%val) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked_mixed>) -> ()
      // CHECK: "use1"
      // CHECK: ttng.fence_async_shared {bCluster = false, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      nvws.semaphore.release %empty, %gtok0 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>, !ttg.async.token
      %gtok1 = nvws.semaphore.acquire %full {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]> -> !ttg.async.token
      %gbuf1 = nvws.semaphore.buffer %full, %gtok1 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_mixed, #smem_mixed, mutable>
      "use2_load"(%gbuf1) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<128x64xf16, #shared_mixed, #smem_mixed, mutable>) -> ()
      // CHECK: "use2_load"
      // CHECK: ttng.fence_async_shared {bCluster = false, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty, %gtok1 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>, !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----

#blocked_mixed = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared_mixed = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem_mixed = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @load_used_as_reg_and_smem_same_partition
  tt.func @load_used_as_reg_and_smem_same_partition(%arg0: !tt.tensordesc<128x64xf16, #shared_mixed>, %arg1: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: [[EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,
    // CHECK: [[EMPTY_INIT:%.*]] = ttg.memdesc_index [[EMPTY]][{{.*}}]
    // CHECK: ttng.init_barrier [[EMPTY_INIT]], 1
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>
    %empty = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>
    %full = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32 : i32 {
      %ptok = nvws.semaphore.acquire %empty {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]> -> !ttg.async.token
      %pbuf = nvws.semaphore.buffer %empty, %ptok {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_mixed, #smem_mixed, mutable>
      nvws.descriptor_load %arg0[%arg2, %arg2] 16384 %pbuf {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared_mixed>, i32, i32, !ttg.memdesc<128x64xf16, #shared_mixed, #smem_mixed, mutable>
      nvws.semaphore.release %full, %ptok [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>, !ttg.async.token
      %gtok = nvws.semaphore.acquire %full {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]> -> !ttg.async.token
      %gbuf = nvws.semaphore.buffer %full, %gtok {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_mixed, #smem_mixed, mutable>
      %val = ttg.local_load %gbuf {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared_mixed, #smem_mixed, mutable> -> tensor<128x64xf16, #blocked_mixed>
      // CHECK: ttng.wait_barrier {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: "use1"
      // CHECK: "use2_load"
      // CHECK: ttng.fence_async_shared {bCluster = false, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]][{{.*}}] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: ttng.arrive_barrier [[EMPTYSLICE]], 1 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      "use1"(%val) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked_mixed>) -> ()
      "use2_load"(%gbuf) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<128x64xf16, #shared_mixed, #smem_mixed, mutable>) -> ()
      nvws.semaphore.release %empty, %gtok [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_mixed, #smem_mixed, mutable>]>, !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----

#blocked7 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked8 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared7 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared8 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem7 = #ttg.shared_memory
#tmem7 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @semaphore_buffer
  tt.func @semaphore_buffer(%arg0: !tt.tensordesc<128x64xf16, #shared7>, %arg1: !tt.tensordesc<64x128xf16, #shared7>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked7>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked7>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: [[BUF:%.*]] = ttng.tmem_alloc
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>
    %empty = nvws.semaphore.create %result true : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result false : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]>
    %tok = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %buf0 = nvws.semaphore.buffer %empty, %tok : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem7, #ttng.tensor_memory, mutable, 2x128x128>
    %2 = ttng.tmem_store %cst_0, %buf0[], %true : tensor<128x128xf32, #blocked7> -> !ttg.memdesc<128x128xf32, #tmem7, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: scf.for
    %3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %tok) -> (!ttg.async.token) : i32 {
      %4:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared7> -> tensor<128x64xf16, #blocked8>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared7> -> tensor<64x128xf16, #blocked8>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked8>) -> !ttg.memdesc<128x64xf16, #shared7, #smem7>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked8>) -> !ttg.memdesc<64x128xf16, #shared7, #smem7>
      // CHECK: local_alloc
      // CHECK: local_alloc
      // CHECK: [[VIEW:%.*]] = ttg.memdesc_index [[BUF]][{{.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: tc_gen5_mma {{.*}}, {{.*}}, [[VIEW]][]{{.*}}
      %9 = nvws.semaphore.buffer %empty, %arg3 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem7, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared7, #smem7>, !ttg.memdesc<64x128xf16, #shared7, #smem7>, !ttg.memdesc<128x128xf32, #tmem7, #ttng.tensor_memory, mutable, 2x128x128>
      %11 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: [[RET_IF:%.*]]:{{.*}} = scf.if
      %12 = scf.if %11 -> (!ttg.async.token) {
        // CHECK: tc_gen5_commit
        // CHECK: ttng.wait_barrier
        // CHECK: [[VIEW2:%.*]] = ttg.memdesc_index [[BUF]]
        // CHECK: tmem_load [[VIEW2]]
        // CHECK: ttng.arrive_barrier
        nvws.semaphore.release %full, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %tok_get = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        %buf_get = nvws.semaphore.buffer %full, %tok_get {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem7, #ttng.tensor_memory, mutable, 2x128x128>
        %result_3, %token_4 = ttng.tmem_load %buf_get[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem7, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked7>
        nvws.semaphore.release %empty, %tok_get [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked7>) -> ()
        // CHECK: ttng.wait_barrier
        // CHECK: scf.yield
        %tok_reacq = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
        scf.yield %tok_reacq : !ttg.async.token
      } else {
        // CHECK: scf.yield
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: [[VIEW3:%.*]] = ttg.memdesc_index [[BUF]][[[RET_IF]]#{{.*}}]
      // CHECK: tmem_store {{.*}}, [[VIEW3]][]
      %13 = nvws.semaphore.buffer %empty, %12 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem7, #ttng.tensor_memory, mutable, 2x128x128>
      %14 = ttng.tmem_store %cst, %13[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked7> -> !ttg.memdesc<128x128xf32, #tmem7, #ttng.tensor_memory, mutable, 2x128x128>
      scf.yield %12 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    nvws.semaphore.release %full, %3 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x128x128xf32, #tmem7, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}

// -----

#blocked5 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked6 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared6 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem5 = #ttg.shared_memory
#tmem5 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @semaphore_not_in_loop
  tt.func @semaphore_not_in_loop(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf16, #shared5>, %arg4: !tt.tensordesc<128x64xf16, #shared5>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked5>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: [[TMEM:%.*]] = ttng.tmem_alloc
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem5, #ttng.tensor_memory, mutable>
    // EMPTY mbar (1 slot)
    // CHECK: [[MBAR_E:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
    // CHECK: ttng.init_barrier {{.*}}, 1
    // FULL mbar (1 slot)
    // CHECK: [[MBAR_F:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64,
    // CHECK: ttng.init_barrier {{.*}}, 1
    %empty = nvws.semaphore.create %result true : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem5, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result false : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem5, #ttng.tensor_memory, mutable>]>
    // Pre-loop: wait EMPTY, get buffer view, tmem_store
    // CHECK: ttng.wait_barrier
    // CHECK: [[BUF_PRE:%.*]] = ttg.memdesc_index [[TMEM]]
    // CHECK: ttng.tmem_store {{.*}}, [[BUF_PRE]][]
    %tok = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem5, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %buf = nvws.semaphore.buffer %empty, %tok : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem5, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem5, #ttng.tensor_memory, mutable, 1x128x128>
    %2 = ttng.tmem_store %cst, %buf[], %true : tensor<128x128xf32, #blocked5> -> !ttg.memdesc<128x128xf32, #tmem5, #ttng.tensor_memory, mutable, 1x128x128>
    // In-loop: buffer view from pre-loop token, MMA
    // CHECK: scf.for
    scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 : i32 {
      %4 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %5 = tt.descriptor_load %arg3[%arg1, %4] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared5> -> tensor<128x64xf16, #blocked6>
      %6 = tt.descriptor_load %arg4[%arg2, %4] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared5> -> tensor<128x64xf16, #blocked6>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked6>) -> !ttg.memdesc<128x64xf16, #shared5, #smem5>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked6>) -> !ttg.memdesc<128x64xf16, #shared5, #smem5>
      // CHECK: [[BUF_LOOP:%.*]] = ttg.memdesc_index [[TMEM]][{{.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF_LOOP]][]{{.*}} {ttg.partition = array<i32: 1>}
      %9 = ttg.memdesc_trans %8 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared5, #smem5> -> !ttg.memdesc<64x128xf16, #shared6, #smem5>
      %10 = nvws.semaphore.buffer %empty, %tok {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem5, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem5, #ttng.tensor_memory, mutable, 1x128x128>
      %11 = ttng.tc_gen5_mma %7, %9, %10[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared5, #smem5>, !ttg.memdesc<64x128xf16, #shared6, #smem5>, !ttg.memdesc<128x128xf32, #tmem5, #ttng.tensor_memory, mutable, 1x128x128>
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>}
    // Post-loop: tc_gen5_commit (release FULL), wait FULL, buffer view, tmem_load, arrive (release EMPTY)
    // CHECK: ttng.tc_gen5_commit
    // CHECK: ttng.wait_barrier
    // CHECK: [[BUF_POST:%.*]] = ttg.memdesc_index [[TMEM]]
    // CHECK: ttng.tmem_load [[BUF_POST]][]
    // CHECK: ttng.arrive_barrier {{.*}}, 1
    nvws.semaphore.release %full, %tok [#nvws.async_op<tc5mma>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem5, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %tok_get = nvws.semaphore.acquire %full : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem5, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %buf_get = nvws.semaphore.buffer %full, %tok_get : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem5, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem5, #ttng.tensor_memory, mutable, 1x128x128>
    %result_2, %token_3 = ttng.tmem_load %buf_get[] : !ttg.memdesc<128x128xf32, #tmem5, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked5>
    nvws.semaphore.release %empty, %tok_get [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem5, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    "use"(%result_2) : (tensor<128x128xf32, #blocked5>) -> ()
    tt.return
  }
}

// -----

#blocked9 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked10 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#linear9 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear10 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared9 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem9 = #ttg.shared_memory
#tmem9 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales9 = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @semaphore_scale_mma_user
  tt.func @semaphore_scale_mma_user(%arg0: !ttg.memdesc<128x64xf16, #shared9, #smem9>, %arg1: !ttg.memdesc<64x128xf16, #shared9, #smem9>, %arg2: !tt.tensordesc<8x128xi8, #shared9>, %arg3: !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>, %arg4: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked9>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>
    %empty = nvws.semaphore.create %result true : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]>
    %full = nvws.semaphore.create %result false : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]>
    %tok = nvws.semaphore.acquire %empty : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %buf0 = nvws.semaphore.buffer %empty, %tok : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable, 1x128x128>
    %2 = ttng.tmem_store %cst, %buf0[], %true : tensor<128x128xf32, #blocked9> -> !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: scf.for
    %3 = scf.for %arg5 = %c0_i32 to %arg4 step %c1_i32 iter_args(%arg6 = %tok) -> (!ttg.async.token) : i32 {
      %5 = tt.descriptor_load %arg2[%arg5, %arg5] {ttg.partition = array<i32: 2>} : !tt.tensordesc<8x128xi8, #shared9> -> tensor<8x128xi8, #blocked10>
      %6 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<8x128xi8, #blocked10>) -> !ttg.memdesc<8x128xi8, #shared9, #smem9>
      %7 = ttg.local_load %6 {ttg.partition = array<i32: 0>} : !ttg.memdesc<8x128xi8, #shared9, #smem9> -> tensor<8x128xi8, #linear10>
      %8 = tt.trans %7 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<8x128xi8, #linear10> -> tensor<128x8xi8, #linear9>
      // CHECK: tmem_alloc {{.*}} {ttg.partition = array<i32: 0, 1>}
      %result_4 = ttng.tmem_alloc %8 {ttg.partition = array<i32: 0, 1>} : (tensor<128x8xi8, #linear9>) -> !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>
      // CHECK: tc_gen5_mma_scaled {{.*}} {ttg.partition = array<i32: 1>}
      %9 = nvws.semaphore.buffer %empty, %arg6 {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable, 1x128x128>
      %10 = ttng.tc_gen5_mma_scaled %arg0, %arg1, %9[], %result_4, %arg3, %true, %true lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared9, #smem9>, !ttg.memdesc<64x128xf16, #shared9, #smem9>, !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales9, #ttng.tensor_memory>
      // CHECK: tc_gen5_commit
      // CHECK: ttng.wait_barrier
      // CHECK: ttng.tmem_load
      // CHECK: ttng.arrive_barrier
      // CHECK: ttng.wait_barrier
      nvws.semaphore.release %full, %arg6 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %tok_get = nvws.semaphore.acquire %full {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %buf_get = nvws.semaphore.buffer %full, %tok_get {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable, 1x128x128>
      %result_7, %token_8 = ttng.tmem_load %buf_get[] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked9>
      nvws.semaphore.release %empty, %tok_get [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "user"(%result_7) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked9>) -> ()
      %tok_reacq = nvws.semaphore.acquire %empty {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      scf.yield %tok_reacq : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 16 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // Post-loop: release + acquire + load + release
    // CHECK: ttng.arrive_barrier
    // CHECK: ttng.wait_barrier
    // CHECK: ttng.tmem_load
    // CHECK: ttng.arrive_barrier
    nvws.semaphore.release %full, %3 [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %tok_final = nvws.semaphore.acquire %full : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
    %buf_final = nvws.semaphore.buffer %full, %tok_final : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable, 1x128x128>
    %result_2, %token_3 = ttng.tmem_load %buf_final[] : !ttg.memdesc<128x128xf32, #tmem9, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked9>
    nvws.semaphore.release %empty, %tok_final [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<1x128x128xf32, #tmem9, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    "use"(%result_2) : (tensor<128x128xf32, #blocked9>) -> ()
    tt.return
  }
}

// -----

#blocked_attn = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared_attn = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_attn_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem_attn = #ttg.shared_memory
#tmem_attn = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @attention_forward
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared_attn, #smem_attn>, %arg1: !tt.tensordesc<64x64xf16, #shared_attn>, %arg2: f32, %arg3: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked_attn>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %acc = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>
    %acc_empty = nvws.semaphore.create %acc true : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]>
    %acc_full = nvws.semaphore.create %acc false : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]>
    %tmp = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>
    %tmp_empty = nvws.semaphore.create %tmp true : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]>
    %tmp_full = nvws.semaphore.create %tmp false : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]>
    %kbuf = ttg.local_alloc : () -> !ttg.memdesc<1x64x64xf16, #shared_attn, #smem_attn, mutable>
    %k_empty = nvws.semaphore.create %kbuf true : !nvws.semaphore<[!ttg.memdesc<1x64x64xf16, #shared_attn, #smem_attn, mutable>]>
    %k_full = nvws.semaphore.create %kbuf false : !nvws.semaphore<[!ttg.memdesc<1x64x64xf16, #shared_attn, #smem_attn, mutable>]>
    %prob = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared_attn, #smem_attn, mutable>
    %prob_empty = nvws.semaphore.create %prob true : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #shared_attn, #smem_attn, mutable>]>
    %prob_full = nvws.semaphore.create %prob false : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #shared_attn, #smem_attn, mutable>]>

    scf.for %iv = %c0_i32 to %arg3 step %c64_i32 : i32 {
      %k_ptok = nvws.semaphore.acquire %k_empty {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x64x64xf16, #shared_attn, #smem_attn, mutable>]> -> !ttg.async.token
      %k_pbuf = nvws.semaphore.buffer %k_empty, %k_ptok {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x64x64xf16, #shared_attn, #smem_attn, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x64xf16, #shared_attn, #smem_attn, mutable>
      nvws.descriptor_load %arg1[%iv, %c0_i32] 8192 %k_pbuf {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<64x64xf16, #shared_attn>, i32, i32, !ttg.memdesc<64x64xf16, #shared_attn, #smem_attn, mutable>
      nvws.semaphore.release %k_full, %k_ptok [#nvws.async_op<tma_load>] {loop.cluster = 4 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<1x64x64xf16, #shared_attn, #smem_attn, mutable>]>, !ttg.async.token

      %k_gtok = nvws.semaphore.acquire %k_full {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x64x64xf16, #shared_attn, #smem_attn, mutable>]> -> !ttg.async.token
      %k_cbuf = nvws.semaphore.buffer %k_full, %k_gtok {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x64x64xf16, #shared_attn, #smem_attn, mutable>]>, !ttg.async.token -> !ttg.memdesc<64x64xf16, #shared_attn, #smem_attn, mutable>
      %k_trans = ttg.memdesc_trans %k_cbuf {loop.cluster = 2 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x64xf16, #shared_attn, #smem_attn, mutable> -> !ttg.memdesc<64x64xf16, #shared_attn_t, #smem_attn, mutable>
      %acc_ptok = nvws.semaphore.acquire %acc_empty {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %acc_pbuf = nvws.semaphore.buffer %acc_empty, %acc_ptok {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable, 1x256x64>
      %mma = ttng.tc_gen5_mma %arg0, %k_trans, %acc_pbuf[], %false, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #shared_attn, #smem_attn>, !ttg.memdesc<64x64xf16, #shared_attn_t, #smem_attn, mutable>, !ttg.memdesc<256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.semaphore.release %acc_full, %acc_ptok [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      nvws.semaphore.release %k_empty, %k_gtok [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x64x64xf16, #shared_attn, #smem_attn, mutable>]>, !ttg.async.token

      %acc_gtok = nvws.semaphore.acquire %acc_full {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %acc_gbuf = nvws.semaphore.buffer %acc_full, %acc_gtok {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable, 1x256x64>
      %acc_val, %acc_tok = ttng.tmem_load %acc_gbuf[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked_attn>
      nvws.semaphore.release %acc_empty, %acc_gtok [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %prob_val = arith.truncf %acc_val {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked_attn> to tensor<256x64xf16, #blocked_attn>
      %prob_ptok = nvws.semaphore.acquire %prob_empty {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #shared_attn, #smem_attn, mutable>]> -> !ttg.async.token
      %prob_pbuf = nvws.semaphore.buffer %prob_empty, %prob_ptok {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #shared_attn, #smem_attn, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared_attn, #smem_attn, mutable>
      // CHECK: ttg.local_store
      // CHECK: ttng.fence_async_shared
      // CHECK: ttng.arrive_barrier
      ttg.local_store %prob_val, %prob_pbuf {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : tensor<256x64xf16, #blocked_attn> -> !ttg.memdesc<256x64xf16, #shared_attn, #smem_attn, mutable>
      nvws.semaphore.release %prob_full, %prob_ptok [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #shared_attn, #smem_attn, mutable>]>, !ttg.async.token

      %prob_gtok = nvws.semaphore.acquire %prob_full {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #shared_attn, #smem_attn, mutable>]> -> !ttg.async.token
      %prob_gbuf = nvws.semaphore.buffer %prob_full, %prob_gtok {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #shared_attn, #smem_attn, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf16, #shared_attn, #smem_attn, mutable>
      %prob_tensor = ttg.local_load %prob_gbuf {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 3>} : !ttg.memdesc<256x64xf16, #shared_attn, #smem_attn, mutable> -> tensor<256x64xf16, #blocked_attn>
      nvws.semaphore.release %prob_empty, %prob_gtok [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf16, #shared_attn, #smem_attn, mutable>]>, !ttg.async.token
      %prob_f32 = arith.extf %prob_tensor {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 3>} : tensor<256x64xf16, #blocked_attn> to tensor<256x64xf32, #blocked_attn>
      %tmp_ptok = nvws.semaphore.acquire %tmp_empty {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]> -> !ttg.async.token
      %tmp_pbuf = nvws.semaphore.buffer %tmp_empty, %tmp_ptok {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable, 1x256x64>
      %tmp_old, %tmp_oldtok = ttng.tmem_load %tmp_pbuf[] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 3>} : !ttg.memdesc<256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable, 1x256x64> -> tensor<256x64xf32, #blocked_attn>
      %tmp_new = arith.addf %tmp_old, %prob_f32 {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked_attn>
      // CHECK: ttng.tmem_store
      // CHECK-NOT: ttng.fence_async_shared
      // CHECK: ttng.arrive_barrier
      %tmp_store = ttng.tmem_store %tmp_new, %tmp_pbuf[], %true {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked_attn> -> !ttg.memdesc<256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable, 1x256x64>
      nvws.semaphore.release %tmp_full, %tmp_ptok [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 3>} : !nvws.semaphore<[!ttg.memdesc<1x256x64xf32, #tmem_attn, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @multi_buffer_trans_chain
  // Tests multi-buffer semaphore.buffer rewriting when one buffer flows
  // through memdesc_trans and another is consumed directly.
  tt.func @multi_buffer_trans_chain() {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %buf_a = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    %buf_b = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    %sem = nvws.semaphore.create %buf_a, %buf_b true : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]> -> !ttg.async.token
    %views:2 = nvws.semaphore.buffer %sem, %tok {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
    %t = ttg.memdesc_trans %views#0 {loop.cluster = 6 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64> -> !ttg.memdesc<64x128xf16, #shared_t, #smem, mutable, 1x64x128>
    // CHECK-COUNT-2: ttg.memdesc_index %{{.*}}[%{{.*}}] {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: [[A_TRANS:%.*]] = ttg.memdesc_trans %{{.*}} {loop.cluster = 6 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
    // CHECK: "use_pair_load"([[A_TRANS]], %{{.*}}) : (!ttg.memdesc<64x128xf16, #shared2, #smem, mutable>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable>) -> ()
    "use_pair_load"(%t, %views#1) : (!ttg.memdesc<64x128xf16, #shared_t, #smem, mutable, 1x64x128>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>) -> ()
    nvws.semaphore.release %sem, %tok [#nvws.async_op<tc5mma>] {loop.cluster = 6 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf_a : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %buf_b : !ttg.memdesc<2x128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared_ws = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_ws_t = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem_ws = #ttg.shared_memory
#tmem_ws = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @ws_multibuffer_trans_chain
  // Regression: semaphore.buffer with allocShape-bearing shared buffers feeding
  // memdesc_trans inside a warp-specialized TMA/MMA pipeline must lower to the
  // same stage slice shape that aref lowering uses.
  tt.func @ws_multibuffer_trans_chain(%desc_a: !tt.tensordesc<128x64xf16, #shared_ws>, %desc_b: !tt.tensordesc<128x64xf16, #shared_ws>, %lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>
    %result, %acc = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem_ws, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %empty_a = nvws.semaphore.create %alloc_a true : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>
    %full_a = nvws.semaphore.create %alloc_a false : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>
    %empty_b = nvws.semaphore.create %alloc_b true : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>
    %full_b = nvws.semaphore.create %alloc_b false : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>
    // Full pipeline multibuffers both shared buffers to depth 3.
    // CHECK: [[ALLOC_A:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16,
    // CHECK: [[ALLOC_B:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16,
    // The two EMPTY/FULL pairs are combined into one pair of mbar arrays.
    // CHECK: [[MBAR_EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,
    // CHECK-COUNT-3: ttng.init_barrier %{{.*}}, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: [[MBAR_FULL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,
    // CHECK-COUNT-3: ttng.init_barrier %{{.*}}, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %0 = scf.for %iv = %lb to %ub step %step iter_args(%tok_acc = %acc) -> (!ttg.async.token) : i32 {
      // Producer side: wait EMPTY and issue TMA loads.
      %tok_a = nvws.semaphore.acquire %empty_a[%c0_i32, %c1_i32] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]> -> !ttg.async.token
      %buf_a = nvws.semaphore.buffer %empty_a[%c0_i32], %tok_a {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_ws, #smem_ws, mutable, 1x128x64>
      nvws.descriptor_load %desc_a[%iv, %iv] 16384 %buf_a {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared_ws>, i32, i32, !ttg.memdesc<128x64xf16, #shared_ws, #smem_ws, mutable, 1x128x64>
      nvws.semaphore.release %full_a[%c0_i32], %tok_a [#nvws.async_op<tma_load>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>, !ttg.async.token
      %tok_b = nvws.semaphore.acquire %empty_b[%c0_i32, %c1_i32] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]> -> !ttg.async.token
      %buf_b = nvws.semaphore.buffer %empty_b[%c0_i32], %tok_b {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_ws, #smem_ws, mutable, 1x128x64>
      nvws.descriptor_load %desc_b[%iv, %iv] 16384 %buf_b {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared_ws>, i32, i32, !ttg.memdesc<128x64xf16, #shared_ws, #smem_ws, mutable, 1x128x64>
      nvws.semaphore.release %full_b[%c0_i32], %tok_b [#nvws.async_op<tma_load>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>, !ttg.async.token
      // Consumer side: same stage-sliced shared buffers feed memdesc_trans/MMA.
      %tok_ca = nvws.semaphore.acquire %full_a[%c0_i32, %c1_i32] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]> -> !ttg.async.token
      %cbuf_a = nvws.semaphore.buffer %full_a[%c0_i32], %tok_ca {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_ws, #smem_ws, mutable, 1x128x64>
      %tok_cb = nvws.semaphore.acquire %full_b[%c0_i32, %c1_i32] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]> -> !ttg.async.token
      %cbuf_b = nvws.semaphore.buffer %full_b[%c0_i32], %tok_cb {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_ws, #smem_ws, mutable, 1x128x64>
      %rhs = ttg.memdesc_trans %cbuf_b {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared_ws, #smem_ws, mutable, 1x128x64> -> !ttg.memdesc<64x128xf16, #shared_ws_t, #smem_ws, mutable, 1x64x128>
      // CHECK-DAG: ttg.memdesc_index [[ALLOC_A]][%{{.*}}] {ttg.partition = array<i32: 0>}
      // CHECK-DAG: ttg.memdesc_index [[ALLOC_B]][%{{.*}}] {ttg.partition = array<i32: 0>}
      // CHECK-DAG: ttng.async_tma_copy_global_to_local %arg0[%{{.*}}, %{{.*}}] %{{.*}}, %{{.*}}, %{{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-DAG: ttng.async_tma_copy_global_to_local %arg1[%{{.*}}, %{{.*}}] %{{.*}}, %{{.*}}, %{{.*}} {ttg.partition = array<i32: 0>}
      // CHECK: [[CBUF_B:%.*]] = ttg.memdesc_index [[ALLOC_B]][%{{.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: [[CBUF_A:%.*]] = ttg.memdesc_index [[ALLOC_A]][%{{.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: [[RHS:%.*]] = ttg.memdesc_trans [[CBUF_B]] {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
      // CHECK: ttng.tc_gen5_mma [[CBUF_A]], [[RHS]], %{{.*}}[%{{.*}}], %true, %true {is_async, ttg.partition = array<i32: 1>}
      %mma = ttng.tc_gen5_mma %cbuf_a, %rhs, %result[%tok_acc], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared_ws, #smem_ws, mutable, 1x128x64>, !ttg.memdesc<64x128xf16, #shared_ws_t, #smem_ws, mutable, 1x64x128>, !ttg.memdesc<128x128xf32, #tmem_ws, #ttng.tensor_memory, mutable>
      nvws.semaphore.release %empty_a[%c0_i32], %tok_ca [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>, !ttg.async.token
      nvws.semaphore.release %empty_b[%c0_i32], %tok_cb [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_ws, #smem_ws, mutable>]>, !ttg.async.token
      scf.yield %mma : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], tt.warp_specialize}
    tt.return
  }
}

// -----

#shared_desc = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @dual_tma_load_order
  // Tests that barrier_expect is emitted before both rewritten async TMA
  // loads and that both loads use the same mbar/predicate operands.
  tt.func @dual_tma_load_order(%desc_a: !tt.tensordesc<128x64xf16, #shared_desc>, %desc_b: !tt.tensordesc<128x64xf16, #shared_desc>) {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %buf_a = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>
    %buf_b = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>
    %sem = nvws.semaphore.create %buf_a, %buf_b false : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>]>
    %tok = nvws.semaphore.acquire %sem {loop.cluster = 7 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>]> -> !ttg.async.token
    %views:2 = nvws.semaphore.buffer %sem, %tok {loop.cluster = 7 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_desc, #smem, mutable, 1x128x64>, !ttg.memdesc<128x64xf16, #shared_desc, #smem, mutable, 1x128x64>
    nvws.descriptor_load %desc_a[%c0_i32, %c0_i32] 4096 %views#0 {loop.cluster = 7 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared_desc>, i32, i32, !ttg.memdesc<128x64xf16, #shared_desc, #smem, mutable, 1x128x64>
    nvws.descriptor_load %desc_b[%c0_i32, %c0_i32] 4096 %views#1 {loop.cluster = 7 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared_desc>, i32, i32, !ttg.memdesc<128x64xf16, #shared_desc, #smem, mutable, 1x128x64>
    // CHECK-COUNT-2: ttg.memdesc_index %{{.*}}[%{{.*}}] {loop.cluster = 7 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable>
    // CHECK: [[TMA_MBAR:%.*]] = ttg.memdesc_index %{{.*}}[%{{.*}}] {loop.cluster = 7 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !ttg.memdesc<3x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: [[TMA_PRED:%.*]] = arith.constant {loop.cluster = 7 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} true
    // CHECK: ttng.barrier_expect [[TMA_MBAR]], 8192 {loop.cluster = 7 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}, [[TMA_PRED]]
    // CHECK: ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %{{.*}}, [[TMA_MBAR]], [[TMA_PRED]] {loop.cluster = 7 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: ttng.async_tma_copy_global_to_local %arg1[%c0_i32, %c0_i32] %{{.*}}, [[TMA_MBAR]], [[TMA_PRED]] {loop.cluster = 7 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    nvws.semaphore.release %sem, %tok [#nvws.async_op<tma_load>] {loop.cluster = 7 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !nvws.semaphore<[!ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>, !ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>]>, !ttg.async.token
    ttg.local_dealloc %buf_a : !ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>
    ttg.local_dealloc %buf_b : !ttg.memdesc<2x128x64xf16, #shared_desc, #smem, mutable>
    tt.return
  }
}

// -----
// depth=1 unassigned semaphores → multiBuffer(3) + AssignStagePhase + lower.
// Verifies the full phase2 pipeline: alloc expanded from 1→3, mbar has 3 slots,
// stage/phase threaded through loop, TMA lowered to barrier_expect + async_tma_copy.

#blocked2 = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared_desc2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem2 = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @phase2_multibuffer
  tt.func @phase2_multibuffer(%desc: !tt.tensordesc<128x64xf16, #shared_desc2>, %lb: i32, %ub: i32, %step: i32) {
    // Multi-buffered alloc: 1x → 3x
    // CHECK: [[BUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16,
    // EMPTY mbar: 3 slots, init_barrier on each
    // CHECK: [[MBAR_E:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,
    // CHECK: ttng.init_barrier {{.*}}, 1
    // CHECK: ttng.init_barrier {{.*}}, 1
    // CHECK: ttng.init_barrier {{.*}}, 1
    // FULL mbar: 3 slots, init_barrier on each
    // CHECK: [[MBAR_F:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,
    // CHECK: ttng.init_barrier {{.*}}, 1
    // CHECK: ttng.init_barrier {{.*}}, 1
    // CHECK: ttng.init_barrier {{.*}}, 1
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared_desc2, #smem2, mutable>
    %sem_empty = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc2, #smem2, mutable>]>
    %sem_full = nvws.semaphore.create %buf false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc2, #smem2, mutable>]>
    // Stage/phase threaded as iter_args
    // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (i32, i32, i32)
    scf.for %i = %lb to %ub step %step : i32 {
      // Producer: wait_barrier on EMPTY, then TMA load
      // CHECK: ttng.wait_barrier {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK: [[PBUF:%.*]] = ttg.memdesc_index [[BUF]][{{.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.barrier_expect {{.*}}, 16384 {ttg.partition = array<i32: 0>}
      // CHECK: ttng.async_tma_copy_global_to_local {{.*}} [[PBUF]], {{.*}} {ttg.partition = array<i32: 0>}
      %ptok = nvws.semaphore.acquire %sem_empty {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc2, #smem2, mutable>]> -> !ttg.async.token
      %pbuf = nvws.semaphore.buffer %sem_empty, %ptok {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc2, #smem2, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_desc2, #smem2, mutable>
      nvws.descriptor_load %desc[%i, %i] 16384 %pbuf {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared_desc2>, i32, i32, !ttg.memdesc<128x64xf16, #shared_desc2, #smem2, mutable>
      nvws.semaphore.release %sem_full, %ptok [#nvws.async_op<tma_load>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc2, #smem2, mutable>]>, !ttg.async.token
      // Consumer: wait_barrier on FULL, local_load, fence, arrive on EMPTY
      // CHECK: ttng.wait_barrier {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK: [[CBUF:%.*]] = ttg.memdesc_index [[BUF]][{{.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_load [[CBUF]] {ttg.partition = array<i32: 1>}
      // CHECK: ttng.fence_async_shared {bCluster = false, ttg.partition = array<i32: 1>}
      // CHECK: ttng.arrive_barrier {{.*}}, 1 {ttg.partition = array<i32: 1>}
      %gtok = nvws.semaphore.acquire %sem_full {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc2, #smem2, mutable>]> -> !ttg.async.token
      %gbuf = nvws.semaphore.buffer %sem_full, %gtok {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc2, #smem2, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared_desc2, #smem2, mutable>
      %v = ttg.local_load %gbuf {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared_desc2, #smem2, mutable> -> tensor<128x64xf16, #blocked2>
      nvws.semaphore.release %sem_empty, %gtok [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared_desc2, #smem2, mutable>]>, !ttg.async.token
      "use"(%v) {ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked2>) -> ()
    } {ttg.partition = array<i32: 0, 1>, tt.warp_specialize}
    // Cleanup MBAR_E: inval 3 slots then dealloc
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttg.local_dealloc [[MBAR_E]]
    // Cleanup MBAR_F: inval 3 slots then dealloc
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttg.local_dealloc [[MBAR_F]]
    // Dealloc buffer
    // CHECK: ttg.local_dealloc [[BUF]]
    ttg.local_dealloc %buf : !ttg.memdesc<1x128x64xf16, #shared_desc2, #smem2, mutable>
    tt.return
  }
}

// -----
// combine test: two separate SMEM semaphore pairs feeding the same
// tc_gen5_mma consumer → combined into 1 pair.  Verifies: 2 mbar arrays
// (not 4), barrier_expect 32768 (16384+16384), both async_tma_copy to same mbar,
// tc_gen5_mma with is_async, tc_gen5_commit on consumer release.

#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem3 = #ttg.shared_memory
#tmem3 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @combine_two_tma_loads
  tt.func @combine_two_tma_loads(%desc_a: !tt.tensordesc<128x64xf16, #shared3>, %desc_b: !tt.tensordesc<128x64xf16, #shared3>, %lb: i32, %ub: i32, %step: i32) {
    %true = arith.constant true
    // Two allocs multi-buffered to 3x
    // CHECK: [[ALLOC_A:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16,
    // CHECK: [[ALLOC_B:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16,
    %alloc_a = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>
    %alloc_b = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem3, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // Combined: only 2 mbar arrays (EMPTY + FULL), 3 slots each
    // CHECK: [[MBAR_E:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,
    // CHECK: ttng.init_barrier {{.*}}, 1
    // CHECK: ttng.init_barrier {{.*}}, 1
    // CHECK: ttng.init_barrier {{.*}}, 1
    // CHECK: [[MBAR_F:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64,
    // CHECK: ttng.init_barrier {{.*}}, 1
    // CHECK: ttng.init_barrier {{.*}}, 1
    // CHECK: ttng.init_barrier {{.*}}, 1
    %empty_a = nvws.semaphore.create %alloc_a true : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>
    %full_a = nvws.semaphore.create %alloc_a false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>
    %empty_b = nvws.semaphore.create %alloc_b true : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>
    %full_b = nvws.semaphore.create %alloc_b false : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>
    // Stage/phase iter_args: 1 shared stage + 1 producer phase + 1 consumer phase = 4 iter_args (+ acc token)
    // CHECK: scf.for {{.*}} iter_args({{.*}}) -> (!ttg.async.token, i32, i32, i32)
    %0 = scf.for %iv = %lb to %ub step %step iter_args(%acc = %token) -> (!ttg.async.token) : i32 {
      // Producer: wait EMPTY mbar, get both buffer views, TMA loads, barrier_expect with summed TX
      // CHECK: ttng.wait_barrier {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK: [[PBUF_B:%.*]] = ttg.memdesc_index [[ALLOC_B]][{{.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: [[PBUF_A:%.*]] = ttg.memdesc_index [[ALLOC_A]][{{.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: [[TMA_MBAR:%.*]] = ttg.memdesc_index [[MBAR_F]][{{.*}}] {ttg.partition = array<i32: 0>}
      // CHECK: [[TMA_PRED:%.*]] = arith.constant {ttg.partition = array<i32: 0>} true
      // CHECK: ttng.barrier_expect [[TMA_MBAR]], 32768 {ttg.partition = array<i32: 0>}, [[TMA_PRED]]
      // CHECK: ttng.async_tma_copy_global_to_local %arg0[{{.*}}] [[PBUF_A]], [[TMA_MBAR]], [[TMA_PRED]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.async_tma_copy_global_to_local %arg1[{{.*}}] [[PBUF_B]], [[TMA_MBAR]], [[TMA_PRED]] {ttg.partition = array<i32: 0>}
      %tok_a = nvws.semaphore.acquire %empty_a {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]> -> !ttg.async.token
      %buf_a = nvws.semaphore.buffer %empty_a, %tok_a {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable>
      nvws.descriptor_load %desc_a[%iv, %iv] 16384 %buf_a {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared3>, i32, i32, !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable>
      nvws.semaphore.release %full_a, %tok_a [#nvws.async_op<tma_load>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>, !ttg.async.token
      %tok_b = nvws.semaphore.acquire %empty_b {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]> -> !ttg.async.token
      %buf_b = nvws.semaphore.buffer %empty_b, %tok_b {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable>
      nvws.descriptor_load %desc_b[%iv, %iv] 16384 %buf_b {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x64xf16, #shared3>, i32, i32, !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable>
      nvws.semaphore.release %full_b, %tok_b [#nvws.async_op<tma_load>] {ttg.partition = array<i32: 0>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>, !ttg.async.token
      // Consumer: wait FULL mbar, get both buffer views, MMA
      // CHECK: ttng.wait_barrier {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK: [[CBUF_B:%.*]] = ttg.memdesc_index [[ALLOC_B]][{{.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: [[CBUF_A:%.*]] = ttg.memdesc_index [[ALLOC_A]][{{.*}}] {ttg.partition = array<i32: 1>}
      // CHECK: [[RHS:%.*]] = ttg.memdesc_trans [[CBUF_B]] {{{.*}}ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma [[CBUF_A]], [[RHS]], {{.*}} {is_async, ttg.partition = array<i32: 1>}
      %tok_ca = nvws.semaphore.acquire %full_a {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]> -> !ttg.async.token
      %cbuf_a = nvws.semaphore.buffer %full_a, %tok_ca {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable>
      %tok_cb = nvws.semaphore.acquire %full_b {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]> -> !ttg.async.token
      %cbuf_b = nvws.semaphore.buffer %full_b, %tok_cb {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable>
      %rhs = ttg.memdesc_trans %cbuf_b {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable> -> !ttg.memdesc<64x128xf16, #shared4, #smem3, mutable>
      %mma = ttng.tc_gen5_mma %cbuf_a, %rhs, %result[%acc], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared3, #smem3, mutable>, !ttg.memdesc<64x128xf16, #shared4, #smem3, mutable>, !ttg.memdesc<128x128xf32, #tmem3, #ttng.tensor_memory, mutable>
      // Consumer release: tc_gen5_commit (from tc5mma async_op)
      // CHECK: ttng.tc_gen5_commit {{.*}} {ttg.partition = array<i32: 1>}
      nvws.semaphore.release %empty_a, %tok_ca [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>, !ttg.async.token
      nvws.semaphore.release %empty_b, %tok_cb [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.semaphore<[!ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>]>, !ttg.async.token
      scf.yield {ttg.partition = array<i32: 0, 1>} %mma : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], tt.warp_specialize}
    // Cleanup: 3 inval + dealloc per mbar, then buffer deallocs
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttg.local_dealloc [[MBAR_E]]
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttg.local_dealloc [[MBAR_F]]
    // CHECK: ttg.local_dealloc [[ALLOC_A]]
    // CHECK: ttg.local_dealloc [[ALLOC_B]]
    ttg.local_dealloc %alloc_a : !ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>
    ttg.local_dealloc %alloc_b : !ttg.memdesc<1x128x64xf16, #shared3, #smem3, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cleanup_after_last_user
  // Tests that barrier invalidation and dealloc happen before "after_last_user"
  tt.func @cleanup_after_last_user() {
    %c0_i32 = arith.constant 0 : i32
    %cm1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    // CHECK: [[MBAR_CL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
    // CHECK: [[CL_S0:%.*]] = ttg.memdesc_index [[MBAR_CL]][{{%.*}}]
    // CHECK: ttng.init_barrier [[CL_S0]], 1
    // CHECK: [[CL_S1:%.*]] = ttg.memdesc_index [[MBAR_CL]][{{%.*}}]
    // CHECK: ttng.init_barrier [[CL_S1]], 1
    %sem = nvws.semaphore.create %buf true : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>

    // Acquire: full pipeline computes the phase bit before wait_barrier.
    // CHECK: [[CL_MBAR:%.*]] = ttg.memdesc_index [[MBAR_CL]][%{{.*}}]
    // CHECK: ttng.wait_barrier [[CL_MBAR]], {{%.*}}
    // CHECK: [[CL_BUF:%.*]] = ttg.memdesc_index %{{.*}}[%{{.*}}]
    // CHECK: ttg.local_load [[CL_BUF]]
    // CHECK: "sink"
    %tok = nvws.semaphore.acquire %sem : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.async.token
    %view = nvws.semaphore.buffer %sem, %tok : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %v = ttg.local_load %view : !ttg.memdesc<1xi32, #shared, #smem, mutable> -> tensor<1xi32>
    "sink"(%v) : (tensor<1xi32>) -> ()
    // Release: arrive, then cleanup BEFORE "after_last_user"
    // CHECK: [[CL_MBAR_REL:%.*]] = ttg.memdesc_index [[MBAR_CL]][%{{.*}}]
    // CHECK: ttng.arrive_barrier [[CL_MBAR_REL]], 1
    // CHECK: [[CL_INV0:%.*]] = ttg.memdesc_index [[MBAR_CL]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[CL_INV0]]
    // CHECK: [[CL_INV1:%.*]] = ttg.memdesc_index [[MBAR_CL]][{{%.*}}]
    // CHECK: ttng.inval_barrier [[CL_INV1]]
    // CHECK: ttg.local_dealloc [[MBAR_CL]]
    // CHECK: "after_last_user"()
    nvws.semaphore.release %sem, %tok [#nvws.async_op<none>] : !nvws.semaphore<[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

    "after_last_user"() : () -> ()
    ttg.local_dealloc %buf : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
