// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-prepare-consan-captures="target=nvidia" -tritoninstrument-concurrency-sanitizer | FileCheck %s --implicit-check-not=cluster_waiting --implicit-check-not=always_use_warp_shuffle
// RUN: env TRITON_CONSAN_INIT_ALLOCATIONS=0 triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-prepare-consan-captures="target=nvidia" -tritoninstrument-concurrency-sanitizer | FileCheck %s --check-prefix=NO-INIT

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-DAG: #[[BUFS_L:.*]] = #ttg.linear<{register = [], lane = {{\[}}[0], [0], [0], [0], [0]], warp = [], block = []}>
  // CHECK-DAG: #[[BUFS_THREADS_L:.*]] = #ttg.linear<{register = [], lane = {{\[}}[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]], warp = [], block = []}>
  // CHECK-DAG: #[[BUFS_BARS_L:.*]] = #ttg.linear<{register = [], lane = {{\[}}[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], warp = [], block = []}>
  // CHECK-LABEL: tt.func private @__triton_consan_verify_write_visibility_
  // CHECK: %[[WRITE_VISIBILITY:.*]] = tt.load
  // CHECK: arith.cmpi eq, %[[WRITE_VISIBILITY]],
  // CHECK: %[[SELECTED_THREAD_BIT:.*]] = arith.shli
  // CHECK: %[[VISIBLE_THREAD_BITS:.*]] = arith.andi %[[WRITE_VISIBILITY]], %[[SELECTED_THREAD_BIT]]
  // CHECK: arith.cmpi eq, %[[VISIBLE_THREAD_BITS]], %[[SELECTED_THREAD_BIT]]
  // CHECK: @single_local_alloc
  tt.func public @single_local_alloc() {
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i32

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i32

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#call_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [1, 0], CGALayout = [[1, 0]]}>
#call_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#call_smem = #ttg.shared_memory
#call_load_blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 1024 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  // Scratch in a non-entry function is summarized by the call's virtual
  // shared-memory frame. The callee body itself is not instrumented.
  // CHECK-LABEL: tt.func private @scratch_only_callee
  // CHECK-NOT: tt.call @__triton_consan
  // CHECK: tt.gather
  tt.func private @scratch_only_callee(
      %indices: tensor<1024x256xi32, #call_blocked>,
      %values: tensor<128x256xf32, #call_blocked>) {
    %0 = tt.gather %values[%indices] {axis = 0 : i32, allocation.offset = 0 : i32, allocation.size = 512 : i32}
        : (tensor<128x256xf32, #call_blocked>, tensor<1024x256xi32, #call_blocked>) -> tensor<1024x256xf32, #call_blocked>
    tt.return
  }

  // CHECK-LABEL: tt.func public @summarize_scratch_call
  // The explicit buffer and call frame partially overlap, so both masks select
  // the shared state lane.
  // CHECK: arith.constant dense<[true, true, false, false]> : tensor<4xi1
  // CHECK: ttg.local_load
  // CHECK: arith.constant dense<[false, true, true, false]> : tensor<4xi1
  // A valid private callee only touches the caller's CTA-local frame.
  // CHECK: %[[CALL_CTA_ID:.*]] = tti.experimental_cluster_cta_id
  // CHECK: %[[CALL_CTA:.*]] = arith.shli {{.*}}, %[[CALL_CTA_ID]] : i32
  // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[CALL_CTA]]
  // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}%[[CALL_CTA]]
  // CHECK: tt.call @__triton_consan_publish_write_visibility{{.*}}%[[CALL_CTA]]
  // CHECK: tt.call @scratch_only_callee
  tt.func public @summarize_scratch_call(
      %indices: tensor<1024x256xi32, #call_blocked>,
      %values: tensor<128x256xf32, #call_blocked>) {
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<256xi32, #call_shared, #call_smem, mutable>
    %loaded = ttg.local_load %buf : !ttg.memdesc<256xi32, #call_shared, #call_smem, mutable> -> tensor<256xi32, #call_load_blocked>
    tt.call @scratch_only_callee(%indices, %values) {allocation.offset = 128 : i32, allocation.size = 512 : i32}
        : (tensor<1024x256xi32, #call_blocked>, tensor<128x256xf32, #call_blocked>) -> ()
    // Keep the NVIDIA dialect loaded in this standalone split module.
    ttng.cluster_barrier {relaxed = true}
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1], CGALayout = [[1, 0]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: tti.experimental_cluster_cta_id : i32
  // Updating the active mask must not read or overwrite another CTA's entry.
  // CHECK-LABEL: tt.func private @__triton_consan_set_active_mask_
  // CHECK-SAME: (%[[ACTIVE_VALUE:[^:]+]]: i32,
  // CHECK-NOT: tt.load
  // CHECK: %[[ACTIVE_VALUES:.*]] = tt.splat %[[ACTIVE_VALUE]] : i32 -> tensor<2xi32
  // CHECK-NOT: tt.load
  // CHECK: %[[ACTIVE_CTA:.*]] = tti.experimental_cluster_cta_id : i32
  // CHECK-NEXT: %[[ACTIVE_CTAS:.*]] = tt.splat %[[ACTIVE_CTA]] : i32 -> tensor<2xi32
  // CHECK-NEXT: %[[ACTIVE_STORE_MASK:.*]] = arith.cmpi eq, {{.*}}, %[[ACTIVE_CTAS]] : tensor<2xi32
  // CHECK-NOT: tt.load
  // CHECK: tt.store %{{[^,]+}}, %[[ACTIVE_VALUES]], %[[ACTIVE_STORE_MASK]] {ignore_cta} : tensor<2x!tt.ptr<i32>
  // CHECK-NEXT: tt.return

  // Publishing and clearing overwrite selected entries without reading old
  // values. The buffer and CTA predicates remain masks on all four stores.
  // CHECK-LABEL: tt.func private @__triton_consan_publish_write_visibility_
  // CHECK-SAME: (%[[PUBLISH_BUFFERS:[^:]+]]: tensor<2xi1, {{.*}}>, %[[PUBLISH_PRED:[^:]+]]: i1, %{{[^:]+}}: i64, %[[PUBLISH_CTAS:[^:]+]]: i32,
  // CHECK-NOT: tt.load
  // CHECK: cf.cond_br %[[PUBLISH_PRED]],
  // CHECK-NOT: tt.load
  // CHECK: %[[PUBLISH_RESHAPED:.*]] = tt.reshape %[[PUBLISH_BUFFERS]] : {{.*}} -> tensor<1x2x1xi1
  // CHECK-NEXT: %[[PUBLISH_LAYOUT:.*]] = ttg.convert_layout %[[PUBLISH_RESHAPED]]
  // CHECK-NEXT: %[[PUBLISH_BUFFER_MASK:.*]] = tt.broadcast %[[PUBLISH_LAYOUT]] : {{.*}} -> tensor<2x2x2xi1
  // CHECK-NOT: tt.load
  // CHECK: %[[PUBLISH_CTA_BITS:.*]] = tt.splat %[[PUBLISH_CTAS]] : i32 -> tensor<2x2x2xi32
  // CHECK-NOT: tt.load
  // CHECK: %[[PUBLISH_OWNER_MASK:.*]] = arith.cmpi ne, {{.*}} : tensor<2x2x2xi32
  // CHECK-NOT: tt.load
  // CHECK: %[[PUBLISH_RELATION_MASK:.*]] = arith.andi %[[PUBLISH_OWNER_MASK]], {{.*}} : tensor<2x2x2xi1
  // CHECK-NEXT: %[[PUBLISH_STORE_MASK:.*]] = arith.andi %[[PUBLISH_BUFFER_MASK]], %[[PUBLISH_RELATION_MASK]] : tensor<2x2x2xi1
  // CHECK-NOT: tt.load
  // CHECK: tt.store %{{[^,]+}}, %{{[^,]+}}, %[[PUBLISH_STORE_MASK]] {ignore_cta} : tensor<2x2x2x!tt.ptr<i{{32|64}}>
  // CHECK-NOT: tt.load
  // CHECK: %[[CLEAR_WRITES_BUFFERS:.*]] = tt.broadcast {{.*}} -> tensor<2x2x2x1x2xi1
  // CHECK-NOT: tt.load
  // CHECK: tt.splat %[[PUBLISH_CTAS]] : i32 -> tensor<2x2x2x1x2xi32
  // CHECK-NOT: tt.load
  // CHECK: %[[CLEAR_WRITES_CTAS:.*]] = arith.cmpi ne, {{.*}} : tensor<2x2x2x1x2xi32
  // CHECK-NEXT: %[[CLEAR_WRITES_MASK:.*]] = arith.andi %[[CLEAR_WRITES_BUFFERS]], %[[CLEAR_WRITES_CTAS]] : tensor<2x2x2x1x2xi1
  // CHECK-NEXT: %[[CLEAR_WRITES_ZERO:.*]] = arith.constant dense<0> : tensor<2x2x2x1x2xi8
  // CHECK-NOT: tt.load
  // CHECK: tt.store %{{[^,]+}}, %[[CLEAR_WRITES_ZERO]], %[[CLEAR_WRITES_MASK]] {ignore_cta} : tensor<2x2x2x1x2x!tt.ptr<i8>
  // CHECK-NOT: tt.load
  // CHECK: %[[CLEAR_READS_BUFFERS:.*]] = tt.broadcast {{.*}} -> tensor<2x2x2x1x2xi1
  // CHECK-NOT: tt.load
  // CHECK: tt.splat %[[PUBLISH_CTAS]] : i32 -> tensor<2x2x2x1x2xi32
  // CHECK-NOT: tt.load
  // CHECK: %[[CLEAR_READS_CTAS:.*]] = arith.cmpi ne, {{.*}} : tensor<2x2x2x1x2xi32
  // CHECK-NEXT: %[[CLEAR_READS_MASK:.*]] = arith.andi %[[CLEAR_READS_BUFFERS]], %[[CLEAR_READS_CTAS]] : tensor<2x2x2x1x2xi1
  // CHECK-NEXT: %[[CLEAR_READS_ZERO:.*]] = arith.constant dense<0> : tensor<2x2x2x1x2xi{{32|64}}
  // CHECK-NOT: tt.load
  // CHECK: tt.store %{{[^,]+}}, %[[CLEAR_READS_ZERO]], %[[CLEAR_READS_MASK]] {ignore_cta} : tensor<2x2x2x1x2x!tt.ptr<i{{32|64}}>
  // CHECK-NOT: tt.load
  // CHECK: %[[CLEAR_TRACKING_BUFFERS:.*]] = tt.broadcast {{.*}} -> tensor<2x2x2x1x2x2xi1
  // CHECK-NOT: tt.load
  // CHECK: tt.splat %[[PUBLISH_CTAS]] : i32 -> tensor<2x2x2x1x2x2xi32
  // CHECK-NOT: tt.load
  // CHECK: %[[CLEAR_TRACKING_CTAS:.*]] = arith.cmpi ne, {{.*}} : tensor<2x2x2x1x2x2xi32
  // CHECK-NEXT: %[[CLEAR_TRACKING_MASK:.*]] = arith.andi %[[CLEAR_TRACKING_BUFFERS]], %[[CLEAR_TRACKING_CTAS]] : tensor<2x2x2x1x2x2xi1
  // CHECK-NEXT: %[[CLEAR_TRACKING_ZERO:.*]] = arith.constant dense<0> : tensor<2x2x2x1x2x2xi{{32|64}}
  // CHECK-NOT: tt.load
  // CHECK: tt.store %{{[^,]+}}, %[[CLEAR_TRACKING_ZERO]], %[[CLEAR_TRACKING_MASK]] {ignore_cta} : tensor<2x2x2x1x2x2x!tt.ptr<i{{32|64}}>
  // CHECK-NOT: tt.load
  // CHECK: tt.return

  // CHECK-LABEL: @single_local_alloc_multi_cta
  tt.func public @single_local_alloc_multi_cta() {
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i32
    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 64 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i32
    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8
    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 128 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i32
    // Matching register and shared ownership keeps an ordinary load local.
    // CHECK: ttng.init_barrier
    // CHECK: %[[LOCAL_CTAS:.*]] = arith.shli {{.*}} : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[LOCAL_CTAS]]{{.*}})
    // CHECK-NOT: publish_cluster_visibility
    // CHECK: tt.return
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 96 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // Candidate IDs are assigned while the six exact regions are sorted. The
  // state tensor is padded to eight lanes, but the highest-address region must
  // still select lane five.
  // CHECK-LABEL: @non_power_of_two_region_count
  // CHECK: arith.constant dense<[false, false, false, false, false, true, false, false]> : tensor<8xi1
  tt.func public @non_power_of_two_region_count() {
    %a = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4xi32, #shared, #smem, mutable>
    %b = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<4xi32, #shared, #smem, mutable>
    %c = ttg.local_alloc {allocation.offset = 32 : i32} : () -> !ttg.memdesc<4xi32, #shared, #smem, mutable>
    %d = ttg.local_alloc {allocation.offset = 48 : i32} : () -> !ttg.memdesc<4xi32, #shared, #smem, mutable>
    %e = ttg.local_alloc {allocation.offset = 64 : i32} : () -> !ttg.memdesc<4xi32, #shared, #smem, mutable>
    %f = ttg.local_alloc {allocation.offset = 80 : i32} : () -> !ttg.memdesc<4xi32, #shared, #smem, mutable>
    %0 = ttg.local_load %a : !ttg.memdesc<4xi32, #shared, #smem, mutable> -> tensor<4xi32>
    %1 = ttg.local_load %b : !ttg.memdesc<4xi32, #shared, #smem, mutable> -> tensor<4xi32>
    %2 = ttg.local_load %c : !ttg.memdesc<4xi32, #shared, #smem, mutable> -> tensor<4xi32>
    %3 = ttg.local_load %d : !ttg.memdesc<4xi32, #shared, #smem, mutable> -> tensor<4xi32>
    %4 = ttg.local_load %e : !ttg.memdesc<4xi32, #shared, #smem, mutable> -> tensor<4xi32>
    %5 = ttg.local_load %f : !ttg.memdesc<4xi32, #shared, #smem, mutable> -> tensor<4xi32>
    tt.return
  }
}

// -----

#barrier_fromCTA = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2], [4]]}>
#smem_fromCTA = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 8 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @mbarrier_fromCTA_basis_mask
  tt.func public @mbarrier_fromCTA_basis_mask() {
    %true = arith.constant true
    %bar = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8xi64, #barrier_fromCTA, #smem_fromCTA, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<8xi64, #barrier_fromCTA, #smem_fromCTA, mutable>

    // fromCTA=5 preserves CTA-ID bits 0 and 2, selects CTA0, CTA1, CTA4,
    // and CTA5 as issuers, and sends to CTA groups with matching fixed bits.
    // CHECK: ttng.init_barrier
    // CHECK: %[[EXPECT_CTA:.*]] = tti.experimental_cluster_cta_id : i32
    // CHECK-NEXT: %[[EXPECT_OMITTED:.*]] = arith.constant 2 : i32
    // CHECK-NEXT: %[[EXPECT_NON_ISSUER:.*]] = arith.andi %[[EXPECT_CTA]], %[[EXPECT_OMITTED]] : i32
    // CHECK-NEXT: %[[EXPECT_ZERO:.*]] = arith.constant 0 : i32
    // CHECK-NEXT: %[[EXPECT_ISSUER:.*]] = arith.cmpi eq, %[[EXPECT_NON_ISSUER]], %[[EXPECT_ZERO]] : i32
    // CHECK-NEXT: %[[EXPECT_PRED:.*]] = arith.andi %true, %[[EXPECT_ISSUER]] : i1
    // CHECK: arith.shli
    // CHECK: %[[EXPECT_RECIPIENT_CTA:.*]] = tti.experimental_cluster_cta_id : i32
    // CHECK: %[[EXPECT_FIXED:.*]] = arith.constant 5 : i32
    // CHECK: %[[EXPECT_BASE:.*]] = arith.andi %[[EXPECT_RECIPIENT_CTA]], %[[EXPECT_FIXED]] : i32
    // CHECK: %[[EXPECT_PATTERN:.*]] = arith.constant 5 : i32
    // CHECK: %[[EXPECT_RECIPIENTS:.*]] = arith.shli %[[EXPECT_PATTERN]], %[[EXPECT_BASE]] : i32
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state{{.*}}({{.*}}%[[EXPECT_PRED]]{{.*}}%[[EXPECT_RECIPIENTS]], {{.*}})
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK: ttng.barrier_expect
    ttng.barrier_expect %bar, 16 {fromCTA = 5 : i32}, %true : !ttg.memdesc<8xi64, #barrier_fromCTA, #smem_fromCTA, mutable>

    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK: ttng.arrive_barrier
    ttng.arrive_barrier %bar, 1, %true {fromCTA = 5 : i32} : !ttg.memdesc<8xi64, #barrier_fromCTA, #smem_fromCTA, mutable>
    tt.return
  }
}

// -----

#barrier_multicast_two = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem_multicast_two = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 16 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @mbarrier_multicast_two_ctas
  tt.func public @mbarrier_multicast_two_ctas() {
    %bar = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #barrier_multicast_two, #smem_multicast_two, mutable>
    // CHECK: ttng.init_barrier
    ttng.init_barrier %bar, 2 : !ttg.memdesc<2xi64, #barrier_multicast_two, #smem_multicast_two, mutable>
    // CHECK: %[[TWO_PATTERN:.*]] = arith.constant 3 : i32
    // CHECK: %[[TWO_SHIFT:.*]] = arith.shli %[[TWO_PATTERN]], {{.*}} : i32
    // CHECK: %[[TWO_RECIPIENTS:.*]] = arith.ori {{.*}}, %[[TWO_SHIFT]] : i32
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state{{.*}}({{.*}}%[[TWO_RECIPIENTS]], {{.*}})
    // CHECK: ttng.arrive_barrier {{.*}}multicastCTA = 1 : i32
    ttng.arrive_barrier %bar, 1 {multicastCTA = 1 : i32} : !ttg.memdesc<2xi64, #barrier_multicast_two, #smem_multicast_two, mutable>
    tt.return
  }
}

// -----

#barrier_multicast_four = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1], [2]]}>
#smem_multicast_four = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:107", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @mbarrier_multicast_four_ctas
  tt.func public @mbarrier_multicast_four_ctas() {
    %bar0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4xi64, #barrier_multicast_four, #smem_multicast_four, mutable>
    %bar1 = ttg.local_alloc {allocation.offset = 32 : i32} : () -> !ttg.memdesc<4xi64, #barrier_multicast_four, #smem_multicast_four, mutable>
    // CHECK: ttng.init_barrier
    ttng.init_barrier %bar0, 2 : !ttg.memdesc<4xi64, #barrier_multicast_four, #smem_multicast_four, mutable>
    // CHECK: ttng.init_barrier
    ttng.init_barrier %bar1, 2 : !ttg.memdesc<4xi64, #barrier_multicast_four, #smem_multicast_four, mutable>
    // multicastCTA=1 reaches {0,1} or {2,3}.
    // CHECK: %[[LOW_FIXED:.*]] = arith.constant 2 : i32
    // CHECK: %[[LOW_BASE:.*]] = arith.andi {{.*}}, %[[LOW_FIXED]] : i32
    // CHECK: %[[LOW_PATTERN:.*]] = arith.constant 3 : i32
    // CHECK: %[[LOW_SHIFT:.*]] = arith.shli %[[LOW_PATTERN]], %[[LOW_BASE]] : i32
    // CHECK: %[[LOW_RECIPIENTS:.*]] = arith.ori {{.*}}, %[[LOW_SHIFT]] : i32
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state{{.*}}({{.*}}%[[LOW_RECIPIENTS]], {{.*}})
    // CHECK: ttng.arrive_barrier {{.*}}multicastCTA = 1 : i32
    ttng.arrive_barrier %bar0, 1 {multicastCTA = 1 : i32} : !ttg.memdesc<4xi64, #barrier_multicast_four, #smem_multicast_four, mutable>
    // multicastCTA=2 reaches {0,2} or {1,3}.
    // CHECK: arith.constant 0 : i32
    // CHECK-NEXT: %[[HIGH_FIXED:.*]] = arith.constant 1 : i32
    // CHECK-NEXT: %[[HIGH_BASE:.*]] = arith.andi {{.*}}, %[[HIGH_FIXED]] : i32
    // CHECK: %[[HIGH_PATTERN:.*]] = arith.constant 5 : i32
    // CHECK: %[[HIGH_SHIFT:.*]] = arith.shli %[[HIGH_PATTERN]], %[[HIGH_BASE]] : i32
    // CHECK: %[[HIGH_RECIPIENTS:.*]] = arith.ori {{.*}}, %[[HIGH_SHIFT]] : i32
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state{{.*}}({{.*}}%[[HIGH_RECIPIENTS]], {{.*}})
    // CHECK: ttng.arrive_barrier {{.*}}multicastCTA = 2 : i32
    ttng.arrive_barrier %bar1, 1 {multicastCTA = 2 : i32} : !ttg.memdesc<4xi64, #barrier_multicast_four, #smem_multicast_four, mutable>
    tt.return
  }
}

// -----

#shared_cluster_ws = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#smem_cluster_ws = #ttg.shared_memory
#blocked_cluster_ws = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4104 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 6 : i32} {
  // CHECK-LABEL: @cluster_barrier_partition_scopes
  tt.func public @cluster_barrier_partition_scopes() {
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared_cluster_ws, #smem_cluster_ws, mutable>
    %value = ttg.local_load %buf : !ttg.memdesc<32x32xf32, #shared_cluster_ws, #smem_cluster_ws, mutable> -> tensor<32x32xf32, #blocked_cluster_ws>

    // A top-level barrier keeps the all-partition publisher.
    // CHECK: tt.call @__triton_consan_publish_cluster_visibility{{.*}}_I0
    ttng.cluster_barrier

    // CHECK: ttg.warp_specialize
    // CHECK-SAME: tti.disable_setmaxregister
    ttg.warp_specialize(%buf) attributes {actualRegisters = array<i32: 32, 32, 32>, allocation.offset = 4096 : i32, requestedRegisters = array<i32: 32, 32>, warpGroupStartIds = array<i32: 4, 5>}
    default {
      // The default region is thread 0, but its cluster barrier is still
      // partition-scoped.
      // CHECK: default
      // CHECK: tt.call @__triton_consan_publish_cluster_visibility{{.*}}_I1({{.*}}, %c0_i32_{{[0-9]+}}, %c1_i64_{{[0-9]+}},
      ttng.cluster_barrier
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<32x32xf32, #shared_cluster_ws, #smem_cluster_ws, mutable>) num_warps(1) {
      ttg.warp_return
    }
    partition1(%arg1: !ttg.memdesc<32x32xf32, #shared_cluster_ws, #smem_cluster_ws, mutable>) num_warps(1) {
      // Nested operations must retain partition1's thread identity.
      // CHECK: partition1
      // CHECK: scf.execute_region
      // CHECK: tt.call @__triton_consan_publish_cluster_visibility{{.*}}_I1({{.*}}, %c2_i32_{{[0-9]+}}, %c4_i64_{{[0-9]+}},
      scf.execute_region {
        ttng.cluster_barrier
        scf.yield
      }
      ttg.warp_return
    } : (!ttg.memdesc<32x32xf32, #shared_cluster_ws, #smem_cluster_ws, mutable>) -> ()
    tt.return
  }
}

// -----

#proxy_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#proxy_bar_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#proxy_smem = #ttg.shared_memory
#proxy_blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 4104 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: tt.func private @__triton_consan_verify_proxy_access_nw1
  // CHECK: arith.xori {{.*}} : tensor<{{.*}}xi64
  // CHECK-NEXT: %[[MISSING_PROXY_BITS:.*]] = arith.andi {{.*}} : tensor<{{.*}}xi64
  // CHECK-NEXT: %[[HAS_MISSING_PROXY_BITS:.*]] = arith.cmpi ne, %[[MISSING_PROXY_BITS]], {{.*}} : tensor<{{.*}}xi64
  // CHECK-NEXT: %[[MISSING_PROXY_PREDICATES:.*]] = tt.reshape %[[HAS_MISSING_PROXY_BITS]]{{.*}} -> tensor<{{.*}}xi1
  // CHECK: "tt.reduce"(%[[MISSING_PROXY_PREDICATES]])
  // CHECK: arith.ori {{.*}} : i1
  // CHECK-LABEL: @proxy_fence_state_transitions
  tt.func public @proxy_fence_state_transitions(%out: !tt.tensordesc<32x32xf32, #proxy_shared>) {
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #proxy_shared, #proxy_smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #proxy_bar_shared, #proxy_smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #proxy_bar_shared, #proxy_smem, mutable>

    // CHECK: tt.call @__triton_consan_set_proxy_access
    %value = ttg.local_load %buf : !ttg.memdesc<32x32xf32, #proxy_shared, #proxy_smem, mutable> -> tensor<32x32xf32, #proxy_blocked>
    // CHECK: tt.call @__triton_consan_track_proxy_accesses
    ttng.arrive_barrier %bar, 1, %true : !ttg.memdesc<1xi64, #proxy_bar_shared, #proxy_smem, mutable>
    // CHECK: ttng.wait_barrier {{.*}}, %[[PROXY_WAIT_PHASE:[^, ]+]],
    // CHECK: tt.call @__triton_consan_complete_barrier_wait{{.*}}(%{{[^,]+}}, %{{[^,]+}}, %[[PROXY_WAIT_PHASE]],
    ttng.wait_barrier %bar, %c0, %true : !ttg.memdesc<1xi64, #proxy_bar_shared, #proxy_smem, mutable>
    // CHECK: tt.call @__triton_consan_fence_proxy_accesses
    ttng.fence_async_shared {bCluster = false}
    // CHECK: tt.call @__triton_consan_verify_proxy_access
    ttng.async_tma_copy_local_to_global %out[%c0, %c0] %buf : !tt.tensordesc<32x32xf32, #proxy_shared>, !ttg.memdesc<32x32xf32, #proxy_shared, #proxy_smem, mutable>
    tt.return
  }
}

// -----

#proxy_cp_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#proxy_cp_smem = #ttg.shared_memory
#proxy_cp_blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#proxy_cp_mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @proxy_fence_tracks_cp_async_and_wgmma
  tt.func public @proxy_fence_tracks_cp_async_and_wgmma(%ptr: tensor<128x128x!tt.ptr<f16>, #proxy_cp_blocked>, %acc: tensor<128x128xf16, #proxy_cp_mma>) {
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #proxy_cp_shared, #proxy_cp_smem, mutable>
    // CHECK: tt.call @__triton_consan_set_proxy_access
    ttg.async_copy_global_to_local %ptr, %buf : tensor<128x128x!tt.ptr<f16>, #proxy_cp_blocked> -> <128x128xf16, #proxy_cp_shared, #proxy_cp_smem, mutable>
    ttg.async_commit_group
    ttg.async_wait {num = 0 : i32}
    // CHECK: tt.call @__triton_consan_verify_proxy_access
    // CHECK: tt.call @__triton_consan_verify_proxy_access
    ttng.warp_group_dot %buf, %buf, %acc : !ttg.memdesc<128x128xf16, #proxy_cp_shared, #proxy_cp_smem, mutable> * !ttg.memdesc<128x128xf16, #proxy_cp_shared, #proxy_cp_smem, mutable> -> tensor<128x128xf16, #proxy_cp_mma>
    tt.return
  }
}

// -----

#local_gather_scatter_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32, CGALayout = [[0, 1]]}>
#local_gather_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1]]}>
#local_gather_scatter_smem = #ttg.shared_memory
#local_gather_scatter_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 528 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_gather_scatter_effects
  tt.func public @local_gather_scatter_effects(%out: !tt.tensordesc<8x32xi32, #local_gather_scatter_shared>) {
    %c0 = arith.constant 0 : i32
    %indices = arith.constant dense<0> : tensor<8x32xi32, #local_gather_scatter_blocked>
    %values = arith.constant dense<1> : tensor<8x32xi32, #local_gather_scatter_blocked>
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #local_gather_shared, #local_gather_scatter_smem, mutable>
    %dst = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<8x32xi32, #local_gather_scatter_shared, #local_gather_scatter_smem, mutable>

    // Indexing the sharded axis can target either CTA row.
    // CHECK: %[[GATHER_CTAS:.*]] = arith.constant 3 : i32
    // CHECK: arith.constant dense<[true, true, false, false]> : tensor<4xi1
    // CHECK: tt.call @__triton_consan_set_proxy_access{{.*}}({{.*}}%[[GATHER_CTAS]]{{.*}})
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[GATHER_CTAS]]{{.*}})
    // CHECK: tt.call @__triton_consan_set_read_visibility{{.*}}({{.*}}%[[GATHER_CTAS]]{{.*}})
    // CHECK: ttg.local_gather
    %gathered = ttg.local_gather %src[%indices] {axis = 1 : i32} : !ttg.memdesc<8x32xi32, #local_gather_shared, #local_gather_scatter_smem, mutable>, tensor<8x32xi32, #local_gather_scatter_blocked> -> tensor<8x32xi32, #local_gather_scatter_blocked>

    // Indexing the unsharded axis leaves the scatter local to the issuing CTA.
    // CHECK: %[[SCATTER_CTAS:.*]] = arith.shli {{.*}} : i32
    // CHECK: arith.constant dense<[false, true, true, false]> : tensor<4xi1
    // CHECK: tt.call @__triton_consan_set_proxy_access{{.*}}({{.*}}%[[SCATTER_CTAS]]{{.*}})
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[SCATTER_CTAS]]{{.*}})
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}({{.*}}%[[SCATTER_CTAS]]{{.*}})
    // CHECK: tt.call @__triton_consan_publish_write_visibility{{.*}}({{.*}}%[[SCATTER_CTAS]]{{.*}})
    // CHECK: ttg.local_scatter
    ttg.local_scatter %dst[%indices], %values {axis = 0 : i32} : !ttg.memdesc<8x32xi32, #local_gather_scatter_shared, #local_gather_scatter_smem, mutable>, tensor<8x32xi32, #local_gather_scatter_blocked>, tensor<8x32xi32, #local_gather_scatter_blocked>

    // An async-proxy consumer makes the generic-proxy classification above
    // observable in ConSan's output.
    // CHECK: tt.call @__triton_consan_verify_proxy_access
    // CHECK: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %out[%c0, %c0] %dst : !tt.tensordesc<8x32xi32, #local_gather_scatter_shared>, !ttg.memdesc<8x32xi32, #local_gather_scatter_shared, #local_gather_scatter_smem, mutable>
    tt.return
  }
}

// -----

#local_atomic_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1]]}>
#local_atomic_tma_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 32, CGALayout = [[0, 1]]}>
#local_atomic_smem = #ttg.shared_memory
#local_atomic_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 4], order = [1, 0], CGALayout = [[0, 1]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 1536 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @local_atomic_scatter_rmw_effects
  tt.func public @local_atomic_scatter_rmw_effects(
      %out: !tt.tensordesc<8x32xi32, #local_atomic_tma_shared>,
      %result: tensor<8x32x!tt.ptr<i32>, #local_atomic_blocked>) {
    // The atomic is the allocation's only user, so its state mask proves
    // BufferRegion discovers the full destination.
    %c0 = arith.constant 0 : i32
    %indices = arith.constant dense<0> : tensor<8x32xi32, #local_atomic_blocked>
    %values = arith.constant dense<1> : tensor<8x32xi32, #local_atomic_blocked>
    // CHECK: %[[ATOMIC_DST:.*]] = ttg.local_alloc {allocation.offset = 0 : i32}
    %dst = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<8x32xi32, #local_atomic_shared, #local_atomic_smem, mutable>
    %proxy = ttg.local_alloc {allocation.offset = 512 : i32} : () -> !ttg.memdesc<8x32xi32, #local_atomic_tma_shared, #local_atomic_smem, mutable>

    // Runtime indices along the sharded axis can target either CTA row.
    // CHECK: %[[ATOMIC_CTAS:.*]] = arith.constant 3 : i32
    // CHECK: arith.constant dense<[true, false, false, false]> : tensor<4xi1
    // CHECK: tt.call @__triton_consan_set_proxy_access{{.*}}({{.*}}%[[ATOMIC_CTAS]])
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[ATOMIC_CTAS]]{{.*}})
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}({{.*}}%[[ATOMIC_CTAS]]{{.*}})
    // CHECK: tt.call @__triton_consan_publish_write_visibility{{.*}}({{.*}}%[[ATOMIC_CTAS]]{{.*}})
    // The existing destination reaches both CTAs, while independent result
    // staging remains private to the issuer's shared-memory frame.
    // CHECK: arith.constant dense<[false, false, true, false]> : tensor<4xi1
    // CHECK: %[[ATOMIC_SCRATCH_CTA_ID:.*]] = tti.experimental_cluster_cta_id
    // CHECK: %[[ATOMIC_SCRATCH_CTA:.*]] = arith.shli {{.*}}, %[[ATOMIC_SCRATCH_CTA_ID]] : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[ATOMIC_SCRATCH_CTA]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}%[[ATOMIC_SCRATCH_CTA]]
    // CHECK: tt.call @__triton_consan_publish_write_visibility{{.*}}%[[ATOMIC_SCRATCH_CTA]]
    // CHECK: ttg.local_atomic_scatter_rmw
    %old = ttg.local_atomic_scatter_rmw add, %dst[%indices], %values {allocation.offset = 1024 : i32, allocation.size = 512 : i32, axis = 1 : i32} : (!ttg.memdesc<8x32xi32, #local_atomic_shared, #local_atomic_smem, mutable>, tensor<8x32xi32, #local_atomic_blocked>, tensor<8x32xi32, #local_atomic_blocked>) -> tensor<8x32xi32, #local_atomic_blocked>
    tt.store %result, %old : tensor<8x32x!tt.ptr<i32>, #local_atomic_blocked>
    // Enable proxy tracking without giving the atomic destination another user.
    ttng.async_tma_copy_local_to_global %out[%c0, %c0] %proxy : !tt.tensordesc<8x32xi32, #local_atomic_tma_shared>, !ttg.memdesc<8x32xi32, #local_atomic_tma_shared, #local_atomic_smem, mutable>
    tt.return
  }
}

// -----

#local_access_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[0, 1]]}>
#local_access_smem = #ttg.shared_memory
#local_access_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @local_load_store_cross_cta_effects
  tt.func public @local_load_store_cross_cta_effects() {
    %values = arith.constant dense<1> : tensor<8x32xi32, #local_access_blocked>

    // A source-backed allocation lowers through the local-store path.
    // CHECK: ttg.local_alloc %{{.*}}
    // CHECK: %[[ALLOC_CTAS:.*]] = arith.constant 3 : i32
    // CHECK: tt.call @__triton_consan_publish_write_visibility{{.*}}({{.*}}%[[ALLOC_CTAS]]{{.*}})
    %buf = ttg.local_alloc %values {allocation.offset = 0 : i32} : (tensor<8x32xi32, #local_access_blocked>) -> !ttg.memdesc<8x32xi32, #local_access_shared, #local_access_smem, mutable>

    // The register and shared layouts shard different logical axes, so every
    // issuer reads from both CTA rows.
    // CHECK: %[[LOAD_CTAS:.*]] = arith.constant 3 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[LOAD_CTAS]]{{.*}})
    // CHECK: ttg.local_load
    %loaded = ttg.local_load %buf : !ttg.memdesc<8x32xi32, #local_access_shared, #local_access_smem, mutable> -> tensor<8x32xi32, #local_access_blocked>

    // The corresponding store writes both CTA rows.
    // CHECK: %[[STORE_CTAS:.*]] = arith.constant 3 : i32
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}({{.*}}%[[STORE_CTAS]]{{.*}})
    // CHECK: ttg.local_store
    ttg.local_store %loaded, %buf : tensor<8x32xi32, #local_access_blocked> -> !ttg.memdesc<8x32xi32, #local_access_shared, #local_access_smem, mutable>
    // Keep the target dialect loaded in this standalone split module.
    ttng.cluster_barrier {relaxed = true}
    tt.return
  }
}

// -----

#frontier_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#frontier_barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#frontier_smem = #ttg.shared_memory
#frontier_src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8200 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // The helper consumes the analysis-derived completion mask directly.
  // CHECK-LABEL: tt.func private @__triton_consan_track_proxy_accesses_for_buffer
  // CHECK-SAME: %arg8: i32, %arg9: tensor<8xi1{{.*}}, %arg10: i32
  // CHECK: ttg.convert_layout %arg9 {force_warp_shuffle}
  // CHECK-LABEL: @tma_completion_tracks_contained_proxy_frontier
  tt.func public @tma_completion_tracks_contained_proxy_frontier(
      %desc: !tt.tensordesc<1024xi32, #frontier_shared>) {
    // The first explicit region is contained in the TMA destination. The third
    // region only partially overlaps it, so only its overlapping atom may be
    // published by TMA completion. Its remainder and the fourth, disjoint
    // region must remain unpublished.
    %c0 = arith.constant 0 : i32
    %true = arith.constant true
    %value = arith.constant dense<0> : tensor<128xi32, #frontier_src>
    %contained = ttg.local_alloc %value {allocation.offset = 0 : i32}
        : (tensor<128xi32, #frontier_src>) -> !ttg.memdesc<128xi32, #frontier_shared, #frontier_smem, mutable>
    %partial = ttg.local_alloc %value {allocation.offset = 3840 : i32}
        : (tensor<128xi32, #frontier_src>) -> !ttg.memdesc<128xi32, #frontier_shared, #frontier_smem, mutable>
    %dst = ttg.local_alloc {allocation.offset = 0 : i32}
        : () -> !ttg.memdesc<1024xi32, #frontier_shared, #frontier_smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32}
        : () -> !ttg.memdesc<1xi64, #frontier_barrier, #frontier_smem, mutable>
    ttng.init_barrier %bar, 1
        : !ttg.memdesc<1xi64, #frontier_barrier, #frontier_smem, mutable>
    ttng.barrier_expect %bar, 4096, %true
        : !ttg.memdesc<1xi64, #frontier_barrier, #frontier_smem, mutable>
    ttng.fence_async_shared {bCluster = false}

    // This access occurs after the fence and is outside the TMA destination.
    %unrelated = ttg.local_alloc %value {allocation.offset = 4608 : i32}
        : (tensor<128xi32, #frontier_src>) -> !ttg.memdesc<128xi32, #frontier_shared, #frontier_smem, mutable>

    // CHECK: arith.constant dense<[true, true, true, false, false, false, false, false]> : tensor<8xi1
    // CHECK: tt.call @__triton_consan_track_barrier_write_for_buffer
    // CHECK: tt.call @__triton_consan_track_proxy_accesses_for_buffer
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0] %dst, %bar, %true
        : !tt.tensordesc<1024xi32, #frontier_shared>,
          !ttg.memdesc<1xi64, #frontier_barrier, #frontier_smem, mutable>
          -> !ttg.memdesc<1024xi32, #frontier_shared, #frontier_smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @two_local_alloc
  tt.func public @two_local_alloc() {
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i32

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i32

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    ttg.local_load %1 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @three_local_alloc
  tt.func public @three_local_alloc() {
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i32

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i32

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    ttg.local_load %1 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    ttg.local_load %2 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @three_sub_bufs
  tt.func public @three_sub_bufs() {
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i32

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i32

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_load %1 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: @read_bars_alloc
  tt.func public @read_bars_alloc() {
    // CHECK: %[[READ_BARS_G:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_BARS_G]], %c0_i8
    %c0 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<4x1xi64, #shared1, #smem, mutable>
    %bar_sub = ttg.memdesc_index %bar[%c0] : !ttg.memdesc<4x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar_sub, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %buf_sub = ttg.memdesc_index %0[%c0] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    ttg.local_load %buf_sub : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[BUFS_L:.*]] = #ttg.linear<{register = [], lane = {{\[}}[0], [0], [0], [0], [0]], warp = [], block = []}>
  // CHECK: @tmem_alloc
  tt.func public @tmem_alloc() {
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_descriptors [4096], [{{.*}}], shared_mem : tensor<1xi64, #{{.*}}>
    %0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.tmem_load %0 : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: tt.func private @__triton_consan_verify_and_update_barrier_state
  // CHECK: %[[BARRIER_INIT_BITS:.*]] = arith.extui {{.*}} : tensor<{{.*}}xi1{{.*}}> to tensor<{{.*}}xi32
  // CHECK: %[[BARRIER_VALID_BITS:.*]] = arith.extui {{.*}} : tensor<{{.*}}xi1{{.*}}> to tensor<{{.*}}xi32
  // CHECK: %[[BARRIER_SHIFTED_VALID_BITS:.*]] = arith.shli %[[BARRIER_VALID_BITS]]
  // CHECK: %[[BARRIER_STATUS_BITS:.*]] = arith.ori %[[BARRIER_INIT_BITS]], %[[BARRIER_SHIFTED_VALID_BITS]]
  // CHECK: %[[BARRIER_PACKED_STATUS:.*]] = "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  // CHECK: arith.andi {{.*}} : i32
  // CHECK: "tt.reduce"
  // CHECK: tt.return %[[BARRIER_PACKED_STATUS]] : i32
  // CHECK-LABEL: @async_tma_copy_global_to_local
  tt.func public @async_tma_copy_global_to_local(%arg0: !tt.tensordesc<32x32xf32, #shared>) {
    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i32

    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i32

    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1xi64
    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_set_proxy_access
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // Model the async TMA completion mechanism: barrier_expect corresponds to
    // mbarrier.arrive.expect_tx and is what should update ConSan's barrier state.
    ttng.barrier_expect %bar, 4096, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    // CHECK: tt.call @__triton_consan_track_visible_accesses
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_publish_write_visibility
    // CHECK: tt.call @__triton_consan_set_read_visibility
    // CHECK: tt.call @__triton_consan_track_barrier_write_for_buffer
    // CHECK: tt.call @__triton_consan_track_barrier_read_for_buffer
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %arg0[%c0_i32, %c0_i32] %0, %bar, %true : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // Invalidation is another generic-proxy write. It must observe the
    // synthetic engine's deferred completion use of the barrier bytes.
    // CHECK: tt.call @__triton_consan_set_proxy_access
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_invalidate_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier invalidated while a thread is waiting"
    // CHECK: ttng.inval_barrier
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared_clc = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // Read the issuing observer's [buffer CTA, buffer, reader CTA] view, not
  // every observer. Reader CTAs retain stride 16 from the 2x2x2x2x2 table.
  // CHECK-LABEL: tt.func private @__triton_consan_track_visible_accesses_
  // CHECK-SAME: (%{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i1, %[[OBSERVER_THREAD:[^:]+]]: i32,
  // CHECK-SAME: !tt.ptr<i8>{{[^,]*}}, %[[OBSERVER_VISIBILITY:[^:]+]]: !tt.ptr<i32>
  // CHECK: tt.store {{.*}} : tensor<{{.*}}x!tt.ptr<i8>
  // CHECK-NOT: tt.load
  // CHECK: %[[OBSERVER_CTA:.*]] = tti.experimental_cluster_cta_id : i32
  // CHECK-NEXT: %[[OBSERVER_CTA_STRIDE:.*]] = arith.constant 4 : i32
  // CHECK-NEXT: %[[OBSERVER_CTA_OFFSET:.*]] = arith.muli %[[OBSERVER_CTA]], %[[OBSERVER_CTA_STRIDE]] : i32
  // CHECK-NEXT: %[[OBSERVER_THREAD_STRIDE:.*]] = arith.constant 8 : i32
  // CHECK-NEXT: %[[OBSERVER_THREAD_OFFSET:.*]] = arith.muli %[[OBSERVER_THREAD]], %[[OBSERVER_THREAD_STRIDE]] : i32
  // CHECK-NEXT: %[[OBSERVER_OFFSET:.*]] = arith.addi %[[OBSERVER_CTA_OFFSET]], %[[OBSERVER_THREAD_OFFSET]] : i32
  // CHECK-NEXT: %[[OBSERVER_PTR:.*]] = tt.addptr %[[OBSERVER_VISIBILITY]], %[[OBSERVER_OFFSET]] : !tt.ptr<i32>, i32
  // CHECK-NEXT: %[[OBSERVER_PTRS:.*]] = tt.splat %[[OBSERVER_PTR]] : !tt.ptr<i32> -> tensor<2x2x2x!tt.ptr<i32>
  // CHECK-NOT: tt.load
  // CHECK: %[[OBSERVER_OWNER_PTRS:.*]] = tt.addptr %[[OBSERVER_PTRS]], {{.*}} : tensor<2x2x2x!tt.ptr<i32>
  // CHECK-NOT: tt.load
  // CHECK: %[[OBSERVER_BUFFER_PTRS:.*]] = tt.addptr %[[OBSERVER_OWNER_PTRS]], {{.*}} : tensor<2x2x2x!tt.ptr<i32>
  // CHECK-NEXT: %[[READER_CTAS:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32}
  // CHECK-NEXT: %[[READER_CTA_STRIDE:.*]] = arith.constant dense<16> : tensor<2xi32
  // CHECK-NEXT: %[[READER_CTA_OFFSETS:.*]] = arith.muli %[[READER_CTAS]], %[[READER_CTA_STRIDE]]
  // CHECK-NEXT: %[[READER_CTA_RESHAPED:.*]] = tt.reshape %[[READER_CTA_OFFSETS]] : {{.*}} -> tensor<1x1x2xi32
  // CHECK-NEXT: %[[READER_CTA_LAYOUT:.*]] = ttg.convert_layout %[[READER_CTA_RESHAPED]]
  // CHECK-NEXT: %[[READER_CTA_BROADCAST:.*]] = tt.broadcast %[[READER_CTA_LAYOUT]] : {{.*}} -> tensor<2x2x2xi32
  // CHECK-NEXT: %[[OBSERVER_READER_PTRS:.*]] = tt.addptr %[[OBSERVER_BUFFER_PTRS]], %[[READER_CTA_BROADCAST]] : tensor<2x2x2x!tt.ptr<i32>
  // CHECK-NEXT: tt.load %[[OBSERVER_READER_PTRS]] : tensor<2x2x2x!tt.ptr<i32>

  // CHECK-LABEL: @clc_try_cancel_diagonal_effect_recipients
  tt.func public @clc_try_cancel_diagonal_effect_recipients() {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: tti.experimental_cluster_cta_id
    // CHECK: arith.cmpi eq
    // CHECK: %[[CLC_THREAD:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[CLC_THREAD]]
    // CHECK: %[[CLC_MASK:.*]] = arith.constant 2 : i64
    // CHECK: tt.call @__triton_consan_publish_write_visibility{{.*}}%[[CLC_MASK]]
    // CHECK: tt.call @__triton_consan_track_barrier_write_for_buffer{{.*}}_I1
    ttng.clc_try_cancel %result, %bar : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable>, !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.barrier_expect %bar, 16, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: ttng.clc_load_result
    %clc_res = ttng.clc_load_result %result : !ttg.memdesc<2xi64, #shared_clc, #smem, mutable> -> i128
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1], CGALayout = [[0, 0]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: tt.func private @__triton_consan_check_outstanding_commits{{.*}}T2x2x1xI8
  // CHECK-SAME: %arg4: i32
  // CHECK-NOT: tti.experimental_cluster_cta_id
  // CHECK: tt.splat %arg4 : i32 -> tensor<2x2x1xi32
  // CHECK: arith.shrui
  // CHECK-LABEL: @outstanding_commits_multicast_tma_recipients
  tt.func public @outstanding_commits_multicast_tma_recipients(
      %desc: !tt.tensordesc<32x32xf32, #shared>,
      %ptr: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %bar, 4096, %true : !ttg.memdesc<2xi64, #shared1, #smem, mutable>
    %cta = tti.experimental_cluster_cta_id : i32
    %non_issuer_cta = arith.cmpi eq, %cta, %c1_i32 : i32
    %mask = tt.splat %non_issuer_cta : i1 -> tensor<32x32xi1, #blocked>
    ttg.async_copy_global_to_local %ptr, %shmem mask %mask : tensor<32x32x!tt.ptr<f32>, #blocked> -> <32x32xf32, #shared, #smem, mutable>
    ttg.async_commit_group
    // CHECK: tt.call @__triton_consan_stage_access_for_commit
    // CHECK: ttg.async_copy_global_to_local
    // CHECK: tt.call @__triton_consan_commit_accesses
    // CHECK: ttg.async_commit_group
    // CHECK: %[[PATTERN:.*]] = arith.constant 3 : i32
    // CHECK: %[[RECIPIENTS:.*]] = arith.shli %[[PATTERN]],
    // CHECK: tt.call @__triton_consan_check_outstanding_commits{{.*}}({{.*}}, %[[RECIPIENTS]])
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %shmem, %bar, %true {multicast} : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<2xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[0]]}>
#smem = #ttg.shared_memory
#offset_parent = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#offsets = #ttg.slice<{dim = 0, parent = #offset_parent}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, "ttng.two-ctas" = true, ttg.shared = 65544 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @multicast_gather_two_cta_tx_count
  tt.func public @multicast_gather_two_cta_tx_count(%desc: !tt.tensordesc<1x32xf32, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %x_offsets = arith.constant dense<0> : tensor<32xi32, #offsets>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: scf.for
    scf.for %i = %c0 to %c2 step %c1 {
      // CHECK: arith.constant 4096 : i64
      // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
      // CHECK: ttng.barrier_expect
      ttng.barrier_expect %bar, 4096, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      // CHECK: arith.constant -8192 : i64
      // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
      // CHECK: ttng.async_tma_gather
      ttng.async_tma_gather %desc[%x_offsets, %c0_i32] %result, %bar, %true {multicast} : !tt.tensordesc<1x32xf32, #shared>, tensor<32xi32, #offsets>, i32, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, i1
    }
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_tma_copy_global_to_local_two_bufs_one_barrier
  tt.func public @async_tma_copy_global_to_local_two_bufs_one_barrier(
      %a: !tt.tensordesc<32x32xf32, #shared>,
      %b: !tt.tensordesc<32x32xf32, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32

    %a_smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %b_smem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // Two TMA copies contribute to a single expected transaction.
    ttng.barrier_expect %bar, 8192, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    // CHECK: tt.call @__triton_consan_init_barrier_state
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    // CHECK: ttng.barrier_expect
    // CHECK-COUNT-2: tt.call @__triton_consan_track_barrier_write_for_buffer
    ttng.async_tma_copy_global_to_local %a[%c0_i32, %c0_i32] %a_smem, %bar, %true : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_global_to_local %b[%c0_i32, %c0_i32] %b_smem, %bar, %true : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>

    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    // Consume results to prevent DCE / to keep realistic ordering.
    %va = ttg.local_load %a_smem : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    %vb = ttg.local_load %b_smem : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    %_ = arith.addf %va, %vb : tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65552 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_tma_copy_global_to_local_two_bufs_two_barriers
  // CHECK: %[[A_SMEM:.*]] = ttg.local_alloc {allocation.offset = 0 : i32}
  // CHECK: %[[B_SMEM:.*]] = ttg.local_alloc {allocation.offset = 4096 : i32}
  // CHECK: %[[BAR0:.*]] = ttg.local_alloc {allocation.offset = 65536 : i32}
  // CHECK: %[[BAR1:.*]] = ttg.local_alloc {allocation.offset = 65544 : i32}
  // CHECK: ttng.barrier_expect %[[BAR0]], 4096, %true
  // CHECK: tt.call @__triton_consan_track_barrier_write_for_buffer{{.*}}({{[^,]+}}, {{[^,]+}}, %true, %[[A_TRACK:.*]], {{[^,]+}},
  // CHECK: ttng.async_tma_copy_global_to_local %arg0
  // CHECK: ttng.barrier_expect %[[BAR1]], 4096, %true
  // CHECK-NOT: tt.call @__triton_consan_track_barrier_write_for_buffer{{.*}}({{[^,]+}}, {{[^,]+}}, %true, %[[A_TRACK]], {{[^,]+}},
  // CHECK: tt.call @__triton_consan_track_barrier_write_for_buffer{{.*}}({{[^,]+}}, {{[^,]+}}, %true, %[[B_TRACK:.*]], {{[^,]+}},
  // CHECK: ttng.async_tma_copy_global_to_local %arg1
  tt.func public @async_tma_copy_global_to_local_two_bufs_two_barriers(
      %a: !tt.tensordesc<32x32xf32, #shared>,
      %b: !tt.tensordesc<32x32xf32, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %a_smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %b_smem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar0 = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %bar1 = ttg.local_alloc {allocation.offset = 65544 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.barrier_expect %bar0, 4096, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %a[%c0_i32, %c0_i32] %a_smem, %bar0, %true : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    ttng.barrier_expect %bar1, 4096, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.async_tma_copy_global_to_local %b[%c0_i32, %c0_i32] %b_smem, %bar1, %true : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    ttng.wait_barrier %bar1, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %va = ttg.local_load %a_smem : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    %vb = ttg.local_load %b_smem : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    %_ = arith.addf %va, %vb : tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_tma_copy_local_to_global
  tt.func public @async_tma_copy_local_to_global(%arg0: !tt.tensordesc<32x32xf32, #shared>, %ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %acc: tensor<128x128xf16, #mma>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>

    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_stage_access_for_commit
    // CHECK: tt.call @__triton_consan_commit_accesses
    ttng.async_tma_copy_local_to_global %arg0[%c0_i32, %c0_i32] %0 : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_tma_store_wait
  tt.func public @async_tma_store_wait(%arg0: !tt.tensordesc<32x32xf32, #shared>, %ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %acc: tensor<128x128xf16, #mma>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>

    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_reads
    ttng.async_tma_store_wait {pendings = 0 : i32}

    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_tma_gather
  tt.func public @async_tma_gather(%arg0: !tt.tensordesc<1x32xf32, #shared>, %ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %acc: tensor<128x128xf16, #mma>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %x_offsets = arith.constant dense<1> : tensor<32xi32>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>
    ttng.warp_group_dot %shmem, %shmem, %acc : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    // CHECK: ttng.warp_group_dot

    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_publish_write_visibility
    // CHECK: tt.call @__triton_consan_track_barrier_write_for_buffer
    ttng.async_tma_gather %arg0[%x_offsets, %c0_i32] %0, %bar, %true : !tt.tensordesc<1x32xf32, #shared>, tensor<32xi32>, i32, !ttg.memdesc<1xi64, #shared1, #smem, mutable>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, i1
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_tma_scatter
  tt.func public @async_tma_scatter(%arg0: !tt.tensordesc<1x32xf32, #shared>, %ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %acc: tensor<128x128xf16, #mma>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %x_offsets = arith.constant dense<1> : tensor<32xi32>
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>
    ttng.warp_group_dot %shmem, %shmem, %acc : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    // CHECK: ttng.warp_group_dot

	    // CHECK: tt.call @__triton_consan_verify_write_visibility
	    // CHECK: tt.call @__triton_consan_check_outstanding_commits
	    // CHECK: tt.call @__triton_consan_stage_access_for_commit
	    // CHECK: tt.call @__triton_consan_commit_accesses
	    ttng.async_tma_scatter %arg0[%x_offsets, %c0_i32] %0 : !tt.tensordesc<1x32xf32, #shared>, tensor<32xi32>, i32, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
	    tt.return
	  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_tma_reduce
  tt.func public @async_tma_reduce(%arg0: !tt.tensordesc<32x32xf32, #shared>, %ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %acc: tensor<128x128xf16, #mma>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>

    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_stage_access_for_commit
    // CHECK: tt.call @__triton_consan_commit_accesses
    ttng.async_tma_reduce add, %arg0[%c0_i32, %c0_i32] %0 : !tt.tensordesc<32x32xf32, #shared>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @barrier_reinit_requires_invalidate
  tt.func public @barrier_reinit_requires_invalidate() {
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32xi32, #shared1, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    %tmp = ttg.local_load %buf : !ttg.memdesc<32xi32, #shared1, #smem, mutable> -> tensor<32xi32>
    // CHECK: tt.call @__triton_consan_invalidate_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier invalidated while a thread is waiting"
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_clear_barrier_write_tracking
    // CHECK: tt.call @__triton_consan_clear_barrier_read_tracking
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @wait_barrier
  tt.func public @wait_barrier(%arg0: !tt.tensordesc<32x32xf32, #shared>) {
    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i32

    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i32

    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1xi64, #linear{{[0-9]*}}>

    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i32
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK-DAG: tt.call @__triton_consan_set_waiting
    // CHECK-DAG: tt.call @__triton_consan_check_all_active_waiting
    // CHECK: ttng.wait_barrier {{.*}}, %[[WAIT_PHASE:[^, ]+]],
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_transfer_visible_accesses{{.*}}(%{{[^,]+}}, %{{[^,]+}}, %[[WAIT_PHASE]], {{.*}}%[[BARRIERS]], %[[WRITE_VISIBILITY_GLOB]], %[[WRITE_TRACKING_GLOB]], %[[READ_VISIBILITY_GLOB]], %[[READ_TRACKING_GLOB]]) : {{.*}}!tt.ptr<i32>, !tt.ptr<i8>, !tt.ptr<i32>, !tt.ptr<i32>) -> ()
    // CHECK: tt.call @__triton_consan_clear_waiting
    // CHECK: tti.experimental_lock_release
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @arrive_barrier
  tt.func public @arrive_barrier(%arg0: !tt.tensordesc<32x32xf32, #shared>) {
    // CHECK-DAG: %[[BSTATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[BSTATE_GLOB]], %c0_i64
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_track_visible_accesses
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    // CHECK: tti.experimental_lock_release
    ttng.arrive_barrier %bar, 2, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @wait_barrier_without_init
  tt.func public @wait_barrier_without_init() {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK: tt.call @__triton_consan_set_waiting
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @arrive_barrier_without_init
  tt.func public @arrive_barrier_without_init() {
    %true = arith.constant true
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    ttng.arrive_barrier %bar, 1, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen5_mma
  tt.func public @tcgen5_mma(%arg0: !tt.tensordesc<32x32xf32, #shared>) {
    // CHECK-DAG: %[[SM_WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: tt.call @[[FILL_FOUR_I32:__triton_consan_fill_global_tensor[^ (]*T4xI32]](%[[SM_WRITE_VISIBILITY_GLOB]],
    // CHECK-DAG: %[[SM_READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: %[[TM_WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: tt.call @[[FILL_TWO_I32:__triton_consan_fill_global_tensor[^ (]*T2xI32]]
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1xi64

    // CHECK-DAG: %[[SM_WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK-DAG: %[[SM_READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: %[[TM_WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 2 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK-DAG: %[[TM_READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>

    // CHECK: ttng.init_barrier
    // CHECK: arith.shli
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[SM_WRITE_VISIBILITY_GLOB]]
    // CHECK: %[[TC_MASK:.*]] = arith.constant 2 : i64
    // CHECK: %[[TC_OBSERVER_MASK:.*]] = arith.constant 2 : i64
    // CHECK: tt.call @__triton_consan_set_read_visibility{{.*}}%[[TC_MASK]], %[[TC_OBSERVER_MASK]], %[[SM_READ_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[SM_WRITE_VISIBILITY_GLOB]]
    // CHECK: %[[TC_MASK:.*]] = arith.constant 2 : i64
    // CHECK: %[[TC_OBSERVER_MASK:.*]] = arith.constant 2 : i64
    // CHECK: tt.call @__triton_consan_set_read_visibility{{.*}}%[[TC_MASK]], %[[TC_OBSERVER_MASK]], %[[SM_READ_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[TM_WRITE_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}T1x1x1x2x1xI32
    // CHECK: %[[TC_MASK:.*]] = arith.constant 2 : i64
    // CHECK: tt.call @__triton_consan_publish_write_visibility
    // CHECK: %[[BAR_I64:.*]] = tti.experimental_memdesc_to_i32 %[[BAR:.*]] :
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_track_visible_accesses{{.*}}%[[BAR_I64]]{{.*}}%[[TC_BIT]]{{.*}}%[[BARRIERS]]{{.*}}%[[SM_WRITE_VISIBILITY_GLOB]]{{.*}}%[[SM_WRITE_TRACKING_GLOB]]{{.*}}%[[SM_READ_VISIBILITY_GLOB]], %{{[^,)]+}}) : {{.*}}!tt.ptr<i8>, !tt.ptr<i32>, !tt.ptr<i32>) -> ()
    // CHECK: %[[BAR_I64:.*]] = tti.experimental_memdesc_to_i32 %[[BAR]] :
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_track_visible_accesses{{.*}}%[[BAR_I64]]{{.*}}%[[TC_BIT]]{{.*}}%[[BARRIERS]]{{.*}}%[[TM_WRITE_VISIBILITY_GLOB]]{{.*}}%[[TM_WRITE_TRACKING_GLOB]]{{.*}}%[[TM_READ_VISIBILITY_GLOB:.*]], %{{[^,)]+}}) : {{.*}}!tt.ptr<i8>, !tt.ptr<i32>, !tt.ptr<i32>) -> ()
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK: ttng.tc_gen5_mma
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result = ttng.tmem_alloc  {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %true = arith.constant true
    ttng.tc_gen5_mma %0, %1, %result[], %true, %true, %bar[%true] {is_async} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen5_mma_lhs_in_tmem
  tt.func public @tcgen5_mma_lhs_in_tmem(%arg0: !tt.tensordesc<32x32xf32, #shared>) {
    // CHECK-DAG: %[[SM_WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: %[[SM_READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: %[[TM_READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1xi64

    // CHECK-DAG: %[[SM_WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK-DAG: %[[SM_READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: %[[TM_WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK-DAG: %[[TM_READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>

    // CHECK: ttng.init_barrier
    // CHECK: arith.shli
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{[^ (]*}}({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, %[[TM_WRITE_VISIBILITY_GLOB:[^,]+]],
    // CHECK: %[[TC_MASK:.*]] = arith.constant 2 : i64
    // CHECK: %[[TC_OBSERVER_MASK:.*]] = arith.constant 2 : i64
    // CHECK: tt.call @__triton_consan_set_read_visibility{{.*}}%[[TC_MASK]], %[[TC_OBSERVER_MASK]], %[[TM_READ_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[SM_WRITE_VISIBILITY_GLOB]]
    // CHECK: %[[TC_MASK:.*]] = arith.constant 2 : i64
    // CHECK: %[[TC_OBSERVER_MASK:.*]] = arith.constant 2 : i64
    // CHECK: tt.call @__triton_consan_set_read_visibility{{.*}}%[[TC_MASK]], %[[TC_OBSERVER_MASK]], %[[SM_READ_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[TM_WRITE_VISIBILITY_GLOB]]
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}%[[TM_READ_VISIBILITY_GLOB]]
    // CHECK: %[[TC_MASK:.*]] = arith.constant 2 : i64
    // CHECK: tt.call @__triton_consan_publish_write_visibility
    // CHECK: %[[BAR_I64:.*]] = tti.experimental_memdesc_to_i32 %[[BAR:.*]] :
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_track_visible_accesses{{.*}}%[[BAR_I64]]{{.*}}%[[TC_BIT]]{{.*}}%[[BARRIERS]]{{.*}}%[[SM_WRITE_VISIBILITY_GLOB]]{{.*}}%[[SM_WRITE_TRACKING_GLOB]]{{.*}}%[[SM_READ_VISIBILITY_GLOB]], %{{[^,)]+}}) : {{.*}}!tt.ptr<i8>, !tt.ptr<i32>, !tt.ptr<i32>) -> ()
    // CHECK: %[[BAR_I64:.*]] = tti.experimental_memdesc_to_i32 %[[BAR]] :
    // CHECK: %[[TC_BIT:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_track_visible_accesses{{.*}}%[[BAR_I64]]{{.*}}%[[TC_BIT]]{{.*}}%[[BARRIERS]]{{.*}}%[[TM_WRITE_VISIBILITY_GLOB]]{{.*}}%[[TM_WRITE_TRACKING_GLOB]]{{.*}}%[[TM_READ_VISIBILITY_GLOB]], %{{[^,)]+}}) : {{.*}}!tt.ptr<i8>, !tt.ptr<i32>, !tt.ptr<i32>) -> ()
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    // CHECK: tti.experimental_lock_release
    // CHECK: ttng.tc_gen5_mma
    %c0_i32 = arith.constant 0 : i32
    %0 = ttng.tmem_alloc  {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %result = ttng.tmem_alloc  {tensor_memory_col_offset = 128 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %true = arith.constant true
    ttng.tc_gen5_mma %0, %1, %result[], %true, %true, %bar[%true] {is_async} : !ttg.memdesc<128x128xf16, #tmem1, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tcgen5_commit
  tt.func public @tcgen5_commit(%arg0: !tt.tensordesc<32x32xf32, #shared>) {

    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %result = ttng.tmem_alloc  {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    %true = arith.constant true
    // CHECK-COUNT-2: tt.call @__triton_consan_track_visible_accesses
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    ttng.tc_gen5_commit %bar : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_load %0 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
    ttng.tmem_load %result : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf16>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[0, 0]]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1, CGALayout = [[0, 0]]>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32, "ttng.two-ctas" = true} {
  // CHECK-LABEL: @tmem_copy_2cta
  tt.func public @tmem_copy_2cta() {
    %src = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %dst = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: arith.constant 3 : i32
    // CHECK: ttng.tmem_copy
    ttng.tmem_copy %src, %dst : !ttg.memdesc<128x128xf32, #shared, #smem, mutable>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_copy_global_to_local
  tt.func public @async_copy_global_to_local(%ptr: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    // CHECK: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRT_COMMITS_GLOB]], %c0_i8

    // CHECK: tt.call @__triton_consan_verify_write_visibility_nw1
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_check_outstanding_commits{{.*}}%[[THREAD_BIT]], %[[WRT_COMMITS_GLOB]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility_nw1
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_stage_access_for_commit_nw1{{.*}}%[[THREAD_BIT]], %[[WRT_COMMITS_GLOB]]
    // CHECK: ttg.async_copy_global_to_local

    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_copy_global_to_local_with_barriers
  tt.func public @async_copy_global_to_local_with_barriers(%ptr: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>

    // CHECK-DAG: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 2 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>

    // CHECK: tt.call @__triton_consan_init_barrier_state

    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_check_outstanding_commits{{.*}}%[[THREAD_BIT]], %[[WRT_COMMITS_GLOB]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}({{[^,]+}}
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_stage_access_for_commit{{.*}}%[[THREAD_BIT]], %[[WRT_COMMITS_GLOB]]
    // CHECK: ttg.async_copy_global_to_local
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: tt.func private @__triton_consan_check_all_active_waiting
  // CHECK: %[[DEADLOCK_WAITING_BITS:.*]] = arith.extui {{.*}} : tensor<{{.*}}xi1{{.*}}> to tensor<{{.*}}xi32
  // CHECK: %[[DEADLOCK_IDLE_BITS:.*]] = arith.extui {{.*}} : tensor<{{.*}}xi1{{.*}}> to tensor<{{.*}}xi32
  // CHECK: %[[DEADLOCK_SHIFTED_IDLE_BITS:.*]] = arith.shli %[[DEADLOCK_IDLE_BITS]]
  // CHECK: %[[DEADLOCK_BITS:.*]] = arith.ori %[[DEADLOCK_WAITING_BITS]], %[[DEADLOCK_SHIFTED_IDLE_BITS]]
  // CHECK: %[[DEADLOCK_STATUS:.*]] = "tt.reduce"(%[[DEADLOCK_BITS]]) <{axis = 0 : i32}>
  // CHECK: arith.andi {{.*}} : i32
  // CHECK-NOT: "tt.reduce"
  // CHECK: arith.cmpi eq, %[[DEADLOCK_STATUS]], {{.*}} : i32
  // CHECK-LABEL: @wait_barrier_multi_cta
  tt.func public @wait_barrier_multi_cta() {
    // The dummy descriptor is the virtual cluster-barrier slot. It uses the
    // ordinary barrier state, waiting, and active-mask captures.
    // CHECK: tti.experimental_buffer_descriptors [65536, 0], [8, 0], shared_mem
    // CHECK: ttg.global_scratch_alloc
    // CHECK-COUNT-5: ttg.global_scratch_alloc
    // CHECK-NOT: ttg.global_scratch_alloc
    // CHECK: tti.experimental_lock_release
    // CHECK-NEXT: ttng.cluster_barrier
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: ttng.init_barrier
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: %[[WAIT_PRED:.*]] = arith.andi %true, %{{.*}} : i1
    // CHECK-NEXT: tti.experimental_lock_acquire %{{.*}}, %[[WAIT_PRED]]
    // CHECK: tt.call @__triton_consan_check_all_active_waiting
    // CHECK-NEXT: tti.experimental_lock_release %{{.*}}, %[[WAIT_PRED]]
    // CHECK-NEXT: tti.experimental_assert_uniform
    // CHECK: ttng.wait_barrier
    // CHECK-NEXT: tti.experimental_lock_acquire %{{.*}}, %[[WAIT_PRED]]
    // CHECK: tt.call @__triton_consan_clear_waiting
    // CHECK-NEXT: tti.experimental_lock_release %{{.*}}, %[[WAIT_PRED]]
    ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: ttg.barrier global_read
    // CHECK: tt.call @__triton_consan_check_all_active_waiting
    // CHECK-NEXT: tti.experimental_lock_release
    // CHECK-NEXT: tti.experimental_assert_uniform {{.*}}, "Deadlock detected at a cluster barrier"
    // CHECK-NEXT: cf.br ^[[POLL:bb[0-9]+]]
    // CHECK: ^[[POLL]]:
    // CHECK: tti.experimental_lock_acquire
    // CHECK: ttg.barrier global_read
    // CHECK: tt.call @__triton_consan_check_all_active_waiting
    // CHECK-NEXT: tti.experimental_lock_release
    // CHECK-NEXT: tti.experimental_assert_uniform {{.*}}, "Deadlock detected at a cluster barrier"
    // CHECK: cf.cond_br {{.*}}, ^[[CONTINUE:bb[0-9]+]], ^[[POLL]]
    // CHECK: ^[[CONTINUE]]:
    // CHECK-NEXT: ttng.cluster_barrier
    // CHECK-NOT: ttng.cluster_barrier
    // CHECK: tt.return
    ttng.cluster_barrier
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_commit_group
  tt.func public @async_commit_group() {
    // CHECK: tt.call @__triton_consan_commit_accesses
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.local_load %shmem : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_commit_group
  tt.func public @async_commit_group() {
    // CHECK: tti.experimental_lock_acquire
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 1 : i64
    // CHECK: %[[OUTSTANDING_NUM:.*]] = arith.constant 42 : i32
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_writes{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]], %[[OUTSTANDING_NUM]]
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_wait {num = 42 : i32}
    ttg.local_load %shmem : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_load
  tt.func public @tmem_load() {
    %result = ttng.tmem_alloc  {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    ttng.tmem_load %result : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf16>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @warp_group_dot
  tt.func public @warp_group_dot(%acc: tensor<128x128xf16, #mma>) {
    // CHECK-DAG: %[[SM_WGMMA_WRITES_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 2 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[SM_WGMMA_WRITES_GLOB]], %c0_i8

    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_stage_access_for_commit{{.*}}%[[THREAD_BIT]], %[[SM_WGMMA_WRITES_GLOB]]
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_stage_access_for_commit{{.*}}%[[THREAD_BIT]], %[[SM_WGMMA_WRITES_GLOB]]
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_commit_accesses{{.*}}(%[[THREAD_BIT]], {{.*}}, %[[SM_WGMMA_WRITES_GLOB]]
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %true = arith.constant true
    ttng.warp_group_dot %0, %1, %acc, %true {isAsync = true} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @warp_group_dot_sync
  tt.func public @warp_group_dot_sync(%acc: tensor<128x128xf16, #mma>) {
    // CHECK-DAG: %[[SM_WGMMA_WRITES_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 2 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[SM_WGMMA_WRITES_GLOB]], %c0_i8

    // CHECK: "before_dot"
    // CHECK-NOT: tt.call @__triton_consan_stage_access_for_commit
    // CHECK-NOT: tt.call @__triton_consan_commit_accesses
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 32768 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %true = arith.constant true
    "before_dot"() : () -> ()
    ttng.warp_group_dot %0, %1, %acc, %true {isAsync = false} : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @warp_group_dot_wait
  tt.func public @warp_group_dot_wait(%acc: tensor<128x128xf16, #mma>) {
    // Dummy buffer just to make the pass run
    %dummy = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_reads
    ttng.warp_group_dot_wait %acc { pendings = 42 : i32 } : tensor<128x128xf16, #mma>
    ttg.local_load %dummy : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @local_alloc_with_src
  tt.func public @local_alloc_with_src(%acc: tensor<128x128xf16, #mma>) {
    // CHECK: %[[BUF:.*]] = ttg.local_alloc
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{[^,]+}}
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}({{[^,]+}}
    %buf = ttg.local_alloc %acc {allocation.offset = 0 : i32} : (tensor<128x128xf16, #mma>) -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tmem_alloc_with_src
  tt.func public @tmem_alloc_with_src(%acc: tensor<128x128xf16, #blocked>) {
    // CHECK: %[[BUF:.*]] = ttng.tmem_alloc
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{[^,]+}}
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}({{[^,]+}}
    %buf = ttng.tmem_alloc %acc { tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32 } : (tensor<128x128xf16, #blocked>) -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @local_load_barriers
  tt.func public @local_load_barriers() {
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: arith.constant dense<[true, false]> : tensor<2xi1
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_set_read_visibility
    // CHECK: tti.experimental_lock_release
    ttg.local_load %buf : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @local_load_barriers
  tt.func public @local_load_barriers_cp_async(%ptr: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>

    // CHECK: ttg.async_copy_global_to_local

    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_set_read_visibility
    // CHECK: tti.experimental_lock_release
    // CHECK: ttg.local_load
    ttg.local_load %buf : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 32, 16]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @local_store_barriers_cp_async_wgmma
  tt.func public @local_store_barriers_cp_async_wgmma(%ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %acc: tensor<128x128xf16, #mma>) {
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>
    ttng.warp_group_dot %shmem, %shmem, %acc : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> * !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #mma>
    // CHECK: ttng.warp_group_dot

    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_publish_write_visibility
    // CHECK: tti.experimental_lock_release
    // CHECK: ttg.local_store
    ttg.local_store %acc, %buf : tensor<128x128xf16, #mma> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_allocation
  tt.func public @ws_allocation(%arg0: !tt.tensordesc<32x32xf32, #shared>) {
    // CHECK-DAG: tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1xi64,
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_set_active_mask
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 2 : i64
    // CHECK: tt.call @__triton_consan_copy_write_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_copy_read_visibility{{.*}}(%[[THREAD_BIT]]
    ttg.warp_specialize(%smem, %bar) attributes {actualRegisters = array<i32: 480, 32>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    default {
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_verify_write_visibility
      // CHECK: tt.call @__triton_consan_set_read_visibility
      // CHECK: tti.experimental_lock_release
      ttg.local_load %smem : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
      ttg.warp_yield
    }
    partition0(%arg1: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      // CHECK: partition0
      // CHECK-DAG: tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1xi64,
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_verify_write_visibility
      // CHECK: tt.call @__triton_consan_set_read_visibility
      // CHECK: tti.experimental_lock_release
      ttg.local_load %arg1 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
      ttg.warp_return
    } : (!ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_publish_cta_visibility
    // CHECK: tti.experimental_lock_release
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_buf_ptrs_default
  tt.func public @ws_buf_ptrs_default(%arg0: !tt.tensordesc<32x32xf32, #shared>) {
    // CHECK-DAG: tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_set_active_mask
    // CHECK: tti.experimental_lock_release
    ttg.warp_specialize(%smem, %bar) attributes {actualRegisters = array<i32: 480, 32>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    default {
      %c0_i32 = arith.constant 0 : i32
      %1 = ttg.memdesc_index %smem[%c0_i32] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_load %1 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
      ttg.warp_yield
    }
    partition0(%arg1: !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      ttg.warp_return
    } : (!ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    tt.return
  }
}

// -----


#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_buf_ptrs_partition0
  tt.func public @ws_buf_ptrs_partition0(%arg0: !tt.tensordesc<32x32xf32, #shared>) {
    // CHECK-DAG: tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_set_active_mask
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 2 : i64
    // CHECK: tt.call @__triton_consan_copy_write_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_copy_read_visibility{{.*}}(%[[THREAD_BIT]]
    ttg.warp_specialize(%smem, %bar) attributes {actualRegisters = array<i32: 480, 32>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%arg1: !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      %c0_i32 = arith.constant 0 : i32
      %1 = ttg.memdesc_index %arg1[%c0_i32] : !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      ttg.local_load %1 : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16>
      ttg.warp_return
    } : (!ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 2>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_wait_barrier
  tt.func public @ws_wait_barrier() {
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: %[[ACTIVE_MASK:.*]] = arith.constant 5 : i32
    // CHECK: tt.call @__triton_consan_set_active_mask{{.*}}(%[[ACTIVE_MASK]],
    ttg.warp_specialize(%smem, %bar) attributes {actualRegisters = array<i32: 480, 32>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    default {
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_set_waiting
      // CHECK: tt.call @__triton_consan_check_all_active_waiting
      // CHECK: tti.experimental_lock_release
      %c0_i32 = arith.constant 0 : i32
      %true = arith.constant true
      ttng.wait_barrier %bar, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg1: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      // CHECK: partition0
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_set_waiting
      // CHECK: tt.call @__triton_consan_check_all_active_waiting
      // CHECK: tti.experimental_lock_release
      %c0_i32 = arith.constant 0 : i32
      %true = arith.constant true
      ttng.wait_barrier %arg2, %c0_i32, %true : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    tt.return
  }
}

// -----


#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @alias_matrix_shared
  tt.func public @alias_matrix_shared() {
    // CHECK-DAG: arith.constant dense<[true, true, false, false]> : tensor<4xi1
    // CHECK-DAG: arith.constant dense<[false, true, true, false]> : tensor<4xi1
    %buf0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %buf1 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.local_load %buf0 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    ttg.local_load %buf1 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @dynamic_shared_index
  tt.func public @dynamic_shared_index(%idx: i32) {
    // CHECK: %[[RUNTIME_BASE:.*]] = tti.experimental_memdesc_to_i32
    // CHECK-DAG: %[[PAGE0:.*]] = tti.experimental_memory_offset_to_i32 0, shared_mem
    // CHECK-DAG: %[[PAGE1:.*]] = tti.experimental_memory_offset_to_i32 128, shared_mem
    // CHECK-DAG: arith.cmpi eq, %[[RUNTIME_BASE]], %[[PAGE0]]
    // CHECK-DAG: arith.cmpi eq, %[[RUNTIME_BASE]], %[[PAGE1]]
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x32xf32, #shared, #smem, mutable>
    %buf = ttg.memdesc_index %smem[%idx] : !ttg.memdesc<2x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %0 = ttg.local_load %buf : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [16, 16]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 4096 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @padded_dynamic_nested_subslice
  tt.func public @padded_dynamic_nested_subslice(%idx: i32, %cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %multi = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable>
    %dynamic = ttg.memdesc_index %multi[%idx] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    %page0 = ttg.memdesc_index %multi[%c0] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    %page1 = ttg.memdesc_index %multi[%c1] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    %dynamic_row = ttg.memdesc_subslice %dynamic [8, 0] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x16xf32, #shared, #smem, mutable, 16x16>
    %dynamic_nested = ttg.memdesc_subslice %dynamic_row [0, 8] : !ttg.memdesc<8x16xf32, #shared, #smem, mutable, 16x16> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %row0 = ttg.memdesc_subslice %page0 [8, 0] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x16xf32, #shared, #smem, mutable, 16x16>
    %nested0 = ttg.memdesc_subslice %row0 [0, 8] : !ttg.memdesc<8x16xf32, #shared, #smem, mutable, 16x16> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %row1 = ttg.memdesc_subslice %page1 [8, 0] : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<8x16xf32, #shared, #smem, mutable, 16x16>
    %nested1 = ttg.memdesc_subslice %row1 [0, 8] : !ttg.memdesc<8x16xf32, #shared, #smem, mutable, 16x16> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %complement1 = ttg.memdesc_subslice %row1 [0, 0] : !ttg.memdesc<8x16xf32, #shared, #smem, mutable, 16x16> -> !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>
    %selected = arith.select %cond, %dynamic_nested, %nested1 : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16>

    // The full pages contain their slices, and complementary page-one slices
    // remain disjoint in the exact five-lane plan.
    // CHECK-DAG: arith.constant dense<[true, true, false, false, false, false, false, false]> : tensor<8xi1
    // CHECK-DAG: arith.constant dense<[false, false, true, true, true, false, false, false]> : tensor<8xi1
    // CHECK-DAG: arith.constant dense<[false, true, false, false, false, false, false, false]> : tensor<8xi1
    // CHECK-DAG: arith.constant dense<[false, false, false, false, true, false, false, false]> : tensor<8xi1
    // CHECK-DAG: arith.constant dense<[false, false, false, true, false, false, false, false]> : tensor<8xi1
    %0 = ttg.local_load %page0 : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> tensor<16x16xf32>
    %1 = ttg.local_load %page1 : !ttg.memdesc<16x16xf32, #shared, #smem, mutable> -> tensor<16x16xf32>
    %2 = ttg.local_load %nested0 : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %3 = ttg.local_load %nested1 : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    %4 = ttg.local_load %complement1 : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>

    // The padded page stride is 1152 bytes, and the nested affine offset of
    // 544 bytes maps to the physical runtime candidates 608 and 1760.
    // CHECK: %[[DYNAMIC_BASE:.*]] = tti.experimental_memdesc_to_i32
    // CHECK-DAG: %[[DYNAMIC_0:.*]] = tti.experimental_memory_offset_to_i32 608, shared_mem
    // CHECK-DAG: %[[DYNAMIC_1:.*]] = tti.experimental_memory_offset_to_i32 1760, shared_mem
    // CHECK-DAG: arith.cmpi eq, %[[DYNAMIC_BASE]], %[[DYNAMIC_0]]
    // CHECK-DAG: arith.cmpi eq, %[[DYNAMIC_BASE]], %[[DYNAMIC_1]]
    // CHECK: tti.experimental_assert_uniform {{.*}}, "internal ConSan error: active memdesc resolved to no buffer state"
    %5 = ttg.local_load %dynamic_nested : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>

    // CHECK: %[[SELECT_BASE:.*]] = tti.experimental_memdesc_to_i32
    // CHECK-DAG: %[[SELECT_0:.*]] = tti.experimental_memory_offset_to_i32 608, shared_mem
    // CHECK-DAG: %[[SELECT_1:.*]] = tti.experimental_memory_offset_to_i32 1760, shared_mem
    // CHECK-DAG: arith.cmpi eq, %[[SELECT_BASE]], %[[SELECT_0]]
    // CHECK-DAG: arith.cmpi eq, %[[SELECT_BASE]], %[[SELECT_1]]
    // CHECK: tti.experimental_assert_uniform {{.*}}, "internal ConSan error: active memdesc resolved to no buffer state"
    %6 = ttg.local_load %selected : !ttg.memdesc<8x8xf32, #shared, #smem, mutable, 16x16> -> tensor<8x8xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @alias_matrix_shared_indexed
  tt.func public @alias_matrix_shared_indexed() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK-DAG: arith.constant dense<[true, false, false, false]> : tensor<4xi1
    // CHECK-DAG: arith.constant dense<[false, true, false, false]> : tensor<4xi1
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x32xf32, #shared, #smem, mutable>
    %buf0 = ttg.memdesc_index %smem[%c0_i32] : !ttg.memdesc<2x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %buf1 = ttg.memdesc_index %smem[%c1_i32] : !ttg.memdesc<2x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.local_load %buf0 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    ttg.local_load %buf1 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @alias_matrix_shared_subslice
  tt.func public @alias_matrix_shared_subslice() {
    // CHECK-DAG: arith.constant dense<[true, true, false, false]> : tensor<4xi1
    // CHECK-DAG: arith.constant dense<[false, true, false, false]> : tensor<4xi1
    %buf0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %buf1 = ttg.memdesc_subslice %buf0 [32] : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable, 64>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.local_load %buf0 : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32>
    ttg.local_load %buf1 : !ttg.memdesc<32xf32, #shared, #smem, mutable, 64> -> tensor<32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @alias_matrix_tensor
  tt.func public @alias_matrix_tensor() {
    // CHECK-DAG: arith.constant dense<[true, true, false, false]> : tensor<4xi1
    // CHECK-DAG: arith.constant dense<[false, false, true, false]> : tensor<4xi1
    // CHECK-DAG: arith.constant dense<[false, true, false, false]> : tensor<4xi1
    %buf0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %buf1 = ttng.tmem_alloc {tensor_memory_col_offset = 64 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    %buf3 = ttng.tmem_subslice %buf0 {offset = 32 : i32} : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable, 64x64>
    ttng.tmem_load %buf0 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32>
    ttng.tmem_load %buf1 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32>
    ttng.tmem_load %buf3 : !ttg.memdesc<64x32xf32, #tmem, #ttng.tensor_memory, mutable, 64x64> -> tensor<64x32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 64, blockN = 64, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @alias_matrix_mixed
  tt.func public @alias_matrix_mixed() {
    %smem0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %tmem0 = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // The tensor-memory read has one atom; each overlapping shared-memory
    // read spans two atoms and must keep the full state-mask path.
    // CHECK: arith.constant dense<true> : tensor<1xi1
    // CHECK: tt.call @__triton_consan_set_read_visibility_nw1_I32_
    // CHECK: ttng.tmem_load
    ttng.tmem_load %tmem0 : !ttg.memdesc<64x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<64x64xf32>
    // CHECK: arith.constant dense<[true, true, false, false]> : tensor<4xi1
    // CHECK: tt.call @__triton_consan_set_read_visibility_nw1_T4xI1_
    // CHECK: ttg.local_load
    ttg.local_load %smem0 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    // CHECK: arith.constant dense<[false, true, true, false]> : tensor<4xi1
    // CHECK: tt.call @__triton_consan_set_read_visibility_nw1_T4xI1_
    // CHECK: ttg.local_load
    ttg.local_load %smem1 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @ws_alias_matrix
  tt.func public @ws_alias_matrix() {
    // We expect exact overlapping-atom masks in the default region and
    // partition0 after lowering warp_specialize.
    // CHECK-DAG: arith.constant dense<[true, true, false, false]> : tensor<4xi1
    // CHECK-DAG: arith.constant dense<[false, true, true, false]> : tensor<4xi1
    %smem0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

    ttg.warp_specialize(%smem0, %smem1, %bar) attributes {actualRegisters = array<i32: 32, 32>, allocation.offset = 0 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 0>}
    default {
      %c0 = arith.constant 0 : i32
      ttg.local_load %smem0 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
      ttg.local_load %smem1 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<32xf32, #shared, #smem, mutable>, %arg1: !ttg.memdesc<32xf32, #shared, #smem, mutable>, %arg2: !ttg.memdesc<1xi64, #shared, #smem, mutable>) num_warps(1) {
      // CHECK: arith.constant dense<[true, true, false, false]> : tensor<4xi1
      // CHECK: arith.constant dense<[false, true, true, false]> : tensor<4xi1
      %c0 = arith.constant 0 : i32
      ttg.local_load %arg0 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
      ttg.local_load %arg1 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
      ttg.warp_return
    } : (!ttg.memdesc<32xf32, #shared, #smem, mutable>, !ttg.memdesc<32xf32, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @initialize_uninitialized_allocs
  // NO-INIT-LABEL: @initialize_uninitialized_allocs
  tt.func public @initialize_uninitialized_allocs() {
    // CHECK: %[[SMEM:.*]] = ttg.local_alloc
    // CHECK: ttg.barrier local
    // CHECK: %[[SMEM_POISON:.*]] = arith.constant dense<0x7FC00000> : tensor<128x128xf32
    // CHECK: ttg.local_store %[[SMEM_POISON]], %[[SMEM]]
    // CHECK: ttg.barrier local
    // CHECK: %[[TMEM:.*]] = ttng.tmem_alloc
    // CHECK: ttg.barrier tensor_read|tensor_write
    // CHECK: %[[TMEM_POISON:.*]] = arith.constant dense<0x7FC00000> : tensor<128x128xf32
    // CHECK: %[[TRUE:.*]] = arith.constant true
    // CHECK: ttng.tmem_store %[[TMEM_POISON]], %[[TMEM]], %[[TRUE]]
    // CHECK: ttg.barrier tensor_read|tensor_write
    // NO-INIT-NOT: ttg.local_store
    // NO-INIT-NOT: ttng.tmem_store
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %tmem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // NO-INIT: tt.return
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 256 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @initialize_multibuffer_allocs
  tt.func public @initialize_multibuffer_allocs() {
    // CHECK: %[[SMEM:.*]] = ttg.local_alloc
    // CHECK: %[[SMEM_0:.*]] = ttg.memdesc_index %[[SMEM]]
    // CHECK: %[[SMEM_1:.*]] = ttg.memdesc_index %[[SMEM]]
    // CHECK: ttg.barrier local
    // CHECK: ttg.local_store {{.*}}, %[[SMEM_0]]
    // CHECK: ttg.local_store {{.*}}, %[[SMEM_1]]
    // CHECK: ttg.barrier local
    // CHECK: %[[TMEM:.*]] = ttng.tmem_alloc
    // CHECK: %[[TMEM_0:.*]] = ttg.memdesc_index %[[TMEM]]
    // CHECK: %[[TMEM_1:.*]] = ttg.memdesc_index %[[TMEM]]
    // CHECK: ttg.barrier tensor_read|tensor_write
    // CHECK: ttng.tmem_store {{.*}}, %[[TMEM_0]],
    // CHECK: ttng.tmem_store {{.*}}, %[[TMEM_1]],
    // CHECK: ttg.barrier tensor_read|tensor_write
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #shared, #smem, mutable>
    %tmem = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @skip_source_backed_allocs
  tt.func public @skip_source_backed_allocs(%src: tensor<128x128xf32, #blocked>) {
    // CHECK: ttg.local_alloc %{{.*}}
    // CHECK-NOT: ttg.local_store
    // CHECK: ttng.tmem_alloc %{{.*}}
    // CHECK-NOT: ttng.tmem_store
    %smem = ttg.local_alloc %src {allocation.offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %tmem = ttng.tmem_alloc %src {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

#shared_cluster_publish = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#smem_cluster_publish = #ttg.shared_memory
#blocked_cluster_publish = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1], CGALayout = [[1, 0]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // All logical threads in this module are synchronous. The cluster
  // publication must still propagate both visibility frontiers, but it need
  // not filter either frontier against nonexistent asynchronous threads.
  // CHECK-LABEL: tt.func private @__triton_consan_publish_cluster_visibility{{.*}}_I0
  // CHECK-NOT: arith.cmpi ne
  // CHECK: tt.return
  // CHECK-LABEL: @cluster_barrier_publish_protocol
  tt.func public @cluster_barrier_publish_protocol() {
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared_cluster_publish, #smem_cluster_publish, mutable>
    // CHECK: ttg.local_load
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_publish_cluster_visibility
    // CHECK: tti.experimental_lock_release
    // CHECK: ttng.cluster_barrier
    // CHECK-NOT: ttng.cluster_barrier
    // CHECK: tt.return
    ttg.local_load %buf : !ttg.memdesc<32x32xf32, #shared_cluster_publish, #smem_cluster_publish, mutable> -> tensor<32x32xf32, #blocked_cluster_publish>
    ttng.cluster_barrier
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @shared_shrinking_reinterpret
  tt.func public @shared_shrinking_reinterpret() {
    // CHECK-DAG: arith.constant dense<[true, false]> : tensor<2xi1
    // CHECK-DAG: arith.constant dense<[false, true]> : tensor<2xi1
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %smaller = ttg.memdesc_reinterpret %parent : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> !ttg.memdesc<8xi32, #shared, #smem, mutable>
    %upper = ttg.memdesc_subslice %parent [8] : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> !ttg.memdesc<8xi32, #shared, #smem, mutable, 16>
    ttg.local_load %smaller : !ttg.memdesc<8xi32, #shared, #smem, mutable> -> tensor<8xi32>
    ttg.local_load %upper : !ttg.memdesc<8xi32, #shared, #smem, mutable, 16> -> tensor<8xi32>
    tt.return
  }
}

// -----

#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 0 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 128 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @tensor_shrinking_reinterpret
  tt.func public @tensor_shrinking_reinterpret() {
    // CHECK-DAG: arith.constant dense<[true, false]> : tensor<2xi1
    // CHECK-DAG: arith.constant dense<[false, true]> : tensor<2xi1
    %parent = ttng.tmem_alloc {tensor_memory_col_offset = 0 : i32, tensor_memory_row_offset = 0 : i32} : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %smaller = ttg.memdesc_reinterpret %parent : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %upper = ttng.tmem_subslice %parent {offset = 64 : i32, dim = 1 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128>
    ttng.tmem_load %smaller : !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf16>
    ttng.tmem_load %upper : !ttg.memdesc<128x64xf32, #tmem, #ttng.tensor_memory, mutable, 128x128> -> tensor<128x64xf32>
    tt.return
  }
}

// -----

// Cross-CTA subviews must direct every local memory access to the CTA that
// owns its physical bytes, including dynamically selected subviews whose
// physical offsets are equal.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  // CHECK-LABEL: @cross_cta_affine_subslice_recipients
  // CHECK-SAME: (%[[CHOOSE:.*]]: i1)
  tt.func public @cross_cta_affine_subslice_recipients(%choose: i1) {
    // CHECK: %[[AFFINE_PARENT:.*]] = ttg.local_alloc
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x32xi32, #shared, #smem, mutable>
    // CHECK: %[[LOCAL_VIEW:.*]] = ttg.memdesc_subslice %[[AFFINE_PARENT]][0, 0]
    %local = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<4x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>
    // CHECK: %[[REMOTE_VIEW:.*]] = ttg.memdesc_subslice %[[AFFINE_PARENT]][2, 0]
    %remote = ttg.memdesc_subslice %parent [2, 0] : !ttg.memdesc<4x32xi32, #shared, #smem, mutable> -> !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>
    // CHECK: arith.select {{.*}}, %[[LOCAL_VIEW]], %[[REMOTE_VIEW]]
    %selected = arith.select %choose, %local, %remote : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>
    %indices = arith.constant dense<0> : tensor<2x32xi32, #blocked>
    %values = arith.constant dense<1> : tensor<2x32xi32, #blocked>
    // CHECK: ttng.cluster_barrier
    ttng.cluster_barrier
    // CHECK: %[[LOCAL_CTAS:.*]] = arith.constant 1 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[LOCAL_CTAS]])
    // CHECK: ttg.local_load
    %l = ttg.local_load %local : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32> -> tensor<2x32xi32, #blocked>
    // CHECK: %[[REMOTE_CTAS:.*]] = arith.constant 2 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[REMOTE_CTAS]])
    // CHECK: ttg.local_load
    %r = ttg.local_load %remote : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32> -> tensor<2x32xi32, #blocked>
    // CHECK: %[[SELECTED_LOCAL_CTA_BIT:.*]] = arith.constant 1 : i32
    // CHECK: %[[LOCAL_SELECTED:.*]] = arith.andi {{.*}}, %[[CHOOSE]] : i1
    // CHECK: %[[REMOTE_CHOICE:.*]] = arith.xori %[[CHOOSE]], {{.*}} : i1
    // CHECK: %[[REMOTE_SELECTED:.*]] = arith.andi {{.*}}, %[[REMOTE_CHOICE]] : i1
    // CHECK: %[[LOCAL_CTA:.*]] = arith.select %[[LOCAL_SELECTED]], %[[SELECTED_LOCAL_CTA_BIT]], {{.*}} : i32
    // CHECK: %[[SELECTED_REMOTE_CTA_BIT:.*]] = arith.constant 2 : i32
    // CHECK: %[[REMOTE_CTA:.*]] = arith.select %[[REMOTE_SELECTED]], %[[SELECTED_REMOTE_CTA_BIT]], {{.*}} : i32
    // CHECK: %[[SELECTED_CTAS:.*]] = arith.ori %[[LOCAL_CTA]], %[[REMOTE_CTA]] : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[SELECTED_CTAS]])
    // Same-base candidates with different CTA masks share one physical atom.
    // CHECK: tt.call @__triton_consan_set_read_visibility_nw4_I32_
    // CHECK: ttg.local_load
    %s = ttg.local_load %selected : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32> -> tensor<2x32xi32, #blocked>
    // CHECK: %[[STORE_CTAS:.*]] = arith.constant 2 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[STORE_CTAS]])
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}({{.*}}%[[STORE_CTAS]])
    // CHECK: ttg.local_store
    ttg.local_store %values, %remote : tensor<2x32xi32, #blocked> -> !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>
    // CHECK: %[[GATHER_CTAS:.*]] = arith.constant 2 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[GATHER_CTAS]])
    // CHECK: ttg.local_gather
    %g = ttg.local_gather %remote[%indices] {axis = 1 : i32} : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>, tensor<2x32xi32, #blocked> -> tensor<2x32xi32, #blocked>
    // CHECK: %[[SCATTER_CTAS:.*]] = arith.constant 2 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[SCATTER_CTAS]])
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}({{.*}}%[[SCATTER_CTAS]])
    // CHECK: ttg.local_scatter
    ttg.local_scatter %remote[%indices], %values {axis = 1 : i32} : !ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>, tensor<2x32xi32, #blocked>, tensor<2x32xi32, #blocked>
    // CHECK: %[[ATOMIC_CTAS:.*]] = arith.constant 2 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[ATOMIC_CTAS]])
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}({{.*}}%[[ATOMIC_CTAS]])
    // CHECK: ttg.local_atomic_scatter_rmw
    %a = ttg.local_atomic_scatter_rmw add, %remote[%indices], %values {axis = 1 : i32} : (!ttg.memdesc<2x32xi32, #shared, #smem, mutable, 4x32>, tensor<2x32xi32, #blocked>, tensor<2x32xi32, #blocked>) -> tensor<2x32xi32, #blocked>
    tt.return
  }
}

// -----

// Dynamically selected subviews can have the same physical byte offset while
// belonging to different CTAs. Barrier checks retain the selected owner and
// verify every overlapping barrier with one call.
#owner_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#owner_barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#owner_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#owner_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  // CHECK-LABEL: @dynamic_barrier_owner_selection
  // CHECK-SAME: (%[[CHOOSE:.*]]: i1)
  tt.func public @dynamic_barrier_owner_selection(%choose: i1) {
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<4x32xi32, #owner_shared, #owner_smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2xi64, #owner_barrier, #owner_smem, mutable>
    %other = ttg.local_alloc {allocation.offset = 8 : i32} : () -> !ttg.memdesc<2xi64, #owner_barrier, #owner_smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #owner_barrier, #owner_smem, mutable>
    ttng.init_barrier %other, 1 : !ttg.memdesc<2xi64, #owner_barrier, #owner_smem, mutable>
    %local = ttg.memdesc_subslice %parent [0, 0] : !ttg.memdesc<4x32xi32, #owner_shared, #owner_smem, mutable> -> !ttg.memdesc<2x32xi32, #owner_shared, #owner_smem, mutable, 4x32>
    %remote = ttg.memdesc_subslice %parent [2, 0] : !ttg.memdesc<4x32xi32, #owner_shared, #owner_smem, mutable> -> !ttg.memdesc<2x32xi32, #owner_shared, #owner_smem, mutable, 4x32>
    // CHECK: arith.select %[[CHOOSE]], {{.*}} : !ttg.memdesc
    %selected = arith.select %choose, %local, %remote : !ttg.memdesc<2x32xi32, #owner_shared, #owner_smem, mutable, 4x32>
    // CHECK: %[[LOCAL_SELECTED:.*]] = arith.andi {{.*}}, %[[CHOOSE]] : i1
    // CHECK: %[[REMOTE_CHOICE:.*]] = arith.xori %[[CHOOSE]], {{.*}} : i1
    // CHECK: %[[REMOTE_SELECTED:.*]] = arith.andi {{.*}}, %[[REMOTE_CHOICE]] : i1
    // CHECK: arith.select %[[LOCAL_SELECTED]], {{.*}} : i32
    // CHECK: arith.select %[[REMOTE_SELECTED]], {{.*}} : i32
    // CHECK: %[[BARRIER_OWNERS:.*]] = arith.ori {{.*}} : tensor<2xi32
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init{{.*}}(%[[BARRIER_OWNERS]],
    // CHECK-NOT: tt.call @__triton_consan_verify_barrier_can_init
    // CHECK: ttg.local_load
    %value = ttg.local_load %selected : !ttg.memdesc<2x32xi32, #owner_shared, #owner_smem, mutable, 4x32> -> tensor<2x32xi32, #owner_blocked>
    tt.return
  }
}

// -----

// A known allocation and an external descriptor share the known state lane;
// the unknown descriptor additionally covers its dedicated wildcard lane.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @known_and_unknown_descriptor_state
  tt.func public @known_and_unknown_descriptor_state(%incoming: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    // CHECK: %[[KNOWN:.*]] = ttg.local_alloc
    %known = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    // CHECK: arith.constant dense<[true, false]> : tensor<2xi1
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_set_read_visibility_nw1_I32_
    // CHECK: ttg.local_load %[[KNOWN]]
    %0 = ttg.local_load %known : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    // CHECK: arith.constant dense<true> : tensor<2xi1
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_set_read_visibility_nw1_T2xI1_
    // CHECK: ttg.local_load %arg0
    %1 = ttg.local_load %incoming : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }
}

// -----

// An unknown-only module still allocates sanitizer state without fabricating
// or dynamically comparing a physical allocation base.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @unknown_only_descriptor_state
  tt.func public @unknown_only_descriptor_state(%incoming: !ttg.memdesc<16xi32, #shared, #smem, mutable>) {
    // CHECK: arith.constant dense<true> : tensor<1xi1
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK-NOT: tti.experimental_memdesc_to_i32
    // An unknown descriptor stays on the mask path even when B is one.
    // CHECK: tt.call @__triton_consan_set_read_visibility_nw1_T1xI1_
    // CHECK-NOT: tti.experimental_memdesc_to_i32
    // CHECK: ttg.local_load %arg0
    %0 = ttg.local_load %incoming : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }
}

// -----

// A warp-group wait forwards descriptor provenance to its result while
// separately preserving its asynchronous read-completion semantics.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @warp_group_wait_preserves_descriptor
  tt.func public @warp_group_wait_preserves_descriptor() {
    // CHECK: %[[BUFFER:.*]] = ttg.local_alloc
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_reads
    // CHECK: %[[WAITED:.*]] = ttng.warp_group_dot_wait %[[BUFFER]]
    %waited = ttng.warp_group_dot_wait %buffer {pendings = 0 : i32} : !ttg.memdesc<16xi32, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: ttg.local_load %[[WAITED]]
    %value = ttg.local_load %waited : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32>
    tt.return
  }
}

// -----

#async_remote_src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
#async_remote_dst = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#async_remote_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 520 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @async_shared_store_absolute_remote_recipient
  tt.func public @async_shared_store_absolute_remote_recipient(%src: tensor<128xi32, #async_remote_src>) {
    %true = arith.constant true
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<256xi32, #async_remote_dst, #async_remote_smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 512 : i32} : () -> !ttg.memdesc<2xi64, #async_remote_dst, #async_remote_smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #async_remote_dst, #async_remote_smem, mutable>
    ttng.barrier_expect %bar, 256, %true : !ttg.memdesc<2xi64, #async_remote_dst, #async_remote_smem, mutable>
    %view = ttg.memdesc_subslice %parent [128] : !ttg.memdesc<256xi32, #async_remote_dst, #async_remote_smem, mutable> -> !ttg.memdesc<128xi32, #async_remote_dst, #async_remote_smem, mutable, 256>
    // CHECK: %[[REMOTE_CTA:.*]] = arith.constant 2 : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{.*}}%[[REMOTE_CTA]])
    // CHECK: tt.call @__triton_consan_track_barrier_write_for_buffer{{.*}}({{.*}}%[[REMOTE_CTA]], %[[REMOTE_CTA]])
    // CHECK: %[[BYTES_PER_CTA:.*]] = arith.constant -256 : i64
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state{{.*}}({{.*}}%[[BYTES_PER_CTA]], {{.*}}%[[REMOTE_CTA]], {{.*}})
    // CHECK: ttng.async_shared_store
    ttng.async_shared_store %src, %view, %bar : tensor<128xi32, #async_remote_src> -> !ttg.memdesc<128xi32, #async_remote_dst, #async_remote_smem, mutable, 256>, !ttg.memdesc<2xi64, #async_remote_dst, #async_remote_smem, mutable>
    tt.return
  }
}

// -----

#lifetime_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#lifetime_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 64 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @barrier_storage_lifetime
  tt.func public @barrier_storage_lifetime() {
    %payload = ttg.local_alloc {allocation.offset = 0 : i32}
        : () -> !ttg.memdesc<16xi32, #lifetime_shared, #lifetime_smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 0 : i32}
        : () -> !ttg.memdesc<1xi64, #lifetime_shared, #lifetime_smem, mutable>
    %uninitialized = ttg.local_alloc {allocation.offset = 8 : i32}
        : () -> !ttg.memdesc<1xi64, #lifetime_shared, #lifetime_smem, mutable>
    // Init and invalidate are generic writes to the barrier bytes.
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: ttng.init_barrier
    ttng.init_barrier %barrier, 1
        : !ttg.memdesc<1xi64, #lifetime_shared, #lifetime_smem, mutable>
    // Ordinary accesses cannot reuse live barrier storage.
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    // CHECK: ttg.local_load
    %before = ttg.local_load %payload
        : !ttg.memdesc<16xi32, #lifetime_shared, #lifetime_smem, mutable>
        -> tensor<16xi32>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_invalidate_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier invalidated while a thread is waiting"
    // CHECK: ttng.inval_barrier
    ttng.inval_barrier %barrier
        : !ttg.memdesc<1xi64, #lifetime_shared, #lifetime_smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK: ttng.async_copy_mbarrier_arrive
    ttng.async_copy_mbarrier_arrive %uninitialized
        : !ttg.memdesc<1xi64, #lifetime_shared, #lifetime_smem, mutable>
    tt.return
  }
}

// -----

#nested_src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nested_dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#nested_dst = #ttg.slice<{dim = 1, parent = #nested_dst_parent}>
#nested_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nested_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 640 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  // Reachable private functions can combine register-only compiler scratch,
  // volatile global accesses, and nested calls without instrumentation.
  // CHECK-LABEL: tt.func private @nested_scratch_leaf
  // CHECK-NOT: tt.call @__triton_consan
  // CHECK: tt.load {{.*}}isVolatile = true
  // CHECK: ttg.convert_layout
  tt.func private @nested_scratch_leaf(
      %value: tensor<128xi32, #nested_src>, %global: !tt.ptr<i32>) {
    %loaded = tt.load %global {isVolatile = true} : !tt.ptr<i32>
    %converted = ttg.convert_layout %value {allocation.offset = 0 : i32, allocation.size = 512 : i32}
        : tensor<128xi32, #nested_src> -> tensor<128xi32, #nested_dst>
    tt.return
  }

  tt.func private @nested_scratch_middle(
      %value: tensor<128xi32, #nested_src>, %global: !tt.ptr<i32>) {
    tt.call @nested_scratch_leaf(%value, %global) {allocation.offset = 0 : i32, allocation.size = 512 : i32}
        : (tensor<128xi32, #nested_src>, !tt.ptr<i32>) -> ()
    tt.return
  }

  tt.func private @forward_shared_descriptor(
      %incoming: !ttg.memdesc<16xi32, #nested_shared, #nested_smem, mutable>)
      -> !ttg.memdesc<16xi32, #nested_shared, #nested_smem, mutable> {
    tt.return %incoming : !ttg.memdesc<16xi32, #nested_shared, #nested_smem, mutable>
  }

  // CHECK-LABEL: tt.func public @nested_scratch_and_descriptor_forwarding
  tt.func public @nested_scratch_and_descriptor_forwarding(
      %value: tensor<128xi32, #nested_src>, %global: !tt.ptr<i32>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32}
        : () -> !ttg.memdesc<16xi32, #nested_shared, #nested_smem, mutable>
    %live_barrier = ttg.local_alloc {allocation.offset = 128 : i32}
        : () -> !ttg.memdesc<1xi64, #nested_shared, #nested_smem, mutable>
    // A summarized call cannot reuse storage occupied by a live barrier.
    // CHECK: ttng.init_barrier
    ttng.init_barrier %live_barrier, 1
        : !ttg.memdesc<1xi64, #nested_shared, #nested_smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_publish_write_visibility
    // CHECK: tt.call @nested_scratch_middle
    tt.call @nested_scratch_middle(%value, %global) {allocation.offset = 128 : i32, allocation.size = 512 : i32}
        : (tensor<128xi32, #nested_src>, !tt.ptr<i32>) -> ()
    // CHECK: %[[FORWARDED:.*]] = tt.call @forward_shared_descriptor
    %forwarded = tt.call @forward_shared_descriptor(%buffer)
        : (!ttg.memdesc<16xi32, #nested_shared, #nested_smem, mutable>)
        -> !ttg.memdesc<16xi32, #nested_shared, #nested_smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: ttg.local_load %[[FORWARDED]]
    %loaded = ttg.local_load %forwarded
        : !ttg.memdesc<16xi32, #nested_shared, #nested_smem, mutable>
        -> tensor<16xi32>
    tt.return
  }
}

// -----

#direct_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 32, rank = 1}>
#direct_smem = #ttg.shared_memory
#direct_src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#direct_dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#direct_dst = #ttg.slice<{dim = 1, parent = #direct_dst_parent}>
#direct_reduce = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 1024 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  // CHECK-LABEL: @conversion_and_reduction_detect_outstanding_tma
  tt.func public @conversion_and_reduction_detect_outstanding_tma(
      %desc: !tt.tensordesc<256xi32, #direct_shared>,
      %value: tensor<128xi32, #direct_src>,
      %reduce: tensor<1x256xf32, #direct_reduce>) {
    %zero = arith.constant 0 : i32
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32}
        : () -> !ttg.memdesc<256xi32, #direct_shared, #direct_smem, mutable>
    ttng.async_tma_copy_local_to_global %desc[%zero] %buffer
        : !tt.tensordesc<256xi32, #direct_shared>,
          !ttg.memdesc<256xi32, #direct_shared, #direct_smem, mutable>
    ttg.local_dealloc %buffer
        : !ttg.memdesc<256xi32, #direct_shared, #direct_smem, mutable>

    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_publish_write_visibility
    // CHECK: ttg.convert_layout
    %converted = ttg.convert_layout %value {allocation.offset = 0 : i32, allocation.size = 512 : i32}
        : tensor<128xi32, #direct_src> -> tensor<128xi32, #direct_dst>

    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_publish_write_visibility
    // CHECK: "tt.reduce"
    %reduced = "tt.reduce"(%reduce) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) {allocation.offset = 256 : i32, allocation.size = 16 : i32}
        : (tensor<1x256xf32, #direct_reduce>)
        -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #direct_reduce}>>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}

// -----

#reduce_groups = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0], [0, 1]]}>

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 32 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  // CHECK-LABEL: @reduce_cross_cta_scratch_is_cta_local
  tt.func public @reduce_cross_cta_scratch_is_cta_local(
      %value: tensor<8x256xf32, #reduce_groups>) {
    // Reduction groups {0,2} and {1,3} read peers only after their intrinsic
    // cluster barrier; scratch writes touch the current CTA alone.
    // CHECK: tti.experimental_lock_acquire
    // CHECK: arith.constant dense<true> : tensor<1xi1
    // CHECK: %[[REDUCE_CTA:.*]] = tti.experimental_cluster_cta_id
    // CHECK: %[[REDUCE_ONE:.*]] = arith.constant 1 : i32
    // CHECK: %[[REDUCE_OWNER:.*]] = arith.shli %[[REDUCE_ONE]], %[[REDUCE_CTA]] : i32
    // CHECK: %[[REDUCE_GROUP_CTA:.*]] = tti.experimental_cluster_cta_id
    // CHECK: %[[REDUCE_GROUP_FIXED:.*]] = arith.constant 1 : i32
    // CHECK: %[[REDUCE_GROUP_BASE:.*]] = arith.andi %[[REDUCE_GROUP_CTA]], %[[REDUCE_GROUP_FIXED]] : i32
    // CHECK: %[[REDUCE_GROUP_PATTERN:.*]] = arith.constant 5 : i32
    // CHECK: %[[REDUCE_GROUP_SHIFT:.*]] = arith.shli %[[REDUCE_GROUP_PATTERN]], %[[REDUCE_GROUP_BASE]] : i32
    // CHECK: %[[REDUCE_READERS:.*]] = arith.ori {{.*}}, %[[REDUCE_GROUP_SHIFT]] : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[REDUCE_READERS]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}%[[REDUCE_OWNER]]
    // CHECK: tt.call @__triton_consan_publish_write_visibility{{.*}}%[[REDUCE_OWNER]]
    // CHECK: "tt.reduce"
    %reduced = "tt.reduce"(%value) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) {allocation.offset = 0 : i32, allocation.size = 32 : i32}
        : (tensor<8x256xf32, #reduce_groups>)
        -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #reduce_groups}>>
    ttng.cluster_barrier {relaxed = true}
    tt.return
  }
}

// -----

#cross_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#cross_dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1]]}>
#cross_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0], CGALayout = [[1, 0]]}>
#cross_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 512 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  // CHECK-LABEL: @convert_layout_cross_cta_scratch
  tt.func public @convert_layout_cross_cta_scratch(
      %value: tensor<8x32xi32, #cross_src>) {
    %buffer = ttg.local_alloc %value {allocation.offset = 0 : i32}
        : (tensor<8x32xi32, #cross_src>) -> !ttg.memdesc<8x32xi32, #cross_shared, #cross_smem, mutable>
    // CHECK: ttg.local_load
    %loaded = ttg.local_load %buffer
        : !ttg.memdesc<8x32xi32, #cross_shared, #cross_smem, mutable> -> tensor<8x32xi32, #cross_src>
    // CHECK: ttg.local_dealloc
    ttg.local_dealloc %buffer : !ttg.memdesc<8x32xi32, #cross_shared, #cross_smem, mutable>
    // CHECK: ttng.cluster_barrier {relaxed = true}
    ttng.cluster_barrier {relaxed = true}
    // Peer reads follow the conversion's intrinsic cluster barrier; the
    // compiler-owned scratch write belongs only to its issuing CTA.
    // CHECK: tti.experimental_lock_acquire
    // CHECK: arith.constant dense<true> : tensor<1xi1
    // CHECK: %[[CONVERT_CTA:.*]] = tti.experimental_cluster_cta_id
    // CHECK: %[[CONVERT_ONE:.*]] = arith.constant 1 : i32
    // CHECK: %[[CONVERT_OWNER:.*]] = arith.shli %[[CONVERT_ONE]], %[[CONVERT_CTA]] : i32
    // CHECK: %[[CONVERT_PEERS:.*]] = arith.constant 3 : i32
    // CHECK: %[[CONVERT_READERS:.*]] = arith.ori {{.*}}, %[[CONVERT_PEERS]] : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[CONVERT_READERS]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}%[[CONVERT_OWNER]]
    // CHECK: tt.call @__triton_consan_publish_write_visibility{{.*}}%[[CONVERT_OWNER]]
    // CHECK: ttg.convert_layout
    %converted = ttg.convert_layout %loaded {allocation.offset = 0 : i32, allocation.size = 512 : i32}
        : tensor<8x32xi32, #cross_src> -> tensor<8x32xi32, #cross_dst>
    ttng.cluster_barrier {relaxed = true}
    tt.return
  }
}

// -----

#atomic_barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#atomic_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32, ttg.shared = 8 : i32, ttg.target = "cuda:90", ttg.tensor_memory_size = 0 : i32} {
  // CHECK-LABEL: @scalar_atomic_scratch_broadcast
  tt.func public @scalar_atomic_scratch_broadcast(
      %ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %one = arith.constant 1 : i32
    %barrier = ttg.local_alloc {allocation.offset = 0 : i32}
        : () -> !ttg.memdesc<2xi64, #atomic_barrier, #atomic_smem, mutable>
    %cta = tti.experimental_cluster_cta_id : i32
    %is_cta_one = arith.cmpi eq, %cta, %one : i32
    scf.if %is_cta_one {
      ttng.init_barrier %barrier, 1 : !ttg.memdesc<2xi64, #atomic_barrier, #atomic_smem, mutable>
    }
    // CHECK: ttng.cluster_barrier {relaxed = true}
    ttng.cluster_barrier {relaxed = true}
    // Both CTAs consume the result, but only CTA zero owns the scratch. The
    // live barrier in CTA one does not overlap the atomic's physical storage.
    // CHECK: tti.experimental_lock_acquire
    // CHECK: %[[SCALAR_PRODUCER:.*]] = arith.cmpi eq, {{.*}} : i32
    // CHECK: arith.constant dense<[true, false]> : tensor<2xi1
    // CHECK: %[[SCALAR_CTA:.*]] = tti.experimental_cluster_cta_id
    // CHECK: %[[SCALAR_ONE:.*]] = arith.constant 1 : i32
    // CHECK: %[[SCALAR_OWNER:.*]] = arith.shli %[[SCALAR_ONE]], %[[SCALAR_CTA]] : i32
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[SCALAR_PRODUCER]]{{.*}}%[[SCALAR_OWNER]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}%[[SCALAR_PRODUCER]]{{.*}}%[[SCALAR_OWNER]]
    // CHECK: tt.call @__triton_consan_publish_write_visibility{{.*}}%[[SCALAR_PRODUCER]]{{.*}}%[[SCALAR_OWNER]]
    // CHECK: tt.atomic_rmw
    %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %one
        {allocation.offset = 0 : i32, allocation.size = 4 : i32}
        : (!tt.ptr<i32>, i32) -> i32
    tt.store %out, %old : !tt.ptr<i32>
    ttng.cluster_barrier {relaxed = true}
    tt.return
  }
}

// -----

#tma_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[1, 0]]}>
#tma_dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1]]}>
#tma_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, CGALayout = [[1, 0]]}>
#tma_barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#tma_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32, "ttng.two-ctas" = true, ttg.shared = 8192 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32} {
  // CHECK-LABEL: @remote_tma_cross_cta_conversion_scratch
  tt.func public @remote_tma_cross_cta_conversion_scratch(
      %desc: !tt.tensordesc<8x32xi32, #tma_shared>, %value: tensor<16x32xi32, #tma_src>) {
    %zero = arith.constant 0 : i32
    %cta = tti.experimental_cluster_cta_id : i32
    %lead = arith.cmpi eq, %cta, %zero : i32
    %parent = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x32xi32, #tma_shared, #tma_smem, mutable>
    %remote = ttg.memdesc_subslice %parent [8, 0] : !ttg.memdesc<16x32xi32, #tma_shared, #tma_smem, mutable> -> !ttg.memdesc<8x32xi32, #tma_shared, #tma_smem, mutable, 16x32>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<2xi64, #tma_barrier, #tma_smem, mutable>
    ttng.init_barrier %bar, 1 : !ttg.memdesc<2xi64, #tma_barrier, #tma_smem, mutable>
    ttng.barrier_expect %bar, 1024, %lead : !ttg.memdesc<2xi64, #tma_barrier, #tma_smem, mutable>
    // CHECK: ttng.async_tma_copy_global_to_local
    ttng.async_tma_copy_global_to_local %desc[%zero, %zero] %remote, %bar, %lead : !tt.tensordesc<8x32xi32, #tma_shared>, !ttg.memdesc<2xi64, #tma_barrier, #tma_smem, mutable> -> !ttg.memdesc<8x32xi32, #tma_shared, #tma_smem, mutable, 16x32>
    // CHECK: ttg.local_dealloc
    ttg.local_dealloc %parent : !ttg.memdesc<16x32xi32, #tma_shared, #tma_smem, mutable>
    // CTA zero's pending async write targets CTA one. Conversion must check
    // that remote owner but publish its own CTA-local scratch write only.
    // CHECK: tti.experimental_lock_acquire
    // CHECK: arith.constant dense<[true, false]> : tensor<2xi1
    // CHECK: %[[TMA_OWNER_CTA:.*]] = tti.experimental_cluster_cta_id
    // CHECK: %[[TMA_OWNER_ONE:.*]] = arith.constant 1 : i32
    // CHECK: %[[TMA_OWNER:.*]] = arith.shli %[[TMA_OWNER_ONE]], %[[TMA_OWNER_CTA]] : i32
    // CHECK: %[[TMA_PEERS:.*]] = arith.constant 3 : i32
    // CHECK: %[[TMA_READERS:.*]] = arith.ori {{.*}}, %[[TMA_PEERS]] : i32
    // CHECK: tt.call @__triton_consan_set_proxy_access{{.*}}%[[TMA_READERS]]
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}%[[TMA_READERS]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}%[[TMA_OWNER]]
    // CHECK: tt.call @__triton_consan_publish_write_visibility{{.*}}%[[TMA_OWNER]]
    // CHECK: ttg.convert_layout
    %converted = ttg.convert_layout %value {allocation.offset = 0 : i32, allocation.size = 1024 : i32} : tensor<16x32xi32, #tma_src> -> tensor<16x32xi32, #tma_dst>
    tt.return
  }
}

// -----

// Selecting a barrier from a ring selects exactly one state atom at runtime.
// The B=1 viewport must retain the original B=2 strides, every observer and
// barrier column, and both origin/phase banks.
#row_barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0], CGALayout = [[1]]}>
#row_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32, ttg.shared = 16 : i32, ttg.target = "cuda:100", ttg.tensor_memory_size = 0 : i32} {
  // CHECK-LABEL: tt.func private @__triton_consan_set_read_visibility_nw1_I32_
  // CHECK-SAME: (%[[ROW_INDEX:[^:]+]]: i32,
  // CHECK: %[[ROW_STRIDE:.*]] = arith.constant 2 : i32
  // CHECK-NEXT: %[[ROW_OFFSET:.*]] = arith.muli %[[ROW_INDEX]], %[[ROW_STRIDE]] : i32
  // CHECK-NEXT: tt.addptr {{.*}}, %[[ROW_OFFSET]] : !tt.ptr<i{{32|64}}>, i32
  // The last visibility axis retains stride 16, not the viewport's stride 8.
  // CHECK: arith.constant dense<16> : tensor<2xi32
  // CHECK: tt.load {{.*}} : tensor<2x1x2x2x2x!tt.ptr<i{{32|64}}>
  // CHECK: arith.constant dense<16> : tensor<2xi32
  // CHECK: tt.store {{.*}} : tensor<2x1x2x2x2x!tt.ptr<i{{32|64}}>
  // Each origin/phase bank spans 16 backing elements, not 8 viewport elements.
  // CHECK: %[[ROW_ORIGIN:.*]] = tti.experimental_cluster_cta_id
  // CHECK-NEXT: %[[ROW_BANK_SIZE:.*]] = arith.constant 16 : i32
  // CHECK-NEXT: %[[ROW_BANK_OFFSET:.*]] = arith.muli %[[ROW_ORIGIN]], %[[ROW_BANK_SIZE]] : i32
  // CHECK-NEXT: tt.addptr {{.*}}, %[[ROW_BANK_OFFSET]] : !tt.ptr<i{{32|64}}>, i32
  // CHECK: arith.constant dense<8> : tensor<2xi32
  // CHECK: tt.load {{.*}} : tensor<2x1x2x2x!tt.ptr<i{{32|64}}>
  // CHECK: arith.constant dense<8> : tensor<2xi32
  // CHECK: tt.store {{.*}} : tensor<2x1x2x2x!tt.ptr<i{{32|64}}>
  // CHECK: %[[ROW_PHASE_STRIDE:.*]] = arith.constant 2 : i32
  // CHECK-NEXT: %[[ROW_PHASE_ORIGIN:.*]] = arith.addi %[[ROW_ORIGIN]], %[[ROW_PHASE_STRIDE]] : i32
  // CHECK-NEXT: %[[ROW_NEXT_BANK_SIZE:.*]] = arith.constant 16 : i32
  // CHECK-NEXT: arith.muli %[[ROW_PHASE_ORIGIN]], %[[ROW_NEXT_BANK_SIZE]] : i32
  // CHECK: tt.load {{.*}} : tensor<2x1x2x2x!tt.ptr<i{{32|64}}>
  // CHECK: tt.store {{.*}} : tensor<2x1x2x2x!tt.ptr<i{{32|64}}>
  // CHECK: tt.return
  // CHECK-LABEL: @single_buffer_dynamic_barrier_ring
  // CHECK-SAME: (%[[ROW_DYNAMIC_INDEX:[^:]+]]: i32,
  tt.func public @single_buffer_dynamic_barrier_ring(%idx: i32, %phase: i32) {
    // Each entry has one eight-byte barrier per CTA.
    // CHECK: tti.experimental_buffer_descriptors [0, 8], [8, 8], shared_mem : tensor<2xi64,
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %true = arith.constant true
    %ring = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x2xi64, #row_barrier, #row_smem, mutable>
    %bar0 = ttg.memdesc_index %ring[%c0] : !ttg.memdesc<2x2xi64, #row_barrier, #row_smem, mutable> -> !ttg.memdesc<2xi64, #row_barrier, #row_smem, mutable>
    %bar1 = ttg.memdesc_index %ring[%c1] : !ttg.memdesc<2x2xi64, #row_barrier, #row_smem, mutable> -> !ttg.memdesc<2xi64, #row_barrier, #row_smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<2xi64, #row_barrier, #row_smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<2xi64, #row_barrier, #row_smem, mutable>
    // CHECK: %[[ROW_DYNAMIC:.*]] = ttg.memdesc_index {{.*}}[%[[ROW_DYNAMIC_INDEX]]]
    %dynamic = ttg.memdesc_index %ring[%idx] : !ttg.memdesc<2x2xi64, #row_barrier, #row_smem, mutable> -> !ttg.memdesc<2xi64, #row_barrier, #row_smem, mutable>
    // CHECK: tti.experimental_memdesc_to_i32 %[[ROW_DYNAMIC]]
    // CHECK: internal ConSan error: active memdesc resolved to no buffer state
    // CHECK: arith.select {{.*}} : i32
    // CHECK: %[[ROW_SELECTED:.*]] = arith.select {{.*}} : i32
    // CHECK: tt.call @__triton_consan_set_read_visibility_nw1_I32_{{.*}}(%[[ROW_SELECTED]],
    // CHECK: ttng.tc_gen5_commit %[[ROW_DYNAMIC]]
    ttng.tc_gen5_commit %dynamic : !ttg.memdesc<2xi64, #row_barrier, #row_smem, mutable>
    ttng.wait_barrier %dynamic, %phase, %true : !ttg.memdesc<2xi64, #row_barrier, #row_smem, mutable>
    ttng.inval_barrier %bar0 : !ttg.memdesc<2xi64, #row_barrier, #row_smem, mutable>
    ttng.inval_barrier %bar1 : !ttg.memdesc<2xi64, #row_barrier, #row_smem, mutable>
    tt.return
  }
}
