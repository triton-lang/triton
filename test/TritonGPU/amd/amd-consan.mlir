// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-prepare-consan-captures="target=amd" -tritoninstrument-concurrency-sanitizer | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: tt.func private @__triton_consan_verify_write_visibility_
  // CHECK: %[[WRITE_VISIBILITY:.*]] = tt.load
  // CHECK: arith.cmpi eq, %[[WRITE_VISIBILITY]],
  // CHECK: %[[SELECTED_THREAD_BIT:.*]] = arith.shli
  // CHECK: %[[VISIBLE_THREAD_BITS:.*]] = arith.andi %[[WRITE_VISIBILITY]], %[[SELECTED_THREAD_BIT]]
  // CHECK: arith.cmpi eq, %[[VISIBLE_THREAD_BITS]], %[[SELECTED_THREAD_BIT]]
  // CHECK-LABEL: @single_local_alloc
  tt.func public @single_local_alloc() {
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i64
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @two_local_alloc
  tt.func public @two_local_alloc() {
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 2 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i64
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    ttg.local_load %1 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @three_local_alloc
  tt.func public @three_local_alloc() {
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i64
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %2 = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 12288 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    ttg.local_load %1 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    ttg.local_load %2 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @three_sub_bufs
  tt.func public @three_sub_bufs() {
    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i64
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<3x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.local_load %1 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: #[[READ_BARS_L:.*]] = #ttg.blocked<{sizePerThread = [2, 4], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
  // CHECK: @read_bars_alloc
  tt.func public @read_bars_alloc() {
    // CHECK: %[[READ_BARS_G:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_BARS_G]], %c0_i8
    %c0 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<4x1xi64, #shared1, #smem, mutable>
    %bar_sub = ttg.memdesc_index %bar[%c0] : !ttg.memdesc<4x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar_sub, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %buf_sub = ttg.memdesc_index %0[%c0] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    ttg.local_load %buf_sub : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_copy_global_to_local
  tt.func public @async_copy_global_to_local(%ptr: tensor<32x32x!tt.ptr<f16>, #blocked>) {
    // CHECK: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRT_COMMITS_GLOB]], %c0_i8

    // CHECK: tt.call @__triton_consan_verify_write_visibility_nw1
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_check_outstanding_commits{{.*}}%[[THREAD_BIT]], %[[WRT_COMMITS_GLOB]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility_nw1
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_update_outstanding_commits_nw1{{.*}}%[[THREAD_BIT]], %[[WRT_COMMITS_GLOB]]
    // CHECK: ttg.async_copy_global_to_local

    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_copy_global_to_local_with_barriers
  tt.func public @async_copy_global_to_local_with_barriers(%ptr: tensor<32x32x!tt.ptr<f16>, #blocked>) {
    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>

    // CHECK-DAG: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>

    // CHECK: tt.call @__triton_consan_init_barrier_state

    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_check_outstanding_commits{{.*}}%[[THREAD_BIT]], %[[WRT_COMMITS_GLOB]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}({{[^,]+}}
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_update_outstanding_commits{{.*}}%[[THREAD_BIT]], %[[WRT_COMMITS_GLOB]]
    // CHECK: ttg.async_copy_global_to_local
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_commit_group
  tt.func public @async_commit_group() {
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    ttg.async_commit_group
    ttg.local_load %shmem : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @async_wait
  tt.func public @async_wait() {
    // CHECK: tti.experimental_lock_acquire
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 1 : i64
    // CHECK: %[[OUTSTANDING_NUM:.*]] = arith.constant 42 : i32
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_writes{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]], %[[OUTSTANDING_NUM]]
    // CHECK-NOT: tt.call @__triton_consan_clear_outstanding_commits_transfer_
    // CHECK: ttg.async_wait
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    ttg.async_wait {num = 42 : i32}
    ttg.local_load %shmem : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @local_alloc_with_src
  tt.func public @local_alloc_with_src(%data: tensor<32x32xf16, #blocked>) {
    // CHECK: %[[BUF:.*]] = ttg.local_alloc
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}({{[^,]+}}
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}({{[^,]+}}
    %buf = ttg.local_alloc %data {allocation.offset = 0 : i32} : (tensor<32x32xf16, #blocked>) -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @alias_matrix_shared
  tt.func public @alias_matrix_shared() {
    // CHECK-DAG: arith.constant dense<[true, true, false, false]> : tensor<4xi1
    // CHECK-DAG: arith.constant dense<[false, true, true, false]> : tensor<4xi1
    %buf0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %buf1 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.local_load %buf0 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    ttg.local_load %buf1 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @alias_matrix_shared_indexed
  tt.func public @alias_matrix_shared_indexed() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK-DAG: arith.constant dense<[true, false]> : tensor<2xi1
    // CHECK-DAG: arith.constant dense<[false, true]> : tensor<2xi1
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<2x32xf32, #shared, #smem, mutable>
    %buf0 = ttg.memdesc_index %smem[%c0_i32] : !ttg.memdesc<2x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %buf1 = ttg.memdesc_index %smem[%c1_i32] : !ttg.memdesc<2x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.local_load %buf0 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    ttg.local_load %buf1 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @alias_matrix_shared_subslice
  tt.func public @alias_matrix_shared_subslice() {
    // CHECK-DAG: arith.constant dense<true> : tensor<2xi1
    // CHECK-DAG: arith.constant dense<[false, true]> : tensor<2xi1
    %buf0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %buf1 = ttg.memdesc_subslice %buf0 [32] : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable, 64>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.local_load %buf0 : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32>
    ttg.local_load %buf1 : !ttg.memdesc<32xf32, #shared, #smem, mutable, 64> -> tensor<32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @amdg_init_barrier
  tt.func public @amdg_init_barrier() {
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    ttg.local_load %buf : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @amdg_wait_barrier
  tt.func public @amdg_wait_barrier() {
    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1xi64

    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i64
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK-DAG: tt.call @__triton_consan_update_waiting
    // CHECK-DAG: tt.call @__triton_consan_check_all_active_waiting
    // CHECK: amdg.wait_barrier
    amdg.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_transfer_visible_accesses{{.*}}%[[BARRIERS]], %[[WRITE_VISIBILITY_GLOB]], %[[WRITE_TRACKING_GLOB]], %[[READ_VISIBILITY_GLOB]], %[[READ_TRACKING_GLOB]]) : {{.*}}!tt.ptr<i64>, !tt.ptr<i8>, !tt.ptr<i64>, !tt.ptr<i64>) -> ()
    // CHECK: tt.call @__triton_consan_update_waiting
    // CHECK: tti.experimental_lock_release
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK: tt.func private @__triton_consan_verify_and_update_barrier_state
  // CHECK: %[[BARRIER_INIT_BITS:.*]] = arith.extui {{.*}} : tensor<{{.*}}xi1{{.*}}> to tensor<{{.*}}xi32
  // CHECK: %[[BARRIER_VALID_BITS:.*]] = arith.extui {{.*}} : tensor<{{.*}}xi1{{.*}}> to tensor<{{.*}}xi32
  // CHECK: %[[BARRIER_SHIFTED_VALID_BITS:.*]] = arith.shli %[[BARRIER_VALID_BITS]]
  // CHECK: %[[BARRIER_STATUS_BITS:.*]] = arith.ori %[[BARRIER_INIT_BITS]], %[[BARRIER_SHIFTED_VALID_BITS]]
  // CHECK: %[[BARRIER_PACKED_STATUS:.*]] = "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  // CHECK: arith.andi {{.*}} : i32
  // CHECK-NOT: "tt.reduce"
  // CHECK: tt.return %[[BARRIER_PACKED_STATUS]] : i32
  // CHECK-LABEL: @amdg_arrive_barrier
  tt.func public @amdg_arrive_barrier() {
    // CHECK-DAG: %[[BSTATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[BSTATE_GLOB]], %c0_i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_track_visible_accesses
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    // CHECK: tti.experimental_lock_release
    %phase = amdg.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> i32
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @amdg_wait_barrier_without_init
  tt.func public @amdg_wait_barrier_without_init() {
    %c0_i32 = arith.constant 0 : i32
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK: tt.call @__triton_consan_update_waiting
    amdg.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @amdg_arrive_barrier_without_init
  tt.func public @amdg_arrive_barrier_without_init() {
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    %phase = amdg.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> i32
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [32, 32]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @async_tdm_copy_global_to_local
  tt.func public @async_tdm_copy_global_to_local(%desc: !tt.tensordesc<32x32xf32>) {
    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1xi64
    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation, tt.divisibility = 16 : i64} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}(%[[READ_TRACKING_GLOB]], %c0_i64
    %c0_i32 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    amdg.init_barrier %bar, 8 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_publish_write_visibility
    // CHECK: tt.call @__triton_consan_track_visible_accesses
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    %1 = amdg.async_tdm_copy_global_to_local %desc into %0, barrier = %bar : !tt.tensordesc<32x32xf32>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [32, 32]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @async_tdm_copy_global_to_local_two_bufs_one_barrier
  tt.func public @async_tdm_copy_global_to_local_two_bufs_one_barrier(
      %a: !tt.tensordesc<32x32xf32>,
      %b: !tt.tensordesc<32x32xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32

    %a_smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %b_smem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    amdg.init_barrier %bar, 16 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    // CHECK: tt.call @__triton_consan_init_barrier_state
    // First TDM copy: full effects + barrier instrumentation
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_publish_write_visibility
    // CHECK: tt.call @__triton_consan_track_visible_accesses
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    %0 = amdg.async_tdm_copy_global_to_local %a into %a_smem, barrier = %bar : !tt.tensordesc<32x32xf32>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>

    // Second TDM copy: same full instrumentation
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_publish_write_visibility
    // CHECK: tt.call @__triton_consan_track_visible_accesses
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    %1 = amdg.async_tdm_copy_global_to_local %b into %b_smem, barrier = %bar : !tt.tensordesc<32x32xf32>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>

    %c0_phase = arith.constant 0 : i32
    amdg.wait_barrier %bar, %c0_phase : !ttg.memdesc<1xi64, #shared1, #smem, mutable>

    %va = ttg.local_load %a_smem : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    %vb = ttg.local_load %b_smem : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    %_ = arith.addf %va, %vb : tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @async_tdm_copy_global_to_local_no_barrier
  tt.func public @async_tdm_copy_global_to_local_no_barrier(%desc: !tt.tensordesc<32x32xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits_excl_self
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits_excl_self
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK-NOT: tt.call @__triton_consan_verify_and_update_barrier_state
    %1 = amdg.async_tdm_copy_global_to_local %desc into %0 : !tt.tensordesc<32x32xf32> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @async_tdm_copy_local_to_global
  tt.func public @async_tdm_copy_local_to_global(%desc: !tt.tensordesc<32x32xf32>, %ptr: tensor<128x128x!tt.ptr<f16>, #blocked>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<128x128x!tt.ptr<f16>, #blocked> -> <128x128xf16, #shared, #smem, mutable>

    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_check_outstanding_commits_excl_self
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    amdg.async_tdm_copy_local_to_global %desc from %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> !tt.tensordesc<32x32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @async_tdm_load_store_no_barrier
  tt.func public @async_tdm_load_store_no_barrier(%in_desc: !tt.tensordesc<32x32xf32>, %out_desc: !tt.tensordesc<32x32xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits_excl_self
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    %1 = amdg.async_tdm_copy_global_to_local %in_desc into %0 : !tt.tensordesc<32x32xf32> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_check_outstanding_commits_excl_self
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    amdg.async_tdm_copy_local_to_global %out_desc from %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> !tt.tensordesc<32x32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @async_tdm_copy_local_to_global_with_barrier
  tt.func public @async_tdm_copy_local_to_global_with_barrier(%desc: !tt.tensordesc<32x32xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    amdg.init_barrier %bar, 8 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_set_read_visibility
    // CHECK: tt.call @__triton_consan_track_visible_accesses
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    // CHECK-NOT: tt.call @__triton_consan_update_outstanding_commits
    amdg.async_tdm_copy_local_to_global %desc from %0, barrier = %bar : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !tt.tensordesc<32x32xf32>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @async_tdm_wait
  tt.func public @async_tdm_wait() {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>

    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_both
    amdg.async_tdm_wait {num = 0 : i32}

    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @async_tdm_intrinsic_wait
  tt.func public @async_tdm_intrinsic_wait() {
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>

    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_both
    amdg.async_tdm_intrinsic_wait {"ttg.num_tdm_ops" = 2 : i64, count = 2 : i32}

    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @amdg_async_wait
  tt.func public @amdg_async_wait() {
    // CHECK: tti.experimental_lock_acquire
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 1 : i64
    // CHECK: %[[OUTSTANDING_NUM:.*]] = arith.constant 42 : i32
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_writes{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]], %[[OUTSTANDING_NUM]]
    %shmem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    amdg.async_wait {"ttg.num_commit_groups" = 42 : i64, num_inst = 42 : i32}
    ttg.local_load %shmem : !ttg.memdesc<32x32xf16, #shared, #smem, mutable> -> tensor<32x32xf16, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @tdm_load_no_barrier_wait
  tt.func public @tdm_load_no_barrier_wait(%desc: !tt.tensordesc<32x32xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc into %0 : !tt.tensordesc<32x32xf32> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_both
    amdg.async_tdm_wait {num = 0 : i32}
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @tdm_store_no_barrier_wait
  tt.func public @tdm_store_no_barrier_wait(%desc: !tt.tensordesc<32x32xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    amdg.async_tdm_copy_local_to_global %desc from %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> !tt.tensordesc<32x32xf32>
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_both
    amdg.async_tdm_wait {num = 0 : i32}
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [32, 32]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @tdm_load_store_no_barrier_wait
  tt.func public @tdm_load_store_no_barrier_wait(%desc: !tt.tensordesc<32x32xf32>) {
    %c0_i32 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %1 = amdg.async_tdm_copy_global_to_local %desc into %0 : !tt.tensordesc<32x32xf32> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    amdg.async_tdm_copy_local_to_global %desc from %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> !tt.tensordesc<32x32xf32>
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_both
    amdg.async_tdm_wait {num = 0 : i32}
    ttg.local_load %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @local_load_barriers
  tt.func public @local_load_barriers() {
    // CHECK: tti.experimental_buffer_descriptors
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_set_read_visibility
    // CHECK: tti.experimental_lock_release
    ttg.local_load %buf : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @local_load_barriers_cp_async
  tt.func public @local_load_barriers_cp_async(%ptr: tensor<32x32x!tt.ptr<f16>, #blocked>) {
    // CHECK: tti.experimental_buffer_descriptors
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %shmem = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.async_copy_global_to_local %ptr, %shmem : tensor<32x32x!tt.ptr<f16>, #blocked> -> <32x32xf16, #shared, #smem, mutable>
    // CHECK: ttg.async_copy_global_to_local
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: tt.call @__triton_consan_set_read_visibility
    // CHECK: tti.experimental_lock_release
    // CHECK: ttg.local_load
    ttg.local_load %buf : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_allocation
  tt.func public @ws_allocation() {
    // CHECK-DAG: tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1xi64,
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
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
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_buf_ptrs_default
  tt.func public @ws_buf_ptrs_default() {
    // CHECK-DAG: tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
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

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK-LABEL: @ws_buf_ptrs_partition0
  tt.func public @ws_buf_ptrs_partition0() {
    // CHECK-DAG: tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
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

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 8 : i32} {
  // CHECK: tt.func private @__triton_consan_check_all_active_waiting
  // CHECK: %[[DEADLOCK_WAITING_BITS:.*]] = arith.extui {{.*}} : tensor<{{.*}}xi1{{.*}}> to tensor<{{.*}}xi32
  // CHECK: %[[DEADLOCK_IDLE_BITS:.*]] = arith.extui {{.*}} : tensor<{{.*}}xi1{{.*}}> to tensor<{{.*}}xi32
  // CHECK: %[[DEADLOCK_SHIFTED_IDLE_BITS:.*]] = arith.shli %[[DEADLOCK_IDLE_BITS]]
  // CHECK: %[[DEADLOCK_BITS:.*]] = arith.ori %[[DEADLOCK_WAITING_BITS]], %[[DEADLOCK_SHIFTED_IDLE_BITS]]
  // CHECK: %[[DEADLOCK_STATUS:.*]] = "tt.reduce"(%[[DEADLOCK_BITS]]) <{axis = 0 : i32}>
  // CHECK: arith.andi {{.*}} : i32
  // CHECK-NOT: "tt.reduce"
  // CHECK: arith.cmpi eq, %[[DEADLOCK_STATUS]], {{.*}} : i32
  // CHECK-LABEL: @ws_wait_barrier
  tt.func public @ws_wait_barrier() {
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: %[[ACTIVE_MASK:.*]] = arith.constant 5 : i32
    // CHECK: tt.call @__triton_consan_set_active_mask{{.*}}(%[[ACTIVE_MASK]],
    ttg.warp_specialize(%smem, %bar) attributes {actualRegisters = array<i32: 480, 32>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    default {
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_update_waiting
      // CHECK: tt.call @__triton_consan_check_all_active_waiting
      // CHECK: tti.experimental_lock_release
      %c0_i32 = arith.constant 0 : i32
      amdg.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg1: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      // CHECK: partition0
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_update_waiting
      // CHECK: tt.call @__triton_consan_check_all_active_waiting
      // CHECK: tti.experimental_lock_release
      %c0_i32 = arith.constant 0 : i32
      amdg.wait_barrier %arg2, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<128x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @ws_alias_matrix
  tt.func public @ws_alias_matrix() {
    // We expect exact overlapping-atom masks in the default region and
    // partition0 after lowering warp_specialize.
    // CHECK-DAG: arith.constant dense<[true, true, false, false]> : tensor<4xi1
    // CHECK-DAG: arith.constant dense<[false, true, true, false]> : tensor<4xi1
    %smem0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc {allocation.offset = 16 : i32} : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>

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

// Partitioned padded allocations have several simultaneous physical bases.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner_padded = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [16, 16]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner_padded}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.shared = 66592 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 2 : i32} {
  // CHECK-LABEL: @partitioned_padded_local_load_store
  // CHECK: ttg.local_alloc {allocation.offset = [0 : i32, 65536 : i32]}
  // CHECK: tt.call @__triton_consan_verify_write_visibility
  // CHECK: tt.call @__triton_consan_verify_read_visibility
  // CHECK: ttg.local_store
  // CHECK: tt.call @__triton_consan_verify_write_visibility
  // CHECK: ttg.local_load
  tt.func public @partitioned_padded_local_load_store(%value: tensor<16x16xf16, #blocked>) {
    %buffer = ttg.local_alloc {allocation.offset = [0 : i32, 65536 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    ttg.local_store %value, %buffer : tensor<16x16xf16, #blocked> -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %loaded = ttg.local_load %buffer : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> tensor<16x16xf16, #blocked>
    tt.return
  }
}

// -----

// Each partitioned subview has its own sanitizer lane; runtime descriptor
// selection must retain the physical base and state mask of either piece.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [2, 1], order = [1, 0]}>
#inner = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#partitioned = #ttg.partitioned_shared<{numPartitions = 2, numGroups = 2, partitionDim = 0, partitionLayout = #inner}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.shared = 66592 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 2 : i32} {
  // CHECK-LABEL: @partitioned_shared_subslice_masks
  // CHECK: arith.constant dense<[true, false]> : tensor<2xi1
  // CHECK: tt.call @__triton_consan_verify_write_visibility
  // CHECK: tt.call @__triton_consan_verify_read_visibility
  // CHECK: ttg.local_store
  // CHECK: arith.constant dense<[false, true]> : tensor<2xi1
  // CHECK: tt.call @__triton_consan_verify_write_visibility
  // CHECK: ttg.local_load
  // CHECK: tti.experimental_memdesc_to_i32
  // CHECK: tti.experimental_memory_offset_to_i32 0, shared_mem
  // CHECK: tti.experimental_memory_offset_to_i32 65536, shared_mem
  // CHECK: ttg.local_load
  tt.func public @partitioned_shared_subslice_masks(%value: tensor<4x16xf16, #blocked>, %choose: i1) {
    %buffer = ttg.local_alloc {allocation.offset = [0 : i32, 65536 : i32]} : () -> !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable>
    %first = ttg.memdesc_subslice %buffer [0, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    %second = ttg.memdesc_subslice %buffer [4, 0] : !ttg.memdesc<16x16xf16, #partitioned, #smem, mutable> -> !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    ttg.local_store %value, %first : tensor<4x16xf16, #blocked> -> !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    %loaded = ttg.local_load %second : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16> -> tensor<4x16xf16, #blocked>
    %selected = arith.select %choose, %first, %second : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16>
    %dynamic = ttg.local_load %selected : !ttg.memdesc<4x16xf16, #partitioned, #smem, mutable, 16x16> -> tensor<4x16xf16, #blocked>
    tt.return
  }
}

// -----

// A fused TDM load must track every physical destination independently.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 16384 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @fused_tdm_tracks_all_destinations
  // CHECK: arith.constant dense<[true, false]> : tensor<2xi1
  // CHECK: tt.call @__triton_consan_verify_write_visibility
  // CHECK: tt.call @__triton_consan_verify_read_visibility
  // CHECK: tt.call @__triton_consan_update_outstanding_commits
  // CHECK: arith.constant dense<[false, true]> : tensor<2xi1
  // CHECK: tt.call @__triton_consan_verify_write_visibility
  // CHECK: tt.call @__triton_consan_verify_read_visibility
  // CHECK: tt.call @__triton_consan_update_outstanding_commits
  // CHECK: tt.call @__triton_consan_update_outstanding_commits
  // CHECK: amdg.async_tdm_fused_copy_global_to_local
  tt.func public @fused_tdm_tracks_all_destinations(%desc0: !tt.tensordesc<64x64xf16, #shared>, %desc1: !tt.tensordesc<64x64xf16, #shared>) {
    %first = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %second = ttg.local_alloc {allocation.offset = 8192 : i32} : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %token = amdg.async_tdm_fused_copy_global_to_local %desc0, %desc1 into %first, %second {warp_used_hints = array<i32: 3, 12>} : !tt.tensordesc<64x64xf16, #shared>, !tt.tensordesc<64x64xf16, #shared> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Direct-to-LDS buffer loads participate in ordinary asynchronous-copy
// commit groups instead of being treated as synchronous shared-memory writes.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 8192 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @buffer_load_to_local_tracks_async_commit
  tt.func public @buffer_load_to_local_tracks_async_commit(%ptr: !tt.ptr<f32>, %offsets: tensor<32x64xi32, #blocked>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: amdg.buffer_load_to_local
    %token = amdg.buffer_load_to_local %ptr[%offsets] into %buffer : <f32>[tensor<32x64xi32, #blocked>] -> <32x64xf32, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: ttg.async_commit_group
    %group = ttg.async_commit_group tokens %token
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_writes
    // CHECK-NOT: tt.call @__triton_consan_clear_outstanding_commits_transfer_
    // CHECK: ttg.async_wait
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
}

// -----

// Packed transposed LDS loads receive the synchronous read instrumentation
// described by their declared memory effects.
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 1024 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @packed_transposed_load_tracks_read
  tt.func public @packed_transposed_load_tracks_read() {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<16x64xi8, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_set_read_visibility
    // CHECK: amdg.local_load_packed_transposed
    %value = amdg.local_load_packed_transposed %buffer : !ttg.memdesc<16x64xi8, #shared, #smem, mutable> -> tensor<32x32xi8, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>>
    tt.return
  }
}

// -----

// An asynchronous LDS-to-global store is an outstanding shared-memory read;
// a following write must check it and the native wait must publish both sides.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4096 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @async_copy_local_to_global_tracks_reads
  tt.func public @async_copy_local_to_global_tracks_reads(%dst: tensor<32x32x!tt.ptr<f32>, #blocked>, %value: tensor<32x32xf32, #blocked>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: amdg.async_copy_local_to_global
    %token = amdg.async_copy_local_to_global %buffer, %dst : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: ttg.async_commit_group
    %group = ttg.async_commit_group tokens %token
    // CHECK: tt.call @__triton_consan_check_outstanding_commits
    // CHECK: "Accessing buffer with pending access. Pending access type: async_copy_shared_to_global"
    // CHECK: ttg.local_store
    ttg.local_store %value, %buffer : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK-NOT: tt.call @__triton_consan_clear_outstanding_commits_transfer_writes
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_both
    // CHECK-NOT: tt.call @__triton_consan_clear_outstanding_commits_transfer_
    // CHECK: ttg.async_wait
    %wait = ttg.async_wait %group {num = 0 : i32}
    tt.return
  }
}

// -----

// Lowered AMD waits retain the same read-and-write completion behavior.
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 4096 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @lowered_async_wait_transfers_reads
  tt.func public @lowered_async_wait_transfers_reads(%dst: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %token = amdg.async_copy_local_to_global %buffer, %dst : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %group = ttg.async_commit_group tokens %token
    // CHECK-NOT: tt.call @__triton_consan_clear_outstanding_commits_transfer_writes
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_both
    // CHECK-NOT: tt.call @__triton_consan_clear_outstanding_commits_transfer_
    // CHECK: amdg.async_wait
    amdg.async_wait {"ttg.num_commit_groups" = 0 : i64, num_inst = 0 : i32}
    tt.return
  }
}

// -----

// TDM gather and scatter use the TDM commit class and transfer both their
// outstanding writes and reads when the TDM wait completes.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
#idx_parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65536 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @tdm_gather_scatter_track_commits
  tt.func public @tdm_gather_scatter_track_commits(%desc: !tt.tensordesc<64x128xf16>, %rows: tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: amdg.async_tdm_gather
    %gather = amdg.async_tdm_gather %desc[%rows] to %buffer : tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>, !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: amdg.async_tdm_scatter
    %scatter = amdg.async_tdm_scatter %desc[%rows] from %buffer : tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>, !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_both
    // CHECK: amdg.async_tdm_wait
    %wait = amdg.async_tdm_wait %scatter {num = 0 : i32}
    tt.return
  }
}

// -----

// Barrier-completing TDM gather and scatter publish their own accesses through
// the real barrier instead of creating implicit TDM commit groups.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#idx_parent = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 4 : i32} {
  // CHECK-LABEL: @tdm_gather_scatter_track_barrier
  tt.func public @tdm_gather_scatter_track_barrier(%desc: !tt.tensordesc<64x128xf16>, %rows: tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>) {
    %buffer = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    amdg.init_barrier %barrier, 8 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    // CHECK-NOT: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: amdg.async_tdm_gather
    %gather = amdg.async_tdm_gather %desc[%rows] to %buffer, barrier = %barrier : tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>, !ttg.memdesc<256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    // CHECK: tt.call @__triton_consan_verify_and_update_barrier_state
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier used before initialization or after invalidation"
    // CHECK: tti.experimental_assert_uniform {{.*}}, "Barrier arrive underflow: current count or tx-count would become invalid"
    // CHECK-NOT: tt.call @__triton_consan_update_barrier_state
    // CHECK-NOT: tt.call @__triton_consan_update_outstanding_commits
    // CHECK: amdg.async_tdm_scatter
    %scatter = amdg.async_tdm_scatter %desc[%rows] from %buffer, barrier = %barrier : tensor<64xi32, #ttg.slice<{dim = 0, parent = #idx_parent}>>, !ttg.memdesc<256x128xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable> -> !tt.tensordesc<64x128xf16>
    tt.return
  }
}
