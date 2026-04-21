// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -tritoninstrument-concurrency-sanitizer | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 4, maxPhase = 4, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.shared = 65544 : i32, ttg.target = "hip:gfx1250", "ttg.threads-per-warp" = 32 : i32, "ttg.total-num-warps" = 1 : i32} {
  // CHECK-LABEL: @single_local_alloc
  tt.func public @single_local_alloc() {
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_descriptors [0], [{{.*}}], shared_mem : tensor<1x1xi64

    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1xI64(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 512 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1x64xI64(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1x1xI8(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1x1xI64(%[[READ_TRACKING_GLOB]], %c0_i64
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_descriptors [0, 4096], [{{.*}}], shared_mem : tensor<1x2xi64

    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x2xI64(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1024 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x2x64xI64(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 2 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x2x1xI8(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x2x1xI64(%[[READ_TRACKING_GLOB]], %c0_i64
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_descriptors [0, 4096, 8192, 0], [{{.*}}], shared_mem : tensor<1x4xi64,

    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x4xI64(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 2048 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x4x64xI64(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x4x1xI8(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x4x1xI64(%[[READ_TRACKING_GLOB]], %c0_i64
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_descriptors [0, 4096, 8192, 0], [{{.*}}], shared_mem : tensor<1x4xi64,

    // CHECK: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x4xI64(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 2048 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x4x64xI64(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x4x1xI8(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 32 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x4x1xI64(%[[READ_TRACKING_GLOB]], %c0_i64
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
    // CHECK: %[[READ_BARS_G:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x2x4xI8(%[[READ_BARS_G]], %c0_i8
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
    // CHECK: %[[BUFFERS:.*]] = tti.experimental_buffer_descriptors [0], [{{.*}}], shared_mem : tensor<1x1xi64
    // CHECK: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1x16xI8(%[[WRT_COMMITS_GLOB]], %c0_i8

    // CHECK: %[[A_I64:.*]] = tti.experimental_memdesc_to_i32 %[[A:.*]] :
    // CHECK: tt.call @__triton_consan_verify_write_visibility_noalias_nw1{{.*}}(%[[A_I64]]
    // CHECK: %[[A_I64:.*]] = tti.experimental_memdesc_to_i32 %[[A]] :
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_check_outstanding_commits{{.*}}(%[[A_I64]], {{.*}}, %[[THREAD_BIT]], %[[BUFFERS]], %[[WRT_COMMITS_GLOB]]
    // CHECK: tt.call @__triton_consan_verify_read_visibility_noalias_nw1
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[A_I64:.*]] = tti.experimental_memdesc_to_i32 %[[A]] :
    // CHECK: tt.call @__triton_consan_stage_access_for_commit_nw1{{.*}}(%[[A_I64]], {{.*}}, %[[THREAD_BIT]], %[[BUFFERS]], %[[WRT_COMMITS_GLOB]]
    // CHECK: ttg.async_copy_global_to_local %{{.*}}, %[[A]]

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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_descriptors [0], [{{.*}}], shared_mem : tensor<1x1xi64
    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 512 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i8>
    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>

    // CHECK-DAG: %[[WRT_COMMITS_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 16 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i8>

    // CHECK: tt.call @__triton_consan_init_barrier_state

    // CHECK: %[[A_I64:.*]] = tti.experimental_memdesc_to_i32 %[[A:.*]] :
    // CHECK: tt.call @__triton_consan_verify_write_visibility_noalias{{.*}}(%[[A_I64]]
    // CHECK: %[[A_I64:.*]] = tti.experimental_memdesc_to_i32 %[[A]] :
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: tt.call @__triton_consan_check_outstanding_commits{{.*}}(%[[A_I64]], {{.*}}, %[[THREAD_BIT]], %[[BUFFERS]], %[[WRT_COMMITS_GLOB]]
    // CHECK: %[[A_I64:.*]] = tti.experimental_memdesc_to_i32 %[[A]] :
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}(%[[A_I64]]
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[A_I64:.*]] = tti.experimental_memdesc_to_i32 %[[A]] :
    // CHECK: tt.call @__triton_consan_stage_access_for_commit{{.*}}(%[[A_I64]], {{.*}}, %[[THREAD_BIT]], %[[BUFFERS]], %[[WRT_COMMITS_GLOB]]
    // CHECK: ttg.async_copy_global_to_local %{{.*}}, %[[A]]
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
    // CHECK: tt.call @__triton_consan_commit_accesses
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
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 281479271743489 : i64
    // CHECK: %[[OUTSTANDING_NUM:.*]] = arith.constant 42 : i32
    // CHECK: tt.call @__triton_consan_clear_outstanding_commits_transfer_writes{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]], %[[OUTSTANDING_NUM]]
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
    // CHECK: %[[BUF_I64:.*]] = tti.experimental_memdesc_to_i32 %[[BUF:.*]] :
    // CHECK: tt.call @__triton_consan_verify_write_visibility{{.*}}(%[[BUF_I64]]
    // CHECK: %[[BUF_I64:.*]] = tti.experimental_memdesc_to_i32 %[[BUF:.*]] :
    // CHECK: tt.call @__triton_consan_verify_read_visibility{{.*}}(%[[BUF_I64]]
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
    // CHECK-DAG: tti.experimental_buffer_descriptors [0, 16], [128, 128], shared_mem : tensor<1x2xi64
    // CHECK-DAG: arith.constant dense<true> : tensor<2x2xi1
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
    // CHECK-DAG: tti.experimental_buffer_descriptors [0, 128], [128, 128], shared_mem : tensor<1x2xi64
    // CHECK-NOT: arith.constant dense<{{\[\[true, false\], \[false, true\]\]}}> : tensor<2x2xi1
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
    // CHECK-DAG: tti.experimental_buffer_descriptors [0, 128], [256, 128], shared_mem : tensor<1x2xi64
    // CHECK-DAG: arith.constant dense<true> : tensor<2x2xi1
    %buf0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<64xf32, #shared, #smem, mutable>
    %buf1 = ttg.memdesc_subslice %buf0 [32] : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttg.local_load %buf0 : !ttg.memdesc<64xf32, #shared, #smem, mutable> -> tensor<64xf32>
    ttg.local_load %buf1 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_descriptors [0], [{{.*}}], shared_mem : tensor<1x1xi64

    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1xI64(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 512 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1x64xI64(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1x1xi64

    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1x1xI8(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1x1xI64(%[[READ_TRACKING_GLOB]], %c0_i64
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK-DAG: tt.call @__triton_consan_set_waiting
    // CHECK-DAG: tt.call @__triton_consan_check_all_active_waiting
    // CHECK: amdg.wait_barrier
    amdg.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_transfer_visible_writes{{.*}}%[[BARRIERS]], %[[WRITE_VISIBILITY_GLOB]], %[[WRITE_TRACKING_GLOB]]
    // CHECK: tt.call @__triton_consan_transfer_visible_reads{{.*}}%[[BARRIERS]], %[[READ_VISIBILITY_GLOB]], %[[READ_TRACKING_GLOB]]
    // CHECK: tt.call @__triton_consan_clear_waiting
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
  // CHECK-LABEL: @amdg_arrive_barrier
  tt.func public @amdg_arrive_barrier() {
    // CHECK-DAG: %[[BSTATE_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 4 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i32>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1xI32(%[[BSTATE_GLOB]], %c0_i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    // CHECK: tti.experimental_lock_acquire
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK: tt.call @__triton_consan_track_visible_writes
    // CHECK: tt.call @__triton_consan_track_visible_reads
    // CHECK: tt.call @__triton_consan_verify_barrier_arrive
    // CHECK: tt.call @__triton_consan_update_barrier_state
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
    // CHECK: tt.call @__triton_consan_set_waiting
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
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK: tt.call @__triton_consan_verify_barrier_arrive
    // CHECK: tt.call @__triton_consan_update_barrier_state
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
    // CHECK-DAG: %[[BUFFERS:.*]] = tti.experimental_buffer_descriptors [0], [{{.*}}], shared_mem : tensor<1x1xi64

    // CHECK-DAG: %[[WRITE_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1xI64(%[[WRITE_VISIBILITY_GLOB]], %c0_i64

    // CHECK-DAG: %[[READ_VISIBILITY_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 512 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1x64xI64(%[[READ_VISIBILITY_GLOB]], %c0_i64

    // CHECK-DAG: %[[BARRIERS:.*]] = tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1x1xi64
    // CHECK-DAG: %[[WRITE_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 1 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i8>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1x1xI8(%[[WRITE_TRACKING_GLOB]], %c0_i8

    // CHECK-DAG: %[[READ_TRACKING_GLOB:.*]] = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 8 : i32, shared_cluster_state, third_party_allocation} : !tt.ptr<i64>
    // CHECK: call {{.*}}fill_global_tensor{{.*}}T1x1x1xI64(%[[READ_TRACKING_GLOB]], %c0_i64
    %c0_i32 = arith.constant 0 : i32
    %pred = arith.constant 1 : i32
    %0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_verify_barrier_can_init
    amdg.init_barrier %bar, 8 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tt.call @__triton_consan_init_barrier_state
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_set_write_visibility
    // CHECK: tt.call @__triton_consan_clear_write_tracking
    // CHECK: tt.call @__triton_consan_clear_read_visibility
    // CHECK: tt.call @__triton_consan_clear_read_tracking
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK: tt.call @__triton_consan_track_visible_writes
    // CHECK: tt.call @__triton_consan_verify_barrier_arrive
    // CHECK: tt.call @__triton_consan_update_barrier_state
    %1 = amdg.async_tdm_copy_global_to_local %desc[%c0_i32, %c0_i32] into %0, pred = %pred, barrier = %bar : !tt.tensordesc<32x32xf32>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
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
    // CHECK: tt.call @__triton_consan_set_write_visibility
    // CHECK: tt.call @__triton_consan_clear_write_tracking
    // CHECK: tt.call @__triton_consan_clear_read_visibility
    // CHECK: tt.call @__triton_consan_clear_read_tracking
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK: tt.call @__triton_consan_track_visible_writes
    // CHECK: tt.call @__triton_consan_verify_barrier_arrive
    // CHECK: tt.call @__triton_consan_update_barrier_state
    %0 = amdg.async_tdm_copy_global_to_local %a[%c0_i32, %c0_i32] into %a_smem, pred = %pred, barrier = %bar : !tt.tensordesc<32x32xf32>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>

    // Second TDM copy: same full instrumentation
    // CHECK: tt.call @__triton_consan_verify_write_visibility
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_set_write_visibility
    // CHECK: tt.call @__triton_consan_clear_write_tracking
    // CHECK: tt.call @__triton_consan_clear_read_visibility
    // CHECK: tt.call @__triton_consan_clear_read_tracking
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK: tt.call @__triton_consan_track_visible_writes
    // CHECK: tt.call @__triton_consan_verify_barrier_arrive
    // CHECK: tt.call @__triton_consan_update_barrier_state
    %1 = amdg.async_tdm_copy_global_to_local %b[%c0_i32, %c0_i32] into %b_smem, pred = %pred, barrier = %bar : !tt.tensordesc<32x32xf32>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>

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
    // CHECK: tt.call @__triton_consan_check_outstanding_commits_excl_self_noalias
    // CHECK: tt.call @__triton_consan_verify_read_visibility
    // CHECK: tt.call @__triton_consan_check_outstanding_commits_excl_self_noalias
    // CHECK: tt.call @__triton_consan_stage_access_for_commit
    // CHECK: tt.call @__triton_consan_commit_accesses
    // CHECK-NOT: tt.call @__triton_consan_verify_barrier_arrive
    %1 = amdg.async_tdm_copy_global_to_local %desc[%c0_i32, %c0_i32] into %0, pred = %pred : !tt.tensordesc<32x32xf32> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
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
    // CHECK: tt.call @__triton_consan_check_outstanding_commits_noalias
    // CHECK: tt.call @__triton_consan_check_outstanding_commits_excl_self_noalias
    // CHECK: tt.call @__triton_consan_stage_access_for_commit
    // CHECK: tt.call @__triton_consan_commit_accesses
    amdg.async_tdm_copy_local_to_global %desc[%c0_i32, %c0_i32] from %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> !tt.tensordesc<32x32xf32>
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
    // CHECK: tt.call @__triton_consan_check_outstanding_commits_excl_self_noalias
    // CHECK: tt.call @__triton_consan_stage_access_for_commit
    // CHECK: tt.call @__triton_consan_commit_accesses
    %1 = amdg.async_tdm_copy_global_to_local %in_desc[%c0_i32, %c0_i32] into %0, pred = %pred : !tt.tensordesc<32x32xf32> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK: tt.call @__triton_consan_check_outstanding_commits_excl_self_noalias
    // CHECK: tt.call @__triton_consan_stage_access_for_commit
    // CHECK: tt.call @__triton_consan_commit_accesses
    amdg.async_tdm_copy_local_to_global %out_desc[%c0_i32, %c0_i32] from %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> !tt.tensordesc<32x32xf32>
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
    // CHECK: tt.call @__triton_consan_verify_barrier_initialized
    // CHECK: tt.call @__triton_consan_track_visible_writes
    // CHECK: tt.call @__triton_consan_track_visible_reads
    // CHECK: tt.call @__triton_consan_verify_barrier_arrive
    // CHECK: tt.call @__triton_consan_update_barrier_state
    // CHECK-NOT: tt.call @__triton_consan_stage_access_for_commit
    amdg.async_tdm_copy_local_to_global %desc[%c0_i32, %c0_i32] from %0, barrier = %bar : !ttg.memdesc<32x32xf32, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared1, #smem, mutable> -> !tt.tensordesc<32x32xf32>
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
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 281479271743489 : i64
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
    %1 = amdg.async_tdm_copy_global_to_local %desc[%c0_i32, %c0_i32] into %0, pred = %pred : !tt.tensordesc<32x32xf32> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
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
    amdg.async_tdm_copy_local_to_global %desc[%c0_i32, %c0_i32] from %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> !tt.tensordesc<32x32xf32>
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
    %1 = amdg.async_tdm_copy_global_to_local %desc[%c0_i32, %c0_i32] into %0, pred = %pred : !tt.tensordesc<32x32xf32> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    amdg.async_tdm_copy_local_to_global %desc[%c0_i32, %c0_i32] from %0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> !tt.tensordesc<32x32xf32>
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
    // CHECK-DAG: tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1x1xi64,
    // CHECK-DAG: tti.experimental_buffer_descriptors [0], [{{.*}}], shared_mem : tensor<1x1xi64
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 562958543486978 : i64
    // CHECK: tt.call @__triton_consan_copy_write_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 562958543486978 : i64
    // CHECK: tt.call @__triton_consan_copy_read_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
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
      // CHECK-DAG: tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem : tensor<1x1xi64,
      // CHECK-DAG: tti.experimental_buffer_descriptors [0], [{{.*}}], shared_mem : tensor<1x1xi64
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
    // CHECK-DAG: tti.experimental_buffer_descriptors [0, {{.*}}], [{{.*}}], shared_mem
    // CHECK-DAG: tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 562958543486978 : i64
    // CHECK: tt.call @__triton_consan_copy_write_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 562958543486978 : i64
    // CHECK: tt.call @__triton_consan_copy_read_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
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
    // CHECK-DAG: tti.experimental_buffer_descriptors [0, {{.*}}], [{{.*}}], shared_mem
    // CHECK-DAG: tti.experimental_buffer_descriptors [65536], [{{.*}}], shared_mem
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<3x128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    // CHECK: tti.experimental_lock_acquire
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 562958543486978 : i64
    // CHECK: tt.call @__triton_consan_copy_write_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
    // CHECK: %[[THREAD_BIT:.*]] = arith.constant 0 : i32
    // CHECK: %[[THREAD_MASK:.*]] = arith.constant 562958543486978 : i64
    // CHECK: tt.call @__triton_consan_copy_read_visibility{{.*}}(%[[THREAD_BIT]], %[[THREAD_MASK]]
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
  // CHECK-LABEL: @ws_wait_barrier
  tt.func public @ws_wait_barrier() {
    %smem = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc {allocation.offset = 65536 : i32} : () -> !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    amdg.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
    ttg.warp_specialize(%smem, %bar) attributes {actualRegisters = array<i32: 480, 32>, allocation.offset = 512 : i32, requestedRegisters = array<i32: 32>, warpGroupStartIds = array<i32: 4>}
    default {
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_set_waiting
      // CHECK: %[[ACTIVE_MASK:.*]] = arith.constant 5 : i32
      // CHECK: tt.call @__triton_consan_check_all_active_waiting{{.*}}(%[[ACTIVE_MASK]], {{.*}}, {{.*}}, {{.*}})
      // CHECK: tti.experimental_lock_release
      %c0_i32 = arith.constant 0 : i32
      amdg.wait_barrier %bar, %c0_i32 : !ttg.memdesc<1xi64, #shared1, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%arg1: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %arg2: !ttg.memdesc<1xi64, #shared1, #smem, mutable>) num_warps(4) {
      // CHECK: partition0
      // CHECK: tti.experimental_lock_acquire
      // CHECK: tt.call @__triton_consan_set_waiting
      // CHECK: %[[ACTIVE_MASK:.*]] = arith.constant 5 : i32
      // CHECK: tt.call @__triton_consan_check_all_active_waiting{{.*}}(%[[ACTIVE_MASK]], {{.*}}, {{.*}}, {{.*}})
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
    // We expect the alias matrix constant to appear once for the default region
    // and once for partition0 when we lower warp_specialize.
    // CHECK-DAG: arith.constant dense<true> : tensor<2x2xi1
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
      // CHECK: arith.constant dense<true> : tensor<2x2xi1
      %c0 = arith.constant 0 : i32
      ttg.local_load %arg0 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
      ttg.local_load %arg1 : !ttg.memdesc<32xf32, #shared, #smem, mutable> -> tensor<32xf32>
      ttg.warp_return
    } : (!ttg.memdesc<32xf32, #shared, #smem, mutable>, !ttg.memdesc<32xf32, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
