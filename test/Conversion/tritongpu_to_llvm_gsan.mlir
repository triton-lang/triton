// RUN: triton-opt %s -split-input-file --set-minimum-shared-memory='minimum-size=123456' | FileCheck %s --check-prefix=CHECK-SHARED
// RUN: triton-opt %s -split-input-file -tritoninstrument-global-sanitizer --allocate-shared-memory-nv --convert-triton-gpu-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-SHARED: module attributes {
  // CHECK-SHARED-DAG: ttg.shared = 123456 : i32
  // CHECK-LABEL: llvm.func @load_store
  // CHECK: llvm.call @__triton_gsan_init({{.*}}) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, i32, i32, !llvm.ptr, i32) -> ()
  // CHECK: nvvm.barrier
  // CHECK: llvm.store %{{.*}} : i64, !llvm.ptr
  // CHECK: llvm.store %{{.*}} : i8, !llvm.ptr
  // CHECK: llvm.call @__triton_gsan_load_tensor(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr, i32) -> ()
  // CHECK-2: ld.global
  // CHECK: llvm.call @__triton_gsan_store_tensor(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr, i32) -> ()
  // CHECK-2: st.global
  tt.func @load_store(%ptrs: tensor<256x!tt.ptr<f32>, #blocked>, %mask: tensor<256xi1, #blocked>,
                      %other: tensor<256xf32, #blocked>, %vals: tensor<256xf32, #blocked>) {
    %loaded = tt.load %ptrs, %mask, %other : tensor<256x!tt.ptr<f32>, #blocked>
    tt.store %ptrs, %vals, %mask : tensor<256x!tt.ptr<f32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: llvm.func @unmasked_store
  // CHECK: llvm.call @__triton_gsan_store_tensor(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32, !llvm.ptr, i32) -> ()
  tt.func @unmasked_store(%ptrs: tensor<128x!tt.ptr<i32>, #blocked>, %vals: tensor<128xi32, #blocked>) {
    tt.store %ptrs, %vals : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // CHECK-LABEL: llvm.func @unmasked_atomic_add
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  tt.func @unmasked_atomic_add(%ptr: !tt.ptr<i32>, %val: i32) {
    %0 = tt.atomic_rmw add, relaxed, gpu, %ptr, %val : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }

  // CHECK-LABEL: llvm.func @atomic_poll
  // CHECK: llvm.load %{{.*}} atomic monotonic
  // CHECK: llvm.fence acquire
  // CHECK: nvvm.barrier
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  // CHECK: nvvm.barrier
  tt.func @atomic_poll(%ptr: !tt.ptr<i32>, %expected: i32) {
    %matched = tt.atomic_poll acquire, sys, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    tt.return
  }
}

// -----

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.profile_scratch_memory_size" = 128 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: llvm.func @cluster_barrier
  // CHECK: %[[NUM_CTAS:.*]] = llvm.mlir.constant(2 : i64) : i64
  // CHECK: %[[CLUSTER_BASE:.*]] = llvm.mul %{{.*}}, %[[NUM_CTAS]] : i64
  // CHECK: %[[PROFILE_BYTES:.*]] = llvm.mlir.constant(128 : i64) : i64
  // CHECK: %{{.*}} = llvm.mul %[[CLUSTER_BASE]], %[[PROFILE_BYTES]] : i64
  // CHECK: %[[SCRATCH:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i8
  // CHECK: %[[ELECT_INIT:.*]] = nvvm.elect.sync -> i1
  // CHECK: %[[CTA_RANK_INIT:.*]] = nvg.cluster_id
  // CHECK: %[[INIT_SCRATCH:.*]] = llvm.addrspacecast %[[SCRATCH]] : !llvm.ptr<1> to !llvm.ptr
  // CHECK: llvm.call @__triton_gsan_cluster_barrier_init(%[[INIT_SCRATCH]], %{{.*}}) : (!llvm.ptr, i32) -> ()
  // CHECK: %[[ELECT_SYNC:.*]] = nvvm.elect.sync -> i1
  // CHECK: %[[CTA_RANK_SYNC:.*]] = nvg.cluster_id
  // CHECK: %[[SYNC_SCRATCH:.*]] = llvm.addrspacecast %[[SCRATCH]] : !llvm.ptr<1> to !llvm.ptr
  // CHECK: %[[TWO:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: llvm.call @__triton_gsan_cluster_barrier_sync(%{{.*}}, %[[SYNC_SCRATCH]], %{{.*}}, %[[TWO]], %[[CTA_RANK_SYNC]], %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, !llvm.ptr, i32) -> ()
  tt.func @cluster_barrier() {
    %scratch = ttg.global_scratch_alloc {alignment = 16 : i32, nbytes = 128 : i32, shared_cluster_state, third_party_allocation, ttg.global_scratch_memory_offset = 0 : i32} : !tt.ptr<i8>
    tti.experimental_gsan_cluster_barrier_init %scratch : <i8>
    tti.experimental_gsan_cluster_barrier_sync %scratch : <i8>
    tt.return
  }
}

// -----

#shared_f16 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @tma_f16_gsan_merge
  tt.func @tma_f16_gsan_merge(%desc: !tt.tensordesc<32x64xf16, #shared_f16>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<32x64xf16, #shared_f16, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 4096 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: llvm.alloca %{{.*}} x !llvm.struct<(array<32 x i64>, array<32 x i8>)>
    // CHECK: %[[COUNT:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[BYTES:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: llvm.call @__triton_gsan_load_tensor(%{{.*}}, %{{.*}}, %[[COUNT]], %[[BYTES]], %{{.*}}, %{{.*}})
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<32x64xf16, #shared_f16>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<32x64xf16, #shared_f16, #smem, mutable>
    tt.return
  }
}

// -----

#shared_f16 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @tma_f16_gsan_merge_4warps
  tt.func @tma_f16_gsan_merge_4warps(%desc: !tt.tensordesc<128x64xf16, #shared_f16>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %buf = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<128x64xf16, #shared_f16, #smem, mutable>
    %barrier = ttg.local_alloc {allocation.offset = 16384 : i32} : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: llvm.alloca %{{.*}} x !llvm.struct<(array<32 x i64>, array<32 x i8>)>
    // CHECK: %[[COUNT_4W:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: %[[BYTES_4W:.*]] = llvm.mlir.constant(4 : i32) : i32
    // CHECK: llvm.call @__triton_gsan_load_tensor(%{{.*}}, %{{.*}}, %[[COUNT_4W]], %[[BYTES_4W]], %{{.*}}, %{{.*}})
    ttng.async_tma_copy_global_to_local %desc[%c0_i32, %c0_i32] %buf, %barrier, %true : !tt.tensordesc<128x64xf16, #shared_f16>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared_f16, #smem, mutable>
    tt.return
  }
}

// -----

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @gsan_atomic_ordering_one_cta
  // CHECK: llvm.call @__triton_gsan_init
  // CHECK: nvvm.barrier
  // CHECK: nvvm.barrier
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  // CHECK: nvvm.barrier
  // CHECK: nvvm.barrier
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  // CHECK: nvvm.barrier
  tt.func @gsan_atomic_ordering_one_cta(%ptr: !tt.ptr<i32>, %val: i32, %mask: i1) {
    %c0 = arith.constant 0 : i32
    %rmw = tt.atomic_rmw add, acq_rel, gpu, %ptr, %val, %mask : (!tt.ptr<i32>, i32, i1) -> i32
    %cas = tt.atomic_cas acq_rel, gpu, %ptr, %c0, %val : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: llvm.func @atomic_cluster_sync
  // CHECK: llvm.call @__triton_gsan_cluster_barrier_init
  // CHECK: llvm.call @__triton_gsan_cluster_barrier_sync
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  // CHECK: llvm.call @__triton_gsan_cluster_barrier_sync
  tt.func @atomic_cluster_sync(%ptr: !tt.ptr<i32>) {
    %one = arith.constant 1 : i32
    %release = tt.atomic_rmw add, release, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32
    %acquire = tt.atomic_rmw add, acquire, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @gsan_atomic_ordering_two_ctas
  // CHECK: llvm.call @__triton_gsan_init
  // CHECK: nvvm.barrier
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK: llvm.call @__triton_gsan_atomic_begin_scalar
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  tt.func @gsan_atomic_ordering_two_ctas(%ptr: !tt.ptr<i32>, %val: i32, %mask: i1) {
    %c0 = arith.constant 0 : i32
    %rmw = tt.atomic_rmw add, acq_rel, gpu, %ptr, %val, %mask : (!tt.ptr<i32>, i32, i1) -> i32
    %cas = tt.atomic_cas acq_rel, gpu, %ptr, %c0, %val : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: llvm.func @atomic_cluster_sync_warp_specialized
  // CHECK: llvm.call @__triton_gsan_cluster_barrier_init
  // CHECK: llvm.call @__triton_gsan_cluster_barrier_sync
  // CHECK: atom.global.gpu.release
  tt.func @atomic_cluster_sync_warp_specialized(%ptr: !tt.ptr<i32>) {
    ttg.warp_specialize()
    default {
      %one = arith.constant 1 : i32
      %release = tt.atomic_rmw add, release, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @gsan_atomic_broadcast_one_cta
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  // CHECK: st.shared
  // CHECK: nvvm.barrier
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3>
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  // CHECK: st.shared
  // CHECK: nvvm.barrier
  // CHECK: llvm.load {{.*}} : !llvm.ptr<3>
  tt.func @gsan_atomic_broadcast_one_cta(%ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>, %val: i32, %mask: i1) {
    %c0 = arith.constant 0 : i32
    %rmw = tt.atomic_rmw add, relaxed, gpu, %ptr, %val, %mask : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %rmw : !tt.ptr<i32>
    %cas = tt.atomic_cas relaxed, gpu, %ptr, %c0, %val : (!tt.ptr<i32>, i32, i32) -> i32
    tt.store %out, %cas : !tt.ptr<i32>
    tt.return
  }
}

// -----

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @gsan_atomic_broadcast_two_ctas
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  // CHECK: st.shared
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK: nvvm.mapa
  // CHECK: llvm.load {{.*}} : !llvm.ptr<7>
  // CHECK: llvm.call @__triton_gsan_atomic_end_scalar
  // CHECK: st.shared
  // CHECK: nvvm.cluster.arrive
  // CHECK-NEXT: nvvm.cluster.wait
  // CHECK: nvvm.mapa
  // CHECK: llvm.load {{.*}} : !llvm.ptr<7>
  tt.func @gsan_atomic_broadcast_two_ctas(%ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>, %val: i32, %mask: i1) {
    %c0 = arith.constant 0 : i32
    %rmw = tt.atomic_rmw add, relaxed, gpu, %ptr, %val, %mask : (!tt.ptr<i32>, i32, i1) -> i32
    tt.store %out, %rmw : !tt.ptr<i32>
    %cas = tt.atomic_cas relaxed, gpu, %ptr, %c0, %val : (!tt.ptr<i32>, i32, i32) -> i32
    tt.store %out, %cas : !tt.ptr<i32>
    tt.return
  }
}

// -----

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 1 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @gsan_relaxed_dead_atomic_results
  // CHECK: llvm.call @__triton_gsan_init
  // CHECK: nvvm.barrier
  // CHECK-NOT: nvvm.barrier
  // CHECK-NOT: nvvm.cluster.arrive
  // CHECK: llvm.return
  tt.func @gsan_relaxed_dead_atomic_results(%ptr: !tt.ptr<i32>, %val: i32, %mask: i1) {
    %c0 = arith.constant 0 : i32
    %rmw = tt.atomic_rmw add, relaxed, gpu, %ptr, %val, %mask : (!tt.ptr<i32>, i32, i1) -> i32
    %cas = tt.atomic_cas relaxed, gpu, %ptr, %c0, %val : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: llvm.func @relaxed_atomic_has_no_cluster_sync
  // CHECK-NOT: __triton_gsan_cluster_barrier
  // CHECK: llvm.return
  tt.func @relaxed_atomic_has_no_cluster_sync(%ptr: !tt.ptr<i32>) {
    %one = arith.constant 1 : i32
    %relaxed = tt.atomic_rmw add, relaxed, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32
    tt.return
  }
}

// -----

#blockedSplitM = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[1, 0]]}>
#blockedSplitN = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1]]}>

module attributes {"ttg.instrumentation_mode" = "gsan", "ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: llvm.func @convert_layout_cluster_sync
  // CHECK: llvm.call @__triton_gsan_cluster_barrier_init
  // CHECK: llvm.call @__triton_gsan_cluster_barrier_sync
  tt.func @convert_layout_cluster_sync() {
    %value = arith.constant dense<0.000000e+00> : tensor<256x128xf16, #blockedSplitM>
    %converted = ttg.convert_layout %value : tensor<256x128xf16, #blockedSplitM> -> tensor<256x128xf16, #blockedSplitN>
    tt.return
  }
}
