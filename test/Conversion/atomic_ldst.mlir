// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 2>&1 | FileCheck %s --check-prefixes=CHECK-TTG2NVGPU,CHECK-COMMON
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 --convert-nv-gpu-to-llvm 2>&1 | FileCheck %s --check-prefixes=CHECK-NVGPU2LLVM,CHECK-COMMON
#blocked4 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_r(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = arith.cmpi slt, %1, %c512_i32 : i32

    // CHECK-TTG2NVGPU: nvg.ld_acquire acquire, gpu
    // CHECK-NVGPU2LLVM: ld.global.gpu.acquire.b32
    %3 = tt.atomic_rmw fadd, acquire, gpu, %arg0, %cst, %2 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %arg0, %3 : !tt.ptr<f32>

    // CHECK-TTG2NVGPU: nvg.ld_acquire acquire, cta
    // CHECK-NVGPU2LLVM: ld.global.cta.acquire.b32
    %4 = tt.atomic_rmw fadd, acquire, cta, %arg0, %cst, %true : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %arg0, %4 : !tt.ptr<f32>

    // CHECK-TTG2NVGPU: nvg.ld_acquire acquire, sys
    // CHECK-NVGPU2LLVM: ld.global.sys.acquire.b32
    %5 = tt.atomic_rmw fadd, acquire, sys, %arg0, %cst, %2 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %arg0, %5 : !tt.ptr<f32>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @atomic_load_store
  // CHECK-TTG2NVGPU: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-TTG2NVGPU-NOT: nvvm.barrier
  // CHECK-TTG2NVGPU: llvm.fence syncscope("device") acquire
  // CHECK-TTG2NVGPU: nvvm.barrier
  // CHECK-TTG2NVGPU: llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CHECK-TTG2NVGPU: llvm.fence release
  // CHECK-TTG2NVGPU: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  // CHECK-NVGPU2LLVM-LABEL: @atomic_load_store
  // CHECK-NVGPU2LLVM: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-NVGPU2LLVM-NOT: nvvm.barrier
  // CHECK-NVGPU2LLVM: llvm.fence syncscope("device") acquire
  // CHECK-NVGPU2LLVM: nvvm.barrier
  // CHECK-NVGPU2LLVM: llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CHECK-NVGPU2LLVM: llvm.fence release
  // CHECK-NVGPU2LLVM: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  tt.func public @atomic_load_store(%ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>, %mask: i1) {
    %loaded = tt.atomic_load acquire, gpu, %ptr, %mask : (!tt.ptr<i32>, i1) -> i32
    tt.atomic_store release, sys, %out, %loaded, %mask : !tt.ptr<i32>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @atomic_load_store_relaxed
  // CHECK-TTG2NVGPU: llvm.load %{{.*}} atomic syncscope("block") monotonic
  // CHECK-TTG2NVGPU: llvm.store %{{.*}}, %{{.*}} atomic syncscope("device") monotonic
  tt.func public @atomic_load_store_relaxed(%ptr: !tt.ptr<i64>, %value: i64) {
    %loaded = tt.atomic_load relaxed, cta, %ptr : (!tt.ptr<i64>) -> i64
    tt.atomic_store relaxed, gpu, %ptr, %value : !tt.ptr<i64>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @unused_replicated_atomic_load
  // CHECK-TTG2NVGPU: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-TTG2NVGPU: llvm.return
  // CHECK-NVGPU2LLVM-LABEL: @unused_replicated_atomic_load
  // CHECK-NVGPU2LLVM: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-NVGPU2LLVM: llvm.return
  tt.func public @unused_replicated_atomic_load(%ptrs: tensor<1x!tt.ptr<i32>, #blocked4>) {
    %loaded = tt.atomic_load relaxed, gpu, %ptrs : (tensor<1x!tt.ptr<i32>, #blocked4>) -> tensor<1xi32, #blocked4>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @unpredicated_tensor_atomic_load_store
  // CHECK-TTG2NVGPU: %{{.*}} = llvm.mlir.undef : i32
  // CHECK-TTG2NVGPU-NEXT: %{{.*}} = llvm.load %{{.*}} atomic monotonic
  // CHECK-TTG2NVGPU-NEXT: %{{.*}} = llvm.mlir.undef : i32
  // CHECK-TTG2NVGPU-NEXT: %{{.*}} = llvm.load %{{.*}} atomic monotonic
  // CHECK-TTG2NVGPU-NEXT: %{{.*}} = llvm.mlir.undef : i32
  // CHECK-TTG2NVGPU-NEXT: %{{.*}} = llvm.load %{{.*}} atomic monotonic
  // CHECK-TTG2NVGPU-NEXT: %{{.*}} = llvm.mlir.undef : i32
  // CHECK-TTG2NVGPU-NEXT: %{{.*}} = llvm.load %{{.*}} atomic monotonic
  // CHECK-TTG2NVGPU: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  // CHECK-TTG2NVGPU-NEXT: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  // CHECK-TTG2NVGPU-NEXT: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  // CHECK-TTG2NVGPU-NEXT: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  // CHECK-TTG2NVGPU-NEXT: llvm.return
  // CHECK-NVGPU2LLVM-LABEL: @unpredicated_tensor_atomic_load_store
  // CHECK-NVGPU2LLVM: %{{.*}} = llvm.load %{{.*}} atomic monotonic
  // CHECK-NVGPU2LLVM-NEXT: %{{.*}} = llvm.load %{{.*}} atomic monotonic
  // CHECK-NVGPU2LLVM-NEXT: %{{.*}} = llvm.load %{{.*}} atomic monotonic
  // CHECK-NVGPU2LLVM-NEXT: %{{.*}} = llvm.load %{{.*}} atomic monotonic
  // CHECK-NVGPU2LLVM: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  // CHECK-NVGPU2LLVM-NEXT: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  // CHECK-NVGPU2LLVM-NEXT: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  // CHECK-NVGPU2LLVM-NEXT: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  // CHECK-NVGPU2LLVM-NEXT: llvm.return
  tt.func public @unpredicated_tensor_atomic_load_store(
      %ptrs: tensor<512x!tt.ptr<i32>, #blocked4>) {
    %loaded = tt.atomic_load relaxed, sys, %ptrs : (tensor<512x!tt.ptr<i32>, #blocked4>) -> tensor<512xi32, #blocked4>
    tt.atomic_store relaxed, sys, %ptrs, %loaded : tensor<512x!tt.ptr<i32>, #blocked4>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @sharded_atomic_load_acquire
  // CHECK-TTG2NVGPU-COUNT-4: llvm.load %{{.*}} atomic monotonic
  // CHECK-TTG2NVGPU-NOT: nvvm.barrier
  // CHECK-TTG2NVGPU: llvm.fence acquire
  // CHECK-TTG2NVGPU: nvvm.barrier
  // CHECK-TTG2NVGPU-NEXT: llvm.return
  // CHECK-NVGPU2LLVM-LABEL: @sharded_atomic_load_acquire
  // CHECK-NVGPU2LLVM-COUNT-4: llvm.load %{{.*}} atomic monotonic
  // CHECK-NVGPU2LLVM-NOT: nvvm.barrier
  // CHECK-NVGPU2LLVM: llvm.fence acquire
  // CHECK-NVGPU2LLVM-NEXT: nvvm.barrier
  // CHECK-NVGPU2LLVM-NEXT: llvm.return
  tt.func public @sharded_atomic_load_acquire(
      %ptrs: tensor<512x!tt.ptr<i32>, #blocked4>,
      %mask: tensor<512xi1, #blocked4>) {
    %loaded = tt.atomic_load acquire, sys, %ptrs, %mask : (tensor<512x!tt.ptr<i32>, #blocked4>, tensor<512xi1, #blocked4>) -> tensor<512xi32, #blocked4>
    tt.return
  }

  // CHECK-COMMON-LABEL: @replicated_atomic_load_acquire
  // CHECK-COMMON-COUNT-4: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-COMMON-NOT: nvvm.barrier
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.load %{{.*}} : !llvm.ptr<3>
  // CHECK-COMMON-NOT: llvm.fence
  // CHECK-COMMON: llvm.return
  tt.func public @replicated_atomic_load_acquire(
      %ptrs: tensor<16x!tt.ptr<i32>, #blocked4>,
      %out: tensor<16x!tt.ptr<i32>, #blocked4>) {
    %loaded = tt.atomic_load acquire, gpu, %ptrs : (tensor<16x!tt.ptr<i32>, #blocked4>) -> tensor<16xi32, #blocked4>
    tt.store %out, %loaded : tensor<16x!tt.ptr<i32>, #blocked4>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @sharded_atomic_store_release
  // CHECK-TTG2NVGPU: nvvm.barrier
  // CHECK-TTG2NVGPU: llvm.fence release
  // CHECK-TTG2NVGPU-COUNT-4: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  // CHECK-NVGPU2LLVM-LABEL: @sharded_atomic_store_release
  // CHECK-NVGPU2LLVM: llvm.fence release
  // CHECK-NVGPU2LLVM-COUNT-4: llvm.store %{{.*}}, %{{.*}} atomic monotonic
  tt.func public @sharded_atomic_store_release(
      %ptrs: tensor<512x!tt.ptr<i32>, #blocked4>,
      %values: tensor<512xi32, #blocked4>,
      %mask: tensor<512xi1, #blocked4>) {
    tt.atomic_store release, sys, %ptrs, %values, %mask : tensor<512x!tt.ptr<i32>, #blocked4>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_poll
  // CHECK-COMMON: nvvm.read.ptx.sreg.tid.x
  // CHECK-COMMON: %[[START:.*]] = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
  // CHECK-COMMON: llvm.cond_br %[[ELECTED:.*]], ^[[LOOP:bb[0-9]+]], ^[[DONE:bb[0-9]+]](%{{.*}} : i1)
  // CHECK-COMMON: ^[[LOOP]]:
  // CHECK-COMMON: %[[LOADED:.*]] = llvm.load %{{.*}} atomic monotonic
  // CHECK-COMMON: %[[MATCHED:.*]] = llvm.icmp "eq" %[[LOADED]], %{{.*}} : i32
  // CHECK-COMMON: llvm.cond_br %[[MATCHED]], ^[[SUCCESS:bb[0-9]+]], ^[[TIMEOUT:bb[0-9]+]]
  // CHECK-COMMON: ^[[SUCCESS]]:
  // CHECK-COMMON: llvm.fence acquire
  // CHECK-COMMON: llvm.br ^[[DONE]](%{{.*}} : i1)
  // CHECK-COMMON: ^[[TIMEOUT]]:
  // CHECK-COMMON: %[[NOW:.*]] = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
  // CHECK-COMMON: %[[ELAPSED:.*]] = llvm.sub %[[NOW]], %[[START]] : i64
  // CHECK-COMMON: %[[TIMED_OUT:.*]] = llvm.icmp "uge" %[[ELAPSED]], %{{.*}} : i64
  // CHECK-COMMON: llvm.cond_br %[[TIMED_OUT]], ^[[DONE]](%{{.*}} : i1), ^[[LOOP]]
  // CHECK-COMMON: ^[[DONE]](%[[RESULT:.*]]: i1):
  // CHECK-COMMON: llvm.insertelement %[[RESULT]],
  // CHECK-COMMON: llvm.inline_asm has_side_effects
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<3> -> i1
  // CHECK-COMMON-NOT: nvvm.barrier
  // CHECK-COMMON: llvm.return
  tt.func public @atomic_poll(%ptr: !tt.ptr<i32>, %expected: i32, %timeout: i64, %out: !tt.ptr<i32>) {
    %matched = tt.atomic_poll acquire, sys, %ptr, %expected timeout %timeout : !tt.ptr<i32>, i32 -> i1
    %result = arith.extui %matched : i1 to i32
    tt.store %out, %result : !tt.ptr<i32>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @atomic_poll_cta
  // CHECK-TTG2NVGPU: llvm.load %{{.*}} atomic syncscope("block") monotonic
  // CHECK-TTG2NVGPU: llvm.fence syncscope("block") acquire
  // CHECK-NVGPU2LLVM-LABEL: @atomic_poll_cta
  // CHECK-NVGPU2LLVM: llvm.load %{{.*}} atomic syncscope("block") monotonic
  // CHECK-NVGPU2LLVM: llvm.fence syncscope("block") acquire
  tt.func public @atomic_poll_cta(%ptr: !tt.ptr<i32>, %expected: i32) {
    %matched = tt.atomic_poll acquire, cta, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @atomic_poll_relaxed
  // CHECK-TTG2NVGPU: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-TTG2NVGPU-NOT: llvm.fence
  // CHECK-NVGPU2LLVM-LABEL: @atomic_poll_relaxed
  // CHECK-NVGPU2LLVM: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-NVGPU2LLVM-NOT: llvm.fence
  tt.func public @atomic_poll_relaxed(%ptr: !tt.ptr<i32>, %expected: i32) {
    %matched = tt.atomic_poll relaxed, gpu, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    tt.return
  }
}

// -----

#poll = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-COMMON-LABEL: @atomic_poll_tensor
  // CHECK-COMMON: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON-NOT: llvm.load
  // CHECK-COMMON: llvm.return
  tt.func public @atomic_poll_tensor(%ptr: tensor<256x!tt.ptr<i32>, #poll>, %expected: tensor<256xi32, #poll>) {
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected : tensor<256x!tt.ptr<i32>, #poll>, tensor<256xi32, #poll> -> tensor<256xi1, #poll>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_poll_tensor_timeout
  // CHECK-COMMON: llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"
  // CHECK-COMMON: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"
  // CHECK-COMMON: llvm.icmp "uge"
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON-NOT: !llvm.ptr<3>
  // CHECK-COMMON: llvm.return
  tt.func public @atomic_poll_tensor_timeout(%ptr: tensor<128x!tt.ptr<i32>, #poll>, %expected: tensor<128xi32, #poll>, %timeout: i64, %out: tensor<128x!tt.ptr<i32>, #poll>) {
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected timeout %timeout : tensor<128x!tt.ptr<i32>, #poll>, tensor<128xi32, #poll> -> tensor<128xi1, #poll>
    %result = arith.extui %matched : tensor<128xi1, #poll> to tensor<128xi32, #poll>
    tt.store %out, %result : tensor<128x!tt.ptr<i32>, #poll>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_poll_shared_timeout
  // CHECK-COMMON: %[[START:.*]] = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
  // CHECK-COMMON: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-COMMON: %[[NOW0:.*]] = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
  // CHECK-COMMON: llvm.sub %[[NOW0]], %[[START]] : i64
  // CHECK-COMMON-NOT: llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"
  // CHECK-COMMON: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-COMMON: %[[NOW1:.*]] = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
  // CHECK-COMMON: llvm.sub %[[NOW1]], %[[START]] : i64
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.return
  tt.func public @atomic_poll_shared_timeout(%ptr: tensor<256x!tt.ptr<i32>, #poll>, %expected: tensor<256xi32, #poll>, %timeout: i64) {
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected timeout %timeout : tensor<256x!tt.ptr<i32>, #poll>, tensor<256xi32, #poll> -> tensor<256xi1, #poll>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_poll_replicated_tensor
  // CHECK-COMMON: llvm.cond_br
  // CHECK-COMMON: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.return
  tt.func public @atomic_poll_replicated_tensor(%ptr: tensor<16x!tt.ptr<i32>, #poll>, %expected: tensor<16xi32, #poll>, %timeout: i64, %out: tensor<16x!tt.ptr<i32>, #poll>) {
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected timeout %timeout : tensor<16x!tt.ptr<i32>, #poll>, tensor<16xi32, #poll> -> tensor<16xi1, #poll>
    %result = arith.extui %matched : tensor<16xi1, #poll> to tensor<16xi32, #poll>
    tt.store %out, %result : tensor<16x!tt.ptr<i32>, #poll>
    tt.return
  }
}

// -----

#poll = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-COMMON-LABEL: @atomic_poll_cluster_replicas
  // CHECK-COMMON: llvm.load %{{.*}} atomic syncscope("device") monotonic
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.cluster.arrive
  // CHECK-COMMON: nvvm.cluster.wait
  // CHECK-COMMON: nvvm.mapa
  // CHECK-COMMON: llvm.load %{{.*}} {{.*}} : !llvm.ptr<7> -> i8
  // CHECK-COMMON: llvm.trunc %{{.*}} : i8 to i1
  // CHECK-COMMON: llvm.return
  tt.func public @atomic_poll_cluster_replicas(%ptr: tensor<128x!tt.ptr<i32>, #poll>, %expected: tensor<128xi32, #poll>, %timeout: i64, %out: tensor<128x!tt.ptr<i32>, #poll>) {
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected timeout %timeout : tensor<128x!tt.ptr<i32>, #poll>, tensor<128xi32, #poll> -> tensor<128xi1, #poll>
    %result = arith.extui %matched : tensor<128xi1, #poll> to tensor<128xi32, #poll>
    tt.store %out, %result : tensor<128x!tt.ptr<i32>, #poll>
    tt.return
  }
}
