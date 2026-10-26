// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 2>&1 | FileCheck %s --check-prefixes=CHECK-TTG2NVGPU,CHECK-COMMON
// RUN: triton-opt %s -split-input-file --convert-scf-to-cf --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 --convert-nv-gpu-to-llvm 2>&1 | FileCheck %s --check-prefixes=CHECK-NVGPU2LLVM,CHECK-COMMON
// RUN: triton-opt %s -split-input-file --tritongpu-optimize-thread-regions | FileCheck %s --check-prefix=THREAD-REGIONS
// RUN: triton-opt %s -split-input-file --tritongpu-optimize-thread-regions --convert-scf-to-cf --allocate-shared-memory-nv=compute-capability=90 --triton-nvidia-gpu-membar='compute-capability=90 ptx-version=80' | FileCheck %s --check-prefix=THREAD-MEMBAR
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
  // CHECK-TTG2NVGPU: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-TTG2NVGPU-NOT: nvvm.barrier
  // CHECK-TTG2NVGPU: llvm.fence syncscope("device") acquire
  // CHECK-TTG2NVGPU: nvvm.barrier
  // CHECK-TTG2NVGPU: llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CHECK-TTG2NVGPU: llvm.fence release
  // CHECK-TTG2NVGPU: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  // CHECK-NVGPU2LLVM-LABEL: @atomic_load_store
  // CHECK-NVGPU2LLVM: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-NVGPU2LLVM-NOT: nvvm.barrier
  // CHECK-NVGPU2LLVM: llvm.fence syncscope("device") acquire
  // CHECK-NVGPU2LLVM: nvvm.barrier
  // CHECK-NVGPU2LLVM: llvm.load %{{.*}} : !llvm.ptr<3> -> i32
  // CHECK-NVGPU2LLVM: llvm.fence release
  // CHECK-NVGPU2LLVM: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  tt.func public @atomic_load_store(%ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>, %mask: i1) {
    %loaded = tt.atomic_load acquire, gpu, %ptr, %mask : (!tt.ptr<i32>, i1) -> i32
    tt.atomic_store release, sys, %out, %loaded, %mask : !tt.ptr<i32>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @atomic_load_store_relaxed
  // CHECK-TTG2NVGPU: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.cta.global.b64
  // CHECK-TTG2NVGPU: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.b64
  tt.func public @atomic_load_store_relaxed(%ptr: !tt.ptr<i64>, %value: i64) {
    %loaded = tt.atomic_load relaxed, cta, %ptr : (!tt.ptr<i64>) -> i64
    tt.atomic_store relaxed, gpu, %ptr, %value : !tt.ptr<i64>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_i8
  // CHECK-COMMON: %[[BYTE:.*]] = llvm.inline_asm has_side_effects {{.*}}mov.u16 $0, 0x0;{{.*}}@$2 ld.relaxed.gpu.global.b8 $0, [ $1 + 0 ];{{.*}}"=h,l,b" {{.*}} -> i16
  // CHECK-COMMON: llvm.trunc %[[BYTE]] : i16 to i8
  tt.func public @atomic_load_i8(%ptr: !tt.ptr<i8>, %out: !tt.ptr<i8>, %mask: i1) {
    %loaded = tt.atomic_load relaxed, gpu, %ptr, %mask : (!tt.ptr<i8>, i1) -> i8
    tt.store %out, %loaded : !tt.ptr<i8>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_f16
  // CHECK-COMMON: %[[HALF:.*]] = llvm.inline_asm has_side_effects {{.*}}@$2 ld.relaxed.cta.global.b16 $0, [ $1 + 0 ];{{.*}}"=h,l,b" {{.*}} -> i16
  // CHECK-COMMON: llvm.bitcast %[[HALF]] : i16 to f16
  tt.func public @atomic_load_f16(%ptr: !tt.ptr<f16>, %out: !tt.ptr<f16>, %mask: i1) {
    %loaded = tt.atomic_load relaxed, cta, %ptr, %mask : (!tt.ptr<f16>, i1) -> f16
    tt.store %out, %loaded : !tt.ptr<f16>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_f64
  // CHECK-COMMON: %[[DOUBLE:.*]] = llvm.inline_asm has_side_effects {{.*}}@$2 ld.relaxed.sys.global.b64 $0, [ $1 + 0 ];{{.*}}"=l,l,b" {{.*}} -> i64
  // CHECK-COMMON: llvm.bitcast %[[DOUBLE]] : i64 to f64
  tt.func public @atomic_load_f64(%ptr: !tt.ptr<f64>, %out: !tt.ptr<f64>, %mask: i1) {
    %loaded = tt.atomic_load relaxed, sys, %ptr, %mask : (!tt.ptr<f64>, i1) -> f64
    tt.store %out, %loaded : !tt.ptr<f64>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @unused_replicated_atomic_load
  // CHECK-TTG2NVGPU: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-TTG2NVGPU: llvm.return
  // CHECK-NVGPU2LLVM-LABEL: @unused_replicated_atomic_load
  // CHECK-NVGPU2LLVM: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-NVGPU2LLVM: llvm.return
  tt.func public @unused_replicated_atomic_load(%ptrs: tensor<1x!tt.ptr<i32>, #blocked4>) {
    %loaded = tt.atomic_load relaxed, gpu, %ptrs : (tensor<1x!tt.ptr<i32>, #blocked4>) -> tensor<1xi32, #blocked4>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @unpredicated_tensor_atomic_load_store
  // CHECK-TTG2NVGPU: %{{.*}} = llvm.inline_asm has_side_effects {{.*}}ld.relaxed.sys.global.b32
  // CHECK-TTG2NVGPU: %{{.*}} = llvm.inline_asm has_side_effects {{.*}}ld.relaxed.sys.global.b32
  // CHECK-TTG2NVGPU: %{{.*}} = llvm.inline_asm has_side_effects {{.*}}ld.relaxed.sys.global.b32
  // CHECK-TTG2NVGPU: %{{.*}} = llvm.inline_asm has_side_effects {{.*}}ld.relaxed.sys.global.b32
  // CHECK-TTG2NVGPU: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  // CHECK-TTG2NVGPU-NEXT: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  // CHECK-TTG2NVGPU-NEXT: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  // CHECK-TTG2NVGPU-NEXT: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  // CHECK-TTG2NVGPU-NEXT: llvm.return
  // CHECK-NVGPU2LLVM-LABEL: @unpredicated_tensor_atomic_load_store
  // CHECK-NVGPU2LLVM: %{{.*}} = llvm.inline_asm has_side_effects {{.*}}ld.relaxed.sys.global.b32
  // CHECK-NVGPU2LLVM-NEXT: %{{.*}} = llvm.inline_asm has_side_effects {{.*}}ld.relaxed.sys.global.b32
  // CHECK-NVGPU2LLVM-NEXT: %{{.*}} = llvm.inline_asm has_side_effects {{.*}}ld.relaxed.sys.global.b32
  // CHECK-NVGPU2LLVM-NEXT: %{{.*}} = llvm.inline_asm has_side_effects {{.*}}ld.relaxed.sys.global.b32
  // CHECK-NVGPU2LLVM: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  // CHECK-NVGPU2LLVM-NEXT: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  // CHECK-NVGPU2LLVM-NEXT: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  // CHECK-NVGPU2LLVM-NEXT: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  // CHECK-NVGPU2LLVM-NEXT: llvm.return
  tt.func public @unpredicated_tensor_atomic_load_store(
      %ptrs: tensor<512x!tt.ptr<i32>, #blocked4>) {
    %loaded = tt.atomic_load relaxed, sys, %ptrs : (tensor<512x!tt.ptr<i32>, #blocked4>) -> tensor<512xi32, #blocked4>
    tt.atomic_store relaxed, sys, %ptrs, %loaded : tensor<512x!tt.ptr<i32>, #blocked4>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @sharded_atomic_load_acquire
  // CHECK-TTG2NVGPU-COUNT-4: llvm.inline_asm has_side_effects {{.*}}@$2 ld.relaxed.sys.global.b32
  // CHECK-TTG2NVGPU-NOT: nvvm.barrier
  // CHECK-TTG2NVGPU: llvm.fence acquire
  // CHECK-TTG2NVGPU: nvvm.barrier
  // CHECK-TTG2NVGPU-NEXT: llvm.return
  // CHECK-NVGPU2LLVM-LABEL: @sharded_atomic_load_acquire
  // CHECK-NVGPU2LLVM-COUNT-4: llvm.inline_asm has_side_effects {{.*}}@$2 ld.relaxed.sys.global.b32
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
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.v4.b32
  // CHECK-COMMON-NOT: nvvm.barrier
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.load %{{.*}} : !llvm.ptr<3>
  // CHECK-COMMON-NOT: llvm.fence
  // CHECK-COMMON: llvm.return
  tt.func public @replicated_atomic_load_acquire(
      %ptrs: tensor<16x!tt.ptr<i32>, #blocked4> {tt.contiguity = 4 : i32, tt.divisibility = 16 : i32},
      %out: tensor<16x!tt.ptr<i32>, #blocked4>) {
    %loaded = tt.atomic_load acquire, gpu, %ptrs : (tensor<16x!tt.ptr<i32>, #blocked4>) -> tensor<16xi32, #blocked4>
    tt.store %out, %loaded : tensor<16x!tt.ptr<i32>, #blocked4>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @sharded_atomic_store_release
  // CHECK-TTG2NVGPU: nvvm.barrier
  // CHECK-TTG2NVGPU: llvm.fence release
  // CHECK-TTG2NVGPU-COUNT-4: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  // CHECK-NVGPU2LLVM-LABEL: @sharded_atomic_store_release
  // CHECK-NVGPU2LLVM: llvm.fence release
  // CHECK-NVGPU2LLVM-COUNT-4: llvm.inline_asm has_side_effects {{.*}}st.relaxed.sys.global.b32
  tt.func public @sharded_atomic_store_release(
      %ptrs: tensor<512x!tt.ptr<i32>, #blocked4>,
      %values: tensor<512xi32, #blocked4>,
      %mask: tensor<512xi1, #blocked4>) {
    tt.atomic_store release, sys, %ptrs, %values, %mask : tensor<512x!tt.ptr<i32>, #blocked4>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_poll
  // CHECK-COMMON: nvvm.read.ptx.sreg.tid.x
  // CHECK-COMMON: %[[START:.*]] = nvvm.read.ptx.sreg.globaltimer : i64
  // CHECK-COMMON: llvm.br ^[[LOOP:bb[0-9]+]]
  // CHECK-COMMON: ^[[LOOP]]:
  // CHECK-COMMON: %[[LOADED:.*]] = llvm.inline_asm has_side_effects {{.*}}@$2 ld.relaxed.sys.global.b32 $0, [ $1 + 0 ];{{.*}}"=r,l,b" %{{.*}}, %[[ELECTED:.*]] : (!llvm.ptr<1>, i1) -> i32
  // CHECK-COMMON: %[[MATCHED:.*]] = llvm.icmp "eq" %[[LOADED]], %{{.*}} : i32
  // CHECK-COMMON: %[[COMPLETE:.*]] = llvm.select %[[ELECTED]], %[[MATCHED]], %{{.*}} : i1, i1
  // CHECK-COMMON: llvm.cond_br %[[COMPLETE]], ^[[SUCCESS:bb[0-9]+]], ^[[TIMEOUT:bb[0-9]+]]
  // CHECK-COMMON: ^[[SUCCESS]]:
  // CHECK-COMMON: llvm.fence acquire
  // CHECK-COMMON: llvm.br ^[[DONE:bb[0-9]+]](%[[ELECTED]] : i1)
  // CHECK-COMMON: ^[[TIMEOUT]]:
  // CHECK-COMMON: %[[NOW:.*]] = nvvm.read.ptx.sreg.globaltimer : i64
  // CHECK-COMMON: %[[ELAPSED:.*]] = llvm.sub %[[NOW]], %[[START]] : i64
  // CHECK-COMMON: %[[TIMED_OUT:.*]] = llvm.icmp "uge" %[[ELAPSED]], %{{.*}} : i64
  // CHECK-COMMON: llvm.cond_br %[[TIMED_OUT]], ^[[DONE]](%{{.*}} : i1), ^[[LOOP]]
  // CHECK-COMMON: ^[[DONE]](%[[RESULT:.*]]: i1):
  // CHECK-COMMON: llvm.insertelement %[[RESULT]],
  // CHECK-COMMON: llvm.inline_asm has_side_effects
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.load %{{.*}} {{.*}} : !llvm.ptr<3> -> i8
  // CHECK-COMMON: llvm.trunc %{{.*}} : i8 to i1
  // CHECK-COMMON-NOT: nvvm.barrier
  // CHECK-COMMON: llvm.return
  tt.func public @atomic_poll(%ptr: !tt.ptr<i32>, %expected: i32, %timeout: i64, %out: !tt.ptr<i32>) {
    %matched = tt.atomic_poll acquire, sys, %ptr, %expected timeout %timeout : !tt.ptr<i32>, i32 -> i1
    %result = arith.extui %matched : i1 to i32
    tt.store %out, %result : !tt.ptr<i32>
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @atomic_poll_cta
  // CHECK-TTG2NVGPU: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.cta.global.b32
  // CHECK-TTG2NVGPU: llvm.fence syncscope("block") acquire
  // CHECK-NVGPU2LLVM-LABEL: @atomic_poll_cta
  // CHECK-NVGPU2LLVM: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.cta.global.b32
  // CHECK-NVGPU2LLVM: llvm.fence syncscope("block") acquire
  tt.func public @atomic_poll_cta(%ptr: !tt.ptr<i32>, %expected: i32) {
    %matched = tt.atomic_poll acquire, cta, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    tt.return
  }

  // CHECK-TTG2NVGPU-LABEL: @atomic_poll_relaxed
  // CHECK-TTG2NVGPU: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-TTG2NVGPU-NOT: llvm.fence
  // CHECK-NVGPU2LLVM-LABEL: @atomic_poll_relaxed
  // CHECK-NVGPU2LLVM: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
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
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON-NOT: llvm.load
  // CHECK-COMMON: llvm.return
  tt.func public @atomic_poll_tensor(%ptr: tensor<256x!tt.ptr<i32>, #poll>, %expected: tensor<256xi32, #poll>) {
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected : tensor<256x!tt.ptr<i32>, #poll>, tensor<256xi32, #poll> -> tensor<256xi1, #poll>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_poll_tensor_timeout
  // CHECK-COMMON: nvvm.read.ptx.sreg.globaltimer
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.read.ptx.sreg.globaltimer
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
  // CHECK-COMMON: %[[START:.*]] = nvvm.read.ptx.sreg.globaltimer : i64
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-COMMON: %[[NOW0:.*]] = nvvm.read.ptx.sreg.globaltimer : i64
  // CHECK-COMMON: llvm.sub %[[NOW0]], %[[START]] : i64
  // CHECK-COMMON-NOT: nvvm.read.ptx.sreg.globaltimer
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-COMMON: %[[NOW1:.*]] = nvvm.read.ptx.sreg.globaltimer : i64
  // CHECK-COMMON: llvm.sub %[[NOW1]], %[[START]] : i64
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.return
  tt.func public @atomic_poll_shared_timeout(%ptr: tensor<256x!tt.ptr<i32>, #poll>, %expected: tensor<256xi32, #poll>, %timeout: i64) {
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected timeout %timeout : tensor<256x!tt.ptr<i32>, #poll>, tensor<256xi32, #poll> -> tensor<256xi1, #poll>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_poll_replicated_tensor
  // CHECK-COMMON: llvm.br
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
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
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.cluster.arrive
  // CHECK-COMMON: nvvm.cluster.wait
  // CHECK-COMMON: nvvm.mapa
  // CHECK-COMMON: llvm.load %{{.*}} {{.*}} : !llvm.ptr<7> -> i8
  // CHECK-COMMON: llvm.trunc %{{.*}} overflow<nuw> : i8 to i1
  // CHECK-COMMON: llvm.return
  tt.func public @atomic_poll_cluster_replicas(%ptr: tensor<128x!tt.ptr<i32>, #poll>, %expected: tensor<128xi32, #poll>, %timeout: i64, %out: tensor<128x!tt.ptr<i32>, #poll>) {
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected timeout %timeout : tensor<128x!tt.ptr<i32>, #poll>, tensor<128xi32, #poll> -> tensor<128xi1, #poll>
    %result = arith.extui %matched : tensor<128xi1, #poll> to tensor<128xi32, #poll>
    tt.store %out, %result : tensor<128x!tt.ptr<i32>, #poll>
    tt.return
  }
}

// -----

#vec = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-COMMON-LABEL: @atomic_load_store_vector_aligned
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.v4.b32
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.fence syncscope("device") release
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.v4.b32
  tt.func public @atomic_load_store_vector_aligned(
      %ptrs: tensor<512x!tt.ptr<i32>, #vec> {tt.contiguity = 4 : i32, tt.divisibility = 16 : i32}) {
    %loaded = tt.atomic_load acquire, gpu, %ptrs : (tensor<512x!tt.ptr<i32>, #vec>) -> tensor<512xi32, #vec>
    tt.atomic_store release, gpu, %ptrs, %loaded : tensor<512x!tt.ptr<i32>, #vec>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_store_vector_mask_pairs
  // CHECK-COMMON-COUNT-2: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.v2.b32
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.fence syncscope("device") release
  // CHECK-COMMON-COUNT-2: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.v2.b32
  tt.func public @atomic_load_store_vector_mask_pairs(
      %ptrs: tensor<512x!tt.ptr<i32>, #vec> {tt.contiguity = 4 : i32, tt.divisibility = 16 : i32},
      %mask: tensor<512xi1, #vec> {tt.constancy = 2 : i32}) {
    %loaded = tt.atomic_load acquire, gpu, %ptrs, %mask : (tensor<512x!tt.ptr<i32>, #vec>, tensor<512xi1, #vec>) -> tensor<512xi32, #vec>
    tt.atomic_store release, gpu, %ptrs, %loaded, %mask : tensor<512x!tt.ptr<i32>, #vec>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_store_vector_unaligned
  // CHECK-COMMON-COUNT-2: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.v2.b32
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.fence syncscope("device") release
  // CHECK-COMMON-COUNT-2: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.v2.b32
  tt.func public @atomic_load_store_vector_unaligned(
      %ptrs: tensor<512x!tt.ptr<i32>, #vec> {tt.contiguity = 4 : i32, tt.divisibility = 8 : i32}) {
    %loaded = tt.atomic_load acquire, gpu, %ptrs : (tensor<512x!tt.ptr<i32>, #vec>) -> tensor<512xi32, #vec>
    tt.atomic_store release, gpu, %ptrs, %loaded : tensor<512x!tt.ptr<i32>, #vec>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_store_vector_noncontiguous
  // CHECK-COMMON-COUNT-4: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.fence syncscope("device") release
  // CHECK-COMMON-COUNT-4: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.b32
  tt.func public @atomic_load_store_vector_noncontiguous(
      %ptrs: tensor<512x!tt.ptr<i32>, #vec> {tt.contiguity = 1 : i32, tt.divisibility = 16 : i32}) {
    %loaded = tt.atomic_load acquire, gpu, %ptrs : (tensor<512x!tt.ptr<i32>, #vec>) -> tensor<512xi32, #vec>
    tt.atomic_store release, gpu, %ptrs, %loaded : tensor<512x!tt.ptr<i32>, #vec>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_store_vector_varying_mask
  // CHECK-COMMON-COUNT-4: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.fence syncscope("device") release
  // CHECK-COMMON-COUNT-4: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.b32
  tt.func public @atomic_load_store_vector_varying_mask(
      %ptrs: tensor<512x!tt.ptr<i32>, #vec> {tt.contiguity = 4 : i32, tt.divisibility = 16 : i32},
      %mask: tensor<512xi1, #vec> {tt.constancy = 1 : i32}) {
    %loaded = tt.atomic_load acquire, gpu, %ptrs, %mask : (tensor<512x!tt.ptr<i32>, #vec>, tensor<512xi1, #vec>) -> tensor<512xi32, #vec>
    tt.atomic_store release, gpu, %ptrs, %loaded, %mask : tensor<512x!tt.ptr<i32>, #vec>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_store_vector_bytes
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b32
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.fence syncscope("device") release
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.b32
  tt.func public @atomic_load_store_vector_bytes(
      %ptrs: tensor<512x!tt.ptr<i8>, #vec> {tt.contiguity = 4 : i32, tt.divisibility = 4 : i32},
      %mask: tensor<512xi1, #vec> {tt.constancy = 4 : i32}) {
    %loaded = tt.atomic_load acquire, gpu, %ptrs, %mask : (tensor<512x!tt.ptr<i8>, #vec>, tensor<512xi1, #vec>) -> tensor<512xi8, #vec>
    tt.atomic_store release, gpu, %ptrs, %loaded, %mask : tensor<512x!tt.ptr<i8>, #vec>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_store_vector_half
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.v2.b32
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.fence syncscope("device") release
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.v2.b32
  tt.func public @atomic_load_store_vector_half(
      %ptrs: tensor<512x!tt.ptr<f16>, #vec> {tt.contiguity = 4 : i32, tt.divisibility = 8 : i32},
      %mask: tensor<512xi1, #vec> {tt.constancy = 4 : i32}) {
    %loaded = tt.atomic_load acquire, gpu, %ptrs, %mask : (tensor<512x!tt.ptr<f16>, #vec>, tensor<512xi1, #vec>) -> tensor<512xf16, #vec>
    tt.atomic_store release, gpu, %ptrs, %loaded, %mask : tensor<512x!tt.ptr<f16>, #vec>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_store_vector_double
  // CHECK-COMMON-COUNT-2: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.v2.b64
  // CHECK-COMMON: llvm.fence syncscope("device") acquire
  // CHECK-COMMON: nvvm.barrier
  // CHECK-COMMON: llvm.fence syncscope("device") release
  // CHECK-COMMON-COUNT-2: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.v2.b64
  tt.func public @atomic_load_store_vector_double(
      %ptrs: tensor<512x!tt.ptr<f64>, #vec> {tt.contiguity = 4 : i32, tt.divisibility = 32 : i32},
      %mask: tensor<512xi1, #vec> {tt.constancy = 4 : i32}) {
    %loaded = tt.atomic_load acquire, gpu, %ptrs, %mask : (tensor<512x!tt.ptr<f64>, #vec>, tensor<512xi1, #vec>) -> tensor<512xf64, #vec>
    tt.atomic_store release, gpu, %ptrs, %loaded, %mask : tensor<512x!tt.ptr<f64>, #vec>
    tt.return
  }

}

// -----

#packed = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-COMMON-LABEL: @atomic_load_store_packed_bytes
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.v4.b32
  // CHECK-COMMON: llvm.fence syncscope("device") release
  // CHECK-COMMON: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.v4.b32
  tt.func public @atomic_load_store_packed_bytes(
      %ptrs: tensor<2048x!tt.ptr<i8>, #packed> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) {
    %loaded = tt.atomic_load relaxed, gpu, %ptrs : (tensor<2048x!tt.ptr<i8>, #packed>) -> tensor<2048xi8, #packed>
    tt.atomic_store release, gpu, %ptrs, %loaded : tensor<2048x!tt.ptr<i8>, #packed>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_store_packed_half
  // CHECK-COMMON-COUNT-2: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.v4.b32
  // CHECK-COMMON: llvm.fence syncscope("device") release
  // CHECK-COMMON-COUNT-2: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.v4.b32
  tt.func public @atomic_load_store_packed_half(
      %ptrs: tensor<2048x!tt.ptr<f16>, #packed> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32}) {
    %loaded = tt.atomic_load relaxed, gpu, %ptrs : (tensor<2048x!tt.ptr<f16>, #packed>) -> tensor<2048xf16, #packed>
    tt.atomic_store release, gpu, %ptrs, %loaded : tensor<2048x!tt.ptr<f16>, #packed>
    tt.return
  }

  // CHECK-COMMON-LABEL: @atomic_load_store_packed_byte_pairs
  // CHECK-COMMON-COUNT-8: llvm.inline_asm has_side_effects {{.*}}ld.relaxed.gpu.global.b16
  // CHECK-COMMON: llvm.fence syncscope("device") release
  // CHECK-COMMON-COUNT-8: llvm.inline_asm has_side_effects {{.*}}st.relaxed.gpu.global.b16
  tt.func public @atomic_load_store_packed_byte_pairs(
      %ptrs: tensor<2048x!tt.ptr<i8>, #packed> {tt.contiguity = 16 : i32, tt.divisibility = 16 : i32},
      %mask: tensor<2048xi1, #packed> {tt.constancy = 2 : i32}) {
    %loaded = tt.atomic_load relaxed, gpu, %ptrs, %mask : (tensor<2048x!tt.ptr<i8>, #packed>, tensor<2048xi1, #packed>) -> tensor<2048xi8, #packed>
    tt.atomic_store release, gpu, %ptrs, %loaded, %mask : tensor<2048x!tt.ptr<i8>, #packed>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // THREAD-REGIONS-LABEL: @thread_regions_scalar_chain(
  // THREAD-REGIONS: %[[LEADER:.*]] = ttg.thread_predicate <31, 3>
  // THREAD-REGIONS-NEXT: scf.if %[[LEADER]]
  // THREAD-REGIONS: ttg.thread_scope
  // THREAD-REGIONS: tt.atomic_load relaxed, gpu,
  // THREAD-REGIONS: arith.subi
  // THREAD-REGIONS: arith.shli
  // THREAD-REGIONS: tt.store
  // THREAD-REGIONS-NOT: ttg.thread_handoff
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_scalar_chain(%ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %one = arith.constant 1 : i32
    %value = tt.atomic_load relaxed, gpu, %ptr : (!tt.ptr<i32>) -> i32
    %amount = arith.subi %value, %one : i32
    %shifted = arith.shli %one, %amount : i32
    tt.store %out, %shifted : !tt.ptr<i32>
    tt.return
  }

  // THREAD-REGIONS-LABEL: @thread_regions_warp_tail(
  // THREAD-REGIONS: %[[W_LEADER:.*]] = ttg.thread_predicate <31, 3>
  // THREAD-REGIONS-NEXT: %[[W_RESULT:.*]] = scf.if %[[W_LEADER]] -> (i32)
  // THREAD-REGIONS: ttg.thread_scope
  // THREAD-REGIONS: tt.atomic_load relaxed, gpu,
  // THREAD-REGIONS: arith.addi
  // THREAD-REGIONS: %[[W_BROADCAST:.*]] = ttg.thread_handoff %[[W_RESULT]] <31, 3> -> <0, 3> : i32
  // THREAD-REGIONS: %[[W_WARP:.*]] = ttg.thread_predicate <0, 3>
  // THREAD-REGIONS-NEXT: scf.if %[[W_WARP]]
  // THREAD-REGIONS: tt.splat %[[W_BROADCAST]]
  // THREAD-REGIONS: tt.store
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_warp_tail(%ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>) {
    %one = arith.constant 1 : i32
    %value = tt.atomic_load relaxed, gpu, %ptr : (!tt.ptr<i32>) -> i32
    %result = arith.addi %value, %one : i32
    %values = tt.splat %result : i32 -> tensor<32xi32, #blocked>
    %offsets = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    %base = tt.splat %out : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>, #blocked>
    %ptrs = tt.addptr %base, %offsets : tensor<32x!tt.ptr<i32>, #blocked>, tensor<32xi32, #blocked>
    tt.store %ptrs, %values : tensor<32x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // THREAD-REGIONS-LABEL: @thread_regions_cta_tail(
  // THREAD-REGIONS: %[[C_LEADER:.*]] = ttg.thread_predicate <31, 3>
  // THREAD-REGIONS-NEXT: %[[C_RESULT:.*]] = scf.if %[[C_LEADER]] -> (i32)
  // THREAD-REGIONS: ttg.thread_scope
  // THREAD-REGIONS: tt.atomic_load relaxed, gpu,
  // THREAD-REGIONS: } else {
  // THREAD-REGIONS: scf.yield
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NEXT: %[[C_BROADCAST:.*]] = ttg.thread_handoff %[[C_RESULT]] <31, 3> -> <0, 0> : i32
  // THREAD-REGIONS: tt.splat %[[C_BROADCAST]]
  // THREAD-REGIONS: tt.store
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_cta_tail(%ptr: !tt.ptr<i32>, %out: tensor<128x!tt.ptr<i32>, #blocked>) {
    %value = tt.atomic_load relaxed, gpu, %ptr : (!tt.ptr<i32>) -> i32
    %values = tt.splat %value : i32 -> tensor<128xi32, #blocked>
    tt.store %out, %values : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // THREAD-REGIONS-LABEL: @thread_regions_acquire_without_value(
  // THREAD-REGIONS: %[[A_LEADER:.*]] = ttg.thread_predicate <31, 3>
  // THREAD-REGIONS-NEXT: scf.if %[[A_LEADER]]
  // THREAD-REGIONS: ttg.thread_scope
  // THREAD-REGIONS: tt.atomic_load acquire, gpu,
  // THREAD-REGIONS-NEXT: ttg.thread_yield
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NEXT: tt.return
  tt.func @thread_regions_acquire_without_value(%ptr: !tt.ptr<i32>) {
    %value = tt.atomic_load acquire, gpu, %ptr : (!tt.ptr<i32>) -> i32
    tt.return
  }

  // THREAD-REGIONS-LABEL: @thread_regions_plain_load_cta_tail(
  // THREAD-REGIONS-NOT: ttg.thread_predicate
  // THREAD-REGIONS: %[[PLAIN:.*]] = tt.load
  // THREAD-REGIONS-NOT: ttg.thread_handoff
  // THREAD-REGIONS: tt.splat %[[PLAIN]]
  // THREAD-REGIONS-NOT: ttg.thread_predicate
  // THREAD-REGIONS-NOT: ttg.thread_handoff
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_plain_load_cta_tail(%ptr: !tt.ptr<i32>, %out: tensor<128x!tt.ptr<i32>, #blocked>) {
    %value = tt.load %ptr : !tt.ptr<i32>
    %values = tt.splat %value : i32 -> tensor<128xi32, #blocked>
    tt.store %out, %values : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }

  // THREAD-MEMBAR-LABEL: @thread_regions_async_atomic_store(
  // THREAD-MEMBAR: ttg.async_wait
  // THREAD-MEMBAR-NEXT: ttg.barrier local
  // THREAD-MEMBAR-NEXT: %[[ATOMIC_LEADER:.*]] = ttg.thread_predicate <31, 3>
  // THREAD-MEMBAR-NEXT: cf.cond_br %[[ATOMIC_LEADER]]
  // THREAD-MEMBAR-NOT: ttg.barrier
  // THREAD-MEMBAR: tt.atomic_store relaxed, gpu
  // THREAD-MEMBAR-NOT: ttg.barrier
  // THREAD-MEMBAR: tt.return
  tt.func @thread_regions_async_atomic_store(%out: !tt.ptr<i32>) {
    %one = arith.constant 1 : i32
    ttg.async_wait {num = 0 : i32}
    tt.atomic_store relaxed, gpu, %out, %one : !tt.ptr<i32>
    tt.return
  }

  // THREAD-MEMBAR-LABEL: @thread_regions_async_store(
  // THREAD-MEMBAR: ttg.async_wait
  // THREAD-MEMBAR-NEXT: ttg.barrier local
  // THREAD-MEMBAR-NEXT: %[[STORE_LEADER:.*]] = ttg.thread_predicate <31, 3>
  // THREAD-MEMBAR-NEXT: cf.cond_br %[[STORE_LEADER]]
  // THREAD-MEMBAR-NOT: ttg.barrier
  // THREAD-MEMBAR: tt.store
  // THREAD-MEMBAR-NOT: ttg.barrier
  // THREAD-MEMBAR: tt.return
  tt.func @thread_regions_async_store(%out: !tt.ptr<i32>) {
    %one = arith.constant 1 : i32
    ttg.async_wait {num = 0 : i32}
    tt.store %out, %one : !tt.ptr<i32>
    tt.return
  }

  // THREAD-REGIONS-LABEL: @thread_regions_success_only_offset(
  // THREAD-REGIONS: scf.if
  // THREAD-REGIONS: %[[LOADED:.*]] = tt.atomic_load relaxed, gpu
  // THREAD-REGIONS: %[[READY:.*]] = arith.cmpi
  // THREAD-REGIONS-NEXT: %[[OFFSET:.*]] = arith.muli %[[LOADED]]
  // THREAD-REGIONS-NEXT: %[[DEST:.*]] = tt.addptr %{{.*}}, %[[OFFSET]]
  // THREAD-REGIONS-NEXT: scf.if %[[READY]]
  // THREAD-REGIONS: tt.store %[[DEST]], %[[LOADED]]
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_success_only_offset(%ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>, %stride: i32) {
    %zero = arith.constant 0 : i32
    %loaded = tt.atomic_load relaxed, gpu, %ptr : (!tt.ptr<i32>) -> i32
    %ready = arith.cmpi ne, %loaded, %zero : i32
    %offset = arith.muli %loaded, %stride : i32
    %dest = tt.addptr %out, %offset : !tt.ptr<i32>, i32
    scf.if %ready {
      tt.store %dest, %loaded : !tt.ptr<i32>
    }
    tt.return
  }

  // THREAD-REGIONS-LABEL: @thread_regions_independent_offset(
  // THREAD-REGIONS-NOT: tt.addptr
  // THREAD-REGIONS: tt.atomic_store relaxed, gpu
  // THREAD-REGIONS-NEXT: ttg.thread_yield
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NEXT: %[[INDEPENDENT_BASE:.*]] = tt.addptr
  // THREAD-REGIONS-NOT: ttg.thread_handoff
  // THREAD-REGIONS: tt.splat %[[INDEPENDENT_BASE]]
  // THREAD-REGIONS-NOT: ttg.thread_handoff
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_independent_offset(%flag: !tt.ptr<i32>, %out: !tt.ptr<i32>, %offset: i32, %data: tensor<128xi32, #blocked>) {
    %one = arith.constant 1 : i32
    tt.atomic_store relaxed, gpu, %flag, %one : !tt.ptr<i32>
    %base = tt.addptr %out, %offset : !tt.ptr<i32>, i32
    %ptrs = tt.splat %base : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %offsets = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #blocked>
    %dst = tt.addptr %ptrs, %offsets : tensor<128x!tt.ptr<i32>, #blocked>, tensor<128xi32, #blocked>
    tt.store %dst, %data : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

#sharded = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // THREAD-REGIONS-LABEL: @thread_regions_multi_cta_loop(
  // THREAD-REGIONS: %[[LEADER:.*]] = ttg.thread_predicate <31, 3>
  // THREAD-REGIONS: %[[VALUE:.*]] = scf.if %[[LEADER]] -> (i32)
  // THREAD-REGIONS: ttg.thread_scope
  // THREAD-REGIONS: scf.for
  // THREAD-REGIONS: tt.load
  // THREAD-REGIONS: ttg.barrier all
  // THREAD-REGIONS: ttg.thread_handoff %[[VALUE]] <31, 3> -> <0, 0> : i32
  // THREAD-REGIONS: tt.store
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_multi_cta_loop(%ptr: !tt.ptr<i32>, %out: tensor<256x!tt.ptr<i32>, #sharded>, %iterations: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %value = scf.for %i = %zero to %iterations step %one iter_args(%sum = %zero) -> (i32) : i32 {
      %source = tt.addptr %ptr, %i : !tt.ptr<i32>, i32
      %element = tt.load %source : !tt.ptr<i32>
      %next = arith.addi %sum, %element : i32
      scf.yield %next : i32
    }
    %values = tt.splat %value : i32 -> tensor<256xi32, #sharded>
    tt.store %out, %values : tensor<256x!tt.ptr<i32>, #sharded>
    tt.return
  }

  // THREAD-REGIONS-LABEL: @thread_regions_multi_cta_atomic_loop(
  // THREAD-REGIONS: ttng.cluster_barrier
  // THREAD-REGIONS: %[[LEADER:.*]] = ttg.thread_predicate <31, 3, 1>
  // THREAD-REGIONS: %[[VALUE:.*]] = scf.if %[[LEADER]] -> (i32)
  // THREAD-REGIONS: ttg.thread_scope
  // THREAD-REGIONS: scf.for
  // THREAD-REGIONS: tt.atomic_rmw add, acq_rel, gpu
  // THREAD-REGIONS-NOT: ttng.cluster_barrier
  // THREAD-REGIONS: ttg.thread_yield
  // THREAD-REGIONS: } else {
  // THREAD-REGIONS: scf.yield
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NEXT: ttng.cluster_barrier
  // THREAD-REGIONS-NEXT: ttg.thread_handoff %[[VALUE]] <31, 3, 1> -> <0, 0> : i32
  // THREAD-REGIONS: tt.store
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_multi_cta_atomic_loop(%ptr: !tt.ptr<i32>, %out: tensor<256x!tt.ptr<i32>, #sharded>, %iterations: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %value = scf.for %i = %zero to %iterations step %one iter_args(%sum = %zero) -> (i32) : i32 {
      %element = tt.atomic_rmw add, acq_rel, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32
      %next = arith.addi %sum, %element : i32
      scf.yield %next : i32
    }
    %values = tt.splat %value : i32 -> tensor<256xi32, #sharded>
    tt.store %out, %values : tensor<256x!tt.ptr<i32>, #sharded>
    tt.return
  }

  // THREAD-REGIONS-LABEL: @thread_regions_multi_cta_unrelated_loads(
  // THREAD-REGIONS: tt.load
  // THREAD-REGIONS: ttg.thread_predicate <31, 3, 1>
  // THREAD-REGIONS: ttg.thread_scope
  // THREAD-REGIONS-NEXT: {{.*}}tt.atomic_rmw add, relaxed, gpu
  // THREAD-REGIONS-NEXT: ttg.thread_yield
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NOT: ttg.thread_handoff
  // THREAD-REGIONS: tt.load
  // THREAD-REGIONS-NOT: ttg.thread_predicate
  // THREAD-REGIONS-NOT: ttg.thread_handoff
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_multi_cta_unrelated_loads(%counter: !tt.ptr<i32>, %ptr: !tt.ptr<i32>, %out: tensor<256x!tt.ptr<i32>, #sharded>) {
    %one = arith.constant 1 : i32
    %before = tt.load %ptr : !tt.ptr<i32>
    %old = tt.atomic_rmw add, relaxed, gpu, %counter, %one : (!tt.ptr<i32>, i32) -> i32
    %after = tt.load %ptr : !tt.ptr<i32>
    %sum = arith.addi %before, %after : i32
    %values = tt.splat %sum : i32 -> tensor<256xi32, #sharded>
    tt.store %out, %values : tensor<256x!tt.ptr<i32>, #sharded>
    tt.return
  }

  // THREAD-REGIONS-LABEL: @thread_regions_multi_cta_atomic_capture(
  // THREAD-REGIONS: ttg.thread_predicate <31, 3, 1>
  // THREAD-REGIONS: ttg.thread_scope
  // THREAD-REGIONS: tt.atomic_load relaxed, gpu
  // THREAD-REGIONS: arith.addi
  // THREAD-REGIONS: tt.addptr
  // THREAD-REGIONS: tt.load
  // THREAD-REGIONS: scf.if
  // THREAD-REGIONS: tt.store
  // THREAD-REGIONS: scf.yield
  // THREAD-REGIONS: tt.store
  // THREAD-REGIONS-NEXT: ttg.thread_yield
  // THREAD-REGIONS-NOT: ttg.thread_handoff
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_multi_cta_atomic_capture(%index: !tt.ptr<i32>, %ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>, %condition: i1) {
    %one = arith.constant 1 : i32
    %old = tt.atomic_load relaxed, gpu, %index : (!tt.ptr<i32>) -> i32
    %offset = arith.addi %old, %one : i32
    %source = tt.addptr %ptr, %offset : !tt.ptr<i32>, i32
    %value = tt.load %source : !tt.ptr<i32>
    %result = scf.if %condition -> (i32) {
      tt.store %out, %value : !tt.ptr<i32>
      scf.yield %value : i32
    } else {
      scf.yield %one : i32
    }
    tt.store %out, %result : !tt.ptr<i32>
    tt.return
  }

  // THREAD-REGIONS-LABEL: @thread_regions_multi_cta_separated_dependents(
  // THREAD-REGIONS: ttg.thread_predicate <31, 3, 1>
  // THREAD-REGIONS: %[[RESULT:.*]] = scf.if
  // THREAD-REGIONS: tt.atomic_load relaxed, gpu
  // THREAD-REGIONS: ttg.thread_yield
  // THREAD-REGIONS: } else {
  // THREAD-REGIONS: scf.yield
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NEXT: ttg.thread_handoff %[[RESULT]] <31, 3, 1> -> <31, 3, 1> : i32
  // THREAD-REGIONS-NOT: ttg.thread_predicate
  // THREAD-REGIONS: tt.load
  // THREAD-REGIONS-NEXT: tt.store
  // THREAD-REGIONS: ttg.thread_predicate <31, 3, 1>
  // THREAD-REGIONS: ttg.thread_scope
  // THREAD-REGIONS: tt.store
  // THREAD-REGIONS: ttg.thread_yield
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS: tt.splat
  // THREAD-REGIONS-NEXT: tt.store
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_multi_cta_separated_dependents(%index: !tt.ptr<i32>, %ptr: !tt.ptr<i32>, %out: !tt.ptr<i32>, %out_atomic: !tt.ptr<i32>, %out_tensor: tensor<256x!tt.ptr<i32>, #sharded>) {
    %old = tt.atomic_load relaxed, gpu, %index : (!tt.ptr<i32>) -> i32
    %value = tt.load %ptr : !tt.ptr<i32>
    tt.store %out, %value : !tt.ptr<i32>
    tt.store %out_atomic, %old : !tt.ptr<i32>
    %values = tt.splat %value : i32 -> tensor<256xi32, #sharded>
    tt.store %out_tensor, %values : tensor<256x!tt.ptr<i32>, #sharded>
    tt.return
  }

  // THREAD-REGIONS-LABEL: @thread_regions_multi_cta_arithmetic_boundary(
  // THREAD-REGIONS: ttg.thread_predicate <31, 3, 1>
  // THREAD-REGIONS: %[[OLD:.*]] = scf.if
  // THREAD-REGIONS: tt.atomic_load relaxed, gpu
  // THREAD-REGIONS: %[[OLD_VALUE:.*]] = ttg.thread_handoff %[[OLD]] <31, 3, 1> -> <31, 3, 1> : i32
  // THREAD-REGIONS-NEXT: %[[PER_CTA:.*]] = ttg.thread_predicate <31, 3>
  // THREAD-REGIONS-NEXT: scf.if %[[PER_CTA]]
  // THREAD-REGIONS-NEXT: ttg.thread_scope
  // THREAD-REGIONS-NEXT: %[[LOADED_VALUE:.*]] = tt.load
  // THREAD-REGIONS-NEXT: tt.store %{{.*}}, %[[LOADED_VALUE]]
  // THREAD-REGIONS-NEXT: ttg.thread_yield
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NEXT: }
  // THREAD-REGIONS-NEXT: %[[DESTINATION:.*]] = tt.addptr
  // THREAD-REGIONS-NEXT: %[[CTA_ZERO:.*]] = ttg.thread_predicate <31, 3, 1>
  // THREAD-REGIONS-NEXT: scf.if %[[CTA_ZERO]]
  // THREAD-REGIONS-NEXT: ttg.thread_scope
  // THREAD-REGIONS-NEXT: %[[SUM:.*]] = arith.addi %[[OLD_VALUE]]
  // THREAD-REGIONS-NEXT: tt.store %[[DESTINATION]], %[[SUM]]
  // THREAD-REGIONS: tt.return
  tt.func @thread_regions_multi_cta_arithmetic_boundary(%flag: !tt.ptr<i32>, %src: !tt.ptr<i32>, %dst: !tt.ptr<i32>, %out: !tt.ptr<i32>, %offset: i32) {
    %old = tt.atomic_load relaxed, gpu, %flag : (!tt.ptr<i32>) -> i32
    %v = tt.load %src : !tt.ptr<i32>
    tt.store %dst, %v : !tt.ptr<i32>
    %q = tt.addptr %out, %offset : !tt.ptr<i32>, i32
    %val = arith.addi %old, %offset : i32
    tt.store %q, %val : !tt.ptr<i32>
    tt.return
  }
}
