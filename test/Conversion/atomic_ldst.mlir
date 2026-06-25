// RUN: triton-opt %s --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 2>&1 | FileCheck %s --check-prefixes=CHECK-TTG2NVGPU,CHECK-POLL
// RUN: triton-opt %s --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 --convert-nv-gpu-to-llvm 2>&1 | FileCheck %s --check-prefixes=CHECK-NVGPU2LLVM,CHECK-POLL
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

  // CHECK-POLL-LABEL: @atomic_poll
  // CHECK-POLL: nvvm.read.ptx.sreg.tid.x
  // CHECK-POLL: llvm.cond_br %[[ELECTED:.*]], ^[[INIT:bb[0-9]+]], ^[[DONE:bb[0-9]+]](%{{.*}} : i1)
  // CHECK-POLL: ^[[INIT]]:
  // CHECK-POLL: %[[START:.*]] = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
  // CHECK-POLL: llvm.br ^[[LOOP:bb[0-9]+]]
  // CHECK-POLL: ^[[LOOP]]:
  // CHECK-POLL: %[[LOADED:.*]] = llvm.load %{{.*}} atomic monotonic
  // CHECK-POLL: %[[MATCHED:.*]] = llvm.icmp "eq" %[[LOADED]], %{{.*}} : i32
  // CHECK-POLL: llvm.cond_br %[[MATCHED]], ^[[SUCCESS:bb[0-9]+]], ^[[TIMEOUT:bb[0-9]+]]
  // CHECK-POLL: ^[[SUCCESS]]:
  // CHECK-POLL: llvm.fence acquire
  // CHECK-POLL: llvm.br ^[[DONE]](%{{.*}} : i1)
  // CHECK-POLL: ^[[TIMEOUT]]:
  // CHECK-POLL: %[[NOW:.*]] = llvm.call_intrinsic "llvm.nvvm.read.ptx.sreg.globaltimer"() : () -> i64
  // CHECK-POLL: %[[ELAPSED:.*]] = llvm.sub %[[NOW]], %[[START]] : i64
  // CHECK-POLL: %[[TIMED_OUT:.*]] = llvm.icmp "uge" %[[ELAPSED]], %{{.*}} : i64
  // CHECK-POLL: llvm.cond_br %[[TIMED_OUT]], ^[[DONE]](%{{.*}} : i1), ^[[LOOP]]
  // CHECK-POLL: ^[[DONE]](%[[RESULT:.*]]: i1):
  // CHECK-POLL: llvm.insertelement %[[RESULT]],
  // CHECK-POLL: llvm.inline_asm has_side_effects
  // CHECK-POLL: nvvm.barrier
  // CHECK-POLL: %{{.*}} = llvm.load %{{.*}} : !llvm.ptr<3> -> i1
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
