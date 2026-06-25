// RUN: triton-opt %s -split-input-file -convert-triton-amdgpu-to-llvm="gfx-arch=gfx942" -cse | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @atomic_cas_0(%arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: @atomic_cas_0
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: %[[C32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: llvm.cmpxchg %{{.*}}, %[[C32]], %[[C64]] syncscope("agent") acquire monotonic
    %0 = tt.atomic_cas acquire, gpu, %arg3, %c32_i32, %c64_i32 : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @atomic_cas_1(%arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: @atomic_cas_1
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: %[[C32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: llvm.cmpxchg %{{.*}}, %[[C32]], %[[C64]] syncscope("agent") monotonic monotonic
    %0 = tt.atomic_cas relaxed, gpu, %arg3, %c32_i32, %c64_i32 : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @atomic_cas_2(%arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: @atomic_cas_2
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: %[[C32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: llvm.cmpxchg %{{.*}}, %[[C32]], %[[C64]] syncscope("agent") acq_rel monotonic
    %0 = tt.atomic_cas acq_rel, gpu, %arg3, %c32_i32, %c64_i32 : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @atomic_cas_3(%arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: @atomic_cas_3
    %c64_i32 = arith.constant 64 : i32
    %c32_i32 = arith.constant 32 : i32
    // CHECK: %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
    // CHECK: %[[C32:.*]] = llvm.mlir.constant(32 : i32) : i32
    // CHECK: llvm.cmpxchg %{{.*}}, %[[C32]], %[[C64]] acquire monotonic
    %0 = tt.atomic_cas acquire, sys, %arg3, %c32_i32, %c64_i32 : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @atomic_poll(%ptr: !tt.ptr<i32>, %expected: i32, %timeout: i64) {
    // CHECK-LABEL: @atomic_poll
    // CHECK: %[[WARP_ELECTED:.*]] = llvm.and %{{.*}}, %{{.*}} : i1
    // CHECK: %[[ELECTED:.*]] = llvm.and %[[WARP_ELECTED]], %{{.*}} : i1
    // CHECK: %[[FALSE:.*]] = llvm.mlir.constant(false) : i1
    // CHECK: llvm.cond_br %[[ELECTED]], ^[[INIT:bb[0-9]+]], ^[[DONE:bb[0-9]+]](%[[FALSE]] : i1)
    // CHECK: ^[[INIT]]:
    // CHECK: %[[START_RAW:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"() : () -> i64
    // CHECK: %[[TEN:.*]] = llvm.mlir.constant(10 : i64) : i64
    // CHECK: %[[START:.*]] = llvm.mul %[[START_RAW]], %[[TEN]] : i64
    // CHECK: llvm.br ^[[LOOP:bb[0-9]+]]
    // CHECK: ^[[LOOP]]:
    // CHECK: %[[LOADED:.*]] = llvm.load %{{.*}} atomic syncscope("agent") monotonic
    // CHECK: %[[MATCHED:.*]] = llvm.icmp "eq" %[[LOADED]], %{{.*}} : i32
    // CHECK: llvm.cond_br %[[MATCHED]], ^[[SUCCESS:bb[0-9]+]], ^[[TIMEOUT:bb[0-9]+]]
    // CHECK: ^[[SUCCESS]]:
    // CHECK: llvm.fence syncscope("agent") acquire
    // CHECK: %[[TRUE:.*]] = llvm.mlir.constant(true) : i1
    // CHECK: llvm.br ^[[DONE]](%[[TRUE]] : i1)
    // CHECK: ^[[TIMEOUT]]:
    // CHECK: %[[NOW_RAW:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.memrealtime"() : () -> i64
    // CHECK: %[[NOW:.*]] = llvm.mul %[[NOW_RAW]], %[[TEN]] : i64
    // CHECK: %[[ELAPSED:.*]] = llvm.sub %[[NOW]], %[[START]] : i64
    // CHECK: %[[TIMED_OUT:.*]] = llvm.icmp "uge" %[[ELAPSED]], %{{.*}} : i64
    // CHECK: llvm.cond_br %[[TIMED_OUT]], ^[[DONE]](%[[FALSE]] : i1), ^[[LOOP]]
    // CHECK: ^[[DONE]](%{{.*}}: i1):
    // CHECK: llvm.fence syncscope("workgroup") release
    // CHECK: rocdl.s.barrier
    // CHECK: llvm.fence syncscope("workgroup") acquire
    %matched = tt.atomic_poll acquire, gpu, %ptr, %expected timeout %timeout : !tt.ptr<i32>, i32 -> i1
    tt.return
  }

  tt.func public @atomic_poll_relaxed(%ptr: !tt.ptr<i32>, %expected: i32) {
    // CHECK-LABEL: @atomic_poll_relaxed
    // CHECK: llvm.load %{{.*}} atomic syncscope("workgroup") monotonic
    %matched = tt.atomic_poll relaxed, cta, %ptr, %expected : !tt.ptr<i32>, i32 -> i1
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @atomic_cas_f32(%arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    // CHECK-LABEL: @atomic_cas_f32
    %c64_f32 = arith.constant 64. : f32
    %c32_f32 = arith.constant 32. : f32
    // CHECK-DAG: %[[C64:.*]] = llvm.mlir.constant(6.400000e+01 : f32) : f32
    // CHECK-DAG: %[[C32:.*]] = llvm.mlir.constant(3.200000e+01 : f32) : f32
    // CHECK-DAG: %[[C64I:.*]] = llvm.bitcast %[[C64]] : f32 to i32
    // CHECK-DAG: %[[C32I:.*]] = llvm.bitcast %[[C32]] : f32 to i32
    // CHECK: %[[CMPXCHG:.*]] = llvm.cmpxchg %{{.*}}, %[[C32I]], %[[C64I]] acquire monotonic
    // CHECK: %[[RESI:.*]] = llvm.extractvalue %[[CMPXCHG]][0] : !llvm.struct<(i32, i1)>
    // CHECK: %[[RES:.*]] = llvm.bitcast %[[RESI]] : i32 to f32
    // CHECK: llvm.store %[[RES]], %{{.*}} : f32, !llvm.ptr<3>
    %0 = tt.atomic_cas acquire, sys, %arg3, %c32_f32, %c64_f32 { allocation.offset = 0 : i32 }: (!tt.ptr<f32>, f32, f32) -> f32
    tt.print "some print" {hex = false, isSigned = array<i32: 0>} : %0: f32
    tt.return
  }
}
