// RUN: triton-opt %s -split-input-file -convert-triton-amdgpu-to-llvm="arch=gfx942" -cse | FileCheck %s

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
