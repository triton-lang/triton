// RUN: triton-opt %s -split-input-file -tritonamdgpu-move-up-prologue-loads | FileCheck %s

// CHECK-LABEL: move_up_slice
// CHECK: arith.cmpi
// CHECK: tt.splat
// CHECK: tt.load
// CHECK: ttg.local_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @move_up_slice(%arg0: tensor<32x128x!tt.ptr<f16>, #blocked>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>
    %1 = arith.cmpi sgt, %arg1, %c0_i32 : i32
    %2 = tt.splat %1 : i1 -> tensor<32x128xi1, #blocked>
    %3 = tt.load %arg0, %2 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: keep_load_order
// CHECK: arith.cmpi sgt
// CHECK: tt.splat
// CHECK: tt.load %arg0
// CHECK: tt.addptr
// CHECK: arith.cmpi slt
// CHECK: tt.splat
// CHECK: tt.load
// CHECK: ttg.local_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @keep_load_order(%arg0: tensor<32x128x!tt.ptr<f16>, #blocked>, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<128> : tensor<32x128xi32, #blocked>
    %0 = tt.addptr %arg0, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>
    %2 = arith.cmpi sgt, %arg1, %c0_i32 : i32
    %3 = tt.splat %2 : i1 -> tensor<32x128xi1, #blocked>
    %4 = tt.load %arg0, %3 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    %5 = arith.cmpi slt, %arg2, %c0_i32 : i32
    %6 = tt.splat %5 : i1 -> tensor<32x128xi1, #blocked>
    %7 = tt.load %0, %6 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: break_at_atomic
// CHECK: tt.atomic_rmw
// CHECK: arith.cmpi
// CHECK: tt.splat
// CHECK: tt.load
// CHECK: ttg.local_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @break_at_atomic(%arg0: tensor<32x128x!tt.ptr<f16>, #blocked>, %arg1: i32, %arg2: !tt.ptr<i32>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.atomic_rmw fadd, relaxed, gpu, %arg2, %c1_i32 : (!tt.ptr<i32>, i32) -> i32
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>
    %2 = arith.cmpi sgt, %arg1, %c0_i32 : i32
    %3 = tt.splat %2 : i1 -> tensor<32x128xi1, #blocked>
    %4 = tt.load %arg0, %3 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: break_at_barrier
// CHECK: gpu.barrier
// CHECK: arith.cmpi
// CHECK: tt.splat
// CHECK: tt.load
// CHECK: ttg.local_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @break_at_barrier(%arg0: tensor<32x128x!tt.ptr<f16>, #blocked>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    gpu.barrier
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>
    %1 = arith.cmpi sgt, %arg1, %c0_i32 : i32
    %2 = tt.splat %1 : i1 -> tensor<32x128xi1, #blocked>
    %3 = tt.load %arg0, %2 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: break_at_loop
// CHECK: scf.for
// CHECK: tt.load %arg0
// CHECK: ttg.local_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @break_at_loop(%arg0: tensor<32x128x!tt.ptr<f16>, #blocked>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
    }
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>
    %1 = tt.load %arg0 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// Negative test: load without amd.pipeliner_part attribute should not be moved
// CHECK-LABEL: no_prologue_attribute
// CHECK: ttg.local_alloc
// CHECK: arith.cmpi
// CHECK: tt.splat
// CHECK: tt.load
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @no_prologue_attribute(%arg0: tensor<32x128x!tt.ptr<f16>, #blocked>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>
    %1 = arith.cmpi sgt, %arg1, %c0_i32 : i32
    %2 = tt.splat %1 : i1 -> tensor<32x128xi1, #blocked>
    %3 = tt.load %arg0, %2 : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
