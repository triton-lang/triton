// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx1100 | FileCheck %s
// RUN: triton-opt %s -split-input-file --convert-triton-amdgpu-to-llvm=gfx-arch=gfx942 | FileCheck %s

// The store conversions skip register indices that a layout maps to an element
// another register in the same thread already holds, which they detect via the
// "register" free-variable mask of getFreeVariableMasks(). A layout whose
// register basis is zero broadcasts that register, so only the canonical index
// is stored.

// Control: a non-zero register basis addresses two distinct elements, so both
// registers are stored.
#plain = #ttg.linear<{register = [[1]], lane = [[2], [4], [8], [16], [32]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @distinct_registers_are_all_stored
  // CHECK-COUNT-2: rocdl.raw.ptr.buffer.store
  // CHECK-NOT: rocdl.raw.ptr.buffer.store
  tt.func @distinct_registers_are_all_stored(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<64xi32, #plain>, %val: tensor<64xf32, #plain>) {
    amdg.buffer_store %val, %arg0[%offset] : !tt.ptr<f32> -> tensor<64xf32, #plain>
    tt.return
  }
}

// -----

// A zero register basis makes both registers hold the same element, so the
// redundant one is not stored.
#bcast = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @broadcast_register_buffer_store
  // CHECK-COUNT-1: rocdl.raw.ptr.buffer.store
  // CHECK-NOT: rocdl.raw.ptr.buffer.store
  tt.func @broadcast_register_buffer_store(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<32xi32, #bcast>, %val: tensor<32xf32, #bcast>) {
    amdg.buffer_store %val, %arg0[%offset] : !tt.ptr<f32> -> tensor<32xf32, #bcast>
    tt.return
  }
}

// -----

// Same deduplication for the non-buffer store path.
#bcast = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @broadcast_register_store
  // CHECK-COUNT-1: llvm.store
  // CHECK-NOT: llvm.store
  tt.func @broadcast_register_store(%ptr: tensor<32x!tt.ptr<f32>, #bcast>, %val: tensor<32xf32, #bcast>) {
    tt.store %ptr, %val : tensor<32x!tt.ptr<f32>, #bcast>
    tt.return
  }
}

// -----

// A scalar store has no layout, so getFreeVariableMasks() falls back to
// getAllFreeVarMasks(), which marks every dimension redundant. The store must
// still happen exactly once, under a predicate that elects a single thread.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @scalar_store_is_predicated
  // CHECK: llvm.cond_br
  // CHECK: llvm.store
  // CHECK-NOT: llvm.store
  tt.func @scalar_store_is_predicated(%ptr: !tt.ptr<f32>, %val: f32) {
    tt.store %ptr, %val : !tt.ptr<f32>
    tt.return
  }
}

// -----

// The atomic emits one operation for the canonical register only, but every
// register still needs a result value, so the redundant one reuses the value
// loaded by the canonical register.
#bcast = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @broadcast_register_buffer_atomic_rmw
  // CHECK-COUNT-1: llvm.amdgcn.raw.ptr.buffer.atomic.fadd
  // CHECK-NOT: llvm.amdgcn.raw.ptr.buffer.atomic.fadd
  tt.func @broadcast_register_buffer_atomic_rmw(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %offset: tensor<32xi32, #bcast>, %val: tensor<32xf32, #bcast>) -> tensor<32xf32, #bcast> {
    %0 = amdg.buffer_atomic_rmw fadd, acq_rel, gpu, %val, %arg0[%offset] : !tt.ptr<f32> -> tensor<32xf32, #bcast>
    tt.return %0 : tensor<32xf32, #bcast>
  }
}

// -----

// Same for the compare-and-swap path.
#bcast = #ttg.linear<{register = [[0]], lane = [[1], [2], [4], [8], [16]], warp = [], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @broadcast_register_atomic_cas
  // CHECK-COUNT-1: llvm.cmpxchg
  // CHECK-NOT: llvm.cmpxchg
  tt.func @broadcast_register_atomic_cas(%ptr: tensor<32x!tt.ptr<i32>, #bcast>, %cmp: tensor<32xi32, #bcast>, %val: tensor<32xi32, #bcast>) -> tensor<32xi32, #bcast> {
    %0 = tt.atomic_cas acq_rel, gpu, %ptr, %cmp, %val : (tensor<32x!tt.ptr<i32>, #bcast>, tensor<32xi32, #bcast>, tensor<32xi32, #bcast>) -> tensor<32xi32, #bcast>
    tt.return %0 : tensor<32xi32, #bcast>
  }
}
