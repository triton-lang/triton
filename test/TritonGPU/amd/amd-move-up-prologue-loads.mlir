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

// -----

// CHECK-LABEL: break_at_store
// CHECK: tt.store
// CHECK: arith.cmpi
// CHECK: tt.splat
// CHECK: tt.load
// CHECK: ttg.local_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @break_at_store(%ptrs: tensor<32x128x!tt.ptr<f16>, #blocked>,
                          %value: tensor<32x128xf16, #blocked>, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    tt.store %ptrs, %value : tensor<32x128x!tt.ptr<f16>, #blocked>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>
    %1 = arith.cmpi sgt, %arg2, %c0_i32 : i32
    %2 = tt.splat %1 : i1 -> tensor<32x128xi1, #blocked>
    %3 = tt.load %ptrs, %2 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: break_at_nested_store
// CHECK: scf.if
// CHECK: tt.store
// CHECK: arith.cmpi
// CHECK: tt.splat
// CHECK: tt.load
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @break_at_nested_store(%ptrs: tensor<32x128x!tt.ptr<f16>, #blocked>,
                                 %value: tensor<32x128xf16, #blocked>, %predicate: i1, %arg3: i32) {
    %c0_i32 = arith.constant 0 : i32
    scf.if %predicate {
      tt.store %ptrs, %value : tensor<32x128x!tt.ptr<f16>, #blocked>
    }
    %0 = arith.cmpi sgt, %arg3, %c0_i32 : i32
    %1 = tt.splat %0 : i1 -> tensor<32x128xi1, #blocked>
    %2 = tt.load %ptrs, %1 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: break_at_unknown_effect
// CHECK: tt.call @overwrite
// CHECK: arith.cmpi
// CHECK: tt.splat
// CHECK: tt.load
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func private @overwrite(%ptrs: tensor<32x128x!tt.ptr<f16>, #blocked>,
                             %value: tensor<32x128xf16, #blocked>) {
    tt.store %ptrs, %value : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }

  tt.func @break_at_unknown_effect(%ptrs: tensor<32x128x!tt.ptr<f16>, #blocked>,
                                   %value: tensor<32x128xf16, #blocked>, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    tt.call @overwrite(%ptrs, %value) : (tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xf16, #blocked>) -> ()
    %0 = arith.cmpi sgt, %arg2, %c0_i32 : i32
    %1 = tt.splat %0 : i1 -> tensor<32x128xi1, #blocked>
    %2 = tt.load %ptrs, %1 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: move_across_shared_store
// CHECK: arith.cmpi
// CHECK: tt.splat
// CHECK: tt.load
// CHECK: ttg.local_store
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @move_across_shared_store(%ptrs: tensor<32x128x!tt.ptr<f16>, #blocked>,
                                    %value: tensor<32x128xf16, #blocked>,
                                    %buffer: !ttg.memdesc<32x128xf16, #shared, #smem, mutable>, %arg3: i32) {
    %c0_i32 = arith.constant 0 : i32
    ttg.local_store %value, %buffer : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %0 = arith.cmpi sgt, %arg3, %c0_i32 : i32
    %1 = tt.splat %0 : i1 -> tensor<32x128xi1, #blocked>
    %2 = tt.load %ptrs, %1 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: move_across_l2_prefetch
// CHECK: arith.cmpi
// CHECK: tt.splat
// CHECK: tt.load
// CHECK: amdg.tdm_prefetch
// CHECK: ttg.local_alloc
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#desc_shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @move_across_l2_prefetch(%base: !tt.ptr<f16>,
                                    %ptrs: tensor<32x128x!tt.ptr<f16>, #blocked>, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_true = arith.constant true
    %desc = tt.make_tensor_descriptor %base, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <64x64xf16, #desc_shared>
    amdg.tdm_prefetch %desc[%c_offset, %c_offset], %c_true, speculative = false : !tt.tensordesc<64x64xf16, #desc_shared>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>
    %1 = arith.cmpi sgt, %arg2, %c0_i32 : i32
    %2 = tt.splat %1 : i1 -> tensor<32x128xi1, #blocked>
    %3 = tt.load %ptrs, %2 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK-LABEL: do_not_move_volatile_load
// CHECK: ttg.local_alloc
// CHECK: arith.cmpi
// CHECK: tt.splat
// CHECK: tt.load
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @do_not_move_volatile_load(%ptrs: tensor<32x128x!tt.ptr<f16>, #blocked>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x32x128xf16, #shared, #smem, mutable>
    %1 = arith.cmpi sgt, %arg1, %c0_i32 : i32
    %2 = tt.splat %1 : i1 -> tensor<32x128xi1, #blocked>
    %3 = tt.load %ptrs, %2 {amd.pipeliner_part = "prologue", isVolatile = true} : tensor<32x128x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
