// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: order_load_alloc_local_load_local_store
//       CHECK:   %[[LOAD:.+]] = tt.load
//       CHECK:   %[[ALLOC:.+]] = ttg.local_alloc
//       CHECK:   ttg.local_store %[[LOAD]], %[[ALLOC]]
//       CHECK:   ttg.local_load %[[ALLOC]]
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @order_load_alloc_local_load_local_store(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) {
    %9 = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    ttg.local_store %9, %10 : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %cst_0 = arith.constant dense<1.230000e+02> : tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    %11 = ttg.local_load %10 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    %12 = tt.dot %11, %cst_0, %cst : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<32x32xf32, #mma>
    %13 = ttg.convert_layout %12 : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %13 : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

//   CHECK-LABEL: sink_convert_dealloc
// CHECK-COUNT-2: ttg.local_dealloc %{{.+}} : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
//         CHECK: ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_convert_dealloc(%arg0: tensor<32x32xf32, #blocked>) {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %2 = ttg.convert_layout %arg0 : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    %3 = arith.addf %2, %2 : tensor<32x32xf32, #blocked1>
    tt.return
  }
}

// -----

//   CHECK-LABEL: anchor_barrier
//         CHECK: gpu.barrier
//         CHECK: tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @anchor_barrier(%arg0: tensor<32x32x!tt.ptr<f16>, #blocked>) {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    gpu.barrier
    %2 = tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %1 = ttg.local_alloc %2 : (tensor<32x32xf16, #blocked>) -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----

#mfma = #ttg.amd_mfma<{version = 2, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx90a", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: dont_hoist_scf_ops
  // Make sure we don't hoist scf ops above its dependencies.
  tt.func public @dont_hoist_scf_ops(%init: tensor<256x128xf32, #mfma>,
    %base: tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>,
    %p1: tensor<128x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>, %i1: i1) -> (tensor<256x128xf32, #mfma>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 1.44269502 : f32
    %c128_i32 = arith.constant 128 : i32
    // CHECK: scf.for
    %54 = scf.for %arg21 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg = %init) -> (tensor<256x128xf32, #mfma>)  : i32 {
      // CHECK: arith.addi
      %f = arith.addi %arg21, %c128_i32 : i32
      // CHECK: scf.if
      // CHECK: tt.load
      %p0 = scf.if %i1 -> tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>{
        %t = tt.splat %f : i32 -> tensor<256x128xi32>
        %padd = tt.addptr %base, %t : tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>, tensor<256x128xi32>
        scf.yield %padd : tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      } else {
        scf.yield %base : tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      }
      %l = tt.load %p0 : tensor<256x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>>
      %r = tt.load %p1 : tensor<128x128x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>>
      %acc = tt.dot %l, %r, %arg : tensor<256x128xf16, #ttg.dot_op<{opIdx = 0, parent = #mfma, kWidth = 4}>> * tensor<128x128xf16, #ttg.dot_op<{opIdx = 1, parent = #mfma, kWidth = 4}>> -> tensor<256x128xf32, #mfma>
      scf.yield %acc : tensor<256x128xf32, #mfma>
    }
    tt.return %54 : tensor<256x128xf32, #mfma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
// This example tests the case where global loads in the prologue are moved early.
// CHECK-LABEL: move_up_global_load_in_prologue
// CHECK: tt.addptr
// CHECK: tt.splat
// CHECK: tt.load
// CHECK: tt.addptr
// CHECK: tt.splat
// CHECK: tt.load
// CHECK: ttg.local_alloc
// CHECK: ttg.local_alloc
  tt.func @move_up_global_load_in_prologue(
      %arg0: tensor<128x128x!tt.ptr<f16>, #blocked>,
      %arg1: tensor<128x128x!tt.ptr<f8E5M2FNUZ>, #blocked1>,
      %arg2: i32) {
    %cst = arith.constant dense<128> : tensor<128x128xi32, #blocked>
    %cst_0 = arith.constant dense<128> : tensor<128x128xi32, #blocked1>
    %c0_i32 = arith.constant 0 : i32

    %0 = tt.addptr %arg0, %cst : tensor<128x128x!tt.ptr<f16>, #blocked>, tensor<128x128xi32, #blocked>
    %1 = tt.addptr %arg1, %cst_0 : tensor<128x128x!tt.ptr<f8E5M2FNUZ>, #blocked1>, tensor<128x128xi32, #blocked1>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf8E5M2FNUZ, #shared1, #smem, mutable>
    %4 = arith.cmpi sgt, %arg2, %c0_i32 : i32
    %5 = tt.splat %4 : i1 -> tensor<128x128xi1, #blocked>
    %6 = tt.load %0, %5 {amd.pipeliner_part = "prologue"} : tensor<128x128x!tt.ptr<f16>, #blocked>
    %7 = tt.splat %4 : i1 -> tensor<128x128xi1, #blocked1>
    %8 = tt.load %1, %7 {amd.pipeliner_part = "prologue"} : tensor<128x128x!tt.ptr<f8E5M2FNUZ>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#smem = #ttg.shared_memory

// Category 1: Single dot with two loads, we make sure the optimization is applied when tile size is large enough
// The following tile sizes should apply the optimization
// 256x256x128
// 256x256x64
// The following tile sizes should NOT apply the optimization
// 256x64x128
// 256x256x32
//

// Should apply: tile size 256x256x128 with single dot
// CHECK-LABEL: sink_2nd_load_256x256x128
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: ttg.local_store %[[tileA]]
//  CHECK-NEXT: ttg.local_store %[[tileB]]
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x256x128(%A_ptr: tensor<256x128x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<128x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>, %A_LDS: !ttg.memdesc<256x128xf16, #shared, #smem, mutable>, %B_LDS: !ttg.memdesc<128x256xf16, #shared1, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %4 = tt.load %A_ptr : tensor<256x128x!tt.ptr<f16>, #blocked>
      %1 = ttg.local_load %A_LDS : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #dotOp0>
      %5 = tt.load %B_ptr : tensor<128x256x!tt.ptr<f16>, #blocked1>
      %2 = ttg.local_load %B_LDS : !ttg.memdesc<128x256xf16, #shared1, #smem, mutable> -> tensor<128x256xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x128xf16, #dotOp0> * tensor<128x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      ttg.local_store %4, %A_LDS : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
      ttg.local_store %5, %B_LDS : tensor<128x256xf16, #blocked1> -> !ttg.memdesc<128x256xf16, #shared1, #smem, mutable>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#smem = #ttg.shared_memory

// Should apply: tile size 256x256x128 with nested single dot
// CHECK-LABEL: nested_sink_2nd_load_256x256x128
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: ttg.local_store %[[tileA]]
//  CHECK-NEXT: ttg.local_store %[[tileB]]
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @nested_sink_2nd_load_256x256x128(%A_ptr: tensor<256x128x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<128x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>, %A_LDS: !ttg.memdesc<256x128xf16, #shared, #smem, mutable>, %B_LDS: !ttg.memdesc<128x256xf16, #shared1, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    scf.for %arg2 = %c0 to %c1 step %c1  : i32 {
      %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
        %4 = tt.load %A_ptr : tensor<256x128x!tt.ptr<f16>, #blocked>
        %1 = ttg.local_load %A_LDS : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #dotOp0>
        %5 = tt.load %B_ptr : tensor<128x256x!tt.ptr<f16>, #blocked1>
        %2 = ttg.local_load %B_LDS : !ttg.memdesc<128x256xf16, #shared1, #smem, mutable> -> tensor<128x256xf16, #dotOp1>
        %3 = tt.dot %1, %2, %arg1 : tensor<256x128xf16, #dotOp0> * tensor<128x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
        ttg.local_store %4, %A_LDS : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
        ttg.local_store %5, %B_LDS : tensor<128x256xf16, #blocked1> -> !ttg.memdesc<128x256xf16, #shared1, #smem, mutable>
        scf.yield %3 : tensor<256x256xf32, #mma>
      }
      tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#smem = #ttg.shared_memory

// Should apply: tile size 256x256x64 with single dot
// CHECK-LABEL: sink_2nd_load_256x256x64
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: ttg.local_store %[[tileA]]
//  CHECK-NEXT: ttg.local_store %[[tileB]]
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x256x64(%A_ptr: tensor<256x64x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<64x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>, %A_LDS: !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, %B_LDS: !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %4 = tt.load %A_ptr : tensor<256x64x!tt.ptr<f16>, #blocked>
      %1 = ttg.local_load %A_LDS : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #dotOp0>
      %5 = tt.load %B_ptr : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %2 = ttg.local_load %B_LDS : !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<64x256xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x64xf16, #dotOp0> * tensor<64x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      ttg.local_store %4, %A_LDS : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      ttg.local_store %5, %B_LDS : tensor<64x256xf16, #blocked1> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#smem = #ttg.shared_memory

// Should NOT apply: tile size 256x64x128 with single dot
// CHECK-LABEL: sink_2nd_load_256x64x128
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: ttg.local_store %[[tileA]]
//  CHECK-NEXT: ttg.local_store %[[tileB]]
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x64x128(%A_ptr: tensor<256x128x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<128x64x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x64x!tt.ptr<f32>, #mma>, %A_LDS: !ttg.memdesc<256x128xf16, #shared, #smem, mutable>, %B_LDS: !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x64xf32, #mma>)  : i32 {
      %4 = tt.load %A_ptr : tensor<256x128x!tt.ptr<f16>, #blocked>
      %1 = ttg.local_load %A_LDS : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> tensor<256x128xf16, #dotOp0>
      %5 = tt.load %B_ptr : tensor<128x64x!tt.ptr<f16>, #blocked1>
      %2 = ttg.local_load %B_LDS : !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x128xf16, #dotOp0> * tensor<128x64xf16, #dotOp1> -> tensor<256x64xf32, #mma>
      ttg.local_store %4, %A_LDS : tensor<256x128xf16, #blocked> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
      ttg.local_store %5, %B_LDS : tensor<128x64xf16, #blocked1> -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
      scf.yield %3 : tensor<256x64xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#smem = #ttg.shared_memory

// Should NOT apply: tile size 256x256x32 with single dot
// CHECK-LABEL: sink_2nd_load_256x256x32
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: ttg.local_store %[[tileA]]
//  CHECK-NEXT: ttg.local_store %[[tileB]]
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x256x32(%A_ptr: tensor<256x32x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<32x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>, %A_LDS: !ttg.memdesc<256x32xf16, #shared, #smem, mutable>, %B_LDS: !ttg.memdesc<32x256xf16, #shared1, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %4 = tt.load %A_ptr : tensor<256x32x!tt.ptr<f16>, #blocked>
      %1 = ttg.local_load %A_LDS : !ttg.memdesc<256x32xf16, #shared, #smem, mutable> -> tensor<256x32xf16, #dotOp0>
      %5 = tt.load %B_ptr : tensor<32x256x!tt.ptr<f16>, #blocked1>
      %2 = ttg.local_load %B_LDS : !ttg.memdesc<32x256xf16, #shared1, #smem, mutable> -> tensor<32x256xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x32xf16, #dotOp0> * tensor<32x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      ttg.local_store %4, %A_LDS : tensor<256x32xf16, #blocked> -> !ttg.memdesc<256x32xf16, #shared, #smem, mutable>
      ttg.local_store %5, %B_LDS : tensor<32x256xf16, #blocked1> -> !ttg.memdesc<32x256xf16, #shared1, #smem, mutable>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#smem = #ttg.shared_memory

// Category 2: single dot with two loads and tile size is large enough (128x128x128).
//             We make sure the move is legal.
// Should NOT apply: the 2nd load has a user before the dot
// CHECK-LABEL: sink_2nd_load_128x128x128_user_before_dot
//       CHECK: %[[tileA:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: %[[tileB:.*]] = tt.load
//  CHECK-NEXT: local_load
//  CHECK-NEXT: tt.store
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: ttg.local_store %[[tileA]]
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_128x128x128_user_before_dot(%A_ptr: tensor<128x128x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<128x128x!tt.ptr<i64>, #blocked>, %B_ptr2: tensor<128x128x!tt.ptr<f16>, #blocked>, %C_ptr: tensor<128x128x!tt.ptr<f32>, #mma>, %A_LDS: !ttg.memdesc<128x128xf16, #shared, #smem, mutable>, %B_LDS: !ttg.memdesc<128x128xf16, #shared1, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<128x128xf32, #mma>)  : i32 {
      %4 = tt.load %A_ptr : tensor<128x128x!tt.ptr<f16>, #blocked>
      %1 = ttg.local_load %A_LDS : !ttg.memdesc<128x128xf16, #shared, #smem, mutable> -> tensor<128x128xf16, #dotOp0>
      %5 = tt.load %B_ptr : tensor<128x128x!tt.ptr<i64>, #blocked>
      %2 = ttg.local_load %B_LDS : !ttg.memdesc<128x128xf16, #shared1, #smem, mutable> -> tensor<128x128xf16, #dotOp1>
      tt.store %B_ptr, %5 : tensor<128x128x!tt.ptr<i64>, #blocked>
      %3 = tt.dot %1, %2, %arg1 : tensor<128x128xf16, #dotOp0> * tensor<128x128xf16, #dotOp1> -> tensor<128x128xf32, #mma>
      ttg.local_store %4, %A_LDS : tensor<128x128xf16, #blocked> -> !ttg.memdesc<128x128xf16, #shared, #smem, mutable>
      scf.yield %3 : tensor<128x128xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<128x128x!tt.ptr<f32>, #mma>
    tt.return
  }
}


// -----

// Category 3: two dots in the for loop. Make sure the optimization is not applied
// should NOT apply: two dots
// CHECK-LABEL: sink_2nd_load_256x256x64_two_dot
//       CHECK: tt.load
//  CHECK-NEXT: tt.load
//  CHECK-NEXT: ttg.local_load
//  CHECK-NEXT: ttg.local_load
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: tt.dot
//  CHECK-NEXT: ttg.local_store
//  CHECK-NEXT: ttg.local_store
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @sink_2nd_load_256x256x64_two_dot(%A_ptr: tensor<256x64x!tt.ptr<f16>, #blocked>, %B_ptr: tensor<64x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>, %A_LDS: !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, %B_LDS: !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %4 = tt.load %A_ptr : tensor<256x64x!tt.ptr<f16>, #blocked>
      %5 = tt.load %B_ptr : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %1 = ttg.local_load %A_LDS : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> tensor<256x64xf16, #dotOp0>
      %2 = ttg.local_load %B_LDS : !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> tensor<64x256xf16, #dotOp1>
      %3 = tt.dot %1, %2, %arg1 : tensor<256x64xf16, #dotOp0> * tensor<64x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      %6 = tt.dot %1, %2, %3 : tensor<256x64xf16, #dotOp0> * tensor<64x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      ttg.local_store %4, %A_LDS : tensor<256x64xf16, #blocked> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
      ttg.local_store %5, %B_LDS : tensor<64x256xf16, #blocked1> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      scf.yield %3 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}


// -----

// Category 4: load a scalar. Make sure the optimization is not applied
// should NOT apply: load scalar
// CHECK-LABEL: sink_2nd_load_scalar
//       CHECK: tt.load
//  CHECK-NEXT: tt.splat
//  CHECK-NEXT: tt.broadcast
//  CHECK-NEXT: tt.load
//  CHECK-NEXT: ttg.convert_layout
//  CHECK-NEXT: ttg.convert_layout
//  CHECK-NEXT: tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
tt.func public @sink_2nd_load_scalar(%A_ptr: !tt.ptr<f16>, %B_ptr: tensor<64x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %1 = tt.load %A_ptr : !tt.ptr<f16>
      %2 = tt.splat %1 : f16 -> tensor<1x64xf16, #blocked>
      %3 = tt.broadcast %2 : tensor<1x64xf16, #blocked> -> tensor<256x64xf16, #blocked>
      %4 = tt.load %B_ptr : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %5 = ttg.convert_layout %3 : tensor<256x64xf16, #blocked> -> tensor<256x64xf16, #dotOp0>
      %6 = ttg.convert_layout %4 : tensor<64x256xf16, #blocked1> -> tensor<64x256xf16, #dotOp1>
      %7 = tt.dot %5, %6, %arg1 : tensor<256x64xf16, #dotOp0> * tensor<64x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      scf.yield %7 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}


// -----

// Category 5: load a 1D tensor. Make sure the optimization is not applied
// should NOT apply: load scalar
// CHECK-LABEL: sink_2nd_load_1D_tensor
//       CHECK: tt.load
//  CHECK-NEXT: tt.expand_dims
//  CHECK-NEXT: tt.broadcast
//  CHECK-NEXT: tt.load
//  CHECK-NEXT: ttg.convert_layout
//  CHECK-NEXT: ttg.convert_layout
//  CHECK-NEXT: tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 1], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [16, 16], isTransposed = true}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 64 : i32} {
tt.func public @sink_2nd_load_1D_tensor(%A_ptr: tensor<256x!tt.ptr<f16>, #ttg.slice<{dim = 1, parent = #blocked}>>, %B_ptr: tensor<64x256x!tt.ptr<f16>, #blocked1>, %C_ptr: tensor<256x256x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:1 = scf.for %arg0 = %c0 to %c1 step %c1 iter_args(%arg1 = %cst) -> (tensor<256x256xf32, #mma>)  : i32 {
      %1 = tt.load %A_ptr : tensor<256x!tt.ptr<f16>, #ttg.slice<{dim = 1, parent = #blocked}>>
      %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor<256xf16, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf16, #blocked>
      %3 = tt.broadcast %2 : tensor<256x1xf16, #blocked> -> tensor<256x64xf16, #blocked>
      %4 = tt.load %B_ptr : tensor<64x256x!tt.ptr<f16>, #blocked1>
      %5 = ttg.convert_layout %3 : tensor<256x64xf16, #blocked> -> tensor<256x64xf16, #dotOp0>
      %6 = ttg.convert_layout %4 : tensor<64x256xf16, #blocked1> -> tensor<64x256xf16, #dotOp1>
      %7 = tt.dot %5, %6, %arg1 : tensor<256x64xf16, #dotOp0> * tensor<64x256xf16, #dotOp1> -> tensor<256x256xf32, #mma>
      scf.yield %7 : tensor<256x256xf32, #mma>
    }
    tt.store %C_ptr, %0#0: tensor<256x256x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL: keep_double_loads_order
// CHECK: %[[A0:.*]] = tt.load %arg0
// CHECK-NEXT: %[[B0:.*]] = tt.load %arg1
// CHECK-COUNT-4: arith.constant
// CHECK-NEXT: %[[APTR:.*]] = tt.addptr %arg0
// CHECK-NEXT: %[[A1:.*]] = tt.load %[[APTR]]
// CHECK-NEXT: %[[BPTR:.*]] = tt.addptr %arg1
// CHECK-NEXT: %[[B1:.*]] = tt.load %[[BPTR]]
// CHECK: ttg.local_store %[[A0]]
// CHECK-NEXT: ttg.local_store %[[B0]]
// CHECK-NEXT: ttg.local_store %[[A1]]
// CHECK-NEXT: ttg.local_store %[[B1]]
#shared=#ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1=#ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
#blocked=#ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1=#ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @keep_double_loads_order(
    %arg0: tensor<32x128x!tt.ptr<f16>, #blocked>,
    %arg1: tensor<128x32x!tt.ptr<f8E5M2FNUZ>, #blocked1>
  ) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<128> : tensor<32x128xi32, #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>>
    %cst_0 = arith.constant dense<128> : tensor<128x32xi32, #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>>
    %0 = tt.addptr %arg0, %cst : tensor<32x128x!tt.ptr<f16>, #blocked>, tensor<32x128xi32, #blocked>
    %1 = tt.addptr %arg1, %cst_0 : tensor<128x32x!tt.ptr<f8E5M2FNUZ>, #blocked1>, tensor<128x32xi32, #blocked1>

    %2 = ttg.local_alloc : () -> !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<2x128x32xf8E5M2FNUZ, #shared1, #smem, mutable>
    %4 = tt.load %arg0 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    %5 = tt.load %arg1 {amd.pipeliner_part = "prologue"} : tensor<128x32x!tt.ptr<f8E5M2FNUZ>, #blocked1>

    %6 = tt.load %0 {amd.pipeliner_part = "prologue"} : tensor<32x128x!tt.ptr<f16>, #blocked>
    %7 = tt.load %1 {amd.pipeliner_part = "prologue"} : tensor<128x32x!tt.ptr<f8E5M2FNUZ>, #blocked1>

    %8 = ttg.memdesc_index %2[%c0_i32] : !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %9 = ttg.memdesc_index %3[%c0_i32] : !ttg.memdesc<2x128x32xf8E5M2FNUZ, #shared1, #smem, mutable> -> !ttg.memdesc<128x32xf8E5M2FNUZ, #shared1, #smem, mutable>
    %10 = ttg.memdesc_index %2[%c1_i32] : !ttg.memdesc<2x32x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    %11 = ttg.memdesc_index %3[%c1_i32] : !ttg.memdesc<2x128x32xf8E5M2FNUZ, #shared1, #smem, mutable> -> !ttg.memdesc<128x32xf8E5M2FNUZ, #shared1, #smem, mutable>

    ttg.local_store %4, %8 : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    ttg.local_store %5, %9 : tensor<128x32xf8E5M2FNUZ, #blocked1> -> !ttg.memdesc<128x32xf8E5M2FNUZ, #shared1, #smem, mutable>

    ttg.local_store %6, %10 : tensor<32x128xf16, #blocked> -> !ttg.memdesc<32x128xf16, #shared, #smem, mutable>
    ttg.local_store %7, %11 : tensor<128x32xf8E5M2FNUZ, #blocked1> -> !ttg.memdesc<128x32xf8E5M2FNUZ, #shared1, #smem, mutable>
    tt.return
  }
}
