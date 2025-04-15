// RUN: triton-opt %s -split-input-file -tritonamdgpu-reorder-instructions | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: order_load_alloc_local_load_local_store
//       CHECK:   %[[LOAD:.+]] = tt.load
//       CHECK:   %[[ALLOC:.+]] = ttg.local_alloc
//       CHECK:   ttg.local_store %[[LOAD]], %[[ALLOC]]
//       CHECK:   ttg.local_load %[[ALLOC]]
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @order_load_alloc_local_load_local_store(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>) attributes {noinline = false} {
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
  tt.func public @sink_convert_dealloc(%arg0: tensor<32x32xf32, #blocked>) attributes {noinline = false} {
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
  tt.func public @anchor_barrier(%arg0: tensor<32x32x!tt.ptr<f16>, #blocked>) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    gpu.barrier
    %2 = tt.load %arg0 : tensor<32x32x!tt.ptr<f16>, #blocked>
    %1 = ttg.local_alloc %2 : (tensor<32x32xf16, #blocked>) -> !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<4x128x64xf16, #shared, #smem, mutable>
    tt.return
  }
}


// -----

#mfma = #ttg.amd_mfma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [32, 32], isTransposed = true}>
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
#mma = #ttg.amd_mfma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 16], isTransposed = true}>
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
    %6 = tt.load %0, %5 {OpIdx = #amdgpu.OpIdx<0>, amd.pipeliner_part = "prologue"} : tensor<128x128x!tt.ptr<f16>, #blocked>
    %7 = tt.splat %4 : i1 -> tensor<128x128xi1, #blocked1>
    %8 = tt.load %1, %7 {OpIdx = #amdgpu.OpIdx<1>, amd.pipeliner_part = "prologue"} : tensor<128x128x!tt.ptr<f8E5M2FNUZ>, #blocked1>
    tt.return
  }
}
