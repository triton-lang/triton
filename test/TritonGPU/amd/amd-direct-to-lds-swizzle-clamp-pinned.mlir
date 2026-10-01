//
// Regression test for triton-patches/patch-direct-to-lds-swizzle-chunk.patch.
//
// The load feeds a dot and a mutable local_alloc that pins maxPhase = 16. With
// one f32 per lane only maxPhase = 8 stays inside the warp, and the pipeliner
// cannot stage a conversion to the pinned layout, so the load is stream copied
// into the pinned layout instead of being clamped.

// RUN: triton-opt %s -tritonamdgpu-pipeline='use_async_copy=1' | FileCheck %s

// CHECK: #[[$SHARED:.+]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
// CHECK-LABEL: @pinned_swizzle_stream_copied
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: ttg.local_store {{.*}} -> !ttg.memdesc<64x128xf32, #[[$SHARED]], #smem, mutable>
// CHECK-NOT: ttg.async_copy_global_to_local

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
#shared16 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @pinned_swizzle_stream_copied(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<64x64xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %b = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #dotB>
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %row = tt.expand_dims %range {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %offs = tt.broadcast %row : tensor<1x128xi32, #blocked> -> tensor<64x128xi32, #blocked>
    %base = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked>
    %ptrs = tt.addptr %base, %offs : tensor<64x128x!tt.ptr<f32>, #blocked>, tensor<64x128xi32, #blocked>
    %res = scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc = %acc0) -> (tensor<64x64xf32, #mma>) : i32 {
      %a = tt.load %ptrs {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f32>, #blocked>
      %aDot = ttg.convert_layout %a {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dotA>
      %aBf16 = arith.truncf %aDot {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #dotA> to tensor<64x128xbf16, #dotA>
      %dot = tt.dot %aBf16, %b, %acc {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x128xbf16, #dotA> * tensor<128x64xbf16, #dotB> -> tensor<64x64xf32, #mma>
      %aMem = ttg.local_alloc %a {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<64x128xf32, #blocked>) -> !ttg.memdesc<64x128xf32, #shared16, #smem, mutable>
      %aLd = ttg.local_load %aMem {loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<64x128xf32, #shared16, #smem, mutable> -> tensor<64x128xf32, #dotA>
      %aLdBf16 = arith.truncf %aLd {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x128xf32, #dotA> to tensor<64x128xbf16, #dotA>
      %dot2 = tt.dot %aLdBf16, %b, %dot {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x128xbf16, #dotA> * tensor<128x64xbf16, #dotB> -> tensor<64x64xf32, #mma>
      scf.yield %dot2 : tensor<64x64xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return %res : tensor<64x64xf32, #mma>
  }
}
