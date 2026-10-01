//
// Regression test for triton-patches/patch-direct-to-lds-swizzle-check-vec.patch
// and triton-patches/patch-direct-to-lds-swizzle-chunk.patch.
//
// The clamp from patch11295.patch has to measure the lane shuffle with the
// vector the load uses, which the register-to-shared run, the target's LDS
// write widths, the pointer contiguity and the mask alignment cap. Each tile
// below hits one of those caps and loads one f32 per lane, so a warp writes 64
// consecutive f32, half of a 128-wide row. maxPhase = 8 keeps the swizzle
// inside that chunk and the deduced maxPhase = 16 does not, whatever layout
// CoalesceAsyncCopy then gives the copy.

// RUN: triton-opt %s -split-input-file -tritonamdgpu-pipeline='use_async_copy=1' | FileCheck %s

// One f32 per thread over contiguous pointers, so the layout caps the load at
// one element. 64 lanes cover half the K columns and maxPhase = 16 has to halve
// once.

// CHECK: #[[$SHARED:.+]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
// CHECK-LABEL: @swizzle_clamped_one_elem_per_lane
// CHECK: ttg.async_copy_global_to_local {{.*}} -> <64x128xf32, #[[$SHARED]],

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 64], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @swizzle_clamped_one_elem_per_lane(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<64x64xf32, #mma> {
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
      %aDot = ttg.convert_layout %a : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dotA>
      %aBf16 = arith.truncf %aDot : tensor<64x128xf32, #dotA> to tensor<64x128xbf16, #dotA>
      %dot = tt.dot %aBf16, %b, %acc {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x128xbf16, #dotA> * tensor<128x64xbf16, #dotB> -> tensor<64x64xf32, #mma>
      scf.yield %dot : tensor<64x64xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return %res : tensor<64x64xf32, #mma>
  }
}

// -----

// Two f32 per thread over contiguous pointers, but CDNA4 writes only 32 and 128
// bits to LDS, so the load vector is still one element and maxPhase = 16 has to
// halve once. The two f32 a lane holds here must not be read as two lanes.

// CHECK: #[[$SHARED:.+]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
// CHECK-LABEL: @swizzle_clamped_two_elems_per_lane
// CHECK: ttg.async_copy_global_to_local {{.*}} -> <64x128xf32, #[[$SHARED]],

#blocked = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 64], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @swizzle_clamped_two_elems_per_lane(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<64x64xf32, #mma> {
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
      %aDot = ttg.convert_layout %a : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dotA>
      %aBf16 = arith.truncf %aDot : tensor<64x128xf32, #dotA> to tensor<64x128xbf16, #dotA>
      %dot = tt.dot %aBf16, %b, %acc {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x128xbf16, #dotA> * tensor<128x64xbf16, #dotB> -> tensor<64x64xf32, #mma>
      scf.yield %dot : tensor<64x64xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return %res : tensor<64x64xf32, #mma>
  }
}

// -----

// Four f32 per thread, which fits a 128-bit LDS write, but every lane reads the
// same pointer, so the async copy loads one element at a time and the layout's
// four must not be used. maxPhase = 16 has to halve once.

// CHECK: #[[$SHARED:.+]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
// CHECK-LABEL: @swizzle_clamped_splat_pointer
// CHECK: ttg.async_copy_global_to_local {{.*}} -> <64x128xf32, #[[$SHARED]],

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @swizzle_clamped_splat_pointer(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<64x64xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %acc0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %b = arith.constant dense<0.000000e+00> : tensor<128x64xbf16, #dotB>
    %ptrs = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x128x!tt.ptr<f32>, #blocked>
    %res = scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc = %acc0) -> (tensor<64x64xf32, #mma>) : i32 {
      %a = tt.load %ptrs {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f32>, #blocked>
      %aDot = ttg.convert_layout %a : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dotA>
      %aBf16 = arith.truncf %aDot : tensor<64x128xf32, #dotA> to tensor<64x128xbf16, #dotA>
      %dot = tt.dot %aBf16, %b, %acc {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x128xbf16, #dotA> * tensor<128x64xbf16, #dotB> -> tensor<64x64xf32, #mma>
      scf.yield %dot : tensor<64x64xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return %res : tensor<64x64xf32, #mma>
  }
}

// -----

// Four f32 per thread over contiguous pointers, which fits a 128-bit LDS write,
// but the mask changes from one element to the next, so the async copy loads
// one element at a time. maxPhase = 16 has to halve once.

// CHECK: #[[$SHARED:.+]] = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
// CHECK-LABEL: @swizzle_clamped_mask_alignment
// CHECK: ttg.async_copy_global_to_local {{.*}} -> <64x128xf32, #[[$SHARED]],

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 2], instrShape = [32, 32, 16], isTransposed = true}>
#dotA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>
#dotB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @swizzle_clamped_mask_alignment(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %bound: i32) -> tensor<64x64xf32, #mma> {
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
    %boundv = tt.splat %bound : i32 -> tensor<64x128xi32, #blocked>
    %mask = arith.cmpi slt, %offs, %boundv : tensor<64x128xi32, #blocked>
    %res = scf.for %k = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%acc = %acc0) -> (tensor<64x64xf32, #mma>) : i32 {
      %a = tt.load %ptrs, %mask {loop.cluster = 0 : i32, loop.stage = 0 : i32} : tensor<64x128x!tt.ptr<f32>, #blocked>
      %aDot = ttg.convert_layout %a : tensor<64x128xf32, #blocked> -> tensor<64x128xf32, #dotA>
      %aBf16 = arith.truncf %aDot : tensor<64x128xf32, #dotA> to tensor<64x128xbf16, #dotA>
      %dot = tt.dot %aBf16, %b, %acc {loop.cluster = 1 : i32, loop.stage = 1 : i32} : tensor<64x128xbf16, #dotA> * tensor<128x64xbf16, #dotB> -> tensor<64x64xf32, #mma>
      scf.yield %dot : tensor<64x64xf32, #mma>
    } {tt.scheduled_max_stage = 1 : i32}
    tt.return %res : tensor<64x64xf32, #mma>
  }
}
