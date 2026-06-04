// RUN: triton-opt %s -split-input-file --tritonamdgpu-coalesce-async-copy=gfx-arch=gfx950 | FileCheck %s --check-prefix=GFX950
// RUN: triton-opt %s -split-input-file --tritonamdgpu-coalesce-async-copy=gfx-arch=gfx1100 | FileCheck %s --check-prefix=GFX1100

// VGPR buffer_load: element width controls the maximum legal 128-bit
// transaction size. These are the pass-level counterparts of AMDGCN
// `buffer_load_* ... offen` coalescing.
#blocked_i8 = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked_i16 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked_i32 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked_f32 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: @vgpr_i8_axisinfo_contig16
  tt.func @vgpr_i8_axisinfo_contig16(%ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<4096xi8, #blocked_i8> {
    %r = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked_i8>
    // GFX950: amdg.buffer_load {{.*}} {contiguity = 16 : i32}
    %ret = amdg.buffer_load %ptr[%r] : tensor<4096xi8, #blocked_i8>
    tt.return %ret : tensor<4096xi8, #blocked_i8>
  }

  // GFX950-LABEL: @vgpr_i16_axisinfo_contig8
  tt.func @vgpr_i16_axisinfo_contig8(%ptr: !tt.ptr<i16> {tt.divisibility = 16 : i32}) -> tensor<2048xi16, #blocked_i16> {
    %r = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #blocked_i16>
    // GFX950: amdg.buffer_load {{.*}} {contiguity = 8 : i32}
    %ret = amdg.buffer_load %ptr[%r] : tensor<2048xi16, #blocked_i16>
    tt.return %ret : tensor<2048xi16, #blocked_i16>
  }

  // GFX950-LABEL: @vgpr_i32_axisinfo_contig4
  tt.func @vgpr_i32_axisinfo_contig4(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}) -> tensor<1024xi32, #blocked_i32> {
    %r = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32, #blocked_i32>
    // GFX950: amdg.buffer_load {{.*}} {contiguity = 4 : i32}
    %ret = amdg.buffer_load %ptr[%r] : tensor<1024xi32, #blocked_i32>
    tt.return %ret : tensor<1024xi32, #blocked_i32>
  }

  // GFX950-LABEL: @vgpr_f32_axisinfo_clamp_contig4
  tt.func @vgpr_f32_axisinfo_clamp_contig4(%ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> tensor<2048xf32, #blocked_f32> {
    %r = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #blocked_f32>
    // GFX950: amdg.buffer_load {{.*}} {contiguity = 4 : i32}
    %ret = amdg.buffer_load %ptr[%r] : tensor<2048xf32, #blocked_f32>
    tt.return %ret : tensor<2048xf32, #blocked_f32>
  }
}

// -----

// Direct-to-LDS buffer_load_to_local: pass-level counterparts of AMDGCN
// `buffer_load_* ... offen lds` coalescing.
#blocked_lds_i8 = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#blocked_lds_i16 = #ttg.blocked<{sizePerThread = [8], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: @lds_i8_axisinfo_contig16
  tt.func @lds_i8_axisinfo_contig16(%ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %dst: !ttg.memdesc<4096xi8, #shared, #smem, mutable>) {
    %r = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked_lds_i8>
    // GFX950: amdg.buffer_load_to_local {{.*}} into {{.*}} {contiguity = 16 : i32}
    %tok = amdg.buffer_load_to_local %ptr[%r] into %dst : <i8>[tensor<4096xi32, #blocked_lds_i8>] -> <4096xi8, #shared, #smem, mutable>
    tt.return
  }

  // GFX950-LABEL: @lds_i16_axisinfo_contig8
  tt.func @lds_i16_axisinfo_contig8(%ptr: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %dst: !ttg.memdesc<2048xi16, #shared, #smem, mutable>) {
    %r = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #blocked_lds_i16>
    // GFX950: amdg.buffer_load_to_local {{.*}} into {{.*}} {contiguity = 8 : i32}
    %tok = amdg.buffer_load_to_local %ptr[%r] into %dst : <i16>[tensor<2048xi32, #blocked_lds_i16>] -> <2048xi16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Mask alignment preserves only the two-element group that is safe for both
// VGPR and direct-to-LDS coalescing.
#blocked_mask = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared_mask = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: @vgpr_i16_mask_clamps_to2
  tt.func @vgpr_i16_mask_clamps_to2(%ptr: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<2048xi1, #blocked_mask> {tt.constancy = 2 : i32}) -> tensor<2048xi16, #blocked_mask> {
    %r = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #blocked_mask>
    // GFX950: amdg.buffer_load {{.*}} {contiguity = 2 : i32}
    %ret = amdg.buffer_load %ptr[%r], %mask : tensor<2048xi16, #blocked_mask>
    tt.return %ret : tensor<2048xi16, #blocked_mask>
  }

  // GFX950-LABEL: @lds_i16_mask_clamps_to2
  tt.func @lds_i16_mask_clamps_to2(%ptr: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %mask: tensor<2048xi1, #blocked_mask> {tt.constancy = 2 : i32}, %dst: !ttg.memdesc<2048xi16, #shared_mask, #smem, mutable>) {
    %r = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32, #blocked_mask>
    // GFX950: amdg.buffer_load_to_local {{.*}} mask {{.*}} into {{.*}} {contiguity = 2 : i32}
    %tok = amdg.buffer_load_to_local %ptr[%r] mask=%mask into %dst : <i16>[tensor<2048xi32, #blocked_mask>] -> <2048xi16, #shared_mask, #smem, mutable>
    tt.return
  }
}

// -----

// Non-power-of-two input contiguity should be floored to the largest legal
// power-of-two transaction for both VGPR and LDS paths.
#blocked_np2 = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared_np2 = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: @vgpr_i16_non_power_two_contig3_to2
  tt.func @vgpr_i16_non_power_two_contig3_to2(%ptr: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %off: tensor<1024xi32, #blocked_np2> {tt.contiguity = 3 : i32, tt.divisibility = 16 : i32}) -> tensor<1024xi16, #blocked_np2> {
    // GFX950: amdg.buffer_load {{.*}} {contiguity = 2 : i32}
    %ret = amdg.buffer_load %ptr[%off] : tensor<1024xi16, #blocked_np2>
    tt.return %ret : tensor<1024xi16, #blocked_np2>
  }

  // GFX950-LABEL: @lds_i16_non_power_two_contig3_to2
  tt.func @lds_i16_non_power_two_contig3_to2(%ptr: !tt.ptr<i16> {tt.divisibility = 16 : i32}, %off: tensor<1024xi32, #blocked_np2> {tt.contiguity = 3 : i32, tt.divisibility = 16 : i32}, %dst: !ttg.memdesc<1024xi16, #shared_np2, #smem, mutable>) {
    // GFX950: amdg.buffer_load_to_local {{.*}} into {{.*}} {contiguity = 2 : i32}
    %tok = amdg.buffer_load_to_local %ptr[%off] into %dst : <i16>[tensor<1024xi32, #blocked_np2>] -> <1024xi16, #shared_np2, #smem, mutable>
    tt.return
  }
}

// -----

// Symbolic register-order proof: AxisInfo cannot describe the per-register four-byte
// pattern, but direct evaluation proves offsets [x, x+1, x+2, x+3].
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [[64, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: @vgpr_const_eval_rem_div_contig4
  tt.func @vgpr_const_eval_rem_div_contig4(%ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<128x8xi8, #linear> {
    %rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %rows2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi32, #linear>
    %cols2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x8xi32, #linear>
    %c256 = arith.constant dense<256> : tensor<128x1xi32, #linear>
    %c4 = arith.constant dense<4> : tensor<1x8xi32, #linear>
    %c128 = arith.constant dense<128> : tensor<1x8xi32, #linear>
    %row = arith.muli %rows2d, %c256 : tensor<128x1xi32, #linear>
    %col_lo = arith.remui %cols2d, %c4 : tensor<1x8xi32, #linear>
    %col_hi_0 = arith.divui %cols2d, %c4 : tensor<1x8xi32, #linear>
    %col_hi = arith.muli %col_hi_0, %c128 : tensor<1x8xi32, #linear>
    %col = arith.addi %col_lo, %col_hi : tensor<1x8xi32, #linear>
    %row_b = tt.broadcast %row : tensor<128x1xi32, #linear> -> tensor<128x8xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x8xi32, #linear> -> tensor<128x8xi32, #linear>
    %off = arith.addi %row_b, %col_b : tensor<128x8xi32, #linear>
    // GFX950: amdg.buffer_load {{.*}} {contiguity = 4 : i32}
    %ret = amdg.buffer_load %ptr[%off] : tensor<128x8xi8, #linear>
    tt.return %ret : tensor<128x8xi8, #linear>
  }
}

// -----

// Symbolic register-order proof negative: per-register deltas are scalar-dependent and
// must not be assumed contiguous.
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [[64, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: @vgpr_const_eval_scalar_dependent_reject
  tt.func @vgpr_const_eval_scalar_dependent_reject(%ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %stride: i32) -> tensor<128x4xi8, #linear> {
    %rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %rows2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi32, #linear>
    %cols2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x4xi32, #linear>
    %c128 = arith.constant dense<128> : tensor<128x1xi32, #linear>
    %row = arith.muli %rows2d, %c128 : tensor<128x1xi32, #linear>
    %row_b = tt.broadcast %row : tensor<128x1xi32, #linear> -> tensor<128x4xi32, #linear>
    %stride_splat = tt.splat %stride : i32 -> tensor<1x4xi32, #linear>
    %col = arith.muli %cols2d, %stride_splat : tensor<1x4xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x4xi32, #linear> -> tensor<128x4xi32, #linear>
    %off = arith.addi %row_b, %col_b : tensor<128x4xi32, #linear>
    // GFX950: %[[RET:.*]] = amdg.buffer_load
    // GFX950-NOT: contiguity =
    // GFX950: tt.return %[[RET]]
    %ret = amdg.buffer_load %ptr[%off] : tensor<128x4xi8, #linear>
    tt.return %ret : tensor<128x4xi8, #linear>
  }
}

// -----

// Existing contiguity should not be lowered or rewritten.
#blocked_existing = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared_existing = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: @vgpr_existing_contiguity_noop
  tt.func @vgpr_existing_contiguity_noop(%ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}) -> tensor<4096xi8, #blocked_existing> {
    %r = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked_existing>
    // GFX950: %[[RET:.*]] = amdg.buffer_load {{.*}} {contiguity = 16 : i32}
    // GFX950-NOT: amdg.buffer_load
    // GFX950: tt.return %[[RET]]
    %ret = amdg.buffer_load %ptr[%r] {contiguity = 16 : i32} : tensor<4096xi8, #blocked_existing>
    tt.return %ret : tensor<4096xi8, #blocked_existing>
  }

  // GFX950-LABEL: @lds_existing_contiguity_noop
  tt.func @lds_existing_contiguity_noop(%ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %dst: !ttg.memdesc<4096xi8, #shared_existing, #smem, mutable>) {
    %r = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked_existing>
    // GFX950: %[[TOK:.*]] = amdg.buffer_load_to_local {{.*}} into {{.*}} {contiguity = 16 : i32}
    // GFX950-NOT: amdg.buffer_load_to_local
    // GFX950: tt.return
    %tok = amdg.buffer_load_to_local %ptr[%r] into %dst {contiguity = 16 : i32} : <i8>[tensor<4096xi32, #blocked_existing>] -> <4096xi8, #shared_existing, #smem, mutable>
    tt.return
  }
}

// -----

// Non-CDNA pass option should early-return and leave both VGPR and LDS ops
// unstamped even though the IR would coalesce on gfx950.
#blocked_skip = #ttg.blocked<{sizePerThread = [16], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared_skip = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX1100-LABEL: @non_cdna_no_transform
  tt.func @non_cdna_no_transform(%ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %dst: !ttg.memdesc<4096xi8, #shared_skip, #smem, mutable>) -> tensor<4096xi8, #blocked_skip> {
    %r = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32, #blocked_skip>
    // GFX1100: %[[RET:.*]] = amdg.buffer_load
    // GFX1100-NOT: contiguity =
    // GFX1100: %[[TOK:.*]] = amdg.buffer_load_to_local
    // GFX1100-NOT: contiguity =
    // GFX1100: tt.return %[[RET]]
    %ret = amdg.buffer_load %ptr[%r] : tensor<4096xi8, #blocked_skip>
    %tok = amdg.buffer_load_to_local %ptr[%r] into %dst : <i8>[tensor<4096xi32, #blocked_skip>] -> <4096xi8, #shared_skip, #smem, mutable>
    tt.return %ret : tensor<4096xi8, #blocked_skip>
  }
}

// -----

// LinearLayout-native soundness guard. The per-register stride carries a
// kernel-arg term  poison(arg) = arg*(arg-1)  (built in tensor ops so the
// symbolic evaluator walks it). It is ZERO at arg in {0, 1} but NON-zero for
// arg >= 2, so the true per-thread contiguity is NOT 4 for all inputs. The
// symbolic register-order proof must refuse to stamp contiguity here.
#linear = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [[64, 0], [0, 0]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: @vgpr_symbolic_scalar_dep_reject
  tt.func @vgpr_symbolic_scalar_dep_reject(%ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg: i32) -> tensor<128x4xi8, #linear> {
    %rows = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>>
    %cols = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #linear}>>
    %rows2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #linear}>> -> tensor<128x1xi32, #linear>
    %cols2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #linear}>> -> tensor<1x4xi32, #linear>
    %c256 = arith.constant dense<256> : tensor<128x1xi32, #linear>
    %c1000 = arith.constant dense<1000> : tensor<1x4xi32, #linear>
    %c1t = arith.constant dense<1> : tensor<1x4xi32, #linear>
    %row = arith.muli %rows2d, %c256 : tensor<128x1xi32, #linear>
    %p = tt.splat %arg : i32 -> tensor<1x4xi32, #linear>
    %pm1 = arith.subi %p, %c1t : tensor<1x4xi32, #linear>
    %poison = arith.muli %p, %pm1 : tensor<1x4xi32, #linear>
    %cp0 = arith.muli %cols2d, %poison : tensor<1x4xi32, #linear>
    %cp = arith.muli %cp0, %c1000 : tensor<1x4xi32, #linear>
    %col = arith.addi %cols2d, %cp : tensor<1x4xi32, #linear>
    %row_b = tt.broadcast %row : tensor<128x1xi32, #linear> -> tensor<128x4xi32, #linear>
    %col_b = tt.broadcast %col : tensor<1x4xi32, #linear> -> tensor<128x4xi32, #linear>
    %off = arith.addi %row_b, %col_b : tensor<128x4xi32, #linear>
    // GFX950: %[[RET:.*]] = amdg.buffer_load
    // GFX950-NOT: contiguity =
    // GFX950: tt.return %[[RET]]
    %ret = amdg.buffer_load %ptr[%off] : tensor<128x4xi8, #linear>
    tt.return %ret : tensor<128x4xi8, #linear>
  }
}

// -----

// Lane-straddle soundness (regression for the lane-0-only over-claim). offsets =
// make_range % 6 (non-power-of-two) with a blocked layout where each thread
// holds 4 contiguous coords. Lane 0 sees coords 0,1,2,3 -> 0,1,2,3 (looks
// contig 4), but lane 1 sees 4,5,6,7 -> 4,5,0,1 (true contig 2). The analysis
// models the lane contribution symbolically and must stamp 2, not 4.
#blk_straddle = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: @vgpr_lane_straddle_mod6
  tt.func @vgpr_lane_straddle_mod6(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}) -> tensor<256xi32, #blk_straddle> {
    %r = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blk_straddle>
    %c6 = arith.constant dense<6> : tensor<256xi32, #blk_straddle>
    %m = arith.remsi %r, %c6 : tensor<256xi32, #blk_straddle>
    // GFX950: amdg.buffer_load {{.*}} {contiguity = 2 : i32}
    %ret = amdg.buffer_load %ptr[%m] : tensor<256xi32, #blk_straddle>
    tt.return %ret : tensor<256xi32, #blk_straddle>
  }
}

// -----

// Warp-straddle soundness. Similar to the lane-straddle test above, but the
// non-register contribution that crosses the `% 6` period comes from the warp
// dimension. Warp 0 sees cols 0,1,2,3 -> contig 4, but warp 1 sees cols
// 4,5,6,7 -> 4,5,0,1, so the theorem over all warps is only contig 2.
#linear_warp_straddle = #ttg.linear<{register = [[0, 1], [0, 2]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0]], warp = [[0, 4]], block = []}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: @vgpr_warp_straddle_mod6
  tt.func @vgpr_warp_straddle_mod6(%ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}) -> tensor<64x8xi32, #linear_warp_straddle> {
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear_warp_straddle}>>
    %cols = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #linear_warp_straddle}>>
    %rows2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #linear_warp_straddle}>> -> tensor<64x1xi32, #linear_warp_straddle>
    %cols2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 0, parent = #linear_warp_straddle}>> -> tensor<1x8xi32, #linear_warp_straddle>
    %c128 = arith.constant dense<128> : tensor<64x1xi32, #linear_warp_straddle>
    %c6 = arith.constant dense<6> : tensor<1x8xi32, #linear_warp_straddle>
    %row = arith.muli %rows2d, %c128 : tensor<64x1xi32, #linear_warp_straddle>
    %mod = arith.remui %cols2d, %c6 : tensor<1x8xi32, #linear_warp_straddle>
    %row_b = tt.broadcast %row : tensor<64x1xi32, #linear_warp_straddle> -> tensor<64x8xi32, #linear_warp_straddle>
    %mod_b = tt.broadcast %mod : tensor<1x8xi32, #linear_warp_straddle> -> tensor<64x8xi32, #linear_warp_straddle>
    %off = arith.addi %row_b, %mod_b : tensor<64x8xi32, #linear_warp_straddle>
    // GFX950: amdg.buffer_load {{.*}} {contiguity = 2 : i32}
    %ret = amdg.buffer_load %ptr[%off] : tensor<64x8xi32, #linear_warp_straddle>
    tt.return %ret : tensor<64x8xi32, #linear_warp_straddle>
  }
}

// -----

// Direct-to-LDS uses the same shared global-memory contiguity proof as VGPR
// buffer_load. Here AxisInfo alone cannot prove that `(r * 16 + scalar) % 16`
// is register-invariant, but the symbolic proof drops the register-varying
// multiple of 16 and the LDS path stamps contiguity 4.
#blocked_lds_symbolic = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>
#shared_lds_symbolic = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // GFX950-LABEL: @lds_mod_residual_keying_scalar_1d
  tt.func @lds_mod_residual_keying_scalar_1d(%ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg: i32, %dst: !ttg.memdesc<256xi8, #shared_lds_symbolic, #smem, mutable>) {
    %r = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #blocked_lds_symbolic>
    %c16 = arith.constant dense<16> : tensor<256xi32, #blocked_lds_symbolic>
    %p = tt.splat %arg : i32 -> tensor<256xi32, #blocked_lds_symbolic>
    %wide = arith.muli %r, %c16 : tensor<256xi32, #blocked_lds_symbolic>
    %sum = arith.addi %wide, %p : tensor<256xi32, #blocked_lds_symbolic>
    %m = arith.remui %sum, %c16 : tensor<256xi32, #blocked_lds_symbolic>
    %off = arith.addi %r, %m : tensor<256xi32, #blocked_lds_symbolic>
    // GFX950: amdg.buffer_load_to_local {{.*}} into {{.*}} {contiguity = 4 : i32}
    %tok = amdg.buffer_load_to_local %ptr[%off] into %dst : <i8>[tensor<256xi32, #blocked_lds_symbolic>] -> <256xi8, #shared_lds_symbolic, #smem, mutable>
    tt.return
  }
}
