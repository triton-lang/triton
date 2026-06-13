// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

// The post-trans `ttg.convert_layout` here is a cross-warp conversion that
// would otherwise lower to shared memory. The load is "expensive" by the
// anchor heuristic (128 elements == 128 threads), but it is small and poorly
// coalesced under its own layout, so backward rematerialization may retype it
// to the encoding inferred backward through the trans: the load and all
// upstream pointer arithmetic / elementwise ops are rewritten, the trans's
// result becomes #blocked2 directly, and the convert is removed.

#blocked = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 4, 1], order = [0, 2, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 4, 8], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [2, 4, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: [[L_NEW:#[a-z0-9_]+]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 4, 2], warpsPerCTA = [1, 1, 4], order = [1, 0, 2]}>
  // CHECK-DAG: [[STORE:#[a-z0-9_]+]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [2, 4, 4], warpsPerCTA = [4, 1, 1], order = [2, 1, 0]}>
  // CHECK: @permute_eliminates_convert
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.load {{.*}} : tensor<4x4x8x!tt.ptr<f32>, [[L_NEW]]>
  // CHECK: tt.trans {{.*}} {order = array<i32: 2, 0, 1>} : tensor<4x4x8xf32, [[L_NEW]]> -> tensor<8x4x4xf32, [[STORE]]>
  // CHECK: tt.store {{.*}} : tensor<8x4x4x!tt.ptr<f32>, [[STORE]]>
  tt.func public @permute_eliminates_convert(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %c1 = arith.constant dense<1.000000e+00> : tensor<4x4x8xf32, #blocked1>
    %r0_sl = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>>
    %r0_2d = tt.expand_dims %r0_sl {axis = 1 : i32} : tensor<4xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked1}>}>> -> tensor<4x1xi32, #ttg.slice<{dim = 2, parent = #blocked1}>>
    %r0_3d = tt.expand_dims %r0_2d {axis = 2 : i32} : tensor<4x1xi32, #ttg.slice<{dim = 2, parent = #blocked1}>> -> tensor<4x1x1xi32, #blocked1>
    %off = tt.broadcast %r0_3d : tensor<4x1x1xi32, #blocked1> -> tensor<4x4x8xi32, #blocked1>
    %p_in = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<4x4x8x!tt.ptr<f32>, #blocked1>
    %p_in2 = tt.addptr %p_in, %off : tensor<4x4x8x!tt.ptr<f32>, #blocked1>, tensor<4x4x8xi32, #blocked1>
    %x = tt.load %p_in2 : tensor<4x4x8x!tt.ptr<f32>, #blocked1>
    %x1 = arith.addf %x, %c1 : tensor<4x4x8xf32, #blocked1>
    %x_t = tt.trans %x1 {order = array<i32: 2, 0, 1>} : tensor<4x4x8xf32, #blocked1> -> tensor<8x4x4xf32, #blocked>
    %x_c = ttg.convert_layout %x_t : tensor<8x4x4xf32, #blocked> -> tensor<8x4x4xf32, #blocked2>
    %off_out_sl = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>>
    %off_out_2d = tt.expand_dims %off_out_sl {axis = 1 : i32} : tensor<8xi32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 2, parent = #blocked2}>}>> -> tensor<8x1xi32, #ttg.slice<{dim = 2, parent = #blocked2}>>
    %off_out_3d = tt.expand_dims %off_out_2d {axis = 2 : i32} : tensor<8x1xi32, #ttg.slice<{dim = 2, parent = #blocked2}>> -> tensor<8x1x1xi32, #blocked2>
    %off_out = tt.broadcast %off_out_3d : tensor<8x1x1xi32, #blocked2> -> tensor<8x4x4xi32, #blocked2>
    %p_out = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<8x4x4x!tt.ptr<f32>, #blocked2>
    %p_out2 = tt.addptr %p_out, %off_out : tensor<8x4x4x!tt.ptr<f32>, #blocked2>, tensor<8x4x4xi32, #blocked2>
    tt.store %p_out2, %x_c : tensor<8x4x4x!tt.ptr<f32>, #blocked2>
    tt.return
  }
}

// -----

// Negative case: backward propagation through the `tt.reshape` infers a
// `#ttg.linear` encoding for the load. The coalescing comparison is only
// conclusive for blocked encodings, so the rematerialization is refused and
// the convert stays. (Extending the run-length computation to linear layouts
// could enable this case in the future.)

#blocked_load = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked_reshape = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_store = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: @reshape_linear_src_refused
  // CHECK: tt.reshape
  // CHECK: ttg.convert_layout
  // CHECK: tt.store
  tt.func public @reshape_linear_src_refused(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %r0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked_load>
    %p_in = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked_load>
    %p_in2 = tt.addptr %p_in, %r0 : tensor<128x!tt.ptr<f32>, #blocked_load>, tensor<128xi32, #blocked_load>
    %x = tt.load %p_in2 : tensor<128x!tt.ptr<f32>, #blocked_load>
    %x_r = tt.reshape %x allow_reorder : tensor<128xf32, #blocked_load> -> tensor<8x16xf32, #blocked_reshape>
    %x_c = ttg.convert_layout %x_r : tensor<8x16xf32, #blocked_reshape> -> tensor<8x16xf32, #blocked_store>
    %r1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_store}>>
    %r1_2d = tt.expand_dims %r1 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked_store}>> -> tensor<1x16xi32, #blocked_store>
    %off_out = tt.broadcast %r1_2d : tensor<1x16xi32, #blocked_store> -> tensor<8x16xi32, #blocked_store>
    %p_out = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>, #blocked_store>
    %p_out2 = tt.addptr %p_out, %off_out : tensor<8x16x!tt.ptr<f32>, #blocked_store>, tensor<8x16xi32, #blocked_store>
    tt.store %p_out2, %x_c : tensor<8x16x!tt.ptr<f32>, #blocked_store>
    tt.return
  }
}

// -----

// Negative case: the load is small enough, but it is perfectly coalesced
// under its own layout (32 contiguous elements along dim 1) and the target
// encoding would shrink the coalesced run to 1. The rematerialization must be
// refused and the convert kept.

#blocked_row = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_col = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: @uncoalescing_retype_refused
  // CHECK: tt.load
  // CHECK: ttg.convert_layout
  // CHECK: tt.store
  tt.func public @uncoalescing_retype_refused(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %c32 = arith.constant dense<32> : tensor<32x1xi32, #blocked_row>
    %r0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked_row}>>
    %r1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_row}>>
    %r0_2d = tt.expand_dims %r0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked_row}>> -> tensor<32x1xi32, #blocked_row>
    %r0_scaled = arith.muli %r0_2d, %c32 : tensor<32x1xi32, #blocked_row>
    %r1_2d = tt.expand_dims %r1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_row}>> -> tensor<1x32xi32, #blocked_row>
    %off_rows = tt.broadcast %r0_scaled : tensor<32x1xi32, #blocked_row> -> tensor<32x32xi32, #blocked_row>
    %off_cols = tt.broadcast %r1_2d : tensor<1x32xi32, #blocked_row> -> tensor<32x32xi32, #blocked_row>
    %off = arith.addi %off_rows, %off_cols : tensor<32x32xi32, #blocked_row>
    %p_in = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked_row>
    %p_in2 = tt.addptr %p_in, %off : tensor<32x32x!tt.ptr<f32>, #blocked_row>, tensor<32x32xi32, #blocked_row>
    %x = tt.load %p_in2 : tensor<32x32x!tt.ptr<f32>, #blocked_row>
    %x_c = ttg.convert_layout %x : tensor<32x32xf32, #blocked_row> -> tensor<32x32xf32, #blocked_col>
    %c32b = arith.constant dense<32> : tensor<32x1xi32, #blocked_col>
    %s0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked_col}>>
    %s1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_col}>>
    %s0_2d = tt.expand_dims %s0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked_col}>> -> tensor<32x1xi32, #blocked_col>
    %s0_scaled = arith.muli %s0_2d, %c32b : tensor<32x1xi32, #blocked_col>
    %s1_2d = tt.expand_dims %s1 {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked_col}>> -> tensor<1x32xi32, #blocked_col>
    %off_b_rows = tt.broadcast %s0_scaled : tensor<32x1xi32, #blocked_col> -> tensor<32x32xi32, #blocked_col>
    %off_b_cols = tt.broadcast %s1_2d : tensor<1x32xi32, #blocked_col> -> tensor<32x32xi32, #blocked_col>
    %off_b = arith.addi %off_b_rows, %off_b_cols : tensor<32x32xi32, #blocked_col>
    %p_out = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked_col>
    %p_out2 = tt.addptr %p_out, %off_b : tensor<32x32x!tt.ptr<f32>, #blocked_col>, tensor<32x32xi32, #blocked_col>
    tt.store %p_out2, %x_c : tensor<32x32x!tt.ptr<f32>, #blocked_col>
    tt.return
  }
}

// -----

// Negative case: the conversion target would even improve coalescing, but the
// tensor exceeds the small-load cap (64x64 = 4096 elements > 8 elements per
// thread for 4 warps), so the load keeps its layout and the convert stays.

#blocked_big = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_big_t = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: @large_load_retype_refused
  // CHECK: tt.load
  // CHECK: ttg.convert_layout
  // CHECK: tt.store
  tt.func public @large_load_retype_refused(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %r0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_big}>>
    %r0_2d = tt.expand_dims %r0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_big}>> -> tensor<64x1xi32, #blocked_big>
    %off = tt.broadcast %r0_2d : tensor<64x1xi32, #blocked_big> -> tensor<64x64xi32, #blocked_big>
    %p_in = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked_big>
    %p_in2 = tt.addptr %p_in, %off : tensor<64x64x!tt.ptr<f32>, #blocked_big>, tensor<64x64xi32, #blocked_big>
    %x = tt.load %p_in2 : tensor<64x64x!tt.ptr<f32>, #blocked_big>
    %x_c = ttg.convert_layout %x : tensor<64x64xf32, #blocked_big> -> tensor<64x64xf32, #blocked_big_t>
    %s0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_big_t}>>
    %s0_2d = tt.expand_dims %s0 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_big_t}>> -> tensor<64x1xi32, #blocked_big_t>
    %off_b = tt.broadcast %s0_2d : tensor<64x1xi32, #blocked_big_t> -> tensor<64x64xi32, #blocked_big_t>
    %p_out = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>, #blocked_big_t>
    %p_out2 = tt.addptr %p_out, %off_b : tensor<64x64x!tt.ptr<f32>, #blocked_big_t>, tensor<64x64xi32, #blocked_big_t>
    tt.store %p_out2, %x_c : tensor<64x64x!tt.ptr<f32>, #blocked_big_t>
    tt.return
  }
}
