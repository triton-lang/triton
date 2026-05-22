// RUN: triton-opt %s -split-input-file -tritongpu-optimize-trans-layout | FileCheck %s

// The post-trans `ttg.convert_layout` here is the cross-warp conversion that
// would otherwise lower to shared memory. The pass propagates the store
// encoding (#blocked2) backward through the trans: the load and all upstream
// pointer arithmetic / elementwise ops are retyped to L_new, the trans's
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

// Same idea as above, but the shape-permuting op is `tt.reshape` (with
// allowReorder), not `tt.trans`. The pass should still fire and eliminate
// the post-reshape convert.

#blocked_load = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked_reshape = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_store = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: @reshape_eliminates_convert
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.reshape
  // CHECK: tt.store
  tt.func public @reshape_eliminates_convert(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
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

// Sanity: a `ttg.convert_layout` with no tt.trans / tt.reshape (reorder) in
// the upstream slice must be left alone. This is the territory of
// RemoveLayoutConversions, not this pass.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: @convert_without_trans_unchanged
  // CHECK: ttg.convert_layout
  tt.func public @convert_without_trans_unchanged(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %r0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %r1 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %r0_2d = tt.expand_dims %r0 {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<32x1xi32, #blocked>
    %off = tt.broadcast %r0_2d : tensor<32x1xi32, #blocked> -> tensor<32x32xi32, #blocked>
    %p_in = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked>
    %p_in2 = tt.addptr %p_in, %off : tensor<32x32x!tt.ptr<f32>, #blocked>, tensor<32x32xi32, #blocked>
    %x = tt.load %p_in2 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %x_c = ttg.convert_layout %x : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked1>
    %r0_b = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %r0_b_2d = tt.expand_dims %r0_b {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<32x1xi32, #blocked1>
    %off_b = tt.broadcast %r0_b_2d : tensor<32x1xi32, #blocked1> -> tensor<32x32xi32, #blocked1>
    %p_out = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<32x32x!tt.ptr<f32>, #blocked1>
    %p_out2 = tt.addptr %p_out, %off_b : tensor<32x32x!tt.ptr<f32>, #blocked1>, tensor<32x32xi32, #blocked1>
    tt.store %p_out2, %x_c : tensor<32x32x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
