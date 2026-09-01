// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-optimize-cta-locality | FileCheck %s

#orig = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#planned = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [1, 0]]}>

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @preserve_yield_layout
  // CHECK-SAME: %arg1: tensor<128x128xf32, #[[$SRC:[a-zA-Z0-9_]+]]>, %arg2: tensor<128x128xf32, #[[$DST:[a-zA-Z0-9_]+]]>
  // CHECK: %[[RESULT:.*]] = scf.if %arg0
  // CHECK: %[[YIELD:.*]] = ttg.convert_layout %arg1 : tensor<128x128xf32, #[[$SRC]]> -> tensor<128x128xf32, #[[$DST]]>
  // CHECK: scf.yield %[[YIELD]] : tensor<128x128xf32, #[[$DST]]>
  // CHECK: tt.return %[[RESULT]] : tensor<128x128xf32, #[[$DST]]>
  tt.func @preserve_yield_layout(%cond: i1, %src: tensor<128x128xf32, #planned>, %init: tensor<128x128xf32, #orig>) -> tensor<128x128xf32, #orig> {
    %result = scf.if %cond -> tensor<128x128xf32, #orig> {
      %yield = ttg.convert_layout %src : tensor<128x128xf32, #planned> -> tensor<128x128xf32, #orig>
      scf.yield %yield : tensor<128x128xf32, #orig>
    } else {
      scf.yield %init : tensor<128x128xf32, #orig>
    }
    tt.return %result : tensor<128x128xf32, #orig>
  }
}

// -----

#orig = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#planned = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [1, 0]]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #planned}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #planned}>

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @store_dot_result
  // CHECK-SAME: %arg0: tensor<128x128x!tt.ptr<f32>, #[[$PLANNED:[a-zA-Z0-9_]+]]>
  // CHECK-SAME: %arg1: tensor<128x128xi1, #[[$PLANNED]]>
  // CHECK: %[[DOT:.*]] = tt.dot {{.*}} -> tensor<128x128xf32, #[[$PLANNED]]>
  // CHECK: %[[MASK_ORIG:.*]] = ttg.convert_layout %arg1 : tensor<128x128xi1, #[[$PLANNED]]> -> tensor<128x128xi1, #[[$ORIG:[a-zA-Z0-9_]+]]>
  // CHECK: %[[DOT_ORIG:.*]] = ttg.convert_layout %[[DOT]] : tensor<128x128xf32, #[[$PLANNED]]> -> tensor<128x128xf32, #[[$ORIG]]>
  // CHECK: %[[PTRS_TARGET:.*]] = ttg.convert_layout %arg0 : tensor<128x128x!tt.ptr<f32>, #[[$PLANNED]]> -> tensor<128x128x!tt.ptr<f32>, #[[$TARGET:[a-zA-Z0-9_]+]]>
  // CHECK: %[[DOT_TARGET:.*]] = ttg.convert_layout %[[DOT_ORIG]] : tensor<128x128xf32, #[[$ORIG]]> -> tensor<128x128xf32, #[[$TARGET]]>
  // CHECK: %[[MASK_TARGET:.*]] = ttg.convert_layout %[[MASK_ORIG]] : tensor<128x128xi1, #[[$ORIG]]> -> tensor<128x128xi1, #[[$TARGET]]>
  // CHECK: tt.store %[[PTRS_TARGET]], %[[DOT_TARGET]], %[[MASK_TARGET]] : tensor<128x128x!tt.ptr<f32>, #[[$TARGET]]>
  tt.func @store_dot_result(
    %ptrs: tensor<128x128x!tt.ptr<f32>, #planned>,
    %mask: tensor<128x128xi1, #planned>,
    %c: tensor<128x128xf32, #planned>) {
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #dot_a>
    %b = arith.constant dense<2.000000e+00> : tensor<32x128xf16, #dot_b>
    %dot = tt.dot %a, %b, %c : tensor<128x32xf16, #dot_a> * tensor<32x128xf16, #dot_b> -> tensor<128x128xf32, #planned>
    %ptrs_orig = ttg.convert_layout %ptrs : tensor<128x128x!tt.ptr<f32>, #planned> -> tensor<128x128x!tt.ptr<f32>, #orig>
    %mask_orig = ttg.convert_layout %mask : tensor<128x128xi1, #planned> -> tensor<128x128xi1, #orig>
    %dot_orig = ttg.convert_layout %dot : tensor<128x128xf32, #planned> -> tensor<128x128xf32, #orig>
    tt.store %ptrs_orig, %dot_orig, %mask_orig : tensor<128x128x!tt.ptr<f32>, #orig>
    tt.return
  }

  // CHECK-LABEL: tt.func @store_dot_with_splat_ptr
  // CHECK-SAME: %arg0: !tt.ptr<f32>
  // CHECK-SAME: %arg1: tensor<128x128xi1, #[[$SPLAT_PLANNED:[a-zA-Z0-9_]+]]>
  // CHECK: %[[DOT:.*]] = tt.dot {{.*}} -> tensor<128x128xf32, #[[$SPLAT_PLANNED]]>
  // CHECK: %[[PTRS_ORIG:.*]] = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #[[$SPLAT_ORIG:[a-zA-Z0-9_]+]]>
  // CHECK: %[[DOT_ORIG:.*]] = ttg.convert_layout %[[DOT]] : tensor<128x128xf32, #[[$SPLAT_PLANNED]]> -> tensor<128x128xf32, #[[$SPLAT_ORIG]]>
  // CHECK: %[[PTRS_TARGET:.*]] = ttg.convert_layout %[[PTRS_ORIG]] : tensor<128x128x!tt.ptr<f32>, #[[$SPLAT_ORIG]]> -> tensor<128x128x!tt.ptr<f32>, #[[$SPLAT_TARGET:[a-zA-Z0-9_]+]]>
  // CHECK: %[[DOT_TARGET:.*]] = ttg.convert_layout %[[DOT_ORIG]] : tensor<128x128xf32, #[[$SPLAT_ORIG]]> -> tensor<128x128xf32, #[[$SPLAT_TARGET]]>
  // CHECK: %[[MASK_TARGET:.*]] = ttg.convert_layout %arg1 : tensor<128x128xi1, #[[$SPLAT_PLANNED]]> -> tensor<128x128xi1, #[[$SPLAT_TARGET]]>
  // CHECK: tt.store %[[PTRS_TARGET]], %[[DOT_TARGET]], %[[MASK_TARGET]] : tensor<128x128x!tt.ptr<f32>, #[[$SPLAT_TARGET]]>
  tt.func @store_dot_with_splat_ptr(
    %ptr: !tt.ptr<f32>,
    %mask: tensor<128x128xi1, #planned>,
    %c: tensor<128x128xf32, #planned>) {
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #dot_a>
    %b = arith.constant dense<2.000000e+00> : tensor<32x128xf16, #dot_b>
    %dot = tt.dot %a, %b, %c : tensor<128x32xf16, #dot_a> * tensor<32x128xf16, #dot_b> -> tensor<128x128xf32, #planned>
    %ptrs_orig = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #orig>
    %mask_orig = ttg.convert_layout %mask : tensor<128x128xi1, #planned> -> tensor<128x128xi1, #orig>
    %dot_orig = ttg.convert_layout %dot : tensor<128x128xf32, #planned> -> tensor<128x128xf32, #orig>
    tt.store %ptrs_orig, %dot_orig, %mask_orig : tensor<128x128x!tt.ptr<f32>, #orig>
    tt.return
  }
}

// -----

#orig = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#planned = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [1, 0]]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #planned}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #planned}>

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @descriptor_store_dot_result
  // CHECK: %[[DOT:.*]] = tt.dot {{.*}} -> tensor<128x128xf32, #[[$DESC_PLANNED:[a-zA-Z0-9_]+]]>
  // CHECK: %[[DOT_TARGET:.*]] = ttg.convert_layout %[[DOT]] : tensor<128x128xf32, #[[$DESC_PLANNED]]> -> tensor<128x128xf32, #[[$DESC_TARGET:[a-zA-Z0-9_]+]]>
  // CHECK: tt.descriptor_store %{{.*}}[%{{.*}}, %{{.*}}], %[[DOT_TARGET]] : !tt.tensordesc<128x128xf32>, tensor<128x128xf32, #[[$DESC_TARGET]]>
  tt.func @descriptor_store_dot_result(
    %desc: !tt.tensordesc<128x128xf32>,
    %i: i32,
    %j: i32,
    %c: tensor<128x128xf32, #planned>) {
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #dot_a>
    %b = arith.constant dense<2.000000e+00> : tensor<32x128xf16, #dot_b>
    %dot = tt.dot %a, %b, %c : tensor<128x32xf16, #dot_a> * tensor<32x128xf16, #dot_b> -> tensor<128x128xf32, #planned>
    %dot_orig = ttg.convert_layout %dot : tensor<128x128xf32, #planned> -> tensor<128x128xf32, #orig>
    tt.descriptor_store %desc[%i, %j], %dot_orig : !tt.tensordesc<128x128xf32>, tensor<128x128xf32, #orig>
    tt.return
  }
}

// -----

#src_orig = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#src_planned = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#out_orig = #ttg.slice<{dim = 1, parent = #src_orig}>
#out_planned = #ttg.slice<{dim = 1, parent = #src_planned}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @store_reduce_result
  // CHECK-SAME: %arg0: tensor<128x64xf32, #[[$REDUCE_SRC_PLANNED:[a-zA-Z0-9_]+]]>
  // CHECK-SAME: %arg1: tensor<128x!tt.ptr<f32>, #[[$REDUCE_OUT_PLANNED:.*]]>
  // CHECK: %[[RED:.*]] = "tt.reduce"(%arg0) <{axis = 1 : i32}>
  // CHECK: }) : (tensor<128x64xf32, #[[$REDUCE_SRC_PLANNED]]>) -> tensor<128xf32, #[[$REDUCE_OUT_PLANNED]]>
  // CHECK: %[[RED_ORIG:.*]] = ttg.convert_layout %[[RED]] : tensor<128xf32, #[[$REDUCE_OUT_PLANNED]]> -> tensor<128xf32, #[[$REDUCE_OUT_ORIG:.*]]>
  // CHECK: %[[PTR_TARGET:.*]] = ttg.convert_layout %arg1 : tensor<128x!tt.ptr<f32>, #[[$REDUCE_OUT_PLANNED]]> -> tensor<128x!tt.ptr<f32>, #[[$REDUCE_OUT_TARGET:.*]]>
  // CHECK: %[[RED_TARGET:.*]] = ttg.convert_layout %[[RED_ORIG]] : tensor<128xf32, #[[$REDUCE_OUT_ORIG]]> -> tensor<128xf32, #[[$REDUCE_OUT_TARGET]]>
  // CHECK: tt.store %[[PTR_TARGET]], %[[RED_TARGET]] : tensor<128x!tt.ptr<f32>, #[[$REDUCE_OUT_TARGET]]>
  tt.func @store_reduce_result(
    %src: tensor<128x64xf32, #src_planned>,
    %out: tensor<128x!tt.ptr<f32>, #out_planned>) {
    %red = "tt.reduce"(%src) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<128x64xf32, #src_planned>) -> tensor<128xf32, #out_planned>
    %out_orig = ttg.convert_layout %out : tensor<128x!tt.ptr<f32>, #out_planned> -> tensor<128x!tt.ptr<f32>, #out_orig>
    %red_orig = ttg.convert_layout %red : tensor<128xf32, #out_planned> -> tensor<128xf32, #out_orig>
    tt.store %out_orig, %red_orig : tensor<128x!tt.ptr<f32>, #out_orig>
    tt.return
  }
}

// -----

#blocked_src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0], [1]]}>
#blocked_dst = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1], [0]]}>
#linear_dst = #ttg.linear<{register = [], lane = [[1], [2], [4], [8], [16]], warp = [[32], [64]], block = [[128], [0]]}>
#nested_src_parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [4, 1, 1], order = [0, 1, 2], CGALayout = [[0, 1, 0], [1, 0, 0]]}>
#nested_src_middle = #ttg.slice<{dim = 1, parent = #nested_src_parent}>
#nested_src = #ttg.slice<{dim = 1, parent = #nested_src_middle}>
#nested_dst_parent = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 4, 1], order = [1, 0, 2], CGALayout = [[0, 1, 0], [1, 0, 0]]}>
#nested_dst_middle = #ttg.slice<{dim = 0, parent = #nested_dst_parent}>
#nested_dst = #ttg.slice<{dim = 1, parent = #nested_dst_middle}>

// Preserve the replicated CTA basis before the split CTA basis.
// CHECK-DAG: #[[$BLOCK_TARGET:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = {{\[\[0\], \[1\]\]}}}>
// CHECK-DAG: #[[$NESTED_TARGET_PARENT:.*]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 32, 1], warpsPerCTA = [1, 4, 1], order = [1, 0, 2], CGALayout = {{\[\[0, 0, 0\], \[0, 1, 0\]\]}}}>
// CHECK-DAG: #[[$LINEAR_TARGET:.*]] = #ttg.linear<{register = [], lane = {{.*}}, warp = {{.*}}, block = {{\[\[0\], \[128\]\]}}}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @store_nested_slice_as_blocked
  // CHECK: tt.descriptor_store %arg0[%arg1], %{{.*}} : !tt.tensordesc<256xf32>, tensor<256xf32, #[[$BLOCK_TARGET]]>
  tt.func @store_nested_slice_as_blocked(%desc: !tt.tensordesc<256xf32>, %i: i32, %src: tensor<256xf32, #nested_src>) {
    %converted = ttg.convert_layout %src : tensor<256xf32, #nested_src> -> tensor<256xf32, #blocked_dst>
    tt.descriptor_store %desc[%i], %converted : !tt.tensordesc<256xf32>, tensor<256xf32, #blocked_dst>
    tt.return
  }

  // CHECK-LABEL: tt.func @store_blocked_as_nested_slice
  // CHECK: tt.descriptor_store %arg0[%arg1], %{{.*}} : !tt.tensordesc<256xf32>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #ttg.slice<{dim = 0, parent = #[[$NESTED_TARGET_PARENT]]}>}>>
  tt.func @store_blocked_as_nested_slice(%desc: !tt.tensordesc<256xf32>, %i: i32, %src: tensor<256xf32, #blocked_src>) {
    %converted = ttg.convert_layout %src : tensor<256xf32, #blocked_src> -> tensor<256xf32, #nested_dst>
    tt.descriptor_store %desc[%i], %converted : !tt.tensordesc<256xf32>, tensor<256xf32, #nested_dst>
    tt.return
  }

  // CHECK-LABEL: tt.func @store_blocked_as_linear
  // CHECK: tt.descriptor_store %arg0[%arg1], %{{.*}} : !tt.tensordesc<256xf32>, tensor<256xf32, #[[$LINEAR_TARGET]]>
  tt.func @store_blocked_as_linear(%desc: !tt.tensordesc<256xf32>, %i: i32, %src: tensor<256xf32, #blocked_src>) {
    %converted = ttg.convert_layout %src : tensor<256xf32, #blocked_src> -> tensor<256xf32, #linear_dst>
    tt.descriptor_store %desc[%i], %converted : !tt.tensordesc<256xf32>, tensor<256xf32, #linear_dst>
    tt.return
  }
}

// -----

#src_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 1], [1, 0]]}>
#dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = [[0, 1], [1, 0]]}>
#src = #ttg.slice<{dim = 1, parent = #src_parent}>
#dst = #ttg.slice<{dim = 0, parent = #dst_parent}>

// CTA coordinates belong to the logical tensor, not the source's parent axes.
// CHECK-DAG: #[[$SLICE_TARGET_PARENT:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0], CGALayout = {{\[\[0, 0\], \[0, 1\]\]}}}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @store_between_slice_axes
  // CHECK: tt.descriptor_store %arg0[%arg1], %{{.*}} : !tt.tensordesc<256xf32>, tensor<256xf32, #ttg.slice<{dim = 0, parent = #[[$SLICE_TARGET_PARENT]]}>>
  tt.func @store_between_slice_axes(%desc: !tt.tensordesc<256xf32>, %i: i32, %src: tensor<256xf32, #src>) {
    %converted = ttg.convert_layout %src : tensor<256xf32, #src> -> tensor<256xf32, #dst>
    tt.descriptor_store %desc[%i], %converted : !tt.tensordesc<256xf32>, tensor<256xf32, #dst>
    tt.return
  }
}
