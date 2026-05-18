// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-optimize-cta-locality | FileCheck %s

#orig = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#planned = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [1, 0]]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #planned}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #planned}>

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @propagate_same_layout_group
  // CHECK-SAME: %arg0: tensor<128x128xf32, #[[$GROUP_PLANNED:[a-zA-Z0-9_]+]]>
  // CHECK-SAME: %arg1: tensor<128x128xf32, #[[$GROUP_PLANNED]]>
  // CHECK: ttg.convert_layout %arg0 : tensor<128x128xf32, #[[$GROUP_PLANNED]]> -> tensor<128x128xf32, #[[$GROUP_ORIG:[a-zA-Z0-9_]+]]>
  // CHECK: ttg.convert_layout %arg1 : tensor<128x128xf32, #[[$GROUP_PLANNED]]> -> tensor<128x128xf32, #[[$GROUP_ORIG]]>
  // CHECK: arith.addf {{.*}} : tensor<128x128xf32, #[[$GROUP_ORIG]]>
  tt.func @propagate_same_layout_group(
    %a: tensor<128x128xf32, #planned>,
    %b: tensor<128x128xf32, #planned>) {
    %a_orig = ttg.convert_layout %a : tensor<128x128xf32, #planned> -> tensor<128x128xf32, #orig>
    %b_orig = ttg.convert_layout %b : tensor<128x128xf32, #planned> -> tensor<128x128xf32, #orig>
    %sum = arith.addf %a_orig, %b_orig : tensor<128x128xf32, #orig>
    tt.return
  }

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
  // CHECK: %[[RED_PLANNED:.*]] = ttg.convert_layout %[[RED_ORIG]] : tensor<128xf32, #[[$REDUCE_OUT_ORIG]]> -> tensor<128xf32, #[[$REDUCE_OUT_PLANNED]]>
  // CHECK: tt.store %arg1, %[[RED_PLANNED]] : tensor<128x!tt.ptr<f32>, #[[$REDUCE_OUT_PLANNED]]>
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

#orig = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#planned = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [1, 0]]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #planned}>

module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @skip_convert_from_dot_operand
  // CHECK: ttg.convert_layout %arg0 : tensor<128x32xf16, {{#ttg\.dot_op<.*>}}> -> tensor<128x32xf16, #[[$ORIG:[a-zA-Z0-9_]+]]>
  // CHECK: ttg.convert_layout %{{.*}} : tensor<128x32xf16, #[[$ORIG]]> -> tensor<128x32xf16, {{#ttg\.dot_op<.*>}}>
  tt.func @skip_convert_from_dot_operand(%a: tensor<128x32xf16, #dot_a>) {
    %a_orig = ttg.convert_layout %a : tensor<128x32xf16, #dot_a> -> tensor<128x32xf16, #orig>
    %a_dot = ttg.convert_layout %a_orig : tensor<128x32xf16, #orig> -> tensor<128x32xf16, #dot_a>
    tt.return
  }

  // CHECK-LABEL: tt.func @skip_convert_to_dot_operand
  // CHECK: ttg.convert_layout %arg0 : tensor<128x32xf16, #[[$PLANNED:[a-zA-Z0-9_]+]]> -> tensor<128x32xf16, {{#ttg\.dot_op<.*>}}>
  // CHECK: ttg.convert_layout %{{.*}} : tensor<128x32xf16, {{#ttg\.dot_op<.*>}}> -> tensor<128x32xf16, #[[$ORIG_2:[a-zA-Z0-9_]+]]>
  tt.func @skip_convert_to_dot_operand(%a: tensor<128x32xf16, #planned>) {
    %a_dot = ttg.convert_layout %a : tensor<128x32xf16, #planned> -> tensor<128x32xf16, #dot_a>
    %a_orig = ttg.convert_layout %a_dot : tensor<128x32xf16, #dot_a> -> tensor<128x32xf16, #orig>
    tt.return
  }
}
