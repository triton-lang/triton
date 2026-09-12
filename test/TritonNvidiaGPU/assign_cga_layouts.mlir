// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-assign-cga-layouts | FileCheck %s
// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-assign-cga-layouts -tritongpu-remove-layout-conversions | FileCheck %s --check-prefix=E2E
// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-assign-cga-layouts -tritongpu-remove-layout-conversions -triton-nvidia-gpu-optimize-cta-locality -tritongpu-remove-layout-conversions | FileCheck %s --check-prefix=PIPELINE

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduce_loaded_plus_constant
  // CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<128x128x!tt.ptr<f32>, #[[$PLANNED:.*]]>
  // CHECK: %[[CONST:.*]] = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #[[$ORIG:.*]]>
  // CHECK: %[[CONVERT:.*]] = ttg.convert_layout %[[CONST]] : tensor<128x128xf32, #[[$ORIG]]> -> tensor<128x128xf32, #[[$PLANNED]]>
  // CHECK: %[[SUM:.*]] = arith.addf %[[LOAD]], %[[CONVERT]] : tensor<128x128xf32, #[[$PLANNED]]>
  // CHECK: "tt.reduce"(%[[SUM]])
  // E2E-LABEL: tt.func @reduce_loaded_plus_constant
  // E2E: %[[CONST:.*]] = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #[[$PLANNED:.*]]>
  // E2E: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<128x128x!tt.ptr<f32>, #[[$PLANNED]]>
  // E2E: %[[SUM:.*]] = arith.addf %[[LOAD]], %[[CONST]] : tensor<128x128xf32, #[[$PLANNED]]>
  // E2E: "tt.reduce"(%[[SUM]])
  tt.func @reduce_loaded_plus_constant(%ptrs: tensor<128x128x!tt.ptr<f32>, #blocked>) -> tensor<128xf32, #slice> {
    %x = tt.load %ptrs : tensor<128x128x!tt.ptr<f32>, #blocked>
    %c = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %sum = arith.addf %x, %c : tensor<128x128xf32, #blocked>
    %r = "tt.reduce"(%sum) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %v = arith.addf %lhs, %rhs : f32
      tt.reduce.return %v : f32
    }) : (tensor<128x128xf32, #blocked>) -> tensor<128xf32, #slice>
    tt.return %r : tensor<128xf32, #slice>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0], [0]]}>

  // CHECK-DAG: #[[$ORIG:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = {{\[\[0\], \[0\]\]}}}>
  // CHECK-DAG: #[[$REDUCE:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = {{\[\[1\], \[2\]\]}}}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduce_1d_split_ctas
  // CHECK: %[[CVT:.*]] = ttg.convert_layout %{{.*}} : tensor<65536xf32, #[[$ORIG]]> -> tensor<65536xf32, #[[$REDUCE]]>
  // CHECK: "tt.reduce"(%[[CVT]]) <{axis = 0 : i32}>
  // CHECK: tt.reduce.return %{{.*}} : f32
  // CHECK-NEXT: }) : (tensor<65536xf32, #[[$REDUCE]]>) -> f32
  tt.func @reduce_1d_split_ctas() -> f32 {
    %cst = arith.constant dense<0.000000e+00> : tensor<65536xf32, #blocked>
    %red = "tt.reduce"(%cst) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<65536xf32, #blocked>) -> f32
    tt.return %red : f32
  }
}

// -----

#dot_default = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1], [0, 2]]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dot_default}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dot_default}>

// CHECK-DAG: #[[$DOT_DEFAULT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\], \[0, 2\]\]}}}>
// CHECK-DAG: #[[$DOT_OPT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\], \[1, 0\]\]}}}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @dot_split_mn
  // CHECK: ttg.convert_layout %{{.*}} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_DEFAULT]]}>> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_OPT]]}>>
  // CHECK: ttg.convert_layout %{{.*}} : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DOT_DEFAULT]]}>> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DOT_OPT]]}>>
  // CHECK: %[[D:.*]] = ttg.convert_layout %{{.*}} : tensor<128x128xf32, #[[$DOT_DEFAULT]]> -> tensor<128x128xf32, #[[$DOT_OPT]]>
  // CHECK: %[[DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[D]] : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_OPT]]}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DOT_OPT]]}>> -> tensor<128x128xf32, #[[$DOT_OPT]]>
  // CHECK: ttg.convert_layout %[[DOT]] : tensor<128x128xf32, #[[$DOT_OPT]]> -> tensor<128x128xf32, #[[$DOT_DEFAULT]]>
  tt.func @dot_split_mn(%ptr: !tt.ptr<f32>) {
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #dot_a>
    %b = arith.constant dense<2.000000e+00> : tensor<32x128xf16, #dot_b>
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #dot_default>
    %dot = tt.dot %a, %b, %c : tensor<128x32xf16, #dot_a> * tensor<32x128xf16, #dot_b> -> tensor<128x128xf32, #dot_default>
    %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #dot_default>
    tt.store %ptrs, %dot : tensor<128x128x!tt.ptr<f32>, #dot_default>
    tt.return
  }
}

// -----

#load_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#load_src_b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#dot_default_load = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#dot_a_load = #ttg.dot_op<{opIdx = 0, parent = #dot_default_load}>
#dot_b_load = #ttg.dot_op<{opIdx = 1, parent = #dot_default_load}>

// CHECK-DAG: #[[$LOAD_ORIG:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// CHECK-DAG: #[[$LOAD_B_ORIG:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// CHECK-DAG: #[[$LOAD_B_PLANNED:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// CHECK-DAG: #[[$DOT_DEFAULT_LOAD:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// CHECK-DAG: #[[$DOT_OPT_LOAD:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// E2E-DAG: #[[$E2E_LOAD_ORIG:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// E2E-DAG: #[[$E2E_LOAD_B_ORIG:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// E2E-DAG: #[[$E2E_LOAD_B_PLANNED:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// E2E-DAG: #[[$E2E_DOT_OPT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @dot_rematerializes_exclusive_load_source
  // CHECK-NOT: tt.load
  // CHECK: %[[ORIG_LOAD:.*]] = tt.load %{{.*}} : tensor<128x32x!tt.ptr<f16>, #[[$LOAD_ORIG]]>
  // CHECK-NOT: tt.load
  // CHECK: %[[B_LOAD:.*]] = tt.load %{{.*}} : tensor<32x128x!tt.ptr<f16>, #[[$LOAD_B_PLANNED]]>
  // CHECK-NOT: tt.load
  // CHECK: tt.store %{{.*}}, %[[ORIG_LOAD]] : tensor<128x32x!tt.ptr<f16>, #[[$LOAD_ORIG]]>
  // CHECK: %[[B_SUM:.*]] = arith.addf %[[B_LOAD]], %[[B_LOAD]] : tensor<32x128xf16, #[[$LOAD_B_PLANNED]]>
  // CHECK: %[[B_DOT:.*]] = ttg.convert_layout %[[B_SUM]] : tensor<32x128xf16, #[[$LOAD_B_PLANNED]]> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DOT_OPT_LOAD]]}>>
  // CHECK: ttg.convert_layout %{{.*}} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_DEFAULT_LOAD]]}>> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_OPT_LOAD]]}>>
  // CHECK: tt.dot %{{.*}}, %[[B_DOT]], %{{.*}} :
  // E2E-LABEL: tt.func @dot_rematerializes_exclusive_load_source
  // E2E: %[[E2E_OLD_A:.*]] = tt.load %{{.*}} : tensor<128x32x!tt.ptr<f16>, #[[$E2E_LOAD_ORIG]]>
  // E2E: %[[E2E_B_PTRS:.*]] = ttg.convert_layout %{{.*}} : tensor<32x128x!tt.ptr<f16>, #[[$E2E_LOAD_B_ORIG]]> -> tensor<32x128x!tt.ptr<f16>, #[[$E2E_LOAD_B_PLANNED]]>
  // E2E: %[[E2E_B:.*]] = tt.load %[[E2E_B_PTRS]] : tensor<32x128x!tt.ptr<f16>, #[[$E2E_LOAD_B_PLANNED]]>
  // E2E: tt.store %{{.*}}, %[[E2E_OLD_A]] : tensor<128x32x!tt.ptr<f16>, #[[$E2E_LOAD_ORIG]]>
  // E2E: %[[E2E_SUM:.*]] = arith.addf %[[E2E_B]], %[[E2E_B]] : tensor<32x128xf16, #[[$E2E_LOAD_B_PLANNED]]>
  // E2E-DAG: %[[E2E_AD:.*]] = ttg.convert_layout %{{.*}} : tensor<128x32xf16, {{.*}}> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$E2E_DOT_OPT]]}>>
  // E2E-DAG: %[[E2E_BD:.*]] = ttg.convert_layout %[[E2E_SUM]] : tensor<32x128xf16, #[[$E2E_LOAD_B_PLANNED]]> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$E2E_DOT_OPT]]}>>
  // E2E: tt.dot %[[E2E_AD]], %[[E2E_BD]], %{{.*}} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$E2E_DOT_OPT]]}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$E2E_DOT_OPT]]}>> -> tensor<128x128xf32, #[[$E2E_DOT_OPT]]>
  tt.func @dot_rematerializes_exclusive_load_source(
    %ptrs: tensor<128x32x!tt.ptr<f16>, #load_src>,
    %out: tensor<128x32x!tt.ptr<f16>, #load_src>,
    %b_ptrs: tensor<32x128x!tt.ptr<f16>, #load_src_b>,
    %c: tensor<128x128xf32, #dot_default_load>) -> tensor<128x128xf32, #dot_default_load> {
    %a = tt.load %ptrs : tensor<128x32x!tt.ptr<f16>, #load_src>
    %b = tt.load %b_ptrs : tensor<32x128x!tt.ptr<f16>, #load_src_b>
    tt.store %out, %a : tensor<128x32x!tt.ptr<f16>, #load_src>
    %a_blocked = ttg.convert_layout %a : tensor<128x32xf16, #load_src> -> tensor<128x32xf16, #dot_default_load>
    %ad = ttg.convert_layout %a_blocked : tensor<128x32xf16, #dot_default_load> -> tensor<128x32xf16, #dot_a_load>
    %sum = arith.addf %b, %b : tensor<32x128xf16, #load_src_b>
    %bd = ttg.convert_layout %sum : tensor<32x128xf16, #load_src_b> -> tensor<32x128xf16, #dot_b_load>
    %dot = tt.dot %ad, %bd, %c : tensor<128x32xf16, #dot_a_load> * tensor<32x128xf16, #dot_b_load> -> tensor<128x128xf32, #dot_default_load>
    tt.return %dot : tensor<128x128xf32, #dot_default_load>
  }
}

// -----

#planner_src_b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#planner_gather_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#planner_dot_default = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#planner_dot_a = #ttg.dot_op<{opIdx = 0, parent = #planner_dot_default}>
#planner_dot_b = #ttg.dot_op<{opIdx = 1, parent = #planner_dot_default}>

// CHECK-DAG: #[[$PLANNER_SRC_B:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// CHECK-DAG: #[[$PLANNER_B:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// CHECK-DAG: #[[$PLANNER_DOT_DEFAULT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// CHECK-DAG: #[[$PLANNER_DOT_OPT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @dot_rematerializes_masked_load
  // CHECK-NOT: tt.load
  // CHECK-DAG: %[[MASKED_PTRS:.*]] = ttg.convert_layout %{{.*}} : tensor<32x128x!tt.ptr<f16>, #[[$PLANNER_SRC_B]]> -> tensor<32x128x!tt.ptr<f16>, #[[$PLANNER_B]]>
  // CHECK-DAG: %[[MASKED_MASK:.*]] = ttg.convert_layout %{{.*}} : tensor<32x128xi1, #[[$PLANNER_SRC_B]]> -> tensor<32x128xi1, #[[$PLANNER_B]]>
  // CHECK-DAG: %[[MASKED_OTHER:.*]] = ttg.convert_layout %{{.*}} : tensor<32x128xf16, #[[$PLANNER_SRC_B]]> -> tensor<32x128xf16, #[[$PLANNER_B]]>
  // CHECK: %[[MASKED_LOAD:.*]] = tt.load %[[MASKED_PTRS]], %[[MASKED_MASK]], %[[MASKED_OTHER]] : tensor<32x128x!tt.ptr<f16>, #[[$PLANNER_B]]>
  // CHECK-NOT: tt.load
  // CHECK: ttg.convert_layout %[[MASKED_LOAD]] : tensor<32x128xf16, #[[$PLANNER_B]]> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_OPT]]}>>
  tt.func @dot_rematerializes_masked_load(
      %ptrs: tensor<32x128x!tt.ptr<f16>, #planner_src_b>,
      %mask: tensor<32x128xi1, #planner_src_b>,
      %other: tensor<32x128xf16, #planner_src_b>) {
    %b = tt.load %ptrs, %mask, %other : tensor<32x128x!tt.ptr<f16>, #planner_src_b>
    %bd = ttg.convert_layout %b : tensor<32x128xf16, #planner_src_b> -> tensor<32x128xf16, #planner_dot_b>
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #planner_dot_a>
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #planner_dot_default>
    %dot = tt.dot %a, %bd, %c : tensor<128x32xf16, #planner_dot_a> * tensor<32x128xf16, #planner_dot_b> -> tensor<128x128xf32, #planner_dot_default>
    tt.return
  }

  // CHECK-LABEL: tt.func @dot_rematerializes_gather_indices
  // CHECK-NOT: tt.load
  // CHECK: %[[GATHER_INDEX_PTRS:.*]] = ttg.convert_layout %{{.*}} : tensor<32x128x!tt.ptr<i32>, #[[$PLANNER_SRC_B]]> -> tensor<32x128x!tt.ptr<i32>, #[[$PLANNER_B]]>
  // CHECK: %[[GATHER_INDICES:.*]] = tt.load %[[GATHER_INDEX_PTRS]] : tensor<32x128x!tt.ptr<i32>, #[[$PLANNER_B]]>
  // CHECK-NOT: tt.load
  // CHECK: %[[GATHER:.*]] = tt.gather %[[GATHER_SRC:.*]][%[[GATHER_INDICES]]] {axis = 0 : i32} : (tensor<64x128xf16, #[[$PLANNER_SRC_B]]>, tensor<32x128xi32, #[[$PLANNER_B]]>) -> tensor<32x128xf16, #[[$PLANNER_B]]>
  // CHECK: ttg.convert_layout %[[GATHER]] : tensor<32x128xf16, #[[$PLANNER_B]]> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_OPT]]}>>
  tt.func @dot_rematerializes_gather_indices(
      %src: tensor<64x128xf16, #planner_gather_src>,
      %index_ptrs: tensor<32x128x!tt.ptr<i32>, #planner_src_b>) {
    %indices = tt.load %index_ptrs : tensor<32x128x!tt.ptr<i32>, #planner_src_b>
    %gather = tt.gather %src[%indices] {axis = 0 : i32} : (tensor<64x128xf16, #planner_gather_src>, tensor<32x128xi32, #planner_src_b>) -> tensor<32x128xf16, #planner_src_b>
    %bd = ttg.convert_layout %gather : tensor<32x128xf16, #planner_src_b> -> tensor<32x128xf16, #planner_dot_b>
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #planner_dot_a>
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #planner_dot_default>
    %dot = tt.dot %a, %bd, %c : tensor<128x32xf16, #planner_dot_a> * tensor<32x128xf16, #planner_dot_b> -> tensor<128x128xf32, #planner_dot_default>
    tt.return
  }

  // CHECK-LABEL: tt.func @dot_does_not_rematerialize_efficient_gather
  // CHECK: %[[EFFICIENT_INDICES:.*]] = tt.load %{{.*}} : tensor<32x128x!tt.ptr<i32>, #[[$PLANNER_SRC_B]]>
  // CHECK-NOT: tt.load
  // CHECK: %[[EFFICIENT_GATHER:.*]] = tt.gather %{{.*}}[%[[EFFICIENT_INDICES]]] {axis = 0 : i32, efficient_layout} : (tensor<64x128xf16, #[[$PLANNER_SRC_B]]>, tensor<32x128xi32, #[[$PLANNER_SRC_B]]>) -> tensor<32x128xf16, #[[$PLANNER_SRC_B]]>
  // CHECK: %[[EFFICIENT_BD:.*]] = ttg.convert_layout %[[EFFICIENT_GATHER]] : tensor<32x128xf16, #[[$PLANNER_SRC_B]]> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_DEFAULT]]}>>
  // CHECK: ttg.convert_layout %[[EFFICIENT_BD]] : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_DEFAULT]]}>> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_OPT]]}>>
  tt.func @dot_does_not_rematerialize_efficient_gather(
      %src: tensor<64x128xf16, #planner_gather_src>,
      %index_ptrs: tensor<32x128x!tt.ptr<i32>, #planner_src_b>) {
    %indices = tt.load %index_ptrs : tensor<32x128x!tt.ptr<i32>, #planner_src_b>
    %gather = tt.gather %src[%indices] {axis = 0 : i32, efficient_layout} : (tensor<64x128xf16, #planner_gather_src>, tensor<32x128xi32, #planner_src_b>) -> tensor<32x128xf16, #planner_src_b>
    %bd = ttg.convert_layout %gather : tensor<32x128xf16, #planner_src_b> -> tensor<32x128xf16, #planner_dot_b>
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #planner_dot_a>
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #planner_dot_default>
    %dot = tt.dot %a, %bd, %c : tensor<128x32xf16, #planner_dot_a> * tensor<32x128xf16, #planner_dot_b> -> tensor<128x128xf32, #planner_dot_default>
    tt.return
  }

  // CHECK-LABEL: tt.func @dot_does_not_rematerialize_block_argument
  // CHECK: %[[BLOCK_ARG_B:.*]] = ttg.convert_layout %[[BLOCK_ARG:.*]] : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_DEFAULT]]}>> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_OPT]]}>>
  // CHECK: tt.dot %{{.*}}, %[[BLOCK_ARG_B]], %{{.*}}
  tt.func @dot_does_not_rematerialize_block_argument(
      %b: tensor<32x128xf16, #planner_dot_b>) {
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #planner_dot_a>
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #planner_dot_default>
    %dot = tt.dot %a, %b, %c : tensor<128x32xf16, #planner_dot_a> * tensor<32x128xf16, #planner_dot_b> -> tensor<128x128xf32, #planner_dot_default>
    tt.return
  }

  // CHECK-LABEL: tt.func @dot_does_not_rematerialize_region_op
  // CHECK: %[[REGION_VALUE:.*]] = scf.if %{{.*}} -> (tensor<32x128xf16, #[[$PLANNER_SRC_B]]>)
  // CHECK: %[[REGION_BD:.*]] = ttg.convert_layout %[[REGION_VALUE]] : tensor<32x128xf16, #[[$PLANNER_SRC_B]]> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_DEFAULT]]}>>
  // CHECK: ttg.convert_layout %[[REGION_BD]] : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_DEFAULT]]}>> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_OPT]]}>>
  tt.func @dot_does_not_rematerialize_region_op(
      %cond: i1,
      %lhs_ptrs: tensor<32x128x!tt.ptr<f16>, #planner_src_b>,
      %rhs_ptrs: tensor<32x128x!tt.ptr<f16>, #planner_src_b>) {
    %selected = scf.if %cond -> tensor<32x128xf16, #planner_src_b> {
      %lhs = tt.load %lhs_ptrs : tensor<32x128x!tt.ptr<f16>, #planner_src_b>
      scf.yield %lhs : tensor<32x128xf16, #planner_src_b>
    } else {
      %rhs = tt.load %rhs_ptrs : tensor<32x128x!tt.ptr<f16>, #planner_src_b>
      scf.yield %rhs : tensor<32x128xf16, #planner_src_b>
    }
    %bd = ttg.convert_layout %selected : tensor<32x128xf16, #planner_src_b> -> tensor<32x128xf16, #planner_dot_b>
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #planner_dot_a>
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #planner_dot_default>
    %dot = tt.dot %a, %bd, %c : tensor<128x32xf16, #planner_dot_a> * tensor<32x128xf16, #planner_dot_b> -> tensor<128x128xf32, #planner_dot_default>
    tt.return
  }

  // CHECK-LABEL: tt.func @dot_does_not_rematerialize_atomic
  // CHECK: %[[ATOMIC:.*]] = tt.atomic_rmw fadd, relaxed, gpu, %{{.*}}, %{{.*}}, %{{.*}} : (tensor<32x128x!tt.ptr<f16>, #[[$PLANNER_SRC_B]]>, tensor<32x128xf16, #[[$PLANNER_SRC_B]]>, tensor<32x128xi1, #[[$PLANNER_SRC_B]]>) -> tensor<32x128xf16, #[[$PLANNER_SRC_B]]>
  // CHECK-NOT: tt.atomic_rmw
  // CHECK: %[[ATOMIC_BD:.*]] = ttg.convert_layout %[[ATOMIC]] : tensor<32x128xf16, #[[$PLANNER_SRC_B]]> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_DEFAULT]]}>>
  // CHECK: ttg.convert_layout %[[ATOMIC_BD]] : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_DEFAULT]]}>> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_OPT]]}>>
  tt.func @dot_does_not_rematerialize_atomic(
      %ptrs: tensor<32x128x!tt.ptr<f16>, #planner_src_b>,
      %value: tensor<32x128xf16, #planner_src_b>,
      %mask: tensor<32x128xi1, #planner_src_b>) {
    %atomic = tt.atomic_rmw fadd, relaxed, gpu, %ptrs, %value, %mask : (tensor<32x128x!tt.ptr<f16>, #planner_src_b>, tensor<32x128xf16, #planner_src_b>, tensor<32x128xi1, #planner_src_b>) -> tensor<32x128xf16, #planner_src_b>
    %bd = ttg.convert_layout %atomic : tensor<32x128xf16, #planner_src_b> -> tensor<32x128xf16, #planner_dot_b>
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #planner_dot_a>
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #planner_dot_default>
    %dot = tt.dot %a, %bd, %c : tensor<128x32xf16, #planner_dot_a> * tensor<32x128xf16, #planner_dot_b> -> tensor<128x128xf32, #planner_dot_default>
    tt.return
  }

  // CHECK-LABEL: tt.func @dot_does_not_rematerialize_multi_result_op
  // CHECK: %[[ASM_LOAD:.*]] = tt.load %{{.*}} : tensor<32x128x!tt.ptr<i8>, #[[$PLANNER_SRC_B]]>
  // CHECK: %[[ASM_RESULTS:.*]]:2 = tt.elementwise_inline_asm {{.*}} %[[ASM_LOAD]] : tensor<32x128xi8, #[[$PLANNER_SRC_B]]> -> tensor<32x128xbf16, #[[$PLANNER_SRC_B]]>, tensor<32x128xbf16, #[[$PLANNER_SRC_B]]>
  // CHECK-NOT: tt.elementwise_inline_asm
  // CHECK: %[[ASM_BD:.*]] = ttg.convert_layout %[[ASM_RESULTS]]#0 : tensor<32x128xbf16, #[[$PLANNER_SRC_B]]> -> tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_DEFAULT]]}>>
  // CHECK: ttg.convert_layout %[[ASM_BD]] : tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_DEFAULT]]}>> -> tensor<32x128xbf16, #ttg.dot_op<{opIdx = 1, parent = #[[$PLANNER_DOT_OPT]]}>>
  tt.func @dot_does_not_rematerialize_multi_result_op(
      %ptrs: tensor<32x128x!tt.ptr<i8>, #planner_src_b>) {
    %loaded = tt.load %ptrs : tensor<32x128x!tt.ptr<i8>, #planner_src_b>
    %lhs, %rhs = tt.elementwise_inline_asm "" {constraints = "=r,=r,=r,=r,r", packed_element = 4 : i32, pure = true} %loaded : tensor<32x128xi8, #planner_src_b> -> tensor<32x128xbf16, #planner_src_b>, tensor<32x128xbf16, #planner_src_b>
    %bd = ttg.convert_layout %lhs : tensor<32x128xbf16, #planner_src_b> -> tensor<32x128xbf16, #planner_dot_b>
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xbf16, #planner_dot_a>
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #planner_dot_default>
    %dot = tt.dot %a, %bd, %c : tensor<128x32xbf16, #planner_dot_a> * tensor<32x128xbf16, #planner_dot_b> -> tensor<128x128xf32, #planner_dot_default>
    tt.return
  }

}

// -----

#volatile_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#volatile_default = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#volatile_a = #ttg.dot_op<{opIdx = 0, parent = #volatile_default}>
#volatile_b = #ttg.dot_op<{opIdx = 1, parent = #volatile_default}>

// CHECK-DAG: #[[$VOLATILE_SRC:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// CHECK-DAG: #[[$VOLATILE_DEFAULT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// CHECK-DAG: #[[$VOLATILE_OPT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @dot_does_not_rematerialize_volatile_load
  // CHECK-NOT: tt.load
  // CHECK: tt.load %{{.*}} {isVolatile = true} : tensor<128x32x!tt.ptr<f16>, #[[$VOLATILE_SRC]]>
  // CHECK-NOT: tt.load
  // CHECK: ttg.convert_layout %{{.*}} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$VOLATILE_DEFAULT]]}>> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$VOLATILE_OPT]]}>>
  tt.func @dot_does_not_rematerialize_volatile_load(
      %ptrs: tensor<128x32x!tt.ptr<f16>, #volatile_src>) {
    %a = tt.load %ptrs {isVolatile = true} : tensor<128x32x!tt.ptr<f16>, #volatile_src>
    %a_blocked = ttg.convert_layout %a : tensor<128x32xf16, #volatile_src> -> tensor<128x32xf16, #volatile_default>
    %ad = ttg.convert_layout %a_blocked : tensor<128x32xf16, #volatile_default> -> tensor<128x32xf16, #volatile_a>
    %b = arith.constant dense<2.000000e+00> : tensor<32x128xf16, #volatile_b>
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #volatile_default>
    %dot = tt.dot %ad, %b, %c : tensor<128x32xf16, #volatile_a> * tensor<32x128xf16, #volatile_b> -> tensor<128x128xf32, #volatile_default>
    tt.return
  }
}

// -----

#desc_load_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#desc_load_src_b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#dot_default_desc_load = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#dot_a_desc_load = #ttg.dot_op<{opIdx = 0, parent = #dot_default_desc_load}>
#dot_b_desc_load = #ttg.dot_op<{opIdx = 1, parent = #dot_default_desc_load}>

// CHECK-DAG: #[[$DESC_LOAD_ORIG:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// CHECK-DAG: #[[$DESC_LOAD_B_PLANNED:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// CHECK-DAG: #[[$DOT_DEFAULT_DESC_LOAD:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// CHECK-DAG: #[[$DOT_OPT_DESC_LOAD:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @dot_rematerializes_exclusive_descriptor_load_source
  // CHECK-NOT: tt.descriptor_load
  // CHECK: %[[ORIG_DESC_LOAD:.*]] = tt.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] : !tt.tensordesc<128x32xf16> -> tensor<128x32xf16, #[[$DESC_LOAD_ORIG]]>
  // CHECK-NOT: tt.descriptor_load
  // CHECK: %[[B_DESC_LOAD:.*]] = tt.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] : !tt.tensordesc<32x128xf16> -> tensor<32x128xf16, #[[$DESC_LOAD_B_PLANNED]]>
  // CHECK-NOT: tt.descriptor_load
  // CHECK: ttg.convert_layout %{{.*}} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_DEFAULT_DESC_LOAD]]}>> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_OPT_DESC_LOAD]]}>>
  // CHECK: tt.return %[[ORIG_DESC_LOAD]], %{{.*}} : tensor<128x32xf16, #[[$DESC_LOAD_ORIG]]>,
  // E2E-LABEL: tt.func @dot_rematerializes_exclusive_descriptor_load_source
  // E2E: %[[E2E_OLD_DESC:.*]] = tt.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] : !tt.tensordesc<128x32xf16> -> tensor<128x32xf16, #[[$E2E_DESC_ORIG:.*]]>
  // E2E: %[[E2E_NEW_B_DESC:.*]] = tt.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] : !tt.tensordesc<32x128xf16> -> tensor<32x128xf16, #[[$E2E_DESC_B_PLANNED:.*]]>
  // E2E: ttg.convert_layout %[[E2E_NEW_B_DESC]]
  // E2E: tt.return %[[E2E_OLD_DESC]], %{{.*}} : tensor<128x32xf16, #[[$E2E_DESC_ORIG]]>,
  tt.func @dot_rematerializes_exclusive_descriptor_load_source(
    %a_desc: !tt.tensordesc<128x32xf16>,
    %b_desc: !tt.tensordesc<32x128xf16>,
    %i: i32,
    %j: i32,
    %c: tensor<128x128xf32, #dot_default_desc_load>) -> (tensor<128x32xf16, #desc_load_src>, tensor<128x128xf32, #dot_default_desc_load>) {
    %a = tt.descriptor_load %a_desc[%i, %j] : !tt.tensordesc<128x32xf16> -> tensor<128x32xf16, #desc_load_src>
    %b = tt.descriptor_load %b_desc[%i, %j] : !tt.tensordesc<32x128xf16> -> tensor<32x128xf16, #desc_load_src_b>
    %a_blocked = ttg.convert_layout %a : tensor<128x32xf16, #desc_load_src> -> tensor<128x32xf16, #dot_default_desc_load>
    %ad = ttg.convert_layout %a_blocked : tensor<128x32xf16, #dot_default_desc_load> -> tensor<128x32xf16, #dot_a_desc_load>
    %bd = ttg.convert_layout %b : tensor<32x128xf16, #desc_load_src_b> -> tensor<32x128xf16, #dot_b_desc_load>
    %dot = tt.dot %ad, %bd, %c : tensor<128x32xf16, #dot_a_desc_load> * tensor<32x128xf16, #dot_b_desc_load> -> tensor<128x128xf32, #dot_default_desc_load>
    tt.return %a, %dot : tensor<128x32xf16, #desc_load_src>, tensor<128x128xf32, #dot_default_desc_load>
  }
}

// -----

#transpose_src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#transpose_dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1], CGALayout = [[1, 0]]}>
#transpose_result = #ttg.slice<{dim = 0, parent = #transpose_dst}>

// CHECK-DAG: #[[$PROP_TRANSPOSE_LOAD:.*]] = #ttg.blocked<{{.*}}order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// CHECK-DAG: #[[$PROP_TRANSPOSE_REDUCE:.*]] = #ttg.blocked<{{.*}}order = [0, 1], CGALayout = {{\[\[0, 1\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduce_rematerializes_transpose
  // CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<128x128x!tt.ptr<f32>, #[[$PROP_TRANSPOSE_LOAD]]>
  // CHECK: %[[TRANSPOSE:.*]] = tt.trans %[[LOAD]] {{.*}} : tensor<128x128xf32, #[[$PROP_TRANSPOSE_LOAD]]> -> tensor<128x128xf32, #[[$PROP_TRANSPOSE_REDUCE]]>
  // CHECK: "tt.reduce"(%[[TRANSPOSE]]) <{axis = 0 : i32}>
  tt.func @reduce_rematerializes_transpose(%ptrs: tensor<128x128x!tt.ptr<f32>, #transpose_src>) -> tensor<128xf32, #transpose_result> {
    %loaded = tt.load %ptrs : tensor<128x128x!tt.ptr<f32>, #transpose_src>
    %transposed = tt.trans %loaded {order = array<i32: 1, 0>} : tensor<128x128xf32, #transpose_src> -> tensor<128x128xf32, #transpose_dst>
    %sum = "tt.reduce"(%transposed) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<128x128xf32, #transpose_dst>) -> tensor<128xf32, #transpose_result>
    tt.return %sum : tensor<128xf32, #transpose_result>
  }
}

// -----

#broadcast_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#broadcast_slice = #ttg.slice<{dim = 1, parent = #broadcast_parent}>

// CHECK-DAG: #[[$PROP_BROADCAST:.*]] = #ttg.blocked<{{.*}}CGALayout = {{\[\[1, 0\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduce_rematerializes_expand_broadcast
  // CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<128x!tt.ptr<f32>, #ttg.slice<{dim = 1, parent = #[[$PROP_BROADCAST]]}>>
  // CHECK: %[[EXPAND:.*]] = tt.expand_dims %[[LOAD]] {{.*}} -> tensor<128x1xf32, #[[$PROP_BROADCAST]]>
  // CHECK: %[[BROADCAST:.*]] = tt.broadcast %[[EXPAND]] : tensor<128x1xf32, #[[$PROP_BROADCAST]]> -> tensor<128x128xf32, #[[$PROP_BROADCAST]]>
  // CHECK: "tt.reduce"(%[[BROADCAST]]) <{axis = 1 : i32}>
  tt.func @reduce_rematerializes_expand_broadcast(%ptrs: tensor<128x!tt.ptr<f32>, #broadcast_slice>) -> tensor<128xf32, #broadcast_slice> {
    %loaded = tt.load %ptrs : tensor<128x!tt.ptr<f32>, #broadcast_slice>
    %expanded = tt.expand_dims %loaded {axis = 1 : i32} : tensor<128xf32, #broadcast_slice> -> tensor<128x1xf32, #broadcast_parent>
    %broadcast = tt.broadcast %expanded : tensor<128x1xf32, #broadcast_parent> -> tensor<128x128xf32, #broadcast_parent>
    %sum = "tt.reduce"(%broadcast) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<128x128xf32, #broadcast_parent>) -> tensor<128xf32, #broadcast_slice>
    tt.return %sum : tensor<128xf32, #broadcast_slice>
  }
}

// -----

#reshape_src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[1]]}>
#reshape_dst = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0], CGALayout = [[1, 0]]}>
#reshape_result = #ttg.slice<{dim = 0, parent = #reshape_dst}>

module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // The requested column split becomes an interleaved block basis in 1D.
  // CHECK-LABEL: tt.func @reduce_does_not_rematerialize_interleaved_reshape
  // CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<4096x!tt.ptr<f32>, {{.*}}>
  // CHECK: %[[RESHAPE:.*]] = tt.reshape %[[LOAD]] : tensor<4096xf32, {{.*}}> -> tensor<64x64xf32, {{.*}}>
  // CHECK: %[[CONVERT:.*]] = ttg.convert_layout %[[RESHAPE]] : tensor<64x64xf32, {{.*}}> -> tensor<64x64xf32, {{.*}}>
  // CHECK: "tt.reduce"(%[[CONVERT]]) <{axis = 0 : i32}>
  tt.func @reduce_does_not_rematerialize_interleaved_reshape(%ptrs: tensor<4096x!tt.ptr<f32>, #reshape_src>) -> tensor<64xf32, #reshape_result> {
    %loaded = tt.load %ptrs : tensor<4096x!tt.ptr<f32>, #reshape_src>
    %reshaped = tt.reshape %loaded : tensor<4096xf32, #reshape_src> -> tensor<64x64xf32, #reshape_dst>
    %sum = "tt.reduce"(%reshaped) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<64x64xf32, #reshape_dst>) -> tensor<64xf32, #reshape_result>
    tt.return %sum : tensor<64xf32, #reshape_result>
  }
}

// -----

#factorable = #ttg.linear<{register = [[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]], lane = [[0, 1], [0, 2], [0, 4], [1, 0], [2, 0]], warp = [[4, 0], [8, 0]], block = [[0, 64]]}>
#factorable_blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#factorable_result = #ttg.slice<{dim = 1, parent = #factorable_blocked}>
#linear_result = #ttg.slice<{dim = 1, parent = #factorable}>

// The ordinary linear layout factors as a 128x64 CTA tile and a column CGA.
// CHECK-DAG: #[[$FACTORABLE_LOAD:.*]] = #ttg.linear<{{.*}}block = {{\[\[64, 0\]\]}}}>
// CHECK-DAG: #[[$FACTORABLE_REDUCE:.*]] = #ttg.blocked<{{.*}}CGALayout = {{\[\[1, 0\]\]}}}>
// E2E-DAG: #[[$E2E_FACTORABLE_LOAD:.*]] = #ttg.linear<{{.*}}block = {{\[\[64, 0\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduce_rematerializes_factorable_linear_conversion
  // CHECK-NOT: tt.load
  // CHECK: %[[PTRS:.*]] = ttg.convert_layout %arg0 : tensor<128x128x!tt.ptr<f32>, {{.*}}> -> tensor<128x128x!tt.ptr<f32>, #[[$FACTORABLE_LOAD]]>
  // CHECK: %[[LOAD:.*]] = tt.load %[[PTRS]] : tensor<128x128x!tt.ptr<f32>, #[[$FACTORABLE_LOAD]]>
  // CHECK-NOT: tt.load
  // CHECK: %[[CONVERT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<128x128xf32, #[[$FACTORABLE_LOAD]]> -> tensor<128x128xf32, #[[$FACTORABLE_REDUCE]]>
  // CHECK: "tt.reduce"(%[[CONVERT]]) <{axis = 1 : i32}>
  // E2E-LABEL: tt.func @reduce_rematerializes_factorable_linear_conversion
  // E2E: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<128x128x!tt.ptr<f32>, #[[$E2E_FACTORABLE_LOAD]]>
  // E2E-NOT: tt.load
  // E2E: "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
  tt.func @reduce_rematerializes_factorable_linear_conversion(%ptrs: tensor<128x128x!tt.ptr<f32>, #factorable>) -> tensor<128xf32, #factorable_result> {
    %loaded = tt.load %ptrs : tensor<128x128x!tt.ptr<f32>, #factorable>
    %converted = ttg.convert_layout %loaded : tensor<128x128xf32, #factorable> -> tensor<128x128xf32, #factorable_blocked>
    %sum = "tt.reduce"(%converted) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<128x128xf32, #factorable_blocked>) -> tensor<128xf32, #factorable_result>
    tt.return %sum : tensor<128xf32, #factorable_result>
  }

  // CHECK-LABEL: tt.func @reduce_rematerializes_factorable_linear_input
  // CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<128x128x!tt.ptr<f32>, #[[$FACTORABLE_LOAD]]>
  // CHECK-NOT: tt.load
  // CHECK: "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
  // E2E-LABEL: tt.func @reduce_rematerializes_factorable_linear_input
  // E2E: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<128x128x!tt.ptr<f32>, #[[$E2E_FACTORABLE_LOAD]]>
  // E2E-NOT: tt.load
  // E2E: "tt.reduce"(%[[LOAD]]) <{axis = 1 : i32}>
  // PIPELINE-LABEL: tt.func @reduce_rematerializes_factorable_linear_input
  // PIPELINE: %[[REDUCE:.*]] = "tt.reduce"
  // PIPELINE: %[[CONVERT:.*]] = ttg.convert_layout %[[REDUCE]]
  // PIPELINE: tt.store %arg1, %[[CONVERT]]
  tt.func @reduce_rematerializes_factorable_linear_input(%ptrs: tensor<128x128x!tt.ptr<f32>, #factorable>, %out: tensor<128x!tt.ptr<f32>, #linear_result>) {
    %loaded = tt.load %ptrs : tensor<128x128x!tt.ptr<f32>, #factorable>
    %sum = "tt.reduce"(%loaded) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.addf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<128x128xf32, #factorable>) -> tensor<128xf32, #linear_result>
    tt.store %out, %sum : tensor<128x!tt.ptr<f32>, #linear_result>
    tt.return
  }
}

// -----

#generic_parent = #ttg.generic_linear<{register = [[0, 0, 8], [0, 0, 16], [0, 0, 32], [0, 16, 0], [0, 32, 0], [0, 64, 0]], lane = [[0, 0, 1], [0, 0, 2], [0, 0, 4], [0, 1, 0], [0, 2, 0]], warp = [[0, 4, 0], [0, 8, 0]], block = [[0, 0, 64]]}>
#generic_slice = #ttg.slice<{dim = 0, parent = #generic_parent}>
#generic_default = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#generic_dot_a = #ttg.dot_op<{opIdx = 0, parent = #generic_default}>
#generic_dot_b = #ttg.dot_op<{opIdx = 1, parent = #generic_default}>

// Generic encodings remain excluded even when they factor and are wrapped.
// CHECK-DAG: #[[$GENERIC_PARENT:.*]] = #ttg.generic_linear<
// CHECK-DAG: #[[$GENERIC_DOT:.*]] = #ttg.blocked<{{.*}}CGALayout = {{\[\[1, 0\]\]}}}>
// E2E-DAG: #[[$E2E_GENERIC_PARENT:.*]] = #ttg.generic_linear<
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @dot_does_not_rematerialize_factorable_generic_slice
  // CHECK: %[[LOAD:.*]] = tt.load %arg0 : tensor<128x128x!tt.ptr<f16>, #ttg.slice<{dim = 0, parent = #[[$GENERIC_PARENT]]}>>
  // CHECK-NOT: tt.load
  // CHECK: ttg.convert_layout %[[LOAD]]
  // CHECK: tt.dot {{.*}} -> tensor<128x64xf32, #[[$GENERIC_DOT]]>
  // E2E-LABEL: tt.func @dot_does_not_rematerialize_factorable_generic_slice
  // E2E: %[[LOAD:.*]] = tt.load %arg0 : tensor<128x128x!tt.ptr<f16>, #ttg.slice<{dim = 0, parent = #[[$E2E_GENERIC_PARENT]]}>>
  // E2E-NOT: tt.load
  // E2E: ttg.convert_layout %[[LOAD]]
  // E2E: tt.dot
  tt.func @dot_does_not_rematerialize_factorable_generic_slice(%ptrs: tensor<128x128x!tt.ptr<f16>, #generic_slice>) -> tensor<128x64xf32, #generic_default> {
    %loaded = tt.load %ptrs : tensor<128x128x!tt.ptr<f16>, #generic_slice>
    %a = ttg.convert_layout %loaded : tensor<128x128xf16, #generic_slice> -> tensor<128x128xf16, #generic_dot_a>
    %b = arith.constant dense<1.000000e+00> : tensor<128x64xf16, #generic_dot_b>
    %c = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #generic_default>
    %dot = tt.dot %a, %b, %c : tensor<128x128xf16, #generic_dot_a> * tensor<128x64xf16, #generic_dot_b> -> tensor<128x64xf32, #generic_default>
    tt.return %dot : tensor<128x64xf32, #generic_default>
  }
}
