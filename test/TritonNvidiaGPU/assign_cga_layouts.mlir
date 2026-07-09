// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-assign-cga-layouts | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s --check-prefix=REMAT
// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-assign-cga-layouts -tritongpu-remove-layout-conversions | FileCheck %s --check-prefix=E2E

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0], [0]]}>

  // CHECK-DAG: #[[$ORIG:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = {{\[\[0\], \[0\]\]}}}>
  // CHECK-DAG: #[[$REDUCE:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = {{\[\[1\], \[2\]\]}}}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduce_1d_split_ctas
  // CHECK: %[[CVT:.*]] = ttg.convert_layout %{{.*}} {ttg.force_cga_rematerialization} : tensor<65536xf32, #[[$ORIG]]> -> tensor<65536xf32, #[[$REDUCE]]>
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
  // CHECK: ttg.convert_layout %{{.*}} {ttg.force_cga_rematerialization} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_DEFAULT]]}>> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_OPT]]}>>
  // CHECK: ttg.convert_layout %{{.*}} {ttg.force_cga_rematerialization} : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DOT_DEFAULT]]}>> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DOT_OPT]]}>>
  // CHECK: %[[D:.*]] = ttg.convert_layout %{{.*}} {ttg.force_cga_rematerialization} : tensor<128x128xf32, #[[$DOT_DEFAULT]]> -> tensor<128x128xf32, #[[$DOT_OPT]]>
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
// CHECK-DAG: #[[$DOT_DEFAULT_LOAD:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// CHECK-DAG: #[[$DOT_OPT_LOAD:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// E2E-DAG: #[[$E2E_LOAD_ORIG:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// E2E-DAG: #[[$E2E_LOAD_PLANNED:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 0\]\]}}}>
// E2E-DAG: #[[$E2E_LOAD_B_ORIG:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// E2E-DAG: #[[$E2E_LOAD_B_PLANNED:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// E2E-DAG: #[[$E2E_DOT_OPT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @dot_clones_load_source
  // CHECK: %[[ORIG_LOAD:.*]] = tt.load %{{.*}} : tensor<128x32x!tt.ptr<f16>, #[[$LOAD_ORIG]]>
  // CHECK: tt.store %{{.*}}, %[[ORIG_LOAD]] : tensor<128x32x!tt.ptr<f16>, #[[$LOAD_ORIG]]>
  // CHECK: ttg.convert_layout %{{.*}} {ttg.force_cga_rematerialization} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_DEFAULT_LOAD]]}>> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_OPT_LOAD]]}>>
  // CHECK: ttg.convert_layout %{{.*}} {ttg.force_cga_rematerialization} : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DOT_DEFAULT_LOAD]]}>> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DOT_OPT_LOAD]]}>>
  // E2E-LABEL: tt.func @dot_clones_load_source
  // E2E: %[[E2E_PTRS:.*]] = ttg.convert_layout %{{.*}} : tensor<128x32x!tt.ptr<f16>, #[[$E2E_LOAD_ORIG]]> -> tensor<128x32x!tt.ptr<f16>, #[[$E2E_LOAD_PLANNED]]>
  // E2E: %[[E2E_A:.*]] = tt.load %[[E2E_PTRS]] : tensor<128x32x!tt.ptr<f16>, #[[$E2E_LOAD_PLANNED]]>
  // E2E: %[[E2E_OLD_A:.*]] = tt.load %{{.*}} : tensor<128x32x!tt.ptr<f16>, #[[$E2E_LOAD_ORIG]]>
  // E2E: %[[E2E_B_PTRS:.*]] = ttg.convert_layout %{{.*}} : tensor<32x128x!tt.ptr<f16>, #[[$E2E_LOAD_B_ORIG]]> -> tensor<32x128x!tt.ptr<f16>, #[[$E2E_LOAD_B_PLANNED]]>
  // E2E: %[[E2E_B:.*]] = tt.load %[[E2E_B_PTRS]] : tensor<32x128x!tt.ptr<f16>, #[[$E2E_LOAD_B_PLANNED]]>
  // E2E: tt.store %{{.*}}, %[[E2E_OLD_A]] : tensor<128x32x!tt.ptr<f16>, #[[$E2E_LOAD_ORIG]]>
  // E2E: %[[E2E_AD:.*]] = ttg.convert_layout %[[E2E_A]] : tensor<128x32xf16, #[[$E2E_LOAD_PLANNED]]> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$E2E_DOT_OPT]]}>>
  // E2E: %[[E2E_BD:.*]] = ttg.convert_layout %[[E2E_B]] : tensor<32x128xf16, #[[$E2E_LOAD_B_PLANNED]]> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$E2E_DOT_OPT]]}>>
  // E2E: tt.dot %[[E2E_AD]], %[[E2E_BD]], %{{.*}} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$E2E_DOT_OPT]]}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$E2E_DOT_OPT]]}>> -> tensor<128x128xf32, #[[$E2E_DOT_OPT]]>
  tt.func @dot_clones_load_source(
    %ptrs: tensor<128x32x!tt.ptr<f16>, #load_src>,
    %out: tensor<128x32x!tt.ptr<f16>, #load_src>,
    %b_ptrs: tensor<32x128x!tt.ptr<f16>, #load_src_b>,
    %c: tensor<128x128xf32, #dot_default_load>) -> tensor<128x128xf32, #dot_default_load> {
    %a = tt.load %ptrs : tensor<128x32x!tt.ptr<f16>, #load_src>
    %b = tt.load %b_ptrs : tensor<32x128x!tt.ptr<f16>, #load_src_b>
    tt.store %out, %a : tensor<128x32x!tt.ptr<f16>, #load_src>
    %a_blocked = ttg.convert_layout %a : tensor<128x32xf16, #load_src> -> tensor<128x32xf16, #dot_default_load>
    %ad = ttg.convert_layout %a_blocked : tensor<128x32xf16, #dot_default_load> -> tensor<128x32xf16, #dot_a_load>
    %bd = ttg.convert_layout %b : tensor<32x128xf16, #load_src_b> -> tensor<32x128xf16, #dot_b_load>
    %dot = tt.dot %ad, %bd, %c : tensor<128x32xf16, #dot_a_load> * tensor<32x128xf16, #dot_b_load> -> tensor<128x128xf32, #dot_default_load>
    tt.return %dot : tensor<128x128xf32, #dot_default_load>
  }
}

// -----

#desc_load_src = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#desc_load_src_b = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#dot_default_desc_load = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#dot_a_desc_load = #ttg.dot_op<{opIdx = 0, parent = #dot_default_desc_load}>
#dot_b_desc_load = #ttg.dot_op<{opIdx = 1, parent = #dot_default_desc_load}>

// CHECK-DAG: #[[$DESC_LOAD_ORIG:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// CHECK-DAG: #[[$DESC_LOAD_B_ORIG:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// CHECK-DAG: #[[$DOT_DEFAULT_DESC_LOAD:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// CHECK-DAG: #[[$DOT_OPT_DESC_LOAD:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// E2E-DAG: #[[$E2E_DESC_ORIG:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// E2E-DAG: #[[$E2E_DESC_PLANNED:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 0\]\]}}}>
// E2E-DAG: #[[$E2E_DESC_B_PLANNED:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @dot_clones_descriptor_load_source
  // CHECK: %[[ORIG_DESC_LOAD:.*]] = tt.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] : !tt.tensordesc<128x32xf16> -> tensor<128x32xf16, #[[$DESC_LOAD_ORIG]]>
  // CHECK: ttg.convert_layout %{{.*}} {ttg.force_cga_rematerialization} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_DEFAULT_DESC_LOAD]]}>> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[$DOT_OPT_DESC_LOAD]]}>>
  // CHECK: ttg.convert_layout %{{.*}} {ttg.force_cga_rematerialization} : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DOT_DEFAULT_DESC_LOAD]]}>> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[$DOT_OPT_DESC_LOAD]]}>>
  // CHECK: tt.return %[[ORIG_DESC_LOAD]], %{{.*}} : tensor<128x32xf16, #[[$DESC_LOAD_ORIG]]>,
  // E2E-LABEL: tt.func @dot_clones_descriptor_load_source
  // E2E: %[[E2E_NEW_DESC:.*]] = tt.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] : !tt.tensordesc<128x32xf16> -> tensor<128x32xf16, #[[$E2E_DESC_PLANNED]]>
  // E2E: %[[E2E_OLD_DESC:.*]] = tt.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] : !tt.tensordesc<128x32xf16> -> tensor<128x32xf16, #[[$E2E_DESC_ORIG]]>
  // E2E: %[[E2E_NEW_B_DESC:.*]] = tt.descriptor_load %{{.*}}[%{{.*}}, %{{.*}}] : !tt.tensordesc<32x128xf16> -> tensor<32x128xf16, #[[$E2E_DESC_B_PLANNED]]>
  // E2E: ttg.convert_layout %[[E2E_NEW_DESC]]
  // E2E: ttg.convert_layout %[[E2E_NEW_B_DESC]]
  // E2E: tt.return %[[E2E_OLD_DESC]], %{{.*}} : tensor<128x32xf16, #[[$E2E_DESC_ORIG]]>,
  tt.func @dot_clones_descriptor_load_source(
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

#remat_orig = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[1, 0]]}>
#remat_planned = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>

// REMAT-DAG: #[[$REMAT_ORIG:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[1, 0\]\]}}}>
// REMAT-DAG: #[[$REMAT_PLANNED:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 0\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // REMAT-LABEL: tt.func @force_cga_rematerializes_through_elementwise
  // REMAT: %[[PTRS:.*]] = ttg.convert_layout %{{.*}} : tensor<128x32x!tt.ptr<f16>, #[[$REMAT_ORIG]]> -> tensor<128x32x!tt.ptr<f16>, #[[$REMAT_PLANNED]]>
  // REMAT: %[[NEW_LOAD:.*]] = tt.load %[[PTRS]] : tensor<128x32x!tt.ptr<f16>, #[[$REMAT_PLANNED]]>
  // REMAT: %[[OLD_LOAD:.*]] = tt.load %{{.*}} : tensor<128x32x!tt.ptr<f16>, #[[$REMAT_ORIG]]>
  // REMAT: tt.store %{{.*}}, %[[OLD_LOAD]] : tensor<128x32x!tt.ptr<f16>, #[[$REMAT_ORIG]]>
  // REMAT: %[[SUM:.*]] = arith.addf %[[NEW_LOAD]], %[[NEW_LOAD]] : tensor<128x32xf16, #[[$REMAT_PLANNED]]>
  // REMAT: tt.store %{{.*}}, %[[SUM]] : tensor<128x32x!tt.ptr<f16>, #[[$REMAT_PLANNED]]>
  tt.func @force_cga_rematerializes_through_elementwise(
      %ptrs: tensor<128x32x!tt.ptr<f16>, #remat_orig>,
      %old_out: tensor<128x32x!tt.ptr<f16>, #remat_orig>,
      %new_out: tensor<128x32x!tt.ptr<f16>, #remat_planned>) {
    %value = tt.load %ptrs : tensor<128x32x!tt.ptr<f16>, #remat_orig>
    tt.store %old_out, %value : tensor<128x32x!tt.ptr<f16>, #remat_orig>
    %sum = arith.addf %value, %value : tensor<128x32xf16, #remat_orig>
    %planned = ttg.convert_layout %sum {ttg.force_cga_rematerialization} : tensor<128x32xf16, #remat_orig> -> tensor<128x32xf16, #remat_planned>
    tt.store %new_out, %planned : tensor<128x32x!tt.ptr<f16>, #remat_planned>
    tt.return
  }
}
