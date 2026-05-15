// RUN: triton-opt %s -split-input-file -triton-nvidia-gpu-plan-cta | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0], [0]]}>

  // CHECK-DAG: #[[ORIG:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = {{\[\[0\], \[0\]\]}}}>
  // CHECK-DAG: #[[REDUCE:.*]] = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = {{\[\[1\], \[2\]\]}}}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduce_1d_split_ctas
  // CHECK: %[[CVT:.*]] = ttg.convert_layout %{{.*}} : tensor<65536xf32, #[[ORIG]]> -> tensor<65536xf32, #[[REDUCE]]>
  // CHECK: "tt.reduce"(%[[CVT]]) <{axis = 0 : i32}>
  // CHECK: tt.reduce.return %{{.*}} : f32
  // CHECK-NEXT: }) : (tensor<65536xf32, #[[REDUCE]]>) -> f32
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

// CHECK-DAG: #[[DOT_DEFAULT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\], \[0, 2\]\]}}}>
// CHECK-DAG: #[[DOT_OPT:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\], \[1, 0\]\]}}}>
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @dot_split_mn
  // CHECK: ttg.convert_layout %{{.*}} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DOT_DEFAULT]]}>> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DOT_OPT]]}>>
  // CHECK: ttg.convert_layout %{{.*}} : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DOT_DEFAULT]]}>> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DOT_OPT]]}>>
  // CHECK: %[[D:.*]] = ttg.convert_layout %{{.*}} : tensor<128x128xf32, #[[DOT_DEFAULT]]> -> tensor<128x128xf32, #[[DOT_OPT]]>
  // CHECK: %[[DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[D]] : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DOT_OPT]]}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DOT_OPT]]}>> -> tensor<128x128xf32, #[[DOT_OPT]]>
  // CHECK: ttg.convert_layout %[[DOT]] : tensor<128x128xf32, #[[DOT_OPT]]> -> tensor<128x128xf32, #[[DOT_DEFAULT]]>
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

#dot_default_2cta = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 1]]}>
#dot_a_2cta = #ttg.dot_op<{opIdx = 0, parent = #dot_default_2cta}>
#dot_b_2cta = #ttg.dot_op<{opIdx = 1, parent = #dot_default_2cta}>

// CHECK-DAG: #[[DOT_DEFAULT_2CTA:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
// CHECK-DAG: #[[DOT_OPT_2CTA:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = {{\[\[0, 1\]\]}}}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @dot_split_m_2cta
  // CHECK: ttg.convert_layout %{{.*}} : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DOT_DEFAULT_2CTA]]}>> -> tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DOT_OPT_2CTA]]}>>
  // CHECK: ttg.convert_layout %{{.*}} : tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DOT_DEFAULT_2CTA]]}>> -> tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DOT_OPT_2CTA]]}>>
  // CHECK: %[[D:.*]] = ttg.convert_layout %{{.*}} : tensor<128x128xf32, #[[DOT_DEFAULT_2CTA]]> -> tensor<128x128xf32, #[[DOT_OPT_2CTA]]>
  // CHECK: %[[DOT:.*]] = tt.dot %{{.*}}, %{{.*}}, %[[D]] : tensor<128x32xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DOT_OPT_2CTA]]}>> * tensor<32x128xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DOT_OPT_2CTA]]}>> -> tensor<128x128xf32, #[[DOT_OPT_2CTA]]>
  // CHECK: ttg.convert_layout %[[DOT]] : tensor<128x128xf32, #[[DOT_OPT_2CTA]]> -> tensor<128x128xf32, #[[DOT_DEFAULT_2CTA]]>
  tt.func @dot_split_m_2cta(%ptr: !tt.ptr<f32>) {
    %a = arith.constant dense<1.000000e+00> : tensor<128x32xf16, #dot_a_2cta>
    %b = arith.constant dense<2.000000e+00> : tensor<32x128xf16, #dot_b_2cta>
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #dot_default_2cta>
    %dot = tt.dot %a, %b, %c : tensor<128x32xf16, #dot_a_2cta> * tensor<32x128xf16, #dot_b_2cta> -> tensor<128x128xf32, #dot_default_2cta>
    %ptrs = tt.splat %ptr : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #dot_default_2cta>
    tt.store %ptrs, %dot : tensor<128x128x!tt.ptr<f32>, #dot_default_2cta>
    tt.return
  }
}
