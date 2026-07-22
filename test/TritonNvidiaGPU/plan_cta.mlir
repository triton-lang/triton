// RUN: env -u TRITON_ENABLE_MMA_V5_TWO_CTA triton-opt %s -split-input-file -triton-nvidia-gpu-plan-cta | FileCheck %s
// RUN: env TRITON_ENABLE_MMA_V5_TWO_CTA=1 triton-opt %s -split-input-file -triton-nvidia-gpu-plan-cta | FileCheck %s --check-prefix=TWOCTA

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = [[0], [0]]}>
#blocked_2d = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1], CGALayout = [[0, 0], [0, 0]]}>
#slice = #ttg.slice<{dim = 1, parent = #blocked_2d}>

  // CHECK: #blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0], CGALayout = {{\[\[1\], \[2\]\]}}}>
  // CHECK-DAG: #[[$RESHAPE_DST:.+]] = #ttg.blocked<{{.*}}CGALayout = {{\[\[1, 0\], \[2, 0\]\]}}}>
  // CHECK-DAG: #[[$RESHAPE_SRC:.+]] = #ttg.linear
module attributes {"ttg.num-ctas" = 4 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reduce_1d_split_ctas
  // CHECK: "tt.reduce"(%{{.*}}) <{axis = 0 : i32}>
  // CHECK: tt.reduce.return %{{.*}} : f32
  // CHECK-NEXT: }) : (tensor<65536xf32, #blocked>) -> f32
  tt.func @reduce_1d_split_ctas() -> f32 {
    %cst = arith.constant dense<0.000000e+00> : tensor<65536xf32, #blocked>
    %red = "tt.reduce"(%cst) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<65536xf32, #blocked>) -> f32
    tt.return %red : f32
  }

  // CHECK-LABEL: tt.func @reduce_reshape
  // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<65536xf32, #[[$RESHAPE_SRC]]>
  // CHECK-NEXT: %[[RESHAPE:.*]] = tt.reshape %[[CST]] : tensor<65536xf32, #[[$RESHAPE_SRC]]> -> tensor<65536x1xf32, #[[$RESHAPE_DST]]>
  // CHECK-NEXT: %[[RED:.*]] = "tt.reduce"(%[[RESHAPE]]) <{axis = 1 : i32}>
  tt.func @reduce_reshape() -> tensor<65536xf32, #slice> {
    %cst = arith.constant dense<0.000000e+00> : tensor<65536xf32, #slice>
    %expanded = tt.reshape %cst : tensor<65536xf32, #slice> -> tensor<65536x1xf32, #blocked_2d>
    %red = "tt.reduce"(%expanded) <{axis = 1 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %sum = arith.addf %lhs, %rhs : f32
      tt.reduce.return %sum : f32
    }) : (tensor<65536x1xf32, #blocked_2d>) -> tensor<65536xf32, #slice>
    tt.return %red : tensor<65536xf32, #slice>
  }
}

// -----

#dot_parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#load_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#load_b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #dot_parent}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #dot_parent}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:110", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: #[[$DEFAULT_DOT:.+]] = #ttg.blocked<{{.*}}CGALayout = {{\[\[0, 1\]\]}}}>
  // CHECK-LABEL: @descriptor_rhs_dot_layout
  // CHECK: tt.dot {{.*}} -> tensor<128x128xf32, #[[$DEFAULT_DOT]]>
  // TWOCTA-DAG: #[[$TWOCTA_DOT:.+]] = #ttg.blocked<{{.*}}CGALayout = {{\[\[1, 0\]\]}}}>
  // TWOCTA-LABEL: @descriptor_rhs_dot_layout
  // TWOCTA: tt.dot {{.*}} -> tensor<128x128xf32, #[[$TWOCTA_DOT]]>
  tt.func @descriptor_rhs_dot_layout(
      %a_desc: !tt.tensordesc<128x64xbf16>,
      %b_desc: !tt.tensordesc<64x128xbf16>) -> tensor<128x128xf32, #dot_parent> {
    %zero = arith.constant 0 : i32
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #dot_parent>
    %a = tt.descriptor_load %a_desc[%zero, %zero] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #load_a>
    %b = tt.descriptor_load %b_desc[%zero, %zero] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16, #load_b>
    %ad = ttg.convert_layout %a : tensor<128x64xbf16, #load_a> -> tensor<128x64xbf16, #dot_a>
    %bd = ttg.convert_layout %b : tensor<64x128xbf16, #load_b> -> tensor<64x128xbf16, #dot_b>
    %d = tt.dot %ad, %bd, %c : tensor<128x64xbf16, #dot_a> * tensor<64x128xbf16, #dot_b> -> tensor<128x128xf32, #dot_parent>
    tt.return %d : tensor<128x128xf32, #dot_parent>
  }
}

// -----

#sm90_dot_parent = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#sm90_load_a = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#sm90_load_b = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0], CGALayout = [[0, 0]]}>
#sm90_dot_a = #ttg.dot_op<{opIdx = 0, parent = #sm90_dot_parent}>
#sm90_dot_b = #ttg.dot_op<{opIdx = 1, parent = #sm90_dot_parent}>
module attributes {"ttg.num-ctas" = 2 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // TWOCTA-DAG: #[[$SM90_DOT:.+]] = #ttg.blocked<{{.*}}CGALayout = {{\[\[0, 1\]\]}}}>
  // TWOCTA-LABEL: @descriptor_rhs_dot_sm90_fallback
  // TWOCTA: tt.dot {{.*}} -> tensor<128x128xf32, #[[$SM90_DOT]]>
  tt.func @descriptor_rhs_dot_sm90_fallback(
      %a_desc: !tt.tensordesc<128x64xbf16>,
      %b_desc: !tt.tensordesc<64x128xbf16>) -> tensor<128x128xf32, #sm90_dot_parent> {
    %zero = arith.constant 0 : i32
    %c = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #sm90_dot_parent>
    %a = tt.descriptor_load %a_desc[%zero, %zero] : !tt.tensordesc<128x64xbf16> -> tensor<128x64xbf16, #sm90_load_a>
    %b = tt.descriptor_load %b_desc[%zero, %zero] : !tt.tensordesc<64x128xbf16> -> tensor<64x128xbf16, #sm90_load_b>
    %ad = ttg.convert_layout %a : tensor<128x64xbf16, #sm90_load_a> -> tensor<128x64xbf16, #sm90_dot_a>
    %bd = ttg.convert_layout %b : tensor<64x128xbf16, #sm90_load_b> -> tensor<64x128xbf16, #sm90_dot_b>
    %d = tt.dot %ad, %bd, %c : tensor<128x64xbf16, #sm90_dot_a> * tensor<64x128xbf16, #sm90_dot_b> -> tensor<128x128xf32, #sm90_dot_parent>
    tt.return %d : tensor<128x128xf32, #sm90_dot_parent>
  }
}
