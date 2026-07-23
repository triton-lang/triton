// RUN: triton-opt %s -triton-nvidia-gpu-plan-cta | FileCheck %s

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
