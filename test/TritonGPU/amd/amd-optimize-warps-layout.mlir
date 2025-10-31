// RUN: triton-opt %s -split-input-file --tritonamdgpu-optimize-warp-layout | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @row_reduce_sum_kernel_v2(%OpA: tensor<8x1024xf32, #blocked>,
                                           %0: tensor<8xi1, #blocked1>,
                                           %2: tensor<8x!tt.ptr<f32>, #blocked1>)
                                           attributes {noinline = false} {
    // CHECK: #[[NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [8, 1], order = [1, 0]}>
    %out = "tt.reduce"(%OpA) <{axis = 1 : i32}> ({
    // CHECK: %[[CONV_OUT:.*]] = ttg.convert_layout %[[IN:.*]] : tensor<8x1024xf32, #blocked> -> tensor<8x1024xf32, #[[NEW_BLOCKED]]>
    // CHECK: %[[REDUCE_RESULT:.*]] = "tt.reduce"(%[[CONV_OUT]]) <{axis = 1 : i32}> ({
    // CHECK-NEXT: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
    // CHECK-NEXT:   %[[SUM:.*]] = arith.addf %[[ARG0]], %[[ARG1]] : f32
    // CHECK-NEXT:   tt.reduce.return %[[SUM]] : f32
    // CHECK-NEXT: }) : (tensor<8x1024xf32, #[[NEW_BLOCKED]]>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #[[NEW_BLOCKED]]}>>
    ^bb0(%out_22: f32, %out_23: f32):
      %out_24 = arith.addf %out_22, %out_23 : f32
      tt.reduce.return %out_24 : f32
    }) : (tensor<8x1024xf32, #blocked>) -> tensor<8xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %3 = ttg.convert_layout %out : tensor<8xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<8xf32, #blocked1>
    tt.store %2, %3, %0 : tensor<8x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [2, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [8], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @row_reduce_sum_kernel_v2(%OpA: tensor<4x1024xf32, #blocked>,
                                           %0: tensor<4xi1, #blocked1>,
                                           %2: tensor<4x!tt.ptr<f32>, #blocked1>) attributes {noinline = false} {
    // CHECK: #[[NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 64], warpsPerCTA = [4, 2], order = [1, 0]}>
    %out_22 = "tt.reduce"(%OpA) <{axis = 1 : i32}> ({
    ^bb0(%out_23: f32, %out_24: f32):
      %out_25 = arith.addf %out_23, %out_24 : f32
      tt.reduce.return %out_25 : f32
    }) : (tensor<4x1024xf32, #blocked>) -> tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    // CHECK: %[[CONV_OUT:.*]] = ttg.convert_layout %[[IN:.*]] : tensor<4x1024xf32, #blocked> -> tensor<4x1024xf32, #[[NEW_BLOCKED]]>
    // CHECK: %[[REDUCE_RESULT:.*]] = "tt.reduce"(%[[CONV_OUT]]) <{axis = 1 : i32}> ({
    // CHECK-NEXT: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
    // CHECK-NEXT:   %[[SUM:.*]] = arith.addf %[[ARG0]], %[[ARG1]] : f32
    // CHECK-NEXT:   tt.reduce.return %[[SUM]] : f32
    // CHECK-NEXT: }) : (tensor<4x1024xf32, #[[NEW_BLOCKED]]>) -> tensor<4xf32, #ttg.slice<{dim = 1, parent = #[[NEW_BLOCKED]]}>>
    %3 = ttg.convert_layout %out_22 : tensor<4xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<4xf32, #blocked1>
    tt.store %2, %3, %0 : tensor<4x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
