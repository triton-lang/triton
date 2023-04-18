// RUN: triton-opt %s -tritongpu-convert-slice-reduction-to-block-reduction | FileCheck %s

#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [32, 1, 1], warpsPerCTA = [2, 2, 1], order = [0, 1, 2]}>
#out_slice = #triton_gpu.slice<{dim = 1, parent = #triton_gpu.slice<{dim = 0, parent = #blocked}>}>

// CHECK-LABEL: @reduction_with_sliced_layout
tt.func public @reduction_with_sliced_layout(%arg: tensor<16x32xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>)
  	-> tensor<16xf32, #out_slice> {
  // CHECK:      %[[BLOCKED_ARG:.*]] = triton_gpu.convert_layout
  // CHECK-SAME:   -> tensor<16x32xf32, #blocked>
  // CHECK:      tt.reduce(%[[BLOCKED_ARG]])
  %reduction = "tt.reduce"(%arg) ({
  ^bb0(%lhs: f32, %rhs: f32):
    %cmp = "triton_gpu.cmpf"(%lhs, %rhs) {predicate = 2 : i64} : (f32, f32) -> i1
    %result = arith.select %cmp, %lhs, %rhs : f32
    tt.reduce.return %result : f32
  }) {axis = 1 : i32} : (tensor<16x32xf32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>) -> tensor<16xf32, #out_slice>
  tt.return %reduction : tensor<16xf32, #out_slice>
}
