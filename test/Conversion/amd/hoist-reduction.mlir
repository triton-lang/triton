// RUN: triton-opt %s -split-input-file --tritonamdgpu-hoist-reduction | FileCheck %s

// CHECK-LABEL: hoist_reduction_accumulate
// CHECK: %[[CST:.+]] = arith.constant dense<0>
// CHECK: %[[PRE_DOT_CVT:.+]] = triton_gpu.convert_layout %[[CST]]
// CHECK: %[[FOR_RESULT:.+]] = scf.for {{.*}}
// CHECK: %[[DOT_RESULT:.+]] = tt.dot
// CHECK: scf.yield %[[DOT_RESULT]]
// CHECK: %[[REDUCED:.+]] = "tt.reduce"(%[[FOR_RESULT]])
// CHECK: %[[FINAL_RESULT:.+]] = triton_gpu.convert_layout %[[REDUCED]]
// CHECK: tt.store {{.*}} %[[FINAL_RESULT]]
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
module attributes {"triton_gpu.target" = "hip:gfx1030", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @hoist_reduction(%x : tensor<8x1x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, %y : tensor<8x64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, %ptr : tensor<1x32x!tt.ptr<i32>, #blocked1>) {
    %iter_begin = arith.constant 0 : i32
    %iter_end = arith.constant 3 : i32
    %iter_step = arith.constant 1 : i32
    %zero = arith.constant dense<0> : tensor<1x32xi32, #blocked1>
    %acc = scf.for %iter = %iter_begin to %iter_end step %iter_step iter_args(%acc = %zero) -> (tensor<1x32xi32, #blocked1>)  : i32 {
      %acc_ext = tt.reshape %acc {allow_reorder = true} : tensor<1x32xi32, #blocked1> -> tensor<1x1x32xi32, #blocked2>
      %acc3d_batched = tt.broadcast %acc_ext : tensor<1x1x32xi32, #blocked2> -> tensor<8x1x32xi32, #blocked2>
      %acc3d_in = triton_gpu.convert_layout %acc3d_batched : tensor<8x1x32xi32, #blocked2> -> tensor<8x1x32xi32, #blocked>
      %acc3d_out = tt.dot %x, %y, %acc3d_in : tensor<8x1x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x1x32xi32, #blocked>
      %acc_next_slice = "tt.reduce"(%acc3d_out) <{axis = 0 : i32}> ({
        ^bb0(%arg0: i32, %arg1: i32):
            %sum = arith.addi %arg0, %arg1 : i32
            tt.reduce.return %sum : i32
      }) : (tensor<8x1x32xi32, #blocked>) -> tensor<1x32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %acc_next = triton_gpu.convert_layout %acc_next_slice : tensor<1x32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked1>
      scf.yield %acc_next : tensor<1x32xi32, #blocked1>
    }
    tt.store %ptr, %acc : tensor<1x32x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}

// -----

// CHECK-LABEL: hoist_reduction_add
#blocked = #triton_gpu.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
#blocked1 = #triton_gpu.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #triton_gpu.blocked<{sizePerThread = [1, 1, 2], threadsPerWarp = [1, 1, 32], warpsPerCTA = [1, 2, 2], order = [2, 1, 0]}>
module attributes {"triton_gpu.target" = "hip:gfx1030", "triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32, "triton_gpu.threads-per-warp" = 32 : i32} {
  tt.func @hoist_reduction(%x : tensor<8x1x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>>, %y : tensor<8x64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>>, %ptr : tensor<1x32x!tt.ptr<i32>, #blocked1>) {
    %iter_begin = arith.constant 0 : i32
    %iter_end = arith.constant 3 : i32
    %iter_step = arith.constant 1 : i32
    %zero = arith.constant dense<0> : tensor<1x32xi32, #blocked1>
    %acc = scf.for %iter = %iter_begin to %iter_end step %iter_step iter_args(%acc = %zero) -> (tensor<1x32xi32, #blocked1>)  : i32 {
      %acc3d_in = arith.constant dense<0> : tensor<8x1x32xi32, #blocked>
      %acc3d_out = tt.dot %x, %y, %acc3d_in : tensor<8x1x64xi8, #triton_gpu.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<8x64x32xi8, #triton_gpu.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<8x1x32xi32, #blocked>
      %acc_next_slice = "tt.reduce"(%acc3d_out) <{axis = 0 : i32}> ({
        ^bb0(%arg0: i32, %arg1: i32):
            %sum = arith.addi %arg0, %arg1 : i32
            tt.reduce.return %sum : i32
      }) : (tensor<8x1x32xi32, #blocked>) -> tensor<1x32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>>
      %acc_next = triton_gpu.convert_layout %acc_next_slice : tensor<1x32xi32, #triton_gpu.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked1>
      %acc_next_add = arith.addi %acc, %acc_next : tensor<1x32xi32, #blocked1>
      scf.yield %acc_next_add : tensor<1x32xi32, #blocked1>
    }
    tt.store %ptr, %acc : tensor<1x32x!tt.ptr<i32>, #blocked1>
    tt.return
  }
}
