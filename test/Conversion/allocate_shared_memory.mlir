// RUN: triton-opt %s --allocate-shared-memory | FileCheck %s

// CHECK-LABEL: module
// CHECK-SAME: triton_gpu.shared = 131072 : i32
module attributes {"triton_gpu.num-ctas" = 1 : i32, "triton_gpu.num-warps" = 4 : i32} {

// CHECK-LABEL: @gather_op
// TODO(jeff): Optimize the lowering to reduce shared memory usage.
tt.func @gather_op(%arg0: tensor<1024x4xi32>, %arg1: tensor<128x256xf32>) {
  // CHECK-NEXT: allocation.offset = 0 : i32
  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<128x256xf32>, tensor<1024x4xi32>) -> tensor<1024x4xf32>
  tt.return
}

}
