// RUN: triton-opt %s --allocate-shared-memory | FileCheck %s

// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 131072 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @gather_op
// TODO(jeff): Optimize the lowering to reduce shared memory usage.
tt.func @gather_op(%arg0: tensor<1024x256xi32>, %arg1: tensor<128x256xf32>) {
  // CHECK-NEXT: allocation.offset = 0 : i32
  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<128x256xf32>, tensor<1024x256xi32>) -> tensor<1024x256xf32>
  tt.return
}

}
