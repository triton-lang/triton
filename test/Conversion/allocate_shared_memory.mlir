// RUN: triton-opt %s -split-input-file --allocate-shared-memory | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 131072 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @gather_op
// TODO(jeff): Optimize the lowering to reduce shared memory usage.
tt.func @gather_op(%arg0: tensor<1024x256xi32, #blocked>, %arg1: tensor<128x256xf32, #blocked>) {
  // CHECK-NEXT: allocation.offset = 0 : i32
  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<128x256xf32, #blocked>, tensor<1024x256xi32, #blocked>) -> tensor<1024x256xf32, #blocked>
  tt.return
}

}

// -----

#reduce = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @reduce_scratch_size
tt.func @reduce_scratch_size(%arg0: tensor<1x256xf32, #reduce>) {
  // CHECK: "tt.reduce"
  // CHECK: }) {allocation.offset = 0 : i32, allocation.size = 16 : i32}
  %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
  ^bb0(%lhs: f32, %rhs: f32):
    %sum = arith.addf %lhs, %rhs : f32
    tt.reduce.return %sum : f32
  }) : (tensor<1x256xf32, #reduce>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #reduce}>>
  tt.return
}

}

// -----

#src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#dst = #ttg.slice<{dim = 1, parent = #dst_parent}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @convert_layout_scratch_size
tt.func @convert_layout_scratch_size(%arg0: tensor<128xi32, #src>) {
  // CHECK: ttg.convert_layout {{.*}} {allocation.offset = 0 : i32, allocation.size = 512 : i32}
  %0 = ttg.convert_layout %arg0 : tensor<128xi32, #src> -> tensor<128xi32, #dst>
  tt.return
}

}
