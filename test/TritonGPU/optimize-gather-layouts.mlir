// RUN: triton-opt %s -split-input-file --tritongpu-optimize-gather-layouts | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [2, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_square_axis_0
tt.func @set_warp_shuffle_layout_square_axis_0(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked> {
  // CHECK: tt.gather {{.*}} (tensor<64x64xf32, [[LAYOUT]]>, tensor<64x64xi32, [[LAYOUT]]>) -> tensor<64x64xf32, [[LAYOUT]]>
  %0 = tt.gather %arg0[%arg1] {axis = 0 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
  tt.return %0 : tensor<64x64xf32, #blocked>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_square_axis_1
tt.func @set_warp_shuffle_layout_square_axis_1(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked> {
  // CHECK: tt.gather {{.*}} (tensor<64x64xf32, [[LAYOUT]]>, tensor<64x64xi32, [[LAYOUT]]>) -> tensor<64x64xf32, [[LAYOUT]]>
  %0 = tt.gather %arg0[%arg1] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x64xi32, #blocked>) -> tensor<64x64xf32, #blocked>
  tt.return %0 : tensor<64x64xf32, #blocked>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_warp_broadcast
tt.func @set_warp_shuffle_layout_warp_broadcast(%arg0: tensor<64x64xf32, #blocked>, %arg1: tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked> {
  // CHECK: tt.gather {{.*}} [[LAYOUT]]>
  %0 = tt.gather %arg0[%arg1] {axis = 1 : i32} : (tensor<64x64xf32, #blocked>, tensor<64x1xi32, #blocked>) -> tensor<64x1xf32, #blocked>
  tt.return %0 : tensor<64x1xf32, #blocked>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2, 1], threadsPerWarp = [16, 2, 1], warpsPerCTA = [2, 1, 2], order = [1, 0, 2]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 1, 32], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_3d_warp
tt.func @set_warp_shuffle_layout_3d_warp(%arg0: tensor<32x2x32xf32, #blocked>, %arg1: tensor<32x2x2xi32, #blocked>) -> tensor<32x2x2xf32, #blocked> {
  // CHECK: tt.gather {{.*}} [[LAYOUT]]>
    %0 = tt.gather %arg0[%arg1] {axis = 2 : i32} : (tensor<32x2x32xf32, #blocked>, tensor<32x2x2xi32, #blocked>) -> tensor<32x2x2xf32, #blocked>
    tt.return %0 : tensor<32x2x2xf32, #blocked>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2, 1], threadsPerWarp = [16, 2, 1], warpsPerCTA = [2, 1, 2], order = [1, 0, 2]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 2, 16], warpsPerCTA = [2, 2, 1], order = [2, 1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_3d_warp_thread_split
tt.func @set_warp_shuffle_layout_3d_warp_thread_split(%arg0: tensor<32x4x16xf32, #blocked>, %arg1: tensor<32x4x2xi32, #blocked>) -> tensor<32x4x2xf32, #blocked> {
  // CHECK: tt.gather {{.*}} [[LAYOUT]]>
    %0 = tt.gather %arg0[%arg1] {axis = 2 : i32} : (tensor<32x4x16xf32, #blocked>, tensor<32x4x2xi32, #blocked>) -> tensor<32x4x2xf32, #blocked>
    tt.return %0 : tensor<32x4x2xf32, #blocked>
}

}


// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_thread_broadcast
tt.func @set_warp_shuffle_layout_thread_broadcast(%arg0: tensor<16x64xf32, #blocked>, %arg1: tensor<16x1xi32, #blocked>) -> tensor<16x1xf32, #blocked> {
  // CHECK: tt.gather {{.*}} [[LAYOUT]]>
  %0 = tt.gather %arg0[%arg1] {axis = 1 : i32} : (tensor<16x64xf32, #blocked>, tensor<16x1xi32, #blocked>) -> tensor<16x1xf32, #blocked>
  tt.return %0 : tensor<16x1xf32, #blocked>
}

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [16, 2], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK: [[LAYOUT:#.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK: set_warp_shuffle_layout_large_source
tt.func @set_warp_shuffle_layout_large_source(%arg0: tensor<256x256xf32, #blocked>, %arg1: tensor<256x8xi32, #blocked>) -> tensor<256x8xf32, #blocked> {
  // CHECK: tt.gather {{.*}} [[LAYOUT]]>
  %0 = tt.gather %arg0[%arg1] {axis = 1 : i32} : (tensor<256x256xf32, #blocked>, tensor<256x8xi32, #blocked>) -> tensor<256x8xf32, #blocked>
  tt.return %0 : tensor<256x8xf32, #blocked>
}

}
