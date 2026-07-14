// RUN: triton-opt %s -split-input-file --allocate-shared-memory | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 131072 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @gather_op
// TODO(jeff): Optimize the lowering to reduce shared memory usage.
tt.func @gather_op(%arg0: tensor<1024x256xi32, #blocked>, %arg1: tensor<128x256xf32, #blocked>) {
  // CHECK-NEXT: allocation.offset = 0 : i32, allocation.size = 131072 : i32
  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<128x256xf32, #blocked>, tensor<1024x256xi32, #blocked>) -> tensor<1024x256xf32, #blocked>
  tt.return
}

}

// -----

#atomic_blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#atomic_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#atomic_smem = #ttg.shared_memory

// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 132 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @local_atomic_scratch_size
tt.func @local_atomic_scratch_size(
    %indices: tensor<1xi32, #atomic_blocked>,
    %values: tensor<1xi32, #atomic_blocked>,
    %out: tensor<1x!tt.ptr<i32>, #atomic_blocked>) {
  // CHECK: %[[ATOMIC_DST:.*]] = ttg.local_alloc {allocation.offset = 0 : i32}
  %dst = ttg.local_alloc
      : () -> !ttg.memdesc<1xi32, #atomic_shared, #atomic_smem, mutable>
  // CHECK: ttg.local_atomic_scatter_rmw {{.*}} {allocation.offset = 128 : i32, allocation.size = 4 : i32, axis = 0 : i32}
  %old = ttg.local_atomic_scatter_rmw add, %dst[%indices], %values {axis = 0 : i32}
      : (!ttg.memdesc<1xi32, #atomic_shared, #atomic_smem, mutable>,
         tensor<1xi32, #atomic_blocked>, tensor<1xi32, #atomic_blocked>)
      -> tensor<1xi32, #atomic_blocked>
  tt.store %out, %old : tensor<1x!tt.ptr<i32>, #atomic_blocked>
  tt.return
}

}

// -----

#shuffle_src = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4]], lane = [[1, 0], [0, 0], [0, 0], [0, 0], [0, 0]], warp = [], block = []}>
#shuffle_dst = #ttg.linear<{register = [[1, 0], [0, 4]], lane = [[0, 0], [0, 0], [0, 1], [0, 2], [0, 0]], warp = [], block = []}>

// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 0 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {

// CHECK-LABEL: @forced_warp_shuffle_has_no_scratch
tt.func @forced_warp_shuffle_has_no_scratch(
    %arg0: tensor<2x8xi16, #shuffle_src>) attributes {always_use_warp_shuffle} {
  // CHECK-NOT: allocation.offset
  // CHECK: ttg.convert_layout
  %0 = ttg.convert_layout %arg0 : tensor<2x8xi16, #shuffle_src> -> tensor<2x8xi16, #shuffle_dst>
  tt.return
}

}

// -----

#call_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#call_smem = #ttg.shared_memory

// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 4 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {

tt.func private @scratch_callee() attributes {noinline = true} {
  %0 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #call_shared, #call_smem, mutable>
  tt.return
}

// CHECK-LABEL: @virtual_call_frame_has_no_size
tt.func @virtual_call_frame_has_no_size() {
  // CHECK: tt.call @scratch_callee() {allocation.offset = 0 : i32}
  tt.call @scratch_callee() : () -> ()
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
