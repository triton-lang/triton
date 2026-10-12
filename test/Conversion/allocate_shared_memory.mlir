// RUN: triton-opt %s -split-input-file --allocate-shared-memory | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 131072 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @gather_op
// TODO(jeff): Optimize the lowering to reduce shared memory usage.
tt.func @gather_op(%arg0: tensor<1024x256xi32, #blocked>, %arg1: tensor<128x256xf32, #blocked>) {
  // CHECK-NEXT: allocation.offset = 0 : i32
  // CHECK-SAME: allocation.size = 131072 : i32
  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<128x256xf32, #blocked>, tensor<1024x256xi32, #blocked>) -> tensor<1024x256xf32, #blocked>
  tt.return
}
}

// -----

#nested_src = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#nested_dst_parent = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#nested_dst = #ttg.slice<{dim = 1, parent = #nested_dst_parent}>
#nested_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nested_smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @nested_scratch_leaf
  tt.func private @nested_scratch_leaf(%value: tensor<128xi32, #nested_src>) {
    // CHECK: ttg.convert_layout {{.*}}allocation.offset = 0 : i32, allocation.size = 512 : i32
    %converted = ttg.convert_layout %value
        : tensor<128xi32, #nested_src> -> tensor<128xi32, #nested_dst>
    tt.return
  }

  // CHECK-LABEL: @nested_scratch_middle
  tt.func private @nested_scratch_middle(%value: tensor<128xi32, #nested_src>) {
    // CHECK: tt.call @nested_scratch_leaf{{.*}}allocation.offset = 0 : i32, allocation.size = 512 : i32
    tt.call @nested_scratch_leaf(%value) : (tensor<128xi32, #nested_src>) -> ()
    tt.return
  }

  // CHECK-LABEL: @nested_scratch_caller
  tt.func public @nested_scratch_caller(%value: tensor<128xi32, #nested_src>) {
    // CHECK: %[[LIVE_BUFFER:.*]] = ttg.local_alloc {allocation.offset = 0 : i32}
    %live = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #nested_shared, #nested_smem, mutable>
    // CHECK: tt.call @nested_scratch_middle{{.*}}allocation.offset = 512 : i32, allocation.size = 512 : i32
    tt.call @nested_scratch_middle(%value) : (tensor<128xi32, #nested_src>) -> ()
    %preserved = ttg.local_load %live
        : !ttg.memdesc<128xi32, #nested_shared, #nested_smem, mutable>
        -> tensor<128xi32, #nested_src>
    tt.return
  }
}
