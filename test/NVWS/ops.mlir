// RUN: triton-opt --split-input-file %s | FileCheck %s

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_create_single
  // CHECK: nvws.aref.create
  tt.func @aref_create_single(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    tt.return
  }

}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_get
  // CHECK: nvws.aref.get.enter
  // CHECK: nvws.aref.get.exit
  tt.func @aref_get(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    %1:3 = nvws.aref.get.enter %0[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
    nvws.aref.get.exit %0[%c0_i32], %1#2 [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>, !ttg.async.token
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: aref_put
  // CHECK: nvws.aref.put.enter
  // CHECK: nvws.aref.put.exit
  tt.func @aref_put(%d : !ttg.memdesc<1x64x16xf16, #shared0, #smem>, %e : !ttg.memdesc<1x16x32xf16, #shared0, #smem>) {
    %c0_i32 = arith.constant {ttg.partition = [0, 1]} 0 : i32
    %0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>
    %1:3 = nvws.aref.put.enter %0[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
    nvws.aref.put.exit %0[%c0_i32], %1#2 [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<1x64x16xf16, #shared0, #smem>, !ttg.memdesc<1x16x32xf16, #shared0, #smem>]>, !ttg.async.token
    tt.return
  }
}

// -----


// CHECK-LABEL: @warp_group_nothing
tt.func @warp_group_nothing() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  tt.return
}

// CHECK-LABEL: @warp_1_partition
tt.func @warp_1_partition() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  // CHECK-NEXT:  num_warps(4) {
  partition0  num_warps(4) {
  // CHECK-NEXT: nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  tt.return
}

// CHECK-LABEL: @warp_2_partition
tt.func @warp_2_partition() {
  // CHECK-NEXT: nvws.warp_group
  nvws.warp_group
  // CHECK-NEXT: partition0  num_warps(8) {
  partition0  num_warps(8) {
  // CHECK-NEXT: nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  // CHECK-NEXT: partition1 num_warps(4) {
  partition1 num_warps(4) {
  // CHECK-NEXT:   nvws.warp_group.return
    nvws.warp_group.return
  // CHECK-NEXT: }
  }
  tt.return
}

// CHECK-LABEL: @token_producer_consumer
tt.func @token_producer_consumer() {

  // CHECK: nvws.create_token
  // CHECK: nvws.producer_acquire
  // CHECK: nvws.producer_commit
  // CHECK: nvws.consumer_wait
  // CHECK: nvws.consumer_release

  %0 = nvws.create_token {loadType = 1 : i32, numBuffers = 3 : i32} : tensor<3x!nvws.token>

  %c0_i32 = arith.constant {async_task_id = dense<0> : vector<1xi32>} 0 : i32
  %false = arith.constant {async_task_id = dense<0> : vector<1xi32>} false

  nvws.producer_acquire %0, %c0_i32, %false {async_task_id = dense<0> : vector<1xi32>} : tensor<3x!nvws.token>, i32, i1
  nvws.producer_commit %0, %c0_i32 {async_task_id = dense<0> : vector<1xi32>} : tensor<3x!nvws.token>, i32
  nvws.consumer_wait %0, %c0_i32, %false {async_task_id = dense<1> : vector<1xi32>} : tensor<3x!nvws.token>, i32, i1
  nvws.consumer_release %0, %c0_i32 {async_task_id = dense<1> : vector<1xi32>} : tensor<3x!nvws.token>, i32
  tt.return
}
