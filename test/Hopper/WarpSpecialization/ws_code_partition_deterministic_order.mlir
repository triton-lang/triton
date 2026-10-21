// RUN: triton-opt %s -split-input-file --nvgpu-test-ws-code-partition=num-buffers=1 | FileCheck %s

// Two consumer partitions share one buffer (b). insertAsyncComm used to walk a
// pointer-keyed DenseMap, so the order of the consumer_release ops emitted after
// the dots depended on allocation addresses. Every copy below (each split is a
// fresh module, so Channel allocations differ) must emit the same order:
// release of the channel waited second first, then the one waited first.

// CHECK-LABEL: @order_0
// CHECK: partition1
// CHECK: nvws.consumer_wait [[A:%arg[0-9]+]]
// CHECK: nvws.consumer_wait [[B:%arg[0-9]+]]
// CHECK: nvws.consumer_release [[B]],
// CHECK: nvws.consumer_release [[A]],

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @order_0(%pa: tensor<64x64x!tt.ptr<f16>, #blocked>, %pb: tensor<64x128x!tt.ptr<f16>, #blocked1>, %pc: tensor<64x64x!tt.ptr<f16>, #blocked>, %n: i32) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc1 = %acc, %acc2 = %acc) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>) : i32 {
      %a = tt.load %pa {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %a_s = ttg.local_alloc %a {async_task_id = array<i32: 1>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %b = tt.load %pb {async_task_id = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %b_s = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory>
      %c = tt.load %pc {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %c_s = ttg.local_alloc %c {async_task_id = array<i32: 2>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %d1 = ttng.warp_group_dot %a_s, %b_s, %acc1 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %d2 = ttng.warp_group_dot %c_s, %b_s, %acc2 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %d1, %d2 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>
    } {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----

// CHECK-LABEL: @order_1
// CHECK: partition1
// CHECK: nvws.consumer_wait [[A:%arg[0-9]+]]
// CHECK: nvws.consumer_wait [[B:%arg[0-9]+]]
// CHECK: nvws.consumer_release [[B]],
// CHECK: nvws.consumer_release [[A]],

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @order_1(%pa: tensor<64x64x!tt.ptr<f16>, #blocked>, %pb: tensor<64x128x!tt.ptr<f16>, #blocked1>, %pc: tensor<64x64x!tt.ptr<f16>, #blocked>, %n: i32) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc1 = %acc, %acc2 = %acc) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>) : i32 {
      %a = tt.load %pa {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %a_s = ttg.local_alloc %a {async_task_id = array<i32: 1>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %b = tt.load %pb {async_task_id = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %b_s = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory>
      %c = tt.load %pc {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %c_s = ttg.local_alloc %c {async_task_id = array<i32: 2>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %d1 = ttng.warp_group_dot %a_s, %b_s, %acc1 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %d2 = ttng.warp_group_dot %c_s, %b_s, %acc2 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %d1, %d2 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>
    } {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----

// CHECK-LABEL: @order_2
// CHECK: partition1
// CHECK: nvws.consumer_wait [[A:%arg[0-9]+]]
// CHECK: nvws.consumer_wait [[B:%arg[0-9]+]]
// CHECK: nvws.consumer_release [[B]],
// CHECK: nvws.consumer_release [[A]],

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @order_2(%pa: tensor<64x64x!tt.ptr<f16>, #blocked>, %pb: tensor<64x128x!tt.ptr<f16>, #blocked1>, %pc: tensor<64x64x!tt.ptr<f16>, #blocked>, %n: i32) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc1 = %acc, %acc2 = %acc) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>) : i32 {
      %a = tt.load %pa {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %a_s = ttg.local_alloc %a {async_task_id = array<i32: 1>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %b = tt.load %pb {async_task_id = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %b_s = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory>
      %c = tt.load %pc {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %c_s = ttg.local_alloc %c {async_task_id = array<i32: 2>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %d1 = ttng.warp_group_dot %a_s, %b_s, %acc1 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %d2 = ttng.warp_group_dot %c_s, %b_s, %acc2 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %d1, %d2 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>
    } {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----

// CHECK-LABEL: @order_3
// CHECK: partition1
// CHECK: nvws.consumer_wait [[A:%arg[0-9]+]]
// CHECK: nvws.consumer_wait [[B:%arg[0-9]+]]
// CHECK: nvws.consumer_release [[B]],
// CHECK: nvws.consumer_release [[A]],

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @order_3(%pa: tensor<64x64x!tt.ptr<f16>, #blocked>, %pb: tensor<64x128x!tt.ptr<f16>, #blocked1>, %pc: tensor<64x64x!tt.ptr<f16>, #blocked>, %n: i32) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc1 = %acc, %acc2 = %acc) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>) : i32 {
      %a = tt.load %pa {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %a_s = ttg.local_alloc %a {async_task_id = array<i32: 1>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %b = tt.load %pb {async_task_id = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %b_s = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory>
      %c = tt.load %pc {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %c_s = ttg.local_alloc %c {async_task_id = array<i32: 2>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %d1 = ttng.warp_group_dot %a_s, %b_s, %acc1 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %d2 = ttng.warp_group_dot %c_s, %b_s, %acc2 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %d1, %d2 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>
    } {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----

// CHECK-LABEL: @order_4
// CHECK: partition1
// CHECK: nvws.consumer_wait [[A:%arg[0-9]+]]
// CHECK: nvws.consumer_wait [[B:%arg[0-9]+]]
// CHECK: nvws.consumer_release [[B]],
// CHECK: nvws.consumer_release [[A]],

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @order_4(%pa: tensor<64x64x!tt.ptr<f16>, #blocked>, %pb: tensor<64x128x!tt.ptr<f16>, #blocked1>, %pc: tensor<64x64x!tt.ptr<f16>, #blocked>, %n: i32) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc1 = %acc, %acc2 = %acc) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>) : i32 {
      %a = tt.load %pa {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %a_s = ttg.local_alloc %a {async_task_id = array<i32: 1>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %b = tt.load %pb {async_task_id = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %b_s = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory>
      %c = tt.load %pc {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %c_s = ttg.local_alloc %c {async_task_id = array<i32: 2>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %d1 = ttng.warp_group_dot %a_s, %b_s, %acc1 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %d2 = ttng.warp_group_dot %c_s, %b_s, %acc2 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %d1, %d2 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>
    } {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----

// CHECK-LABEL: @order_5
// CHECK: partition1
// CHECK: nvws.consumer_wait [[A:%arg[0-9]+]]
// CHECK: nvws.consumer_wait [[B:%arg[0-9]+]]
// CHECK: nvws.consumer_release [[B]],
// CHECK: nvws.consumer_release [[A]],

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @order_5(%pa: tensor<64x64x!tt.ptr<f16>, #blocked>, %pb: tensor<64x128x!tt.ptr<f16>, #blocked1>, %pc: tensor<64x64x!tt.ptr<f16>, #blocked>, %n: i32) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc1 = %acc, %acc2 = %acc) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>) : i32 {
      %a = tt.load %pa {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %a_s = ttg.local_alloc %a {async_task_id = array<i32: 1>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %b = tt.load %pb {async_task_id = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %b_s = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory>
      %c = tt.load %pc {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %c_s = ttg.local_alloc %c {async_task_id = array<i32: 2>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %d1 = ttng.warp_group_dot %a_s, %b_s, %acc1 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %d2 = ttng.warp_group_dot %c_s, %b_s, %acc2 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %d1, %d2 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>
    } {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----

// CHECK-LABEL: @order_6
// CHECK: partition1
// CHECK: nvws.consumer_wait [[A:%arg[0-9]+]]
// CHECK: nvws.consumer_wait [[B:%arg[0-9]+]]
// CHECK: nvws.consumer_release [[B]],
// CHECK: nvws.consumer_release [[A]],

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @order_6(%pa: tensor<64x64x!tt.ptr<f16>, #blocked>, %pb: tensor<64x128x!tt.ptr<f16>, #blocked1>, %pc: tensor<64x64x!tt.ptr<f16>, #blocked>, %n: i32) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc1 = %acc, %acc2 = %acc) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>) : i32 {
      %a = tt.load %pa {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %a_s = ttg.local_alloc %a {async_task_id = array<i32: 1>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %b = tt.load %pb {async_task_id = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %b_s = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory>
      %c = tt.load %pc {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %c_s = ttg.local_alloc %c {async_task_id = array<i32: 2>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %d1 = ttng.warp_group_dot %a_s, %b_s, %acc1 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %d2 = ttng.warp_group_dot %c_s, %b_s, %acc2 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %d1, %d2 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>
    } {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}

// -----

// CHECK-LABEL: @order_7
// CHECK: partition1
// CHECK: nvws.consumer_wait [[A:%arg[0-9]+]]
// CHECK: nvws.consumer_wait [[B:%arg[0-9]+]]
// CHECK: nvws.consumer_release [[B]],
// CHECK: nvws.consumer_release [[A]],

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @order_7(%pa: tensor<64x64x!tt.ptr<f16>, #blocked>, %pb: tensor<64x128x!tt.ptr<f16>, #blocked1>, %pc: tensor<64x64x!tt.ptr<f16>, #blocked>, %n: i32) {
    %c0 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 0 : i32
    %c1 = arith.constant {async_task_id = array<i32: 0, 1, 2>} 1 : i32
    %acc = arith.constant {async_task_id = array<i32: 1, 2>} dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc1 = %acc, %acc2 = %acc) -> (tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>) : i32 {
      %a = tt.load %pa {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %a_s = ttg.local_alloc %a {async_task_id = array<i32: 1>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %b = tt.load %pb {async_task_id = array<i32: 0>} : tensor<64x128x!tt.ptr<f16>, #blocked1>
      %b_s = ttg.local_alloc %b {async_task_id = array<i32: 1, 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory>
      %c = tt.load %pc {async_task_id = array<i32: 0>} : tensor<64x64x!tt.ptr<f16>, #blocked>
      %c_s = ttg.local_alloc %c {async_task_id = array<i32: 2>} : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory>
      %d1 = ttng.warp_group_dot %a_s, %b_s, %acc1 {async_task_id = array<i32: 1>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      %d2 = ttng.warp_group_dot %c_s, %b_s, %acc2 {async_task_id = array<i32: 2>, inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #ttg.shared_memory> * !ttg.memdesc<64x128xf16, #shared, #ttg.shared_memory> -> tensor<64x128xf32, #mma>
      scf.yield {async_task_id = array<i32: 0, 1, 2>} %d1, %d2 : tensor<64x128xf32, #mma>, tensor<64x128xf32, #mma>
    } {async_task_id = array<i32: 0, 1, 2>}
    tt.return
  }
}
