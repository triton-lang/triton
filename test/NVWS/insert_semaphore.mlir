// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-semaphore | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [128, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>

module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @warp_specialize_tma_matmul
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf16, #shared>, %arg4: !tt.tensordesc<128x64xf16, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // Two cross-partition values => two semaphore pairs
    // CHECK: [[ALLOC1:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY1:%.*]] = nvws.semaphore.create [[ALLOC1]] true
    // CHECK-NEXT: [[FULL1:%.*]] = nvws.semaphore.create [[ALLOC1]] false
    // CHECK: [[ALLOC2:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY2:%.*]] = nvws.semaphore.create [[ALLOC2]] true
    // CHECK-NEXT: [[FULL2:%.*]] = nvws.semaphore.create [[ALLOC2]] false
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      // Producer LHS: acquire EMPTY1, buffer, descriptor_load, release FULL1
      // CHECK: [[PTOK1:%.*]] = nvws.semaphore.acquire [[EMPTY1]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[PBUF1:%.*]] = nvws.semaphore.buffer [[EMPTY1]], [[PTOK1]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.descriptor_load {{.*}} 16384 [[PBUF1]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[FULL1]], [[PTOK1]] [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      %3 = tt.descriptor_load %arg3[%arg1, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      // Producer RHS: acquire EMPTY2, buffer, descriptor_load, release FULL2
      // CHECK: [[PTOK2:%.*]] = nvws.semaphore.acquire [[EMPTY2]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[PBUF2:%.*]] = nvws.semaphore.buffer [[EMPTY2]], [[PTOK2]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.descriptor_load {{.*}} 16384 [[PBUF2]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[FULL2]], [[PTOK2]] [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      %4 = tt.descriptor_load %arg4[%arg2, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>

      %5 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

      // Consumer RHS: acquire FULL2, buffer, memdesc_trans uses buffer
      // CHECK: [[GTOK2:%.*]] = nvws.semaphore.acquire [[FULL2]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF2:%.*]] = nvws.semaphore.buffer [[FULL2]], [[GTOK2]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[RHS:%.*]] = ttg.memdesc_trans [[GBUF2]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>}
      // Consumer LHS: acquire FULL1, buffer, MMA uses both buffers
      // CHECK: [[GTOK1:%.*]] = nvws.semaphore.acquire [[FULL1]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF1:%.*]] = nvws.semaphore.buffer [[FULL1]], [[GTOK1]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma [[GBUF1]], [[RHS]], {{.*}}, {{.*}}, {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %8 = ttng.tc_gen5_mma %5, %7, %result[%arg6], %true, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // Cross-release: consumer releases EMPTY semaphores
      // CHECK: nvws.semaphore.release [[EMPTY2]], [[GTOK2]] [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY1]], [[GTOK1]] [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      scf.yield {ttg.partition = array<i32: 0, 1>} %8 : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @specialize_load_only
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @specialize_load_only(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // Producer: acquire EMPTY, buffer, descriptor_load, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.descriptor_load {{.*}} 16384 [[PBUF]] {loop.cluster = 1 : i32, loop.stage = 0
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 1 : i32, loop.stage = 0, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      // Consumer: acquire FULL, buffer, local_load, release EMPTY
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[GBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[REG:%.*]] = ttg.local_load [[GBUF]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK]] [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: "use"([[REG]]) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      "use"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @no_value_semaphore
  tt.func @no_value_semaphore(%arg0: tensor<128x64xf16, #blocked1>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK-NOT: nvws.semaphore.create
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      %0 = "producer"(%arg0, %arg2) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>, i32) -> tensor<128x64xf16, #blocked1>
      "use"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 1>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @value_semaphore_multiple_producers
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @value_semaphore_multiple_producers(%arg0: tensor<128x64xf16, #blocked1>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // CHECK: [[VAL:%.*]] = "producer"
      // Producer: acquire EMPTY, buffer, store val, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store [[VAL]], [[PBUF]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK: "use0"([[VAL]])
      // CHECK: "use1"([[VAL]])
      // Consumer partition 2: acquire FULL, buffer, load, release EMPTY
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[GBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[LOADED:%.*]] = ttg.local_load [[GBUF]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK]] [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>}
      // CHECK: "use2"([[LOADED]])
      %0 = "producer"(%arg0, %arg2) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1>} : (tensor<128x64xf16, #blocked1>, i32) -> tensor<128x64xf16, #blocked1>
      "use0"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use1"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @load_used_as_reg_and_smem
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @load_used_as_reg_and_smem(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // Producer: acquire EMPTY, buffer, descriptor_load, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.descriptor_load {{.*}} 16384 [[PBUF]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %alloc = ttg.local_alloc %0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // Consumer 1 (register use): acquire FULL, buffer, local_load, release EMPTY
      // CHECK: [[GTOK1:%.*]] = nvws.semaphore.acquire [[FULL]] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[GBUF1:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK1]] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[REG:%.*]] = ttg.local_load [[GBUF1]] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK1]] [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: "use1"([[REG]]) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // Consumer 2 (smem use): acquire FULL, buffer used directly, release EMPTY
      // CHECK: [[GTOK2:%.*]] = nvws.semaphore.acquire [[FULL]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK2]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: "use2"([[GBUF2]]) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK2]] [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      "use1"(%0) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%alloc) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<128x64xf16, #shared, #smem>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @load_used_as_reg_and_smem_same_partition
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @load_used_as_reg_and_smem_same_partition(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // Producer: acquire EMPTY, buffer, descriptor_load, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.descriptor_load {{.*}} 16384 [[PBUF]] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %alloc = ttg.local_alloc %0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // Single consumer partition 0: acquire FULL, buffer, local_load + uses, release EMPTY
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[GBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[REG:%.*]] = ttg.local_load [[GBUF]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: "use1"([[REG]])
      // CHECK: "use2"([[GBUF]])
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK]] [#nvws.async_op<none>]
      "use1"(%0) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%alloc) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<128x64xf16, #shared, #smem>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @matmul_scaled_rhs_scales_tma
  tt.func @matmul_scaled_rhs_scales_tma(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf8E4M3FN, #shared3>, %arg4: !tt.tensordesc<128x64xf8E4M3FN, #shared3>, %arg5: !tt.tensordesc<128x8xi8, #shared2>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<127> : tensor<128x8xi8, #linear>
    %result = ttng.tmem_alloc %cst_0 : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %0 = scf.for %arg6 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg7 = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %1 = arith.muli %arg6, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %2 = tt.descriptor_load %arg3[%arg1, %1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf8E4M3FN, #shared3> -> tensor<128x64xf8E4M3FN, #blocked1>
      %3 = tt.descriptor_load %arg4[%arg2, %1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf8E4M3FN, #shared3> -> tensor<128x64xf8E4M3FN, #blocked1>
      %5 = ttg.local_alloc %2 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>
      %6 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>
      // scales are a register descriptor_load — stays as tt.descriptor_load
      // CHECK: [[REG:%.*]] = tt.descriptor_load
      %4 = tt.descriptor_load %arg5[%arg1, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x8xi8, #shared2> -> tensor<128x8xi8, #linear>
      // CHECK: tmem_alloc [[REG]]
      %result_1 = ttng.tmem_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>
      %result_2, %token = ttng.tmem_alloc %arg7 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %8 = ttng.tc_gen5_mma_scaled %5, %7, %result_2[%token], %result, %result_1, %true, %true lhs = e4m3 rhs = e4m3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      %result_3, %token_4 = ttng.tmem_load %result_2[%8] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %result_3 : tensor<128x128xf32, #blocked>
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], tt.num_stages = 2 : i64, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @local_alloc_default_partition
  tt.func @local_alloc_default_partition(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x128xf16, #shared>, %arg4: !tt.tensordesc<128x128xf16, #shared>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    // Three cross-partition values => three semaphore pairs (6 creates)
    // CHECK: nvws.semaphore.create {{.*}} true
    // CHECK: nvws.semaphore.create {{.*}} false
    // CHECK: nvws.semaphore.create {{.*}} true
    // CHECK: nvws.semaphore.create {{.*}} false
    // CHECK: nvws.semaphore.create {{.*}} true
    // CHECK: nvws.semaphore.create {{.*}} false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c128_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      // Producer for LHS TMA load
      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: nvws.descriptor_load

      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: ttg.local_load
      // CHECK: ttg.local_store

      // CHECK: nvws.semaphore.acquire
      // CHECK: nvws.semaphore.buffer
      // CHECK: ttg.memdesc_trans

      %3 = tt.descriptor_load %arg3[%arg1, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked2>
      %5 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared1, #smem>
      %lhs_trans = ttg.memdesc_trans %5 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared1, #smem> -> !ttg.memdesc<128x128xf16, #shared, #smem>

      %4 = tt.descriptor_load %arg4[%arg2, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf16, #shared> -> tensor<128x128xf16, #blocked1>
      %6 = ttg.local_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared1, #smem>

      // CHECK: ttng.tc_gen5_mma
      %8 = ttng.tc_gen5_mma %lhs_trans, %7, %result[%arg6], %true, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %8 : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!ty = tensor<1xi32, #blocked>
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @two_consumers
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @two_consumers(%lb: i32, %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
      %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[VAL:%.*]] = "op_a"() {ttg.partition = array<i32: 0>}
      // Producer: acquire EMPTY, buffer, store val, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttg.local_store [[VAL]], [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}

      "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> ()
      // Consumer partition 1: acquire FULL, buffer, load, release EMPTY
      // CHECK: [[GTOK1:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF1:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED1:%.*]] = ttg.local_load [[GBUF1]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: "op_b"([[LOADED1]]) {ttg.partition = array<i32: 1>}

      "op_c"(%0) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      // Consumer partition 2: acquire FULL, buffer, load, release EMPTY
      // CHECK: [[GTOK2:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[GBUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK2]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[LOADED2:%.*]] = ttg.local_load [[GBUF2]] {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK: "op_c"([[LOADED2]]) {ttg.partition = array<i32: 2>}
      // CHECK: "op_d"([[LOADED2]]) {ttg.partition = array<i32: 2>}
      "op_d"(%0) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0, 2, 2], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @distance_one
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @distance_one(%lb: i32, %ub: i32, %step: i32) {
    %cst = arith.constant dense<0> : !ty
    scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
      // Producer: acquire EMPTY, buffer, store, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{.*}}, [[PBUF]] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>}
      %0 = "op_a"() {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : () -> !ty
      // Consumer: acquire FULL, buffer, load, release EMPTY
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED:%.*]] = ttg.local_load [[GBUF]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: "op_b"([[LOADED]]) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>}
      "op_b"(%k) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : (!ty) -> ()

      scf.yield {ttg.partition = array<i32: 0, 1>} %0 : !ty
    } {tt.warp_specialize, ttg.partition.stages = [0, 0], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @different_yield_partition
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @different_yield_partition(%lb: i32, %ub: i32, %step: i32) {
    %cst = arith.constant dense<0> : !ty
    scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
      %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[VAL:%.*]] = "op_a"() {ttg.partition = array<i32: 0>}
      // Producer: acquire EMPTY, buffer, store, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store [[VAL]], [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      "op_b"(%k) {ttg.partition = array<i32: 1>} : (!ty) -> ()

      // Consumer: acquire FULL, buffer, load, release EMPTY
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED:%.*]] = ttg.local_load [[GBUF]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}

      scf.yield {ttg.partition = array<i32: 0, 1>} %0 : !ty
    } {tt.warp_specialize, ttg.partition.stages = [0, 0], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // Two cross-partition iter_args => two semaphore pairs (for %k and %l)
  // CHECK: [[ALLOC_K:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY_K:%.*]] = nvws.semaphore.create [[ALLOC_K]] true
  // CHECK-NEXT: [[FULL_K:%.*]] = nvws.semaphore.create [[ALLOC_K]] false
  // CHECK: [[ALLOC_L:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY_L:%.*]] = nvws.semaphore.create [[ALLOC_L]] true
  // CHECK-NEXT: [[FULL_L:%.*]] = nvws.semaphore.create [[ALLOC_L]] false
  tt.func @complex_case(%lb: i32, %ub: i32, %step: i32) {
    %cst = arith.constant dense<0> : !ty
    scf.for %i = %lb to %ub step %step iter_args(%k = %cst, %l = %cst) -> (!ty, !ty) : i32 {
      // Producer put for %l: acquire EMPTY_L, buffer, store, release FULL_L
      // CHECK: [[PTOK_L:%.*]] = nvws.semaphore.acquire [[EMPTY_L]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF_L:%.*]] = nvws.semaphore.buffer [[EMPTY_L]], [[PTOK_L]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{.*}}, [[PBUF_L]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL_L]], [[PTOK_L]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // Producer put for %k: acquire EMPTY_K, buffer, store, release FULL_K
      // CHECK: [[PTOK_K:%.*]] = nvws.semaphore.acquire [[EMPTY_K]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF_K:%.*]] = nvws.semaphore.buffer [[EMPTY_K]], [[PTOK_K]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{.*}}, [[PBUF_K]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL_K]], [[PTOK_K]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}

      %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: "op_a"

      // Consumer for %k in partition 1
      // CHECK: [[GTOK_K1:%.*]] = nvws.semaphore.acquire [[FULL_K]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF_K1:%.*]] = nvws.semaphore.buffer [[FULL_K]], [[GTOK_K1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED_K1:%.*]] = ttg.local_load [[GBUF_K1]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY_K]], [[GTOK_K1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: "op_b"([[LOADED_K1]]) {ttg.partition = array<i32: 1>}
      "op_b"(%k) {ttg.partition = array<i32: 1>} : (!ty) -> ()

      // Consumer for %k in partition 2
      // CHECK: [[GTOK_K2:%.*]] = nvws.semaphore.acquire [[FULL_K]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[GBUF_K2:%.*]] = nvws.semaphore.buffer [[FULL_K]], [[GTOK_K2]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[LOADED_K2:%.*]] = ttg.local_load [[GBUF_K2]] {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[EMPTY_K]], [[GTOK_K2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK: "op_c"([[LOADED_K2]]) {ttg.partition = array<i32: 2>}
      // CHECK: "op_c"([[LOADED_K2]]) {ttg.partition = array<i32: 2>}
      "op_c"(%k) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      "op_c"(%k) {ttg.partition = array<i32: 2>} : (!ty) -> ()

      // Consumer for %l in partition 1
      // CHECK: [[GTOK_L1:%.*]] = nvws.semaphore.acquire [[FULL_L]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF_L1:%.*]] = nvws.semaphore.buffer [[FULL_L]], [[GTOK_L1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED_L1:%.*]] = ttg.local_load [[GBUF_L1]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY_L]], [[GTOK_L1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: "op_d"([[LOADED_L1]]) {ttg.partition = array<i32: 1>}
      "op_d"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()

      // Consumer for %l in partition 2
      // CHECK: [[GTOK_L2:%.*]] = nvws.semaphore.acquire [[FULL_L]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[GBUF_L2:%.*]] = nvws.semaphore.buffer [[FULL_L]], [[GTOK_L2]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[LOADED_L2:%.*]] = ttg.local_load [[GBUF_L2]] {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[EMPTY_L]], [[GTOK_L2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK: "op_d"([[LOADED_L2]]) {ttg.partition = array<i32: 2>}
      "op_d"(%l) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      scf.yield %0, %k : !ty, !ty
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0, 2, 2], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @reuse_argument
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  // CHECK: scf.for
  tt.func @reuse_argument(%lb: i32, %ub: i32, %step: i32) {
    %cst0 = arith.constant dense<0> : !ty
    %cst1 = arith.constant dense<1> : !ty

    scf.for %i = %lb to %ub step %step iter_args(%k = %cst0, %l = %cst1) -> (!ty, !ty) : i32 {
      // Producer: acquire EMPTY, buffer, store %l, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{.*}}, [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK: "op_a"
      %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

      // Consumer partition 1: acquire FULL, buffer, load, release EMPTY
      // CHECK: [[GTOK1:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF1:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED1:%.*]] = ttg.local_load [[GBUF1]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: "op_d"([[LOADED1]]) {ttg.partition = array<i32: 1>}
      "op_d"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()

      // Consumer partition 2: acquire FULL, buffer, load, release EMPTY
      // CHECK: [[GTOK2:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[GBUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK2]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[LOADED2:%.*]] = ttg.local_load [[GBUF2]] {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK: "op_d"([[LOADED2]]) {ttg.partition = array<i32: 2>}
      "op_d"(%l) {ttg.partition = array<i32: 2>} : (!ty) -> ()
      scf.yield %0, %k : !ty, !ty
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.partition.stages = [1, 0, 0], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @multiplicity_branch
  // Three cross-partition iter_args => three semaphore pairs
  // CHECK: [[ALLOC_A:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY_A:%.*]] = nvws.semaphore.create [[ALLOC_A]] true
  // CHECK-NEXT: [[FULL_A:%.*]] = nvws.semaphore.create [[ALLOC_A]] false
  // CHECK: [[ALLOC_B:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY_B:%.*]] = nvws.semaphore.create [[ALLOC_B]] true
  // CHECK-NEXT: [[FULL_B:%.*]] = nvws.semaphore.create [[ALLOC_B]] false
  // CHECK: [[ALLOC_C:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY_C:%.*]] = nvws.semaphore.create [[ALLOC_C]] true
  // CHECK-NEXT: [[FULL_C:%.*]] = nvws.semaphore.create [[ALLOC_C]] false
  tt.func @multiplicity_branch(%lb: i32, %ub: i32, %step: i32) {
    %cst0 = arith.constant dense<0> : !ty
    %cst1 = arith.constant dense<1> : !ty
    %cst2 = arith.constant dense<2> : !ty

    scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
      // Producer puts for %c, %b, %a — all in partition 0
      // CHECK: [[PTOK_C:%.*]] = nvws.semaphore.acquire [[EMPTY_C]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL_C]], [[PTOK_C]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK: [[PTOK_B:%.*]] = nvws.semaphore.acquire [[EMPTY_B]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL_B]], [[PTOK_B]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK: [[PTOK_A:%.*]] = nvws.semaphore.acquire [[EMPTY_A]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL_A]], [[PTOK_A]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK: "op_a"
      %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

      // Consumer for %a in partition 1
      // CHECK: [[GTOK_A:%.*]] = nvws.semaphore.acquire [[FULL_A]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF_A:%.*]] = nvws.semaphore.buffer [[FULL_A]], [[GTOK_A]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED_A:%.*]] = ttg.local_load [[GBUF_A]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY_A]], [[GTOK_A]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: "op_b"([[LOADED_A]]) {ttg.partition = array<i32: 1>}
      "op_b"(%a) {ttg.partition = array<i32: 1>}: (!ty) -> ()

      // Consumer for %b in partition 2
      // CHECK: [[GTOK_B:%.*]] = nvws.semaphore.acquire [[FULL_B]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[GBUF_B:%.*]] = nvws.semaphore.buffer [[FULL_B]], [[GTOK_B]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[LOADED_B:%.*]] = ttg.local_load [[GBUF_B]] {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[EMPTY_B]], [[GTOK_B]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK: "op_c"([[LOADED_B]]) {ttg.partition = array<i32: 2>}
      "op_c"(%b) {ttg.partition = array<i32: 2>}: (!ty) -> ()

      // Consumer for %c in partition 3
      // CHECK: [[GTOK_C:%.*]] = nvws.semaphore.acquire [[FULL_C]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: [[GBUF_C:%.*]] = nvws.semaphore.buffer [[FULL_C]], [[GTOK_C]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: [[LOADED_C:%.*]] = ttg.local_load [[GBUF_C]] {ttg.partition = array<i32: 3>}
      // CHECK: nvws.semaphore.release [[EMPTY_C]], [[GTOK_C]] [#nvws.async_op<none>] {ttg.partition = array<i32: 3>}
      // CHECK: "op_d"([[LOADED_C]]) {ttg.partition = array<i32: 3>}
      "op_d"(%c) {ttg.partition = array<i32: 3>}: (!ty) -> ()

      scf.yield %0, %a, %a : !ty, !ty, !ty
    } {tt.warp_specialize, ttg.partition.stages = [0, 0, 0, 0], ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @multiplicity_branch2
  // CHECK: [[ALLOC_A:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY_A:%.*]] = nvws.semaphore.create [[ALLOC_A]] true
  // CHECK-NEXT: [[FULL_A:%.*]] = nvws.semaphore.create [[ALLOC_A]] false
  // CHECK: [[ALLOC_B:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY_B:%.*]] = nvws.semaphore.create [[ALLOC_B]] true
  // CHECK-NEXT: [[FULL_B:%.*]] = nvws.semaphore.create [[ALLOC_B]] false
  // CHECK: [[ALLOC_C:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY_C:%.*]] = nvws.semaphore.create [[ALLOC_C]] true
  // CHECK-NEXT: [[FULL_C:%.*]] = nvws.semaphore.create [[ALLOC_C]] false
  tt.func @multiplicity_branch2(%lb: i32, %ub: i32, %step: i32) {
    %cst0 = arith.constant dense<0> : !ty
    %cst1 = arith.constant dense<1> : !ty
    %cst2 = arith.constant dense<2> : !ty

    scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
      // Producer puts: %c from partition 2, %b from partition 1, %a from partition 0
      // CHECK: [[PTOK_C:%.*]] = nvws.semaphore.acquire [[EMPTY_C]] {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[FULL_C]], [[PTOK_C]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK: [[PTOK_B:%.*]] = nvws.semaphore.acquire [[EMPTY_B]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[FULL_B]], [[PTOK_B]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: [[PTOK_A:%.*]] = nvws.semaphore.acquire [[EMPTY_A]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL_A]], [[PTOK_A]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK: "op_a"
      %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

      // Consumer for %a in partition 1
      // CHECK: [[GTOK_A:%.*]] = nvws.semaphore.acquire [[FULL_A]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF_A:%.*]] = nvws.semaphore.buffer [[FULL_A]], [[GTOK_A]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED_A:%.*]] = ttg.local_load [[GBUF_A]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY_A]], [[GTOK_A]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: "op_b"([[LOADED_A]]) {ttg.partition = array<i32: 1>}
      %d = "op_b"(%a) {ttg.partition = array<i32: 1>}: (!ty) -> !ty

      // Consumer for %b in partition 2
      // CHECK: [[GTOK_B:%.*]] = nvws.semaphore.acquire [[FULL_B]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[GBUF_B:%.*]] = nvws.semaphore.buffer [[FULL_B]], [[GTOK_B]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[LOADED_B:%.*]] = ttg.local_load [[GBUF_B]] {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[EMPTY_B]], [[GTOK_B]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK: "op_c"([[LOADED_B]]) {ttg.partition = array<i32: 2>}
      %e = "op_c"(%b) {ttg.partition = array<i32: 2>}: (!ty) -> !ty

      // Consumer for %c in partition 3
      // CHECK: [[GTOK_C:%.*]] = nvws.semaphore.acquire [[FULL_C]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: [[GBUF_C:%.*]] = nvws.semaphore.buffer [[FULL_C]], [[GTOK_C]] {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: [[LOADED_C:%.*]] = ttg.local_load [[GBUF_C]] {ttg.partition = array<i32: 3>}
      // CHECK: nvws.semaphore.release [[EMPTY_C]], [[GTOK_C]] [#nvws.async_op<none>] {ttg.partition = array<i32: 3>}
      // CHECK: "op_d"([[LOADED_C]]) {ttg.partition = array<i32: 3>}
      "op_d"(%c) {ttg.partition = array<i32: 3>}: (!ty) -> ()

      scf.yield %0, %d, %e : !ty, !ty, !ty
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>, array<i32: 2>], ttg.partition.stages = [0, 0, 0, 0], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @self_recursion
  tt.func @self_recursion(%lb: i32, %ub: i32, %step: i32) {
    // CHECK-NOT: nvws.semaphore.create
    %cst = arith.constant dense<0> : !ty
    %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
      %0 = "op_a"(%k) {ttg.partition = array<i32: 0>} : (!ty) -> !ty
      scf.yield %0 : !ty
    } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @self_recursion_and_use
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @self_recursion_and_use(%lb: i32, %ub: i32, %step: i32) {
    %cst = arith.constant dense<0> : !ty
    %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
      %0 = "op_a"(%k) {ttg.partition = array<i32: 0>} : (!ty) -> !ty
      // CHECK: [[VAL:%.*]] = "op_a"
      // Producer: acquire EMPTY, buffer, store, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store [[VAL]], [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}

      "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
      // Consumer: acquire FULL, buffer, load, release EMPTY
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED:%.*]] = ttg.local_load [[GBUF]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: "op_b"([[LOADED]])

      scf.yield %0 : !ty
    } {tt.warp_specialize, ttg.partition.stages = [0, 1], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @conditional_consumer
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @conditional_consumer(%lb: i32, %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step : i32 {
      %0 = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[VAL:%.*]] = "producer"() {ttg.partition = array<i32: 0>}
      // Producer: acquire EMPTY, buffer, store, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store [[VAL]], [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %cond = "rand"() {ttg.partition = array<i32: 1>} : () -> i1
      // CHECK: "rand"
      // Consumer: acquire FULL, buffer, load, release EMPTY (before if)
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED:%.*]] = ttg.local_load [[GBUF]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: scf.if
      %1 = scf.if %cond -> !ty {
        "something"() {ttg.partition = array<i32: 1>} : () -> ()
        scf.yield {ttg.partition = array<i32: 1>} %0 : !ty
      } else {
        %2 = "something"() {ttg.partition = array<i32: 1>} : () -> !ty
        scf.yield {ttg.partition = array<i32: 1>} %2 : !ty
      } {ttg.partition = array<i32: 1>, ttg.partition.outputs = [array<i32: 1>]}
      "keep"(%1) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0, 2], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @no_def_op
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  // CHECK: scf.for
  tt.func @no_def_op(%lb: i32, %ub: i32, %step: i32) {
    %c0_i32 = arith.constant 0 : i32
    scf.for %i = %lb to %ub step %step iter_args(%k = %c0_i32) -> i32 : i32 {
      // Producer: acquire EMPTY, buffer, splat, store, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: tt.splat
      // CHECK: ttg.local_store {{.*}}, [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // Consumer: acquire FULL, buffer, load, release EMPTY, unsplat
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED:%.*]] = ttg.local_load [[GBUF]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: tt.unsplat [[LOADED]]
      // CHECK: arith.addi
      arith.addi %k, %k {ttg.partition = array<i32: 1>} : i32
      scf.yield {ttg.partition = array<i32: 0>} %k : i32
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
    tt.return
  }

  // CHECK-LABEL: @scalar_consumers
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @scalar_consumers(%lb: i32, %ub: i32, %step: i32) {
    scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
      %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> i32
      // CHECK: [[SCALAR:%.*]] = "op_a"() {ttg.partition = array<i32: 0>}
      // Producer: acquire EMPTY, buffer, splat scalar, store, release FULL
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: tt.splat [[SCALAR]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store {{.*}}, [[PBUF]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}

      "op_b"(%0) {ttg.partition = array<i32: 1>} : (i32) -> ()
      // Consumer: acquire FULL, buffer, load, release EMPTY, unsplat
      // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF:%.*]] = nvws.semaphore.buffer [[FULL]], [[GTOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED:%.*]] = ttg.local_load [[GBUF]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: [[UNSP:%.*]] = tt.unsplat [[LOADED]] {ttg.partition = array<i32: 1>}
      // CHECK: "op_b"([[UNSP]]) {ttg.partition = array<i32: 1>}

    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0, 2], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @cycle_in_partition(%lb: i32, %ub: i32, %step: i32) {
    // Two cross-partition values => two semaphore pairs
    // CHECK: [[ALLOC1:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY1:%.*]] = nvws.semaphore.create [[ALLOC1]] true
    // CHECK-NEXT: [[FULL1:%.*]] = nvws.semaphore.create [[ALLOC1]] false
    // CHECK: [[ALLOC2:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY2:%.*]] = nvws.semaphore.create [[ALLOC2]] true
    // CHECK-NEXT: [[FULL2:%.*]] = nvws.semaphore.create [[ALLOC2]] false

    scf.for %i = %lb to %ub step %step : i32 {
      %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: [[VAL_A:%.*]] = "op_a"() {ttg.partition = array<i32: 0>}
      // Producer for %0→partition1: acquire EMPTY1, buffer, store, release FULL1
      // CHECK: [[PTOK1:%.*]] = nvws.semaphore.acquire [[EMPTY1]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[PBUF1:%.*]] = nvws.semaphore.buffer [[EMPTY1]], [[PTOK1]] {ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store [[VAL_A]], [[PBUF1]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL1]], [[PTOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}

      %1 = "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
      // Consumer: acquire FULL1, buffer, load, release EMPTY1
      // CHECK: [[GTOK1:%.*]] = nvws.semaphore.acquire [[FULL1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[GBUF1:%.*]] = nvws.semaphore.buffer [[FULL1]], [[GTOK1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LOADED1:%.*]] = ttg.local_load [[GBUF1]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY1]], [[GTOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: [[VAL_B:%.*]] = "op_b"([[LOADED1]]) {ttg.partition = array<i32: 1>}
      // Producer for %1→partition0: acquire EMPTY2, buffer, store, release FULL2
      // CHECK: [[PTOK2:%.*]] = nvws.semaphore.acquire [[EMPTY2]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[PBUF2:%.*]] = nvws.semaphore.buffer [[EMPTY2]], [[PTOK2]] {ttg.partition = array<i32: 1>}
      // CHECK: ttg.local_store [[VAL_B]], [[PBUF2]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[FULL2]], [[PTOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}

      // Consumer: acquire FULL2, buffer, load, release EMPTY2
      // CHECK: [[GTOK2:%.*]] = nvws.semaphore.acquire [[FULL2]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[GBUF2:%.*]] = nvws.semaphore.buffer [[FULL2]], [[GTOK2]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[LOADED2:%.*]] = ttg.local_load [[GBUF2]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[EMPTY2]], [[GTOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK: "op_c"([[LOADED2]]) {ttg.partition = array<i32: 0>}
      "op_c"(%1) {ttg.partition = array<i32: 0>} : (!ty) -> ()
      scf.yield
    } {tt.warp_specialize, ttg.partition.stages = [0, 2], ttg.partition = array<i32: 0, 1>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @cycle_in_partition(%lb: i32, %ub: i32, %step: i32) {
    // Three cross-partition values => three semaphore pairs
    // CHECK: [[ALLOC1:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY1:%.*]] = nvws.semaphore.create [[ALLOC1]] true
    // CHECK-NEXT: [[FULL1:%.*]] = nvws.semaphore.create [[ALLOC1]] false
    // CHECK: [[ALLOC2:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY2:%.*]] = nvws.semaphore.create [[ALLOC2]] true
    // CHECK-NEXT: [[FULL2:%.*]] = nvws.semaphore.create [[ALLOC2]] false
    // CHECK: [[ALLOC3:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY3:%.*]] = nvws.semaphore.create [[ALLOC3]] true
    // CHECK-NEXT: [[FULL3:%.*]] = nvws.semaphore.create [[ALLOC3]] false
    scf.for %j = %lb to %ub step %step : i32 {
      %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
      // CHECK: "op_a"
      // CHECK: nvws.semaphore.acquire [[EMPTY1]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[FULL1]]{{.*}} {ttg.partition = array<i32: 0>}

      %1 = "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
      // CHECK: nvws.semaphore.acquire [[FULL1]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[EMPTY1]]{{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      // CHECK: "op_b"
      // CHECK: nvws.semaphore.acquire [[EMPTY2]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[FULL2]]{{.*}} {ttg.partition = array<i32: 1>}

      %2 = "op_c"(%1) {ttg.partition = array<i32: 2>} : (!ty) -> !ty
      // CHECK: nvws.semaphore.acquire [[FULL2]] {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[EMPTY2]]{{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      // CHECK: "op_c"
      // CHECK: nvws.semaphore.acquire [[EMPTY3]] {ttg.partition = array<i32: 2>}
      // CHECK: nvws.semaphore.release [[FULL3]]{{.*}} {ttg.partition = array<i32: 2>}

      "op_c"(%2) {ttg.partition = array<i32: 0>} : (!ty) -> ()
      // CHECK: nvws.semaphore.acquire [[FULL3]] {ttg.partition = array<i32: 0>}
      // CHECK: nvws.semaphore.release [[EMPTY3]]{{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK: "op_c"
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0, 2, 3], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// CHECK-LABEL: @inner_loop_fixed_operand
// Two cross-partition values (outer LHS + inner RHS)
// CHECK-COUNT-4: nvws.semaphore.create
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @inner_loop_fixed_operand(%arg0: !tt.tensordesc<128x128xf8E4M3FN, #shared>, %arg1: !tt.tensordesc<128x128xf8E4M3FN, #shared>, %arg2: !tt.tensordesc<128x128xf8E4M3FN, #shared>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c148_i32 = arith.constant 148 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c8_i32 = arith.constant 8 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %arg3, %c128_i32 : i32
    %2 = arith.divsi %arg4, %c128_i32 : i32
    %3 = arith.divsi %arg5, %c128_i32 : i32
    %4 = arith.muli %1, %2 : i32
    %5 = arith.muli %2, %c8_i32 : i32
    %result, %token = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: scf.for
    // Producer for outer LHS TMA load
    // CHECK: nvws.semaphore.acquire {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: nvws.semaphore.buffer {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: nvws.descriptor_load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // Consumer for outer LHS
    // CHECK: nvws.semaphore.acquire {{.*}} {ttg.partition = array<i32: 1>}
    // CHECK: nvws.semaphore.buffer {{.*}} {ttg.partition = array<i32: 1>}
    // CHECK: scf.for
    // Producer for inner RHS TMA load
    // CHECK: nvws.semaphore.acquire {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: nvws.semaphore.buffer {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: nvws.descriptor_load {{.*}} {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
    // Consumer for inner RHS
    // CHECK: nvws.semaphore.acquire {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    // CHECK: nvws.semaphore.buffer {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    // CHECK: ttg.memdesc_trans {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>}
    // CHECK: ttng.tc_gen5_mma {{.*}} {loop.cluster = 0 : i32, loop.stage = 2 : i32{{.*}}ttg.partition = array<i32: 1>}
    // CHECK: nvws.semaphore.release {{.*}} [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
    %6 = scf.for %arg6 = %0 to %4 step %c148_i32 iter_args(%arg7 = %token) -> (!ttg.async.token)  : i32 {
      %7 = arith.divsi %arg6, %5 {ttg.partition = array<i32: 0, 2>} : i32
      %8 = arith.muli %7, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %9 = arith.subi %1, %8 {ttg.partition = array<i32: 0, 2>} : i32
      %10 = arith.minsi %9, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %11 = arith.remsi %arg6, %10 {ttg.partition = array<i32: 0, 2>} : i32
      %12 = arith.addi %8, %11 {ttg.partition = array<i32: 0, 2>} : i32
      %13 = arith.remsi %arg6, %5 {ttg.partition = array<i32: 0, 2>} : i32
      %14 = arith.divsi %13, %10 {ttg.partition = array<i32: 0, 2>} : i32
      %15 = arith.muli %12, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %16 = arith.muli %14, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %17 = tt.descriptor_load %arg0[%15, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf8E4M3FN, #shared> -> tensor<128x128xf8E4M3FN, #blocked1>
      %18 = ttg.local_alloc %17 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %19:2 = scf.for %arg8 = %c0_i32 to %3 step %c1_i32 iter_args(%arg9 = %false, %arg10 = %arg7) -> (i1, !ttg.async.token)  : i32 {
        %22 = arith.muli %arg8, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        %23 = tt.descriptor_load %arg1[%16, %22] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<128x128xf8E4M3FN, #shared> -> tensor<128x128xf8E4M3FN, #blocked1>
        %24 = ttg.local_alloc %23 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %25 = ttg.memdesc_trans %24 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
        %26 = ttng.tc_gen5_mma %18, %25, %result[%arg10], %arg9, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %26 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1>]}
      %result_0, %token_1 = ttng.tmem_load %result[%19#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %20 = tt.fp_to_fp %result_0 {ttg.partition = array<i32: 0>}, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %21 = ttg.convert_layout %20 {ttg.partition = array<i32: 0>} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
      tt.descriptor_store %arg2[%15, %16], %21 {ttg.partition = array<i32: 0>} : !tt.tensordesc<128x128xf8E4M3FN, #shared>, tensor<128x128xf8E4M3FN, #blocked1>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %token_1 : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>
module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @semaphore_result_outside_scheduled_loop
  // CHECK: [[ALLOC:%.*]] = ttg.local_alloc
  // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ALLOC]] true
  // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ALLOC]] false
  tt.func @semaphore_result_outside_scheduled_loop(%lb: i32, %ub: i32, %step: i32) {
    // Producer: acquire EMPTY, buffer, store, release FULL
    // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 2>}
    // CHECK: nvws.semaphore.release [[FULL]], [[PTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
    // Consumer: acquire FULL, buffer, load, release EMPTY
    // CHECK: [[GTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
    // CHECK: nvws.semaphore.release [[EMPTY]], [[GTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
    scf.for %i = %lb to %ub step %step : i32 {
      %0 = "op_a"() {ttg.partition = array<i32: 2>} : () -> !ty
      "op_b"(%0) {ttg.partition = array<i32: 0>} : (!ty) -> ()
      scf.for %j = %lb to %ub step %step : i32 {
        %x = arith.addi %lb, %lb {loop.cluster = 0 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : i32
        scf.yield
      } {tt.scheduled_max_stage = 0 : i32, ttg.partition = array<i32: 0>}
      scf.yield
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 2>, ttg.partition.stages = [0, 1], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
