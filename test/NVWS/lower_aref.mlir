// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-aref -canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {

  // CHECK-LABEL: @two_consumers
  tt.func @two_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    // CHECK: [[BUF:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.init_barrier [[EMPTYSLICE]], 2
    // CHECK-NEXT: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.init_barrier [[EMPTYSLICE]], 2
    // CHECK-NEXT: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.init_barrier [[EMPTYSLICE]], 2
    // CHECK-NEXT: [[FULL:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.init_barrier [[FULLSLICE]], 1
    // CHECK-NEXT: [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.init_barrier [[FULLSLICE]], 1
    // CHECK-NEXT: [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.init_barrier [[FULLSLICE]], 1
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    scf.for %arg3 = %arg0 to %arg1 step %arg2 : i32 {
      %3 = "op_a"() {ttg.partition = array<i32: 0>} : () -> tensor<1xi32, #blocked>
      // CHECK: op_a
      // CHECK-NEXT: addi
      // CHECK-NEXT: cmpi
      // CHECK-NEXT: [[STAGE:%.*]] = arith.select
      // CHECK-NEXT: xori
      // CHECK-NEXT: [[PHASE:%.*]] = arith.select
      // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]][[[STAGE]]]
      // CHECK-NEXT: ttng.wait_barrier [[EMPTYMBAR]], [[PHASE]] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      // CHECK: local_store
      // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]][[[STAGE]]]
      // CHECK-NEXT: ttng.arrive_barrier [[FULLMBAR]], 1 {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>}
      %buffers, %token = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %3, %buffers {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%c0_i32], %token [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      // CHECK: addi
      // CHECK-NEXT: cmpi
      // CHECK-NEXT: [[STAGE:%.*]] = arith.select
      // CHECK-NEXT: xori
      // CHECK-NEXT: [[PHASE:%.*]] = arith.select
      // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]][[[STAGE]]]
      // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]], [[PHASE]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>}
      // CHECK: local_load
      // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]][[[STAGE]]]
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR]], 1 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>}
      %buffers_0, %token_1 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %14 = ttg.local_load %buffers_0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_1 [#nvws.async_op<none>] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%14) {ttg.partition = array<i32: 1>} : (tensor<1xi32, #blocked>) -> ()
      // CHECK: addi
      // CHECK-NEXT: cmpi
      // CHECK-NEXT: [[STAGE:%.*]] = arith.select
      // CHECK-NEXT: xori
      // CHECK-NEXT: [[PHASE:%.*]] = arith.select
      // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]][[[STAGE]]]
      // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]], [[PHASE]] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>}
      // CHECK: local_load
      // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]][[[STAGE]]]
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR]], 1 {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>}
      %buffers_2, %token_3 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %20 = ttg.local_load %buffers_2 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_3 [#nvws.async_op<none>] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%20) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
      "op_d"(%20) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {ttg.partition.stages =
    // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT: ttg.local_dealloc
    // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.inval_barrier [[FULLMBAR]]
    // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.inval_barrier [[FULLMBAR]]
    // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.inval_barrier [[FULLMBAR]]
    // CHECK-NEXT: ttg.local_dealloc
    ttg.local_dealloc %0 : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }

  //CHECK-LABEL: @three_consumers
  tt.func @three_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    // CHECK: [[BUF:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.init_barrier [[EMPTYSLICE]], 3
    // CHECK: [[FULL:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.init_barrier [[FULLSLICE]], 1
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    scf.for %arg3 = %arg0 to %arg1 step %arg2 : i32 {
      %3 = "op_a"() {ttg.partition = array<i32: 0>} : () -> tensor<1xi32, #blocked>
      %buffers, %token = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %3, %buffers {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%c0_i32], %token [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %buffers_0, %token_1 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %14 = ttg.local_load %buffers_0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_1 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%14) {ttg.partition = array<i32: 1>} : (tensor<1xi32, #blocked>) -> ()
      %buffers_2, %token_3 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %20 = ttg.local_load %buffers_2 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_3 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%20) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
      "op_d"(%20) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
      %buffers_4, %token_5 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %26 = ttg.local_load %buffers_4 {ttg.partition = array<i32: 3>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_5 [#nvws.async_op<none>] {ttg.partition = array<i32: 3>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_e"(%26) {ttg.partition = array<i32: 3>} : (tensor<1xi32, #blocked>) -> ()
      "op_f"(%26) {ttg.partition = array<i32: 3>} : (tensor<1xi32, #blocked>) -> ()
    } {ttg.partition.stages = [0 : i32, 2 : i32, 2 : i32, 3 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: } {ttg.partition.stages =
    // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT: ttg.local_dealloc
    // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.inval_barrier [[FULLMBAR]]
    // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.inval_barrier [[FULLMBAR]]
    // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.inval_barrier [[FULLMBAR]]
    // CHECK-NEXT: ttg.local_dealloc
    ttg.local_dealloc %0 : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    tt.return
  }


  //CHECK-LABEL: @reuse_argument
  tt.func @reuse_argument(%arg0: i32, %arg1: i32, %arg2: i32) {
    %true = arith.constant true
    %cst = arith.constant dense<1> : tensor<1xi32, #blocked>
    %cst_0 = arith.constant dense<0> : tensor<1xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: ttg.local_alloc
    // CHECK: [[EMPTY1:%.*]] = ttg.local_alloc
    // CHECK: [[FULL1:%.*]] = ttg.local_alloc
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>
    // CHECK: scf.for
    scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg5 = %cst) -> (tensor<1xi32, #blocked>)  : i32 {
      // CHECK: arith.select
      // CHECK: [[PHASE:%.*]] = arith.select
      // CHECK: [[EMPTYBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
      // CHECK: ttng.wait_barrier [[EMPTYBAR1]], [[PHASE]]
      // CHECK: local_store
      // CHECK-NEXT: [[FULLBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
      // CHECK-NEXT: ttng.arrive_barrier [[FULLBAR1]], 1
      // CHECK: op_a
      %buffers, %token = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %arg5, %buffers {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%c0_i32], %token [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %5 = "op_a"() {ttg.partition = array<i32: 0>} : () -> tensor<1xi32, #blocked>

      // CHECK: arith.select
      // CHECK: [[PHASE:%.*]] = arith.select
      // CHECK: [[FULLMBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
      // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR1]], [[PHASE]]
      // CHECK: local_load
      // CHECK-NEXT: [[EMPTYMBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR1]], 1
      // CHECK: op_d
      %buffers_1, %token_2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %8 = ttg.local_load %buffers_1 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_d"(%8) {ttg.partition = array<i32: 1>} : (tensor<1xi32, #blocked>) -> ()

      // CHECK: arith.select
      // CHECK: [[PHASE:%.*]] = arith.select
      // CHECK: [[FULLMBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
      // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR1]], [[PHASE]]
      // CHECK: local_load
      // CHECK-NEXT: [[EMPTYMBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR1]], 1
      // CHECK: op_d
      %buffers_3, %token_4 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %11 = ttg.local_load %buffers_3 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_4 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_d"(%11) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
      scf.yield %5 : tensor<1xi32, #blocked>
    } {ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %0 = ub.poison : !ttg.async.token
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %1 = ttg.memdesc_index %result[%c0_i32] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %2 = ttng.tmem_store %cst, %1[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    // CHECK: [[BUF_A:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[BUF_B:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable>
    // CHECK: [[TMA_EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64, #shared1, #smem, mutable>
    // CHECK: [[TMA_FULL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64, #shared1, #smem, mutable>
    %3 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %4 = nvws.aref.create %3 : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %5 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %6 = nvws.aref.create %5 : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    %7 = arith.subi %arg0, %c1_i32 : i32
    %8 = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64, #shared1, #smem, mutable>
    %9 = ttg.memdesc_index %8[%c0_i32] : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 1x1>
    ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 1x1>
    %10 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %2) -> (!ttg.async.token)  : i32 {
      %11 = arith.muli %arg5, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      // CHECK-COUNT-1: ttng.wait_barrier {{.*}}, {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK: [[BUF_A_SLICE:%.*]] = ttg.memdesc_index [[BUF_A]]
      // CHECK: [[BUF_B_SLICE:%.*]] = ttg.memdesc_index [[BUF_B]]
      // CHECK: [[TMA_FULL_SLICE:%.*]] = ttg.memdesc_index [[TMA_FULL]]
      // CHECK: ttng.async_tma_copy_global_to_local {{.*}} [[BUF_A_SLICE]], [[TMA_FULL_SLICE]], {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK: ttng.async_tma_copy_global_to_local {{.*}} [[BUF_B_SLICE]], [[TMA_FULL_SLICE]], {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      %buffers, %token_2 = nvws.aref.put.enter %4[%c0_i32, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>, !ttg.async.token
      nvws.descriptor_load %arg3[%arg1, %11] 16384 %buffers {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
      nvws.aref.put.exit %4[%c0_i32], %token_2 [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %buffers_3, %token_4 = nvws.aref.get.enter %4[%c0_i32, %c0_i32] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.async.token
      %buffers_5, %token_6 = nvws.aref.put.enter %6[%c0_i32, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>, !ttg.async.token
      nvws.descriptor_load %arg4[%arg2, %11] 16384 %buffers_5 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
      nvws.aref.put.exit %6[%c0_i32], %token_6 [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %buffers_7, %token_8 = nvws.aref.get.enter %6[%c0_i32, %c0_i32] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.async.token

      // CHECK-COUNT-1: ttng.wait_barrier {{.*}}, {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[BUF_A_SLICE:%.*]] = ttg.memdesc_index [[BUF_A]]
      // CHECK: [[BUF_B_SLICE:%.*]] = ttg.memdesc_index [[BUF_B]]
      // CHECK: [[BUF_B_SLICE_TRANS:%.*]] = ttg.memdesc_trans [[BUF_B_SLICE]] {loop.cluster = 0 : i32, loop.stage = 1 : i32
      %12 = ttg.memdesc_trans %buffers_7 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64> -> !ttg.memdesc<64x128xf16, #shared2, #smem, 1x64x128>
      %13 = arith.cmpi eq, %arg5, %7 : i32
      // CHECK: ttng.tc_gen5_mma [[BUF_A_SLICE]], [[BUF_B_SLICE_TRANS]]
      %14 = ttng.tc_gen5_mma %buffers_3, %12, %1[], %true, %true, %9[%13] {is_async, loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.memdesc<64x128xf16, #shared2, #smem, 1x64x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 1x1>
      // CHECK: [[TMA_EMPTY_SLICE:%.*]] = ttg.memdesc_index [[TMA_EMPTY]]
      // CHECK-COUNT-1: ttng.tc_gen5_commit [[TMA_EMPTY_SLICE]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      nvws.aref.get.exit %6[%c0_i32], %token_8 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.aref.get.exit %4[%c0_i32], %token_4 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      scf.yield %0 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @load_used_as_reg_and_smem(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: [[EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.init_barrier [[EMPTYSLICE]], 2
    // CHECK: [[FULL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.init_barrier [[FULLSLICE]], 1
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      %buffers, %token = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>, !ttg.async.token
      nvws.descriptor_load %arg0[%arg2, %arg2] 16384 %buffers {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
      nvws.aref.put.exit %1[%c0_i32], %token [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %buffers_0, %token_1 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.async.token
      %2 = ttg.local_load %buffers_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64> -> tensor<128x64xf16, #blocked>
      // CHECK: ttng.fence_async_shared {bCluster = false, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
      // CHECK: ttng.arrive_barrier [[EMPTYSLICE]], 1 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      nvws.aref.get.exit %1[%c0_i32], %token_1 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %buffers_2, %token_3 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.async.token
      "use1"(%2) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> ()
      // CHECK: "use2"
      // CHECK: ttng.fence_async_shared {bCluster = false, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
      // CHECK: ttng.arrive_barrier [[EMPTYSLICE]], 1 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      "use2"(%buffers_2) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>) -> ()
      nvws.aref.get.exit %1[%c0_i32], %token_3 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @load_used_as_reg_and_smem_same_partition(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: [[EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK: ttng.init_barrier [[EMPTYSLICE]], 1
    // CHECK: [[FULL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK: ttng.init_barrier [[FULLSLICE]], 1
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      %buffers, %token = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>, !ttg.async.token
      nvws.descriptor_load %arg0[%arg2, %arg2] 16384 %buffers {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
      nvws.aref.put.exit %1[%c0_i32], %token [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      %buffers_0, %token_1 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.async.token
      %2 = ttg.local_load %buffers_0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64> -> tensor<128x64xf16, #blocked>
       // CHECK: ttng.wait_barrier {{.*}}, {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
       // CHECK: "use1"
       // CHECK: "use2"
       // CHECK: ttng.fence_async_shared {bCluster = false, loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
       // CHECK: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
       // CHECK: ttng.arrive_barrier [[EMPTYSLICE]], 1 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      "use1"(%2) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked>) -> ()
      "use2"(%buffers_0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>) -> ()
      nvws.aref.get.exit %1[%c0_i32], %token_1 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @lower_aref_buffer
  tt.func @lower_aref_buffer(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // CHECK: [[BUF:%.*]] = ttng.tmem_alloc
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = nvws.aref.create %result : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %buffers, %token = nvws.aref.put.enter %0 : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
    %1 = nvws.aref.buffer %0, %token : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %2 = ttng.tmem_store %cst_0, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: scf.for {{.*}} iter_args([[SPUT:%.*]] = {{.*}}, {{.*}} = {{.*}}, {{.*}} = {{.*}}, {{.*}} = {{.*}})
    %3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token) -> (!ttg.async.token)  : i32 {
      %4:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: local_alloc
      // CHECK-NEXT: local_alloc
      // CHECK-NEXT: [[VIEW:%.*]] = ttg.memdesc_index [[BUF]][[[SPUT]]]
      // CHECK-NEXT: tc_gen5_mma {{.*}}, {{.*}}, [[VIEW]][]
      %9 = nvws.aref.buffer %0, %arg3 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %11 = arith.cmpi eq, %arg2, %c0_i32 : i32
      // CHECK: [[RET_IF:%.*]]:4 = scf.if
      %12 = scf.if %11 -> (!ttg.async.token) {
        // CHECK: tc_gen5_commit
        // CHECK: ttg.memdesc_index {{.*}}[[[SGET:%.*]]]
        // CHECK-NEXT: ttng.wait_barrier
        // CHECK-NEXT: [[VIEW:%.*]] = ttg.memdesc_index [[BUF]][[[SGET]]]
        // CHECK-NEXT: tmem_load [[VIEW]]
        // CHECK-NEXT: ttg.memdesc_index
        // CHECK-NEXT: ttng.arrive_barrier
        nvws.aref.put.exit %0, %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        %buffers_1, %token_2 = nvws.aref.get.enter %0 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
        %15 = nvws.aref.buffer %0, %token_2 {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
        %result_3, %token_4 = ttng.tmem_load %15[] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
        nvws.aref.get.exit %0, %token_2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
        "acc_user"(%result_3) : (tensor<128x128xf32, #blocked>) -> ()
        %buffers_5, %token_6 = nvws.aref.put.enter %0 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
        // CHECK: ttg.memdesc_index {{.*}}[[[SPUT1:%.*]]]
        // CHECK-NEXT: ttng.wait_barrier
        // CHECK-NEXT: scf.yield [[SPUT1]]
        scf.yield %token_6 : !ttg.async.token
      } else {
        // CHECK: scf.yield
        scf.yield %arg3 : !ttg.async.token
      } {ttg.partition = array<i32: 0>}
      // CHECK: [[VIEW:%.*]] = ttg.memdesc_index [[BUF]][[[RET_IF]]#0]
      // CHECK-NEXT: tmem_store {{.*}}, [[VIEW]][]
      %13 = nvws.aref.buffer %0, %12 {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %14 = ttng.tmem_store %cst, %13[], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      scf.yield %12 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32}
    nvws.aref.put.exit %0, %3 [#nvws.async_op<none>] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }


  // CHECK-LABEL: @aref_not_in_loop
  tt.func @aref_not_in_loop(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: ttng.tmem_alloc
    // CHECK-NEXT: local_alloc
    // CHECK-NEXT: memdesc_index
    // CHECK-NEXT: init_barrier {{.*}}, 1
    // CHECK-NEXT: local_alloc
    // CHECK-NEXT: memdesc_index
    // CHECK-NEXT: init_barrier {{.*}}, 1
    %0 = nvws.aref.create %result : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    %buffers, %token = nvws.aref.put.enter %0 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
    %1 = nvws.aref.buffer %0, %token : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %2 = ttng.tmem_store %cst, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32  : i32 {
      %4 = arith.muli %arg5, %c64_i32 : i32
      %5 = tt.descriptor_load %arg3[%arg1, %4] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg4[%arg2, %4] {ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %9 = ttg.memdesc_trans %8 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      %10 = nvws.aref.buffer %0, %token {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %11 = ttng.tc_gen5_mma %7, %9, %10[], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    nvws.aref.put.exit %0, %token [#nvws.async_op<tc5mma>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    %buffers_0, %token_1 = nvws.aref.get.enter %0 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
    %3 = nvws.aref.buffer %0, %token_1 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_2, %token_3 = ttng.tmem_load %3[] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    nvws.aref.get.exit %0, %token_1 [#nvws.async_op<none>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    "use"(%result_2) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

}
