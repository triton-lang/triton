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
    %2:6 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %c2_i32, %arg5 = %c0_i32, %arg6 = %c2_i32, %arg7 = %c1_i32, %arg8 = %c2_i32, %arg9 = %c1_i32) -> (i32, i32, i32, i32, i32, i32)  : i32 {
      %3 = "op_a"() {ttg.partition = 0 : i32} : () -> tensor<1xi32, #blocked>
      %4 = arith.addi %arg4, %c1_i32 : i32
      %5 = arith.cmpi eq, %4, %c3_i32 : i32
      %6 = arith.select %5, %c0_i32, %4 : i32
      %7 = arith.xori %arg5, %c1_i32 : i32
      %8 = arith.select %5, %7, %arg5 : i32
      // CHECK: op_a
      // CHECK-NEXT: addi
      // CHECK-NEXT: cmpi
      // CHECK-NEXT: [[STAGE:%.*]] = arith.select
      // CHECK-NEXT: xori
      // CHECK-NEXT: [[PHASE:%.*]] = arith.select
      // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]][[[STAGE]]]
      // CHECK-NEXT: ttng.wait_barrier [[EMPTYMBAR]], [[PHASE]] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32}
      // CHECK: local_store
      // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]][[[STAGE]]]
      // CHECK-NEXT: ttng.arrive_barrier [[FULLMBAR]], 1 {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32}
      %buffers, %token = nvws.aref.put.enter %1[%6, %8] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %3, %buffers {ttg.partition = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%6], %token [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = 0 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %9 = arith.addi %arg6, %c1_i32 : i32
      %10 = arith.cmpi eq, %9, %c3_i32 : i32
      %11 = arith.select %10, %c0_i32, %9 : i32
      %12 = arith.xori %arg7, %c1_i32 : i32
      %13 = arith.select %10, %12, %arg7 : i32

      // CHECK: addi
      // CHECK-NEXT: cmpi
      // CHECK-NEXT: [[STAGE:%.*]] = arith.select
      // CHECK-NEXT: xori
      // CHECK-NEXT: [[PHASE:%.*]] = arith.select
      // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]][[[STAGE]]]
      // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]], [[PHASE]] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = 1 : i32}
      // CHECK: local_load
      // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]][[[STAGE]]]
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR]], 1 {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = 1 : i32}
      %buffers_0, %token_1 = nvws.aref.get.enter %1[%11, %13] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = 1 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %14 = ttg.local_load %buffers_0 {ttg.partition = 1 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%11], %token_1 [#nvws.async_op<none>] {loop.cluster = 2 : i32, loop.stage = 3 : i32, ttg.partition = 1 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%14) {ttg.partition = 1 : i32} : (tensor<1xi32, #blocked>) -> ()
      %15 = arith.addi %arg8, %c1_i32 : i32
      %16 = arith.cmpi eq, %15, %c3_i32 : i32
      %17 = arith.select %16, %c0_i32, %15 : i32
      %18 = arith.xori %arg9, %c1_i32 : i32
      %19 = arith.select %16, %18, %arg9 : i32

      // CHECK: addi
      // CHECK-NEXT: cmpi
      // CHECK-NEXT: [[STAGE:%.*]] = arith.select
      // CHECK-NEXT: xori
      // CHECK-NEXT: [[PHASE:%.*]] = arith.select
      // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]][[[STAGE]]]
      // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]], [[PHASE]] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = 2 : i32}
      // CHECK: local_load
      // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]][[[STAGE]]]
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR]], 1 {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = 2 : i32}
      %buffers_2, %token_3 = nvws.aref.get.enter %1[%17, %19] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = 2 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %20 = ttg.local_load %buffers_2 {ttg.partition = 2 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%17], %token_3 [#nvws.async_op<none>] {loop.cluster = 3 : i32, loop.stage = 4 : i32, ttg.partition = 2 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%20) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      "op_d"(%20) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      scf.yield %6, %8, %11, %13, %17, %19 : i32, i32, i32, i32, i32, i32
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
    %2:8 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %c2_i32, %arg5 = %c0_i32, %arg6 = %c2_i32, %arg7 = %c1_i32, %arg8 = %c2_i32, %arg9 = %c1_i32, %arg10 = %c2_i32, %arg11 = %c1_i32) -> (i32, i32, i32, i32, i32, i32, i32, i32)  : i32 {
      %3 = "op_a"() {ttg.partition = 0 : i32} : () -> tensor<1xi32, #blocked>
      %4 = arith.addi %arg4, %c1_i32 : i32
      %5 = arith.cmpi eq, %4, %c3_i32 : i32
      %6 = arith.select %5, %c0_i32, %4 : i32
      %7 = arith.xori %arg5, %c1_i32 : i32
      %8 = arith.select %5, %7, %arg5 : i32
      %buffers, %token = nvws.aref.put.enter %1[%6, %8] {ttg.partition = 0 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %3, %buffers {ttg.partition = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%6], %token [#nvws.async_op<none>] {ttg.partition = 0 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %9 = arith.addi %arg6, %c1_i32 : i32
      %10 = arith.cmpi eq, %9, %c3_i32 : i32
      %11 = arith.select %10, %c0_i32, %9 : i32
      %12 = arith.xori %arg7, %c1_i32 : i32
      %13 = arith.select %10, %12, %arg7 : i32
      %buffers_0, %token_1 = nvws.aref.get.enter %1[%11, %13] {ttg.partition = 1 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %14 = ttg.local_load %buffers_0 {ttg.partition = 1 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%11], %token_1 [#nvws.async_op<none>] {ttg.partition = 1 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%14) {ttg.partition = 1 : i32} : (tensor<1xi32, #blocked>) -> ()
      %15 = arith.addi %arg8, %c1_i32 : i32
      %16 = arith.cmpi eq, %15, %c3_i32 : i32
      %17 = arith.select %16, %c0_i32, %15 : i32
      %18 = arith.xori %arg9, %c1_i32 : i32
      %19 = arith.select %16, %18, %arg9 : i32
      %buffers_2, %token_3 = nvws.aref.get.enter %1[%17, %19] {ttg.partition = 2 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %20 = ttg.local_load %buffers_2 {ttg.partition = 2 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%17], %token_3 [#nvws.async_op<none>] {ttg.partition = 2 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%20) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      "op_d"(%20) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      %21 = arith.addi %arg10, %c1_i32 : i32
      %22 = arith.cmpi eq, %21, %c3_i32 : i32
      %23 = arith.select %22, %c0_i32, %21 : i32
      %24 = arith.xori %arg11, %c1_i32 : i32
      %25 = arith.select %22, %24, %arg11 : i32
      %buffers_4, %token_5 = nvws.aref.get.enter %1[%23, %25] {ttg.partition = 3 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %26 = ttg.local_load %buffers_4 {ttg.partition = 3 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%23], %token_5 [#nvws.async_op<none>] {ttg.partition = 3 : i32} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_e"(%26) {ttg.partition = 3 : i32} : (tensor<1xi32, #blocked>) -> ()
      "op_f"(%26) {ttg.partition = 3 : i32} : (tensor<1xi32, #blocked>) -> ()
      scf.yield %6, %8, %11, %13, %17, %19, %23, %25 : i32, i32, i32, i32, i32, i32, i32, i32
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
    // CHECK: [[IDX:%.*]]:5 = scf.for
    %2:8 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %cst_0, %arg5 = %cst, %arg6 = %c0_i32, %arg7 = %c0_i32, %arg8 = %c0_i32, %arg9 = %c1_i32, %arg10 = %c0_i32, %arg11 = %c1_i32) -> (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>, i32, i32, i32, i32, i32, i32)  : i32 {
      %3 = arith.xori %arg7, %c1_i32 : i32
      %4 = arith.select %true, %3, %arg7 : i32
      // CHECK-NEXT: [[PHASE:%.*]] = arith.xori
      // CHECK: [[EMPTYBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
      // CHECK-NEXT: ttng.wait_barrier [[EMPTYBAR1]], [[PHASE]]
      // CHECK: local_store
      // CHECK-NEXT: [[FULLBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
      // CHECK-NEXT: ttng.arrive_barrier [[FULLBAR1]], 1
      // CHECK: op_a
      %buffers, %token = nvws.aref.put.enter %1[%c0_i32, %4] {ttg.partition = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %arg5, %buffers {ttg.partition = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      nvws.aref.put.exit %1[%c0_i32], %token [#nvws.async_op<none>] {ttg.partition = 0 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      %5 = "op_a"() {ttg.partition = 0 : i32} : () -> tensor<1xi32, #blocked>

      // CHECK-NEXT: [[PHASE:%.*]] = arith.xori
      // CHECK: [[FULLMBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
      // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR1]], [[PHASE]]
      // CHECK: local_load
      // CHECK-NEXT: [[EMPTYMBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR1]], 1
      // CHECK: op_d
      %6 = arith.xori %arg9, %c1_i32 : i32
      %7 = arith.select %true, %6, %arg9 : i32
      %buffers_1, %token_2 = nvws.aref.get.enter %1[%c0_i32, %7] {ttg.partition = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %8 = ttg.local_load %buffers_1 {ttg.partition = 1 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_2 [#nvws.async_op<none>] {ttg.partition = 1 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_d"(%8) {ttg.partition = 1 : i32} : (tensor<1xi32, #blocked>) -> ()

      // CHECK-NEXT: [[PHASE:%.*]] = arith.xori
      // CHECK: [[FULLMBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
      // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR1]], [[PHASE]]
      // CHECK: local_load
      // CHECK-NEXT: [[EMPTYMBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
      // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR1]], 1
      // CHECK: op_d
      %9 = arith.xori %arg11, %c1_i32 : i32
      %10 = arith.select %true, %9, %arg11 : i32
      %buffers_3, %token_4 = nvws.aref.get.enter %1[%c0_i32, %10] {ttg.partition = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %11 = ttg.local_load %buffers_3 {ttg.partition = 2 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      nvws.aref.get.exit %1[%c0_i32], %token_4 [#nvws.async_op<none>] {ttg.partition = 2 : i32} : <[!ttg.memdesc<1x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_d"(%11) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      scf.yield %5, %arg4, %c0_i32, %4, %c0_i32, %7, %c0_i32, %10 : tensor<1xi32, #blocked>, tensor<1xi32, #blocked>, i32, i32, i32, i32, i32, i32
    } {ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    ttg.local_dealloc %0 : !ttg.memdesc<1x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
