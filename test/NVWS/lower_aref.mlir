// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-aref | FileCheck %s

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {

  //CHECK-LABEL: @two_consumers
  tt.func @two_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[BUF:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.init_barrier [[EMPTYSLICE]], 2
    // CHECK-NEXT: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.init_barrier [[EMPTYSLICE]], 2
    // CHECK-NEXT: [[FULL:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.init_barrier [[FULLSLICE]], 1
    // CHECK-NEXT: [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.init_barrier [[FULLSLICE]], 1
    nvws.warp_group
    partition0 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %2 = "op_a"() : () -> tensor<1xi32, #blocked>
        %3, %token3 = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 3 : i32}: <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        ttg.local_store %2, %3 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
        // CHECK: op_a
        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
        // CHECK-NEXT: ttng.wait_barrier [[EMPTYMBAR]], {{.*}} {loop.cluster = 1 : i32, loop.stage = 3 : i32}
        // CHECK: local_store
        // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
        // CHECK-NEXT: ttng.arrive_barrier [[FULLMBAR]], 1 {loop.cluster = 1 : i32, loop.stage = 3 : i32}
        nvws.aref.put.exit %1[%c0_i32], %token3 [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 3 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      }
      nvws.warp_group.yield
    }
    partition1 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        // CHECK: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]], {{.*}} {loop.cluster = 2 : i32, loop.stage = 3 : i32}
        // CHECK: [[VAL:%.*]] = ttg.local_load
        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
        // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR]], 1 {loop.cluster = 2 : i32, loop.stage = 3 : i32}
        // CHECK: "op_b"([[VAL]])
        %2:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 3 : i32}: <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %3 = ttg.local_load %2#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %2#1 [#nvws.async_op<none>] {loop.cluster = 2 : i32, loop.stage = 3 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_b"(%3) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    partition2 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        // CHECK: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]], {{.*}} {loop.cluster = 3 : i32, loop.stage = 4 : i32}
        // CHECK: [[VAL:%.*]] = ttg.local_load
        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
        // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR]], 1 {loop.cluster = 3 : i32, loop.stage = 4 : i32}
        // CHECK: "op_c"([[VAL]])
        // CHECK: "op_d"([[VAL]])
        %2:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {loop.cluster = 3 : i32, loop.stage = 4 : i32}: <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %3 = ttg.local_load %2#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %2#1 [#nvws.async_op<none>] {loop.cluster = 3 : i32, loop.stage = 4 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_c"(%3) : (tensor<1xi32, #blocked>) -> ()
        "op_d"(%3) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    // CHECK: nvws.warp_group.return
    // CHECK-NEXT: }
    // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.inval_barrier [[FULLMBAR]]
    // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.inval_barrier [[FULLMBAR]]
    // CHECK-NEXT: ttg.local_dealloc
    // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT: ttg.local_dealloc
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }

  //CHECK-LABEL: @three_consumers
  tt.func @three_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[BUF:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.init_barrier [[EMPTYSLICE]], 3
    // CHECK-NEXT: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT: ttng.init_barrier [[EMPTYSLICE]], 3
    // CHECK-NEXT: [[FULL:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.init_barrier [[FULLSLICE]], 1
    // CHECK-NEXT: [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT: ttng.init_barrier [[FULLSLICE]], 1
    nvws.warp_group
    partition0 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        %2 = "op_a"() : () -> tensor<1xi32, #blocked>
        %3, %token3 = nvws.aref.put.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        ttg.local_store %2, %3 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
        // CHECK: op_a
        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
        // CHECK-NEXT: ttng.wait_barrier [[EMPTYMBAR]]
        // CHECK: local_store
        // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
        // CHECK-NEXT: ttng.arrive_barrier [[FULLMBAR]], 1
        nvws.aref.put.exit %1[%c0_i32], %token3 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      }
      nvws.warp_group.yield
    }
    partition1 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        // CHECK: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]]
        // CHECK: [[VAL:%.*]] = ttg.local_load
        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
        // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR]], 1
        // CHECK: "op_b"([[VAL]])
        %2:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %3 = ttg.local_load %2#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %2#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_b"(%3) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    partition2 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        // CHECK: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]]
        // CHECK: [[VAL:%.*]] = ttg.local_load
        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
        // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR]], 1
        // CHECK: "op_c"([[VAL]])
        // CHECK: "op_d"([[VAL]])
        %2:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %3 = ttg.local_load %2#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %2#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_c"(%3) : (tensor<1xi32, #blocked>) -> ()
        "op_d"(%3) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    partition3 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        // CHECK: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]]
        // CHECK: [[VAL:%.*]] = ttg.local_load
        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
        // CHECK-NEXT: ttng.arrive_barrier [[EMPTYMBAR]], 1
        // CHECK: "op_c"([[VAL]])
        // CHECK: "op_d"([[VAL]])
        %2:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %3 = ttg.local_load %2#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %2#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_c"(%3) : (tensor<1xi32, #blocked>) -> ()
        "op_d"(%3) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:0", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  //CHECK-LABEL: @aref_lowering
  tt.func @aref_lowering(%d : !ttg.memdesc<3x64x16xf16, #shared0, #smem>,
                         %e : !ttg.memdesc<3x16x32xf16, #shared0, #smem>,
                         %cond : i1) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %lb = arith.constant 0 : i32
    // CHECK:   [[C3:%.*]] = arith.constant 3 : i32
    // CHECK:   [[C0:%.*]] = arith.constant 0 : i32
    // CHECK:   [[C1:%.*]] = arith.constant 1 : i32
    %ub = arith.constant 4 : i32

    // CHECK:   [[EMPTY1:%.*]] = ttg.local_alloc
    // CHECK:   init_barrier
    // CHECK:   init_barrier
    // CHECK:   init_barrier
    // CHECK:   [[FULL1:%.*]] = ttg.local_alloc
    // CHECK:   init_barrier
    // CHECK:   init_barrier
    // CHECK:   init_barrier
    %aref0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

    // CHECK:   [[EMPTY0:%.*]] = ttg.local_alloc
    // CHECK:   init_barrier
    // CHECK:   init_barrier
    // CHECK:   init_barrier
    // CHECK:   [[FULL0:%.*]] = ttg.local_alloc
    // CHECK:   init_barrier
    // CHECK:   init_barrier
    // CHECK:   init_barrier
    %aref1 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>

    nvws.warp_group
    partition0  num_warps(4) {
      // CHECK: [[IDX:%.*]]:6 = scf.for [[I:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[C1:%.*]] iter_args([[S0:%.*]] = [[C0]], [[P0:%.*]] = [[C1]], [[S1:%.*]] = [[C0]], [[P1:%.*]] = [[C1]], [[S2:%.*]] = [[C0]],  [[S3:%.*]] = [[C0]])
      scf.for %i = %lb to %ub step %c1_i32 : i32{

        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY0]], [[S0]]
        // CHECK-NEXT: ttng.wait_barrier [[EMPTYMBAR]], [[P0]]
        %1:3 = nvws.aref.put.enter %aref0[%c0_i32, %c0_i32] {aref_tag = "put0"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token

        // CHECK-NEXT: [[BUFA:%.*]] = ttg.memdesc_index %arg0, [[S0]]
        // CHECK-NEXT: [[BUFB:%.*]] = ttg.memdesc_index %arg1, [[S0]]
        // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL0]], [[S2]]
        // CHECK-NEXT: ttng.barrier_expect [[FULLMBAR]], 0
        // CHECK-NEXT: [[S0a:%.*]] = arith.addi [[S0]], [[C1]]
        // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S0a]], [[C3]]
        // CHECK-NEXT: [[S0b:%.*]] = arith.select [[CMP]], [[C0]], [[S0a]]
        // CHECK-NEXT: [[P0a:%.*]] = arith.xori [[P0]], [[C1]]
        // CHECK-NEXT: [[P0b:%.*]] = arith.select [[CMP]], [[P0a]], [[P0]]
        // CHECK-NEXT: "tma_load"([[BUFA]])
        // CHECK-NEXT: "sts"([[BUFB]])
        "tma_load"(%1#0) : (!ttg.memdesc<64x16xf16, #shared0, #smem>) -> ()
        "sts"(%1#1) : (!ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL0]], [[S2]]
        // CHECK-NEXT: ttng.arrive_barrier [[FULLMBAR]], 1
        // CHECK-NEXT: [[S2a:%.*]] = arith.addi [[S2]], [[C1]]
        // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S2a]], [[C3]]
        // CHECK-NEXT: [[S2b:%.*]] = arith.select [[CMP]], [[C0]], [[S2a]]
        nvws.aref.put.exit %aref0[%c0_i32], %1#2 [#nvws.async_op<tma_load>, #nvws.async_op<none>] {aref_tag = "put0"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token

        // CHECK-NEXT: [[SP1S3:%.*]]:3 = scf.if
        scf.if %cond {

          // CHECK-NEXT: [[BAR:%.*]] = ttg.memdesc_index {{.*}}, [[S1]]
          // CHECK-NEXT: ttng.wait_barrier [[BAR]], [[P1]]
          // CHECK: [[S1a:%.*]] = arith.addi [[S1]], [[C1]]
          // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S1a]], [[C3]]
          // CHECK-NEXT: [[S1b:%.*]] = arith.select [[CMP]], [[C0]], [[S1a]]
          // CHECK-NEXT: [[P1a:%.*]] = arith.xori [[P1]], [[C1]]
          // CHECK-NEXT: [[P1b:%.*]] = arith.select [[CMP]], [[P1a]], [[P1]]

          %2:3 = nvws.aref.put.enter %aref1[%c0_i32, %c0_i32] {aref_tag = "put1"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
          "tmem_store"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

          // CHECK: [[BAR:%.*]] = ttg.memdesc_index {{.*}}, [[S3]]
          // CHECK-NEXT: ttng.arrive_barrier [[BAR]], 1
          // CHECK: [[S3a:%.*]] = arith.addi [[S3]], [[C1]]
          // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S3a]], [[C3]]
          // CHECK-NEXT: [[S3b:%.*]] = arith.select [[CMP]], [[C0]], [[S3a]]
          nvws.aref.put.exit %aref1[%c0_i32], %2#2 [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token

          // CHECK: scf.yield [[S1b]], [[P1b]], [[S3b]]
        }
        // CHECK-NEXT: } else {
        // CHECK-NEXT:   scf.yield [[S1]], [[P1]], [[S3]]
        // CHECK-NEXT: }

        // CHECK: scf.yield [[S0b]], [[P0b]], [[SP1S3]]#0, [[SP1S3]]#1, [[S2b]], [[SP1S3]]#2
        // CHECK-NEXT: }
      }

      // CHECK-NEXT: [[IDX1:%.*]]:3 = scf.if
      scf.if %cond {

        // CHECK-NEXT: [[BAR:%.*]] = ttg.memdesc_index {{.*}}, [[IDX]]#0
        // CHECK-NEXT: ttng.wait_barrier [[BAR]], [[IDX]]#1
        %1:3 = nvws.aref.put.enter %aref0[%c0_i32, %c0_i32] {aref_tag = "put1"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
        "tma_load"(%1#0) : (!ttg.memdesc<64x16xf16, #shared0, #smem>) -> ()
        "sts"(%1#1) : (!ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
        //CHECK: sts

        // CHECK: [[BAR:%.*]] = ttg.memdesc_index {{.*}}, [[IDX]]#4
        // CHECK-NEXT: ttng.arrive_barrier [[BAR]]
        nvws.aref.put.exit %aref0[%c0_i32], %1#2 [#nvws.async_op<tma_load>, #nvws.async_op<none>] {aref_tag = "put1"} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      }

      // CHECK: [[BAR:%.*]] = ttg.memdesc_index {{.*}}, [[IDX]]#2
      // CHECK-NEXT: ttng.wait_barrier [[BAR]], [[IDX]]#3
      %1:3 = nvws.aref.put.enter %aref1[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "tmem_store"(%1#0, %1#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      // CHECK: [[BAR:%.*]] = ttg.memdesc_index {{.*}}, [[IDX]]#5
      // CHECK-NEXT: ttng.arrive_barrier [[BAR]], 1
      nvws.aref.put.exit %aref1[%c0_i32], %1#2 [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      nvws.warp_group.return
    }
    partition1 num_warps(8) {
      // CHECK: [[IDX:%.*]]:6 = scf.for [[I:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[C1:%.*]] iter_args([[S0:%.*]] = [[C0]], [[P0:%.*]] = [[C0]], [[S1:%.*]] = [[C0]], [[P1:%.*]] = [[C0]], [[S2:%.*]] = [[C0]], [[S3:%.*]] = [[C0]])
      scf.for %i = %lb to %ub step %c1_i32 : i32{

        // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL0]], [[S0]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLMBAR]], [[P0]]
        %2:3 = nvws.aref.get.enter %aref0[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token

        // CHECK-NEXT: [[BUFA:%.*]] = ttg.memdesc_index %arg0, [[S0]]
        // CHECK-NEXT: [[BUFB:%.*]] = ttg.memdesc_index %arg1, [[S0]]
        // CHECK-NEXT: arith.addi
        // CHECK-NEXT: arith.cmpi
        // CHECK-NEXT: arith.select
        // CHECK-NEXT: arith.xori
        // CHECK-NEXT: arith.select
        // CHECK-NEXT: "tc5mma"([[BUFA]], [[BUFB]])
        "tc5mma"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY0]], [[S2]]
        // CHECK-NEXT: ttng.tc_gen5_commit [[EMPTYMBAR]]
        // CHECK-NEXT: arith.addi
        // CHECK-NEXT: arith.cmpi
        // CHECK-NEXT: arith.select
        // CHECK-NOT: arith.xori
        // CHECK-NOT: arith.select
        nvws.aref.get.exit %aref0[%c0_i32], %2#2 [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token

        // CHECK: [[IDX13:%.*]]:3 = scf.if
        scf.if %cond {
          // CHECK: [[BAR:%.*]] = ttg.memdesc_index {{.*}}, [[S1]]
          // CHECK-NEXT: ttng.wait_barrier  [[BAR]], [[P1]]
          %3:3 = nvws.aref.get.enter %aref1[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
          "tmem_load"(%3#0, %3#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
          // CHECK: tmem_load

          // CHECK-NEXT: [[BAR:%.*]] = ttg.memdesc_index {{.*}}, [[S3]]
          // CHECK-NEXT: ttng.arrive_barrier [[BAR]], 1
          nvws.aref.get.exit %aref1[%c0_i32], %3#2 [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
        }
        // CHECK: } else {
        // CHECK-NEXT:   scf.yield [[S1]], [[P1]], [[S3]]
        // CHECK-NEXT: }

        // CHECK: scf.yield {{.*}}, {{.*}}, [[IDX13]]#0, [[IDX13]]#1, {{.*}}, [[IDX13]]#2
      }
      scf.if %cond {
        // CHECK: [[BAR:%.*]] = ttg.memdesc_index {{.*}}, [[IDX]]#0
        // CHECK-NEXT: ttng.wait_barrier  [[BAR]], [[IDX]]#1
        %2:3 = nvws.aref.get.enter %aref0[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
        "tc5mma"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()

        // CHECK: [[BAR:%.*]] = ttg.memdesc_index {{.*}}, [[IDX]]#4
        // CHECK-NEXT: ttng.tc_gen5_commit  [[BAR]]
        nvws.aref.get.exit %aref0[%c0_i32], %2#2 [#nvws.async_op<tc5mma>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      }
      // CHECK: } else {
      // CHECK-NEXT:   scf.yield [[IDX]]#0, [[IDX]]#1, [[IDX]]#4
      // CHECK-NEXT: }

      %2:3 = nvws.aref.get.enter %aref1[%c0_i32, %c0_i32] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "tmem_load"(%2#0, %2#1) : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      nvws.aref.get.exit %aref1[%c0_i32], %2#2 [#nvws.async_op<none>] : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      nvws.warp_group.return
    }
    // CHECK: warp_group.return
    // CHECK-NEXT: }
    // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL0]]
    // CHECK-NEXT: ttng.inval_barrier [[FULLMBAR]]
    // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL0]]
    // CHECK-NEXT: ttng.inval_barrier [[FULLMBAR]]
    // CHECK-NEXT: [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL0]]
    // CHECK-NEXT: ttng.inval_barrier [[FULLMBAR]]
    // CHECK-NEXT: ttg.local_dealloc
    // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY0]]
    // CHECK-NEXT: ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY0]]
    // CHECK-NEXT: ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT: [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY0]]
    // CHECK-NEXT: ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT: ttg.local_dealloc
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  //CHECK-LABEL: @complex_case
  tt.func @complex_case(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<1xi32, #blocked>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    %2 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %3 = nvws.aref.create %2 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    // CHECK: ttg.local_alloc
    // CHECK-NEXT: ttg.local_alloc
    // CHECK-NEXT: [[EMPTY2:%.*]] = ttg.local_alloc
    // CHECK-NEXT: memdesc_index
    // CHECK-NEXT: init_barrier
    // CHECK-NEXT: memdesc_index
    // CHECK-NEXT: init_barrier
    // CHECK-NEXT: [[FULL2:%.*]] = ttg.local_alloc
    // CHECK-NEXT: memdesc_index
    // CHECK-NEXT: init_barrier
    // CHECK-NEXT: memdesc_index
    // CHECK-NEXT: init_barrier

    // CHECK-NEXT: [[EMPTY1:%.*]] = ttg.local_alloc
    // CHECK-NEXT: memdesc_index
    // CHECK-NEXT: init_barrier
    // CHECK-NEXT: memdesc_index
    // CHECK-NEXT: init_barrier
    // CHECK-NEXT: [[FULL1:%.*]] = ttg.local_alloc
    // CHECK-NEXT: memdesc_index
    // CHECK-NEXT: init_barrier
    // CHECK-NEXT: memdesc_index
    // CHECK-NEXT: init_barrier
    nvws.warp_group
    partition0 num_warps(4) {
      %5:2 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %cst, %arg5 = %cst) -> (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>)  : i32 {
        // CHECK: [[EMPTYBAR2:%.*]] = ttg.memdesc_index [[EMPTY2]]
        // CHECK-NEXT: ttng.wait_barrier [[EMPTYBAR2]]
        // CHECK: local_store
        // CHECK-NEXT: [[FULLBAR2:%.*]] = ttg.memdesc_index [[FULL2]]
        // CHECK-NEXT: ttng.arrive_barrier [[FULLBAR2]], 1
        %6, %token6 = nvws.aref.put.enter %3[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        ttg.local_store %arg5, %6 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
        nvws.aref.put.exit %3[%c0_i32], %token6 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %7, %token7 = nvws.aref.put.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        // CHECK: [[EMPTYBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
        // CHECK-NEXT: ttng.wait_barrier [[EMPTYBAR1]]
        // CHECK: local_store
        // CHECK-NEXT: [[FULLBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
        // CHECK-NEXT: ttng.arrive_barrier [[FULLBAR1]], 1
        ttg.local_store %arg4, %7 : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
        nvws.aref.put.exit %1[%c0_i32], %token7 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %8 = "op_a"() : () -> tensor<1xi32, #blocked>
        scf.yield %8, %arg4 : tensor<1xi32, #blocked>, tensor<1xi32, #blocked>
      }
      nvws.warp_group.yield %5#0, %5#1 : tensor<1xi32, #blocked>, tensor<1xi32, #blocked>
    }
    partition1 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        // CHECK: [[FULLBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLBAR1]]
        // CHECK: [[VAL:%.*]] = ttg.local_load
        // CHECK-NEXT: [[EMPTYBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
        // CHECK-NEXT: ttng.arrive_barrier [[EMPTYBAR1]], 1
        // CHECK: "op_b"([[VAL]])
        %5:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %6 = ttg.local_load %5#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %5#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_b"(%6) : (tensor<1xi32, #blocked>) -> ()
        // CHECK: [[FULLBAR2:%.*]] = ttg.memdesc_index [[FULL2]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLBAR2]]
        // CHECK: [[VAL:%.*]] = ttg.local_load
        // CHECK-NEXT: [[EMPTYBAR2:%.*]] = ttg.memdesc_index [[EMPTY2]]
        // CHECK-NEXT: ttng.arrive_barrier [[EMPTYBAR2]], 1
        // CHECK: "op_d"([[VAL]])
        %7:2 = nvws.aref.get.enter %3[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %8 = ttg.local_load %7#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %3[%c0_i32], %7#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_d"(%8) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    partition2 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        // CHECK: [[FULLBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLBAR1]]
        // CHECK: [[VAL:%.*]] = ttg.local_load
        // CHECK-NEXT: [[EMPTYBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
        // CHECK-NEXT: ttng.arrive_barrier [[EMPTYBAR1]], 1
        // CHECK: "op_c"([[VAL]])
        // CHECK-NEXT: "op_c"([[VAL]])
        %5:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %6 = ttg.local_load %5#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %5#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_c"(%6) : (tensor<1xi32, #blocked>) -> ()
        "op_c"(%6) : (tensor<1xi32, #blocked>) -> ()
        // CHECK: [[FULLBAR2:%.*]] = ttg.memdesc_index [[FULL2]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLBAR2]]
        // CHECK: [[VAL:%.*]] = ttg.local_load
        // CHECK-NEXT: [[EMPTYBAR2:%.*]] = ttg.memdesc_index [[EMPTY2]]
        // CHECK-NEXT: ttng.arrive_barrier [[EMPTYBAR2]], 1
        // CHECK: "op_d"([[VAL]])
        %7:2 = nvws.aref.get.enter %3[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %8 = ttg.local_load %7#0 : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %3[%c0_i32], %7#1 [#nvws.async_op<none>] : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_d"(%8) : (tensor<1xi32, #blocked>) -> ()
      }
      nvws.warp_group.return
    }
    ttg.local_dealloc %2 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reuse_argument(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0> : tensor<1xi32, #blocked>
    %cst_0 = arith.constant dense<1> : tensor<1xi32, #blocked>
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    // CHECK: ttg.local_alloc
    // CHECK: [[EMPTY1:%.*]] = ttg.local_alloc
    // CHECK: [[FULL1:%.*]] = ttg.local_alloc
    nvws.warp_group
    partition0 num_warps(4) {
      %2:2 = scf.for %arg3 = %arg0 to %arg1 step %arg2 iter_args(%arg4 = %cst, %arg5 = %cst_0) -> (tensor<1xi32, #blocked>, tensor<1xi32, #blocked>)  : i32 {
        // CHECK: [[EMPTYBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
        // CHECK-NEXT: ttng.wait_barrier [[EMPTYBAR1]]
        // CHECK: local_store
        // CHECK-NEXT: [[FULLBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
        // CHECK-NEXT: ttng.arrive_barrier [[FULLBAR1]], 1
        // CHECK: op_a
        %3, %token3 = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {ttg.partition = 0 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        ttg.local_store %arg5, %3 {ttg.partition = 0 : i32} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>
        nvws.aref.put.exit %1[%c0_i32], %token3 [#nvws.async_op<none>] {ttg.partition = 0 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        %4 = "op_a"() {ttg.partition = 0 : i32} : () -> tensor<1xi32, #blocked>
        scf.yield %4, %arg4 : tensor<1xi32, #blocked>, tensor<1xi32, #blocked>
      } {ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32]}
      nvws.warp_group.yield %2#0, %2#1 : tensor<1xi32, #blocked>, tensor<1xi32, #blocked>
    }
    partition1 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        // CHECK: [[FULLBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLBAR1]]
        // CHECK: [[VAL:%.*]] = ttg.local_load
        // CHECK-NEXT: [[EMPTYBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
        // CHECK-NEXT: ttng.arrive_barrier [[EMPTYBAR1]], 1
        // CHECK: "op_d"([[VAL]])
        %5:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 1 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %6 = ttg.local_load %5#0 {ttg.partition = 1 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %5#1 [#nvws.async_op<none>] {ttg.partition = 1 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_d"(%6) {ttg.partition = 1 : i32} : (tensor<1xi32, #blocked>) -> ()
      } {ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32]}
      nvws.warp_group.return
    }
    partition2 num_warps(4) {
      scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
        // CHECK: [[FULLBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
        // CHECK-NEXT: ttng.wait_barrier [[FULLBAR1]]
        // CHECK: [[VAL:%.*]] = ttg.local_load
        // CHECK-NEXT: [[EMPTYBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
        // CHECK-NEXT: ttng.arrive_barrier [[EMPTYBAR1]], 1
        // CHECK: "op_d"([[VAL]])
        %7:2 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = 2 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1>, !ttg.async.token
        %8 = ttg.local_load %7#0 {ttg.partition = 2 : i32} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 2x1> -> tensor<1xi32, #blocked>
        nvws.aref.get.exit %1[%c0_i32], %7#1 [#nvws.async_op<none>] {ttg.partition = 2 : i32} : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
        "op_d"(%8) {ttg.partition = 2 : i32} : (tensor<1xi32, #blocked>) -> ()
      } {ttg.partition.stages = [1 : i32, 0 : i32, 0 : i32]}
      nvws.warp_group.return
    }
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tt.return
  }
}
