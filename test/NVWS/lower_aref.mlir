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
    // CHECK-NEXT: [[FULL:%.*]] = ttg.local_alloc
    // CHECK-NEXT: scf.for
    // CHECK-NEXT:   [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT:   ttng.init_barrier [[EMPTYSLICE]], 2
    // CHECK-NEXT:   [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT:   ttng.init_barrier [[FULLSLICE]], 1
    // CHECK-NEXT: }
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
    ttg.local_dealloc %0 : !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    nvws.aref.destroy %1 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    // CHECK: ttg.local_dealloc
    // CHECK-NEXT: scf.for
    // CHECK-NEXT:   [[EMPTYMBAR:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT:   ttng.inval_barrier [[EMPTYMBAR]]
    // CHECK-NEXT:   [[FULLMBAR:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT:   ttng.inval_barrier [[FULLMBAR]]
    tt.return
  }

  //CHECK-LABEL: @three_consumers
  tt.func @three_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<2x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[BUF:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[EMPTY:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[FULL:%.*]] = ttg.local_alloc
    // CHECK-NEXT: scf.for
    // CHECK-NEXT:   [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK-NEXT:   ttng.init_barrier [[EMPTYSLICE]], 3
    // CHECK-NEXT:   [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK-NEXT:   ttng.init_barrier [[FULLSLICE]], 1
    // CHECK-NEXT: }
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
    // CHECK-NEXT: [[EMPTY1:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[FULL1:%.*]] = ttg.local_alloc
    // CHECK-NEXT: scf.for
    // CHECK-NEXT:   [[EMPTYBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
    // CHECK-NEXT:   init_barrier [[EMPTYBAR1]], 2
    // CHECK-NEXT:   [[FULLBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
    // CHECK-NEXT:   init_barrier [[FULLBAR1]], 1

    // CHECK: ttg.local_alloc
    // CHECK-NEXT: [[EMPTY2:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[FULL2:%.*]] = ttg.local_alloc
    // CHECK-NEXT: scf.for
    // CHECK-NEXT:   [[EMPTYBAR2:%.*]] = ttg.memdesc_index [[EMPTY2]]
    // CHECK-NEXT:   init_barrier [[EMPTYBAR2]], 2
    // CHECK-NEXT:   [[FULLBAR2:%.*]] = ttg.memdesc_index [[FULL2]]
    // CHECK-NEXT:   init_barrier [[FULLBAR2]], 1
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
    // CHECK-NEXT: [[EMPTY1:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[FULL1:%.*]] = ttg.local_alloc
    // CHECK-NEXT: scf.for
    // CHECK-NEXT:   [[EMPTYBAR1:%.*]] = ttg.memdesc_index [[EMPTY1]]
    // CHECK-NEXT:   init_barrier [[EMPTYBAR1]], 2
    // CHECK-NEXT:   [[FULLBAR1:%.*]] = ttg.memdesc_index [[FULL1]]
    // CHECK-NEXT:   init_barrier [[FULLBAR1]], 1
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

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %0 = ub.poison : !ttg.async.token
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %1 = ttg.memdesc_index %result, %c0_i32 : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
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
    %9 = ttg.memdesc_index %8, %c0_i32 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 1x1>
    ttng.init_barrier %9, 1 : !ttg.memdesc<1xi64, #shared1, #smem, mutable, 1x1>
    nvws.warp_group
    partition0 num_warps(4) {
      %10 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %2) -> (!ttg.async.token)  : i32 {
        scf.yield %0 : !ttg.async.token
      } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
      nvws.warp_group.yield %10, %c0_i32, %c0_i32 : !ttg.async.token, i32, i32
    }
    partition1 num_warps(4) {
      scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32  : i32 {
        // CHECK-COUNT-1: ttng.wait_barrier {{.*}}, {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32}
        // CHECK: [[BUF_A_SLICE:%.*]] = ttg.memdesc_index [[BUF_A]]
        // CHECK: [[BUF_B_SLICE:%.*]] = ttg.memdesc_index [[BUF_B]]
        %buffers, %token_0 = nvws.aref.get.enter %4[%c0_i32, %c0_i32] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.async.token
        %buffers_1, %token_2 = nvws.aref.get.enter %6[%c0_i32, %c0_i32] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.async.token
	// CHECK: [[BUF_B_SLICE_TRANS:%.*]] = ttg.memdesc_trans [[BUF_B_SLICE]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf16, #shared2, #smem, mutable>
        %10 = ttg.memdesc_trans %buffers_1 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64> -> !ttg.memdesc<64x128xf16, #shared2, #smem, 1x64x128>
        %11 = arith.cmpi eq, %arg5, %7 : i32
        %12 = ttg.memdesc_index %result, %c0_i32 : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
        %13 = ttg.memdesc_index %8, %c0_i32 : !ttg.memdesc<1x1xi64, #shared1, #smem, mutable> -> !ttg.memdesc<1xi64, #shared1, #smem, mutable, 1x1>
	// CHECK: ttng.tc_gen5_mma [[BUF_A_SLICE]], [[BUF_B_SLICE_TRANS]]
        %14 = ttng.tc_gen5_mma %buffers, %10, %12[], %true, %true, %13[%11] {is_async, loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.memdesc<64x128xf16, #shared2, #smem, 1x64x128>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.memdesc<1xi64, #shared1, #smem, mutable, 1x1>
	// CHECK: [[TMA_EMPTY_SLICE:%.*]] = ttg.memdesc_index [[TMA_EMPTY]]
	// CHECK-COUNT-1: ttng.tc_gen5_commit [[TMA_EMPTY_SLICE]] {loop.cluster = 0 : i32, loop.stage = 1 : i32}
        nvws.aref.get.exit %6[%c0_i32], %token_2 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
        nvws.aref.get.exit %4[%c0_i32], %token_0 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
      nvws.warp_group.return
    }
    partition2 num_warps(4) {
      scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32  : i32 {
        %10 = arith.muli %arg5, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
        // CHECK-COUNT-1: ttng.wait_barrier {{.*}}, {{.*}} {loop.cluster = 1 : i32, loop.stage = 0 : i32}
        // CHECK: [[BUF_A_SLICE:%.*]] = ttg.memdesc_index [[BUF_A]]
        // CHECK: [[BUF_B_SLICE:%.*]] = ttg.memdesc_index [[BUF_B]]
        // CHECK: [[TMA_FULL_SLICE:%.*]] = ttg.memdesc_index [[TMA_FULL]]
	// CHECK: ttng.barrier_expect [[TMA_FULL_SLICE]], 32768
        %buffers, %token_0 = nvws.aref.put.enter %4[%c0_i32, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>, !ttg.async.token

        // CHECK: ttng.async_tma_copy_global_to_local {{.*}} [[BUF_A_SLICE]], [[TMA_FULL_SLICE]]
        // CHECK: ttng.async_tma_copy_global_to_local {{.*}} [[BUF_B_SLICE]], [[TMA_FULL_SLICE]]
        nvws.descriptor_load %arg3[%arg1, %10] 16384 %buffers {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
        nvws.aref.put.exit %4[%c0_i32], %token_0 [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
        %buffers_1, %token_2 = nvws.aref.put.enter %6[%c0_i32, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>, !ttg.async.token
        nvws.descriptor_load %arg4[%arg2, %10] 16384 %buffers_1 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
        nvws.aref.put.exit %6[%c0_i32], %token_2 [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
      nvws.warp_group.return
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  tt.func @load_used_as_reg_and_smem(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: [[EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[FULL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: scf.for
    // CHECK:   [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK:   ttng.init_barrier [[EMPTYSLICE]], 2
    // CHECK:   [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK:   ttng.init_barrier [[FULLSLICE]], 1
    nvws.warp_group
    partition0 num_warps(4) {
      scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
        %c0_i32_0 = arith.constant 0 : i32
        %buffers, %token = nvws.aref.get.enter %1[%c0_i32_0, %c0_i32_0] {loop.cluster = 1 : i32, loop.stage = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.async.token
        %2 = ttg.local_load %buffers {loop.cluster = 1 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64> -> tensor<128x64xf16, #blocked>
        %c0_i32_1 = arith.constant 0 : i32
        // CHECK: ttng.fence_async_shared {bCluster = false, loop.cluster = 1 : i32, loop.stage = 1 : i32}
        // CHECK: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
	// CHECK: ttng.arrive_barrier [[EMPTYSLICE]], 1 {loop.cluster = 1 : i32, loop.stage = 1 : i32}
        nvws.aref.get.exit %1[%c0_i32_1], %token [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
        "use1"(%2) {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<128x64xf16, #blocked>) -> ()
      } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
      nvws.warp_group.yield
    }
    partition1 num_warps(4) {
      scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
        %c0_i32_0 = arith.constant 0 : i32
        %buffers, %token = nvws.aref.get.enter %1[%c0_i32_0, %c0_i32_0] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.async.token
	// CHECK: "use2"
        // CHECK: ttng.fence_async_shared {bCluster = false, loop.cluster = 0 : i32, loop.stage = 1 : i32}
        // CHECK: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
	// CHECK: ttng.arrive_barrier [[EMPTYSLICE]], 1 {loop.cluster = 0 : i32, loop.stage = 1 : i32}
        "use2"(%buffers) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (!ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>) -> ()
        %c0_i32_1 = arith.constant 0 : i32
        nvws.aref.get.exit %1[%c0_i32_1], %token [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
      nvws.warp_group.return
    }
    partition2 num_warps(4) {
      scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
        %c0_i32_0 = arith.constant 0 : i32
        %buffers, %token = nvws.aref.put.enter %1[%c0_i32_0, %c0_i32_0] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>, !ttg.async.token
        nvws.descriptor_load %arg0[%arg2, %arg2] 16384 %buffers {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
        %c0_i32_1 = arith.constant 0 : i32
        nvws.aref.put.exit %1[%c0_i32_1], %token [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
      nvws.warp_group.return
    }
    tt.return
  }

  tt.func @load_used_as_reg_and_smem_same_partition(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: [[EMPTY:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    // CHECK: [[FULL:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>
    // CHECK: scf.for
    // CHECK:   [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
    // CHECK:   ttng.init_barrier [[EMPTYSLICE]], 1
    // CHECK:   [[FULLSLICE:%.*]] = ttg.memdesc_index [[FULL]]
    // CHECK:   ttng.init_barrier [[FULLSLICE]], 1
    nvws.warp_group
    partition0 num_warps(4) {
      scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
        %c0_i32_0 = arith.constant 0 : i32
        %buffers, %token = nvws.aref.get.enter %1[%c0_i32_0, %c0_i32_0] {loop.cluster = 0 : i32, loop.stage = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>, !ttg.async.token
        %2 = ttg.local_load %buffers {loop.cluster = 0 : i32, loop.stage = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64> -> tensor<128x64xf16, #blocked>
        // CHECK: ttng.wait_barrier {{.*}}, {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32}
	// CHECK: "use1"
	// CHECK: "use2"
        // CHECK: ttng.fence_async_shared {bCluster = false, loop.cluster = 1 : i32, loop.stage = 1 : i32}
        // CHECK: [[EMPTYSLICE:%.*]] = ttg.memdesc_index [[EMPTY]]
	// CHECK: ttng.arrive_barrier [[EMPTYSLICE]], 1 {loop.cluster = 1 : i32, loop.stage = 1 : i32}
        "use1"(%2) {loop.cluster = 1 : i32, loop.stage = 1 : i32} : (tensor<128x64xf16, #blocked>) -> ()
        "use2"(%buffers) {loop.cluster = 0 : i32, loop.stage = 1 : i32} : (!ttg.memdesc<128x64xf16, #shared, #smem, 1x128x64>) -> ()
        %c0_i32_1 = arith.constant 0 : i32
        nvws.aref.get.exit %1[%c0_i32_1], %token [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 1 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
      nvws.warp_group.yield
    }
    partition1 num_warps(4) {
      scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
        %c0_i32_0 = arith.constant 0 : i32
        %buffers, %token = nvws.aref.put.enter %1[%c0_i32_0, %c0_i32_0] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>, !ttg.async.token
        nvws.descriptor_load %arg0[%arg2, %arg2] 16384 %buffers {loop.cluster = 2 : i32, loop.stage = 0 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>>, i32, i32, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 1x128x64>
        %c0_i32_1 = arith.constant 0 : i32
        nvws.aref.put.exit %1[%c0_i32_1], %token [#nvws.async_op<tma_load>] {loop.cluster = 2 : i32, loop.stage = 0 : i32} : <[!ttg.memdesc<1x128x64xf16, #shared, #smem, mutable>]>, !ttg.async.token
      } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize}
      nvws.warp_group.return
    }
    tt.return
  }
}
