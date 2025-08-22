// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-assign-stage-phase  -canonicalize | FileCheck %s

#shared0 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32} {

  //CHECK-LABEL: @two_consumers
  tt.func @two_consumers(%arg0: i32, %arg1: i32, %arg2: i32) {
    // CHECK:   [[C3:%.*]] = arith.constant 3 : i32
    // CHECK:   [[C1:%.*]] = arith.constant 1 : i32
    // CHECK:   [[C2:%.*]] = arith.constant 2 : i32
    // CHECK:   [[C0:%.*]] = arith.constant 0 : i32
    %ub = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
    // CHECK: [[AREF:%.*]] = nvws.aref.create
    %1 = nvws.aref.create %0 : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>
    // CHECK: [[IDX:%.*]]:6 = scf.for [[I:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[S0:%.*]] = [[C2]], [[P0:%.*]] = [[C0]], [[S1:%.*]] = [[C2]], [[P1:%.*]] = [[C1]], [[S2:%.*]] = [[C2]], [[P2:%.*]] = [[C1]])
    scf.for %arg3 = %arg0 to %arg1 step %arg2  : i32 {
      %2 = "op_a"() {ttg.partition = array<i32: 0>} : () -> tensor<1xi32, #blocked>
      // CHECK: op_a
      // CHECK-NEXT: [[S0a:%.*]] = arith.addi [[S0]], [[C1]]
      // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S0a]], [[C3]]
      // CHECK-NEXT: [[S0b:%.*]] = arith.select [[CMP]], [[C0]], [[S0a]]
      // CHECK-NEXT: [[P0a:%.*]] = arith.xori [[P0]], [[C1]]
      // CHECK-NEXT: [[P0b:%.*]] = arith.select [[CMP]], [[P0a]], [[P0]]
      // CHECK-NEXT: put.enter [[AREF]][[[S0b]], [[P0b]]]
      %buffers, %token = nvws.aref.put.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      ttg.local_store %2, %buffers {ttg.partition = array<i32: 0>} : tensor<1xi32, #blocked> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>
      // CHECK: put.exit [[AREF]][[[S0b]]]
      nvws.aref.put.exit %1[%c0_i32], %token [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token

      // CHECK-NEXT: [[S1a:%.*]] = arith.addi [[S1]], [[C1]]
      // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S1a]], [[C3]]
      // CHECK-NEXT: [[S1b:%.*]] = arith.select [[CMP]], [[C0]], [[S1a]]
      // CHECK-NEXT: [[P1a:%.*]] = arith.xori [[P1]], [[C1]]
      // CHECK-NEXT: [[P1b:%.*]] = arith.select [[CMP]], [[P1a]], [[P1]]
      // CHECK-NEXT: {{.*}}, [[TOK1:%.*]] = nvws.aref.get.enter [[AREF]][[[S1b]], [[P1b]]] {ttg.partition = array<i32: 1>}
      %buffers_0, %token_1 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %3 = ttg.local_load %buffers_0 {ttg.partition = array<i32: 1>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      // CHECK: get.exit [[AREF]][[[S1b]]], [[TOK1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
      nvws.aref.get.exit %1[%c0_i32], %token_1 [#nvws.async_op<none>] {ttg.partition = array<i32: 1>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_b"(%3) {ttg.partition = array<i32: 1>} : (tensor<1xi32, #blocked>) -> ()

      // CHECK: op_b
      // CHECK-NEXT: [[S2a:%.*]] = arith.addi [[S2]], [[C1]]
      // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S2a]], [[C3]]
      // CHECK-NEXT: [[S2b:%.*]] = arith.select [[CMP]], [[C0]], [[S2a]]
      // CHECK-NEXT: [[P2a:%.*]] = arith.xori [[P2]], [[C1]]
      // CHECK-NEXT: [[P2b:%.*]] = arith.select [[CMP]], [[P2a]], [[P2]]
      // CHECK-NEXT: {{.*}}, [[TOK2:%.*]] = nvws.aref.get.enter [[AREF]][[[S2b]], [[P2b]]] {ttg.partition = array<i32: 2>}
      %buffers_2, %token_3 = nvws.aref.get.enter %1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]> -> !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1>, !ttg.async.token
      %4 = ttg.local_load %buffers_2 {ttg.partition = array<i32: 2>} : !ttg.memdesc<1xi32, #shared, #smem, mutable, 1x1> -> tensor<1xi32, #blocked>
      // CHECK: get.exit [[AREF]][[[S2b]]], [[TOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      nvws.aref.get.exit %1[%c0_i32], %token_3 [#nvws.async_op<none>] {ttg.partition = array<i32: 2>} : <[!ttg.memdesc<3x1xi32, #shared, #smem, mutable>]>, !ttg.async.token
      "op_c"(%4) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
      "op_d"(%4) {ttg.partition = array<i32: 2>} : (tensor<1xi32, #blocked>) -> ()
      // CHECK: op_c
      // CHECK-NEXT: op_d
      // CHECK-NEXT: yield [[S0b]], [[P0b]], [[S1b]], [[P1b]], [[S2b]], [[P2b]]

    } {ttg.paArtition.stages = [0 : i32, 2 : i32, 2 : i32], ttg.warp_specialize.tag = 0 : i32}

    ttg.local_dealloc %0 : !ttg.memdesc<3x1xi32, #shared, #smem, mutable>
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
    // CHECK:   [[C3:%.*]] = arith.constant 3 : i32
    // CHECK:   [[C2:%.*]] = arith.constant 2 : i32
    // CHECK:   [[C0:%.*]] = arith.constant 0 : i32
    // CHECK:   [[C1:%.*]] = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %lb = arith.constant 0 : i32
    %ub = arith.constant 4 : i32

    // CHECK: [[AREF0:%.*]] = nvws.aref.create
    // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create
    %aref0 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    %aref1 = nvws.aref.create %d, %e : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>
    // CHECK: [[IDX:%.*]]:8 = scf.for [[I:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[S0:%.*]] = [[C2]], [[P0:%.*]] = [[C0]], [[S1:%.*]] = [[C2]], [[P1:%.*]] = [[C1]], [[S2:%.*]] = [[C2]], [[P2:%.*]] = [[C0]], [[S3:%.*]] = [[C2]], [[P3:%.*]] = [[C1]])
    scf.for %i = %lb to %ub step %c1_i32 : i32{
      // CHECK: [[S0a:%.*]] = arith.addi [[S0]], [[C1]]
      // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S0a]], [[C3]]
      // CHECK-NEXT: [[S0b:%.*]] = arith.select [[CMP]], [[C0]], [[S0a]]
      // CHECK-NEXT: [[P0a:%.*]] = arith.xori [[P0]], [[C1]]
      // CHECK-NEXT: [[P0b:%.*]] = arith.select [[CMP]], [[P0a]], [[P0]]
      // CHECK-NEXT: put.enter [[AREF0]][[[S0b]], [[P0b]]]
      %1:3 = nvws.aref.put.enter %aref0[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "op1"(%1#0) {ttg.partition = array<i32: 0>}: (!ttg.memdesc<64x16xf16, #shared0, #smem>) -> ()
      "op2"(%1#1)  {ttg.partition = array<i32: 0>} : (!ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      // CHECK: op2
      // CHECK-NEXT: put.exit [[AREF0]][[[S0b]]]
      nvws.aref.put.exit %aref0[%c0_i32], %1#2 [#nvws.async_op<tma_load>, #nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token


      // CHECK-NEXT: [[S1a:%.*]] = arith.addi [[S1]], [[C1]]
      // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S1a]], [[C3]]
      // CHECK-NEXT: [[S1b:%.*]] = arith.select [[CMP]], [[C0]], [[S1a]]
      // CHECK-NEXT: [[P1a:%.*]] = arith.xori [[P1]], [[C1]]
      // CHECK-NEXT: [[P1b:%.*]] = arith.select [[CMP]], [[P1a]], [[P1]]
      // CHECK-NEXT: {{.*}}, [[TOK1:%.*]] = nvws.aref.get.enter [[AREF0]][[[S1b]], [[P1b]]] {ttg.partition = array<i32: 1>}
      %2:3 = nvws.aref.get.enter %aref0[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
      "op3"(%2#0, %2#1) {ttg.partition = array<i32: 1>}: (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
      // CHECK: op3
      // CHECK-NEXT: get.exit [[AREF0]][[[S1b]]], [[TOK1]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      nvws.aref.get.exit %aref0[%c0_i32], %2#2 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
      // CHECK: [[IDX1:%.*]]:4 = scf.if
      scf.if %cond {
        // CHECK-NEXT: [[S2a:%.*]] = arith.addi [[S2]], [[C1]]
        // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S2a]], [[C3]]
        // CHECK-NEXT: [[S2b:%.*]] = arith.select [[CMP]], [[C0]], [[S2a]]
        // CHECK-NEXT: [[P2a:%.*]] = arith.xori [[P2]], [[C1]]
        // CHECK-NEXT: [[P2b:%.*]] = arith.select [[CMP]], [[P2a]], [[P2]]
        // CHECK-NEXT: {{.*}}, [[TOK2:%.*]] = nvws.aref.put.enter [[AREF1]][[[S2b]], [[P2b]]] {ttg.partition = array<i32: 0>}
        // CHECK-NEXT: op4
        // CHECK-NEXT: put.exit [[AREF1]][[[S2b]]]
        %4:3 = nvws.aref.put.enter %aref1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
        "op4"(%4#0, %4#1) {ttg.partition = array<i32: 0>} : (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
        nvws.aref.put.exit %aref1[%c0_i32], %4#2 [#nvws.async_op<none>] {ttg.partition = array<i32: 0>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
        // CHECK-NEXT: [[S3a:%.*]] = arith.addi [[S3]], [[C1]]
        // CHECK-NEXT: [[CMP:%.*]] = arith.cmpi eq, [[S3a]], [[C3]]
        // CHECK-NEXT: [[S3b:%.*]] = arith.select [[CMP]], [[C0]], [[S3a]]
        // CHECK-NEXT: [[P3a:%.*]] = arith.xori [[P3]], [[C1]]
        // CHECK-NEXT: [[P3b:%.*]] = arith.select [[CMP]], [[P3a]], [[P3]]
        // CHECK-NEXT: {{.*}}, [[TOK3:%.*]] = nvws.aref.get.enter [[AREF1]][[[S3b]], [[P3b]]] {ttg.partition = array<i32: 1>}
        // CHECK-NEXT: op5
        // CHECK-NEXT: get.exit [[AREF1]][[[S3b]]], [[TOK3]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
        %5:3 = nvws.aref.get.enter %aref1[%c0_i32, %c0_i32] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]> -> !ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>, !ttg.async.token
        "op5"(%5#0, %5#1) {ttg.partition = array<i32: 1>}: (!ttg.memdesc<64x16xf16, #shared0, #smem>, !ttg.memdesc<16x32xf16, #shared0, #smem>) -> ()
        nvws.aref.get.exit %aref1[%c0_i32], %5#2 [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>} : !nvws.aref<[!ttg.memdesc<3x64x16xf16, #shared0, #smem>, !ttg.memdesc<3x16x32xf16, #shared0, #smem>]>, !ttg.async.token
        // CHECK-NEXT: yield [[S2b]], [[P2b]], [[S3b]], [[P3b]]
      }
      // CHECK-NEXT: } else {
      // CHECK-NEXT: yield [[S2]], [[P2]], [[S3]], [[P3]]
      // CHECK-NEXT: }
      // CHECK: scf.yield [[S0b]], [[P0b]], [[S1b]], [[P1b]], [[IDX1]]#0, [[IDX1]]#1, [[IDX1]]#2, [[IDX1]]#3

    } {ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----


#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, unpacked = false>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @warp_specialize_tma_matmul
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    // CHECK: [[C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0:%.*]] = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %0 = nvws.aref.create %result : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: [[AREF:%.*]] = nvws.aref.create
    // CHECK-NEXT: {{.*}}, [[TOK:%.*]] = nvws.aref.put.enter [[AREF]][[[C0]], [[C1]]]
    %buffers, %token = nvws.aref.put.enter %0[%c0_i32, %c0_i32] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
    %1 = nvws.aref.buffer %0[%c0_i32], %token : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %2 = ttng.tmem_store %cst, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32  : i32 {
      %4 = arith.muli %arg5, %c64_i32 : i32
      %5 = tt.descriptor_load %arg3[%arg1, %4] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg4[%arg2, %4] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %9 = ttg.memdesc_trans %8 {order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // CHECK: nvws.aref.buffer [[AREF]][[[C0]]], [[TOK]]
      %10 = nvws.aref.buffer %0[%c0_i32], %token {ttg.partition = 1 : i32} : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
      %11 = ttng.tc_gen5_mma %7, %9, %10[], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    // CHECK: nvws.aref.put.exit [[AREF]][[[C0]]], [[TOK]]
    nvws.aref.put.exit %0[%c0_i32], %token [#nvws.async_op<tc5mma>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    // CHECK: {{.*}}, [[TOK:%.*]] = nvws.aref.get.enter [[AREF]][[[C0]], [[C0]]]
    %buffers_0, %token_1 = nvws.aref.get.enter %0[%c0_i32, %c0_i32] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>, !ttg.async.token
    // CHECK-NEXT: nvws.aref.buffer [[AREF]][[[C0]]], [[TOK]]
    %3 = nvws.aref.buffer %0[%c0_i32], %token_1 : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128>
    %result_2, %token_3 = ttng.tmem_load %3[] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 1x128x128> -> tensor<128x128xf32, #blocked>
    // CHECK: nvws.aref.get.exit [[AREF]][[[C0]]], [[TOK]]
    nvws.aref.get.exit %0[%c0_i32], %token_1 [#nvws.async_op<none>] : <[!ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    "use"(%result_2) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_unconditional_user
  tt.func @matmul_tma_acc_with_unconditional_user(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: !tt.tensordesc<tensor<64x128xf16, #shared>>) {
    // CHECK: [[C1:%.*]] = arith.constant 1 : i32
    // CHECK: [[C0:%.*]] = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %result = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[AREF:%.*]] = nvws.aref.create
    %0 = nvws.aref.create %result : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>
    // CHECK: {{.*}}, [[ATOK:%.*]] = nvws.aref.put.enter [[AREF]][[[C0]], [[C1]]]
    %buffers, %token = nvws.aref.put.enter %0[%c0_i32, %c0_i32] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
    %1 = nvws.aref.buffer %0[%c0_i32], %token : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    %2 = ttng.tmem_store %cst_0, %1[], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
    // CHECK: [[RET:%.*]]:5 = scf.for {{.*}} iter_args([[TOK:%.*]] = [[ATOK:%.*]], [[S0:%.*]] = [[C0]], [[P0:%.*]] = [[C1]], [[S1:%.*]] = [[C1]], [[P1:%.*]] = [[C1]])
    %3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %token) -> (!ttg.async.token)  : i32 {
      %4:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %5 = tt.descriptor_load %arg0[%4#0, %4#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %6 = tt.descriptor_load %arg1[%4#1, %4#2] {ttg.partition = 2 : i32} : !tt.tensordesc<tensor<64x128xf16, #shared>> -> tensor<64x128xf16, #blocked1>
      %7 = ttg.local_alloc %5 {ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {ttg.partition = 2 : i32} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: nvws.aref.buffer [[AREF]][[[S0]]
      %9 = nvws.aref.buffer %0[%c0_i32], %arg3 {ttg.partition = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %10 = ttng.tc_gen5_mma %7, %8, %9[], %true, %true {ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: nvws.aref.put.exit [[AREF]][[[S0]]], [[TOK]]
      nvws.aref.put.exit %0[%c0_i32], %arg3 [#nvws.async_op<tc5mma>] {ttg.partition = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token

      // CHECK: arith.addi
      // CHECK-NEXT: arith.cmpi eq
      // CHECK-NEXT: [[S1a:%.*]] = arith.select
      // CHECK-NEXT: arith.xori
      // CHECK-NEXT: [[P1a:%.*]] = arith.select
      // CHECK-NEXT: {{.*}}, [[TOK1:%.*]] = nvws.aref.get.enter [[AREF]][[[S1a]], [[P1a]]]
      %buffers_1, %token_2 = nvws.aref.get.enter %0[%c0_i32, %c0_i32] {ttg.partition = 0 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
      // CHECK-NEXT: nvws.aref.buffer [[AREF]][[[S1a]]], [[TOK1]]
      %11 = nvws.aref.buffer %0[%c0_i32], %token_2 {ttg.partition = 0 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %result_3, %token_4 = ttng.tmem_load %11[] {ttg.partition = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128> -> tensor<128x128xf32, #blocked>
      // CHECK: nvws.aref.get.exit [[AREF]][[[S1a]]], [[TOK1]]
      nvws.aref.get.exit %0[%c0_i32], %token_2 [#nvws.async_op<none>] {ttg.partition = 0 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      "acc_user"(%result_3) {ttg.partition = 0 : i32} : (tensor<128x128xf32, #blocked>) -> ()

      // CHECK: arith.addi
      // CHECK-NEXT: arith.cmpi eq
      // CHECK-NEXT: [[S0a:%.*]] = arith.select
      // CHECK-NEXT: arith.xori
      // CHECK-NEXT: [[P0a:%.*]] = arith.select
      // CHECK-NEXT: {{.*}}, [[TOK:%.*]] = nvws.aref.put.enter [[AREF]][[[S0a]], [[P0a]]]
      %buffers_5, %token_6 = nvws.aref.put.enter %0[%c0_i32, %c0_i32] {ttg.partition = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>, !ttg.async.token
      // CHECK-NEXT: aref.buffer [[AREF]][[[S0a]]]
      %12 = nvws.aref.buffer %0[%c0_i32], %token_6 {ttg.partition = 1 : i32} : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      %13 = ttng.tmem_store %cst, %12[], %true {ttg.partition = 1 : i32} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable, 2x128x128>
      // CHECK: scf.yield [[TOK]], [[S0a]], [[P0a]], [[S1a]], [[P1a]]
      scf.yield %token_6 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32}
    nvws.aref.put.exit %0[%c0_i32], %3 [#nvws.async_op<none>] : <[!ttg.memdesc<2x128x128xf32, #tmem, #ttng.tensor_memory, mutable>]>, !ttg.async.token
    tt.return
  }
}
