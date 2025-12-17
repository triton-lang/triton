// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-aref | FileCheck %s

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
  // FUNC-LABEL: @warp_specialize_tma_matmul
  // CHECK: @warp_specialize_tma_matmul
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x64xf16, #shared>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    // CHECK: [[AREF_BUF1:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create [[AREF_BUF1]]
    // CHECK: [[AREF_BUF2:%.*]] = ttg.local_alloc
    // CHECK-NEXT: [[AREF2:%.*]] = nvws.aref.create [[AREF_BUF2]]
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>, loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      // CHECK: [[PUT_BUF1:%.*]], [[TOKEN1:%.*]] = nvws.aref.put.enter [[AREF1]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.descriptor_load {{.*}} 16384 [[PUT_BUF1]]
      // CHECK: nvws.aref.put.exit [[AREF1]], [[TOKEN1]] [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>}
      %3 = tt.descriptor_load %arg3[%arg1, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      // CHECK: [[PUT_BUF2:%.*]], [[TOKEN2:%.*]] = nvws.aref.put.enter [[AREF2]]
      // CHECK-NEXT: nvws.descriptor_load {{.*}} 16384 [[PUT_BUF2]]
      // CHECK: nvws.aref.put.exit [[AREF2]]
      %4 = tt.descriptor_load %arg4[%arg2, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>

      %5 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

      // CHECK: [[GET_BUF2:%.*]], [[GET_TOKEN2:%.*]] = nvws.aref.get.enter [[AREF2]]
      // CHECK:  [[RHS:%.*]] = ttg.memdesc_trans [[GET_BUF2]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>}
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // CHECK: [[GET_BUF1:%.*]], [[GET_TOKEN1:%.*]] = nvws.aref.get.enter [[AREF1]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma [[GET_BUF1]], [[RHS]], {{.*}}, {{.*}}, {{.*}}
      %8 = ttng.tc_gen5_mma %5, %7, %result[%arg6], %true, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.aref.get.exit [[AREF2]], [[GET_TOKEN2]]
      // CHECK: nvws.aref.get.exit [[AREF1]], [[GET_TOKEN1]] [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      scf.yield {ttg.partition = array<i32: 0, 1>} %8 : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @specialize_load_only
  tt.func @specialize_load_only(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // CHECK: nvws.aref.put.enter
      // CHECK: nvws.descriptor_load
      // CHECK: nvws.aref.put.exit
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 1 : i32, loop.stage = 0, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      // CHECK: {{.*}}, [[GET_TOKEN:%.*]] = nvws.aref.get.enter
      // CHECK: [[REG:%.*]] = ttg.local_load
      // CHECK: nvws.aref.get.exit {{.*}}, [[GET_TOKEN]] [#nvws.async_op<none>]
      // CHECK: "use"([[REG]])
      "use"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @no_value_aref
  tt.func @no_value_aref(%arg0: tensor<128x64xf16, #blocked1>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK-NOT: nvws.aref.create
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      %0 = "producer"(%arg0, %arg2) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>, i32) -> tensor<128x64xf16, #blocked1>
      "use"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 1>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @value_aref_multiple_producers
  tt.func @value_aref_multiple_producers(%arg0: tensor<128x64xf16, #blocked1>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK: nvws.aref.create
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      %0 = "producer"(%arg0, %arg2) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 0, 1>} : (tensor<128x64xf16, #blocked1>, i32) -> tensor<128x64xf16, #blocked1>
      // CHECK: [[VAL:%.*]] = "producer"
      // CHECK-NEXT: nvws.aref.put.enter
      // CHECK-NEXT: local_store
      // CHECK-NEXT: nvws.aref.put.exit
      // CHECK-NEXT: "use0"([[VAL]])
      // CHECK-NEXT: "use1"([[VAL]])
      // CHECK-NEXT: get.enter
      // CHECK-NEXT: [[VAL1:%.*]] = ttg.local_load
      // CHECK-NEXT: nvws.aref.get.exit
      // CHECK-NEXT: "use2"([[VAL1]])
      "use0"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use1"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @load_used_as_reg_and_smem
  tt.func @load_used_as_reg_and_smem(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // CHECK: nvws.aref.put.enter
      // CHECK: nvws.descriptor_load
      // CHECK: nvws.aref.put.exit
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %alloc = ttg.local_alloc %0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // CHECK-DAG: [[GET_BUF1:%.*]], [[GET_TOKEN1:%.*]] = nvws.aref.get.enter {{.*}} {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-DAG: [[REG:%.*]] = ttg.local_load [[GET_BUF1]] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK-DAG: nvws.aref.get.exit {{.*}}, [[GET_TOKEN1]] [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: "use1"([[REG]])
      // CHECK-DAG: [[GET_BUF2:%.*]], [[GET_TOKEN2:%.*]] = nvws.aref.get.enter {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      // CHECK: "use2"([[GET_BUF2]])
      // CHECK: nvws.aref.get.exit {{.*}}, [[GET_TOKEN2]] [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>}
      "use1"(%0) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%alloc) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (!ttg.memdesc<128x64xf16, #shared, #smem>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @load_used_as_reg_and_smem_same_partition
  tt.func @load_used_as_reg_and_smem_same_partition(%arg0: !tt.tensordesc<tensor<128x64xf16, #shared>>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      // CHECK: nvws.aref.put.enter
      // CHECK: nvws.descriptor_load
      // CHECK: nvws.aref.put.exit
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1>} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %alloc = ttg.local_alloc %0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // CHECK: [[GET_BUF:%.*]], [[GET_TOKEN:%.*]] = nvws.aref.get.enter {{.*}} {loop.cluster = 0 : i32, loop.stage = 1
      // CHECK: [[REG:%.*]] = ttg.local_load [[GET_BUF]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>}
      // CHECK: "use1"([[REG]])
      // CHECK: "use2"([[GET_BUF]])
      // CHECK: nvws.aref.get.exit {{.*}}, [[GET_TOKEN]] {{.*}} {loop.cluster = 1 : i32, loop.stage = 1
      "use1"(%0) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%alloc) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (!ttg.memdesc<128x64xf16, #shared, #smem>) -> ()
    } {ttg.partition = array<i32: 0, 1, 2>, tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @matmul_scaled_rhs_scales_tma
  tt.func @matmul_scaled_rhs_scales_tma(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>>, %arg4: !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>>, %arg5: !tt.tensordesc<tensor<128x8xi8, #shared2>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<127> : tensor<128x8xi8, #linear>
    %result = ttng.tmem_alloc %cst_0 : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %0 = scf.for %arg6 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg7 = %cst) -> (tensor<128x128xf32, #blocked>)  : i32 {
      %1 = arith.muli %arg6, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      %2 = tt.descriptor_load %arg3[%arg1, %1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>> -> tensor<128x64xf8E4M3FN, #blocked1>
      %3 = tt.descriptor_load %arg4[%arg2, %1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>> -> tensor<128x64xf8E4M3FN, #blocked1>
      %5 = ttg.local_alloc %2 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>
      %6 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>
      // CHECK: [[REG:%.*]] = tt.descriptor_load
      %4 = tt.descriptor_load %arg5[%arg1, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x8xi8, #shared2>> -> tensor<128x8xi8, #linear>
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

  // FUNC-LABEL: @local_alloc_default_partition
  // CHECK: @local_alloc_default_partition
  tt.func @local_alloc_default_partition(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<tensor<128x128xf16, #shared>>, %arg4: !tt.tensordesc<tensor<128x128xf16, #shared>>) {
    %true = arith.constant true
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    // CHECK: [[AREF_LHS_TRANS:%.*]] = nvws.aref.create {{.*}} : <[!ttg.memdesc<1x128x128xf16, #shared1, #smem, mutable>]>
    // CHECK: [[AREF_RHS:%.*]] = nvws.aref.create {{.*}} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    // CHECK: [[AREF_LHS:%.*]] = nvws.aref.create {{.*}} : <[!ttg.memdesc<1x128x128xf16, #shared, #smem, mutable>]>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c128_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
      // CHECK: [[AREF_LHS_PUT_BUF:%.*]], {{.*}} = nvws.aref.put.enter [[AREF_LHS]] {{.*}}ttg.partition = array<i32: 2>}
      // CHECK: nvws.descriptor_load {{.*}} 32768 [[AREF_LHS_PUT_BUF]] {{.*}}ttg.partition = array<i32: 2>}

      // CHECK: [[AREF_LHS_TRANS_PUT_BUF:%.*]], {{.*}} = nvws.aref.put.enter [[AREF_LHS_TRANS]] {{.*}}ttg.partition = array<i32: 0>}
      // CHECK: [[AREF_LHS_GET_BUF:%.*]], {{.*}} = nvws.aref.get.enter [[AREF_LHS]] {{.*}}ttg.partition = array<i32: 0>}
      // CHECK: [[TMA_RES_REG:%.*]] = ttg.local_load [[AREF_LHS_GET_BUF]] {{.*}}ttg.partition = array<i32: 0>}
      // CHECK: ttg.local_store [[TMA_RES_REG]], [[AREF_LHS_TRANS_PUT_BUF]] {{.*}}ttg.partition = array<i32: 0>}

      // CHECK: [[AREF_LHS_TRANS_GET_BUF:%.*]], {{.*}} = nvws.aref.get.enter [[AREF_LHS_TRANS]] {{.*}}ttg.partition = array<i32: 1>}
      // CHECK: [[LHS:%.*]] = ttg.memdesc_trans [[AREF_LHS_TRANS_GET_BUF]] {{.*}}ttg.partition = array<i32: 1>}

      %3 = tt.descriptor_load %arg3[%arg1, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked2>
      %5 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 0>} : (tensor<128x128xf16, #blocked2>) -> !ttg.memdesc<128x128xf16, #shared1, #smem>
      %lhs_trans = ttg.memdesc_trans %5 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared1, #smem> -> !ttg.memdesc<128x128xf16, #shared, #smem>

      %4 = tt.descriptor_load %arg4[%arg2, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf16, #shared>> -> tensor<128x128xf16, #blocked1>
      %6 = ttg.local_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf16, #blocked1>) -> !ttg.memdesc<128x128xf16, #shared, #smem>
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf16, #shared, #smem> -> !ttg.memdesc<128x128xf16, #shared1, #smem>

      // CHECK: ttng.tc_gen5_mma [[LHS]]
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
tt.func @two_consumers(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-NEXT: [[ABUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF:%.*]] = nvws.aref.create [[ABUF]]
  scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: [[VAL:%.*]] = "op_a"
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.put.enter [[AREF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttg.local_store [[VAL]], [[BUF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}

    "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> ()
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[VAL:%.*]] = ttg.local_load [[BUF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: "op_b"([[VAL]])

    "op_c"(%0) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: [[VAL:%.*]] = ttg.local_load [[BUF]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: "op_c"([[VAL]])
    // CHECK-NEXT: "op_d"([[VAL]])
    "op_d"(%0) {ttg.partition = array<i32: 2>} : (!ty) -> ()
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0, 2, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @distance_one
tt.func @distance_one(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: [[ABUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF:%.*]] = nvws.aref.create [[ABUF]]
  %cst = arith.constant dense<0> : !ty
  // CHECK: scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[K:%.*]] = {{.*}})
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.put.enter [[AREF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttg.local_store [[K]], [[BUF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: [[VAL:%.*]] = "op_a"
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[VAL:%.*]] = ttg.local_load [[BUF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: "op_b"([[VAL]])
    "op_b"(%k) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    scf.yield {ttg.partition = array<i32: 0, 1>} %0 : !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 0], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @different_yield_partition
tt.func @different_yield_partition(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: [[ABUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF:%.*]] = nvws.aref.create [[ABUF]]
  %cst = arith.constant dense<0> : !ty
  // CHECK: scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[K:%.*]] = {{.*}})
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK-NEXT: [[VAL:%.*]] = "op_a"
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.put.enter [[AREF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttg.local_store [[VAL]], [[BUF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: "op_b"([[K]])
    "op_b"(%k) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[VAL:%.*]] = ttg.local_load [[BUF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: scf.yield {ttg.partition = array<i32: 0, 1>} [[VAL]]

    scf.yield {ttg.partition = array<i32: 0, 1>} %0 : !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 0], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

tt.func @complex_case(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: [[ABUF1:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create [[ABUF1]]
  // CHECK-NEXT: [[ABUF2:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF2:%.*]] = nvws.aref.create [[ABUF2]]
  %cst = arith.constant dense<0> : !ty
  // CHECK: scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[K:%.*]] = {{.*}}, [[L:%.*]] = {{.*}})
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst, %l = %cst) -> (!ty, !ty) : i32 {
    // CHECK: [[BUF:%.*]], [[TOKEN2:%.*]] = nvws.aref.put.enter [[AREF2]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttg.local_store [[L]], [[BUF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF2]], [[TOKEN2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN1:%.*]] = nvws.aref.put.enter [[AREF1]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: ttg.local_store [[K]], [[BUF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF1]], [[TOKEN1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}

    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK-NEXT: op_a
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF1]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[K1:%.*]] = ttg.local_load [[BUF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF1]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: "op_b"([[K1]])
    "op_b"(%k) {ttg.partition = array<i32: 1>} : (!ty) -> ()


    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF1]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: [[K2:%.*]] = ttg.local_load [[BUF]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF1]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: "op_c"([[K2]])
    // CHECK-NEXT: "op_c"([[K2]])
    "op_c"(%k) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    "op_c"(%k) {ttg.partition = array<i32: 2>} : (!ty) -> ()

    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF2]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[L1:%.*]] = ttg.local_load [[BUF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF2]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: "op_d"([[L1]])
    "op_d"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF2]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: [[L2:%.*]] = ttg.local_load [[BUF]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF2]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: "op_d"([[L2]])
    "op_d"(%l) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.partition.stages = [0, 2, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @reuse_argument
tt.func @reuse_argument(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-DAG: [[CST0:%.*]] = arith.constant dense<0>
  // CHECK-DAG: [[CST1:%.*]] = arith.constant dense<1>
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty

  // CHECK: local_alloc
  // CHECK-NEXT: [[AREF:%.*]] = nvws.aref.create
  // CHECK-NEXT: scf.for
  scf.for %i = %lb to %ub step %step iter_args(%k = %cst0, %l = %cst1) -> (!ty, !ty) : i32 {
    // CHECK-NEXT: {{.*}}, [[TOKEN:%.*]] = nvws.aref.put.enter [[AREF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: local_store
    // CHECK-NEXT: nvws.aref.put.exit [[AREF]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: op_a
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

    // CHECK-NEXT: aref.get.enter [[AREF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: local_load {{.*}} {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: aref.get.exit [[AREF]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: op_d
    "op_d"(%l) {ttg.partition = array<i32: 1>} : (!ty) -> ()

    // CHECK-NEXT: aref.get.enter [[AREF]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: local_load {{.*}} {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: aref.get.exit [[AREF]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: op_d
    "op_d"(%l) {ttg.partition = array<i32: 2>} : (!ty) -> ()
    scf.yield %0, %k : !ty, !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>], ttg.partition.stages = [1, 0, 0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @multiplicity_branch
tt.func @multiplicity_branch(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-DAG: [[CST0:%.*]] = arith.constant dense<0>
  // CHECK-DAG: [[CST1:%.*]] = arith.constant dense<1>
  // CHECK-DAG: [[CST2:%.*]] = arith.constant dense<2>
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty
  %cst2 = arith.constant dense<2> : !ty

  // CHECK: local_alloc
  // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create
  // CHECK-NEXT: local_alloc
  // CHECK-NEXT: [[AREF2:%.*]] = nvws.aref.create
  // CHECK-NEXT: local_alloc
  // CHECK-NEXT: [[AREF3:%.*]] = nvws.aref.create

  // CHECK: scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[A:%.*]] = {{.*}}, [[B:%.*]] = {{.*}}, [[C:%.*]] = {{.*}})
  scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN3:%.*]] = nvws.aref.put.enter [[AREF3]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: local_store [[C]], [[BUF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF3]], [[TOKEN3]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN2:%.*]] = nvws.aref.put.enter [[AREF2]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: local_store [[B]], [[BUF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF2]], [[TOKEN2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN1:%.*]] = nvws.aref.put.enter [[AREF1]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: local_store [[A]], [[BUF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF1]], [[TOKEN1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: op_a
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

    // CHECK: aref.get.enter [[AREF1]]
    // CHECK-NEXT: local_load
    // CHECK-NEXT: aref.get.exit [[AREF1]]
    // CHECK-NEXT: op_b
    "op_b"(%a) {ttg.partition = array<i32: 1>}: (!ty) -> ()

    // CHECK: aref.get.enter [[AREF2]]
    // CHECK-NEXT: local_load
    // CHECK-NEXT: aref.get.exit [[AREF2]]
    // CHECK-NEXT: op_c
    "op_c"(%b) {ttg.partition = array<i32: 2>}: (!ty) -> ()

    // CHECK: aref.get.enter [[AREF3]]
    // CHECK-NEXT: local_load
    // CHECK-NEXT: aref.get.exit [[AREF3]]
    // CHECK-NEXT: op_d
    "op_d"(%c) {ttg.partition = array<i32: 3>}: (!ty) -> ()

    scf.yield %0, %a, %a : !ty, !ty, !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 0, 0, 0], ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @multiplicity_branch2
tt.func @multiplicity_branch2(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-DAG: [[CST0:%.*]] = arith.constant dense<0>
  // CHECK-DAG: [[CST1:%.*]] = arith.constant dense<1>
  // CHECK-DAG: [[CST2:%.*]] = arith.constant dense<2>
  %cst0 = arith.constant dense<0> : !ty
  %cst1 = arith.constant dense<1> : !ty
  %cst2 = arith.constant dense<2> : !ty

  // CHECK: local_alloc
  // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create
  // CHECK-NEXT: local_alloc
  // CHECK-NEXT: [[AREF2:%.*]] = nvws.aref.create
  // CHECK-NEXT: local_alloc
  // CHECK-NEXT: [[AREF3:%.*]] = nvws.aref.create

  // CHECK: scf.for [[IV:%.*]] = [[LB:%.*]] to [[UB:%.*]] step [[STEP:%.*]] iter_args([[A:%.*]] = {{.*}}, [[B:%.*]] = {{.*}}, [[C:%.*]] = {{.*}})
  scf.for %i = %lb to %ub step %step iter_args(%a = %cst0, %b = %cst1, %c = %cst2) -> (!ty, !ty, !ty) : i32 {
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN3:%.*]] = nvws.aref.put.enter [[AREF3]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: local_store [[C]], [[BUF]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF3]], [[TOKEN3]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN2:%.*]] = nvws.aref.put.enter [[AREF2]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: local_store [[B]], [[BUF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF2]], [[TOKEN2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN1:%.*]] = nvws.aref.put.enter [[AREF1]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: local_store [[A]], [[BUF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF1]], [[TOKEN1]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: op_a
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty

    // CHECK: aref.get.enter [[AREF1]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[A1:%.*]] = ttg.local_load {{.*}} {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: aref.get.exit [[AREF1]]
    // CHECK-NEXT: "op_b"([[A1]]) {ttg.partition = array<i32: 1>}
    %d = "op_b"(%a) {ttg.partition = array<i32: 1>}: (!ty) -> !ty

    // CHECK: aref.get.enter [[AREF2]] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: [[B1:%.*]] = ttg.local_load {{.*}} {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: aref.get.exit [[AREF2]]
    // CHECK-NEXT: "op_c"([[B1]]) {ttg.partition = array<i32: 2>}
    %e = "op_c"(%b) {ttg.partition = array<i32: 2>}: (!ty) -> !ty

    // CHECK: aref.get.enter [[AREF3]] {ttg.partition = array<i32: 3>}
    // CHECK-NEXT: [[C1:%.*]] = ttg.local_load {{.*}} {ttg.partition = array<i32: 3>}
    // CHECK-NEXT: aref.get.exit [[AREF3]]
    // CHECK-NEXT: "op_d"([[C1]]) {ttg.partition = array<i32: 3>}
    "op_d"(%c) {ttg.partition = array<i32: 3>}: (!ty) -> ()

    scf.yield %0, %d, %e : !ty, !ty, !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>, array<i32: 2>], ttg.partition.stages = [0, 0, 0, 0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @self_recursion
tt.func @self_recursion(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-NOT: nvws.aref.create
  %cst = arith.constant dense<0> : !ty
  // CHECK: iter_args([[ARG:%arg[0-9]+]] = %cst)
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    // CHECK-NEXT: [[OUT:%.*]] = "op_a"([[ARG]])
    %0 = "op_a"(%k) {ttg.partition = array<i32: 0>} : (!ty) -> !ty
    // CHECK: yield [[OUT]]
    scf.yield %0 : !ty
  } {tt.warp_specialize, ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>], ttg.partition.stages = [0], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @self_recursion_and_use
tt.func @self_recursion_and_use(%lb: i32, %ub: i32, %step: i32) {
  %cst = arith.constant dense<0> : !ty
  %0 = scf.for %i = %lb to %ub step %step iter_args(%k = %cst) -> (!ty) : i32 {
    %0 = "op_a"(%k) {ttg.partition = array<i32: 0>} : (!ty) -> !ty
    // CHECK: "op_a"
    // CHECK-NEXT: nvws.aref.put.enter
    // CHECK-NEXT: local_store
    // CHECK-NEXT: nvws.aref.put.exit

    "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
    // CHECK-NEXT: nvws.aref.get.enter
    // CHECK-NEXT: ttg.local_load
    // CHECK-NEXT: nvws.aref.get.exit
    // CHECK-NEXT: "op_b"

    scf.yield %0 : !ty
  } {tt.warp_specialize, ttg.partition.stages = [0, 1], ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

// CHECK-LABEL: @conditional_consumer
tt.func @conditional_consumer(%lb: i32, %ub: i32, %step: i32) {
  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "producer"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: "producer"
    // CHECK-NEXT: nvws.aref.put.enter
    // CHECK-NEXT: local_store
    // CHECK-NEXT: nvws.aref.put.exit
    %cond = "rand"() {ttg.partition = array<i32: 1>} : () -> i1
    // CHECK-NEXT: "rand"
    // CHECK-NEXT: nvws.aref.get.enter
    // CHECK-NEXT: [[VALUE:%.*]] = ttg.local_load
    // CHECK-NEXT: nvws.aref.get.exit{{.*}}, {{.*}}
    // CHECK-NEXT: scf.if
    %1 = scf.if %cond -> !ty {
      // CHECK-NEXT: "something"
      "something"() {ttg.partition = array<i32: 1>} : () -> ()
      // CHECK-NEXT: yield {{.*}} [[VALUE]]
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
tt.func @no_def_op(%lb: i32, %ub: i32, %step: i32) {
  %c0_i32 = arith.constant 0 : i32
  // CHECK: scf.for
  scf.for %i = %lb to %ub step %step iter_args(%k = %c0_i32) -> i32 : i32 {
    // CHECK-NEXT: put.enter
    // CHECK-NEXT: splat
    // CHECK-NEXT: local_store
    // CHECK-NEXT: put.exit
    // CHECK-NEXT: get.enter
    // CHECK-NEXT: local_load
    // CHECK-NEXT: get.exit
    // CHECK-NEXT: [[VAL:%.*]] = tt.unsplat
    // CHECK-NEXT: addi [[VAL]], [[VAL]]
    arith.addi %k, %k {ttg.partition = array<i32: 1>} : i32
    scf.yield {ttg.partition = array<i32: 0>} %k : i32
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 0>]}
  tt.return
}

// CHECK-LABEL: @scalar_consumers
tt.func @scalar_consumers(%lb: i32, %ub: i32, %step: i32) {
  // CHECK-NEXT: [[ABUF:%.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi32, {{.*}}>
  // CHECK-NEXT: [[AREF:%.*]] = nvws.aref.create [[ABUF]]
  scf.for %i = %lb to %ub step %step iter_args() -> () : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> i32
    // CHECK: [[VAL:%.*]] = "op_a"
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.put.enter [[AREF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: [[VAL_TENSOR:%.*]] = tt.splat [[VAL]] {ttg.partition = array<i32: 0>} : i32 -> tensor<1xi32, #blocked>
    // CHECK-NEXT: ttg.local_store [[VAL_TENSOR]], [[BUF]] {ttg.partition = array<i32: 0>}
    // CHECK-NEXT: nvws.aref.put.exit [[AREF]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}

    "op_b"(%0) {ttg.partition = array<i32: 1>} : (i32) -> ()
    // CHECK-NEXT: [[BUF:%.*]], [[TOKEN:%.*]] = nvws.aref.get.enter [[AREF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[VAL:%.*]] = ttg.local_load [[BUF]] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: nvws.aref.get.exit [[AREF]], [[TOKEN]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: [[VAL_SCALAR:%.*]] = tt.unsplat [[VAL]] {ttg.partition = array<i32: 1>} : tensor<1xi32, #blocked>
    // CHECK-NEXT: "op_b"([[VAL_SCALAR]])

  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1>, ttg.partition.stages = [0, 2], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}


}
// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
!ty = tensor<1xi32, #blocked>

module attributes {"ttg.num-warps" = 4 : i32} {

tt.func @cycle_in_partition(%lb: i32, %ub: i32, %step: i32) {
  // CHECK: ttg.local_alloc
  // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create
  // CHECK-NEXT: ttg.local_alloc
  // CHECK-NEXT: [[AREF2:%.*]] = nvws.aref.create

  scf.for %i = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: "op_a"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF1]] {ttg.partition = array<i32: 0>}

    %1 = "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
    // CHECK: nvws.aref.get.exit [[AREF1]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: "op_b"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF2]] {ttg.partition = array<i32: 1>}

    // CHECK: nvws.aref.get.exit [[AREF2]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}

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
  // CHECK: ttg.local_alloc
  // CHECK-NEXT: [[AREF1:%.*]] = nvws.aref.create
  // CHECK-NEXT: ttg.local_alloc
  // CHECK-NEXT: [[AREF2:%.*]] = nvws.aref.create
  // CHECK-NEXT: ttg.local_alloc
  // CHECK-NEXT: [[AREF3:%.*]] = nvws.aref.create
  scf.for %j = %lb to %ub step %step : i32 {
    %0 = "op_a"() {ttg.partition = array<i32: 0>} : () -> !ty
    // CHECK: "op_a"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF1]] {ttg.partition = array<i32: 0>}

    %1 = "op_b"(%0) {ttg.partition = array<i32: 1>} : (!ty) -> !ty
    // CHECK: nvws.aref.get.exit [[AREF1]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 1>}
    // CHECK-NEXT: "op_b"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF2]] {ttg.partition = array<i32: 1>}

    %2 = "op_c"(%1) {ttg.partition = array<i32: 2>} : (!ty) -> !ty
    // CHECK: nvws.aref.get.exit [[AREF2]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
    // CHECK-NEXT: "op_c"
    // CHECK-NEXT: nvws.aref.put.enter [[AREF3]] {ttg.partition = array<i32: 2>}

    "op_c"(%2) {ttg.partition = array<i32: 0>} : (!ty) -> ()
    // CHECK: nvws.aref.get.exit [[AREF3]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
    // CHECK: "op_c"
    scf.yield
  } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0, 2, 3], ttg.warp_specialize.tag = 0 : i32}
  tt.return
}

}


// -----

// CHECK-LABEL: @inner_loop_fixed_operand
#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @inner_loop_fixed_operand(%arg0: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg1: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg2: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    // CHECK-COUNT-2: nvws.aref.create
    // CHECK: scf.for
    // CHECK: nvws.aref.put.enter
    // CHECK: nvws.descriptor_load
    // CHECK: nvws.aref.put.exit {{.*}}, {{.*}} [#nvws.async_op<tma_load>]
    // CHECK: [[LHS:%.*]], {{.*}} = nvws.aref.get.enter
    // CHECK: scf.for
    // CHECK: nvws.aref.put.enter
    // CHECK: nvws.descriptor_load
    // CHECK: nvws.aref.put.exit {{.*}}, {{.*}} [#nvws.async_op<tma_load>]
    // CHECK: [[RHS:%.*]], {{.*}} = nvws.aref.get.enter
    // CHECK: [[RHS_TRANS:%.*]] = ttg.memdesc_trans [[RHS]]
    // CHECK: ttng.tc_gen5_mma [[LHS]], [[RHS_TRANS]]
    // CHECL: }
    // CHECK: nvws.aref.get.exit {{.*}}, {{.*}} [#nvws.async_op<tc5mma>]
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
      %17 = tt.descriptor_load %arg0[%15, %c0_i32] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
      %18 = ttg.local_alloc %17 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
      %19:2 = scf.for %arg8 = %c0_i32 to %3 step %c1_i32 iter_args(%arg9 = %false, %arg10 = %arg7) -> (i1, !ttg.async.token)  : i32 {
        %22 = arith.muli %arg8, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        %23 = tt.descriptor_load %arg1[%16, %22] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
        %24 = ttg.local_alloc %23 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %25 = ttg.memdesc_trans %24 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
        %26 = ttng.tc_gen5_mma %18, %25, %result[%arg10], %arg9, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %26 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1>]}
      %result_0, %token_1 = ttng.tmem_load %result[%19#1] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      %20 = tt.fp_to_fp %result_0 {ttg.partition = array<i32: 0>}, rounding = rtne : tensor<128x128xf32, #blocked> -> tensor<128x128xf8E4M3FN, #blocked>
      %21 = ttg.convert_layout %20 {ttg.partition = array<i32: 0>} : tensor<128x128xf8E4M3FN, #blocked> -> tensor<128x128xf8E4M3FN, #blocked1>
      tt.descriptor_store %arg2[%15, %16], %21 {ttg.partition = array<i32: 0>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, tensor<128x128xf8E4M3FN, #blocked1>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %token_1 : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
