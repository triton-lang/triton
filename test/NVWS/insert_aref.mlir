// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-aref | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, unpacked = true>
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
      %2 = arith.muli %arg5, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      // CHECK: [[C_ZERO1:%.*]] = arith.constant {ttg.partition = 2 : i32} 0
      // CHECK: [[PUT_BUF1:%.*]], [[TOKEN1:%.*]] = nvws.aref.put.enter [[AREF1]][[[C_ZERO1]], [[C_ZERO1]]] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32}
      // CHECK-NEXT: nvws.descriptor_load {{.*}} 16384 [[PUT_BUF1]]
      // CHECK: [[C_ZERO2:%.*]] = arith.constant {ttg.partition = 2 : i32} 0
      // CHECK: nvws.aref.put.exit [[AREF1]][[[C_ZERO2]]], [[TOKEN1]] [#nvws.async_op<tma_load>] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32}
      // CHECK: [[C_ZERO3:%.*]] = arith.constant {ttg.partition = 1 : i32} 0
      // CHECK: [[GET_BUF1:%.*]], [[GET_TOKEN1:%.*]] = nvws.aref.get.enter [[AREF1]][[[C_ZERO3]], [[C_ZERO3]]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 1 : i32}
      %3 = tt.descriptor_load %arg3[%arg1, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      // CHECK: [[PUT_BUF2:%.*]], [[TOKEN2:%.*]] = nvws.aref.put.enter [[AREF2]]
      // CHECK-NEXT: nvws.descriptor_load {{.*}} 16384 [[PUT_BUF2]]
      // CHECK: nvws.aref.put.exit [[AREF2]]
      // CHECK: [[GET_BUF2:%.*]], [[GET_TOKEN2:%.*]] = nvws.aref.get.enter [[AREF2]]
      %4 = tt.descriptor_load %arg4[%arg2, %2] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>

      %5 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>

      // CHECK:  [[RHS:%.*]] = ttg.memdesc_trans [[GET_BUF2]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32}
      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // CHECK: ttng.tc_gen5_mma [[GET_BUF1]], [[RHS]], {{.*}}, {{.*}}, {{.*}}
      %8 = ttng.tc_gen5_mma %5, %7, %result[%arg6], %true, %true {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK: nvws.aref.get.exit [[AREF2]][{{.*}}], [[GET_TOKEN2]]
      // CHECK: [[C_ZERO4:%.*]] = arith.constant {ttg.partition = 1 : i32} 0
      // CHECK: nvws.aref.get.exit [[AREF1]][[[C_ZERO4]]], [[GET_TOKEN1]] [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 1 : i32}
      scf.yield %8 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
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
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 1 : i32, loop.stage = 0, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      // CHECK: {{.*}}, [[GET_TOKEN:%.*]] = nvws.aref.get.enter
      // CHECK: [[REG:%.*]] = ttg.local_load
      // CHECK: nvws.aref.get.exit {{.*}}, [[GET_TOKEN]] [#nvws.async_op<none>]
      // CHECK: "use"([[REG]])
      "use"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> ()
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @no_value_aref
  tt.func @no_value_aref(%arg0: tensor<128x64xf16, #blocked1>, %arg1: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    // CHECK-NOT: nvws.aref.create
    scf.for %arg2 = %c0_i32 to %arg1 step %c1_i32  : i32 {
      %0 = "producer"(%arg0, %arg2) {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 1 : i32} : (tensor<128x64xf16, #blocked1>, i32) -> tensor<128x64xf16, #blocked1>
      "use"(%0) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> ()
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
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
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %alloc = ttg.local_alloc %0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // CHECK-DAG: [[GET_BUF1:%.*]], [[GET_TOKEN1:%.*]] = nvws.aref.get.enter {{.*}} {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32}
      // CHECK-DAG: [[GET_BUF2:%.*]], [[GET_TOKEN2:%.*]] = nvws.aref.get.enter {{.*}} {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 1 : i32}
      // CHECK-DAG: [[REG:%.*]] = ttg.local_load [[GET_BUF1]] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32}
      // CHECK-DAG: nvws.aref.get.exit {{.*}}, [[GET_TOKEN1]] [#nvws.async_op<none>] {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32}
      // CHECK: "use1"([[REG]])
      // CHECK: "use2"([[GET_BUF2]])
      // CHECK: nvws.aref.get.exit {{.*}}, [[GET_TOKEN2]] [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 1 : i32}
      "use1"(%0) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%alloc) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 1 : i32} : (!ttg.memdesc<128x64xf16, #shared, #smem>) -> ()
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
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
      %0 = tt.descriptor_load %arg0[%arg2, %arg2] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = 1 : i32} : !tt.tensordesc<tensor<128x64xf16, #shared>> -> tensor<128x64xf16, #blocked1>
      %alloc = ttg.local_alloc %0 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 1 : i32} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // CHECK: [[GET_BUF:%.*]], [[GET_TOKEN:%.*]] = nvws.aref.get.enter {{.*}} {loop.cluster = 0 : i32, loop.stage = 1
      // CHECK: [[REG:%.*]] = ttg.local_load [[GET_BUF]] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32}
      // CHECK: "use1"([[REG]])
      // CHECK: "use2"([[GET_BUF]])
      // CHECK: nvws.aref.get.exit {{.*}}, [[GET_TOKEN]] {{.*}} {loop.cluster = 1 : i32, loop.stage = 1
      "use1"(%0) {loop.cluster = 1 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : (tensor<128x64xf16, #blocked1>) -> ()
      "use2"(%alloc) {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : (!ttg.memdesc<128x64xf16, #shared, #smem>) -> ()
    } {tt.num_stages = 2 : i32, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
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
      %1 = arith.muli %arg6, %c64_i32 {loop.cluster = 1 : i32, loop.stage = 0 : i32} : i32
      %2 = tt.descriptor_load %arg3[%arg1, %1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>> -> tensor<128x64xf8E4M3FN, #blocked1>
      %3 = tt.descriptor_load %arg4[%arg2, %1] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x64xf8E4M3FN, #shared3>> -> tensor<128x64xf8E4M3FN, #blocked1>
      %5 = ttg.local_alloc %2 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>
      %6 = ttg.local_alloc %3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>

      // CHECK: nvws.aref.put.enter
      // CHECK: nvws.descriptor_load
      // CHECK: nvws.aref.put.exit
      %4 = tt.descriptor_load %arg5[%arg1, %c0_i32] {loop.cluster = 1 : i32, loop.stage = 0 : i32, ttg.partition = 2 : i32} : !tt.tensordesc<tensor<128x8xi8, #shared2>> -> tensor<128x8xi8, #linear>

      // CHECK: nvws.aref.get.enter
      // CHECK: [[REG:%.*]] = ttg.local_load
      // CHECK: nvws.aref.get.exit
      // CHECK: tmem_alloc [[REG]]
      %result_1 = ttng.tmem_alloc %4 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 2 : i32} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>

      %7 = ttg.memdesc_trans %6 {loop.cluster = 0 : i32, loop.stage = 1 : i32, order = array<i32: 1, 0>, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>
      %result_2, %token = ttng.tmem_alloc %arg7 {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %8 = ttng.tc_gen5_mma_scaled %5, %7, %result_2[%token], %result, %result_1, %true, %true lhs = e4m3 rhs = e4m3 {loop.cluster = 0 : i32, loop.stage = 1 : i32, tt.self_latency = 1 : i32, ttg.partition = 1 : i32} : !ttg.memdesc<128x64xf8E4M3FN, #shared3, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      %result_3, %token_4 = ttng.tmem_load %result_2[%8] {loop.cluster = 0 : i32, loop.stage = 1 : i32, ttg.partition = 0 : i32} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      scf.yield %result_3 : tensor<128x128xf32, #blocked>
    } {tt.num_stages = 2 : i64, tt.scheduled_max_stage = 1 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
