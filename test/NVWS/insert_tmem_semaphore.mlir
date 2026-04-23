// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --nvws-insert-tmem-semaphore -cse | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[1, 0], [2, 0], [0, 32], [0, 64], [4, 0]], lane = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [4, 3, 2, 1, 0]}>
#shared4 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#shared5 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, fp4Padded = true, rank = 3}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {

  // CHECK-LABEL: @warp_specialize_tma_matmul
  tt.func @warp_specialize_tma_matmul(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf16, #shared>, %arg4: !tt.tensordesc<128x64xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    // Single-buffered (1x): alloc, create semaphores, initial acquire+store
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[TOK2:%.*]] = scf.for {{.*}} iter_args([[TOK:%.*]] = [[ATOK]])
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %3 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %7 = ttg.memdesc_trans %6 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // Buffer from EMPTY sem used in MMA
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[TOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF]][]{{.*}} {ttg.partition = array<i32: 1>}
      %8 = ttng.tc_gen5_mma %5, %7, %result[%arg6], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %8 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop: release FULL, acquire FULL for load, load, release EMPTY
    // CHECK: nvws.semaphore.release [[FULL]], [[TOK2]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: [[ATOK2:%.*]] = nvws.semaphore.acquire [[FULL]]
    // CHECK-NEXT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[ATOK2]]
    // CHECK-NEXT: ttng.tmem_load [[BUF2]][]
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[ATOK2]] [#nvws.async_op<none>]
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

// CHECK-LABEL: @matmul_tma_acc_with_unconditional_user
  tt.func @matmul_tma_acc_with_unconditional_user(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Double-buffered (2x): alloc, create semaphores, initial acquire+store
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst_0, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[TOK2:%.*]] = scf.for {{.*}} iter_args([[TOK:%.*]] = [[ATOK]])
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA uses buffer from EMPTY sem, then release FULL
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[TOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF]][]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[TOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // Consumer: acquire FULL, buffer, load, release EMPTY
      // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load [[BUF2]][] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "acc_user"
      %result_1, %token_2 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "acc_user"(%result_1) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // Re-acquire EMPTY for next store
      // CHECK: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[BUF3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF3]][]{{.*}} {ttg.partition = array<i32: 1>}
      %8 = ttng.tmem_store %cst, %result[%token_2], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %8 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 4 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop: release FULL, acquire FULL, release EMPTY
    // CHECK: nvws.semaphore.release [[FULL]], [[TOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 4 : i32}
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 4 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 4 : i32}
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_user
  tt.func @matmul_tma_acc_with_conditional_user(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<1.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Double-buffered (2x): alloc, create, initial acquire+store
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst_0, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[TOK2:%.*]] = scf.for {{.*}} iter_args([[TOK:%.*]] = [[ATOK]])
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA uses buffer from EMPTY sem
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[TOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF]][]{{.*}} {ttg.partition = array<i32: 1>}
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>}: i32
      // Conditional release FULL
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], [[TOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // Conditional consumer: acquire FULL, buffer, load, release EMPTY
      // CHECK: scf.if
      // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load [[BUF2]][] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "acc_user"
      %9 = scf.if %8 -> (!ttg.async.token) {
        %result_1, %token_2 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_1) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield %token_2 : !ttg.async.token
      } else {
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // Re-acquire EMPTY via conditional if
      // CHECK: [[TOK1:%.*]] = scf.if
      // CHECK-NEXT: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK: [[BUF3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[TOK1]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF3]][]
      %10 = ttng.tmem_store %cst, %result[%9], %true {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %10 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 5 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs= [array<i32: 1>]}
    // After loop: release FULL, acquire FULL, release EMPTY
    // CHECK: nvws.semaphore.release [[FULL]], [[TOK2]] [#nvws.async_op<none>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 5 : i32}
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 5 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 5 : i32}
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_def
  tt.func @matmul_tma_acc_with_conditional_def(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Double-buffered: alloc, create, initial acquire+store
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.for
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA uses buffer, then release
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF]][]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<none>]
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 : i32
      // Consumer: acquire FULL, buffer, load, release EMPTY
      // CHECK: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load [[BUF2]][] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "acc_user"
      %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // Re-acquire EMPTY, buffer, conditional store
      // CHECK: nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.buffer [[EMPTY]]
      // CHECK: ttng.tmem_store
      %9 = ttng.tmem_store %cst, %result[%token_1], %8 {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %9 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 6 : i32}
    // After loop
    // CHECK: nvws.semaphore.release [[FULL]]
    // CHECK: nvws.semaphore.acquire [[FULL]]
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_def_and_use
  tt.func @matmul_tma_acc_with_conditional_def_and_use(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Double-buffered: alloc, create, initial acquire+store
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.for
    %1 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %0) -> (!ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA uses buffer
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF]][]{{.*}} {ttg.partition = array<i32: 1>}
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg3], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>}: i32
      // Conditional release FULL
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // Conditional consumer: acquire FULL, buffer, load, release EMPTY
      // CHECK: scf.if
      // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load [[BUF2]][] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "acc_user"
      %9 = scf.if %8 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield %token_1 : !ttg.async.token
      } else {
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // Conditional re-acquire EMPTY, buffer, store
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.buffer [[EMPTY]]
      // CHECK: ttng.tmem_store
      %10 = ttng.tmem_store %cst, %result[%9], %8 {ttg.partition = array<i32: 1>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %10 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 7 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop
    // CHECK: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 7 : i32}
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 7 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 7 : i32}
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_def_and_use_no_multibuf_flag
  tt.func @matmul_tma_acc_with_conditional_def_and_use_no_multibuf_flag(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Single-buffered (1x) because of tt.disallow_acc_multi_buffer
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.for
    %1:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0) -> (i1, !ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // MMA uses buffer
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF]][]{{.*}} {ttg.partition = array<i32: 1>}
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg4], %arg3, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>}: i32
      %9 = arith.cmpi ne, %arg2, %c0_i32 {ttg.partition = array<i32: 1>} : i32
      // Conditional release FULL
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // Conditional consumer: some_op, acquire FULL, buffer, load, release EMPTY
      // CHECK: scf.if
      // CHECK-NEXT: "some_op"
      // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load [[BUF2]][] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "acc_user"
      %10 = scf.if %8 -> (!ttg.async.token) {
        "some_op"() {ttg.partition = array<i32: 0>} : () -> ()
        %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield %token_1 : !ttg.async.token
      } else {
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // Conditional re-acquire EMPTY
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      scf.yield %9, %10 : i1, !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>], tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 8 : i32}
    // After loop
    // CHECK: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 8 : i32}
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 8 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 8 : i32}
    tt.return
  }

  // CHECK-LABEL: @matmul_scaled_rhs_scales_tma
  tt.func @matmul_scaled_rhs_scales_tma(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf8E4M3FN, #shared2>, %arg4: !tt.tensordesc<128x64xf8E4M3FN, #shared2>, %arg5: !tt.tensordesc<128x8xi8, #shared3>) {
    %cst = arith.constant dense<127> : tensor<128x8xi8, #linear>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    // LHS scales (no semaphore - static)
    // CHECK: tmem_alloc {{.*}} !ttg.memdesc<128x8xi8, #tmem_scales
    %result = ttng.tmem_alloc %cst : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    // ACC buffer: alloc, create, initial acquire+store
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %result_1, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst_0, %result_1[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // RHS scales: alloc + semaphore pair
    // CHECK: [[SBUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x8xi8,
    // CHECK-NEXT: [[SEMPTY:%.*]] = nvws.semaphore.create [[SBUF]] true
    // CHECK-NEXT: [[SFULL:%.*]] = nvws.semaphore.create [[SBUF]] false
    // CHECK: scf.for
    %1 = scf.for %arg6 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg7 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg6, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %3 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf8E4M3FN, #shared2> -> tensor<128x64xf8E4M3FN, #blocked1>
      %4 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf8E4M3FN, #shared2> -> tensor<128x64xf8E4M3FN, #blocked1>
      %5 = tt.descriptor_load %arg5[%arg1, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x8xi8, #shared3> -> tensor<128x8xi8, #linear>
      %6 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>
      %7 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>
      %8 = ttg.memdesc_trans %7 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem> -> !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>
      // RHS scales: acquire SEMPTY, buffer, store, release SFULL
      // CHECK: [[STOK:%.*]] = nvws.semaphore.acquire [[SEMPTY]] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[SBUF2:%.*]] = nvws.semaphore.buffer [[SEMPTY]], [[STOK]] {ttg.partition = array<i32: 2>}
      // CHECK: ttng.tmem_store {{.*}}, [[SBUF2]]{{.*}} {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: nvws.semaphore.release [[SFULL]], [[STOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 2>}
      %result_2 = ttng.tmem_alloc %5 {ttg.partition = array<i32: 2>} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      // ACC buffer + RHS scales buffer for MMA
      // CHECK: [[BUF3:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[STOK2:%.*]] = nvws.semaphore.acquire [[SFULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[SBUF3:%.*]] = nvws.semaphore.buffer [[SFULL]], [[STOK2]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma_scaled {{.*}}, {{.*}}, [[BUF3]][], {{.*}}, [[SBUF3]]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[SEMPTY]], [[STOK2]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      %9 = ttng.tc_gen5_mma_scaled %6, %8, %result_1[%arg7], %result, %result_2, %true, %true lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf8E4M3FN, #shared2, #smem>, !ttg.memdesc<64x128xf8E4M3FN, #shared4, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %9 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 9 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop: release/acquire/buffer/load/release for ACC
    // CHECK: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 9 : i32}
    // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]]
    // CHECK-NEXT: [[BUF4:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]]
    // CHECK-NEXT: ttng.tmem_load [[BUF4]][]
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>]
    %val, %tok = ttng.tmem_load %result_1[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%val) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @user_partition_has_cycle
  tt.func @user_partition_has_cycle(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf16, #shared>, %arg4: !tt.tensordesc<128x64xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    %0 = tt.descriptor_load %arg3[%c0_i32, %c0_i32] : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
    %1 = ttg.local_alloc %0 : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    // Double-buffered: producer/consumer cycle in loop
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: scf.for
    %2:2 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %cst, %arg7 = %token) -> (tensor<128x128xf32, #blocked>, !ttg.async.token)  : i32 {
      %3 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %4 = tt.descriptor_load %arg4[%arg2, %3] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %5 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.memdesc_trans %5 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF]][]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      %7 = ttng.tc_gen5_mma %1, %6, %result[%arg7], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: arith.addf
      %8 = arith.addf %arg6, %arg6 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked>
      // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load [[BUF2]][] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK-NEXT: arith.mulf
      // CHECK-NEXT: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      %9 = arith.mulf %8, %result_0 {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked>
      scf.yield %9, %token_1 : tensor<128x128xf32, #blocked>, !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 11 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>, array<i32: 1>]}
    // After loop
    // CHECK: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 11 : i32}
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 11 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 11 : i32}
    "use"(%2#0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @matmul_tma_acc_with_conditional_def_and_use_flag
  tt.func @matmul_tma_acc_with_conditional_def_and_use_flag(%arg0: !tt.tensordesc<128x64xf16, #shared>, %arg1: !tt.tensordesc<64x128xf16, #shared>) {
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Double-buffered with use_d flag
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.for
    %1:2 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %0) -> (i1, !ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {ttg.partition = array<i32: 2>} : (i32) -> (i32, i32, i32)
      %3 = tt.descriptor_load %arg0[%2#0, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg1[%2#1, %2#2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %5 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF]][]{{.*}} {ttg.partition = array<i32: 1>}
      %7 = ttng.tc_gen5_mma %5, %6, %result[%arg4], %arg3, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %8 = arith.cmpi eq, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      %9 = arith.cmpi ne, %arg2, %c0_i32 {ttg.partition = array<i32: 0, 1>} : i32
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK: scf.if
      // CHECK-NEXT: "some_op"
      // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load [[BUF2]][] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "acc_user"
      %10 = scf.if %8 -> (!ttg.async.token) {
        "some_op"() {ttg.partition = array<i32: 0>} : () -> ()
        %result_0, %token_1 = ttng.tmem_load %result[%7] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield %token_1 : !ttg.async.token
      } else {
        scf.yield %7 : !ttg.async.token
      } {ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      scf.yield %9, %10 : i1, !ttg.async.token
    } {tt.num_stages = 4 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 12 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0, 1>, array<i32: 1>]}
    // After loop
    // CHECK: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 12 : i32}
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 12 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 12 : i32}
    tt.return
  }

  // CHECK-LABEL: @specialize_mma_only
  tt.func @specialize_mma_only(%arg0: !tt.tensordesc<64x128xf16, #shared>, %arg1: !ttg.memdesc<128x64xf16, #shared, #smem>, %arg2: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // Reversed pattern: partition 0 stores, partition 1 does MMA
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.for
    %1 = scf.for %arg3 = %c0_i32 to %arg2 step %c1_i32 iter_args(%arg4 = %0) -> (!ttg.async.token)  : i32 {
      %2 = tt.descriptor_load %arg0[%arg3, %arg3] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load [[BUF]][] {ttg.partition = array<i32: 0>}
      %result_2, %token_3 = ttng.tmem_load %result[%arg4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK: "some_producer"
      %3:2 = "some_producer"(%2, %result_2) {ttg.partition = array<i32: 0>} : (tensor<64x128xf16, #blocked1>, tensor<128x128xf32, #blocked>) -> (tensor<128x64xf16, #blocked1>, tensor<128x128xf32, #blocked>)
      %4 = ttg.local_alloc %3#0 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %5 = ttg.memdesc_trans %4 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // CHECK: ttng.tmem_store {{.*}}, [[BUF]][]{{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %6 = ttng.tmem_store %3#1, %result[%token_3], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF2]][]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      %7 = ttng.tc_gen5_mma %arg1, %5, %result[%6], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %7 : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 15 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>]}
    // After loop: release/acquire/buffer/load/release
    // CHECK: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 15 : i32}
    // CHECK-NEXT: [[CTOK2:%.*]] = nvws.semaphore.acquire [[FULL]]
    // CHECK-NEXT: [[BUF3:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK2]]
    // CHECK-NEXT: ttng.tmem_load [[BUF3]][]
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK2]] [#nvws.async_op<none>]
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @load_scale_mma_user
  tt.func @load_scale_mma_user(%arg0: !ttg.memdesc<128x64xf16, #shared, #smem>, %arg1: !ttg.memdesc<64x128xf16, #shared, #smem>, %arg2: !tt.tensordesc<8x128xi8, #shared>, %arg3: !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, %arg4: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // ACC buffer + scale buffer each get their own semaphore pairs
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: [[SBUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<128x8xi8,
    // CHECK-NEXT: [[SEMPTY:%.*]] = nvws.semaphore.create [[SBUF]] true
    // CHECK-NEXT: [[SFULL:%.*]] = nvws.semaphore.create [[SBUF]] false
    // CHECK: scf.for
    %1 = scf.for %arg5 = %c0_i32 to %arg4 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = tt.descriptor_load %arg2[%arg5, %arg5] {ttg.partition = array<i32: 2>} : !tt.tensordesc<8x128xi8, #shared> -> tensor<8x128xi8, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = array<i32: 2>} : (tensor<8x128xi8, #blocked1>) -> !ttg.memdesc<8x128xi8, #shared, #smem>
      %4 = ttg.local_load %3 {ttg.partition = array<i32: 0>} : !ttg.memdesc<8x128xi8, #shared, #smem> -> tensor<8x128xi8, #linear1>
      %5 = tt.trans %4 {order = array<i32: 1, 0>, ttg.partition = array<i32: 0>} : tensor<8x128xi8, #linear1> -> tensor<128x8xi8, #linear>
      // CHECK: [[STOK:%.*]] = nvws.semaphore.acquire [[SEMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[SBUF2:%.*]] = nvws.semaphore.buffer [[SEMPTY]], [[STOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store {{.*}}, [[SBUF2]]{{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[SFULL]], [[STOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_2 = ttng.tmem_alloc %5 {ttg.partition = array<i32: 0>} : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
      // CHECK: [[BUF3:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[STOK2:%.*]] = nvws.semaphore.acquire [[SFULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[SBUF3:%.*]] = nvws.semaphore.buffer [[SFULL]], [[STOK2]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma_scaled {{.*}}, {{.*}}, [[BUF3]][], [[SBUF3]]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[SEMPTY]], [[STOK2]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      %6 = ttng.tc_gen5_mma_scaled %arg0, %arg1, %result[%arg6], %result_2, %arg3, %true, %true lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>

      // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BUF4:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load [[BUF4]][] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_3, %token_4 = ttng.tmem_load %result[%6] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK-NEXT: "user"
      // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 1>}
      "user"(%result_3) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield %token_4 : !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 16 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop
    // CHECK: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 16 : i32}
    // CHECK-NEXT: [[CTOK2:%.*]] = nvws.semaphore.acquire [[FULL]]
    // CHECK-NEXT: [[BUF5:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK2]]
    // CHECK-NEXT: ttng.tmem_load [[BUF5]][]
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK2]] [#nvws.async_op<none>]
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }

  // CHECK-LABEL: @store_mma_load
  tt.func @store_mma_load(%arg0: i32, %arg1: !tt.tensordesc<128x64xf16, #shared>, %arg2: !ttg.memdesc<64x128xf16, #shared, #smem>) {
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    // disallow_acc_multi_buffer => single-buffered
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: scf.for
    %0 = scf.for %arg3 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg4 = %token) -> (!ttg.async.token)  : i32 {
      %1 = tt.descriptor_load %arg1[%arg3, %arg3] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %2 = arith.addf %1, %1 {ttg.partition = array<i32: 0>} : tensor<128x64xf16, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      // CHECK: "make_acc"
      // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]][]{{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %4 = "make_acc"() {ttg.partition = array<i32: 0>} : () -> tensor<128x128xf32, #blocked>
      %5 = ttng.tmem_store %4, %result[%arg4], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF2]][]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      %6 = ttng.tc_gen5_mma %3, %arg2, %result[%5], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>

      // CHECK-NEXT: [[PTOK:%.*]] = nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BUF3:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[PTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load [[BUF3]][] {ttg.partition = array<i32: 0>}
      %result_0, %token_1 = ttng.tmem_load %result[%6] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK-NEXT: "use"
      "use"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield %token_1 : !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 17 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>]}
    // After loop
    // CHECK: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 17 : i32}
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 17 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 17 : i32}
    tt.return
  }

  // CHECK-LABEL: @local_alloc_into_mma
  tt.func @local_alloc_into_mma(%arg0: i32, %arg1: tensor<128x64xf16, #blocked1>, %arg2: !tt.tensordesc<64x128xf16, #shared>) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: scf.for
    %5 = scf.for %arg3 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg4 = %token) -> (!ttg.async.token)  : i32 {
      %0 = ttg.local_alloc %arg1 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %1 = tt.descriptor_load %arg2[%arg3, %arg3] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x128xf16, #shared> -> tensor<64x128xf16, #blocked1>
      %2 = arith.addf %1, %1 {ttg.partition = array<i32: 0>} : tensor<64x128xf16, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = array<i32: 0>} : (tensor<64x128xf16, #blocked1>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF]][]{{.*}} {ttg.partition = array<i32: 1>}
      %4 = ttng.tc_gen5_mma %0, %3, %result[%arg4], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %4 : !ttg.async.token
    } {ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>], tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 18 : i32}
    // After loop
    // CHECK: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 18 : i32}
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[FULL]]
    // CHECK-NEXT: [[BUF2:%.*]] = nvws.semaphore.buffer [[FULL]], {{.*}}
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], {{.*}} [#nvws.async_op<none>]
    ttng.tmem_load %result[%5] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    tt.return
  }

  // CHECK-LABEL: @shmem_sink_iterator_invalidation
  tt.func @shmem_sink_iterator_invalidation(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: !tt.tensordesc<128x64xf16, #shared>, %arg4: !tt.tensordesc<128x64xf16, #shared>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c64_i32 = arith.constant 64 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    // Two TMEM allocs: ACC (single-buffered) + LHS (TMEM operand, single-buffered)
    // ACC semaphore pair
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // LHS semaphore pair
    // CHECK: [[LBUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x64xf16,
    // CHECK-NEXT: [[LEMPTY:%.*]] = nvws.semaphore.create [[LBUF]] true
    // CHECK-NEXT: [[LFULL:%.*]] = nvws.semaphore.create [[LBUF]] false
    // CHECK: scf.for
    %1 = scf.for %arg5 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg6 = %0) -> (!ttg.async.token)  : i32 {
      %2 = arith.muli %arg5, %c64_i32 {ttg.partition = array<i32: 2>} : i32
      %3 = tt.descriptor_load %arg4[%arg2, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %4 = tt.descriptor_load %arg3[%arg1, %2] {ttg.partition = array<i32: 2>} : !tt.tensordesc<128x64xf16, #shared> -> tensor<128x64xf16, #blocked1>
      %5 = ttg.local_alloc %4 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %6 = ttg.local_load %5 {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> tensor<128x64xf16, #blocked2>
      %7 = ttg.local_alloc %3 {ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.memdesc_trans %7 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem> -> !ttg.memdesc<64x128xf16, #shared1, #smem>
      // LHS: acquire, buffer, store, release
      // CHECK: [[LTOK:%.*]] = nvws.semaphore.acquire [[LEMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[LBUF2:%.*]] = nvws.semaphore.buffer [[LEMPTY]], [[LTOK]] {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store {{.*}}, [[LBUF2]]{{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[LFULL]], [[LTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_2 = ttng.tmem_alloc %6 {ttg.partition = array<i32: 0>} : (tensor<128x64xf16, #blocked2>) -> !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory>
      // ACC buffer + LHS acquire for MMA
      // CHECK: [[BUF3:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LTOK2:%.*]] = nvws.semaphore.acquire [[LFULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: [[LBUF3:%.*]] = nvws.semaphore.buffer [[LFULL]], [[LTOK2]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma [[LBUF3]], {{.*}}, [[BUF3]][]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[LEMPTY]], [[LTOK2]] [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      %9 = ttng.tc_gen5_mma %result_2, %8, %result[%arg6], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<64x128xf16, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      scf.yield %9 : !ttg.async.token
    } {tt.num_stages = 2 : i32, tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 19 : i32, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>]}
    // After loop: release/acquire/buffer/load/release for ACC
    // CHECK: nvws.semaphore.release [[FULL]], {{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 19 : i32}
    // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]]
    // CHECK-NEXT: [[BUF4:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]]
    // CHECK-NEXT: ttng.tmem_load [[BUF4]][]
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>]
    %result_0, %token_1 = ttng.tmem_load %result[%1] : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
    "use"(%result_0) : (tensor<128x128xf32, #blocked>) -> ()
    tt.return
  }
}

// -----
#blocked = #ttg.blocked<{sizePerThread = [1, 64], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
#tmem1 = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @attention_forward
  tt.func public @attention_forward(%arg0: !ttg.memdesc<256x64xf16, #shared, #smem>, %arg1: !tt.tensordesc<64x64xf16, #shared>, %arg2: !tt.tensordesc<64x64xf16, #shared>, %arg3: f32, %arg4: i32) {
    %cst = arith.constant dense<1.000000e+00> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<256x64xf32, #blocked>
    %cst_1 = arith.constant dense<0xFF800000> : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %false = arith.constant false
    %true = arith.constant true
    // Three TMEM allocs: S (double-buffered), O (single-buffered), P (single-buffered)
    // S semaphore pair
    // CHECK: [[SBUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x256x64xf32,
    // CHECK-NEXT: [[SEMPTY:%.*]] = nvws.semaphore.create [[SBUF]] true
    // CHECK-NEXT: [[SFULL:%.*]] = nvws.semaphore.create [[SBUF]] false
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[SEMPTY]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // O semaphore pair
    // CHECK: [[OBUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf32,
    // CHECK-NEXT: [[OEMPTY:%.*]] = nvws.semaphore.create [[OBUF]] true
    // CHECK-NEXT: [[OFULL:%.*]] = nvws.semaphore.create [[OBUF]] false
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[OEMPTY]]
    // CHECK-NEXT: {{.*}} = nvws.semaphore.buffer [[OEMPTY]]
    // CHECK-NEXT: ttng.tmem_store
    %result_2, %token_3 = ttng.tmem_alloc : () -> (!ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst_0, %result_2[%token_3], %true : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>
    // P semaphore pair
    // CHECK: [[PBUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x64xf16,
    // CHECK-NEXT: [[PEMPTY:%.*]] = nvws.semaphore.create [[PBUF]] true
    // CHECK-NEXT: [[PFULL:%.*]] = nvws.semaphore.create [[PBUF]] false
    // CHECK: scf.for
    %1:4 = scf.for %arg5 = %c0_i32 to %arg4 step %c64_i32 iter_args(%arg6 = %cst, %arg7 = %cst_1, %arg8 = %token, %arg9 = %0) -> (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token)  : i32 {
      %2 = tt.descriptor_load %arg1[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #blocked1>
      %3 = ttg.local_alloc %2 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %4 = ttg.memdesc_trans %3 {order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      // S: buffer, mma, release
      // CHECK: nvws.semaphore.buffer [[SEMPTY]]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK: nvws.semaphore.release [[SFULL]]{{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      %5 = ttng.tc_gen5_mma %arg0, %4, %result[%arg8], %false, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #shared, #smem>, !ttg.memdesc<64x64xf16, #shared1, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

      // S: acquire, buffer, load, release
      // CHECK: nvws.semaphore.acquire [[SFULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.buffer [[SFULL]]{{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[SEMPTY]]{{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_6, %token_7 = ttng.tmem_load %result[%5] {ttg.partition = array<i32: 0>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>

      %6 = "compute_row_max"(%result_6, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, f32) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %7 = "sub_row_max"(%result_6, %6, %arg3) {ttg.partition = array<i32: 0>} : (tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, f32) -> tensor<256x64xf32, #blocked>
      %8 = math.exp2 %7 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked>
      %9 = arith.subf %arg7, %6 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %10 = arith.subf %arg7, %6 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %11 = math.exp2 %9 {ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %12 = math.exp2 %10 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %13 = "tt.reduce"(%8) <{axis = 1 : i32}> ({
      ^bb0(%arg10: f32, %arg11: f32):
        %24 = arith.addf %arg10, %arg11 {ttg.partition = array<i32: 0>}: f32
        tt.reduce.return %24 {ttg.partition = array<i32: 0>} : f32
      }) {ttg.partition = array<i32: 0>, ttg.partition.outputs = [array<i32: 0>]} : (tensor<256x64xf32, #blocked>) -> tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %14 = arith.mulf %arg6, %12 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %15 = arith.addf %14, %13 {ttg.partition = array<i32: 0>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
      %16 = tt.expand_dims %11 {axis = 1 : i32, ttg.partition = array<i32: 3>} : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<256x1xf32, #blocked>
      %17 = tt.broadcast %16 {ttg.partition = array<i32: 3>} : tensor<256x1xf32, #blocked> -> tensor<256x64xf32, #blocked>

      // O: buffer, load
      // CHECK: nvws.semaphore.buffer [[OEMPTY]]{{.*}} {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: ttng.tmem_load {{.*}} {ttg.partition = array<i32: 3>}
      %result_8, %token_9 = ttng.tmem_load %result_2[%arg9] {ttg.partition = array<i32: 3>} : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>

      %18 = arith.mulf %result_8, %17 {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked>
      %19 = tt.descriptor_load %arg2[%arg5, %c0_i32] {ttg.partition = array<i32: 2>} : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #blocked1>
      %20 = ttg.local_alloc %19 {ttg.partition = array<i32: 2>} : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %21 = arith.truncf %8 {ttg.partition = array<i32: 0>} : tensor<256x64xf32, #blocked> to tensor<256x64xf16, #blocked>
      // P: acquire, buffer, store, release
      // CHECK: nvws.semaphore.acquire [[PEMPTY]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.buffer [[PEMPTY]]{{.*}} {ttg.partition = array<i32: 0>}
      // CHECK: ttng.tmem_store {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[PFULL]]{{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %result_10 = ttng.tmem_alloc %21 {ttg.partition = array<i32: 0>} : (tensor<256x64xf16, #blocked>) -> !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>

      // O: store, release
      // CHECK: ttng.tmem_store {{.*}} {ttg.partition = array<i32: 3>}
      // CHECK-NEXT: nvws.semaphore.release [[OFULL]]{{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 3>}
      %22 = ttng.tmem_store %18, %result_2[%token_9], %true {ttg.partition = array<i32: 3>} : tensor<256x64xf32, #blocked> -> !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

      // O: acquire, buffer for MMA
      // CHECK: nvws.semaphore.acquire [[OFULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.buffer [[OFULL]]{{.*}} {ttg.partition = array<i32: 1>}
      // P: acquire, buffer for MMA
      // CHECK: nvws.semaphore.acquire [[PFULL]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.buffer [[PFULL]]{{.*}} {ttg.partition = array<i32: 1>}
      // CHECK: ttng.tc_gen5_mma {{.*}} {ttg.partition = array<i32: 1>}
      // P+O: release after MMA
      // CHECK: nvws.semaphore.release [[PEMPTY]]{{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.release [[OEMPTY]]{{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 1>}
      %23 = ttng.tc_gen5_mma %result_10, %20, %result_2[%22], %true, %true {ttg.partition = array<i32: 1>} : !ttg.memdesc<256x64xf16, #tmem1, #ttng.tensor_memory>, !ttg.memdesc<64x64xf16, #shared, #smem>, !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable>

      // S+O: acquire for next iter
      // CHECK: nvws.semaphore.acquire [[SEMPTY]] {ttg.partition = array<i32: 1>}
      // CHECK-NEXT: nvws.semaphore.acquire [[OEMPTY]] {ttg.partition = array<i32: 3>}
      scf.yield %15, %6, %token_7, %23 : tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, !ttg.async.token, !ttg.async.token
    } {tt.warp_specialize, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 1 : i32], ttg.warp_specialize.tag = 0 : i32, ttg.partition = array<i32: 0, 1, 2, 3>, ttg.partition.outputs = [array<i32: 0>, array<i32: 0>, array<i32: 1>, array<i32: 3>]}
    // After loop: O release, S release, S acquire+release, O acquire+buffer+load+release
    // CHECK: nvws.semaphore.release [[OFULL]]{{.*}} {ttg.partition = array<i32: 3>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[SFULL]]{{.*}} {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[SFULL]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[SEMPTY]]{{.*}} {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK: nvws.semaphore.acquire [[OFULL]]
    // CHECK-NEXT: nvws.semaphore.buffer [[OFULL]]
    // CHECK-NEXT: ttng.tmem_load
    // CHECK-NEXT: nvws.semaphore.release [[OEMPTY]]{{.*}} [#nvws.async_op<none>]
    %result_4, %token_5 = ttng.tmem_load %result_2[%1#3] : !ttg.memdesc<256x64xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<256x64xf32, #blocked>
    "use"(%1#0, %result_4, %1#1) : (tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>, tensor<256x64xf32, #blocked>, tensor<256xf32, #ttg.slice<{dim = 1, parent = #blocked}>>) -> ()
    tt.return
  }

}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @hoisted_alloc
  tt.func @hoisted_alloc(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>) {
    %true = arith.constant true
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // Hoisted alloc with nested loops
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %res, %tok = ttng.tmem_alloc %cst : (tensor<128x128xf32, #blocked>) -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    // CHECK: scf.for
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %tok) -> (!ttg.async.token) : i32 {
      %ptrub = tt.addptr %ptr0, %iv0 {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>, i32
      %ub1 = tt.load %ptrub {ttg.partition = array<i32: 1, 2>} : !tt.ptr<i32>
      %lb1 = "lb1"(%iv0) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      %step1 = "step1"(%iv0) {ttg.partition = array<i32: 1, 2>} : (i32) -> i32
      // CHECK: scf.for
      %tok4 = scf.for %iv = %lb1 to %ub1 step %step1 iter_args(%tok2 = %tok1) -> (!ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        // CHECK: [[BUF2:%.*]] = nvws.semaphore.buffer [[EMPTY]], {{.*}} {ttg.partition = array<i32: 2>}
        // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF2]][]{{.*}} {ttg.partition = array<i32: 2>}
        %tok3 = ttng.tc_gen5_mma %sA, %sB, %res[%tok2], %true, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %tok3 : !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>]}
      // CHECK: nvws.semaphore.release [[FULL]]{{.*}} [#nvws.async_op<tc5mma>] {ttg.partition = array<i32: 2>}
      // CHECK-NEXT: [[CTOK:%.*]] = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: [[BUF3:%.*]] = nvws.semaphore.buffer [[FULL]], [[CTOK]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load [[BUF3]][] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]], [[CTOK]] [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      %val, %tok5 = ttng.tmem_load %res[%tok4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      // CHECK-NEXT: "use"
      "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      // CHECK: nvws.semaphore.acquire [[EMPTY]] {ttg.partition = array<i32: 2>}
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %tok5 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    // After outer loop
    // CHECK: nvws.semaphore.release [[FULL]]{{.*}} {ttg.partition = array<i32: 2>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]]{{.*}} {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked3 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @if_split_workaround
  tt.func @if_split_workaround(%arg0: !tt.tensordesc<1x64xf16, #shared>, %arg1: tensor<64x128x!tt.ptr<f16>, #blocked3> {tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>}) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %c32_i32 = arith.constant 32 : i32
    // Single-buffered (disallow_acc_multi_buffer)
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    // CHECK-NEXT: [[ATOK:%.*]] = nvws.semaphore.acquire [[EMPTY]]
    // CHECK-NEXT: [[BUF:%.*]] = nvws.semaphore.buffer [[EMPTY]], [[ATOK]]
    // CHECK-NEXT: ttng.tmem_store {{.*}}, [[BUF]]
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %0 = ttng.tmem_store %cst, %result[%token], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    // CHECK: scf.for
    %1:3 = scf.for %arg2 = %c0_i32 to %c32_i32 step %c1_i32 iter_args(%arg3 = %true, %arg4 = %arg1, %arg5 = %0) -> (i1, tensor<64x128x!tt.ptr<f16>, #blocked3>, !ttg.async.token)  : i32 {
      %2:3 = "get_offsets"(%arg2) {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 1, 2>} : (i32) -> (i32, tensor<64x128xi32, #blocked3>, i32)
      %3 = tt.splat %2#0 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32 -> tensor<128xi32, #blocked2>
      %4 = tt.descriptor_gather %arg0[%3, %2#2] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : (!tt.tensordesc<1x64xf16, #shared>, tensor<128xi32, #blocked2>, i32) -> tensor<128x64xf16, #blocked1>
      %5 = tt.addptr %arg4, %2#1 {loop.cluster = 3 : i32, loop.stage = 1 : i32, tt.constancy = dense<1> : tensor<2xi32>, tt.contiguity = dense<[1, 64]> : tensor<2xi32>, tt.divisibility = dense<16> : tensor<2xi32>, ttg.partition = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked3>, tensor<64x128xi32, #blocked3>
      %6 = tt.load %5 {loop.cluster = 3 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 1>} : tensor<64x128x!tt.ptr<f16>, #blocked3>
      %7 = ttg.local_alloc %4 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x64xf16, #blocked1>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
      %8 = ttg.local_alloc %6 {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : (tensor<64x128xf16, #blocked3>) -> !ttg.memdesc<64x128xf16, #shared, #smem>
      // CHECK: [[BUF2:%.*]] = nvws.semaphore.buffer [[EMPTY]]{{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
      // CHECK-NEXT: ttng.tc_gen5_mma {{.*}}, {{.*}}, [[BUF2]][]{{.*}} {loop.cluster = 2 : i32, loop.stage = 2 : i32, {{.*}}ttg.partition = array<i32: 1>}
      %9 = ttng.tc_gen5_mma %7, %8, %result[%arg5], %arg3, %true {loop.cluster = 2 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x64xf16, #shared, #smem>, !ttg.memdesc<64x128xf16, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %10 = arith.cmpi eq, %arg2, %c0_i32 {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 1>} : i32
      %11 = arith.select %10, %false, %true {loop.cluster = 1 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 1>} : i1
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.release [[FULL]]{{.*}} [#nvws.async_op<tc5mma>] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.buffer [[FULL]]{{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: ttng.tmem_load {{.*}} {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: nvws.semaphore.release [[EMPTY]]{{.*}} [#nvws.async_op<none>] {ttg.partition = array<i32: 0>}
      // CHECK-NEXT: "acc_user"
      %12 = scf.if %10 -> (!ttg.async.token) {
        %result_0, %token_1 = ttng.tmem_load %result[%9] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
        "acc_user"(%result_0) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
        scf.yield {ttg.partition = array<i32: 0, 1>} %token_1 : !ttg.async.token
      } else {
        scf.yield {ttg.partition = array<i32: 0, 1>} %9 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 3 : i32, ttg.partition = array<i32: 0, 1>, ttg.partition.outputs = [array<i32: 1>]}
      // CHECK: scf.if
      // CHECK-NEXT: nvws.semaphore.acquire [[EMPTY]] {loop.cluster = 2 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>}
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %11, %5, %12 : i1, tensor<64x128x!tt.ptr<f16>, #blocked3>, !ttg.async.token
    } {tt.disallow_acc_multi_buffer, tt.num_stages = 3 : i32, tt.scheduled_max_stage = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 1>, array<i32: 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 2 : i32}
    // After loop
    // CHECK: nvws.semaphore.release [[FULL]]{{.*}} {ttg.partition = array<i32: 1>, ttg.warp_specialize.tag = 2 : i32}
    // CHECK-NEXT: {{.*}} = nvws.semaphore.acquire [[FULL]] {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 2 : i32}
    // CHECK-NEXT: nvws.semaphore.release [[EMPTY]]{{.*}} {ttg.partition = array<i32: 0>, ttg.warp_specialize.tag = 2 : i32}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @nested_loop_yes_double_buffer
  tt.func @nested_loop_yes_double_buffer(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // Double-buffered: inner loop store is in partition 2 (same as MMA producer)
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    %res, %tok = ttng.tmem_alloc : () ->(!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %toka = ttng.tmem_store %cst, %res[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %toka) -> (!ttg.async.token) : i32 {
      %tok1a = ttng.tmem_store %cst, %res[%tok1], %true {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %useD, %tok4 = scf.for %iv = %lb to %ub step %step iter_args(%useD = %false, %tok2 = %tok1a) -> (i1, !ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        %tok3 = ttng.tc_gen5_mma %sA, %sB, %res[%tok2], %useD, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %tok3 : i1, !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}
      %val, %tok5 = ttng.tmem_load %res[%tok4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %tok5 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @nested_loop_no_double_buffer
  tt.func @nested_loop_no_double_buffer(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // Single-buffered: inner loop store is in partition 0 (consumer side)
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    %res, %tok = ttng.tmem_alloc : () ->(!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %toka = ttng.tmem_store %cst, %res[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %toka) -> (!ttg.async.token) : i32 {
      %tok1a = ttng.tmem_store %cst, %res[%tok1], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %useD, %tok4 = scf.for %iv = %lb to %ub step %step iter_args(%useD = %false, %tok2 = %tok1a) -> (i1, !ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        %tok3 = ttng.tc_gen5_mma %sA, %sB, %res[%tok2], %useD, %true {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %tok3 : i1, !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}
      %val, %tok5 = ttng.tmem_load %res[%tok4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %tok5 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 0>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @nested_loop_yes_double_buffer_scaled
  tt.func @nested_loop_yes_double_buffer_scaled(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>,
    %scalesA: tensor<128x8xi8, #linear>, %scalesB: tensor<128x8xi8, #linear>) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    // Double-buffered with scaled MMA
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<2x128x128xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    %res, %tok = ttng.tmem_alloc : () ->(!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %toka = ttng.tmem_store %cst, %res[%tok], %true : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %lhs_scales = ttng.tmem_alloc %scalesA: (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %rhs_scales = ttng.tmem_alloc %scalesB : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %toka) -> (!ttg.async.token) : i32 {
      %tok1a = ttng.tmem_store %cst, %res[%tok1], %true {ttg.partition = array<i32: 2>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %useD, %tok4 = scf.for %iv = %lb to %ub step %step iter_args(%useD = %false, %tok2 = %tok1a) -> (i1, !ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x128xf32, #shared, #smem>
        %tok3 = ttng.tc_gen5_mma_scaled %sA, %sB, %res[%tok2], %lhs_scales, %rhs_scales, %useD, %true lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x128xf32, #shared, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %tok3 : i1, !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}
      %val, %tok5 = ttng.tmem_load %res[%tok4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #blocked>
      "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x128xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %tok5 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

  // CHECK-LABEL: @nested_loop_no_double_buffer_scaled
  tt.func @nested_loop_no_double_buffer_scaled(%lb: i32, %ub: i32, %step: i32, %ptr0: !tt.ptr<i32>,
    %scalesA: tensor<128x8xi8, #linear>, %scalesB: tensor<128x8xi8, #linear>) {
    %true = arith.constant true
    %false = arith.constant false
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    // Single-buffered: inner loop store in partition 2 but 128x256 is too large
    // CHECK: [[ABUF:%.*]] = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x256xf32,
    // CHECK-NEXT: [[EMPTY:%.*]] = nvws.semaphore.create [[ABUF]] true
    // CHECK-NEXT: [[FULL:%.*]] = nvws.semaphore.create [[ABUF]] false
    %res, %tok = ttng.tmem_alloc : () ->(!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %toka = ttng.tmem_store %cst, %res[%tok], %true : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %lhs_scales = ttng.tmem_alloc %scalesA : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %rhs_scales = ttng.tmem_alloc %scalesB : (tensor<128x8xi8, #linear>) -> !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
    %tok0 = scf.for %iv0 = %lb to %ub step %step iter_args(%tok1 = %toka) -> (!ttg.async.token) : i32 {
      %tok1a = ttng.tmem_store %cst, %res[%tok1], %true {ttg.partition = array<i32: 2>} : tensor<128x256xf32, #blocked> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
      %useD, %tok4 = scf.for %iv = %lb to %ub step %step iter_args(%useD = %false, %tok2 = %tok1a) -> (i1, !ttg.async.token)  : i32 {
        %sA = "load1"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<128x64xf32, #shared, #smem>
        %sB = "load2"(%iv) {ttg.partition = array<i32: 1>} : (i32) -> !ttg.memdesc<64x256xf32, #shared, #smem>
        %tok3 = ttng.tc_gen5_mma_scaled %sA, %sB, %res[%tok2], %lhs_scales, %rhs_scales, %useD, %true lhs = e4m3 rhs = e4m3 {ttg.partition = array<i32: 2>} : !ttg.memdesc<128x64xf32, #shared, #smem>, !ttg.memdesc<64x256xf32, #shared, #smem>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x8xi8, #tmem_scales, #ttng.tensor_memory>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %tok3 : i1, !ttg.async.token
      } {ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 2>, array<i32: 2>]}
      %val, %tok5 = ttng.tmem_load %res[%tok4] {ttg.partition = array<i32: 0>} : !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x256xf32, #blocked>
      "use"(%val) {ttg.partition = array<i32: 0>} : (tensor<128x256xf32, #blocked>) -> ()
      scf.yield {ttg.partition = array<i32: 0, 1, 2>} %tok5 : !ttg.async.token
    } {tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.outputs = [array<i32: 2>], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}

// -----

// Test that tmem allocations in functions that do not use warp specialization
// do not trigger an assert if they have multiple uses.

// CHECK-LABEL: @test_tmem_no_ws
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4], [0, 8]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-NOT: nvws.semaphore.create
  tt.func public @test_tmem_no_ws(%arg0: !ttg.memdesc<128x128xi8, #shared, #smem>, %arg1: !ttg.memdesc<128x128xi8, #shared1, #smem>, %arg2: !ttg.memdesc<128x128xi8, #shared1, #smem>, %arg3: tensor<128x16xf8E4M3FN, #linear>, %arg4: tensor<128x16xf8E4M3FN, #linear>, %arg5: tensor<128x16xf8E4M3FN, #linear>) {
    %true = arith.constant true
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_0, %token_1 = ttng.tmem_alloc : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %result_2 = ttng.tmem_alloc %arg3 : (tensor<128x16xf8E4M3FN, #linear>) -> !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>
    %result_3 = ttng.tmem_alloc %arg4 : (tensor<128x16xf8E4M3FN, #linear>) -> !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>
    %result_4 = ttng.tmem_alloc %arg5 : (tensor<128x16xf8E4M3FN, #linear>) -> !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>
    %0 = ttng.tc_gen5_mma_scaled %arg0, %arg1, %result[%token], %result_2, %result_3, %true, %true lhs = e2m1 rhs = e2m1 : !ttg.memdesc<128x128xi8, #shared, #smem>, !ttg.memdesc<128x128xi8, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>
    %1 = ttng.tc_gen5_mma_scaled %arg0, %arg2, %result_0[%token_1], %result_2, %result_4, %true, %true lhs = e2m1 rhs = e2m1 : !ttg.memdesc<128x128xi8, #shared, #smem>, !ttg.memdesc<128x128xi8, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>, !ttg.memdesc<128x16xf8E4M3FN, #tmem_scales, #ttng.tensor_memory>
    tt.return
  }
}
