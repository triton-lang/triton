// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-hoist-tmem-store | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 128], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_nested_persistent_ws_kernel(%arg0: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg1: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg2: !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>>, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    // There is llvm.intr.assume on the inner-loop upper bound, the tmem store can be hoisted to the top level
    // CHECK: {{.*}}, [[TOKEN:%.*]] = ttng.tmem_alloc {{.*}} : (tensor<128x128xf32, #blocked>)
    // CHECK-NOT: tmem_store
    // CHECK: scf.for {{.*}}iter_args([[TOKEN_ARG:%.*]] = [[TOKEN]])
    scf.for %arg6 = %0 to %4 step %c148_i32  : i32 {
      %6 = arith.divsi %arg6, %5 {ttg.partition = array<i32: 0, 2>} : i32
      %7 = arith.muli %6, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %8 = arith.subi %1, %7 {ttg.partition = array<i32: 0, 2>} : i32
      %9 = arith.minsi %8, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %10 = arith.remsi %arg6, %9 {ttg.partition = array<i32: 0, 2>} : i32
      %11 = arith.addi %7, %10 {ttg.partition = array<i32: 0, 2>} : i32
      %12 = arith.remsi %arg6, %5 {ttg.partition = array<i32: 0, 2>} : i32
      %13 = arith.divsi %12, %9 {ttg.partition = array<i32: 0, 2>} : i32
      // CHECK-COUNT-3: arith.muli
      // CHECK-NEXT: arith.addi
      // CHECK-NEXT: arith.cmpi
      // CHECK-NEXT: llvm.intr.assume
      // CHECK-NEXT: scf.for {{.*}}iter_args({{.*}} = {{.*}}, {{.*}} = [[TOKEN_ARG]])
      %14 = arith.muli %11, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %15 = arith.muli %13, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %result, %token = ttng.tmem_alloc {ttg.partition = array<i32: 0, 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %16 = ttng.tmem_store %cst, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %17 = arith.addi %3, %arg6 {ttg.partition = array<i32: 1, 2>} : i32
      %18 = arith.cmpi sgt, %17, %c0_i32 {ttg.partition = array<i32: 1, 2>} : i32
      llvm.intr.assume %18 : i1 {ttg.partition = array<i32: 1, 2>}
      %19:2 = scf.for %arg7 = %c0_i32 to %17 step %c1_i32 iter_args(%arg8 = %false, %arg9 = %16) -> (i1, !ttg.async.token)  : i32 {
        %22 = arith.muli %arg7, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        %23 = tt.descriptor_load %arg0[%14, %22] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
        %24 = ttg.local_alloc %23 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %25 = tt.descriptor_load %arg1[%15, %22] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
        %26 = ttg.local_alloc %25 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %27 = ttg.memdesc_trans %26 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
        %28 = ttng.tc_gen5_mma %24, %27, %result[%arg9], %arg8, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %28 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1>]}
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}

    // There is no llvm.intr.assume in this case
    // CHECK: scf.for
    scf.for %arg6 = %0 to %4 step %c148_i32  : i32 {
      %6 = arith.divsi %arg6, %5 {ttg.partition = array<i32: 0, 2>} : i32
      %7 = arith.muli %6, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %8 = arith.subi %1, %7 {ttg.partition = array<i32: 0, 2>} : i32
      %9 = arith.minsi %8, %c8_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %10 = arith.remsi %arg6, %9 {ttg.partition = array<i32: 0, 2>} : i32
      %11 = arith.addi %7, %10 {ttg.partition = array<i32: 0, 2>} : i32
      %12 = arith.remsi %arg6, %5 {ttg.partition = array<i32: 0, 2>} : i32
      %13 = arith.divsi %12, %9 {ttg.partition = array<i32: 0, 2>} : i32
      %14 = arith.muli %11, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      %15 = arith.muli %13, %c128_i32 {ttg.partition = array<i32: 0, 2>} : i32
      // CHECK: {{.*}}, [[TOKEN:%.*]] = ttng.tmem_alloc {{.*}} {ttg.partition = array<i32: 1>}
      // CHECK-NOT: tmem_store
      // CHECK: scf.for {{.*}}iter_args({{.*}} = {{.*}}, {{.*}} = [[TOKEN]])
      %result, %token = ttng.tmem_alloc {ttg.partition = array<i32: 0, 1>} : () -> (!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
      %16 = ttng.tmem_store %cst, %result[%token], %true {ttg.partition = array<i32: 0>} : tensor<128x128xf32, #blocked> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
      %17 = arith.addi %3, %arg6 {ttg.partition = array<i32: 1, 2>} : i32
      %19:2 = scf.for %arg7 = %c0_i32 to %17 step %c1_i32 iter_args(%arg8 = %false, %arg9 = %16) -> (i1, !ttg.async.token)  : i32 {
        %22 = arith.muli %arg7, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : i32
        %23 = tt.descriptor_load %arg0[%14, %22] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
        %24 = ttg.local_alloc %23 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %25 = tt.descriptor_load %arg1[%15, %22] {loop.cluster = 2 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E4M3FN, #shared>> -> tensor<128x128xf8E4M3FN, #blocked1>
        %26 = ttg.local_alloc %25 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 2>} : (tensor<128x128xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>
        %27 = ttg.memdesc_trans %26 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem> -> !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>
        %28 = ttng.tc_gen5_mma %24, %27, %result[%arg9], %arg8, %true {loop.cluster = 0 : i32, loop.stage = 2 : i32, tt.self_latency = 1 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E4M3FN, #shared, #smem>, !ttg.memdesc<128x128xf8E4M3FN, #shared1, #smem>, !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
        scf.yield {ttg.partition = array<i32: 1, 2>} %true, %28 : i1, !ttg.async.token
      } {tt.scheduled_max_stage = 2 : i32, ttg.partition = array<i32: 1, 2>, ttg.partition.outputs = [array<i32: 1, 2>, array<i32: 1>]}
    } {tt.num_stages = 3 : i32, tt.warp_specialize, ttg.partition = array<i32: 0, 1, 2>, ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }

}
