// RUN: triton-opt %s -split-input-file --allow-unregistered-dialect --nvws-lower-aref  | FileCheck %s
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#linear2 = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [128, 0]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = false, elementBitWidth = 8}>
#smem = #ttg.shared_memory
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 256, colStride = 1>
#tmem_scales = #ttng.tensor_memory_scales_encoding<>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @block_scaled_mma_scale_copy_tmem_store(%arg0: !tt.tensordesc<tensor<128x128xf8E5M2, #shared>>, %arg1: !tt.tensordesc<tensor<256x128xf8E5M2, #shared>>, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: tensor<128x4x!tt.ptr<i8>, #blocked>, %arg4: tensor<256x4x!tt.ptr<i8>, #blocked>, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %false = arith.constant false
    %cst = arith.constant dense<4> : tensor<128x4xi32, #blocked>
    %cst_0 = arith.constant dense<4> : tensor<256x4xi32, #blocked>
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #linear>
    %0 = tt.get_program_id x : i32
    %1 = arith.divsi %arg5, %c128_i32 : i32
    %2 = arith.remsi %0, %1 : i32
    %3 = arith.divsi %0, %1 : i32
    %4 = arith.muli %2, %c128_i32 : i32
    %5 = arith.muli %3, %c256_i32 : i32
    %6 = arith.divsi %arg7, %c128_i32 : i32
    %result, %token = ttng.tmem_alloc : () -> (!ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.async.token)
    %7 = ttng.tmem_store %cst_1, %result[%token], %true : tensor<128x256xf32, #linear> -> !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>
    %8 = ttg.local_alloc : () -> !ttg.memdesc<1x128x128xf8E5M2, #shared, #smem, mutable>
    %9 = nvws.aref.create %8 : <[!ttg.memdesc<1x128x128xf8E5M2, #shared, #smem, mutable>]>
    %10 = ttg.local_alloc : () -> !ttg.memdesc<1x256x128xf8E5M2, #shared, #smem, mutable>
    %11 = nvws.aref.create %10 : <[!ttg.memdesc<1x256x128xf8E5M2, #shared, #smem, mutable>]>
    // CHECK: ttng.tmem_alloc : () -> !ttg.memdesc<2x128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    %result_2 = ttng.tmem_alloc : () -> !ttg.memdesc<1x128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    %12 = nvws.aref.create %result_2 : <[!ttg.memdesc<1x128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>
    %13 = ub.poison : !ttg.async.token
    // CHECK: ttng.tmem_alloc : () -> !ttg.memdesc<2x256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    %result_3 = ttng.tmem_alloc : () -> !ttg.memdesc<1x256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>
    %14 = nvws.aref.create %result_3 : <[!ttg.memdesc<1x256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>
    %15 = ub.poison : !ttg.async.token
    %16:5 = scf.for %arg9 = %c0_i32 to %6 step %c1_i32 iter_args(%arg10 = %c0_i32, %arg11 = %arg3, %arg12 = %arg4, %arg13 = %false, %arg14 = %7) -> (i32, tensor<128x4x!tt.ptr<i8>, #blocked>, tensor<256x4x!tt.ptr<i8>, #blocked>, i1, !ttg.async.token)  : i32 {
      %buffers, %token_4 = nvws.aref.put.enter %9 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf8E5M2, #shared, #smem, mutable>]> -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable, 1x128x128>, !ttg.async.token
      nvws.descriptor_load %arg0[%4, %arg10] 16384 %buffers {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<128x128xf8E5M2, #shared>>, i32, i32, !ttg.memdesc<128x128xf8E5M2, #shared, #smem, mutable, 1x128x128>
      nvws.aref.put.exit %9, %token_4 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x128x128xf8E5M2, #shared, #smem, mutable>]>, !ttg.async.token
      %buffers_5, %token_6 = nvws.aref.put.enter %11 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x128xf8E5M2, #shared, #smem, mutable>]> -> !ttg.memdesc<256x128xf8E5M2, #shared, #smem, mutable, 1x256x128>, !ttg.async.token
      nvws.descriptor_load %arg1[%5, %arg10] 32768 %buffers_5 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : !tt.tensordesc<tensor<256x128xf8E5M2, #shared>>, i32, i32, !ttg.memdesc<256x128xf8E5M2, #shared, #smem, mutable, 1x256x128>
      nvws.aref.put.exit %11, %token_6 [#nvws.async_op<tma_load>] {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 2>} : <[!ttg.memdesc<1x256x128xf8E5M2, #shared, #smem, mutable>]>, !ttg.async.token
      %buffers_7, %token_8 = nvws.aref.get.enter %11 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x128xf8E5M2, #shared, #smem, mutable>]> -> !ttg.memdesc<256x128xf8E5M2, #shared, #smem, 1x256x128>, !ttg.async.token
      %17 = ttg.memdesc_trans %buffers_7 {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>, ttg.partition = array<i32: 1>} : !ttg.memdesc<256x128xf8E5M2, #shared, #smem, 1x256x128> -> !ttg.memdesc<128x256xf8E5M2, #shared1, #smem, 1x128x256>
      %18 = tt.load %arg11 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<128x4x!tt.ptr<i8>, #blocked>
      %19 = ttg.local_alloc %18 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x4xi8, #blocked>) -> !ttg.memdesc<128x4xi8, #shared2, #smem>
      %20 = ttg.local_load %19 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : !ttg.memdesc<128x4xi8, #shared2, #smem> -> tensor<128x4xi8, #linear1>
      %21 = tt.load %arg12 {loop.cluster = 3 : i32, loop.stage = 0 : i32, ttg.partition = array<i32: 3>} : tensor<256x4x!tt.ptr<i8>, #blocked>
      %22 = ttg.local_alloc %21 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<256x4xi8, #blocked>) -> !ttg.memdesc<256x4xi8, #shared2, #smem>
      %23 = ttg.local_load %22 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : !ttg.memdesc<256x4xi8, #shared2, #smem> -> tensor<256x4xi8, #linear2>
      %buffers_9, %token_10 = nvws.aref.put.enter %12 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x128x4>, !ttg.async.token
      %24 = nvws.aref.buffer %12, %token_10 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x128x4>
      %true_11 = arith.constant {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} true
      ttng.tmem_store %20, %24, %true_11 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : tensor<128x4xi8, #linear1> -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x128x4>
      %result_12 = ttng.tmem_alloc %20 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<128x4xi8, #linear1>) -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory>
      nvws.aref.put.exit %12, %token_10 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %buffers_13, %token_14 = nvws.aref.put.enter %14 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x256x4>, !ttg.async.token
      %25 = nvws.aref.buffer %14, %token_14 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x256x4>
      %true_15 = arith.constant {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} true
      ttng.tmem_store %23, %25, %true_15 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : tensor<256x4xi8, #linear2> -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x256x4>
      %result_16 = ttng.tmem_alloc %23 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : (tensor<256x4xi8, #linear2>) -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory>
      nvws.aref.put.exit %14, %token_14 [#nvws.async_op<none>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 3>} : <[!ttg.memdesc<1x256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      %buffers_17, %token_18 = nvws.aref.get.enter %9 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf8E5M2, #shared, #smem, mutable>]> -> !ttg.memdesc<128x128xf8E5M2, #shared, #smem, 1x128x128>, !ttg.async.token
      %buffers_19, %token_20 = nvws.aref.get.enter %12 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x128x4>, !ttg.async.token
      %26 = nvws.aref.buffer %12, %token_20 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x128x4>
      %buffers_21, %token_22 = nvws.aref.get.enter %14 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]> -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x256x4>, !ttg.async.token
      %27 = nvws.aref.buffer %14, %token_22 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token -> !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x256x4>
      // CHECK: ttng.tc_gen5_mma_scaled {{.*}}is_async
      %28 = ttng.tc_gen5_mma_scaled %buffers_17, %17, %result[%arg14], %26, %27, %arg13, %true lhs = e5m2 rhs = e5m2 {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : !ttg.memdesc<128x128xf8E5M2, #shared, #smem, 1x128x128>, !ttg.memdesc<128x256xf8E5M2, #shared1, #smem, 1x128x256>, !ttg.memdesc<128x256xf32, #tmem, #ttng.tensor_memory, mutable>, !ttg.memdesc<128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x128x4>, !ttg.memdesc<256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable, 1x256x4>
      nvws.aref.get.exit %14, %token_22 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      nvws.aref.get.exit %12, %token_20 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x4xi8, #tmem_scales, #ttng.tensor_memory, mutable>]>, !ttg.async.token
      nvws.aref.get.exit %11, %token_8 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x256x128xf8E5M2, #shared, #smem, mutable>]>, !ttg.async.token
      nvws.aref.get.exit %9, %token_18 [#nvws.async_op<tc5mma>] {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.partition = array<i32: 1>} : <[!ttg.memdesc<1x128x128xf8E5M2, #shared, #smem, mutable>]>, !ttg.async.token
      %29 = arith.addi %arg10, %c128_i32 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 2>} : i32
      %30 = tt.addptr %arg11, %cst {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : tensor<128x4x!tt.ptr<i8>, #blocked>, tensor<128x4xi32, #blocked>
      %31 = tt.addptr %arg12, %cst_0 {loop.cluster = 2 : i32, loop.stage = 1 : i32, ttg.partition = array<i32: 3>} : tensor<256x4x!tt.ptr<i8>, #blocked>, tensor<256x4xi32, #blocked>
      scf.yield {ttg.partition = array<i32: 1, 2, 3>} %29, %30, %31, %true, %28 : i32, tensor<128x4x!tt.ptr<i8>, #blocked>, tensor<256x4x!tt.ptr<i8>, #blocked>, i1, !ttg.async.token
    } {tt.num_stages = 3 : i32, tt.scheduled_max_stage = 2 : i32, tt.warp_specialize, ttg.partition = array<i32: 1, 2, 3>, ttg.partition.outputs = [array<i32: 2>, array<i32: 3>, array<i32: 3>, array<i32: 1>, array<i32: 1>], ttg.partition.stages = [0 : i32, 1 : i32, 0 : i32, 0 : i32], ttg.warp_specialize.tag = 0 : i32}
    tt.return
  }
}
