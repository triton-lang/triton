// RUN: triton-opt %s --nvgpu-warp-specialization='dump-intermediate-steps=true num-stages=2' 2>&1 | FileCheck %s
//
// This regression covers two coupled bugs in the warp specialization retry path:
// 1. A failed 3-warp-group attempt must clear the task ids it generated before
//    retrying with 2 warp groups.
// 2. After the retry collapses the consumer to a single task, code partitioning
//    must still append the i64 accum-count bookkeeping used by buffer indexing.
//
// CHECK-LABEL: // -----// WarpSpec internal IR Dump After: doTaskPartition
// CHECK: ttng.warp_group_dot {{.*}} {async_task_id = array<i32: 1, 2>, inputPrecision = 0 : i32}
// CHECK: tt.descriptor_store {{.*}} {async_task_id = array<i32: 1, 2>}
//
// CHECK-LABEL: // -----// WarpSpec internal IR Dump After: doTaskPartition
// CHECK-NOT: async_task_id = array<i32: 1, 2>
// CHECK: ttng.warp_group_dot {{.*}} {async_task_id = array<i32: 1>, inputPrecision = 0 : i32}
// CHECK: tt.descriptor_store {{.*}} {async_task_id = array<i32: 1>}
//
// CHECK-LABEL: // -----// WarpSpec internal IR Dump After: doCodePartition
// CHECK: partition0(
// CHECK: scf.for {{.*}} iter_args({{.*}}%{{.*}} = %c0_i64) -> (
// CHECK-SAME: i64
// CHECK: arith.divui %{{.*}}, %{{.*}} : i64

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @attention_inner_loop_kernel(%arg0: !tt.tensordesc<64x64xf16, #shared>, %arg1: i32, %arg2: i32, %arg3: i64, %arg4: i64, %arg5: !tt.tensordesc<64x64xf16, #shared>, %arg6: i32, %arg7: i32, %arg8: i64, %arg9: i64, %arg10: !tt.tensordesc<64x64xf16, #shared>, %arg11: i32, %arg12: i32, %arg13: i64, %arg14: i64, %arg15: !tt.tensordesc<64x64xf16, #shared>, %arg16: i32, %arg17: i32, %arg18: i64, %arg19: i64, %arg20: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg21: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg22: i32 {tt.divisibility = 16 : i32}, %arg23: i32 {tt.divisibility = 16 : i32}, %arg24: f32) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.descriptor_load %arg0[%1, %c0_i32] : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #blocked>
    %3 = ttg.local_alloc %2 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
    %4 = tt.splat %arg24 : f32 -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
    %5 = tt.splat %arg24 : f32 -> tensor<64x64xf32, #mma>
    %6:3 = scf.for %arg25 = %c0_i32 to %arg23 step %c64_i32 iter_args(%arg26 = %cst_0, %arg27 = %cst_1, %arg28 = %cst) -> (tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64xf32, #mma>)  : i32 {
      %20 = tt.descriptor_load %arg5[%arg25, %c0_i32] : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #blocked>
      %21 = ttg.local_alloc %20 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %22 = ttg.memdesc_trans %21 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xf16, #shared, #smem> -> !ttg.memdesc<64x64xf16, #shared1, #smem>
      %23 = ttng.warp_group_dot %3, %22, %cst {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      %24 = "tt.reduce"(%23) <{axis = 1 : i32}> ({
      ^bb0(%arg29: f32, %arg30: f32):
        %45 = arith.maxnumf %arg29, %arg30 : f32
        tt.reduce.return %45 : f32
      }) : (tensor<64x64xf32, #mma>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %25 = arith.mulf %24, %4 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %26 = arith.maxnumf %arg26, %25 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %27 = arith.mulf %23, %5 : tensor<64x64xf32, #mma>
      %28 = tt.expand_dims %26 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xf32, #mma>
      %29 = tt.broadcast %28 : tensor<64x1xf32, #mma> -> tensor<64x64xf32, #mma>
      %30 = arith.subf %27, %29 : tensor<64x64xf32, #mma>
      %31 = math.exp2 %30 : tensor<64x64xf32, #mma>
      %32 = arith.subf %arg26, %26 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %33 = math.exp2 %32 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %34 = "tt.reduce"(%31) <{axis = 1 : i32}> ({
      ^bb0(%arg29: f32, %arg30: f32):
        %45 = arith.addf %arg29, %arg30 : f32
        tt.reduce.return %45 : f32
      }) : (tensor<64x64xf32, #mma>) -> tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %35 = tt.expand_dims %33 {axis = 1 : i32} : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64x1xf32, #mma>
      %36 = tt.broadcast %35 : tensor<64x1xf32, #mma> -> tensor<64x64xf32, #mma>
      %37 = arith.mulf %arg28, %36 : tensor<64x64xf32, #mma>
      %38 = tt.descriptor_load %arg10[%arg25, %c0_i32] : !tt.tensordesc<64x64xf16, #shared> -> tensor<64x64xf16, #blocked>
      %39 = ttg.local_alloc %38 : (tensor<64x64xf16, #blocked>) -> !ttg.memdesc<64x64xf16, #shared, #smem>
      %40 = arith.truncf %31 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma>
      %41 = ttg.convert_layout %40 : tensor<64x64xf16, #mma> -> tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %42 = ttng.warp_group_dot %41, %39, %37 {inputPrecision = 0 : i32} : tensor<64x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf32, #mma>
      %43 = arith.mulf %arg27, %33 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      %44 = arith.addf %43, %34 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>
      scf.yield %26, %44, %42 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>>, tensor<64x64xf32, #mma>
    } {tt.num_stages = 2 : i32, tt.warp_specialize}
    %7 = arith.truncf %6#2 : tensor<64x64xf32, #mma> to tensor<64x64xf16, #mma>
    %8 = ttg.convert_layout %7 : tensor<64x64xf16, #mma> -> tensor<64x64xf16, #blocked>
    tt.descriptor_store %arg15[%1, %c0_i32], %8 : !tt.tensordesc<64x64xf16, #shared>, tensor<64x64xf16, #blocked>
    %9 = tt.addptr %arg20, %1 : !tt.ptr<f16>, i32
    %10 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked1>
    %11 = tt.splat %9 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked1>
    %12 = tt.addptr %11, %10 : tensor<64x!tt.ptr<f16>, #blocked1>, tensor<64xi32, #blocked1>
    %13 = arith.truncf %6#1 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf16, #ttg.slice<{dim = 1, parent = #mma}>>
    %14 = ttg.convert_layout %13 : tensor<64xf16, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64xf16, #blocked1>
    tt.store %12, %14 : tensor<64x!tt.ptr<f16>, #blocked1>
    %15 = tt.addptr %arg21, %1 : !tt.ptr<f16>, i32
    %16 = tt.splat %15 : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>, #blocked1>
    %17 = tt.addptr %16, %10 : tensor<64x!tt.ptr<f16>, #blocked1>, tensor<64xi32, #blocked1>
    %18 = arith.truncf %6#0 : tensor<64xf32, #ttg.slice<{dim = 1, parent = #mma}>> to tensor<64xf16, #ttg.slice<{dim = 1, parent = #mma}>>
    %19 = ttg.convert_layout %18 : tensor<64xf16, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<64xf16, #blocked1>
    tt.store %17, %19 : tensor<64x!tt.ptr<f16>, #blocked1>
    tt.return
  }
}
