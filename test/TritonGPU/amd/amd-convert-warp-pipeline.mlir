// RUN: triton-opt %s -split-input-file -convert-warp-pipeline | FileCheck %s

// ---- 2-stage pipeline (basic) ----
//

tt.func @two_stage_backend(%n: index, %ptr: !tt.ptr<f32>) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %v0  = arith.constant 0.0 : f32
  %v1  = arith.constant 1.0 : f32

  scf.for %i = %c0 to %n step %c1 {

    // Stage 0 cluster
    scf.execute_region {
      tt.store %ptr, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage0"}

    // Stage 1 cluster
    scf.execute_region {
      tt.store %ptr, %v1 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage1"}

    scf.yield
  } {triton.warp_pipeline.pipelined_for}

  tt.return
}

// CHECK-LABEL: tt.func @two_stage_backend(
// CHECK: %c0 = arith.constant 0 : index
// CHECK: %c1 = arith.constant 1 : index
// CHECK-NOT: no_inline

// === Pre-loop sync + role setup ===
// CHECK: gpu.barrier
// CHECK: arith.divsi
// CHECK: %[[WARPLOW:.+]] = arith.cmpi eq
// CHECK: %[[WARPHIGH:.+]] = arith.cmpi ne
// CHECK: amdg.cond_barrier %[[WARPHIGH]]

// After conversion, the for body is flattened and cluster barriers inserted.
// CHECK: scf.for
// CHECK-NOT:   scf.execute_region
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier
// CHECK-NOT:   scf.execute_region

// CHECK: amdg.cond_barrier %[[WARPLOW]]
// CHECK: tt.return


// ---- 3-stage pipeline (ensures multiple clusters handled) ----

tt.func @three_stage_backend(%n: index, %ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %v0  = arith.constant 0.0 : f32
  %v1  = arith.constant 1.0 : f32
  %v2  = arith.constant 2.0 : f32

  scf.for %i = %c0 to %n step %c1 {

    // Stage 0
    scf.execute_region {
      tt.store %ptr0, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage0"}

    // Stage 1
    scf.execute_region {
      tt.store %ptr0, %v1 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage1"}

    // Stage 2
    scf.execute_region {
      tt.store %ptr1, %v2 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage2"}

    scf.yield
  } {triton.warp_pipeline.pipelined_for}

  tt.return
}

// CHECK-LABEL: tt.func @three_stage_backend(
// CHECK-NOT: no_inline
// CHECK: gpu.barrier
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// CHECK-NOT:   scf.execute_region
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier
// CHECK: amdg.cond_barrier
// CHECK: tt.return


// -- 8-stage pipeline dependency check ----
//
// 0: <lload>-<dot  >-<lload>-<dot  >-<lload>-<dot  >-<lstore>-<dot  >|<lload>-<dot  >-<lload>-<dot  >
// 1:         <lload>-<dot  >-<lload>-<dot  >-<lload>*<dot  >-<lstore>*<dot  >|<lload>-<dot  >-<lload>-<dot>
// < > : a pipeline cluster, relevant operation in it.
// -  : pipeline border with s.barrier
// *  : pipeline border with local_barrier
// |  : end of the loop, begins next iteration.
//
// Dependency comes from the second warp (deferred) to the first warp,
// In this case, local_load(lload) and local_store(lstore) access the same allocation
// we need to insert wait after lload/lstore from the second warp
// and just before lstore/lload in the first warp, that is annotated as (*) above
//
// CHECK-LABEL: tt.func public @eight_stage_dependency
// CHECK-NOT: no_inline
// CHECK: gpu.barrier
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// CHECK-COUNT-2: local_load
// CHECK: s.barrier
// CHECK: tt.dot
// CHECK: s.barrier
// CHECK-COUNT-2: local_load
// CHECK: s.barrier
// CHECK: tt.dot
// CHECK: s.barrier
// CHECK-COUNT-4: local_load
// CHECK: local_barrier
// CHECK: tt.dot
// CHECK: s.barrier
// CHECK-COUNT-2: local_store
// CHECK: local_barrier
// CHECK: tt.dot
// CHECK: s.barrier
// CHECK: scf.yield
// CHECK: amdg.cond_barrier

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#shared1 = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @eight_stage_dependency(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: tensor<256x256xf32, #mma>, %arg4: tensor<64x256xi32, #blocked>, %arg5: tensor<256x64xi32, #blocked1>, %arg6: tensor<256x64x!tt.ptr<f16>, #blocked1>, %arg7: tensor<64x256x!tt.ptr<f16>, #blocked>, %arg8: !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, %arg9: !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>) {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<1x64x256xf16, #shared1, #smem, mutable>
    %2:6 = scf.for %arg10 = %arg0 to %arg1 step %arg2 iter_args(%arg11 = %arg3, %arg12 = %arg6, %arg13 = %arg7, %arg14 = %arg0, %arg15 = %arg8, %arg16 = %arg9) -> (tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>)  : i32 {
      %3:5 = scf.execute_region -> (tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xf16, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) no_inline {
        %11 = tt.addptr %arg12, %arg5 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xi32, #blocked1>
        %12 = tt.load %11 : tensor<256x64x!tt.ptr<f16>, #blocked1>
        %13 = tt.addptr %arg13, %arg4 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked>
        %14 = ttg.memdesc_subslice %arg15[0, 0] : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x16xf16, #shared, #smem, mutable, 256x64>
        %15 = ttg.local_load %14 : !ttg.memdesc<256x16xf16, #shared, #smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %16 = ttg.memdesc_subslice %arg16[0, 0] : !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x256xf16, #shared1, #smem, mutable, 64x256>
        %17 = ttg.local_load %16 : !ttg.memdesc<16x256xf16, #shared1, #smem, mutable, 64x256> -> tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
        scf.yield %11, %12, %13, %15, %17 : tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<256x64xf16, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      } {triton.warp_pipeline.stage = "stage"}
      %4 = scf.execute_region -> tensor<256x256xf32, #mma> no_inline {
        %11 = tt.dot %3#3, %3#4, %arg11 : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
        scf.yield %11 : tensor<256x256xf32, #mma>
      } {triton.warp_pipeline.stage = "stage"}
      %5:3 = scf.execute_region -> (tensor<64x256xf16, #blocked>, tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) no_inline {
        %11 = tt.load %3#2 : tensor<64x256x!tt.ptr<f16>, #blocked>
        %12 = ttg.memdesc_subslice %arg15[0, 16] : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x16xf16, #shared, #smem, mutable, 256x64>
        %13 = ttg.local_load %12 : !ttg.memdesc<256x16xf16, #shared, #smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %14 = ttg.memdesc_subslice %arg16[16, 0] : !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x256xf16, #shared1, #smem, mutable, 64x256>
        %15 = ttg.local_load %14 : !ttg.memdesc<16x256xf16, #shared1, #smem, mutable, 64x256> -> tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
        scf.yield %11, %13, %15 : tensor<64x256xf16, #blocked>, tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      } {triton.warp_pipeline.stage = "stage"}
      %6 = scf.execute_region -> tensor<256x256xf32, #mma> no_inline {
        %11 = tt.dot %5#1, %5#2, %4 : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
        scf.yield %11 : tensor<256x256xf32, #mma>
      } {triton.warp_pipeline.stage = "stage"}
      %7:4 = scf.execute_region -> (tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>) no_inline {
        %11 = ttg.memdesc_subslice %arg15[0, 32] : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x16xf16, #shared, #smem, mutable, 256x64>
        %12 = ttg.local_load %11 : !ttg.memdesc<256x16xf16, #shared, #smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %13 = ttg.memdesc_subslice %arg16[32, 0] : !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x256xf16, #shared1, #smem, mutable, 64x256>
        %14 = ttg.local_load %13 : !ttg.memdesc<16x256xf16, #shared1, #smem, mutable, 64x256> -> tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
        %15 = ttg.memdesc_subslice %arg15[0, 48] : !ttg.memdesc<256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x16xf16, #shared, #smem, mutable, 256x64>
        %16 = ttg.local_load %15 : !ttg.memdesc<256x16xf16, #shared, #smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
        %17 = ttg.memdesc_subslice %arg16[48, 0] : !ttg.memdesc<64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<16x256xf16, #shared1, #smem, mutable, 64x256>
        %18 = ttg.local_load %17 : !ttg.memdesc<16x256xf16, #shared1, #smem, mutable, 64x256> -> tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
        scf.yield %12, %14, %16, %18 : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>, tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>, tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>>
      } {triton.warp_pipeline.stage = "stage"}
      %8 = scf.execute_region -> tensor<256x256xf32, #mma> no_inline {
        %11 = tt.dot %7#0, %7#1, %6 : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
        scf.yield %11 : tensor<256x256xf32, #mma>
      } {triton.warp_pipeline.stage = "stage"}
      %9:3 = scf.execute_region -> (i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>) no_inline {
        %11 = arith.addi %arg14, %arg2 : i32
        %12 = arith.cmpi slt, %11, %arg2 : i32
        %13 = arith.select %12, %11, %arg0 : i32
        %14 = ttg.memdesc_index %0[%13] : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        ttg.local_store %3#1, %14 : tensor<256x64xf16, #blocked1> -> !ttg.memdesc<256x64xf16, #shared, #smem, mutable>
        %15 = ttg.memdesc_index %1[%13] : !ttg.memdesc<1x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
        ttg.local_store %5#0, %15 : tensor<64x256xf16, #blocked> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
        scf.yield %13, %14, %15 : i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
      } {triton.warp_pipeline.stage = "stage"}
      %10 = scf.execute_region -> tensor<256x256xf32, #mma> no_inline {
        %11 = tt.dot %7#2, %7#3, %8 : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * tensor<16x256xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 4}>> -> tensor<256x256xf32, #mma>
        scf.yield %11 : tensor<256x256xf32, #mma>
      } {triton.warp_pipeline.stage = "stage"}
      scf.yield %10, %3#0, %3#2, %9#0, %9#1, %9#2 : tensor<256x256xf32, #mma>, tensor<256x64x!tt.ptr<f16>, #blocked1>, tensor<64x256x!tt.ptr<f16>, #blocked>, i32, !ttg.memdesc<256x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable>
    } {triton.warp_pipeline.pipelined_for}
    ttg.local_dealloc %0 : !ttg.memdesc<1x256x64xf16, #shared, #smem, mutable>
    ttg.local_dealloc %1 : !ttg.memdesc<1x64x256xf16, #shared1, #smem, mutable>
    tt.return
  }
}

// -- Triple buffered 2-stage pipeline dependency check ----
// Currently little conservative, there could be more chance to optimize local_wait
//
// CHECK-LABEL: tt.func public @triple_buf_2stage
// CHECK-NOT: no_inline
// CHECK: gpu.barrier
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// CHECK-COUNT-2: local_load
// CHECK: async_copy_global_to_local

// pre-inserted wait should be preserved.
// CHECK: rocdl.sched.barrier
// CHECK: async_wait
// CHECK: rocdl.sched.barrier

// CHECK: async_copy_global_to_local
// CHECK: local_barrier
// CHECK: scf.yield
// CHECK: amdg.cond_barrier

#linear = #ttg.linear<{register = [[1, 0], [2, 0], [4, 0], [0, 4]], lane = [[8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16]], warp = [[0, 1], [0, 2], [0, 8]], block = []}>
#linear1 = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [4, 0]], lane = [[0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0]], warp = [[1, 0], [2, 0], [8, 0]], block = []}>
#mma2 = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
#shrd_a = #ttg.padded_shared<[512:+16] {offset = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0], [32, 0], [64, 0], [128, 0], [0, 16], [0, 1], [0, 2], [0, 8], [0, 4]], block = []}>
#shrd1 = #ttg.padded_shared<[512:+16] {offset = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32], [0, 64], [0, 128], [16, 0], [1, 0], [2, 0], [8, 0], [4, 0]], block = []}>
#shmem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @triple_buf_2stage(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: tensor<256x256xf32, #mma2>, %arg5: i32, %arg6: i32, %arg7: tensor<256x32xi32, #linear>, %arg8: tensor<32x256xi32, #linear1>, %arg9: !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable>, %arg10: !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable>, %arg11: !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable>, %arg12: !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable>, %arg13: !ttg.async.token, %arg14: !ttg.async.token, %arg15: !ttg.async.token, %arg16: tensor<256x32x!tt.ptr<bf16>, #linear>, %arg17: tensor<32x256x!tt.ptr<bf16>, #linear1>, %arg18: tensor<256xi64, #ttg.slice<{dim = 1, parent = #mma2}>>, %arg19: tensor<256xi64, #ttg.slice<{dim = 0, parent = #mma2}>>, %arg20: i64, %arg21: i64, %arg22: !tt.ptr<bf16>, %arg23: i32) attributes {noinline = false} {
    %0 = ttg.local_alloc : () -> !ttg.memdesc<3x256x32xbf16, #shrd_a, #shmem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<3x32x256xbf16, #shrd1, #shmem, mutable>
    %2:11 = scf.for %arg24 = %arg0 to %arg6 step %arg1 iter_args(%arg25 = %arg4, %arg26 = %arg1, %arg27 = %arg9, %arg28 = %arg11, %arg29 = %arg13, %arg30 = %arg10, %arg31 = %arg12, %arg32 = %arg14, %arg33 = %arg15, %arg34 = %arg16, %arg35 = %arg17) -> (tensor<256x256xf32, #mma2>, i32, !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable>, !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable>, !ttg.async.token, !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable>, !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable>, !ttg.async.token, !ttg.async.token, tensor<256x32x!tt.ptr<bf16>, #linear>, tensor<32x256x!tt.ptr<bf16>, #linear1>)  : i32 {
      %32:8 = scf.execute_region -> (tensor<256x32x!tt.ptr<bf16>, #linear>, tensor<32x256x!tt.ptr<bf16>, #linear1>, i32, !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable>, !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable>, tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>>, tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>>, !ttg.async.token) no_inline {
        %35 = tt.addptr %arg34, %arg7 : tensor<256x32x!tt.ptr<bf16>, #linear>, tensor<256x32xi32, #linear>
        %36 = tt.addptr %arg35, %arg8 : tensor<32x256x!tt.ptr<bf16>, #linear1>, tensor<32x256xi32, #linear1>
        %37 = arith.addi %arg26, %arg1 : i32
        %38 = arith.cmpi slt, %37, %arg3 : i32
        %39 = arith.select %38, %37, %arg0 : i32
        %40 = ttg.memdesc_index %0[%39] : !ttg.memdesc<3x256x32xbf16, #shrd_a, #shmem, mutable> -> !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable>
        %41 = ttg.memdesc_index %1[%39] : !ttg.memdesc<3x32x256xbf16, #shrd1, #shmem, mutable> -> !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable>
        %42 = ttg.local_load %arg27 token %arg29 : !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable> -> tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>>
        %43 = ttg.local_load %arg30 token %arg29 : !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>>
        %44 = ttg.async_copy_global_to_local %35, %40 : tensor<256x32x!tt.ptr<bf16>, #linear> -> <256x32xbf16, #shrd_a, #shmem, mutable>
        %45 = ttg.async_commit_group tokens %44
        scf.yield %35, %36, %39, %40, %41, %42, %43, %45 : tensor<256x32x!tt.ptr<bf16>, #linear>, tensor<32x256x!tt.ptr<bf16>, #linear1>, i32, !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable>, !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable>, tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>>, tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>>, !ttg.async.token
      } {triton.warp_pipeline.stage = "stage"}
      %33 = ttg.async_wait %arg32, %arg33 {num = 0 : i32}
      %34:2 = scf.execute_region -> (!ttg.async.token, tensor<256x256xf32, #mma2>) no_inline {
        %35 = ttg.async_copy_global_to_local %32#1, %32#4 : tensor<32x256x!tt.ptr<bf16>, #linear1> -> <32x256xbf16, #shrd1, #shmem, mutable>
        %36 = ttg.async_commit_group tokens %35
        %37 = tt.dot %32#5, %32#6, %arg25 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>> -> tensor<256x256xf32, #mma2>
        scf.yield %36, %37 : !ttg.async.token, tensor<256x256xf32, #mma2>
      } {triton.warp_pipeline.stage = "stage"}
      scf.yield %34#1, %32#2, %arg28, %32#3, %33, %arg31, %32#4, %32#7, %34#0, %32#0, %32#1 : tensor<256x256xf32, #mma2>, i32, !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable>, !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable>, !ttg.async.token, !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable>, !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable>, !ttg.async.token, !ttg.async.token, tensor<256x32x!tt.ptr<bf16>, #linear>, tensor<32x256x!tt.ptr<bf16>, #linear1>
    } {triton.warp_pipeline.pipelined_for}
    %3 = arith.cmpi sge, %arg5, %arg1 : i32
    %4 = arith.cmpi sge, %arg5, %arg2 : i32
    %5 = ttg.local_load %2#2 token %2#4 : !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable> -> tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>>
    %6 = ttg.local_load %2#5 token %2#4 : !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>>
    %7 = scf.if %3 -> (tensor<256x256xf32, #mma2>) {
      %32 = tt.dot %5, %6, %2#0 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>> -> tensor<256x256xf32, #mma2>
      scf.yield %32 : tensor<256x256xf32, #mma2>
    } else {
      scf.yield %2#0 : tensor<256x256xf32, #mma2>
    }
    %8 = ttg.async_wait %2#7, %2#8 {num = 0 : i32}
    %9 = arith.select %3, %7, %2#0 : tensor<256x256xf32, #mma2>
    %10 = ttg.local_load %2#3 token %8 : !ttg.memdesc<256x32xbf16, #shrd_a, #shmem, mutable> -> tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>>
    %11 = ttg.local_load %2#6 token %8 : !ttg.memdesc<32x256xbf16, #shrd1, #shmem, mutable> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>>
    %12 = scf.if %4 -> (tensor<256x256xf32, #mma2>) {
      %32 = tt.dot %10, %11, %9 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma2, kWidth = 8}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma2, kWidth = 8}>> -> tensor<256x256xf32, #mma2>
      scf.yield %32 : tensor<256x256xf32, #mma2>
    } else {
      scf.yield %9 : tensor<256x256xf32, #mma2>
    }
    %13 = arith.select %4, %12, %9 : tensor<256x256xf32, #mma2>
    ttg.local_dealloc %1 : !ttg.memdesc<3x32x256xbf16, #shrd1, #shmem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<3x256x32xbf16, #shrd_a, #shmem, mutable>
    tt.return
  }
}


// -- Negative: no total_stages → pass should not touch the loop ----
//

tt.func @no_total_stages(%n: index, %ptr: !tt.ptr<f32>) {
  %c0  = arith.constant 0 : index
  %c1  = arith.constant 1 : index
  %v0  = arith.constant 3.0 : f32

  scf.for %i = %c0 to %n step %c1 {
    scf.execute_region {
      tt.store %ptr, %v0 : !tt.ptr<f32>
      scf.yield
    }
    scf.yield
  }

  tt.return
}

// CHECK-LABEL: tt.func @no_total_stages(
// CHECK-NOT: gpu.barrier
// CHECK-NOT: amdg.cond_barrier
// CHECK: scf.for
// CHECK:   scf.execute_region
// CHECK: tt.return
