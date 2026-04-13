// RUN: triton-opt %s -split-input-file -convert-warp-pipeline="arch=gfx1250" | FileCheck %s --check-prefixes CHECK,WAVE32
// RUN: triton-opt %s -split-input-file -convert-warp-pipeline="arch=gfx950" | FileCheck %s --check-prefixes CHECK,WAVE64

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
// CHECK: ttg.barrier local
// CHECK: arith.divsi
// WAVE64-SAME: %c256
// WAVE32-SAME: %c128
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
// CHECK: ttg.barrier local
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
// *  : pipeline border with ttg.barrier local
// |  : end of the loop, begins next iteration.
//
// Dependency comes from the second warp (deferred) to the first warp,
// In this case, local_load(lload) and local_store(lstore) access the same allocation
// we need to insert wait after lload/lstore from the second warp
// and just before lstore/lload in the first warp, that is annotated as (*) above
//
// CHECK-LABEL: tt.func public @eight_stage_dependency
// CHECK-NOT: no_inline
// CHECK: ttg.barrier local
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
// CHECK: ttg.barrier local
// CHECK: tt.dot
// CHECK: s.barrier
// CHECK-COUNT-2: local_store
// CHECK: ttg.barrier local
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
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// CHECK-COUNT-2: local_load
// CHECK: async_copy_global_to_local

// pre-inserted wait should be preserved.
// CHECK: rocdl.sched.barrier
// CHECK: async_wait
// CHECK: rocdl.sched.barrier

// CHECK: async_copy_global_to_local
// CHECK: ttg.barrier local
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
// CHECK-NOT: ttg.barrier
// CHECK-NOT: amdg.cond_barrier
// CHECK: scf.for
// CHECK:   scf.execute_region
// CHECK: tt.return

// -----

// ---- Priority reset: stages without priority reset to 0 when others have it ----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @priority_reset_between_stages(%n: index, %ptr: !tt.ptr<f32>) {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %v0  = arith.constant 0.0 : f32
    %v1  = arith.constant 1.0 : f32

    scf.for %i = %c0 to %n step %c1 {
      // Stage 0 - has priority 3
      scf.execute_region {
        tt.store %ptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "load", triton.warp_pipeline.priority = 3 : i32}

      // Stage 1 - no priority, should reset to 0
      scf.execute_region {
        tt.store %ptr, %v1 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "compute"}

      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    tt.return
  }
}

// CHECK-LABEL: tt.func @priority_reset_between_stages
// Before loop: priority for first cluster
// CHECK: rocdl.s.setprio 3
// CHECK: scf.for
// Inside loop: setprio 0 for second cluster, then setprio 3 for first cluster
// CHECK: rocdl.s.setprio 0
// CHECK: rocdl.s.setprio 3
// After loop: reset to 0
// CHECK: rocdl.s.setprio 0
// CHECK: amdg.cond_barrier
// CHECK: tt.return

// -----

// ---- No priority: no setprio emitted when no stage uses priority ----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @no_priority_no_setprio(%n: index, %ptr: !tt.ptr<f32>) {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %v0  = arith.constant 0.0 : f32
    %v1  = arith.constant 1.0 : f32

    scf.for %i = %c0 to %n step %c1 {
      scf.execute_region {
        tt.store %ptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage0"}

      scf.execute_region {
        tt.store %ptr, %v1 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage1"}

      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    tt.return
  }
}

// CHECK-LABEL: tt.func @no_priority_no_setprio
// CHECK: scf.for
// CHECK-NOT: rocdl.s.setprio
// CHECK: amdg.cond_barrier
// CHECK: tt.return

// -----

// ---- amdg.async_wait recognized as a valid barrier between stages ----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @async_wait_between_stages(%n: index, %ptr: !tt.ptr<f32>) {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %v0  = arith.constant 0.0 : f32
    %v1  = arith.constant 1.0 : f32

    scf.for %i = %c0 to %n step %c1 {
      scf.execute_region {
        tt.store %ptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage1"}

      // amdg.async_wait sits between stages and must be recognized as a barrier.
      amdg.async_wait {num_inst = 0 : i32}

      scf.execute_region {
        tt.store %ptr, %v1 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "stage2"}

      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    tt.return
  }
}

// The pass should succeed (not bail out) and produce barriers.
// CHECK-LABEL: tt.func @async_wait_between_stages
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// CHECK-NOT: scf.execute_region
// CHECK: rocdl.sched.barrier
// CHECK: amdg.cond_barrier
// CHECK: tt.return

// -----

// ---- Back-to-back pipelined loops: redundant cond_barriers eliminated ----
//
// Both loops have local_load (read) in stage0 and local_store (write) in
// stage1, creating a wrap-around ttg.barrier local.  The post-loop
// cond_barrier of loop 1, the pre-barrier of loop 2, and the pre-loop
// cond_barrier of loop 2 should all be eliminated because loop 1's
// wrap-around barrier already includes a local fence.
//
// Expected:
//   ttg.barrier local          (pre-barrier for loop 1)
//   amdg.cond_barrier          (#1 phase shift for loop 1)
//   scf.for { loop 1 }
//   NO amdg.cond_barrier       (#2 eliminated)
//   NO ttg.barrier local       (pre-barrier eliminated)
//   NO amdg.cond_barrier       (#3 eliminated)
//   scf.for { loop 2 }
//   amdg.cond_barrier          (#4 reconverge for loop 2)

#b2b_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#b2b_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#b2b_shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#b2b_smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @back_to_back_elimination(
      %lb: i32, %ub: i32, %step: i32,
      %acc: tensor<256x256xf32, #b2b_mma>,
      %ptr: tensor<256x64x!tt.ptr<f16>, #b2b_blocked>) {

    %smem = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable>

    // Loop 1: stage0 reads LDS, stage1 writes LDS → wrap-around is ttg.barrier local
    %r1:2 = scf.for %i = %lb to %ub step %step
        iter_args(%a1 = %acc, %s1 = %smem)
        -> (tensor<256x256xf32, #b2b_mma>, !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable>) : i32 {
      %ld1 = scf.execute_region -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2b_mma, kWidth = 4}>> no_inline {
        %sub = ttg.memdesc_subslice %s1[0, 0] : !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable> -> !ttg.memdesc<256x16xf16, #b2b_shared, #b2b_smem, mutable, 256x64>
        %v = ttg.local_load %sub : !ttg.memdesc<256x16xf16, #b2b_shared, #b2b_smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2b_mma, kWidth = 4}>>
        scf.yield %v : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2b_mma, kWidth = 4}>>
      } {triton.warp_pipeline.stage = "lds_load"}

      %st1 = scf.execute_region -> !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable> no_inline {
        %data = tt.load %ptr : tensor<256x64x!tt.ptr<f16>, #b2b_blocked>
        ttg.local_store %data, %s1 : tensor<256x64xf16, #b2b_blocked> -> !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable>
        scf.yield %s1 : !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable>
      } {triton.warp_pipeline.stage = "global_load_and_store"}

      scf.yield %a1, %st1 : tensor<256x256xf32, #b2b_mma>, !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable>
    } {triton.warp_pipeline.pipelined_for}

    // Loop 2: same structure (read + write LDS) so it is not optimized away
    %r2:2 = scf.for %j = %lb to %ub step %step
        iter_args(%a2 = %r1#0, %s2 = %r1#1)
        -> (tensor<256x256xf32, #b2b_mma>, !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable>) : i32 {
      %ld2 = scf.execute_region -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2b_mma, kWidth = 4}>> no_inline {
        %sub2 = ttg.memdesc_subslice %s2[0, 0] : !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable> -> !ttg.memdesc<256x16xf16, #b2b_shared, #b2b_smem, mutable, 256x64>
        %v2 = ttg.local_load %sub2 : !ttg.memdesc<256x16xf16, #b2b_shared, #b2b_smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2b_mma, kWidth = 4}>>
        scf.yield %v2 : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2b_mma, kWidth = 4}>>
      } {triton.warp_pipeline.stage = "epilogue_lds_load"}

      %st2 = scf.execute_region -> !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable> no_inline {
        %data2 = tt.load %ptr : tensor<256x64x!tt.ptr<f16>, #b2b_blocked>
        ttg.local_store %data2, %s2 : tensor<256x64xf16, #b2b_blocked> -> !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable>
        scf.yield %s2 : !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable>
      } {triton.warp_pipeline.stage = "epilogue_global_load_and_store"}

      scf.yield %a2, %st2 : tensor<256x256xf32, #b2b_mma>, !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable>
    } {triton.warp_pipeline.pipelined_for}

    ttg.local_dealloc %smem : !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable>
    tt.return
  }
}

// CHECK-LABEL: tt.func @back_to_back_elimination
// Pre-barrier and phase shift for loop 1 are kept.
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// Wrap-around barrier inside loop 1 (local fence from LDS dependency).
// CHECK: ttg.barrier local
// CHECK: scf.yield
// Between the two loops: no cond_barriers, no ttg.barrier local.
// CHECK-NOT: amdg.cond_barrier
// CHECK-NOT: ttg.barrier local
// CHECK: scf.for
// Post-loop reconverge for loop 2 is kept.
// CHECK: amdg.cond_barrier
// CHECK: tt.return

// -----

// ---- Flat (unrolled) pipeline: execute_regions outside scf.for ----
//
// Simulates the output of WarpPipeliner::createFlatPipeline —
// 4 execute_regions from a 2-iteration × 2-stage unrolled epilogue.
// ConvertWarpPipeline should insert pre-barrier, phase shift,
// cluster barriers, priority, and reconverge around them.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @flat_pipeline_backend(%ptr0: !tt.ptr<f32>, %ptr1: !tt.ptr<f32>) {
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32
    %v2 = arith.constant 2.0 : f32
    %v3 = arith.constant 3.0 : f32

    // Iteration 0, stage 0
    scf.execute_region no_inline {
      tt.store %ptr0, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage0_epi", triton.warp_pipeline.priority = 1 : i32}

    // Iteration 0, stage 1
    scf.execute_region no_inline {
      tt.store %ptr1, %v1 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage1_epi", triton.warp_pipeline.priority = 0 : i32}

    // Iteration 1, stage 0
    scf.execute_region no_inline {
      tt.store %ptr0, %v2 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage0_epi", triton.warp_pipeline.priority = 1 : i32}

    // Iteration 1, stage 1
    scf.execute_region no_inline {
      tt.store %ptr1, %v3 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage1_epi", triton.warp_pipeline.priority = 0 : i32}

    tt.return
  }
}

// CHECK-LABEL: tt.func @flat_pipeline_backend
// All execute_regions must be inlined.
// CHECK-NOT: no_inline
//
// Pre-barrier + phase shift.
// CHECK: ttg.barrier local
// CHECK: %[[WARPLOW:.+]] = arith.cmpi eq
// CHECK: %[[WARPHIGH:.+]] = arith.cmpi ne
// CHECK: amdg.cond_barrier %[[WARPHIGH]]
//
// Stage 0 priority.
// CHECK: rocdl.s.setprio 1
// Stage 0 ops (inlined).
// CHECK: tt.store
//
// Cluster barrier between stages 0 and 1.
// CHECK: rocdl.s.setprio 0
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier
// Stage 1 ops.
// CHECK: tt.store
//
// Cluster barrier between iteration 0 stage 1 and iteration 1 stage 0.
// CHECK: rocdl.s.setprio 1
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier
// CHECK: tt.store
//
// Cluster barrier between iteration 1 stages.
// CHECK: rocdl.s.setprio 0
// CHECK: rocdl.sched.barrier
// CHECK: rocdl.s.barrier
// CHECK: rocdl.sched.barrier
// CHECK: tt.store
//
// Post-sequence priority reset + reconverge.
// CHECK: rocdl.s.setprio 0
// CHECK: amdg.cond_barrier %[[WARPLOW]]
// CHECK: tt.return

// -----

// ---- Back-to-back: pipelined scf.for + flat (unrolled) pipeline ----
//
// Loop 1 (scf.for) has local_load in stage0 and local_store in stage1,
// sharing the same LDS allocation → the wrap-around barrier includes a
// ttg.barrier local.  The flat pipeline follows immediately.
//
// The post-loop reconverge of loop 1, the pre-barrier, and the phase
// shift of the flat pipeline should all be eliminated — same logic as
// back-to-back scf.for loops.
//
// Expected:
//   ttg.barrier local          (pre-barrier for loop 1)
//   amdg.cond_barrier          (#1 phase shift for loop 1)
//   scf.for { loop 1 }
//   NO amdg.cond_barrier       (#2 eliminated)
//   NO ttg.barrier local       (pre-barrier eliminated)
//   NO amdg.cond_barrier       (#3 eliminated)
//   [flat pipeline stages]
//   amdg.cond_barrier          (#4 reconverge for flat pipeline)

#b2bf_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#b2bf_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#b2bf_shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#b2bf_smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @back_to_back_for_then_flat(
      %lb: i32, %ub: i32, %step: i32,
      %acc: tensor<256x256xf32, #b2bf_mma>,
      %ptr: tensor<256x64x!tt.ptr<f16>, #b2bf_blocked>,
      %sptr: !tt.ptr<f32>) {

    %smem = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #b2bf_shared, #b2bf_smem, mutable>
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32

    // Loop 1: local_load + local_store → wrap-around is ttg.barrier local
    %r1:2 = scf.for %i = %lb to %ub step %step
        iter_args(%a1 = %acc, %s1 = %smem)
        -> (tensor<256x256xf32, #b2bf_mma>, !ttg.memdesc<256x64xf16, #b2bf_shared, #b2bf_smem, mutable>) : i32 {
      %ld1 = scf.execute_region -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bf_mma, kWidth = 4}>> no_inline {
        %sub = ttg.memdesc_subslice %s1[0, 0] : !ttg.memdesc<256x64xf16, #b2bf_shared, #b2bf_smem, mutable> -> !ttg.memdesc<256x16xf16, #b2bf_shared, #b2bf_smem, mutable, 256x64>
        %v = ttg.local_load %sub : !ttg.memdesc<256x16xf16, #b2bf_shared, #b2bf_smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bf_mma, kWidth = 4}>>
        scf.yield %v : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bf_mma, kWidth = 4}>>
      } {triton.warp_pipeline.stage = "lds_load"}

      %st1 = scf.execute_region -> !ttg.memdesc<256x64xf16, #b2bf_shared, #b2bf_smem, mutable> no_inline {
        %data = tt.load %ptr : tensor<256x64x!tt.ptr<f16>, #b2bf_blocked>
        ttg.local_store %data, %s1 : tensor<256x64xf16, #b2bf_blocked> -> !ttg.memdesc<256x64xf16, #b2bf_shared, #b2bf_smem, mutable>
        scf.yield %s1 : !ttg.memdesc<256x64xf16, #b2bf_shared, #b2bf_smem, mutable>
      } {triton.warp_pipeline.stage = "global_load_and_store"}

      scf.yield %a1, %st1 : tensor<256x256xf32, #b2bf_mma>, !ttg.memdesc<256x64xf16, #b2bf_shared, #b2bf_smem, mutable>
    } {triton.warp_pipeline.pipelined_for}

    // Flat (unrolled) pipeline: 2 stages, simple stores (no LDS dep)
    scf.execute_region no_inline {
      tt.store %sptr, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "flat_stage0"}

    scf.execute_region no_inline {
      tt.store %sptr, %v1 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "flat_stage1"}

    ttg.local_dealloc %smem : !ttg.memdesc<256x64xf16, #b2bf_shared, #b2bf_smem, mutable>
    tt.return
  }
}

// CHECK-LABEL: tt.func @back_to_back_for_then_flat
// Pre-barrier and phase shift for loop 1 are kept.
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// Wrap-around barrier inside loop 1 (local fence from LDS dependency).
// CHECK: ttg.barrier local
// CHECK: scf.yield
// Between loop 1 and flat pipeline: no cond_barriers, no ttg.barrier local.
// CHECK-NOT: amdg.cond_barrier
// CHECK-NOT: ttg.barrier local
// Flat pipeline stages (inlined after conversion).
// CHECK: tt.store
// CHECK: rocdl.s.barrier
// CHECK: tt.store
// Reconverge for flat pipeline is kept.
// CHECK: amdg.cond_barrier
// CHECK: tt.return

// -----

// ---- Flat pipeline with pre-existing barrier between stages ----
//
// When an async_wait (or similar barrier op) already exists between
// flat pipeline stages, the pass should wrap it with sched_barriers
// instead of inserting a redundant s_barrier.
//
// Stage layout: stage0 -- async_wait -- stage1 -- (nothing) -- stage2
//
// Expected between stage0 and stage1:
//   sched_barrier + async_wait + sched_barrier   (wrapped, no s_barrier)
// Expected between stage1 and stage2:
//   sched_barrier + s_barrier + sched_barrier     (inserted, no async_wait)

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @flat_pipeline_existing_barrier(%ptr: !tt.ptr<f32>) {
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32
    %v2 = arith.constant 2.0 : f32

    scf.execute_region no_inline {
      tt.store %ptr, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage0"}

    amdg.async_wait {num_inst = 0 : i32}

    scf.execute_region no_inline {
      tt.store %ptr, %v1 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage1"}

    scf.execute_region no_inline {
      tt.store %ptr, %v2 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "stage2"}

    tt.return
  }
}

// CHECK-LABEL: tt.func @flat_pipeline_existing_barrier
// CHECK-NOT: no_inline
//
// Pre-barrier + phase shift.
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
//
// Stage 0 ops.
// CHECK: tt.store
//
// Between stage 0 and 1: existing async_wait wrapped, no s_barrier.
// CHECK: rocdl.sched.barrier
// CHECK-NEXT: amdg.async_wait
// CHECK-NEXT: rocdl.sched.barrier
// CHECK-NOT: rocdl.s.barrier
// Stage 1 ops.
// CHECK: tt.store
//
// Between stage 1 and 2: no pre-existing barrier, so s_barrier inserted.
// CHECK: rocdl.sched.barrier
// CHECK-NEXT: rocdl.s.barrier
// CHECK-NEXT: rocdl.sched.barrier
// Stage 2 ops.
// CHECK: tt.store
//
// Reconverge.
// CHECK: amdg.cond_barrier
// CHECK: tt.return
