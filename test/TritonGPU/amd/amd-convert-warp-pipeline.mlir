// RUN: triton-opt %s -split-input-file -convert-warp-pipeline="gfx-arch=gfx1250" | FileCheck %s --check-prefixes CHECK,WAVE32
// RUN: triton-opt %s -split-input-file -convert-warp-pipeline="gfx-arch=gfx950" | FileCheck %s --check-prefixes CHECK,WAVE64

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

// ---- Back-to-back: cross-pipeline LDS dep covered by A's wrap-around ----
//
// Both loops access the same shared buffer (read + write).  Loop 1's
// stage1 writes smem and loop 2's stage0 reads it — a cross-pipeline RAW.
//
// Loop 1's wrap-around barrier (bars[0]) is LOCAL because of the in-loop
// RAW between stage1 (write) and the next iteration's stage0 (read).
// That barrier physically sits at the bottom of loop 1's body and is the
// most recent LDS sync after the loop exits, so it already covers the
// (a1, b0) cross-pipeline dep at the boundary.  The boundary barriers
// can therefore be eliminated.
//
// Expected:
//   ttg.barrier local          (pre-barrier for loop 1)
//   amdg.cond_barrier          (#1 phase shift for loop 1)
//   scf.for { loop 1 }
//   NO amdg.cond_barrier       (#2 eliminated — wrap-around covers)
//   NO ttg.barrier local       (prelude eliminated)
//   NO amdg.cond_barrier       (#3 eliminated)
//   scf.for { loop 2 }
//   amdg.cond_barrier          (#4 post-loop reconverge for loop 2)

#b2b_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#b2b_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#b2b_shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#b2b_smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @back_to_back_wrap_around_covers_dep(
      %lb: i32, %ub: i32, %step: i32,
      %acc: tensor<256x256xf32, #b2b_mma>,
      %ptr: tensor<256x64x!tt.ptr<f16>, #b2b_blocked>) {

    %smem = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #b2b_shared, #b2b_smem, mutable>

    // Loop 1: stage0 reads LDS, stage1 writes LDS
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

    // Loop 2: same structure — reads+writes the SAME buffer → cross-pipeline RAW
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

// CHECK-LABEL: tt.func @back_to_back_wrap_around_covers_dep
// Pre-barrier and phase shift for loop 1.
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// Wrap-around barrier inside loop 1 (LOCAL — covers cross-pipeline dep).
// CHECK: ttg.barrier local
// CHECK: scf.yield
// Boundary barriers are eliminated: A's wrap-around already provides the
// LDS sync needed for loop 2's first read; phase carries over.
// CHECK-NOT: amdg.cond_barrier
// CHECK-NOT: ttg.barrier local
// CHECK: scf.for
// Post-loop reconverge for loop 2.
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
// Loop 1 (scf.for) followed immediately by a flat pipeline with no
// intervening operations.  The post-loop reconverge, prelude barrier,
// and phase shift are all eliminated — same logic as back-to-back
// scf.for loops.
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
// Wrap-around barrier inside loop 1.
// CHECK: ttg.barrier local
// CHECK: scf.yield
// Between loop 1 and flat pipeline: no cond_barriers, no ttg.barrier local
// (no intervening ops → phase carries over, prelude barrier redundant).
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

// -----

// ---- Back-to-back: no cross-pipeline LDS dep → barriers eliminated ----
//
// Loop 1 reads+writes shared memory.  Loop 2 only does global ops (no LDS).
// No cross-pipeline LDS dependency exists, so the boundary barriers are
// safely eliminated and the phase carries over.
//
// Expected:
//   ttg.barrier local          (pre-barrier for loop 1)
//   amdg.cond_barrier          (#1 phase shift for loop 1)
//   scf.for { loop 1 }
//   NO amdg.cond_barrier       (eliminated)
//   NO ttg.barrier local       (eliminated)
//   NO amdg.cond_barrier       (eliminated)
//   scf.for { loop 2 }
//   amdg.cond_barrier          (#4 reconverge for loop 2)

#b2bnd_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#b2bnd_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#b2bnd_shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#b2bnd_smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @back_to_back_no_dep_elimination(
      %lb: i32, %ub: i32, %step: i32,
      %acc: tensor<256x256xf32, #b2bnd_mma>,
      %ptr: tensor<256x64x!tt.ptr<f16>, #b2bnd_blocked>,
      %gptr: !tt.ptr<f32>) {

    %smem = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #b2bnd_shared, #b2bnd_smem, mutable>
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32

    // Loop 1: stage0 reads LDS, stage1 writes LDS
    %r1:2 = scf.for %i = %lb to %ub step %step
        iter_args(%a1 = %acc, %s1 = %smem)
        -> (tensor<256x256xf32, #b2bnd_mma>, !ttg.memdesc<256x64xf16, #b2bnd_shared, #b2bnd_smem, mutable>) : i32 {
      %ld1 = scf.execute_region -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bnd_mma, kWidth = 4}>> no_inline {
        %sub = ttg.memdesc_subslice %s1[0, 0] : !ttg.memdesc<256x64xf16, #b2bnd_shared, #b2bnd_smem, mutable> -> !ttg.memdesc<256x16xf16, #b2bnd_shared, #b2bnd_smem, mutable, 256x64>
        %v = ttg.local_load %sub : !ttg.memdesc<256x16xf16, #b2bnd_shared, #b2bnd_smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bnd_mma, kWidth = 4}>>
        scf.yield %v : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bnd_mma, kWidth = 4}>>
      } {triton.warp_pipeline.stage = "lds_load"}

      %st1 = scf.execute_region -> !ttg.memdesc<256x64xf16, #b2bnd_shared, #b2bnd_smem, mutable> no_inline {
        %data = tt.load %ptr : tensor<256x64x!tt.ptr<f16>, #b2bnd_blocked>
        ttg.local_store %data, %s1 : tensor<256x64xf16, #b2bnd_blocked> -> !ttg.memdesc<256x64xf16, #b2bnd_shared, #b2bnd_smem, mutable>
        scf.yield %s1 : !ttg.memdesc<256x64xf16, #b2bnd_shared, #b2bnd_smem, mutable>
      } {triton.warp_pipeline.stage = "global_load_and_store"}

      scf.yield %a1, %st1 : tensor<256x256xf32, #b2bnd_mma>, !ttg.memdesc<256x64xf16, #b2bnd_shared, #b2bnd_smem, mutable>
    } {triton.warp_pipeline.pipelined_for}

    // Loop 2: global-only ops — no LDS access at all
    scf.for %j = %lb to %ub step %step : i32 {
      scf.execute_region no_inline {
        tt.store %gptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "global_store_0"}

      scf.execute_region no_inline {
        tt.store %gptr, %v1 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "global_store_1"}

      scf.yield
    } {triton.warp_pipeline.pipelined_for}

    ttg.local_dealloc %smem : !ttg.memdesc<256x64xf16, #b2bnd_shared, #b2bnd_smem, mutable>
    tt.return
  }
}

// CHECK-LABEL: tt.func @back_to_back_no_dep_elimination
// Pre-barrier and phase shift for loop 1.
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// Wrap-around barrier inside loop 1.
// CHECK: ttg.barrier local
// CHECK: scf.yield
// No cross-pipeline LDS dep → barriers eliminated, phase carries over.
// CHECK-NOT: amdg.cond_barrier
// CHECK-NOT: ttg.barrier local
// CHECK: scf.for
// Post-loop reconverge for loop 2.
// CHECK: amdg.cond_barrier
// CHECK: tt.return

// -----

// ---- Back-to-back: cross-pipeline dep covered by loop A's barrier ----
//
// Loop 1 has 3 stages: stage0 writes LDS, stage1 reads LDS, stage2 is
// compute-only.  The circular dependency analysis places a LOCAL barrier
// between stage1 and stage2 (covering the WAR from stage1 reading what
// stage0 wrote).
//
// Loop 2 has 2 stages: stage0 reads the SAME LDS buffer, stage1 is
// compute-only.  There IS a cross-pipeline dependency (loop1.stage0 writes
// smem that loop2.stage0 reads), but it is already covered by loop 1's
// barrier between stage1 and stage2.
//
// At the boundary with no barrier: warp0 runs b0, warp1 runs a2.
// Since a2 has no LDS access and the LOCAL barrier before a2 already
// flushed all prior LDS writes, b0's read is safe.
//
// Expected:
//   ttg.barrier local          (pre-barrier for loop 1)
//   amdg.cond_barrier          (phase shift for loop 1)
//   scf.for { loop 1 — 3 stages }
//   NO amdg.cond_barrier       (eliminated)
//   NO ttg.barrier local       (eliminated)
//   NO amdg.cond_barrier       (eliminated)
//   scf.for { loop 2 — 2 stages }
//   amdg.cond_barrier          (reconverge for loop 2)

#b2bcov_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#b2bcov_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#b2bcov_shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#b2bcov_smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @back_to_back_dep_covered_elimination(
      %lb: i32, %ub: i32, %step: i32,
      %acc: tensor<256x256xf32, #b2bcov_mma>,
      %ptr: tensor<256x64x!tt.ptr<f16>, #b2bcov_blocked>,
      %gptr: !tt.ptr<f32>) {

    %smem = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #b2bcov_shared, #b2bcov_smem, mutable>
    %v0 = arith.constant 0.0 : f32

    // Loop 1: 3 stages
    //   stage0: writes LDS (local_store)
    //   stage1: reads LDS  (local_load) → RAW with stage0
    //   stage2: compute-only (global store, no LDS)
    // Circular analysis: barrier between stage1 and stage2 is LOCAL.
    %r1:2 = scf.for %i = %lb to %ub step %step
        iter_args(%a1 = %acc, %s1 = %smem)
        -> (tensor<256x256xf32, #b2bcov_mma>, !ttg.memdesc<256x64xf16, #b2bcov_shared, #b2bcov_smem, mutable>) : i32 {
      %st1 = scf.execute_region -> !ttg.memdesc<256x64xf16, #b2bcov_shared, #b2bcov_smem, mutable> no_inline {
        %data = tt.load %ptr : tensor<256x64x!tt.ptr<f16>, #b2bcov_blocked>
        ttg.local_store %data, %s1 : tensor<256x64xf16, #b2bcov_blocked> -> !ttg.memdesc<256x64xf16, #b2bcov_shared, #b2bcov_smem, mutable>
        scf.yield %s1 : !ttg.memdesc<256x64xf16, #b2bcov_shared, #b2bcov_smem, mutable>
      } {triton.warp_pipeline.stage = "global_load_and_store"}

      %ld1 = scf.execute_region -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bcov_mma, kWidth = 4}>> no_inline {
        %sub = ttg.memdesc_subslice %s1[0, 0] : !ttg.memdesc<256x64xf16, #b2bcov_shared, #b2bcov_smem, mutable> -> !ttg.memdesc<256x16xf16, #b2bcov_shared, #b2bcov_smem, mutable, 256x64>
        %v = ttg.local_load %sub : !ttg.memdesc<256x16xf16, #b2bcov_shared, #b2bcov_smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bcov_mma, kWidth = 4}>>
        scf.yield %v : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bcov_mma, kWidth = 4}>>
      } {triton.warp_pipeline.stage = "lds_load"}

      scf.execute_region no_inline {
        tt.store %gptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "compute"}

      scf.yield %a1, %s1 : tensor<256x256xf32, #b2bcov_mma>, !ttg.memdesc<256x64xf16, #b2bcov_shared, #b2bcov_smem, mutable>
    } {triton.warp_pipeline.pipelined_for}

    // Loop 2: stage0 reads the SAME LDS buffer, stage1 is compute-only.
    // Cross-pipeline dep (a0 writes → b0 reads) is covered by loop 1's
    // barrier between stage1 and stage2.
    %r2:2 = scf.for %j = %lb to %ub step %step
        iter_args(%a2 = %r1#0, %s2 = %r1#1)
        -> (tensor<256x256xf32, #b2bcov_mma>, !ttg.memdesc<256x64xf16, #b2bcov_shared, #b2bcov_smem, mutable>) : i32 {
      %ld2 = scf.execute_region -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bcov_mma, kWidth = 4}>> no_inline {
        %sub2 = ttg.memdesc_subslice %s2[0, 0] : !ttg.memdesc<256x64xf16, #b2bcov_shared, #b2bcov_smem, mutable> -> !ttg.memdesc<256x16xf16, #b2bcov_shared, #b2bcov_smem, mutable, 256x64>
        %v2 = ttg.local_load %sub2 : !ttg.memdesc<256x16xf16, #b2bcov_shared, #b2bcov_smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bcov_mma, kWidth = 4}>>
        scf.yield %v2 : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #b2bcov_mma, kWidth = 4}>>
      } {triton.warp_pipeline.stage = "epilogue_lds_load"}

      scf.execute_region no_inline {
        tt.store %gptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "epilogue_compute"}

      scf.yield %a2, %s2 : tensor<256x256xf32, #b2bcov_mma>, !ttg.memdesc<256x64xf16, #b2bcov_shared, #b2bcov_smem, mutable>
    } {triton.warp_pipeline.pipelined_for}

    ttg.local_dealloc %smem : !ttg.memdesc<256x64xf16, #b2bcov_shared, #b2bcov_smem, mutable>
    tt.return
  }
}

// CHECK-LABEL: tt.func @back_to_back_dep_covered_elimination
// Pre-barrier and phase shift for loop 1.
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// Loop 1 has 3 stages; barrier between stage1→stage2 is LOCAL (covers dep).
// CHECK: ttg.barrier local
// CHECK: scf.yield
// Cross-pipeline dep IS covered by loop 1's internal barrier →
// boundary barriers eliminated, phase carries over.
// CHECK-NOT: amdg.cond_barrier
// CHECK-NOT: ttg.barrier local
// CHECK: scf.for
// Post-loop reconverge for loop 2.
// CHECK: amdg.cond_barrier
// CHECK: tt.return

// -----

// ---- Adjacent-stage LDS dependency: barrier must be LOCAL ----
//
// 3-stage loop pipeline where stage0 writes LDS and stage1 reads it.
// Stage2 has no LDS access.
//
// The distance-2+ analysis only checks pairs separated by ≥2 clusters,
// so it never examines (stage0, stage1) directly.  Without the adjacent-
// stage check, the barrier between stage0 and stage1 would be emitted as
// a plain s_barrier, and ModuleMembarAnalysis would later insert a
// redundant ttg.barrier local inside the pipeline — breaking timing.
//
// With the adjacent-stage check:
//   bars[0] (wrap-around) = false  (a2 no LDS, a0 writes — no conflict)
//   bars[1] (a0→a1)       = true   (a0 writes, a1 reads — RAW)
//   bars[2] (a1→a2)       = true   (a1→a0 WAR via distance-2)
//
// Expected inside the loop body:
//   stage0 ops  (local_store)
//   ttg.barrier local             (bars[1] — adjacent dep)
//   stage1 ops  (local_load)
//   ttg.barrier local             (bars[2] — distance-2 dep)
//   stage2 ops  (global store)
//   rocdl.s.barrier               (bars[0] — wrap-around, no LDS dep)
//   scf.yield

#adj_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#adj_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#adj_dot = #ttg.dot_op<{opIdx = 0, parent = #adj_mma, kWidth = 4}>
#adj_shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#adj_smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @adjacent_stage_lds_dep(
      %lb: i32, %ub: i32, %step: i32,
      %acc: tensor<256x16xf16, #adj_dot>,
      %ptr: tensor<256x64x!tt.ptr<f16>, #adj_blocked>,
      %gptr: !tt.ptr<f32>) {

    %smem = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #adj_shared, #adj_smem, mutable>
    %v0 = arith.constant 0.0 : f32

    // The local_load result must be carried as an iter_arg so it is not
    // DCE'd — otherwise the barrier between stage0 and stage1 would merge
    // with the barrier between stage1 and stage2.
    %r:3 = scf.for %i = %lb to %ub step %step
        iter_args(%a = %acc, %s = %smem, %prev = %acc)
        -> (tensor<256x16xf16, #adj_dot>, !ttg.memdesc<256x64xf16, #adj_shared, #adj_smem, mutable>, tensor<256x16xf16, #adj_dot>) : i32 {

      // Stage 0: writes LDS
      %st = scf.execute_region -> !ttg.memdesc<256x64xf16, #adj_shared, #adj_smem, mutable> no_inline {
        %data = tt.load %ptr : tensor<256x64x!tt.ptr<f16>, #adj_blocked>
        ttg.local_store %data, %s : tensor<256x64xf16, #adj_blocked> -> !ttg.memdesc<256x64xf16, #adj_shared, #adj_smem, mutable>
        scf.yield %s : !ttg.memdesc<256x64xf16, #adj_shared, #adj_smem, mutable>
      } {triton.warp_pipeline.stage = "global_load_and_store"}

      // Stage 1: reads LDS — RAW dep with stage 0
      %ld = scf.execute_region -> tensor<256x16xf16, #adj_dot> no_inline {
        %sub = ttg.memdesc_subslice %s[0, 0] : !ttg.memdesc<256x64xf16, #adj_shared, #adj_smem, mutable> -> !ttg.memdesc<256x16xf16, #adj_shared, #adj_smem, mutable, 256x64>
        %v = ttg.local_load %sub : !ttg.memdesc<256x16xf16, #adj_shared, #adj_smem, mutable, 256x64> -> tensor<256x16xf16, #adj_dot>
        scf.yield %v : tensor<256x16xf16, #adj_dot>
      } {triton.warp_pipeline.stage = "lds_load"}

      // Stage 2: compute-only — no LDS access
      scf.execute_region no_inline {
        tt.store %gptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "compute"}

      scf.yield %a, %s, %ld : tensor<256x16xf16, #adj_dot>, !ttg.memdesc<256x64xf16, #adj_shared, #adj_smem, mutable>, tensor<256x16xf16, #adj_dot>
    } {triton.warp_pipeline.pipelined_for}

    ttg.local_dealloc %smem : !ttg.memdesc<256x64xf16, #adj_shared, #adj_smem, mutable>
    tt.return
  }
}

// CHECK-LABEL: tt.func @adjacent_stage_lds_dep
// CHECK: scf.for
//
// Stage 0 ops (local_store).
// CHECK: ttg.local_store
//
// Barrier between stage0→stage1 is LOCAL (adjacent RAW: write→read).
// CHECK: rocdl.sched.barrier
// CHECK-NEXT: ttg.barrier local
// CHECK-NEXT: rocdl.sched.barrier
//
// Stage 1 ops (local_load).
// CHECK: ttg.local_load
//
// Barrier between stage1→stage2 is LOCAL (distance-2 WAR: a1 reads, a0 writes).
// CHECK: rocdl.sched.barrier
// CHECK-NEXT: ttg.barrier local
// CHECK-NEXT: rocdl.sched.barrier
//
// Stage 2 ops (global store).
// CHECK: tt.store
//
// Wrap-around barrier is s_barrier only (a2 has no LDS, a0 writes — no dep).
// CHECK: rocdl.sched.barrier
// CHECK-NEXT: rocdl.s.barrier
// CHECK-NEXT: rocdl.sched.barrier
//
// CHECK: scf.yield

// -----

// ---- Back-to-back: cross-pipeline dep in a later flat stage (b_1) ----
//
// This test exercises `collectNextPipelineClusters` when the next pipeline is
// a flat (unrolled) sequence of more than one stage.  Before the fix, only
// the first stage (b_0) was collected, so a cross-pipeline dependency
// involving a later stage (b_1, b_2, …) was missed and the boundary barriers
// were wrongly eliminated.
//
// Layout:
//   Loop A (2 stages): a_0 tt.store         (no LDS)
//                      a_1 ttg.local_load   (READS LDS)
//   Flat B (2 stages): b_0 tt.store         (no LDS)
//                      b_1 ttg.local_store  (WRITES the same LDS buffer)
//
// A's circular analysis finds no intersecting pair (a_1's read does not
// conflict with itself or with a_0), so all of A's bars are non-LOCAL.
// In particular the wrap-around bars[0] is FALSE, so it cannot seed
// coverage for the merged boundary slot.
//
// Cross-pipeline dep: (a_1, b_1) WAR at merged distance 2, barrierLoc = K = 2
// (the boundary).  No other slot on the path from a_1 to b_1 is LOCAL, so
// the analysis must flag the boundary and preserve the post-loop
// cond_barrier, prelude ttg.barrier local, and phase-shift cond_barrier.
//
// Before the collectNextPipelineClusters fix, the boundary barriers would
// have been removed (false negative) because only b_0 was collected, making
// b_1 invisible to the cross-pipeline analysis.

#crossb_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#crossb_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#crossb_shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#crossb_smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @cross_pipeline_dep_in_b1(
      %lb: i32, %ub: i32, %step: i32,
      %acc: tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossb_mma, kWidth = 4}>>,
      %ptr: tensor<256x64x!tt.ptr<f16>, #crossb_blocked>,
      %gptr: !tt.ptr<f32>,
      %dst: tensor<256x16x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #crossb_mma, kWidth = 4}>>) {

    %smem = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #crossb_shared, #crossb_smem, mutable>
    %v0 = arith.constant 0.0 : f32
    %v1 = arith.constant 1.0 : f32

    // Loop A: stage 0 no LDS, stage 1 reads %smem.  The loaded value is
    // threaded through iter_args + used after the loop so the execute_region
    // (and its ttg.local_load) survives DCE before the redundant-barrier pass.
    %final = scf.for %i = %lb to %ub step %step
        iter_args(%cur = %acc)
        -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossb_mma, kWidth = 4}>> : i32 {
      scf.execute_region no_inline {
        tt.store %gptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "a_compute"}

      %ld = scf.execute_region -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossb_mma, kWidth = 4}>> no_inline {
        %sub = ttg.memdesc_subslice %smem[0, 0] : !ttg.memdesc<256x64xf16, #crossb_shared, #crossb_smem, mutable> -> !ttg.memdesc<256x16xf16, #crossb_shared, #crossb_smem, mutable, 256x64>
        %v = ttg.local_load %sub : !ttg.memdesc<256x16xf16, #crossb_shared, #crossb_smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossb_mma, kWidth = 4}>>
        scf.yield %v : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossb_mma, kWidth = 4}>>
      } {triton.warp_pipeline.stage = "a_load"}

      scf.yield %ld : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossb_mma, kWidth = 4}>>
    } {triton.warp_pipeline.pipelined_for}

    // Flat B: b_0 no LDS (masks the bug), b_1 writes the same %smem (dep).
    scf.execute_region no_inline {
      tt.store %gptr, %v1 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "b_nolds"}

    scf.execute_region no_inline {
      %data = tt.load %ptr : tensor<256x64x!tt.ptr<f16>, #crossb_blocked>
      ttg.local_store %data, %smem : tensor<256x64xf16, #crossb_blocked> -> !ttg.memdesc<256x64xf16, #crossb_shared, #crossb_smem, mutable>
      scf.yield
    } {triton.warp_pipeline.stage = "b_lds"}

    // Use %final after flat B so the loop's iter_arg result is observed and
    // the local_load execute_region survives DCE — without breaking the
    // back-to-back boundary between loop A and flat B.
    tt.store %dst, %final : tensor<256x16x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #crossb_mma, kWidth = 4}>>

    ttg.local_dealloc %smem : !ttg.memdesc<256x64xf16, #crossb_shared, #crossb_smem, mutable>
    tt.return
  }
}

// CHECK-LABEL: tt.func @cross_pipeline_dep_in_b1
// Pre-barrier and phase shift for loop A.
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// Loop body: a_0 (tt.store), internal s_barrier, a_1 (local_load).
// CHECK: tt.store
// CHECK: rocdl.s.barrier
// CHECK: ttg.local_load
// Boundary barriers between loop A and flat B are KEPT because (a_1, b_1)
// is a cross-pipeline WAR dep on %smem and no LOCAL barrier on the path
// a_1 → boundary → b_0 → b_1 covers it (A's wrap-around is not LOCAL).
// CHECK: amdg.cond_barrier
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
// Flat B stages: b_0 (tt.store), internal s_barrier, b_1 (local_store).
// CHECK: tt.store
// CHECK: ttg.local_store
// Reconverge cond_barrier for flat B.
// CHECK: amdg.cond_barrier
// CHECK: tt.return

// -----

// ---- Back-to-back: cross-pipeline dep where placement falls inside A ----
//
// Companion to @cross_pipeline_dep_in_b1.  Where that test puts the
// uncovered cross-pipeline pair at distance == 1 (so the placement falls at
// boundary slot K), this one engineers a pair at distance == K from `a_0`
// to `b_0` so the placement falls at slot K-1 — *inside* A's body.
// isCrossPipelineSafe must still flag this as unsafe: the explicit
// cross-pipeline-pair sweep walks (src, barrierLoc] for coverage and finds
// no LOCAL slot in A (loopBars[1..K-1] are all false).
//
// Layout:
//   Loop A (2 stages): a_0 ttg.local_load   (READS LDS)
//                      a_1 tt.store         (no LDS)
//   Flat B (2 stages): b_0 ttg.local_store  (WRITES the same LDS buffer)
//                      b_1 tt.store         (no LDS)
//
// A's circular analysis: a_0 read-read with itself, no intersection with a_1;
// loopBars = [false, false] and the wrap-around is non-LOCAL.
//
// Cross-pipeline dep (a_0, b_0) WAR on %smem at merged distance K=2 →
// barrierLoc = dst-1 = 1.  isCovered(0, 1) walks slot 1 (loopBars[1]=false)
// and returns false; the pair is intersected → unsafe.  Boundary barriers
// must be kept.

#crossa_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#crossa_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#crossa_shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#crossa_smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @cross_pipeline_dep_in_a0(
      %lb: i32, %ub: i32, %step: i32,
      %acc: tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossa_mma, kWidth = 4}>>,
      %ptr: tensor<256x64x!tt.ptr<f16>, #crossa_blocked>,
      %gptr: !tt.ptr<f32>,
      %dst: tensor<256x16x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #crossa_mma, kWidth = 4}>>) {

    %smem = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #crossa_shared, #crossa_smem, mutable>
    %v0 = arith.constant 0.0 : f32

    // Loop A: stage 0 reads %smem (threaded through iter_args so the
    // local_load survives DCE), stage 1 no LDS.
    %final = scf.for %i = %lb to %ub step %step
        iter_args(%cur = %acc)
        -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossa_mma, kWidth = 4}>> : i32 {
      %ld = scf.execute_region -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossa_mma, kWidth = 4}>> no_inline {
        %sub = ttg.memdesc_subslice %smem[0, 0] : !ttg.memdesc<256x64xf16, #crossa_shared, #crossa_smem, mutable> -> !ttg.memdesc<256x16xf16, #crossa_shared, #crossa_smem, mutable, 256x64>
        %v = ttg.local_load %sub : !ttg.memdesc<256x16xf16, #crossa_shared, #crossa_smem, mutable, 256x64> -> tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossa_mma, kWidth = 4}>>
        scf.yield %v : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossa_mma, kWidth = 4}>>
      } {triton.warp_pipeline.stage = "a_load"}

      scf.execute_region no_inline {
        tt.store %gptr, %v0 : !tt.ptr<f32>
        scf.yield
      } {triton.warp_pipeline.stage = "a_compute"}

      scf.yield %ld : tensor<256x16xf16, #ttg.dot_op<{opIdx = 0, parent = #crossa_mma, kWidth = 4}>>
    } {triton.warp_pipeline.pipelined_for}

    // Flat B: b_0 writes %smem (the dep), b_1 no LDS.
    scf.execute_region no_inline {
      %data = tt.load %ptr : tensor<256x64x!tt.ptr<f16>, #crossa_blocked>
      ttg.local_store %data, %smem : tensor<256x64xf16, #crossa_blocked> -> !ttg.memdesc<256x64xf16, #crossa_shared, #crossa_smem, mutable>
      scf.yield
    } {triton.warp_pipeline.stage = "b_lds"}

    scf.execute_region no_inline {
      tt.store %gptr, %v0 : !tt.ptr<f32>
      scf.yield
    } {triton.warp_pipeline.stage = "b_nolds"}

    tt.store %dst, %final : tensor<256x16x!tt.ptr<f16>, #ttg.dot_op<{opIdx = 0, parent = #crossa_mma, kWidth = 4}>>

    ttg.local_dealloc %smem : !ttg.memdesc<256x64xf16, #crossa_shared, #crossa_smem, mutable>
    tt.return
  }
}

// CHECK-LABEL: tt.func @cross_pipeline_dep_in_a0
// Pre-barrier and phase shift for loop A.
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
// CHECK: scf.for
// Loop body: a_0 (local_load), internal s_barrier, a_1 (tt.store).
// CHECK: ttg.local_load
// CHECK: rocdl.s.barrier
// CHECK: tt.store
// Boundary barriers between loop A and flat B must be KEPT.  The (a_0, b_0)
// WAR on %smem at merged distance K places at slot K-1 (inside A); the
// cross-pipeline-pair sweep finds no LOCAL slot in (0, K-1] (loopBars[1] is
// false because A's intra-cluster barrier is just s_barrier) and reports
// the pair as uncovered.
// CHECK: amdg.cond_barrier
// CHECK: ttg.barrier local
// CHECK: amdg.cond_barrier
// Flat B stages: b_0 (local_store), internal s_barrier, b_1 (tt.store).
// CHECK: ttg.local_store
// CHECK: tt.store
// Reconverge cond_barrier for flat B.
// CHECK: amdg.cond_barrier
// CHECK: tt.return

// -----

// ---- LDS effect nested inside scf.if must be detected ----
//
// Stage 0 wraps its ttg.local_store inside an scf.if, so the effect is not
// visible on the top-level op.  buildBlockInfoFromBlock must walk
// recursively to discover it; otherwise the cross-cluster RAW (stage0
// writes, stage1 reads) is missed and the cluster barriers degrade from
// ttg.barrier local to plain rocdl.s.barrier — leaving the LDS race
// uncovered.

#nest_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#nest_mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 16], isTransposed = true}>
#nest_dot = #ttg.dot_op<{opIdx = 0, parent = #nest_mma, kWidth = 4}>
#nest_shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#nest_smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func @nested_lds_effect_in_if(
      %lb: i32, %ub: i32, %step: i32,
      %cond: i1,
      %acc: tensor<256x16xf16, #nest_dot>,
      %ptr: tensor<256x64x!tt.ptr<f16>, #nest_blocked>) {

    %smem = ttg.local_alloc : () -> !ttg.memdesc<256x64xf16, #nest_shared, #nest_smem, mutable>

    %r:2 = scf.for %i = %lb to %ub step %step
        iter_args(%a = %acc, %s = %smem)
        -> (tensor<256x16xf16, #nest_dot>, !ttg.memdesc<256x64xf16, #nest_shared, #nest_smem, mutable>) : i32 {

      // Stage 0: conditionally writes LDS via scf.if.  The ttg.local_store
      // sits inside the if body, so a flat scan of the cluster body would
      // miss it.
      %st = scf.execute_region -> !ttg.memdesc<256x64xf16, #nest_shared, #nest_smem, mutable> no_inline {
        scf.if %cond {
          %data = tt.load %ptr : tensor<256x64x!tt.ptr<f16>, #nest_blocked>
          ttg.local_store %data, %s : tensor<256x64xf16, #nest_blocked> -> !ttg.memdesc<256x64xf16, #nest_shared, #nest_smem, mutable>
        }
        scf.yield %s : !ttg.memdesc<256x64xf16, #nest_shared, #nest_smem, mutable>
      } {triton.warp_pipeline.stage = "cond_store"}

      // Stage 1: reads LDS — RAW with the conditional write in stage 0.
      %ld = scf.execute_region -> tensor<256x16xf16, #nest_dot> no_inline {
        %sub = ttg.memdesc_subslice %s[0, 0] : !ttg.memdesc<256x64xf16, #nest_shared, #nest_smem, mutable> -> !ttg.memdesc<256x16xf16, #nest_shared, #nest_smem, mutable, 256x64>
        %v = ttg.local_load %sub : !ttg.memdesc<256x16xf16, #nest_shared, #nest_smem, mutable, 256x64> -> tensor<256x16xf16, #nest_dot>
        scf.yield %v : tensor<256x16xf16, #nest_dot>
      } {triton.warp_pipeline.stage = "lds_load"}

      scf.yield %ld, %s : tensor<256x16xf16, #nest_dot>, !ttg.memdesc<256x64xf16, #nest_shared, #nest_smem, mutable>
    } {triton.warp_pipeline.pipelined_for}

    ttg.local_dealloc %smem : !ttg.memdesc<256x64xf16, #nest_shared, #nest_smem, mutable>
    tt.return
  }
}

// CHECK-LABEL: tt.func @nested_lds_effect_in_if
// CHECK: scf.for
// Stage 0 with the nested scf.if + local_store.
// CHECK: scf.if
// CHECK:   ttg.local_store
// Cluster barrier between stage 0 and stage 1 is LOCAL (nested write seen).
// CHECK: rocdl.sched.barrier
// CHECK-NEXT: ttg.barrier local
// CHECK-NEXT: rocdl.sched.barrier
// Stage 1 reads LDS.
// CHECK: ttg.local_load
// Wrap-around barrier is also LOCAL (stage1 read vs stage0 write next iter).
// CHECK: rocdl.sched.barrier
// CHECK-NEXT: ttg.barrier local
// CHECK-NEXT: rocdl.sched.barrier
// CHECK: scf.yield
