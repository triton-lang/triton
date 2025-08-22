// RUN: triton-opt %s -split-input-file --tritonamdgpu-block-pingpong="num-stages=4" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

  // CHECK-LABEL: chained_dots_async_loads

  // CHECK: scf.for
  // CHECK: rocdl.s.setprio 0
  // Compute Cluster1
  // CHECK: tt.dot
  // CHECK: rocdl.s.setprio 1
  // CHECK: ttg.async_wait
  // CHECK: rocdl.sched.barrier 0
  // MemoryCluster2
  // CHECK: ttg.local_load
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.s.barrier
  // CHECK: rocdl.s.setprio 0
  // Compute Cluster2
  // CHECK: tt.dot
  // CHECK: rocdl.s.setprio 1
  // CHECK: ttg.async_wait
  // CHECK: rocdl.sched.barrier 0
  // Memory Cluster2
  // CHECK: ttg.local_load
  // CHECK: ttg.async_copy_global_to_local
  // CHECK: ttg.async_commit_group
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.s.barrier
  // CHECK-NEXT: scf.yield

  tt.func @chained_dots_async_loads(%arg0: tensor<64x16x!tt.ptr<f16>, #blocked>, %arg1: i32, %arg2: i32, %arg3: !ttg.async.token, %arg4: tensor<128x16xf32, #mma>, %arg5: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %arg6: i32, %arg7: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %arg8: tensor<128x16xf32, #mma>, %arg9: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg10: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg11: i32, %arg12: i32, %arg13: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>) -> tensor<128x16xf32, #mma> {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    %2 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
    %3 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
    %4 = ttg.memdesc_index %1[%c1_i32] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
    %5:9 = scf.for %arg14 = %c0_i32 to %arg1 step %arg2 iter_args(%arg15 = %arg4, %arg16 = %arg4, %arg17 = %arg7, %arg18 = %arg3, %arg19 = %arg3, %arg20 = %2, %arg21 = %4, %arg22 = %arg3, %arg23 = %3) -> (tensor<128x16xf32, #mma>, tensor<128x16xf32, #mma>, tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, !ttg.async.token, !ttg.async.token, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.async.token, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>)  : i32 {
      %6 = tt.dot %arg10, %arg17, %arg15 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %7 = ttg.async_wait %arg18 {num = 0 : i32}
      %8 = ttg.local_load %arg20 token %7 : !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %9 = ttg.memdesc_index %0[%arg6] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
      %10 = ttg.async_copy_global_to_local %arg0, %9 : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable, 2x64x16>
      %11 = ttg.async_commit_group tokens %10
      %12 = tt.dot %arg10, %8, %arg16 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %13 = ttg.async_wait %arg22 {num = 0 : i32}
      %14 = ttg.local_load %arg23 token %13 : !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %15 = ttg.memdesc_index %1[%arg6] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
      %16 = ttg.async_copy_global_to_local %arg0, %15 : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable, 2x64x16>
      %17 = ttg.async_commit_group tokens %16
      scf.yield %12, %6, %14, %arg19, %17, %arg21, %15, %11, %9 : tensor<128x16xf32, #mma>, tensor<128x16xf32, #mma>, tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, !ttg.async.token, !ttg.async.token, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.async.token, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
    }
    ttg.local_dealloc %1 : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    tt.return %5#0 : tensor<128x16xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

  // CHECK-LABEL: chained_dots_tt_loads

  // CHECK-NOT: rocdl.s
  // CHECK: scf.for
  // CHECK: rocdl.s.setprio 0
  // Compute Cluster1
  // CHECK: tt.dot
  // CHECK: rocdl.s.setprio 1
  // CHECK: gpu.barrier
  // CHECK: rocdl.sched.barrier 0
  // MemoryCluster2
  // CHECK: ttg.local_store
  // CHECK: ttg.local_load
  // CHECK: tt.load
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.s.barrier
  // CHECK: rocdl.s.setprio 0
  // Compute Cluster2
  // CHECK: tt.dot
  // CHECK: rocdl.s.setprio 1
  // CHECK: gpu.barrier
  // CHECK: rocdl.sched.barrier 0
  // Memory Cluster2
  // CHECK: ttg.local_store
  // CHECK: ttg.local_load
  // CHECK: tt.load
  // CHECK: rocdl.sched.barrier 0
  // CHECK: rocdl.s.barrier
  // CHECK-NEXT: scf.yield

  tt.func @chained_dots_tt_loads(%arg0: tensor<64x16xf16, #blocked>, %arg1: tensor<64x16x!tt.ptr<f16>, #blocked>, %arg2: i32, %arg3: i32, %arg4: tensor<128x16xf32, #mma>, %arg5: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %arg6: i32, %arg7: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %arg8: tensor<128x16xf32, #mma>, %arg9: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg10: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg11: i32, %arg12: i32, %arg13: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>) -> tensor<128x16xf32, #mma> {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    %2 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
    %3 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
    %4 = ttg.memdesc_index %1[%c1_i32] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
    %5:8 = scf.for %arg14 = %c0_i32 to %arg2 step %arg3 iter_args(%arg15 = %arg4, %arg16 = %arg4, %arg17 = %arg7, %arg18 = %2, %arg19 = %4, %arg20 = %3, %arg21 = %arg0, %arg22 = %arg0) -> (tensor<128x16xf32, #mma>, tensor<128x16xf32, #mma>, tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, tensor<64x16xf16, #blocked>, tensor<64x16xf16, #blocked>)  : i32 {
      %6 = tt.dot %arg10, %arg17, %arg15 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      ttg.local_store %arg21, %arg18 : tensor<64x16xf16, #blocked> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
      %7 = ttg.local_load %arg18 : !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %8 = ttg.memdesc_index %0[%arg6] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
      %9 = tt.load %arg1 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %10 = tt.dot %arg10, %7, %arg16 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      ttg.local_store %arg22, %arg20 : tensor<64x16xf16, #blocked> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
      %11 = ttg.local_load %arg20 : !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %12 = ttg.memdesc_index %1[%arg6] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
      %13 = tt.load %arg1 : tensor<64x16x!tt.ptr<f16>, #blocked>
      scf.yield %10, %6, %11, %arg19, %12, %8, %9, %13 : tensor<128x16xf32, #mma>, tensor<128x16xf32, #mma>, tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, tensor<64x16xf16, #blocked>, tensor<64x16xf16, #blocked>
    }
    ttg.local_dealloc %1 : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    tt.return %5#0 : tensor<128x16xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

  // CHECK-LABEL: reject_chained_dots_empty_mem_cluster_1

  // CHECK-NOT: setprio
  // CHECK-NOT: barrier

  tt.func @reject_chained_dots_empty_mem_cluster_1(%arg0: tensor<64x16xf16, #blocked>, %arg1: tensor<64x16x!tt.ptr<f16>, #blocked>, %arg2: i32, %arg3: i32, %arg4: tensor<128x16xf32, #mma>, %arg5: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %arg6: i32, %arg7: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %arg8: tensor<128x16xf32, #mma>, %arg9: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg10: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg11: i32, %arg12: i32, %arg13: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>) -> tensor<128x16xf32, #mma> {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    %2 = ttg.memdesc_index %1[%c0_i32] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
    %3 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
    %4 = ttg.memdesc_index %1[%c1_i32] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
    %5:8 = scf.for %arg14 = %c0_i32 to %arg2 step %arg3 iter_args(%arg15 = %arg4, %arg16 = %arg4, %arg17 = %arg7, %arg18 = %2, %arg19 = %4, %arg20 = %3, %arg21 = %arg0, %arg22 = %arg0) -> (tensor<128x16xf32, #mma>, tensor<128x16xf32, #mma>, tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, tensor<64x16xf16, #blocked>, tensor<64x16xf16, #blocked>)  : i32 {
      %6 = tt.dot %arg10, %arg17, %arg15 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      %10 = tt.dot %arg10, %arg17, %arg16 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      ttg.local_store %arg22, %arg20 : tensor<64x16xf16, #blocked> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
      %11 = ttg.local_load %arg20 : !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %12 = ttg.memdesc_index %1[%arg6] : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
      %13 = tt.load %arg1 : tensor<64x16x!tt.ptr<f16>, #blocked>
      scf.yield %10, %6, %11, %arg19, %12, %12, %13, %13 : tensor<128x16xf32, #mma>, tensor<128x16xf32, #mma>, tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, tensor<64x16xf16, #blocked>, tensor<64x16xf16, #blocked>
    }
    ttg.local_dealloc %1 : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    ttg.local_dealloc %0 : !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>
    tt.return %5#0 : tensor<128x16xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [4, 1], instrShape = [32, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {

  // CHECK-LABEL: reject_chained_dots_empty_mem_cluster_2

  // CHECK-NOT: setprio
  // CHECK-NOT: barrier

  tt.func @reject_chained_dots_empty_mem_cluster_2(%memdesc1: !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, %memdesc2: !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, %alloc1: !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>, %alloc2: !ttg.memdesc<2x64x16xf16, #shared, #smem, mutable>, %arg0: tensor<64x16xf16, #blocked>, %arg1: tensor<64x16x!tt.ptr<f16>, #blocked>, %arg2: i32, %arg3: i32, %arg4: tensor<128x16xf32, #mma>, %arg5: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>, %arg6: i32, %arg7: tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, %arg8: tensor<128x16xf32, #mma>, %arg9: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg10: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg11: i32, %arg12: i32, %arg13: tensor<128xf32, #ttg.slice<{dim = 1, parent = #mma}>>) -> tensor<128x16xf32, #mma> {
    %5:8 = scf.for %arg14 = %arg3 to %arg2 step %arg3 iter_args(%arg15 = %arg4, %arg16 = %arg4, %arg17 = %arg7, %arg18 = %memdesc1, %arg19 = %memdesc1, %arg20 = %memdesc2, %arg21 = %arg0, %arg22 = %arg0) -> (tensor<128x16xf32, #mma>, tensor<128x16xf32, #mma>, tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, tensor<64x16xf16, #blocked>, tensor<64x16xf16, #blocked>)  : i32 {
      %6 = tt.dot %arg10, %arg17, %arg15 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      ttg.local_store %arg22, %arg20 : tensor<64x16xf16, #blocked> -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>
      %11 = ttg.local_load %arg20 : !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
      %13 = tt.load %arg1 : tensor<64x16x!tt.ptr<f16>, #blocked>
      %10 = tt.dot %arg10, %arg17, %arg16 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>> -> tensor<128x16xf32, #mma>
      scf.yield %10, %6, %11, %arg19, %arg20, %arg20, %13, %13 : tensor<128x16xf32, #mma>, tensor<128x16xf32, #mma>, tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, !ttg.memdesc<64x16xf16, #shared, #smem, mutable, 2x64x16>, tensor<64x16xf16, #blocked>, tensor<64x16xf16, #blocked>
    }
    tt.return %5#0 : tensor<128x16xf32, #mma>
  }
}
