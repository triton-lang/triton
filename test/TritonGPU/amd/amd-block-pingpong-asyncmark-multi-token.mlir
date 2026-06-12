// RUN: triton-opt %s --tritonamdgpu-block-pingpong=num-stages=3 | FileCheck %s
// RUN: triton-opt %s --tritonamdgpu-block-pingpong=num-stages=3 --tritonamdgpu-update-async-wait-count=gfx-arch=gfx950 | FileCheck %s

// BlockPingpong's transformTwoClusterWithLocalLoadAndAll combines per-operand
// ttg.async_waits into a single multi-token wait, then reorders async copies
// and commits around it. After the reorder the pass re-runs updateWaits, so
// the merged wait's `num` is derived via minNumInterleavedCommitOps against
// the post-reorder IR (with multi-token support added in the same patch).
//
// On asyncmark targets (CDNA3/CDNA4) this `num` lowers straight to
// rocdl.wait.asyncmark(N), and UpdateAsyncWaitCount is a no-op since PR #9883
// - so whatever num BlockPingpong writes is what reaches the hardware. The
// second RUN line confirms UpdateAsyncWaitCount leaves the wait untouched.

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 4, warpsPerCTA = [2, 4], instrShape = [16, 16, 32], isTransposed = true}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [0, 1]}>
#shared1 = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 16, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx950", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: async_ns3_gemm_pingpong_multi_token
  // The two per-operand waits in the input carry num=2 and num=1; pingpong
  // fuses them into a single multi-token wait whose num is recomputed against
  // the post-reorder IR. In this synthetic test the tokens come from tt.func
  // args (no prologue commit chain), so minNumInterleavedCommitOps bails to
  // its conservative N=0; in a real kernel the chains terminate at prologue
  // async_commit_groups and yield a tighter bound (e.g. 1 for a real
  // gfx950 simple_persistent_matmul kernel).
  // CHECK: scf.for
  // CHECK: ttg.async_wait %{{[^,]+}}, %{{[^,]+}} {num = 0 : i32}
  tt.func public @async_ns3_gemm_pingpong_multi_token(
      %arg0: i32,
      %arg1: tensor<256x32x!tt.ptr<bf16>, #blocked>,
      %arg2: tensor<32x256x!tt.ptr<bf16>, #blocked1>,
      %arg3: !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>,
      %arg4: !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>,
      %arg5: !ttg.async.token,
      %arg6: !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>,
      %arg7: !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>,
      %arg8: !ttg.async.token,
      %arg9: !ttg.async.token,
      %arg10: !ttg.async.token,
      %arg11: tensor<256x32xi32, #blocked>,
      %arg12: tensor<32x256xi32, #blocked1>,
      %arg13: !ttg.memdesc<3x256x32xbf16, #shared, #smem, mutable>,
      %arg14: !ttg.memdesc<3x32x256xbf16, #shared1, #smem, mutable>) {
    %c3_i32 = arith.constant 3 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<256x256xf32, #mma>
    %0:12 = scf.for %arg15 = %c0_i32 to %arg0 step %c1_i32 iter_args(%arg16 = %cst, %arg17 = %arg1, %arg18 = %arg2, %arg19 = %c1_i32, %arg20 = %arg3, %arg21 = %arg4, %arg22 = %arg5, %arg23 = %arg6, %arg24 = %arg7, %arg25 = %arg8, %arg26 = %arg9, %arg27 = %arg10) -> (tensor<256x256xf32, #mma>, tensor<256x32x!tt.ptr<bf16>, #blocked>, tensor<32x256x!tt.ptr<bf16>, #blocked1>, i32, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token)  : i32 {
      %1 = tt.addptr %arg17, %arg11 : tensor<256x32x!tt.ptr<bf16>, #blocked>, tensor<256x32xi32, #blocked>
      %2 = tt.addptr %arg18, %arg12 : tensor<32x256x!tt.ptr<bf16>, #blocked1>, tensor<32x256xi32, #blocked1>
      %3 = arith.addi %arg19, %c1_i32 : i32
      %4 = arith.cmpi slt, %3, %c3_i32 : i32
      %5 = arith.select %4, %3, %c0_i32 : i32
      %6 = ttg.memdesc_index %arg13[%5] : !ttg.memdesc<3x256x32xbf16, #shared, #smem, mutable> -> !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>
      %7 = ttg.async_copy_global_to_local %1, %6 : tensor<256x32x!tt.ptr<bf16>, #blocked> -> <256x32xbf16, #shared, #smem, mutable>
      %8 = ttg.async_commit_group tokens %7
      %9 = ttg.local_load %arg20 token %arg22 : !ttg.memdesc<256x32xbf16, #shared, #smem, mutable> -> tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>>
      %10 = ttg.memdesc_index %arg14[%5] : !ttg.memdesc<3x32x256xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>
      %11 = ttg.async_copy_global_to_local %2, %10 : tensor<32x256x!tt.ptr<bf16>, #blocked1> -> <32x256xbf16, #shared1, #smem, mutable>
      %12 = ttg.async_commit_group tokens %11
      %13 = ttg.local_load %arg23 token %arg25 : !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable> -> tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>>
      %14 = tt.dot %9, %13, %arg16 : tensor<256x32xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 8}>> * tensor<32x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 8}>> -> tensor<256x256xf32, #mma>
      %15 = ttg.async_wait %arg26 {num = 2 : i32}
      %16 = ttg.async_wait %arg27 {num = 1 : i32}
      scf.yield %14, %1, %2, %5, %arg21, %6, %15, %arg24, %10, %16, %8, %12 : tensor<256x256xf32, #mma>, tensor<256x32x!tt.ptr<bf16>, #blocked>, tensor<32x256x!tt.ptr<bf16>, #blocked1>, i32, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.memdesc<256x32xbf16, #shared, #smem, mutable>, !ttg.async.token, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.memdesc<32x256xbf16, #shared1, #smem, mutable>, !ttg.async.token, !ttg.async.token, !ttg.async.token
    }
    tt.return
  }
}
