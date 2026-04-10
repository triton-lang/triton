// GEMM + Scatter Store: input for -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline
//
// Based on the matmul_loop pattern from test/TritonGPU/loop-pipeline.mlir.
// Added: a tt.store AFTER the scf.for to model the scatter write to a peer
// symmetric memory buffer.
//
// Run:
//   triton-opt this_file.mlir -split-input-file \
//     -tritongpu-assign-latencies \
//     -tritongpu-schedule-loops \
//     -tritongpu-pipeline=num-stages=3 \
//     -canonicalize \
//     -o after_pipeline.mlir
//
// Expected outcome:
//   - The tt.load ops inside scf.for → ttg.async_copy_global_to_local
//   - ttg.async_wait inserted before tt.dot consumers
//   - The tt.store AFTER the loop remains UNCHANGED (synchronous scatter)

// 4 warps, matmul: 128x32 @ 32x128 -> 128x128
#AL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#BL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#CL = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#ALs0 = #ttg.slice<{parent=#AL, dim=0}>
#BLs0 = #ttg.slice<{parent=#BL, dim=0}>
#C = #ttg.nvidia_mma<{versionMajor = 2, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#A = #ttg.dot_op<{opIdx = 0, parent = #C, kWidth=2}>
#B = #ttg.dot_op<{opIdx = 1, parent = #C, kWidth=2}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {

// gemm_scatter_kernel:
//   Computes C = A @ B in a pipelined inner loop,
//   then stores the result to `out_ptr` (the peer/scatter destination).
//
// Key observation:
//   - The scf.for (GEMM inner loop) is pipelined by the pass sequence.
//   - The tt.store (scatter) outside the loop is NOT touched.
tt.func @gemm_scatter_kernel(
    %lb : index,
    %ub : index,
    %step : index,
    %A : !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %B : !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %out_ptr : !tt.ptr<f32> {tt.divisibility = 16 : i32}
) -> tensor<128x128xf32, #C> {
  // A ptrs
  %a_ptr_splat = tt.splat %A : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>, #AL>
  %a_tmp0 = tt.make_range {end = 32: i32, start = 0: i32} : tensor<32xi32, #ALs0>
  %a_tmp1 = tt.expand_dims %a_tmp0 {axis = 0 : i32} : tensor<32xi32, #ALs0> -> tensor<1x32xi32, #AL>
  %a_offs = tt.broadcast %a_tmp1 : tensor<1x32xi32, #AL> -> tensor<128x32xi32, #AL>
  %a_ptr_init = tt.addptr %a_ptr_splat, %a_offs : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>

  // B ptrs
  %b_ptr_splat = tt.splat %B : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>, #BL>
  %b_tmp0 = tt.make_range {end = 128: i32, start = 0: i32} : tensor<128xi32, #BLs0>
  %b_tmp1 = tt.expand_dims %b_tmp0 {axis = 0 : i32} : tensor<128xi32, #BLs0> -> tensor<1x128xi32, #BL>
  %b_offs = tt.broadcast %b_tmp1 : tensor<1x128xi32, #BL> -> tensor<32x128xi32, #BL>
  %b_ptr_init = tt.addptr %b_ptr_splat, %b_offs : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>

  %a_mask = arith.constant dense<true> : tensor<128x32xi1, #AL>
  %a_other = arith.constant dense<0.00e+00> : tensor<128x32xf16, #AL>
  %b_mask = arith.constant dense<true> : tensor<32x128xi1, #BL>
  %b_other = arith.constant dense<0.00e+00> : tensor<32x128xf16, #BL>
  %c_init = arith.constant dense<0.00e+00> : tensor<128x128xf32, #C>

  %a_off = arith.constant dense<4> : tensor<128x32xi32, #AL>
  %b_off = arith.constant dense<4> : tensor<32x128xi32, #BL>
  %b_scale = arith.constant dense<4.> : tensor<32x128xf16, #B>

  // =========================================================================
  // GEMM inner loop: pipelined by -tritongpu-assign-latencies +
  //                  -tritongpu-schedule-loops + -tritongpu-pipeline
  //
  // After pipelining:
  //   tt.load %a_ptr → ttg.async_copy_global_to_local (issues cp.async)
  //   tt.load %b_ptr → ttg.async_copy_global_to_local
  //   ttg.async_commit_group inserted after each async copy group
  //   ttg.async_wait {num = 2} inserted before the tt.dot consumer
  //   ttg.local_alloc creates 3-buffered shared memory for A and B
  //   Prologue: first 2 iterations unrolled before the main loop
  // =========================================================================
  %loop:3 = scf.for %iv = %lb to %ub step %step
      iter_args(%a_ptr = %a_ptr_init, %b_ptr = %b_ptr_init, %prev_c = %c_init)
      -> (tensor<128x32x!tt.ptr<f16>, #AL>,
          tensor<32x128x!tt.ptr<f16>, #BL>,
          tensor<128x128xf32, #C>) {

    // Load A tile from global memory
    // After pipelining: becomes ttg.async_copy_global_to_local
    %a_ = tt.load %a_ptr, %a_mask, %a_other : tensor<128x32x!tt.ptr<f16>, #AL>
    %a = ttg.convert_layout %a_ : tensor<128x32xf16, #AL> -> tensor<128x32xf16, #A>

    // Load B tile from global memory
    // After pipelining: becomes ttg.async_copy_global_to_local
    %b__ = tt.load %b_ptr, %b_mask, %b_other : tensor<32x128x!tt.ptr<f16>, #BL>
    %b_ = ttg.convert_layout %b__ : tensor<32x128xf16, #BL> -> tensor<32x128xf16, #B>
    %b = arith.mulf %b_, %b_scale: tensor<32x128xf16, #B>

    // Matrix multiply-accumulate (tensor cores)
    // After pipelining: reads from shared memory buffers via ttg.local_load,
    //                   but this op itself is unchanged
    %c = tt.dot %a, %b, %prev_c : tensor<128x32xf16, #A> * tensor<32x128xf16, #B> -> tensor<128x128xf32, #C>

    %next_a_ptr = tt.addptr %a_ptr, %a_off : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<128x32xi32, #AL>
    %next_b_ptr = tt.addptr %b_ptr, %b_off : tensor<32x128x!tt.ptr<f16>, #BL>, tensor<32x128xi32, #BL>
    scf.yield %next_a_ptr, %next_b_ptr, %c
        : tensor<128x32x!tt.ptr<f16>, #AL>, tensor<32x128x!tt.ptr<f16>, #BL>, tensor<128x128xf32, #C>
  }

  // =========================================================================
  // SCATTER STORE — models writing GEMM result to peer symmetric memory buffer.
  //
  // This tt.store is OUTSIDE the scf.for. The pipeline pass DOES NOT TOUCH IT.
  //
  // Structural reasons:
  //   1. %loop#2 is the final accumulated value — only valid after all scf.for
  //      iterations complete. There is no partially-correct value to scatter.
  //   2. tt.store has no {loop.stage} attribute — AssignLatencies only annotates
  //      tt.LoadOp, tt.DescriptorLoadOp, and tt.DotOp/MMAv5 ops.
  //   3. There is no async scatter op in Triton's IR — no "ttg.AsyncScatterOp"
  //      or peer-write primitive that the pipeliner could schedule.
  //   4. The pipeliner operates on the scf.for body; ops outside the loop are
  //      outside the pipeliner's scope by definition.
  //
  // In multi-GPU deployment: out_ptr is the destination rank's UVA-mapped
  // symmetric memory buffer (peer GPU memory via NVLink). The kernel writes
  // to this buffer without any special IR mechanism — it is simply a pointer
  // that happens to be on another GPU. The pipeliner has no semantic knowledge
  // of this communication nature.
  // =========================================================================
  // Convert from MMA layout to blocked layout for store
  %result_blocked = ttg.convert_layout %loop#2 : tensor<128x128xf32, #C> -> tensor<128x128xf32, #CL>

  %out_ptr_splat = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>, #CL>

  // Synchronous store — NOT pipelined, NOT async, NOT touched by any pass.
  // This is the "scatter" step in GEMM + Reduce-Scatter.
  tt.store %out_ptr_splat, %result_blocked : tensor<128x128x!tt.ptr<f32>, #CL>

  tt.return %loop#2: tensor<128x128xf32, #C>
}

}
