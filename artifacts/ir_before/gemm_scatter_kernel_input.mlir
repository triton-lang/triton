// Hand-crafted TTGIR for gemm_scatter_kernel:
//   - Two tt.load ops (A tile, B tile) inside scf.for
//   - tt.dot accumulation (loop-carried)
//   - tt.store OUTSIDE the loop (scatter to peer/output buffer)
//
// This represents the IR after TritonToTritonGPU + TritonGPUScheduleLoops
// but before TritonGPUPipeline.
//
// GPU target: cuda:80 (Ampere, sm_80) — uses cp.async for pipelining
// Block shape: 128x128 output, 64 reduction dimension
// 4 warps, num_stages=3
//
// Run pipelining pass:
//   triton-opt this_file.mlir -tritongpu-pipeline -canonicalize
//
// The key observation: tt.store (scatter) after the scf.for has no loop.stage
// attribute and is NOT touched by the pipeline pass.

#blocked_a = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked_b = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#blocked_c = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#dot_a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#shared_a = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#shared_b = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {
  "ttg.num-ctas" = 1 : i32,
  "ttg.num-warps" = 4 : i32,
  ttg.target = "cuda:80",
  "ttg.threads-per-warp" = 32 : i32
} {

// gemm_scatter_kernel:
//   a_ptr: !tt.ptr<f16>   — input A (M, K_local)
//   b_ptr: !tt.ptr<f16>   — input B (K_local, N)
//   out_ptr: !tt.ptr<f16> — output / peer symm mem buffer
//   K: i32                — K_local (reduction dimension)
tt.func public @gemm_scatter_kernel(
    %a_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %b_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %out_ptr: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %K: i32 {tt.divisibility = 16 : i32},
    %stride_am: i32 {tt.divisibility = 16 : i32},
    %stride_bk: i32 {tt.divisibility = 16 : i32}
) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst_zero_f32 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>

  // Program IDs (tile indices)
  %pid_m = tt.get_program_id x : i32
  %pid_n = tt.get_program_id y : i32

  // Row/column offsets
  %offs_m_base = arith.muli %pid_m, %c64_i32 : i32  // pid_m * BLOCK_M (128)
  %offs_n_base = arith.muli %pid_n, %c64_i32 : i32

  // Build initial A pointer tensor (shape 128x64)
  %offs_m_range = tt.make_range {end = 128 : i32, start = 0 : i32}
      : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked_a}>>
  %offs_k_range = tt.make_range {end = 64 : i32, start = 0 : i32}
      : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
  %offs_m_splat = tt.splat %offs_m_base
      : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked_a}>>
  %offs_m = arith.addi %offs_m_splat, %offs_m_range
      : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked_a}>>
  %a_ptr_splat = tt.splat %a_ptr
      : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked_a>
  %offs_m_2d = tt.expand_dims %offs_m {axis = 1 : i32}
      : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked_a}>>
        -> tensor<128x1xi32, #blocked_a>
  %offs_k_2d = tt.expand_dims %offs_k_range {axis = 0 : i32}
      : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
        -> tensor<1x64xi32, #blocked_a>
  %offs_m_bc = tt.broadcast %offs_m_2d : tensor<128x1xi32, #blocked_a> -> tensor<128x64xi32, #blocked_a>
  %offs_k_bc = tt.broadcast %offs_k_2d : tensor<1x64xi32, #blocked_a> -> tensor<128x64xi32, #blocked_a>
  %stride_am_tensor = tt.splat %stride_am : i32 -> tensor<128x64xi32, #blocked_a>
  %a_row_offs = arith.muli %offs_m_bc, %stride_am_tensor : tensor<128x64xi32, #blocked_a>
  %a_off = arith.addi %a_row_offs, %offs_k_bc : tensor<128x64xi32, #blocked_a>
  %a_ptrs_init = tt.addptr %a_ptr_splat, %a_off : tensor<128x64x!tt.ptr<f16>, #blocked_a>, tensor<128x64xi32, #blocked_a>

  // Build initial B pointer tensor (shape 64x128)
  %offs_n_range = tt.make_range {end = 128 : i32, start = 0 : i32}
      : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked_b}>>
  %offs_n_splat = tt.splat %offs_n_base
      : i32 -> tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked_b}>>
  %offs_n = arith.addi %offs_n_splat, %offs_n_range
      : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked_b}>>
  %b_ptr_splat = tt.splat %b_ptr
      : !tt.ptr<f16> -> tensor<64x128x!tt.ptr<f16>, #blocked_b>
  %offs_k2_2d = tt.expand_dims %offs_k_range {axis = 1 : i32}
      : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked_a}>>
        -> tensor<64x1xi32, #blocked_b>
  %offs_n_2d = tt.expand_dims %offs_n {axis = 0 : i32}
      : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked_b}>>
        -> tensor<1x128xi32, #blocked_b>
  %offs_k2_bc = tt.broadcast %offs_k2_2d : tensor<64x1xi32, #blocked_b> -> tensor<64x128xi32, #blocked_b>
  %offs_n_bc = tt.broadcast %offs_n_2d : tensor<1x128xi32, #blocked_b> -> tensor<64x128xi32, #blocked_b>
  %stride_bk_tensor = tt.splat %stride_bk : i32 -> tensor<64x128xi32, #blocked_b>
  %b_row_offs = arith.muli %offs_k2_bc, %stride_bk_tensor : tensor<64x128xi32, #blocked_b>
  %b_off = arith.addi %b_row_offs, %offs_n_bc : tensor<64x128xi32, #blocked_b>
  %b_ptrs_init = tt.addptr %b_ptr_splat, %b_off : tensor<64x128x!tt.ptr<f16>, #blocked_b>, tensor<64x128xi32, #blocked_b>

  // Loop bound: K_local / BLOCK_K
  %K_div_BK = arith.divsi %K, %c64_i32 : i32

  // =========================================================================
  // GEMM inner loop — this is what the software pipeline pass targets.
  //
  // {tt.num_stages = 3}: requests 3-stage pipelining (2 loads ahead of dot).
  //
  // After TritonGPUScheduleLoops, the loads have:
  //   {loop.stage = 0, loop.cluster = 0, tt.latency = 2}
  // The dot has:
  //   {loop.stage = 2, loop.cluster = 0}
  //
  // After TritonGPUPipeline:
  //   - tt.load → ttg.async_copy_global_to_local (writes to shared mem)
  //   - ttg.async_commit_group inserted after each async copy
  //   - ttg.async_wait {num = 2} inserted before consuming each buffer
  //   - ttg.local_alloc allocates multi-buffered shared memory (3 buffers)
  //   - ttg.memdesc_index selects which buffer to write/read
  //   - Prologue unrolls first 2 iterations before the loop
  // =========================================================================
  %loop_result:3 = scf.for %k = %c0_i32 to %K_div_BK step %c1_i32
      iter_args(%a_ptrs = %a_ptrs_init, %b_ptrs = %b_ptrs_init, %acc = %cst_zero_f32)
      -> (tensor<128x64x!tt.ptr<f16>, #blocked_a>,
          tensor<64x128x!tt.ptr<f16>, #blocked_b>,
          tensor<128x128xf32, #mma>)
      {tt.num_stages = 3 : i32} {

    // Load A tile: scheduled in stage 0 (earliest), latency 2
    %a_tile = tt.load %a_ptrs {loop.stage = 0 : i32, loop.cluster = 0 : i32, tt.latency = 2 : i32}
        : tensor<128x64x!tt.ptr<f16>, #blocked_a>

    // Load B tile: scheduled in stage 0 (earliest), latency 2
    %b_tile = tt.load %b_ptrs {loop.stage = 0 : i32, loop.cluster = 0 : i32, tt.latency = 2 : i32}
        : tensor<64x128x!tt.ptr<f16>, #blocked_b>

    // Convert to dot operand layouts (MMA-compatible)
    %a_dot = ttg.convert_layout %a_tile
        {loop.stage = 2 : i32, loop.cluster = 0 : i32}
        : tensor<128x64xf16, #blocked_a> -> tensor<128x64xf16, #dot_a>
    %b_dot = ttg.convert_layout %b_tile
        {loop.stage = 2 : i32, loop.cluster = 0 : i32}
        : tensor<64x128xf16, #blocked_b> -> tensor<64x128xf16, #dot_b>

    // Matrix multiply-accumulate: scheduled in stage 2 (last stage)
    %c = tt.dot %a_dot, %b_dot, %acc
        {loop.stage = 2 : i32, loop.cluster = 0 : i32}
        : tensor<128x64xf16, #dot_a> * tensor<64x128xf16, #dot_b> -> tensor<128x128xf32, #mma>

    // Advance pointers by BLOCK_K
    %a_stride = arith.constant dense<64> : tensor<128x64xi32, #blocked_a>
    %b_stride = arith.constant dense<64> : tensor<64x128xi32, #blocked_b>
    %a_ptrs_next = tt.addptr %a_ptrs, %a_stride
        {loop.stage = 2 : i32, loop.cluster = 0 : i32}
        : tensor<128x64x!tt.ptr<f16>, #blocked_a>, tensor<128x64xi32, #blocked_a>
    %b_ptrs_next = tt.addptr %b_ptrs, %b_stride
        {loop.stage = 2 : i32, loop.cluster = 0 : i32}
        : tensor<64x128x!tt.ptr<f16>, #blocked_b>, tensor<64x128xi32, #blocked_b>

    scf.yield %a_ptrs_next, %b_ptrs_next, %c
        : tensor<128x64x!tt.ptr<f16>, #blocked_a>,
          tensor<64x128x!tt.ptr<f16>, #blocked_b>,
          tensor<128x128xf32, #mma>
  }

  // =========================================================================
  // SCATTER STORE — outside the inner loop, NOT pipelined.
  //
  // 'loop_result#2' is the fully-accumulated GEMM result (acc_final).
  // It is only valid after ALL K iterations of the scf.for complete.
  //
  // Structural reasons the pipeliner cannot move this:
  //   1. It is outside the scf.for — the pipeliner only transforms ops
  //      that are inside the loop it is pipelining.
  //   2. It depends on %loop_result#2 which is defined by the scf.for —
  //      it is not possible to compute the final value before the loop ends.
  //   3. There is no 'tt.AsyncStoreOp' or async scatter in Triton's IR.
  //      AssignLatencies.cpp never assigns tt.latency to tt.StoreOp.
  //
  // In multi-GPU deployment: out_ptr would be the destination rank's
  // symmetric memory buffer (peer GPU memory via NVLink/UVA).
  // =========================================================================

  // Convert f32 accumulator to f16 for output
  %result_f16 = tt.fp_to_fp %loop_result#2
      : tensor<128x128xf32, #mma> -> tensor<128x128xf16, #mma>

  // Convert layout back to blocked for store
  %result_blocked = ttg.convert_layout %result_f16
      : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #blocked_c>

  // Compute output pointer tensor
  %out_ptr_splat = tt.splat %out_ptr
      : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked_c>

  // Synchronous scatter store to output buffer.
  // This op has NO {loop.stage} or {tt.latency} attribute.
  // The TritonGPUPipeline pass will leave this UNCHANGED.
  tt.store %out_ptr_splat, %result_blocked
      : tensor<128x128x!tt.ptr<f16>, #blocked_c>

  tt.return
}

// =========================================================================
// reduce_kernel:
//   partial_buf: !tt.ptr<f16> — local symm mem buffer (WORLD_SIZE, M_SHARD, N)
//   out: !tt.ptr<f16>         — final output buffer (M_SHARD, N)
//
// Uses static_range(WORLD_SIZE=2) — unrolled at compile time.
// The pipeliner finds no scf.for to pipeline here.
// =========================================================================
tt.func public @reduce_kernel(
    %partial_buf: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %out: !tt.ptr<f16> {tt.divisibility = 16 : i32}
) {
  %cst_zero = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked_c>
  %c0 = arith.constant 0 : i32
  %stride_n = arith.constant 1024 : i32  // N = 1024

  // Unrolled iteration 0: load rank-0's partial (offset = 0)
  %partial_buf_splat0 = tt.splat %partial_buf
      : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked_c>
  %p0 = tt.load %partial_buf_splat0 : tensor<128x128x!tt.ptr<f16>, #blocked_c>
  %acc0 = arith.addf %cst_zero, %p0 : tensor<128x128xf32, #blocked_c>  // widening via cast elided for clarity

  // Unrolled iteration 1: load rank-1's partial (offset = M_SHARD * N elements)
  %shard_stride = arith.constant dense<131072> : tensor<128x128xi32, #blocked_c>  // 512 * 1024 = 524288 / 4 fp16 = 131072 elements
  %partial_buf_1 = tt.addptr %partial_buf_splat0, %shard_stride
      : tensor<128x128x!tt.ptr<f16>, #blocked_c>, tensor<128x128xi32, #blocked_c>
  %p1 = tt.load %partial_buf_1 : tensor<128x128x!tt.ptr<f16>, #blocked_c>
  %acc1 = arith.addf %acc0, %p1 : tensor<128x128xf32, #blocked_c>

  // Store reduced result
  %out_splat = tt.splat %out
      : !tt.ptr<f16> -> tensor<128x128x!tt.ptr<f16>, #blocked_c>
  tt.store %out_splat, %acc1 : tensor<128x128x!tt.ptr<f16>, #blocked_c>

  tt.return
}

}
