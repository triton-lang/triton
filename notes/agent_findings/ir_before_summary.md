# IR Before Pipeline Pass: Summary

## Confirmed Facts (from Triton source and test files)

### Expected TTGIR Structure Before `TritonGPUPipeline`

This section describes the expected IR for `gemm_scatter_kernel` based on:
1. Knowledge of how Triton compiles `tl.range(num_stages=N)` loops
2. Reference from existing tests (e.g., `test/TritonGPU/loop-pipeline-cuda.mlir`)
3. The `TritonGPUScheduleLoops` pass output structure

### Compilation Stages and IR Names

| Stage | IR name | Extension | Content |
|-------|---------|-----------|---------|
| 1 | TTIR | `.ttir` | Python AST → raw Triton IR (before GPU-specific lowering) |
| 2 | TTGIR | `.ttgir` | After `TritonToTritonGPU` + GPU dialect passes |
| 3 | Pre-pipeline TTGIR | (in memory) | After `assign-latencies` + `schedule-loops`, before `pipeline` |
| 4 | Post-pipeline TTGIR | `.ttgir` (final) | After `TritonGPUPipeline` |
| 5 | LLIR | `.llir` | After lowering to LLVM IR |
| 6 | PTX/AMDGCN | `.ptx` | Final assembly |

### Expected IR: `gemm_scatter_kernel` Before Pipeline Pass

```mlir
// -----// IR Dump Before TritonGPUPipeline (gemm_scatter_kernel)
#blocked = #ttg.blocked<{sizePerThread = [1, 8], ...}>
module attributes {"ttg.num-warps" = 4 : i32, ...} {
  tt.func public @gemm_scatter_kernel(
      %a_ptr: !tt.ptr<f16>,
      %b_ptr: !tt.ptr<f16>,
      %out_ptr: !tt.ptr<f16>,
      %scatter_offset: i32,
      ...) {
    
    // Initialize accumulator
    %acc_init = tt.splat %cst_f32 : f32 -> tensor<128x128xf32, #blocked>
    
    // Compute loop bounds and initial pointers
    // (tl.arange, tl.broadcast, pointer arithmetic ops)
    
    // ─── Inner k-loop ────────────────────────────────────────────
    // After TritonGPUScheduleLoops:
    //   tt.load has {loop.stage = 0, loop.cluster = 0, tt.latency = 2}
    //   tt.dot has  {loop.stage = 2, loop.cluster = 0}
    // This means loads are scheduled 2 stages ahead of the dot.
    %result:3 = scf.for %k = %c0 to %K_div_BK step %c1
        iter_args(%a_ptrs_arg = %a_ptrs_init,
                  %b_ptrs_arg = %b_ptrs_init,
                  %acc_arg = %acc_init)
        -> (tensor<128x64x..>, tensor<64x128x..>, tensor<128x128xf32, #...>)
        {tt.num_stages = 3 : i32} {
      
      // Load A tile from global memory
      %a_tile = tt.load %a_ptrs_arg {loop.stage = 0, tt.latency = 2} :
          tensor<128x64x!tt.ptr<f16>, #blocked> -> tensor<128x64xf16, #blocked>
      
      // Load B tile from global memory
      %b_tile = tt.load %b_ptrs_arg {loop.stage = 0, tt.latency = 2} :
          tensor<64x128x!tt.ptr<f16>, #blocked> -> tensor<64x128xf16, #blocked>
      
      // Convert layouts for dot operands
      %a_dot = ttg.convert_layout %a_tile : ... -> tensor<128x64xf16, #dot_op_a>
      %b_dot = ttg.convert_layout %b_tile : ... -> tensor<64x128xf16, #dot_op_b>
      
      // Matrix multiply-accumulate
      %c = tt.dot %a_dot, %b_dot, %acc_arg {loop.stage = 2} :
          tensor<128x64xf16, #dot_op_a> * tensor<64x128xf16, #dot_op_b>
          -> tensor<128x128xf32, #...>
      
      // Advance pointers for next iteration
      %a_ptrs_next = tt.addptr %a_ptrs_arg, %a_stride
      %b_ptrs_next = tt.addptr %b_ptrs_arg, %b_stride
      
      scf.yield %a_ptrs_next, %b_ptrs_next, %c
    }
    
    // ─── Post-loop: scatter store ─────────────────────────────────
    // This is OUTSIDE the scf.for. It has NO loop.stage or tt.latency attribute.
    // The pipeliner will not touch this operation.
    
    // Convert accumulator to fp16
    %result_fp16 = tt.fp_to_fp %result#2 {rounding = ...} :
        tensor<128x128xf32, ...> -> tensor<128x128xf16, ...>
    
    // Compute output pointer addresses (includes scatter_offset)
    %out_ptrs = tt.addptr %out_ptr, %scatter_offset_tensor
    
    // Synchronous scatter store to (potentially peer) output buffer
    // - No {tt.latency} attribute
    // - No {loop.stage} attribute
    // - Not inside any pipeline loop
    tt.store %out_ptrs, %result_fp16 {cache = 1 : i32, evict = 1 : i32} :
        tensor<128x128x!tt.ptr<f16>, #blocked>
    
    tt.return
  }
}
```

### Key Structural Observations

1. **The `tt.load` ops are inside `scf.for`** and receive `{loop.stage = 0, tt.latency = 2}` from `TritonGPUScheduleLoops`. Stage 0 means "schedule 2 stages before the consumer (stage 2 dot)."

2. **The `tt.dot` is at `loop.stage = 2`** — the last stage. This is the consumer of the loads.

3. **The `tt.store` is outside the loop** — no stage attribute, not a candidate for pipelining. It appears as a simple sequential op after the `scf.for`.

4. **The `scf.for` has `{tt.num_stages = 3}`** — this is the compiler hint from `tl.range(num_stages=3)`.

### Expected IR: `reduce_kernel` Before Pipeline Pass

The `reduce_kernel` uses `tl.static_range(WORLD_SIZE)` which compiles to multiple unrolled `tt.load` + accumulate ops — no `scf.for` at all. The pipeliner finds no loop to pipeline.

```mlir
tt.func @reduce_kernel(%partial_buf: !tt.ptr<f16>, %out: !tt.ptr<f16>, ...) {
  // Unrolled iteration 0: load rank-0 partial
  %p0 = tt.load %partial_buf + %offset_0 : ...
  %acc0 = arith.addf %zeros, %p0 : ...
  
  // Unrolled iteration 1: load rank-1 partial
  %p1 = tt.load %partial_buf + %offset_1 : ...
  %acc1 = arith.addf %acc0, %p1 : ...
  
  // (world_size=1: only one iteration, effectively no-op reduction)
  
  tt.store %out, %acc_final : ...
  tt.return
}
```

## Observed Behavior

(To be updated after actual IR dump from hardware run)

## Hypotheses

- The layout `#blocked` in TTGIR will be converted to shared memory layout (`#ttg.shared`) for the load results after pipelining
- The `tt.load` ops will become `ttg.async_copy_global_to_local` targeting a multi-buffered shared memory allocation of size `NUM_STAGES × BLOCK_M × BLOCK_K × sizeof(f16)`

## Open Questions

- What exact layout attributes (`#blocked`, `#dot_op_a`) does the `TritonToTritonGPU` pass assign for this problem shape and GPU target?
- Does the `other=0.0` argument in `tl.load(..., other=0.0)` prevent async conversion? (Per `AssignLatencies.cpp`: loads with non-zero "other" must load to registers, not shared memory.)

## Next Actions

- Run `python gemm_reduce_scatter_triton.py --mode warmup` with `MLIR_ENABLE_DUMP=1` on hardware
- Read `artifacts/ir_before/before_pipeline.mlir` and update this document with actual IR
