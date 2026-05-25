# [AMD] Layout-aware widening for `amdgpu.buffer_load` (VGPR path) — Plan

## Context

`amdgpu.buffer_load_to_local` (direct-to-LDS) already has a layout-aware
contiguity pass (`CoalesceBufferLoadToLocalWrites` in
`third_party/amd/lib/TritonAMDGPUTransforms/CoalesceAsyncCopy.cpp`) that
stamps `op.setContiguity(N)`. The VGPR sibling `amdgpu.buffer_load` has the
same `$contiguity` IR attribute
(`third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td:733`)
and its lowering already consumes it
(`LoadStoreOpToLLVM.cpp:654` — `vec = std::max(vec, op.getContiguity())`),
but nothing in the pipeline ever sets it. The result is that `BufferLoadOp`
vectorization is determined solely by `AxisInfo` via `getVectorSize` in the
lowering — there is no transforms-pass step that bumps contiguity using
extra information available pre-lowering.

This follow-up adds that missing pass step so the VGPR path can be widened
the same way the direct-to-LDS path already is. See
`docs/amd-followup-buffer-load-coalesce.md` for the full motivating
write-up (MXFP4 B-scale loads on gfx950).

Per discussion, this plan deliberately lands the **layout-only** scaffold
first and explicitly defers the constant-evaluator described in the
write-up to a separate follow-up. The MXFP4 B-scale case relies on
mod/div arithmetic that neither AxisInfo nor a pure layout check can see;
the layout-only pass will be a no-op there, and that's acceptable —
landing the scaffold unblocks the constant-eval work and immediately
helps any case where AxisInfo already reports good contiguity but the
attribute simply wasn't being stamped.

## Approach

Mirror `CoalesceBufferLoadToLocalWrites`:

1. In the existing `TritonAMDGPUCoalesceAsyncCopyPass`, precompute
   per-`BufferLoadOp` contiguity from `ModuleAxisInfoAnalysis` (same
   helpers: `axisAnalysis.getContiguity(offsets, elemNumBits)` and
   `applyMaskAlignment`).
2. Add a third `OpRewritePattern<ttag::BufferLoadOp>` —
   `CoalesceBufferLoadWrites` — that:
   - Looks up the precomputed contiguity.
   - Clamps it by what the VGPR load can actually emit (`vec * elemBits`
     ≤ 128 for `buffer_load_dwordx4`; align to power of two).
   - Bails if `<= op.getContiguity()` (idempotent).
   - Stamps `op.setContiguity(newContig)`.
3. Add a lit test exercising the new stamping behavior with a
   synthetic `amdgpu.buffer_load` whose offsets have known AxisInfo
   contiguity.
4. Document the constant-eval extension as a TODO inline (the
   write-up already describes it).

Since `CoalesceBufferLoadWrites` only mutates the `$contiguity` attribute
(no layout conversion, no SSA rewires), it is safe to add to the existing
greedy pattern set without ordering concerns.

## Files to modify

- `third_party/amd/lib/TritonAMDGPUTransforms/CoalesceAsyncCopy.cpp`
  — add `CoalesceBufferLoadWrites` pattern + precompute map in
  `runOnOperation`.
- `test/TritonGPU/amd/amd-coalesce-async-copy.mlir` (existing file)
  or new `test/TritonGPU/amd/amd-coalesce-buffer-load.mlir` — add a
  test case for the VGPR pattern. Prefer extending the existing file
  to keep one fixture per pass.

No header changes. No new analysis files. No pass-pipeline change
(`tritonamdgpu-coalesce-async-copy` already runs at the right place).

## Reuse

All the building blocks already exist:

- `AMD::ModuleAxisInfoAnalysis` —
  `third_party/amd/include/Analysis/AxisInfoExt.h`.
- `axisAnalysis.getContiguity(value, elemNumBits)` and
  `axisAnalysis.getMaskAlignment(mask)` — already used by both
  existing patterns.
- `mlir::LLVM::AMD::getPointerTypeWithShape(ptr, offsets)` — same
  helper `CoalesceBufferLoadToLocalWrites` uses.
- `triton::getPointeeBitWidth(ptrTy)` — same helper.
- `op.setContiguity(N)` / `op.getContiguity()` — already on
  `BufferLoadOp` via the TD `$contiguity` field.
- VGPR-side legality (max 128-bit transaction) is enforced naturally
  by the lowering's existing clamp; the pattern just needs to clamp
  to `128 / elemBitWidth` so it doesn't request an unsupported width.

## Steps

- [ ] In `CoalesceAsyncCopy.cpp`, add `CoalesceBufferLoadWrites`
      `OpRewritePattern<ttag::BufferLoadOp>` mirroring
      `CoalesceBufferLoadToLocalWrites` but without the
      `canLoadDirectToLDS` / dst-encoding checks (VGPR has no dst
      shared layout). Clamp `loadContig` to `128 / elemBitWidth`
      and to the next-lower power of two.
- [ ] In `TritonAMDGPUCoalesceAsyncCopyPass::runOnOperation`, add a
      `DenseMap<ttag::BufferLoadOp, unsigned> bufferLoadVgprContiguity`
      walk that mirrors the existing `bufferLoadContiguity` walk and
      registers the new pattern.
- [ ] Add an inline `// TODO:` comment in the new pattern pointing
      at `docs/amd-followup-buffer-load-coalesce.md` for the
      constant-evaluator extension needed to catch the MXFP4 B-scale
      case (mod/div offsets).
- [ ] Add a lit test case in
      `test/TritonGPU/amd/amd-coalesce-async-copy.mlir` that builds a
      synthetic `amdgpu.buffer_load` with offsets known-contiguous to
      AxisInfo and CHECKs that `contiguity = N` appears on the op
      after the pass.
- [ ] Sanity-check the existing two patterns' lit cases still pass
      unchanged (the new pattern shouldn't fire on direct-to-LDS ops
      because it's typed on `BufferLoadOp`, not `BufferLoadToLocalOp`).

## Verification

1. Build: `cd $TRITON_REPO && make` (or `ninja triton-opt` in the build
   dir for fast iteration).
2. Lit: `lit -v test/TritonGPU/amd/amd-coalesce-async-copy.mlir` — must
   pass, including the new VGPR case.
3. Existing kernel regression: run
   `kernels/cdna4/gemm/test_mxfp4_gemm_gfx950.py` per
   `docs/amd-followup-buffer-load-coalesce.md` §3. The B-scale loads
   are **not** expected to widen with this layout-only change (the
   write-up's empirical probe relied on the constant-eval pattern); the
   goal here is just to confirm no numerical regressions and no AMDGCN
   width regressions vs. baseline.
4. Inspect AMDGCN per the write-up §5 to confirm any VGPR `buffer_load`
   that previously had AxisInfo-derived contiguity now carries it via
   the attribute path as well (no width loss).
5. Follow-up (separate PR): implement `ConstantTensorValueAnalysis`
   per the write-up to actually widen the MXFP4 B-scale loads.
