# [AMD] Layout-aware widening for `amdgpu.buffer_load` (VGPR path)

## Background

PR #amd-coalesce-lds added `CoalesceBufferLoadToLocalWrites`, mirroring the
existing `CoalesceAsyncCopyWrites` for `amdgpu.buffer_load_to_local` (the
direct-to-LDS path). That pass uses `canLoadDirectToLDS`, which in turn calls
`getNumConsecutiveInOut` on the `regLayout -> sharedLayout` map. That check is
*layout-aware* — it can prove per-thread memory contiguity even when AxisInfo
cannot.

The VGPR sibling `amdgpu.buffer_load` does **not** have an equivalent.
`BufferLoadOpConversion` in `LoadStoreOpToLLVM.cpp` only consults
`AxisInfo` via `getVectorSize(ptr, offset, axisAnalysisPass)`, which is
purely tensor-axis-based.

## Symptom

In the MXFP4 unroll4 GEMM kernel
(`kernels/cdna4/gemm/mxfp4_gemm_gfx950.py`), B-scale loads emit 32 ×
`buffer_load_ubyte` per kernel even though every thread's 4 i8 registers
land on 4 memory-adjacent bytes after the `e8m0_shuffle_opsel_b` pre-shuffle.

Empirically validated: forcing `vec = 4` for i8 `BufferLoadOp` with
`numElems == 4` collapses the loads to 8 × `buffer_load_dword` (4× fewer
instructions, same `dwordx4` traffic everywhere else) and **numerical
results still PASS** on both `M=128 N=128 K=768` and `M=128 N=256 K=1024`.

Layout comparison (computed in this session):

```
A scale (gload_layout, custom):
  register = [[0, 4], [16, 0]]
  lane     = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]]
  warp     = [[32, 0], [64, 0]]

B scale (from get_mfma_scale_layout, opIdx=1, mfmaMDim=16):
  register = [[0, 4], [64, 0]]
  lane     = [[1, 0], [2, 0], [4, 0], [8, 0], [0, 1], [0, 2]]
  warp     = [[16, 0], [32, 0]]
```

The layouts are structurally identical: every thread's 4 registers map to
memory offsets `[X, X+1, X+2, X+3]` after the pre-shuffle. A succeeds because
its load goes through LDS (`buffer_load_to_shared` → layout-aware coalesce
pass). B fails because it loads directly to VGPR and the VGPR lowering only
sees AxisInfo, which reports stride-256 along the K axis (consumed by the
mod-based pre-shuffle math: `b_k_lane = (k%4)*256 + (k%8//4)*1`).

## Root cause

AxisInfo's per-axis contiguity model cannot express stride functions that
depend on mod/div over the tensor coords, because those produce non-affine
per-element memory deltas. The actual per-register memory delta requires
either evaluating the offset SSA at specific tensor coords or proving a
mod/div pattern symbolically.

## Proposed fix

Build a `ConstantTensorValueAnalysis` (or extend `AxisInfo` with a
"per-coord evaluator" side channel) that:

1. Walks back through `arith.{addi,subi,muli,divsi,divui,remsi,remui,shli,shri}`,
   `tt.make_range`, `tt.splat`, `tt.broadcast`, `tt.expand_dims`,
   `arith.constant`, `tt.reshape` (etc.).
2. Provides `evaluate(value, tensorCoord) -> std::optional<int64_t>` that
   returns the value of `value` at a specific tensor coordinate when the
   producer chain is fully constant-foldable.
3. For a `BufferLoadOp` whose offsets value comes from such a chain,
   sample `evaluate` at the per-thread register-coord deltas (using the
   offsets value's linear layout to map register-axis bases to tensor
   coords). The min of the per-register memory deltas gives the safe
   per-thread vector width.

Then add a `CoalesceBufferLoadWrites` pass alongside the existing pattern
that stamps `op.setContiguity(N)`. The lowering already widens via
`vec = std::max(vec, op.getContiguity())`.

## Scope

- `third_party/amd/include/Analysis/ConstantTensorValueAnalysis.h` (new)
- `third_party/amd/lib/Analysis/ConstantTensorValueAnalysis.cpp` (new)
- `third_party/amd/lib/TritonAMDGPUTransforms/CoalesceAsyncCopy.cpp`
  — add a third pattern `CoalesceBufferLoadWrites`.
- `test/TritonGPU/amd/amd-coalesce-buffer-load.mlir` (new)
- An end-to-end check on `mxfp4_gemm_gfx950.py` confirming the B-scale
  loads collapse to `buffer_load_dword` and numerics still pass.

## How to reproduce / validate locally

All paths below are on the gfx950 dev box used in this session.

### 1. Environment

```bash
source /home/djavady/venvs/therock-rocm713-gfx950/bin/activate
export TRITON_REPO=/home/sanketp/work/triton-main
export KERNELS_REPO=/home/sanketp/work/gluon-kernels
export PYTHONPATH=$TRITON_REPO/python:$KERNELS_REPO
```

### 2. Build Triton

```bash
cd $TRITON_REPO
make            # builds libtriton.so + triton-opt under build/cmake.linux-*
```

Build dir:

```bash
BUILD_DIR=$(cd $TRITON_REPO && PYTHONPATH=./python python3 -c \
  'from build_helpers import get_cmake_dir; print(get_cmake_dir())')
# e.g. /home/sanketp/work/triton-main/build/cmake.linux-x86_64-cpython-3.10
```

Pass-only iteration (no Python rebuild needed):

```bash
cd $BUILD_DIR && ninja triton-opt
```

### 3. Run the kernel

```bash
rm -rf ~/.triton/cache/*        # force a fresh compile
cd $KERNELS_REPO
TRITON_ALWAYS_COMPILE=1 \
  python3 kernels/cdna4/gemm/test_mxfp4_gemm_gfx950.py
```

Expected output:

```
[PASS] M=  128 N=  128 K=  768 max_abs=1.6000e+01 ref_max=7.1812e+03 rel=2.2280e-03
[PASS] M=  128 N=  256 K= 1024 max_abs=2.7250e+01 ref_max=8.6672e+03 rel=3.1440e-03
```

### 4. Find the compiled artifacts

Triton caches every compile under `~/.triton/cache/<hash>/`:

```bash
ls -lt ~/.triton/cache/*/mxfp4_gemm_kernel.amdgcn | head
```

Each cache dir contains:

| File | Stage |
|---|---|
| `mxfp4_gemm_kernel.ttgir` | TTGIR after `make_ttgir` / `gluon_to_ttgir` |
| `mxfp4_gemm_kernel.llir`  | LLVM-IR (MLIR form) after `make_llir` |
| `mxfp4_gemm_kernel.amdgcn`| AMDGCN assembly |
| `mxfp4_gemm_kernel.hsaco` | Final HSACO binary |
| `mxfp4_gemm_kernel.json`  | Metadata (num_warps, shared, etc.) |

Useful aliases:

```bash
AMDGCN=$(ls -t ~/.triton/cache/*/mxfp4_gemm_kernel.amdgcn | head -1)
TTGIR=$(ls -t  ~/.triton/cache/*/mxfp4_gemm_kernel.ttgir  | head -1)
```

### 5. Inspect the loads in AMDGCN

```bash
# All buffer_load widths (incl. direct-to-LDS)
grep -oE '^\s*buffer_load_[a-z0-9]+' $AMDGCN | sort | uniq -c | sort -rn

# Direct-to-LDS only (the `lds` suffix)
grep -E 'buffer_load.* offen lds' $AMDGCN | awk '{print $1}' | sort | uniq -c

# VGPR loads only (no `lds`)
grep -E '^\s*buffer_load' $AMDGCN | grep -v 'offen lds' | \
  awk '{print $1}' | sort | uniq -c
```

Current baseline (after `CoalesceBufferLoadToLocalWrites` only):

```
=== Direct-to-LDS by width ===
      8 buffer_load_dword       # A scales (i8, widened by current pass)
     32 buffer_load_dwordx4     # A data + others

=== VGPR loads by width ===
     32 buffer_load_ubyte       # B scales — this follow-up targets these
     ...
```

Target (after this follow-up):

```
=== VGPR loads by width ===
      8 buffer_load_dword       # B scales widened 4x
     ...
```

### 6. Inspect the layouts in TTGIR

```bash
grep -E '^#(blocked|linear|shared|padded)' $TTGIR
grep -nE 'buffer_load_to_local|amdg\.buffer_load[^_]' $TTGIR | head
```

The B-scale load currently looks like:

```
%b_scale = amdg.buffer_load %b_scales_ptr[%b_scale_offsets_108]
           : tensor<128x8xi8, #linear>
```

Note the **absence** of `{contiguity = N}` (the A direct-to-LDS op carries
`{contiguity = 4}` stamped by `CoalesceBufferLoadToLocalWrites`).

### 7. AITER reference assembly

```bash
less /home/sanketp/work/aiter_kernels/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x128.s
```

Search for `v_mfma_scale_f32_16x16x128_f8f6f4` and the surrounding
`buffer_load_dword`. AITER loads one scale-dword per thread then uses
`op_sel` / `op_sel_hi` across 4 MFMAs. This is the target codegen shape.

### 8. Lit tests

```bash
cd $BUILD_DIR && ninja triton-opt
lit -v test/TritonGPU/amd/amd-coalesce-async-copy.mlir
# When this follow-up adds tests:
lit -v test/TritonGPU/amd/amd-coalesce-buffer-load.mlir
```

Lit runs on CPU — no GPU required.

### 9. The empirical probe used in this session

Apply to `BufferLoadOpConversion` in
`third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp`, just after
`vec = std::max(vec, op.getContiguity());`:

```cpp
if (valueElemTy.getIntOrFloatBitWidth() == 8 && numElems == 4 && vec == 1)
  vec = 4;
```

Rebuild (`make`), clear cache, rerun step 3. Numerics still pass; step 5
shows `buffer_load_dword` replacing the 32 × `buffer_load_ubyte`. Revert
before committing.

## Validation already done

- One-line lowering probe (`if (i8 && numElems == 4 && vec == 1) vec = 4;`)
  applied to `BufferLoadOpConversion`:
  - Compiled.
  - `[PASS] M=128 N=128 K=768` and `M=128 N=256 K=1024`.
  - AMDGCN: 32 × `buffer_load_ubyte` → 8 × `buffer_load_dword`.
- Probe reverted after validation; not in tree.

## Out of scope for this follow-up

- A generic kernel-author primitive like `tl.assume_contiguity(offsets, n)`
  — could be added independently but is orthogonal.
- Extending the same analysis to non-AMD backends or to `tt.load` /
  `tt.store` directly. Start narrow.

## References

- Existing layout-aware pattern (the template to mirror):
  `third_party/amd/lib/TritonAMDGPUTransforms/CoalesceAsyncCopy.cpp`
  (`CoalesceAsyncCopyWrites` and `CoalesceBufferLoadToLocalWrites`).
- Lowering site where `op.getContiguity()` is consumed:
  `third_party/amd/lib/TritonAMDGPUToLLVM/LoadStoreOpToLLVM.cpp` line ~654.
- IR attribute already present:
  `third_party/amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td`
  (`BufferLoadOp`'s `$contiguity` field).
- AITER reference assembly:
  `/home/sanketp/work/aiter_kernels/f4gemm_bf16_per1x32Fp4_BpreShuffle_128x128.s`
  — shows the target pattern (one `buffer_load_dword` per thread feeding
  4 `v_mfma_scale_*` ops via `op_sel`).
