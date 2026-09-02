# Triton Gaudi2 native backend

This backend maps Triton's logical block programming model to Gaudi2 TPC
index-space execution. It does not emulate CUDA threads, warps, shared memory,
or barriers.

The initial vertical path is:

```text
@triton.jit -> TTIR -> Gaudi TPC plan -> TPC-C -> tpc-clang -> ELF
             -> GaudiKernelArtifactV1 -> Synapse recipe -> current HPU stream
```

The generated kernel is exposed to Graph Compiler through one fixed GUID,
`triton_gaudi2_v1`. The GUID's perf-library reads a content-addressed ELF from
a private runtime cache; file paths are never embedded in the artifact ABI.
The matching PyTorch Bridge patch registers artifacts and exposes the fixed
GUID as a graph-integrated custom op. Production execution therefore stays in
the current HPU graph and is compatible with `torch.compile(hpu_backend)`;
the direct recipe launcher remains available for ABI diagnostics.

## Target and options

Only Gaudi2 with SynapseAI 1.24.1 is supported. Gaudi scheduling uses
`GaudiConfig` through `backend_options`; CUDA-only launch knobs are not used to
construct a fake SIMT execution model.

```python
schedule = GaudiConfig(engine="tpc", pipeline_depth=1, unroll=1)
kernel[grid](..., backend_options=schedule.as_backend_options())
```

`GaudiKernelArtifactV1` contains canonical metadata, the TPC ELF, and SHA-256
digests. `TRITON_GAUDI_ARTIFACT_DIR` selects the private materialization cache;
`TRITON_GAUDI_PERF_LIB` can override the packaged perf-library for development.
The driver accepts a Bridge advertising launch ABI v1, or launch ABI v2 when
that Bridge explicitly retains the v1 registration/launch compatibility
surface. ArtifactV1 emission remains explicit and fail-closed; this is not an
implicit conversion to the v2 envelope.
Synapse reads `GC_KERNEL_PATH` during graph-compiler initialization, so a
standalone Triton process must prepare the packaged perf library before its
first HPU allocation:

```python
from triton.backends.gaudi import prepare_environment

prepare_environment()
```

The vLLM plugin performs this preparation during operator registration.

## Implemented strict subset

The first production-shaped slice supports contiguous masked FP32 and BF16
elementwise add/subtract/multiply with `program_id(0) * BLOCK + arange`. It
also recognizes the exact Triton TTIR DAG for BF16 residual-add plus RMSNorm,
including the rounded residual output, FP32 reduction, runtime epsilon, and
static hidden sizes up to 8192. Strict matchers also cover BF16 SiLU-and-mul
and the shape-specialized Qwen3.5 packed decode GDN with an in-place FP32
recurrent state. Generated kernels use the full 2048-bit TPC vector, native
reduction intrinsics, VLM row residency, and partial tensor loads/stores for
tails. Unsupported or near-matching TTIR fails closed with a Gaudi lowering
diagnostic. The state-mutating GDN path additionally requires the matching
Bridge reinplace pass so AOTAutograd cannot materialize and copy the complete
state cache.

TPC-C is compiled at `-O2`. `-O3` is intentionally rejected on this Gaudi2
slice because it has not met the hardware-safety gate for the generated
stateful/reduction kernels.

Scan, gather/scatter, generic reductions, MME partitioning, complete
attention, MoE, quantization, DMA/HCCL scheduling, standalone lazy-mode
HPUGraph capture, and graph-level epilogues remain subsequent backend work;
they must not silently take CUDA semantics. vLLM exposes `off`, `hybrid`, and
`strict` rollout modes so this subset can be measured without claiming
unsupported coverage. Canonical TP1 GDN decode clears the hybrid compiled
subgraph gate at batch 8 and above; smaller batches stay on the vendor graph.
SiLU-and-mul remains a strict-mode performance candidate.
