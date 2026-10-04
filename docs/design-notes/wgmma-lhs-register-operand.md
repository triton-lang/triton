# Requirements to pass WGMMA LHS operand in registers

Status: draft / under discussion (tracking issue #4785)

## Background

On Hopper and later, `wgmma.mma_async` accepts its A (LHS) operand either from
shared memory (via a shared-memory descriptor) or from registers. Today Triton
always materializes the A operand in shared memory before the WGMMA, even when
the operand is produced by an element-wise prologue (e.g. a scale, transpose-free
layout change, or activation applied to the A tile). This forces a round trip
through shared memory that a register-resident A operand would avoid.

NVIDIA/OpenXLA have a prototype (openxla/triton#17) implementing this
optimization, and the Triton team is planning a similar feature. This note
records the requirements to align on so the eventual PRs converge.

## Requirements to align on

1. **Operand layout.** A register-resident A operand must already match the
   WGMMA A-fragment register layout expected by the MMA layout of the
   accumulator. The compiler must either verify that the producer layout is
   compatible or insert a conversion (which may itself require shared memory,
   negating the benefit). A layout-compatibility check at the
   `DotOperandLayout` level is the natural place for this.
2. **Prologue fusion scope.** Only element-wise prologues whose result can be
   kept in registers should be eligible. Prologues requiring cross-lane data
   movement (e.g. transposes) are out of scope initially.
3. **Register pressure.** The A tile held in registers competes with
   accumulators and pipeline buffers for the register file. The feature must
   not regress kernels that currently fit; heuristics or a knob may be needed
   to fall back to the shared-memory path.
4. **Pipeliner interaction.** The software pipeliner currently stages the A
   operand through shared memory with async copies. A register path needs a
   distinct staging strategy (or an explicit opt-out) so pipelining remains
   correct.
5. **API surface.** Decide whether register-resident A is selected implicitly
   by the compiler when legal, or exposed explicitly (e.g. via a Gluon-level
   construct or a `tt.dot` flag). Implicit selection keeps the default path
   unchanged; explicit selection aids testing and benchmarking.
6. **Testing.** Numerical parity with the shared-memory path across dtypes
   (fp16, bf16, tf32, fp8) and tile shapes, plus benchmarks showing the
   expected win on prologue-heavy matmuls.

## Open questions

- Minimum viable shape support (which M/N/K tile sizes are supported with A in
  registers on Hopper vs. Blackwell).
- Whether the feature should also cover the tcgen05 (Blackwell) MMA path or
  stay WGMMA-only initially.
- How the choice interacts with warp specialization and TMA-based operand
  loading.
