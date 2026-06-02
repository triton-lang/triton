#ifndef TRITON_ANALYSIS_REGISTERCONTIGUITY_H
#define TRITON_ANALYSIS_REGISTERCONTIGUITY_H

#include "mlir/IR/Value.h"

namespace mlir::triton {

class ModuleAxisInfoAnalysis;

// PROTOTYPE: backend-agnostic, layout-aware per-thread memory contiguity.
//
// Coalescing asks: along the sequence of addresses one thread visits (its
// `register` dimension, per the tensor's LinearLayout), how many *consecutive*
// elements are contiguous in memory? AxisInfo answers a weaker, per-axis
// question and cannot see contiguity that emerges from two tensor axes jointly
// (e.g. the MXFP4 B-scale pre-shuffle: col+4 -> byte+1, row+64 -> byte+2).
//
// This analysis answers the real question directly. It abstractly interprets
// the integer offsets tensor over an "affine-with-symbols" domain:
//
//     value  =  cst  +  sum_i coeff_i * symbol_i
//
// where each `symbol_i` is an opaque unknown scalar SSA value (kernel arg, loop
// induction variable, ...) carried *symbolically* with an integer coefficient.
// Transfer functions:
//   - add/sub/mul-by-constant: exact affine arithmetic.
//   - mul(symbolic, symbolic): non-affine -> top.
//   - rem/div/shr/and by a constant c: sound modular folding. A symbol term
//     drops out of `% c` when `c` divides coeff_i * divisibility(symbol_i),
//     where the symbol's divisibility comes from AxisInfo -- this is the
//     concrete "marry AxisInfo's value lattice with LinearLayout's register
//     order" step: AxisInfo supplies sound per-scalar alignment facts
//     (tt.divisibility, program-id alignment, ...), the linear layout supplies
//     the register visiting order. Otherwise the subterm is carried opaquely.
//
// Non-affine but register-INVARIANT subterms (e.g. block_id * runtime_stride)
// are kept as opaque atoms keyed by structural identity, so they cancel in
// offset(r) - offset(0) instead of poisoning the result.
//
// Contiguity is then read off the register->offset map with a *structural*
// scalar-independence proof: the per-register stride is trusted only when the
// symbol coefficients CANCEL in offset(r) - offset(0). A residual symbol (the
// stride depends on an unknown scalar) forces a bail. This replaces the finite
// multi-substitution probe of the AMD-only ConstantTensorValueAnalysis with an
// actual proof, needs no AxisInfo, and is shared across backends.
//
// Returns the largest power-of-two N such that registers [0..N) of a single
// thread access N consecutive offsets. Returns 1 on any failure (non-tensor,
// missing register dim, non-affine offset, scalar-dependent stride). Never
// returns 0.
unsigned getPerThreadContiguityAlongRegisters(Value offsetsValue,
                                              ModuleAxisInfoAnalysis &axisInfo);

} // namespace mlir::triton

#endif // TRITON_ANALYSIS_REGISTERCONTIGUITY_H
