#ifndef TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H
#define TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H

#include "mlir/IR/Value.h"

namespace mlir::triton {
class ModuleAxisInfoAnalysis;
}

namespace mlir::triton::AMD {

// Per-thread memory contiguity of an integer offsets tensor, measured along the
// LinearLayout "register" order -- the quantity AxisInfo's per-axis model
// cannot express for cross-axis mod/div offsets (e.g. the MXFP4 B-scale
// pre-shuffle).
//
// This is a SYMBOLIC PROOF, not a sampler: there is no substitution of unknown
// scalars with trial values anywhere. Each offset is abstractly interpreted
// over an exact "affine + opaque" domain
//     V = cst + sum a_i*u_i + sum b_k*tau_k
// where u_i are unknown scalars (program-ids, loop IVs, runtime sizes) carried
// symbolically and tau_k are hash-consed opaque atoms for nonlinear subterms.
// mod/floordiv by a constant c are *eliminated* soundly:
//   - rem c: a symbolic term drops when c | coeff * divisibility(term)
//     (literal-coefficient or AxisInfo divisibility -- a for-all-unknowns fact);
//   - floordiv c: the c-divisible-coefficient terms are pulled into the quotient
//     and the bounded residual is resolved via range info.
// Coordinate variables are concrete at a fixed register, so the only symbolic
// register-variation comes from unknown monomials; register-invariant pieces
// (however nonlinear) cancel in offset(r) - offset(0) by structural identity.
//
// Contiguity is the largest power-of-two N such that, for ALL valuations of the
// unknowns, registers [0..N) hit N consecutive offsets -- established when the
// symbolic difference offset(r) - offset(0) reduces to the exact constant r.
//
// Returns 1 on any failure (non-tensor, missing register dim, unprovable).
// Never returns 0.
unsigned getPerThreadContiguityFromLinearLayout(
    Value offsetsValue, mlir::triton::ModuleAxisInfoAnalysis &axisAnalysis);

} // namespace mlir::triton::AMD

#endif // TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H
