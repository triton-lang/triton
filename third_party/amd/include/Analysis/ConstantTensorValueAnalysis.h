#ifndef TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H
#define TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>

namespace mlir::triton {
class ModuleAxisInfoAnalysis;
}

namespace mlir::triton::AMD {

// Evaluate an integer tensor value at a specific tensor coordinate. Walks the
// producer chain through arith.{addi,subi,muli,divsi,divui,remsi,remui,shli,
// shrsi,shrui,andi,ori,xori}, arith.constant, arith.{extsi,extui,trunci},
// tt.make_range, tt.splat, tt.broadcast, tt.expand_dims, ttg.convert_layout.
// Non-constant scalars (e.g. kernel args, block args) are substituted with
// `unknownScalarSubst` -- callers that need to prove a result is independent
// of unknowns should probe with multiple substitutions and compare.
//
// Returns std::nullopt if any op in the chain is unsupported.
std::optional<int64_t> evaluateAt(Value value, ArrayRef<int64_t> coord,
                                  int64_t unknownScalarSubst = 0);

// Per-register memory contiguity for an integer offsets tensor whose linear
// layout's "register" dimension maps to tensor-coord deltas. Samples
// `evaluateAt` at lane=warp=0 and each register basis vector, taking the
// difference from the base register. Returns the largest power-of-two N such
// that register indices [0..N) produce memory deltas [0..N) (i.e. N
// consecutive elements per thread).
//
// To prove independence from kernel-argument scalars, the deltas are sampled
// at TWO unknown-scalar substitutions: 0 and the AxisInfo-derived
// divisibility of the offsets value. If the per-register deltas disagree
// between the two probes, the function returns 1 (cannot prove contiguity).
//
// Returns 1 on any failure. Never returns 0.
unsigned getPerThreadConsecutiveContiguity(
    Value offsetsValue, mlir::triton::ModuleAxisInfoAnalysis &axisAnalysis);

} // namespace mlir::triton::AMD

#endif // TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H
