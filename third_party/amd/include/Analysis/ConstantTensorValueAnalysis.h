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

// LinearLayout-native per-register contiguity (recommended replacement for
// `getPerThreadConsecutiveContiguity`).
//
// Instead of walking the per-register offsets sequentially and probing two
// unknown-scalar substitutions, this:
//   1. Uses the offsets tensor's LinearLayout to map each `register` *basis*
//      bit (lane=warp=block=0) to a tensor coord, and evaluates the offset
//      delta that flipping that single bit produces -- i.e. it recovers the
//      basis images of the register->offset map, which is exactly how a
//      LinearLayout is represented.
//   2. VERIFIES that register->offset is actually GF(2)-linear over the whole
//      register subspace (every composite register value equals the XOR-free
//      sum of its set-bit basis deltas -- the "disjoint / no-carry" property).
//      A function that is merely contiguous on a prefix but not linear is
//      rejected, closing the prefix-only blind spot of the sequential walk.
//   3. Establishes independence from kernel/loop scalars STRUCTURALLY by
//      requiring the recovered basis images to be identical across several
//      unknown-scalar substitutions (strictly stronger than the 2-probe).
//
// Contiguity is then read directly off the verified linear map: the largest
// power-of-two N such that basis bit b maps to offset 2^b for every b < log2 N.
//
// Returns 1 on any failure (non-tensor, missing register dim, non-linear map,
// scalar-dependent stride). Never returns 0.
unsigned getPerThreadContiguityFromLinearLayout(
    Value offsetsValue, mlir::triton::ModuleAxisInfoAnalysis &axisAnalysis);

} // namespace mlir::triton::AMD

#endif // TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H
