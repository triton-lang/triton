#ifndef TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H
#define TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H

#include "mlir/IR/Value.h"

namespace mlir::triton {
class ModuleAxisInfoAnalysis;
}

namespace mlir::triton::AMD {

// Proves per-thread memory-offset contiguity in LinearLayout "register" order.
// AxisInfo describes per-axis properties; this analysis symbolically evaluates
// the integer offset expression at successive register coordinates so it can
// prove contiguity created by combining multiple tensor axes, e.g. AMD
// buffer-load offsets after layout/pre-shuffle arithmetic.
//
// The proof is universal over program-id/function-argument scalars and over
// non-register LinearLayout coordinates such as lane/warp/block. Any dependence
// that cannot be eliminated, shown register-invariant, or reduced exactly
// prevents the proven contiguity from increasing.
//
// Returns the largest proven power-of-two N such that every thread's registers
// [0..N) access N consecutive offsets. Returns 1 if the property cannot be
// proven; never returns 0.
unsigned getPerThreadContiguityFromLinearLayout(
    Value offsetsValue, mlir::triton::ModuleAxisInfoAnalysis &axisAnalysis);

} // namespace mlir::triton::AMD

#endif // TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H
