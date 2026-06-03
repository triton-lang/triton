#ifndef TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H
#define TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H

#include "mlir/IR/Value.h"

namespace mlir::triton {
class ModuleAxisInfoAnalysis;
}

namespace mlir::triton::AMD {

// Per-thread memory contiguity of an integer offsets tensor, measured along the
// LinearLayout "register" order. This complements AxisInfo for AMD buffer-load
// offsets whose true contiguity is formed across multiple tensor axes (for
// example MXFP4 scale pre-shuffles).
//
// Returns the largest proven power-of-two N such that every thread's registers
// [0..N) access N consecutive offsets. Returns 1 if the property cannot be
// proven (never 0).
unsigned getPerThreadContiguityFromLinearLayout(
    Value offsetsValue, mlir::triton::ModuleAxisInfoAnalysis &axisAnalysis);

} // namespace mlir::triton::AMD

#endif // TRITONAMD_ANALYSIS_CONSTANTTENSORVALUEANALYSIS_H
