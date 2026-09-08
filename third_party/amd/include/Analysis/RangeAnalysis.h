#ifndef TRITONAMD_ANALYSIS_RANGE_ANALYSIS_H
#define TRITONAMD_ANALYSIS_RANGE_ANALYSIS_H

#include "triton/Analysis/RangeAnalysis.h"

namespace mlir::triton::AMD {

using triton::collectRanges;
using triton::evaluateCmpI;
using triton::initializeFuncOps;
using triton::isEmptyInitializedRange;
using triton::populateFoldTrueCmpIOpPatterns;
using triton::TritonIntegerRangeAnalysis;

} // namespace mlir::triton::AMD

#endif
