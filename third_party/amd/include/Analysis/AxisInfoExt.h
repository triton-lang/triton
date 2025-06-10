#ifndef TRITONAMD_ANALYSIS_AXIS_INFO_EXT_H
#define TRITONAMD_ANALYSIS_AXIS_INFO_EXT_H

#include "include/triton/Analysis/AxisInfo.h"

namespace mlir::triton::AMD {
struct AxisInfoExt {
  static void addVisitors(mlir::triton::AxisInfoVisitorList &visitors);
};
} // namespace mlir::triton::AMD

#endif
