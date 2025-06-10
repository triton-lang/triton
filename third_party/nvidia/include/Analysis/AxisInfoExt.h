#ifndef TRITONNVIDIA_ANALYSIS_AXIS_INFO_EXT_H
#define TRITONNVIDIA_ANALYSIS_AXIS_INFO_EXT_H

#include "include/triton/Analysis/AxisInfo.h"

namespace mlir::triton::nvidia {
struct AxisInfoExt {
  static void addVisitors(mlir::triton::AxisInfoVisitorList &visitors) {}
};
} // namespace mlir::triton::nvidia

#endif
