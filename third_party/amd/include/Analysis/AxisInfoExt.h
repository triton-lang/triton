#ifndef TRITONAMD_ANALYSIS_AXIS_INFO_EXT_H
#define TRITONAMD_ANALYSIS_AXIS_INFO_EXT_H

#include "include/triton/Analysis/AxisInfo.h"

namespace mlir::triton::AMD {

struct AxisInfoExt {
  static void addVisitors(mlir::triton::AxisInfoVisitorList &visitors);
};

class ModuleAxisInfoAnalysis : public mlir::triton::ModuleAxisInfoAnalysis {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp)
      : mlir::triton::ModuleAxisInfoAnalysis(moduleOp,
                                             AxisInfoExt::addVisitors) {}
};
} // namespace mlir::triton::AMD

#endif
