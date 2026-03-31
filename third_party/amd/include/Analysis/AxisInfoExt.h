#ifndef TRITONAMD_ANALYSIS_AXIS_INFO_EXT_H
#define TRITONAMD_ANALYSIS_AXIS_INFO_EXT_H

#include "include/triton/Analysis/AxisInfo.h"

namespace mlir::triton::AMD {

class AxisInfoAnalysisExt : public triton::AxisInfoAnalysis {
public:
  AxisInfoAnalysisExt(DataFlowSolver &solver);

  static triton::AxisInfoAnalysis *loadAnalysis(DataFlowSolver *solver);
};

class ModuleAxisInfoAnalysis : public mlir::triton::ModuleAxisInfoAnalysis {
public:
  explicit ModuleAxisInfoAnalysis(ModuleOp moduleOp)
      : mlir::triton::ModuleAxisInfoAnalysis(
            moduleOp, AxisInfoAnalysisExt::loadAnalysis) {}
};
} // namespace mlir::triton::AMD

#endif
