#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_ANALYSIS_AXISINFOEXT_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_ANALYSIS_AXISINFOEXT_H_

#include "triton/Analysis/AxisInfo.h"

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

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_ANALYSIS_AXISINFOEXT_H_
