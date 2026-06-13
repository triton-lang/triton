#include "TritonAMDGPUTransforms/Passes.h"
#include "Dialect/TritonAMDGPU/Utility/TDMMergeUtility.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUPREPARETDMMERGE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

struct TritonAMDGPUPrepareTDMMergePass
    : impl::TritonAMDGPUPrepareTDMMergeBase<TritonAMDGPUPrepareTDMMergePass> {
  void runOnOperation() override {
    mlir::triton::AMD::materializeTDMMergeGroups(getOperation());
  }
};

} // namespace
} // namespace mlir
