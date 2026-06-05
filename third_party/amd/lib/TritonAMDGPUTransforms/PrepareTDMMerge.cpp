#include "Dialect/TritonAMDGPU/Utility/TDMMergeUtility.h"
#include "TritonAMDGPUTransforms/Passes.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUPREPARETDMMERGE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

struct TritonAMDGPUPrepareTDMMergePass
    : impl::TritonAMDGPUPrepareTDMMergeBase<TritonAMDGPUPrepareTDMMergePass> {
  void runOnOperation() override {
    mlir::triton::AMD::assignTDMMergeGroupIds(getOperation());
  }
};

} // namespace
} // namespace mlir
