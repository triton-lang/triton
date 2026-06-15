#include "TritonAMDGPUTransforms/Passes.h"
#include "Dialect/TritonAMDGPU/Utility/TDMMergeUtility.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUMATERIALIZETDMMERGE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

struct TritonAMDGPUMaterializeTDMMergePass
    : impl::TritonAMDGPUMaterializeTDMMergeBase<
          TritonAMDGPUMaterializeTDMMergePass> {
  void runOnOperation() override {
    mlir::triton::AMD::materializeTDMMergeGroups(getOperation());
  }
};

} // namespace
} // namespace mlir
