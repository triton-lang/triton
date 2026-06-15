#include "Dialect/TritonAMDGPU/Utility/TDMCopyFuseUtility.h"
#include "TritonAMDGPUTransforms/Passes.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUAUTOFUSETDMCOPY
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

struct TritonAMDGPUAutoFuseTDMCopyPass
    : impl::TritonAMDGPUAutoFuseTDMCopyBase<TritonAMDGPUAutoFuseTDMCopyPass> {
  void runOnOperation() override {
    mlir::triton::AMD::autoFuseTDMCopies(getOperation());
  }
};

} // namespace
} // namespace mlir
