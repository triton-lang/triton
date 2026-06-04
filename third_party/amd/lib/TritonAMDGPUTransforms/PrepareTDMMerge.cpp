#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/TDMUtility.h"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUPREPARETDMMERGE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

struct TritonAMDGPUPrepareTDMMergePass
    : impl::TritonAMDGPUPrepareTDMMergeBase<TritonAMDGPUPrepareTDMMergePass> {
  void runOnOperation() override {
    mlir::LLVM::AMD::assignTDMMergeGroupIds(getOperation());
  }
};

} // namespace
} // namespace mlir
