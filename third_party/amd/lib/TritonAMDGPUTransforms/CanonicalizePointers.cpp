#include "TritonAMDGPUTransforms/Passes.h"
#include "TritonAMDGPUTransforms/PointerCanonicalizer.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

// This pass is simply calling the fat pointer canonicalization utility
// on the given MLIR module
class TritonAMDGPUCanonicalizePointersPass
    : public TritonAMDGPUCanonicalizePointersBase<
          TritonAMDGPUCanonicalizePointersPass> {
public:
  TritonAMDGPUCanonicalizePointersPass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    if (failed(triton::AMD::PointerCanonicalizer(m).run()))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUCanonicalizePointersPass() {
  return std::make_unique<TritonAMDGPUCanonicalizePointersPass>();
}
