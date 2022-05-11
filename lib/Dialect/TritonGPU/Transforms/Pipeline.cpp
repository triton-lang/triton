#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {
struct PipelinePass : public TritonGPUPipelineBase<PipelinePass> {
  void runOnOperation() override {
    getOperation()->walk([&](scf::ForOp forOp) {

    });
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUPipelinePass() {
  return std::make_unique<PipelinePass>();
}
