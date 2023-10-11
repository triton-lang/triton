#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <memory>
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;

class TritonGPUOptimizeThreadLocalityPass
    : public TritonGPUOptimizeThreadLocalityBase<
          TritonGPUOptimizeThreadLocalityPass> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([](triton::ReduceOp reduce)) {}
  };
};

std::unique_ptr<Pass> mlir::createTritonGPUOptimizeThreadLocalityPass() {
  return std::make_unique<TritonGPUOptimizeThreadLocalityPass>();
}
