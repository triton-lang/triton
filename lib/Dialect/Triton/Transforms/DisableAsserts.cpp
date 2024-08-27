#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include <memory>

#define GEN_PASS_DEF_TRITONDISABLEASSERTS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace mlir::triton {
namespace {

class DisableAssertsPass
    : public ::impl::TritonDisableAssertsBase<DisableAssertsPass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    m.walk([&](triton::AssertOp assertOp) { assertOp.erase(); });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createDisableAssertsPass() {
  return std::make_unique<DisableAssertsPass>();
}

} // namespace mlir::triton
