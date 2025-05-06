#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

void eraseResult(Operation *op, unsigned resultIdx, Value replacement) {
  OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                       op->getResultTypes(), op->getAttrs());
  state.types.erase(std::next(state.types.begin(), resultIdx));
  OpBuilder b(op);
  Operation *newOp = b.create(state);
  SmallVector<Value> replacements = newOp->getResults();
  replacements.insert(std::next(replacements.begin(), resultIdx), replacement);
  op->replaceAllUsesWith(replacements);
  op->erase();
}

void removeTMEMToken(Operation *op, Value dummy) {
  if (auto mmaOp = dyn_cast<MMAv5OpInterface>(op)) {
    mmaOp.getAccDepMutable().clear();
    if (mmaOp.getToken())
      eraseResult(mmaOp, 0, dummy);
  } else if (auto store = dyn_cast<TMEMStoreOp>(op)) {
    store.getDepMutable().clear();
    if (store.getToken())
      eraseResult(store, 0, dummy);
  } else if (auto alloc = dyn_cast<TMEMAllocOp>(op)) {
    if (alloc.getToken())
      eraseResult(alloc, 1, dummy);
  } else if (auto load = dyn_cast<TMEMLoadOp>(op)) {
    load.getDepMutable().clear();
    if (load.getToken())
      eraseResult(load, 1, dummy);
  }
}

class TritonNvidiaGPURemoveTMEMTokensPass
    : public TritonNvidiaGPURemoveTMEMTokensPassBase<
          TritonNvidiaGPURemoveTMEMTokensPass> {
public:
  using TritonNvidiaGPURemoveTMEMTokensPassBase::
      TritonNvidiaGPURemoveTMEMTokensPassBase;

  void runOnOperation() override {
    for (auto func : getOperation().getOps<FuncOp>()) {
      auto b = OpBuilder::atBlockBegin(&func.getBody().front());
      // Placeholder value that will get DCE'd by the canonicalizer.
      Value dummy =
          b.create<ub::PoisonOp>(func.getLoc(), b.getType<AsyncTokenType>());
      func.walk([&](Operation *op) { removeTMEMToken(op, dummy); });
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPURemoveTMEMTokensPass() {
  return std::make_unique<TritonNvidiaGPURemoveTMEMTokensPass>();
}
