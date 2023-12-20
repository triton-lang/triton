#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

using namespace mlir;

static inline bool
willIncreaseRegisterPressure(triton::gpu::ConvertLayoutOp op) {
  auto srcType = op.getOperand().getType().cast<RankedTensorType>();
  auto dstType = op.getResult().getType().cast<RankedTensorType>();
  auto srcEncoding = srcType.getEncoding();
  auto dstEncoding = dstType.getEncoding();
  if (srcEncoding.isa<triton::gpu::SharedEncodingAttr>())
    return true;
  if (dstEncoding.isa<triton::gpu::DotOperandEncodingAttr>())
    return true;
  return false;
}

class TritonGPUReorderInstructionsPass
    : public TritonGPUReorderInstructionsBase<
          TritonGPUReorderInstructionsPass> {
public:
  TritonGPUReorderInstructionsPass() = default;

  Operation *getFirstUse(Operation *op) {
    std::vector<Operation *> users;
    for (auto user : op->getUsers()) {
      if (Operation *ancestor = op->getBlock()->findAncestorOpInBlock(*user))
        users.push_back(ancestor);
    }
    auto minOpIt = std::min_element(users.begin(), users.end(),
                                    [](mlir::Operation *a, mlir::Operation *b) {
                                      return a->isBeforeInBlock(b);
                                    });
    return minOpIt != users.end() ? *minOpIt : nullptr;
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    mlir::DominanceInfo dom(m);
    // sink conversion after the last dealloc
    // before the first use ancestor in its block
    m.walk([&](triton::gpu::ConvertLayoutOp op) {
      auto curr = mlir::Block::iterator(op);
      for (; &*curr != getFirstUse(op); curr++)
        if (isa<triton::gpu::DeallocTensorOp>(&*curr))
          op->moveAfter(&*curr);
    });
    // Sink conversions into loops when they will increase
    // register pressure
    DenseMap<Operation *, Operation *> opToMove;
    auto moveAfter = [](Operation *lhs, Operation *rhs) {
      auto lhsId = getWSRoleId(lhs);
      auto rhsId = getWSRoleId(rhs);
      if (lhsId == rhsId)
        lhs->moveAfter(rhs);
    };
    m.walk([&](triton::gpu::ConvertLayoutOp op) {
      if (!willIncreaseRegisterPressure(op))
        return;
      auto user_begin = op->user_begin();
      auto user_end = op->user_end();
      if (std::distance(user_begin, user_end) != 1)
        return;
      if (user_begin->getParentOfType<scf::ForOp>() ==
          op->getParentOfType<scf::ForOp>())
        return;
      opToMove.insert({op, *user_begin});
    });
    for (auto &kv : opToMove)
      kv.first->moveBefore(kv.second);
    // Move convert(load) immediately after dependent load
    m.walk([&](triton::gpu::ConvertLayoutOp op) {
      auto dstType = op.getResult().getType().cast<RankedTensorType>();
      auto dstEncoding = dstType.getEncoding();
      if (!dstEncoding.isa<triton::gpu::SharedEncodingAttr>())
        return;
      Operation *argOp = op.getOperand().getDefiningOp();
      if (!argOp)
        return;
      moveAfter(op, argOp);
    });
    // Move transpositions just after their definition
    opToMove.clear();
    m.walk([&](triton::TransOp op) {
      Operation *argOp = op.getOperand().getDefiningOp();
      if (!argOp)
        return;
      moveAfter(op, argOp);
    });
    // Move `dot` operand so that conversions to opIdx=1 happens after
    // conversions to opIdx=0
    m.walk([&](triton::gpu::ConvertLayoutOp op) {
      auto dstType = op.getResult().getType().cast<RankedTensorType>();
      auto dstEncoding =
          dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
      if (!dstEncoding)
        return;
      int opIdx = dstEncoding.getOpIdx();
      if (opIdx != 1)
        return;
      if (op->getUsers().empty())
        return;
      auto dotUser = dyn_cast<triton::DotOp>(*op->user_begin());
      if (!dotUser)
        return;
      auto AOp =
          dotUser.getOperand(0).getDefiningOp<triton::gpu::ConvertLayoutOp>();
      if (!AOp)
        return;
      // Check that the conversion to OpIdx=1 happens before and can be moved
      // after the conversion to OpIdx=0.
      if (!dom.dominates(op.getOperation(), AOp.getOperation()))
        return;
      moveAfter(op, AOp);
    });
    return;
  }
};

std::unique_ptr<Pass> mlir::triton::gpu::createReorderInstructionsPass() {
  return std::make_unique<TritonGPUReorderInstructionsPass>();
}
