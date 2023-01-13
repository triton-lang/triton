#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
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

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    // Sink conversions into loops when they will increase
    // register pressure
    DenseMap<Operation *, Operation *> opToMove;
    m.walk([&](triton::gpu::ConvertLayoutOp op) {
      if (!willIncreaseRegisterPressure(op))
        return;
      auto user_begin = op->user_begin();
      auto user_end = op->user_end();
      if (std::distance(user_begin, user_end) != 1)
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
      op->moveAfter(argOp);
    });
    // Move transpositions just after their definition
    opToMove.clear();
    m.walk([&](triton::TransOp op) {
      Operation *argOp = op.getOperand().getDefiningOp();
      if (!argOp)
        return;
      op->moveAfter(argOp);
    });
    // Move `dot` operand so that conversions to opIdx=0 happens before
    // conversions to opIdx=1
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
      auto user_begin = op->user_begin();
      op->moveBefore(*user_begin);
    });
    return;
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUReorderInstructionsPass() {
  return std::make_unique<TritonGPUReorderInstructionsPass>();
}
