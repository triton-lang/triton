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

static inline bool willIncreaseRegisterPressure(triton::gpu::ConvertLayoutOp op) {
  auto srcType = op.getOperand().getType().cast<RankedTensorType>();
  auto dstType = op.getResult().getType().cast<RankedTensorType>();
  auto srcEncoding = srcType.getEncoding();
  auto dstEncoding = dstType.getEncoding();
  if(srcEncoding.isa<triton::gpu::SharedEncodingAttr>())
    return true;
  if(dstEncoding.isa<triton::gpu::DotOperandEncodingAttr>())
    return true;
  return false;
}

class TritonGPUSinkConversionsFromSharedPass
    : public TritonGPUSinkConversionsFromSharedBase<TritonGPUSinkConversionsFromSharedPass> {
public:
  TritonGPUSinkConversionsFromSharedPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    // Move convert(load) immediately after dependent load
    m.walk([&](triton::gpu::ConvertLayoutOp op){
      auto load = dyn_cast<triton::LoadOp>(op.getOperand().getDefiningOp());
      if(!load)
        return;
      op->moveAfter(load);
    });
    // Sink conversions into loops when they will increase
    // register pressure
    DenseMap<Operation*, Operation *> opToMove;
    m.walk([&](triton::gpu::ConvertLayoutOp op){
      if(!willIncreaseRegisterPressure(op))
        return;
      auto user_begin = op->user_begin();
      auto user_end = op->user_end();
      if(std::distance(user_begin, user_end) != 1)
        return;
      opToMove.insert({op, *user_begin});
    });
    for(auto &kv: opToMove)
      kv.first->moveBefore(kv.second);
    
    // Move transpositions just before their first use
    opToMove.clear();
    m.walk([&](triton::TransOp op){
      auto user_begin = op->user_begin();
      opToMove.insert({op, *user_begin});
    });
    for(auto &kv: opToMove)
      kv.first->moveBefore(kv.second);


    return;
  }
};

std::unique_ptr<Pass>
mlir::createTritonGPUSinkConversionsFromSharedPass() {
  return std::make_unique<TritonGPUSinkConversionsFromSharedPass>();
}
