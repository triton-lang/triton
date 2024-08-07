#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Verifier.h"

using namespace mlir;
namespace ttg = triton::gpu;

namespace {

// // convert(val) : mma -> blocked
// // tt.store(ptr, val, mask, ...) : blocked
// // ==>
// // convert(ptr) : blocked -> mma
// // convert(mask) : blocked -> mma
// // tt.store(ptr, val, mask, ...) : mma
// //
// // Store with mma layout directly

// static Type getNewType(Type type, Attribute encoding) {
//   RankedTensorType tensorType = type.cast<RankedTensorType>();
//   return RankedTensorType::get(tensorType.getShape(),
//                                tensorType.getElementType(), encoding);
// }

// void convertLayout(Attribute encoding, Operation *op) {
//   OpBuilder builder(op);
//   // Convert operands
//   // For load/store with tensor pointers, we don't have to change the
//   // operands' type, we do this by changing the outputs' type of
//   // `make_tensor_ptr`
//   SmallVector<Value, 4> newArgs;
//   for (auto operand : op->getOperands()) {
//     auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
//     if (tensorType &&
//         !tensorType.getEncoding().isa<ttg::SharedEncodingAttr>()) {
//       Type newType = getNewType(tensorType, encoding);
//       newArgs.push_back(
//           builder.create<ttg::ConvertLayoutOp>(op->getLoc(), newType, operand));
//     } else {
//       newArgs.push_back(operand);
//     }
//   }

//   // Convert output types
//   SmallVector<Type, 4> newTypes;
//   for (auto t : op->getResultTypes()) {
//     bool isAsync = isa<ttg::InsertSliceAsyncOp>(op);
//     newTypes.push_back(isAsync ? t : getNewType(t, encoding));
//   }

//   // Construct new op with the new encoding
//   Operation *newOp = builder.create(op->getLoc(), op->getName().getIdentifier(),
//                                     newArgs, newTypes, op->getAttrs());

//   // Cast the results back to the original layout
//   for (size_t i = 0; i < op->getNumResults(); i++) {
//     Value newResult = newOp->getResult(i);
//     if (newTypes[i] != op->getResultTypes()[i]) {
//       newResult = builder.create<ttg::ConvertLayoutOp>(
//           op->getLoc(), op->getResult(i).getType(), newResult);
//     }
//     op->getResult(i).replaceAllUsesWith(newResult);
//   }
//   op->erase();
// }

// triton::LoadOp getLoadInst(Operation *op, ModuleOp &mod) {
//   SmallVector<triton::LoadOp> loadOpsVec;

//   mod.walk([&](triton::LoadOp loadOp) {
//     SetVector<Operation *> forwardSlices;
//     getForwardSlice((Operation *)loadOp, &forwardSlices);
//     if (std::find(forwardSlices.begin(), forwardSlices.end(), op) !=
//         forwardSlices.end()) {
//       loadOpsVec.push_back(loadOp);
//     }
//   });

//   // Currently, we expect the dot operand to depend only on one tensor
//   // from global memory (applicable for dot ops that don't depend on other dot
//   // ops). This condition can be lifted if necessary.
//   assert(loadOpsVec.size() == 1);
//   return loadOpsVec[0];
// }

class BypassLDSForDotLayout : public mlir::RewritePattern {

public:
  explicit BypassLDSForDotLayout(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(), 1, context) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {

    // auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
    // auto mod = op->getParentOfType<ModuleOp>();
    // static int counter = 0;
    // if(counter > 0){
    //   return mlir::failure();
    // }
    // if (!cvtOp)
    //   return mlir::failure();

    // auto srcType = cvtOp.getOperand().getType().cast<RankedTensorType>();
    // auto dstType = cvtOp.getType().cast<RankedTensorType>();
    // auto srcBlocked =
    //     srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
    // auto dstDotOp =
    //     dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
    // if (!(srcBlocked && dstDotOp)) {
    //   return mlir::failure();
    // }

    // if(dstDotOp.getOpIdx() != 0){
    //   return mlir::failure();
    // }

    // if(srcBlocked.getOrder()[0] == 0){
    //   return mlir::failure();
    // }

    // SmallVector<unsigned> newWarpsPerCTA(2, 4);
    // SmallVector<unsigned> newSizePerThread(2, 4);
    // SmallVector<unsigned> newThreadsPerWarp(2, 4);
    // SmallVector<unsigned> newOrder(2, 4);
    // newOrder[0] = 0;
    // newOrder[1] = 1;
    // newThreadsPerWarp[0] = 32;
    // newThreadsPerWarp[1] = 2;
    // newSizePerThread[0] = 1;
    // newSizePerThread[1] = 8;
    // newWarpsPerCTA[0] = 4;
    // newWarpsPerCTA[1] = 1;
    // auto newBlockedEncoding = triton::gpu::BlockedEncodingAttr::get(
    //     mod.getContext(), newSizePerThread, newThreadsPerWarp, newWarpsPerCTA,
    //     newOrder, srcBlocked.getCTALayout());

    // auto loadInst = getLoadInst(cvtOp, mod);
    // convertLayout(newBlockedEncoding, (Operation *)loadInst);
    // if (failed(mlir::verify(mod))) {
    //   assert(false);
    // }
    // counter+= 1;
    return mlir::success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonAMDGPUBypassLDSForDotLayoutPass
    : public TritonAMDGPUBypassLDSForDotLayoutBase<TritonAMDGPUBypassLDSForDotLayoutPass> {

public:
  TritonAMDGPUBypassLDSForDotLayoutPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<BypassLDSForDotLayout>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUBypassLDSForDotLayout() {
  return std::make_unique<TritonAMDGPUBypassLDSForDotLayoutPass>();
}
