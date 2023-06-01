#include "Utility.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <memory>

using namespace mlir;
namespace {
using triton::DotOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MmaEncodingAttr;
using triton::gpu::SliceEncodingAttr;

// convert(trans(convert(arg)))
// x = convert_layout arg: #distributed -> #shared_x
// y = trans x: #shared_x -> #shared_y
// z = convert_layout y: #shared_y -> #dot_operand
class ConvertTransConvert : public mlir::RewritePattern {

public:
  ConvertTransConvert(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dstOp = cast<triton::gpu::ConvertLayoutOp>(op);
    auto tmpOp =
        dyn_cast_or_null<triton::TransOp>(dstOp.getSrc().getDefiningOp());
    if (!tmpOp)
      return mlir::failure();
    auto srcOp = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
        tmpOp.getSrc().getDefiningOp());
    if (!srcOp)
      return mlir::failure();
    auto arg = srcOp.getSrc();
    auto X = tmpOp.getSrc();
    // types
    auto argType = arg.getType().cast<RankedTensorType>();
    auto XType = X.getType().cast<RankedTensorType>();
    auto ZType = dstOp.getResult().getType().cast<RankedTensorType>();
    // encodings
    auto argEncoding = argType.getEncoding();
    auto XEncoding =
        XType.getEncoding().cast<triton::gpu::SharedEncodingAttr>();
    auto ZEncoding =
        ZType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
    if (!ZEncoding)
      return mlir::failure();
    // new X encoding
    auto newXOrder = triton::gpu::getOrder(argEncoding);
    auto newXEncoding = triton::gpu::SharedEncodingAttr::get(
        getContext(), ZEncoding, XType.getShape(), newXOrder,
        XType.getElementType());
    auto newXType = RankedTensorType::get(XType.getShape(),
                                          XType.getElementType(), newXEncoding);
    if (XEncoding == newXEncoding)
      return mlir::failure();

    auto newX = rewriter.create<triton::gpu::ConvertLayoutOp>(srcOp.getLoc(),
                                                              newXType, arg);
    auto newY = rewriter.create<triton::TransOp>(tmpOp.getLoc(), newX);
    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(dstOp, ZType,
                                                              newY);
    return mlir::success();
  }
};

//

class MoveOpAfterLayoutConversion : public mlir::RewritePattern {

public:
  MoveOpAfterLayoutConversion(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  static mlir::LogicalResult
  isBlockedToDotOperand(mlir::Operation *op,
                        triton::gpu::DotOperandEncodingAttr &retEncoding,
                        triton::gpu::BlockedEncodingAttr &srcEncoding) {
    if (!op)
      return failure();
    auto cvt = cast<triton::gpu::ConvertLayoutOp>(op);
    auto srcTy = cvt.getOperand().getType().cast<RankedTensorType>();
    auto retTy = cvt.getResult().getType().dyn_cast<RankedTensorType>();
    retEncoding =
        retTy.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
    srcEncoding =
        srcTy.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
    if (!retTy)
      return failure();
    if (!retEncoding)
      return failure();
    auto retEncodingParent =
        retEncoding.getParent().dyn_cast<triton::gpu::MmaEncodingAttr>();
    if (!retEncodingParent || retEncodingParent.isVolta())
      return failure();
    if (!srcEncoding)
      return failure();
    return success();
  }

  static bool isTrans(const triton::gpu::DotOperandEncodingAttr &retEncoding,
                      const triton::gpu::BlockedEncodingAttr &srcEncoding) {
    int kOrder = retEncoding.getOpIdx() ^ 1;
    return kOrder != srcEncoding.getOrder()[0];
  }

  static bool isDotNT(triton::DotOp dotOp) {
    triton::gpu::DotOperandEncodingAttr aRetEncoding;
    triton::gpu::DotOperandEncodingAttr bRetEncoding;
    triton::gpu::BlockedEncodingAttr aSrcEncoding;
    triton::gpu::BlockedEncodingAttr bSrcEncoding;
    if (isBlockedToDotOperand(dotOp.getOperand(0).getDefiningOp(), aRetEncoding,
                              aSrcEncoding)
            .failed())
      return false;
    if (isBlockedToDotOperand(dotOp.getOperand(1).getDefiningOp(), bRetEncoding,
                              bSrcEncoding)
            .failed())
      return false;
    if (!aRetEncoding || !bRetEncoding || !aSrcEncoding || !bSrcEncoding)
      return false;
    return !isTrans(aRetEncoding, aSrcEncoding) &&
           !isTrans(bRetEncoding, bSrcEncoding);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = cast<triton::gpu::ConvertLayoutOp>(op);
    triton::gpu::DotOperandEncodingAttr retEncoding;
    triton::gpu::BlockedEncodingAttr srcEncoding;
    if (isBlockedToDotOperand(op, retEncoding, srcEncoding).failed())
      return mlir::failure();

    // only supports dot NT
    auto users = cvt->getUsers();
    auto dotOp = dyn_cast_or_null<DotOp>(*users.begin());
    if (!dotOp)
      return failure();
    if (!isDotNT(dotOp))
      return failure();

    // don't move things around when cvt operand is a block arg
    Operation *argOp = cvt.getOperand().getDefiningOp();
    if (!argOp)
      return failure();
    //
    SetVector<Operation *> processed;
    SetVector<Attribute> layout;
    llvm::MapVector<Value, Attribute> toConvert;
    int numCvts = simulateBackwardRematerialization(cvt, processed, layout,
                                                    toConvert, retEncoding);
    if (numCvts > 1 || toConvert.size() == 1)
      return failure();
    for (Operation *op : processed) {
      if (op->getNumOperands() != 1)
        continue;
      auto srcTy = op->getOperand(0).getType().cast<RankedTensorType>();
      auto dstTy = op->getResult(0).getType().cast<RankedTensorType>();
      // we don't want to push conversions backward if there is a downcast
      // since it would result in more shared memory traffic
      if (srcTy.getElementType().getIntOrFloatBitWidth() >
          dstTy.getElementType().getIntOrFloatBitWidth())
        return failure();
      // we only push back when the first op in the chain has a load operand
      if ((op == processed.back()) &&
          !isa<triton::LoadOp>(op->getOperand(0).getDefiningOp()))
        return failure();
      // we don't want to use ldmatrix for 8-bit data that requires trans
      // since Nvidia GPUs can't do it efficiently
      int kOrder = retEncoding.getOpIdx() ^ 1;
      bool isTrans = kOrder != srcEncoding.getOrder()[0];
      bool isInt8 = srcTy.getElementType().getIntOrFloatBitWidth() == 8;
      if (isTrans && isInt8)
        return failure();
    }
    IRMapping mapping;
    rematerializeConversionChain(toConvert, rewriter, processed, mapping);
    rewriter.replaceOp(cvt, mapping.lookup(cvt->getOperand(0)));
    return mlir::success();
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUOptimizeDotOperandsPass
    : public TritonGPUOptimizeDotOperandsBase<
          TritonGPUOptimizeDotOperandsPass> {
public:
  TritonGPUOptimizeDotOperandsPass() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::PassManager pm(m.getContext());
    pm.addPass(mlir::createCanonicalizerPass());
    auto ret = pm.run(m);

    mlir::RewritePatternSet patterns(context);
    patterns.add<ConvertTransConvert>(context);
    patterns.add<MoveOpAfterLayoutConversion>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
    if (fixupLoops(m).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUOptimizeDotOperandsPass() {
  return std::make_unique<TritonGPUOptimizeDotOperandsPass>();
}
