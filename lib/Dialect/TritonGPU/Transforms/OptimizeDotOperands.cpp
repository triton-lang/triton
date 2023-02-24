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

class OptimizeConvertToDotOperand : public mlir::RewritePattern {
public:
  explicit OptimizeConvertToDotOperand(mlir::MLIRContext *context)
      : RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(), 1,
                       context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = cast<triton::gpu::ConvertLayoutOp>(op);
    auto srcType = cvt.getOperand().getType().cast<RankedTensorType>();
    auto dstType = cvt.getResult().getType().cast<RankedTensorType>();
    // order
    ArrayRef<unsigned> order;
    if (auto srcBlockedLayout =
            srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>())
      order = srcBlockedLayout.getOrder();
    else if (auto srcSharedLayout =
                 srcType.getEncoding()
                     .dyn_cast<triton::gpu::SharedEncodingAttr>())
      order = srcSharedLayout.getOrder();
    else
      return failure();
    // dot operand output
    auto dstDotOperandLayout =
        dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
    if (!dstDotOperandLayout)
      return failure();
    if (!dstDotOperandLayout.getIsMMAv1Row())
      return failure();
    bool isMMAv1Row =
        dstDotOperandLayout.getIsMMAv1Row().cast<BoolAttr>().getValue();
    if ((order[0] == 1 && isMMAv1Row) || (order[0] == 0 && !isMMAv1Row))
      return failure();

    auto newIsRow = BoolAttr::get(op->getContext(), !isMMAv1Row);
    auto newDstEncoding = triton::gpu::DotOperandEncodingAttr::get(
        op->getContext(), dstDotOperandLayout.getOpIdx(),
        dstDotOperandLayout.getParent(), newIsRow);
    auto newDstType = RankedTensorType::get(
        dstType.getShape(), dstType.getElementType(), newDstEncoding);
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        op->getLoc(), newDstType, cvt.getOperand());
    rewriter.replaceOp(op, newCvt.getResult());
    return success();
  }
};

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

class MoveOpAfterLayoutConversion : public mlir::RewritePattern {

public:
  MoveOpAfterLayoutConversion(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto convertLayoutOp = cast<triton::gpu::ConvertLayoutOp>(op);
    auto retTy =
        convertLayoutOp.getResult().getType().dyn_cast<RankedTensorType>();
    if (!retTy)
      return failure();
    if (!isa<triton::gpu::DotOperandEncodingAttr>(retTy.getEncoding()))
      return failure();
    Operation *argOp = convertLayoutOp.getOperand().getDefiningOp();
    if (!argOp)
      return failure();
    if (argOp->getNumOperands() != 1)
      return failure();
    if (!isPure(argOp))
      return failure();

    if (!argOp->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() &&
        !argOp->hasTrait<mlir::OpTrait::Elementwise>())
      return failure();

    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(convertLayoutOp);
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        convertLayoutOp.getLoc(), convertLayoutOp.getOperand().getType(),
        argOp->getOperand(0));
    auto newOp = rewriter.clone(*argOp);
    newOp->setOperand(0, newCvt.getResult());
    newOp->getResult(0).setType(convertLayoutOp.getResult().getType());
    rewriter.replaceOp(convertLayoutOp, newOp->getResult(0));
    return success();
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
    patterns.add<OptimizeConvertToDotOperand>(context);
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
