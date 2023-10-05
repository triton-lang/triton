#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <memory>

using namespace mlir;
namespace {
using triton::DotOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MmaEncodingAttr;
using triton::gpu::SharedEncodingAttr;
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
    // TODO(Qingyi): need to check whether the CTALayout of XEncoding should be
    // used here. For tests where numCTAs = 1, this is not a problem since all
    // CTALayouts are the same.
    auto newXOrder = triton::gpu::getOrder(argEncoding);
    auto newXEncoding = triton::gpu::SharedEncodingAttr::get(
        getContext(), ZEncoding, XType.getShape(), newXOrder,
        XEncoding.getCTALayout(), XType.getElementType());
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

// convert(layout_preserving_op(x), dot_operand)
// -> layout_preserving_op(convert(x, dot_operand))
class MoveOpAfterLayoutConversion : public mlir::RewritePattern {
public:
  MoveOpAfterLayoutConversion(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = cast<triton::gpu::ConvertLayoutOp>(op);
    // conversion should be dependent on a load
    // and all operations between the load and the conversion
    // should be layout preserving
    SetVector<Operation *> slice;
    getBackwardSlice(op, &slice);
    int loadIdx = -1;
    bool checkOp = false;
    for (int i = 0; i < slice.size(); i++) {
      Operation *currOp = *(slice.begin() + i);
      if (currOp->getParentRegion() != op->getParentRegion())
        continue;
      if (isa<triton::LoadOp>(currOp))
        checkOp = true;
      else if (checkOp) {
        if (!isa<triton::FpToFpOp, triton::BitcastOp>(currOp) &&
            currOp->getDialect()->getTypeID() !=
                mlir::TypeID::get<arith::ArithDialect>())
          return mlir::failure();
      }
    }
    if (!checkOp)
      return mlir::failure();

    auto cvtTy = cvt.getType().cast<RankedTensorType>();
    auto cvtArgOp = cvt.getSrc().getDefiningOp();
    if (!cvtArgOp || cvtArgOp->getNumOperands() == 0)
      return mlir::failure();
    // only consider custom conversions or arith ops
    if (!isa<triton::FpToFpOp, triton::BitcastOp>(cvtArgOp) &&
        cvtArgOp->getDialect()->getTypeID() !=
            mlir::TypeID::get<arith::ArithDialect>())
      return mlir::failure();
    // not handled in elementwise lowering.
    if (isa<arith::TruncIOp, arith::TruncFOp>(cvtArgOp))
      return mlir::failure();
    // only considers conversions to dot operand
    if (!cvtTy.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>())
      return mlir::failure();
    auto argTy = cvtArgOp->getOperand(0).getType().cast<RankedTensorType>();
    auto retTy = cvtArgOp->getResult(0).getType().cast<RankedTensorType>();
    if (!argTy || !retTy)
      return mlir::failure();
    Type newRetTy = RankedTensorType::get(
        retTy.getShape(), retTy.getElementType(), cvtTy.getEncoding());
    Type newCvtTy = RankedTensorType::get(
        retTy.getShape(), argTy.getElementType(), cvtTy.getEncoding());
    int numArgs = cvtArgOp->getNumOperands();
    SmallVector<triton::gpu::ConvertLayoutOp> newCvts(numArgs);
    for (int i = 0; i < numArgs; i++)
      newCvts[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
          cvt.getLoc(), newCvtTy, cvtArgOp->getOperand(i));
    auto newRet = rewriter.clone(*cvtArgOp);
    for (int i = 0; i < numArgs; i++)
      newRet->setOperand(i, newCvts[i]);
    newRet->getResult(0).setType(newRetTy);
    rewriter.replaceOp(op, newRet->getResults());
    return mlir::success();
  }
};

// convert(trans(convert(arg)))
// x = convert_layout arg: #distributed -> #shared_x
// y = trans x: #shared_x -> #shared_y
// z = convert_layout y: #shared_y -> #shared_z
class FuseTransHopper : public mlir::RewritePattern {

public:
  FuseTransHopper(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!op->hasOneUse())
      return mlir::failure();
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
        ZType.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
    if (!ZEncoding)
      return mlir::failure();
    // new X encoding
    auto newXOrder = triton::gpu::getOrder(argEncoding);

    auto dotOp = *op->getUsers().begin();
    if (isa<triton::DotOp, triton::nvidia_gpu::DotAsyncOp>(dotOp)) {
      auto dotTy = dotOp->getResult(0).getType().cast<RankedTensorType>();
      auto dotEncoding =
          dotTy.getEncoding().dyn_cast<triton::gpu::MmaEncodingAttr>();
      auto eltType = XType.getElementType();
      if (!dotEncoding || dotEncoding.getVersionMajor() != 3)
        return mlir::failure();
      // MMAv3 with transpose only supports f16 and bf16 data type
      // fallback to MMAv3 without transpose for other data types
      if (!eltType.isF16() && !eltType.isBF16()) {
        if (dstOp.getResult() == dotOp->getOperand(0)) {
          newXOrder = {0, 1};
        } else if (dstOp.getResult() == dotOp->getOperand(1)) {
          newXOrder = {1, 0};
        }
      }
    }

    // TODO(Qingyi): need to check whether the CTALayout of XEncoding should be
    // used here. For tests where numCTAs = 1, this is not a problem since all
    // CTALayouts are the same.
    auto newXEncoding = triton::gpu::SharedEncodingAttr::get(
        getContext(), XType.getShape(), newXOrder, XEncoding.getCTALayout(),
        XType.getElementType());
    auto newXType = RankedTensorType::get(XType.getShape(),
                                          XType.getElementType(), newXEncoding);

    auto newX = rewriter.create<triton::gpu::ConvertLayoutOp>(srcOp.getLoc(),
                                                              newXType, arg);
    rewriter.replaceOpWithNewOp<triton::TransOp>(dstOp, newX);
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
    patterns.add<FuseTransHopper>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUOptimizeDotOperandsPass() {
  return std::make_unique<TritonGPUOptimizeDotOperandsPass>();
}
