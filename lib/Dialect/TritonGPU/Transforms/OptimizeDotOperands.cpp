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
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 1, context) {}

  static mlir::LogicalResult
  isBlockedToDotOperand(mlir::Operation *op,
                        triton::gpu::DotOperandEncodingAttr &retEncoding,
                        triton::gpu::BlockedEncodingAttr &srcEncoding) {
    auto cvt = dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(op);
    if (!cvt)
      return failure();
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
    auto dotOp = cast<triton::DotOp>(op);
    // only supports dot NT
    if (!isDotNT(dotOp))
      return failure();
    bool changed = false;
    for (Value operand : {dotOp.getOperand(0), dotOp.getOperand(1)}) {
      auto cvt = operand.getDefiningOp<triton::gpu::ConvertLayoutOp>();
      triton::gpu::DotOperandEncodingAttr retEncoding;
      triton::gpu::BlockedEncodingAttr srcEncoding;
      bool failed =
          isBlockedToDotOperand(cvt, retEncoding, srcEncoding).failed();
      assert(!failed);

      // don't move things around when cvt operand is a block arg
      Operation *argOp = cvt.getOperand().getDefiningOp();
      if (!argOp)
        continue;
      SetVector<Operation *> processed;
      SetVector<Attribute> layout;
      llvm::MapVector<Value, Attribute> toConvert;
      int numCvts = simulateBackwardRematerialization(cvt, processed, layout,
                                                      toConvert, retEncoding);
      if (numCvts > 1 || toConvert.size() == 1)
        continue;
      bool replaceOperand = true;
      for (Operation *op : processed) {
        if (op->getNumOperands() != 1)
          continue;
        auto srcTy = op->getOperand(0).getType().cast<RankedTensorType>();
        auto dstTy = op->getResult(0).getType().cast<RankedTensorType>();
        // we don't want to push conversions backward if there is a downcast
        // since it would result in more shared memory traffic
        if (srcTy.getElementType().getIntOrFloatBitWidth() >
            dstTy.getElementType().getIntOrFloatBitWidth()) {
          replaceOperand = false;
          break;
        }
        // we only push back when the first op in the chain has a load operand
        if ((op == processed.back()) &&
            !isa<triton::LoadOp>(op->getOperand(0).getDefiningOp())) {
          replaceOperand = false;
          break;
        }
        // we don't want to use ldmatrix for 8-bit data that requires trans
        // since Nvidia GPUs can't do it efficiently
        int kOrder = retEncoding.getOpIdx() ^ 1;
        bool isTrans = kOrder != srcEncoding.getOrder()[0];
        bool isInt8 = srcTy.getElementType().getIntOrFloatBitWidth() == 8;
        if (isTrans && isInt8) {
          replaceOperand = false;
          break;
        }
      }
      if (!replaceOperand)
        continue;
      IRMapping mapping;
      rematerializeConversionChain(toConvert, rewriter, processed, mapping);
      rewriter.replaceOp(cvt, mapping.lookup(cvt->getOperand(0)));
      changed = true;
    }
    return mlir::success(changed);
  }
};

} // namespace

static bool isConvertToDotEncoding(Operation *op) {
  auto convertLayout = llvm::dyn_cast<ConvertLayoutOp>(op);
  if (!convertLayout)
    return false;
  auto tensorType =
      convertLayout.getResult().getType().cast<RankedTensorType>();
  return tensorType.getEncoding().isa<DotOperandEncodingAttr>();
}

static ConvertLayoutOp updateConvert(OpBuilder &builder, ConvertLayoutOp cvt,
                                     IRMapping &mapping, Type smallestType) {
  auto cvtDstTy = cvt.getResult().getType().cast<RankedTensorType>();
  auto cvtDstEnc = cvtDstTy.getEncoding().cast<DotOperandEncodingAttr>();
  Value operand = cvt.getOperand();
  if (mapping.contains(operand))
    operand = mapping.lookup(operand);
  auto newDstTy = RankedTensorType::get(
      cvtDstTy.getShape(), cvtDstTy.getElementType(),
      DotOperandEncodingAttr::get(cvtDstEnc.getContext(), cvtDstEnc.getOpIdx(),
                                  cvtDstEnc.getParent(), smallestType));
  auto newCvt =
      builder.create<ConvertLayoutOp>(cvt.getLoc(), newDstTy, operand);
  mapping.map(cvt.getResult(), newCvt.getResult());
  return newCvt;
}

// Update kWidth based on the smallestType found in the given convert ops and
// propagate the type change.
static void
updateDotEncodingLayout(SmallVector<ConvertLayoutOp> &convertsToDotEncoding,
                        Type smallestType) {
  IRMapping mapping;
  OpBuilder builder(smallestType.getContext());
  SetVector<Operation *> slices(convertsToDotEncoding.begin(),
                                convertsToDotEncoding.end());
  // Collect all the operations where the type needs to be propagated.
  for (auto cvt : convertsToDotEncoding) {
    auto forwardFilter = [&](Operation *op) {
      if (op == cvt.getOperation())
        return true;
      for (Value operand : op->getOperands()) {
        auto tensorType = operand.getType().dyn_cast<RankedTensorType>();
        if (tensorType &&
            tensorType.getEncoding().isa<DotOperandEncodingAttr>())
          return true;
      }
      return false;
    };
    auto backwardFilter = [&](Operation *op) {
      for (Value results : op->getResults()) {
        auto tensorType = results.getType().dyn_cast<RankedTensorType>();
        if (tensorType &&
            tensorType.getEncoding().isa<DotOperandEncodingAttr>())
          return true;
      }
      return false;
    };
    SetVector<Operation *> opSlice =
        getSlice(cvt.getOperation(), {backwardFilter}, {forwardFilter});
    slices.insert(opSlice.begin(), opSlice.end());
  }
  // Apply the type change by walking ops in topological order.
  slices = mlir::topologicalSort(slices);
  for (Operation *op : slices) {
    builder.setInsertionPoint(op);
    if (isConvertToDotEncoding(op)) {
      auto cvt = cast<ConvertLayoutOp>(op);
      ConvertLayoutOp newCvt =
          updateConvert(builder, cvt, mapping, smallestType);
      continue;
    }
    auto *newOp = cloneWithInferType(builder, op, mapping);
    for (auto [result, newResult] :
         llvm::zip(op->getResults(), newOp->getResults())) {
      result.replaceUsesWithIf(newResult, [&](OpOperand &operand) {
        return slices.count(operand.getOwner()) == 0;
      });
    }
  }
  for (Operation *op : llvm::reverse(slices))
    op->erase();
}

// Change the layout of dotOperand layout to use the kWidth from the smallest
// loaded type. This allows better code generation for mixed-mode matmul.
static void optimizeKWidth(triton::FuncOp func) {
  SmallVector<ConvertLayoutOp> convertsToDotEncoding;
  Type smallestType;
  func->walk([&](triton::LoadOp loadOp) {
    if (!loadOp.getResult().hasOneUse())
      return;
    Operation *use = *loadOp.getResult().getUsers().begin();

    // Advance to the first conversion as long as the use resides in shared
    // memory and it has a single use itself
    while (use) {
      if (use->getNumResults() != 1 || !use->getResult(0).hasOneUse())
        break;
      auto tensorType =
          use->getResult(0).getType().dyn_cast<RankedTensorType>();
      if (!tensorType || !tensorType.getEncoding().isa<SharedEncodingAttr>())
        break;
      use = *use->getResult(0).getUsers().begin();
    }

    auto convertLayout = llvm::dyn_cast<ConvertLayoutOp>(use);
    if (!convertLayout)
      return;
    auto tensorType =
        convertLayout.getResult().getType().cast<RankedTensorType>();
    if (!tensorType.getEncoding().isa<DotOperandEncodingAttr>())
      return;
    convertsToDotEncoding.push_back(convertLayout);

    // Update the smallest type.
    auto ty = loadOp.getType().cast<RankedTensorType>();
    Type eltTy = ty.getElementType();
    if (!smallestType ||
        (eltTy.getIntOrFloatBitWidth() < smallestType.getIntOrFloatBitWidth()))
      smallestType = eltTy;
  });
  if (!smallestType)
    return;
  updateDotEncodingLayout(convertsToDotEncoding, smallestType);
}

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

    // Change the layout of dotOperand layout to use the kWidth from the
    // smallest loaded type.
    m->walk([](triton::FuncOp func) { optimizeKWidth(func); });
  }
};

std::unique_ptr<Pass> mlir::createTritonGPUOptimizeDotOperandsPass() {
  return std::make_unique<TritonGPUOptimizeDotOperandsPass>();
}
