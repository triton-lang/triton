#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include <memory>

namespace mlir {
#define GEN_PASS_DEF_TRITONREORDERBROADCAST
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

Operation *cloneWithNewArgsAndResultTypes(PatternRewriter &rewriter,
                                          Operation *op, ValueRange newOperands,
                                          TypeRange newTypes) {
  OperationState newElementwiseState(op->getLoc(), op->getName());
  newElementwiseState.addOperands(newOperands);
  newElementwiseState.addTypes(newTypes);
  newElementwiseState.addAttributes(op->getAttrs());
  return rewriter.create(newElementwiseState);
}

bool isSplat(Operation *op) {
  if (auto splatOp = llvm::dyn_cast<triton::SplatOp>(op)) {
    return true;
  }
  DenseElementsAttr constAttr;
  return (matchPattern(op, m_Constant(&constAttr)) && constAttr.isSplat());
}

// elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))
struct MoveSplatAfterElementwisePattern
    : public mlir::OpTraitRewritePattern<mlir::OpTrait::Elementwise> {

  MoveSplatAfterElementwisePattern(mlir::MLIRContext *context)
      : OpTraitRewritePattern(context) {}

  mlir::LogicalResult match(Operation *op) const override {
    if (!isMemoryEffectFree(op)) {
      return mlir::failure();
    }

    for (auto operand : op->getOperands()) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp)
        return mlir::failure();

      if (!isSplat(definingOp)) {
        return mlir::failure();
      }
    }
    return mlir::success(op->getNumOperands() > 0);
  }

  void rewrite(Operation *op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    auto operands = op->getOperands();

    llvm::SmallVector<Value, 4> scalarOperands(operands.size());
    for (unsigned iOp = 0; iOp < operands.size(); ++iOp) {
      auto definingOp = operands[iOp].getDefiningOp();

      DenseElementsAttr constAttr;
      if (auto splatOp = llvm::dyn_cast<triton::SplatOp>(definingOp)) {
        scalarOperands[iOp] = splatOp.getSrc();
      } else if (matchPattern(definingOp, m_Constant(&constAttr)) &&
                 constAttr.isSplat()) {
        auto value = constAttr.getSplatValue<Attribute>();
        scalarOperands[iOp] = arith::ConstantOp::materialize(
            rewriter, value, constAttr.getElementType(), loc);
      } else {
        llvm_unreachable("Expected a splat");
      }
    }

    auto resultTypes = op->getResultTypes();
    llvm::SmallVector<Type, 4> scalarResultTys;
    for (auto resultTy : resultTypes) {
      auto elemTy = resultTy.dyn_cast<TensorType>().getElementType();
      scalarResultTys.push_back(elemTy);
    }

    auto newOp = cloneWithNewArgsAndResultTypes(rewriter, op, scalarOperands,
                                                scalarResultTys);

    for (unsigned iRes = 0; iRes < resultTypes.size(); ++iRes) {
      auto newResult = rewriter.create<triton::SplatOp>(loc, resultTypes[iRes],
                                                        newOp->getResult(iRes));
      rewriter.replaceAllUsesWith(op->getResult(iRes), newResult);
    }
  }
};

// elementwise(broadcast(a)) => broadcast(elementwise(a))
// This also generalizes to multiple arguments when the rest are splat-like
// Not handled: multiple broadcasted arguments
struct MoveBroadcastAfterElementwisePattern
    : public mlir::OpTraitRewritePattern<mlir::OpTrait::Elementwise> {

  MoveBroadcastAfterElementwisePattern(mlir::MLIRContext *context)
      : OpTraitRewritePattern(context) {}

  mlir::LogicalResult match(Operation *op) const override {
    if (!isMemoryEffectFree(op)) {
      return mlir::failure();
    }

    auto operands = op->getOperands();
    bool seenBroadcast = false;
    ArrayRef<int64_t> srcShape;
    for (auto operand : operands) {
      auto definingOp = operand.getDefiningOp();
      if (!definingOp) {
        return mlir::failure();
      }
      auto getSrcShape = [](triton::BroadcastOp b) {
        return b.getSrc().getType().getShape();
      };
      if (auto broadcastOp = llvm::dyn_cast<triton::BroadcastOp>(definingOp)) {
        if (!seenBroadcast) {
          seenBroadcast = true;
          srcShape = getSrcShape(broadcastOp);
        } else if (srcShape != getSrcShape(broadcastOp)) {
          // If the broadcast have different types we cannot re-order.
          return mlir::failure();
        }
      } else if (!isSplat(definingOp)) {
        // Not splat or broadcast
        return mlir::failure();
      }
    }
    return mlir::success(seenBroadcast);
  }

  void rewrite(Operation *op, mlir::PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // Find broadcast op
    auto operands = op->getOperands();
    triton::BroadcastOp broadcastOp;
    for (auto operand : operands) {
      broadcastOp = operand.getDefiningOp<triton::BroadcastOp>();
      if (broadcastOp) {
        break;
      }
    }

    auto srcTy = broadcastOp.getSrc().getType();
    auto srcShape = srcTy.getShape();
    auto srcEncoding = srcTy.getEncoding();

    // Reshape operands to match srcShape
    llvm::SmallVector<Value, 4> newOperands;
    for (auto operand : operands) {
      auto definingOp = operand.getDefiningOp();
      if (auto broadcastSrcOp =
              llvm::dyn_cast<triton::BroadcastOp>(definingOp)) {
        newOperands.push_back(broadcastSrcOp.getSrc());
        continue;
      }
      auto elemTy =
          operand.getType().dyn_cast<RankedTensorType>().getElementType();
      auto newTy = RankedTensorType::get(srcShape, elemTy, srcEncoding);
      if (auto splatOp = llvm::dyn_cast<triton::SplatOp>(definingOp)) {
        auto newSplat =
            rewriter.create<triton::SplatOp>(loc, newTy, splatOp.getSrc());
        newOperands.push_back(newSplat);
        continue;
      }
      DenseElementsAttr constAttr;
      if (matchPattern(definingOp, m_Constant(&constAttr)) &&
          constAttr.isSplat()) {
        auto scalarValue = constAttr.getSplatValue<Attribute>();
        auto splatValue = SplatElementsAttr::get(newTy, scalarValue);
        auto newConstant =
            rewriter.create<arith::ConstantOp>(loc, newTy, splatValue);
        newOperands.push_back(newConstant);
        continue;
      }
      llvm_unreachable("Expected broadcast or splat");
    }

    // Reshape results to match srcShape
    llvm::SmallVector<Type, 4> newResultTypes;
    auto resultTypes = op->getResultTypes();
    for (auto resultTy : resultTypes) {
      auto elemTy = resultTy.dyn_cast<RankedTensorType>().getElementType();
      newResultTypes.push_back(
          RankedTensorType::get(srcShape, elemTy, srcEncoding));
    }

    // Create new op and broadcast results
    auto newOp = cloneWithNewArgsAndResultTypes(rewriter, op, newOperands,
                                                newResultTypes);
    for (unsigned iRes = 0; iRes < newResultTypes.size(); ++iRes) {
      auto newResult = rewriter.create<triton::BroadcastOp>(
          loc, resultTypes[iRes], newOp->getResult(iRes));
      rewriter.replaceAllUsesWith(op->getResult(iRes), newResult);
    }
  }
};

template <typename OpType>
class CanonicalizePattern : public mlir::OpRewritePattern<OpType> {
public:
  explicit CanonicalizePattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<OpType>(context) {}

  mlir::LogicalResult
  matchAndRewrite(OpType op, mlir::PatternRewriter &rewriter) const override {
    return OpType::canonicalize(op, rewriter);
  }
};

class ReorderBroadcastPass
    : public mlir::impl::TritonReorderBroadcastBase<ReorderBroadcastPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    mlir::ModuleOp m = getOperation();

    patterns.add<CanonicalizePattern<triton::BroadcastOp>>(context);
    patterns.add<CanonicalizePattern<triton::ExpandDimsOp>>(context);
    // elementwise(broadcast(a)) => broadcast(elementwise(a))
    patterns.add<MoveBroadcastAfterElementwisePattern>(context);
    // elementwise(splat(a), splat(b), ...) => splat(elementwise(a, b, ...))
    patterns.add<MoveSplatAfterElementwisePattern>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::triton::createReorderBroadcastPass() {
  return std::make_unique<ReorderBroadcastPass>();
}
