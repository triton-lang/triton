#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include <memory>
#include <unordered_set>

using namespace mlir;

namespace {

bool isZero(mlir::Value val) {
  if (mlir::matchPattern(val, mlir::m_Zero()) ||
      mlir::matchPattern(val, mlir::m_AnyZeroFloat()))
    return true;
  // broadcast(constant_0)
  if (auto bc = val.getDefiningOp<mlir::triton::BroadcastOp>()) {
    if (mlir::matchPattern(bc.getSrc(), mlir::m_Zero()) ||
        mlir::matchPattern(bc.getSrc(), mlir::m_AnyZeroFloat()))
      return true;
  }
  return false;
}

bool isBroadcastConstantCombinable(Attribute value) {
  if (auto denseValue = value.dyn_cast<DenseElementsAttr>()) {
    return denseValue.isSplat();
  }
  return value.isa<FloatAttr, IntegerAttr>();
}

DenseElementsAttr getConstantValue(Builder &builder, Attribute value,
                                   Value bcast_res) {

  auto resType = bcast_res.getType().cast<ShapedType>();
  DenseElementsAttr res;
  if (auto denseValue = value.dyn_cast<DenseElementsAttr>()) {
    res =
        DenseElementsAttr::get(resType, denseValue.getSplatValue<Attribute>());
  } else {
    res = DenseElementsAttr::get(resType, value);
  }
  return res;
}

// TODO(csigg): remove after next LLVM integrate.
using FastMathFlags = arith::FastMathFlags;

#include "TritonCombine.inc"

} // anonymous namespace

// reduce()
// select(cond, load(ptrs, broadcast(cond), ???), other)
//   => load(ptrs, broadcast(cond), other)
class CombineSelectMaskedLoadPattern : public mlir::RewritePattern {
public:
  CombineSelectMaskedLoadPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::arith::SelectOp::getOperationName(), 3,
                             context, {triton::LoadOp::getOperationName()}) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto selectOp = llvm::dyn_cast<mlir::arith::SelectOp>(op);
    if (!selectOp)
      return mlir::failure();

    mlir::Value trueValue = selectOp.getTrueValue();
    mlir::Value falseValue = selectOp.getFalseValue();
    mlir::Value condSelect = selectOp.getCondition();

    auto *loadOpCandidate = trueValue.getDefiningOp();
    auto loadOp = llvm::dyn_cast_or_null<triton::LoadOp>(loadOpCandidate);
    if (!loadOp)
      return mlir::failure();

    mlir::Value mask = loadOp.getMask();
    if (!mask)
      return mlir::failure();

    auto *broadcastOpCandidate = mask.getDefiningOp();
    auto broadcastOp =
        llvm::dyn_cast_or_null<triton::BroadcastOp>(broadcastOpCandidate);
    if (!broadcastOp)
      return mlir::failure();

    auto broadcastCond = broadcastOp.getSrc();
    if (broadcastCond != condSelect)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, loadOp.getPtr(), loadOp.getMask(), falseValue,
        loadOp.getBoundaryCheck(), loadOp.getPadding(), loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile());
    return mlir::success();
  }
};

// expand_dims(dot(x, y), shape) + other
// -> expand_dim(dot(x, y) + squeeze_dims(other), shape)
class CombineExpandAddPattern : public mlir::RewritePattern {
private:
public:
  CombineExpandAddPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::arith::AddFOp::getOperationName(), 1,
                             context) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const {
    auto add = llvm::dyn_cast<mlir::arith::AddFOp>(op);
    if (!add)
      return mlir::failure();
    auto expand = add.getOperand(0).getDefiningOp<mlir::triton::ExpandDimsOp>();
    if (!expand)
      return mlir::failure();
    auto dot = expand.getOperand().getDefiningOp<mlir::triton::DotOp>();
    if (!dot)
      return mlir::failure();
    auto other = add.getOperand(1);
    auto newOther = rewriter.create<mlir::triton::SqueezeDimsOp>(
        add.getLoc(), other, expand.getAxis());
    auto newAdd = rewriter.create<mlir::arith::AddFOp>(
        add.getLoc(), dot.getResult(), newOther);
    rewriter.replaceOpWithNewOp<mlir::triton::ExpandDimsOp>(op, newAdd,
                                                            expand.getAxis());
    return mlir::success();
  }
};

// sum(x[:, :, None] * y[None, :, :], 1)
// -> dot(x, y)
class CombineBroadcastMulReducePattern : public mlir::RewritePattern {
private:
  static bool isAddF32(const Operation *op) {
    if (auto addf = dyn_cast_or_null<arith::AddFOp>(op))
      return addf.getType().isa<mlir::Float32Type>();
    return false;
  }

  static SmallVector<int> getEqualIndices(ArrayRef<int64_t> x,
                                          ArrayRef<int64_t> y) {
    SmallVector<int> res;
    for (int i = 0; i < x.size(); ++i)
      if (x[i] == y[i])
        res.push_back(i);
    return res;
  }

public:
  CombineBroadcastMulReducePattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::ReduceOp::getOperationName(), 1, context) {
  }

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const {
    auto reduceOp = llvm::dyn_cast<triton::ReduceOp>(op);
    if (!reduceOp)
      return mlir::failure();
    // only support reduce with simple addition
    Region &combineOp = reduceOp.getCombineOp();
    bool isReduceAdd = combineOp.hasOneBlock() &&
                       combineOp.front().getOperations().size() == 2 &&
                       isAddF32(&*combineOp.front().getOperations().begin());
    if (!isReduceAdd)
      return mlir::failure();
    // operand of reduce has to be mul
    auto mulOp = llvm::dyn_cast_or_null<arith::MulFOp>(
        reduceOp.getOperand(0).getDefiningOp());
    if (!mulOp)
      return mlir::failure();
    // mul operand has to be broadcast
    auto broadcastLhsOp = llvm::dyn_cast_or_null<triton::BroadcastOp>(
        mulOp.getOperand(0).getDefiningOp());
    if (!broadcastLhsOp)
      return mlir::failure();
    auto broadcastRhsOp = llvm::dyn_cast_or_null<triton::BroadcastOp>(
        mulOp.getOperand(1).getDefiningOp());
    if (!broadcastRhsOp)
      return mlir::failure();
    // get not-broadcast dimensions
    auto lhsInShape =
        broadcastLhsOp.getOperand().getType().cast<ShapedType>().getShape();
    auto lhsOutShape = broadcastLhsOp.getType().cast<ShapedType>().getShape();
    auto rhsInShape =
        broadcastRhsOp.getOperand().getType().cast<ShapedType>().getShape();
    auto rhsOutShape = broadcastRhsOp.getType().cast<ShapedType>().getShape();
    SmallVector<int> keptDimLHS = getEqualIndices(lhsInShape, lhsOutShape);
    SmallVector<int> keptDimRHS = getEqualIndices(rhsInShape, rhsOutShape);
    if (keptDimLHS.size() != 2 || keptDimRHS.size() != 2)
      return mlir::failure();
    if (keptDimLHS[1] != keptDimRHS[0] || reduceOp.getAxis() != keptDimLHS[1])
      return mlir::failure();
    // replace with dot
    Type newLhsType = RankedTensorType::get(
        {lhsInShape[keptDimLHS[0]], lhsOutShape[keptDimLHS[1]]},
        broadcastLhsOp.getOperand()
            .getType()
            .cast<ShapedType>()
            .getElementType());
    Type newRhsType = RankedTensorType::get(
        {rhsOutShape[keptDimRHS[0]], rhsInShape[keptDimRHS[1]]},
        broadcastRhsOp.getOperand()
            .getType()
            .cast<ShapedType>()
            .getElementType());
    Type newAccType = RankedTensorType::get(
        {lhsInShape[keptDimLHS[0]], rhsInShape[keptDimRHS[1]]},
        broadcastLhsOp.getOperand()
            .getType()
            .cast<ShapedType>()
            .getElementType());
    rewriter.setInsertionPoint(op);
    auto newLhs = rewriter.create<triton::SqueezeDimsOp>(
        op->getLoc(), broadcastLhsOp.getOperand(), 2);
    auto newRhs = rewriter.create<triton::SqueezeDimsOp>(
        op->getLoc(), broadcastRhsOp.getOperand(), 0);
    auto newAcc = rewriter.create<triton::SplatOp>(
        op->getLoc(), newAccType,
        rewriter.create<arith::ConstantOp>(op->getLoc(),
                                           rewriter.getF32FloatAttr(0)));
    rewriter.replaceOpWithNewOp<triton::DotOp>(op, newLhs, newRhs, newAcc,
                                               true);
    return mlir::success();
  }
};

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

class CombineOpsPass : public TritonCombineOpsBase<CombineOpsPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    mlir::ModuleOp m = getOperation();

    // Dot Add %{
    patterns.add<CombineDotAddIPattern>(context);
    patterns.add<CombineDotAddFPattern>(context);
    patterns.add<CombineDotAddIRevPattern>(context);
    patterns.add<CombineDotAddFRevPattern>(context);
    // %}
    patterns.add<CombineSelectMaskedLoadPattern>(context);
    // patterns.add<CombineAddPtrPattern>(context);
    patterns.add<CombineBroadcastConstantPattern>(context);
    patterns.add<CombineBroadcastMulReducePattern>(context);
    patterns.add<CombineExpandAddPattern>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> mlir::triton::createCombineOpsPass() {
  return std::make_unique<CombineOpsPass>();
}
