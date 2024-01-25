#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include <memory>

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

// select(cond, load(ptrs, splat(cond), ???), other)
//   => load(ptrs, splat(cond), other)
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

    auto *splatOpCandidate = mask.getDefiningOp();
    auto splatOp = llvm::dyn_cast_or_null<triton::SplatOp>(splatOpCandidate);
    if (!splatOp)
      return mlir::failure();

    auto splatCond = splatOp.getSrc();
    if (splatCond != condSelect)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, loadOp.getPtr(), loadOp.getMask(), falseValue,
        loadOp.getBoundaryCheck(), loadOp.getPadding(), loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile());
    return mlir::success();
  }
};

// sum(x[:, :, None] * y[None, :, :], 1)
// -> dot(x, y)
class CombineBroadcastMulReducePattern : public mlir::RewritePattern {
private:
  static bool isAddF32(const Operation *op) {
    if (auto addf = dyn_cast_or_null<arith::AddFOp>(op))
      return addf.getType().getIntOrFloatBitWidth() <= 32;
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
    // broadcast operand is expand dims
    auto expandLhsOp = llvm::dyn_cast_or_null<triton::ExpandDimsOp>(
        broadcastLhsOp.getOperand().getDefiningOp());
    if (!expandLhsOp)
      return mlir::failure();
    auto expandRhsOp = llvm::dyn_cast_or_null<triton::ExpandDimsOp>(
        broadcastRhsOp.getOperand().getDefiningOp());
    if (!expandRhsOp)
      return mlir::failure();
    // get not-broadcast dimensions
    int expandLhsAxis = expandLhsOp.getAxis();
    int expandRhsAxis = expandRhsOp.getAxis();
    if (expandLhsAxis != 2 || expandRhsAxis != 0)
      return mlir::failure();
    auto broadcastLhsShape =
        broadcastLhsOp.getType().cast<ShapedType>().getShape();
    auto broadcastRhsShape =
        broadcastLhsOp.getType().cast<ShapedType>().getShape();
    if (broadcastLhsShape[2] < 16 || broadcastRhsShape[0] < 16)
      return mlir::failure();
    Type newAccType =
        RankedTensorType::get({broadcastLhsShape[0], broadcastRhsShape[2]},
                              broadcastLhsOp.getOperand()
                                  .getType()
                                  .cast<ShapedType>()
                                  .getElementType());
    rewriter.setInsertionPoint(op);
    auto newAcc = rewriter.create<triton::SplatOp>(
        op->getLoc(), newAccType,
        rewriter.create<arith::ConstantOp>(op->getLoc(),
                                           rewriter.getF32FloatAttr(0)));
    rewriter.replaceOpWithNewOp<triton::DotOp>(op, expandLhsOp.getOperand(),
                                               expandRhsOp.getOperand(), newAcc,
                                               true, 0);
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

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> mlir::triton::createCombineOpsPass() {
  return std::make_unique<CombineOpsPass>();
}
