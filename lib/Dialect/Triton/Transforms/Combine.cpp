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
    if (mlir::matchPattern(bc.src(), mlir::m_Zero()) ||
        mlir::matchPattern(bc.src(), mlir::m_AnyZeroFloat()))
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

  Type resType = bcast_res.getType();
  DenseElementsAttr res;
  if (auto denseValue = value.dyn_cast<DenseElementsAttr>()) {
    res =
        DenseElementsAttr::get(resType, denseValue.getSplatValue<Attribute>());
  } else {
    res = DenseElementsAttr::get(resType, value);
  }
  return res;
}

#include "TritonCombine.inc"

} // anonymous namespace

// select(cond, load(ptrs, broadcast(cond), ???), other)
//   => load(ptrs, broadcast(cond), other)
class CombineSelectMaskedLoadPattern : public mlir::RewritePattern {
public:
  CombineSelectMaskedLoadPattern(mlir::MLIRContext *context)
      : mlir::RewritePattern(mlir::SelectOp::getOperationName(), 3, context,
                             {triton::LoadOp::getOperationName()}) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto selectOp = llvm::dyn_cast<mlir::SelectOp>(op);
    if (!selectOp)
      return mlir::failure();

    mlir::Value trueValue = selectOp.getTrueValue();
    mlir::Value falseValue = selectOp.getFalseValue();

    auto *loadOpCandidate = trueValue.getDefiningOp();
    auto loadOp = llvm::dyn_cast<triton::LoadOp>(loadOpCandidate);
    if (!loadOp)
      return mlir::failure();

    mlir::Value mask = loadOp.mask();
    if (!mask)
      return mlir::failure();

    auto *broadcastOpCandidate = mask.getDefiningOp();
    auto broadcastOp =
        llvm::dyn_cast<triton::BroadcastOp>(broadcastOpCandidate);
    if (!broadcastOp)
      return mlir::failure();

    rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, loadOp.ptr(), loadOp.mask(), falseValue, loadOp.cache(),
        loadOp.evict(), loadOp.isVolatile());
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
    patterns.add<CombineGEPPattern>(context);
    patterns.add<CombineBroadcastConstantPattern>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> mlir::triton::createCombineOpsPass() {
  return std::make_unique<CombineOpsPass>();
}
