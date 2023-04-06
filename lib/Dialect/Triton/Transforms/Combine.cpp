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

// TODO(csigg): remove after next LLVM integrate.
using FastMathFlags = arith::FastMathFlags;

#include "TritonCombine.inc"

} // anonymous namespace

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

// load(ptr, splat(1), ...)        -> load(ptr, ...)
// load(ptr, splat(0), other, ...) -> other
struct CanonicalizeMaskedLoadPattern
    : public mlir::OpRewritePattern<triton::LoadOp> {
  CanonicalizeMaskedLoadPattern(mlir::MLIRContext *context)
      : OpRewritePattern<triton::LoadOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(triton::LoadOp loadOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto mask = loadOp.getMask();
    if (!mask)
      return mlir::failure();

    auto constantMask =
        llvm::dyn_cast_or_null<arith::ConstantOp>(mask.getDefiningOp());
    if (!constantMask)
      return mlir::failure();

    auto splatMask = constantMask.getValue().dyn_cast<SplatElementsAttr>();
    if (!splatMask)
      return mlir::failure();

    if (splatMask.getSplatValue<IntegerAttr>().getValue() == true) {
      // mask = splat(1)
      rewriter.replaceOpWithNewOp<triton::LoadOp>(
          loadOp, loadOp.getType(), loadOp.getPtr(), Value(), Value(),
          loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    } else {
      // mask = splat(0)

      // If there's no "other", the value is "undef".  Perhaps we want to
      // optimize it in the future.x
      auto otherVal = loadOp.getOther();
      if (!otherVal)
        return mlir::failure();
      rewriter.replaceOp(loadOp, otherVal);
    }
    return mlir::success();
  }
};

void triton::LoadOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                 MLIRContext *context) {
  results.add<CanonicalizeMaskedLoadPattern>(context);
}

// store(ptr, value, splat(1), ...) -> store(ptr, value, ...)
// store(ptr, value, splat(0), ...) -> [none]
struct CanonicalizeMaskedStorePattern
    : public mlir::OpRewritePattern<triton::StoreOp> {
  CanonicalizeMaskedStorePattern(mlir::MLIRContext *context)
      : OpRewritePattern<triton::StoreOp>(context, 1) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp storeOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto mask = storeOp.getMask();
    if (!mask)
      return mlir::failure();

    auto constantMask =
        llvm::dyn_cast_or_null<arith::ConstantOp>(mask.getDefiningOp());
    if (!constantMask)
      return mlir::failure();

    auto splatMask = constantMask.getValue().dyn_cast<SplatElementsAttr>();
    if (!splatMask)
      return mlir::failure();

    if (splatMask.getSplatValue<IntegerAttr>().getValue() == true) {
      // mask = splat(1)
      rewriter.replaceOpWithNewOp<triton::StoreOp>(
          storeOp, storeOp.getPtr(), storeOp.getValue(), storeOp.getCache(),
          storeOp.getEvict());
    } else {
      // mask = splat(0)
      rewriter.eraseOp(storeOp);
    }
    return mlir::success();
  }
};

void triton::StoreOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<CanonicalizeMaskedStorePattern>(context);
}

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

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> mlir::triton::createCombineOpsPass() {
  return std::make_unique<CombineOpsPass>();
}
