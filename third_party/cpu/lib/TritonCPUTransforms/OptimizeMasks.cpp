#include "cpu/include/TritonCPUTransforms/OptCommon.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_OPTIMIZEMASKS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

int64_t getDivisibility(Value val) {
  BlockArgument blockArg = dyn_cast<BlockArgument>(val);
  if (!blockArg)
    return 1;

  Operation *argOp = blockArg.getOwner()->getParentOp();
  if (auto fn = dyn_cast<FunctionOpInterface>(argOp)) {
    Attribute attr = fn.getArgAttr(blockArg.getArgNumber(), "tt.divisibility");
    if (auto iattr = dyn_cast_or_null<IntegerAttr>(attr)) {
      return iattr.getInt();
    }
  }

  return 1;
}

bool isAlwaysDivisible(Value val, int64_t divisor) {
  if (auto cst = val.getDefiningOp<arith::ConstantOp>()) {
    auto intAttr = dyn_cast<IntegerAttr>(cst.getValue());
    return intAttr && (intAttr.getInt() % divisor == 0);
  }
  return getDivisibility(val) % divisor == 0;
}

bool isAlwaysDivisible(Value val, Value divisor) {
  if (auto cst = divisor.getDefiningOp<arith::ConstantOp>()) {
    auto intAttr = dyn_cast<IntegerAttr>(cst.getValue());
    if (intAttr)
      return isAlwaysDivisible(val, intAttr.getInt());
  }
  return false;
}

// Optimize cdiv pattern using divisibility hints. If value is known to be
// divisible by N then we can transform
//   (val + K - 1) / K
// to
//   val / K
// if N % K == 0 and val is not negative. Usually, we cannot prove value to be
// non-negative but still can apply transformation for contexts that assume
// positive value (e.g. as an upper bound in a for-loop with non-negative
// lower bound).
struct CdivToDiv : public OpRewritePattern<arith::DivSIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::DivSIOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    auto addOpDef = lhs.getDefiningOp<arith::AddIOp>();
    auto divisorDef = rhs.getDefiningOp<arith::ConstantOp>();
    if (!addOpDef || !divisorDef)
      return failure();

    arith::ConstantOp addCstDef;
    Value addOtherVal;
    if (addCstDef = addOpDef.getLhs().getDefiningOp<arith::ConstantOp>())
      addOtherVal = addOpDef.getRhs();
    else if (addCstDef = addOpDef.getRhs().getDefiningOp<arith::ConstantOp>())
      addOtherVal = addOpDef.getLhs();
    else
      return failure();

    int64_t divisorCst = cast<IntegerAttr>(divisorDef.getValue()).getInt();
    int64_t addCst = cast<IntegerAttr>(addCstDef.getValue()).getInt();
    if (divisorCst <= addCst)
      return failure();

    if (!isAlwaysDivisible(addOtherVal, divisorCst))
      return failure();

    Value res = op.getResult();
    Value newRes =
        rewriter.create<arith::DivSIOp>(loc, addOtherVal, divisorDef);
    int replaced = 0;
    rewriter.replaceUsesWithIf(res, newRes, [&](OpOperand &use) {
      if (auto forOp = dyn_cast<scf::ForOp>(use.getOwner())) {
        auto lowerDef =
            forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
        if (lowerDef && use.getOperandNumber() == 1 &&
            cast<IntegerAttr>(lowerDef.getValue()).getInt() >= 0) {
          ++replaced;
          return true;
        }
      }
      return false;
    });
    if (!replaced)
      return failure();

    return success();
  }
};

// This pattern rewrites for-loops used for tiling to optimize out division
// and multiplication using divisibility hints.
// Typical tiled loop looks like:
//   for i in range(0, tl.cdiv(size, TILE_SIZE)):
//     offs = i * TILE_SIZE
//     ...
// If size is known to be divisible by TILE_SIZE then it can be written as:
//   for offs in range(0, size, TILE_SIZE):
//     ...
// This pattern is used after an attempt to replace cdiv with a regular
// division. Possible input pattern is:
//   %c0 = arith.constant 0 : index
//   %c1 = arith.constant 1 : index
//   %c16 = arith.constant 16 : index
//   %init = arith.constant dense<0x00000000> : vector<16xf32>
//   %1 = arith.divsi %arg4, %c16
//   %2 = scf.for %arg5 = %c0 to %1 step %c1 iter_args(%arg6 = %init) ->
//   (vector<16xf32>) : i32 {
//     %3 = arith.muli %arg5, %c16 : i32
//     ...
//   }
// where %arg4 is known to be divisible by 16. The resulting code would be:
//   %c0 = arith.constant 0 : index
//   %c16 = arith.constant 16 : index
//   %init = arith.constant dense<0x00000000> : vector<16xf32>
//   %2 = scf.for %arg5 = %c0 to %arg4 step %c16 iter_args(%arg6 = %init) ->
//   (vector<16xf32>) : i32 {
//     ...
//   }
// This removes division and simplifies the following analysis to optimize
// masked memory acccess for the tile.
struct ScaleInductionVariable : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value iv = op.getInductionVar();
    Value lower = op.getLowerBound();
    Value upper = op.getUpperBound();
    Value step = op.getStep();
    auto lowerDef = lower.getDefiningOp<arith::ConstantOp>();
    auto upperDef = upper.getDefiningOp<arith::DivSIOp>();
    if (!lowerDef || !upperDef)
      return failure();

    int64_t lowerVal = cast<IntegerAttr>(lowerDef.getValue()).getInt();
    if (lowerVal < 0)
      return failure();

    // TODO: This is a strong requirement. With more generic value range
    // analysis we should be able to not rely on this transformation.
    if (!iv.hasOneUse())
      return failure();

    auto ivUse = dyn_cast<arith::MulIOp>(*iv.getUsers().begin());
    if (!ivUse)
      return failure();

    Value scale = ivUse.getLhs() == iv ? ivUse.getRhs() : ivUse.getLhs();
    auto scaleDef = scale.getDefiningOp<arith::ConstantOp>();
    auto divRhsDef = upperDef.getRhs().getDefiningOp<arith::ConstantOp>();
    auto divLhs = upperDef.getLhs();
    if (!scaleDef || !divRhsDef)
      return failure();

    int64_t scaleVal = cast<IntegerAttr>(scaleDef.getValue()).getInt();
    int64_t divisorVal = cast<IntegerAttr>(divRhsDef.getValue()).getInt();
    if (scaleVal != divisorVal || !isAlwaysDivisible(divLhs, scaleVal) ||
        lowerVal % scaleVal != 0)
      return failure();

    // Build new lower bound.
    Value newLower = lower;
    if (lowerVal != 0) {
      rewriter.setInsertionPointAfterValue(lower);
      newLower = rewriter.create<arith::ConstantIntOp>(
          lower.getLoc(), lowerVal * scaleVal, lower.getType());
    }
    // New Upper bound.
    Value newUpper = divLhs;
    // Build new step.
    rewriter.setInsertionPoint(op);
    auto newStep = rewriter.create<arith::MulIOp>(ivUse.getLoc(), step, scale);

    // Modify ForOp.
    rewriter.startOpModification(op);
    op.setLowerBound(newLower);
    op.setUpperBound(newUpper);
    op.setStep(newStep);
    rewriter.finalizeOpModification(op);

    // Replace iv uses.
    rewriter.replaceAllUsesWith(ivUse, iv);

    return success();
  }
};

// Build affine expression to express min/max value of the given SSA name.
// symbolTable is used to map SSA names to affine symbols.
AffineExpr buildMinOrMaxExpr(Value val, bool isSigned, bool isMax,
                             llvm::DenseMap<Value, unsigned> &symbolTable) {
  if (auto def = val.getDefiningOp<vector::SplatOp>()) {
    return buildMinOrMaxExpr(def.getInput(), isSigned, isMax, symbolTable);
  } else if (auto def = val.getDefiningOp<arith::ConstantOp>()) {
    auto attr = def.getValueAttr();
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return getAffineConstantExpr(intAttr.getInt(), val.getContext());
    if (auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(attr)) {
      auto valueBegin = denseAttr.value_begin<APInt>();
      auto valueEnd = denseAttr.value_end<APInt>();
      auto cmpVals = [isSigned](const APInt &lhs, const APInt &rhs) {
        return isSigned ? lhs.slt(rhs) : lhs.ult(rhs);
      };
      auto valueIt = isMax ? std::max_element(valueBegin, valueEnd, cmpVals)
                           : std::min_element(valueBegin, valueEnd, cmpVals);
      return getAffineConstantExpr((*valueIt).getSExtValue(), val.getContext());
    }
  } else if (auto def = val.getDefiningOp<arith::AddIOp>()) {
    return buildMinOrMaxExpr(def.getLhs(), isSigned, isMax, symbolTable) +
           buildMinOrMaxExpr(def.getRhs(), isSigned, isMax, symbolTable);
  } else if (auto def = val.getDefiningOp<arith::SubIOp>()) {
    return buildMinOrMaxExpr(def.getLhs(), isSigned, isMax, symbolTable) -
           buildMinOrMaxExpr(def.getRhs(), isSigned, !isMax, symbolTable);
  } else if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    auto op = blockArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (val == forOp.getInductionVar()) {
        Value lower = forOp.getLowerBound();
        Value upper = forOp.getUpperBound();
        Value step = forOp.getStep();

        // For min value return lower bound.
        if (!isMax)
          return buildMinOrMaxExpr(forOp.getLowerBound(), isSigned, isMax,
                                   symbolTable);

        // For max value we use upper bound - 1 in generic case and bound - step
        // if both bounds are divisible by the step.
        if (isAlwaysDivisible(lower, step) && isAlwaysDivisible(upper, step)) {
          return buildMinOrMaxExpr(upper, isSigned, isMax, symbolTable) -
                 buildMinOrMaxExpr(step, isSigned, false, symbolTable);
        }
        return buildMinOrMaxExpr(upper, isSigned, isMax, symbolTable) -
               getAffineConstantExpr(1, val.getContext());
      }
    }
  }

  if (symbolTable.count(val))
    return getAffineSymbolExpr(symbolTable.at(val), val.getContext());

  unsigned pos = symbolTable.size();
  symbolTable.insert(std::make_pair(val, pos));
  return getAffineSymbolExpr(pos, val.getContext());
}

// Check if vector mask is all-ones by checking compared values ranges.
// Only simplest cases are covered here, so affine expression is used
// to represent a range for now.
bool isAlwaysAllOnes(Value mask) {
  auto maskDef = mask.getDefiningOp<arith::CmpIOp>();
  if (!maskDef)
    return false;

  auto pred = maskDef.getPredicate();
  if (pred == arith::CmpIPredicate::eq || pred == arith::CmpIPredicate::ne)
    return false;

  bool isSigned =
      pred == arith::CmpIPredicate::sgt || pred == arith::CmpIPredicate::sge ||
      pred == arith::CmpIPredicate::sle || pred == arith::CmpIPredicate::slt;
  llvm::DenseMap<Value, unsigned> symbolTable;
  AffineExpr maxOffs;
  AffineExpr minLen;
  if (pred == arith::CmpIPredicate::slt || pred == arith::CmpIPredicate::sle ||
      pred == arith::CmpIPredicate::ult || pred == arith::CmpIPredicate::ule) {
    maxOffs = buildMinOrMaxExpr(maskDef.getLhs(), isSigned, true, symbolTable);
    minLen = buildMinOrMaxExpr(maskDef.getRhs(), isSigned, false, symbolTable);
  } else {
    maxOffs = buildMinOrMaxExpr(maskDef.getRhs(), isSigned, true, symbolTable);
    minLen = buildMinOrMaxExpr(maskDef.getLhs(), isSigned, false, symbolTable);
  }

  // The mask is all-ones if max offset is always less than min length.
  auto diff = maxOffs - minLen;
  if (auto diffCst = dyn_cast<AffineConstantExpr>(diff)) {
    int64_t diffVal = diffCst.getValue();
    if (pred == arith::CmpIPredicate::slt ||
        pred == arith::CmpIPredicate::ult ||
        pred == arith::CmpIPredicate::sgt || pred == arith::CmpIPredicate::ugt)
      return diffVal < 0;
    else
      return diffVal <= 0;
  }

  return false;
}

struct OptimizeMaskedLoad : public OpRewritePattern<vector::MaskedLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedLoadOp op,
                                PatternRewriter &rewriter) const override {
    if (!isAlwaysAllOnes(op.getMask()))
      return failure();

    rewriter.replaceOpWithNewOp<vector::LoadOp>(op, op.getType(), op.getBase(),
                                                op.getIndices());
    return success();
  }
};

struct OptimizeMaskedStore : public OpRewritePattern<vector::MaskedStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (!isAlwaysAllOnes(op.getMask()))
      return failure();

    rewriter.replaceOpWithNewOp<vector::StoreOp>(op, op.getValueToStore(),
                                                 op.getBase(), op.getIndices());
    return success();
  }
};

struct OptimizeMasks
    : public triton::cpu::impl::OptimizeMasksBase<OptimizeMasks> {
  OptimizeMasks() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // TODO: This pass optimizes out masks applying a set of very strict
    // patterns. We should use more generic range and divisibility analysis
    // to cover more cases and remove dependency on other transformations.
    RewritePatternSet patterns1(context);
    patterns1.add<CdivToDiv>(context);
    if (failed(mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns1))))
      return signalPassFailure();

    RewritePatternSet patterns2(context);
    patterns2.add<ScaleInductionVariable>(context);
    if (failed(mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns2))))
      return signalPassFailure();

    RewritePatternSet patterns3(context);
    patterns3.add<OptimizeMaskedLoad>(context);
    patterns3.add<OptimizeMaskedStore>(context);
    if (failed(mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns3))))
      return signalPassFailure();

    // TODO: if masks removal failed for loads/stores in a for-loop, we might
    // still optimize it using loop peeling.
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createOptimizeMasks() {
  return std::make_unique<OptimizeMasks>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
