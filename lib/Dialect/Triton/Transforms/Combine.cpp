#include <memory>

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace mlir::triton {
namespace {

bool isZero(Value val) {
  if (matchPattern(val, m_Zero()) || matchPattern(val, m_AnyZeroFloat()))
    return true;
  // broadcast(constant_0)
  if (auto bc = val.getDefiningOp<BroadcastOp>()) {
    if (matchPattern(bc.getSrc(), m_Zero()) ||
        matchPattern(bc.getSrc(), m_AnyZeroFloat()))
      return true;
  }
  return false;
}

bool isBroadcastConstantCombinable(Attribute value) {
  if (auto denseValue = dyn_cast<DenseElementsAttr>(value)) {
    return denseValue.isSplat();
  }
  return isa<FloatAttr, IntegerAttr>(value);
}

DenseElementsAttr getConstantValue(Builder &builder, Attribute value,
                                   Value bcast_res) {
  auto resType = cast<ShapedType>(bcast_res.getType());
  DenseElementsAttr res;
  if (auto denseValue = dyn_cast<DenseElementsAttr>(value)) {
    res =
        DenseElementsAttr::get(resType, denseValue.getSplatValue<Attribute>());
  } else {
    res = DenseElementsAttr::get(resType, value);
  }
  return res;
}

bool isAddPtrOffsetCombinable(Value first, Value second) {
  auto GetConstantIntValue = [](Value val) -> std::optional<llvm::APInt> {
    DenseElementsAttr constAttr;
    auto defOp = val.getDefiningOp();
    if (defOp) {
      if (auto splatOp = llvm::dyn_cast<SplatOp>(defOp))
        val = splatOp.getSrc();
      else if (matchPattern(defOp, m_Constant(&constAttr)) &&
               constAttr.isSplat()) {
        auto attr = constAttr.getSplatValue<Attribute>();
        // Check IntegerAttr
        if (auto intAttr = dyn_cast_or_null<IntegerAttr>(attr))
          return intAttr.getValue();
      }
    }

    // Check constant value.
    llvm::APInt intVal;
    if (matchPattern(val, m_ConstantInt(&intVal)))
      return intVal;

    return std::nullopt;
  };

  if (first.getType() == second.getType()) {
    // Whether bitwidth of element type is equal to pointer
    if (getElementTypeOrSelf(first.getType()).getIntOrFloatBitWidth() == 64)
      return true;

    // first + second does not overflow
    auto firstVal = GetConstantIntValue(first);
    auto secondVal = GetConstantIntValue(second);
    if (firstVal && secondVal) {
      bool overflow = false;
      auto resVal = firstVal->sadd_ov(*secondVal, overflow);
      return !overflow;
    }
  }
  return false;
}

// TODO(csigg): remove after next LLVM integrate.
using FastMathFlags = arith::FastMathFlags;

#include "TritonCombine.inc"

/// select(cond, load(ptrs, mask, ???), other)
///   => load(ptrs, mask, other)
/// mask: a dense value related to cond
/// - mask = cond
/// - mask = splat(cond)
/// - cond = extract(mask) &&  mask = denseVal
/// - cond = extract(mask) &&  mask = splat(boolVal)
/// - cond = extract(mask) &&  mask = broadcast(denseVal)
class CombineSelectMaskedLoadPattern : public RewritePattern {
public:
  CombineSelectMaskedLoadPattern(MLIRContext *context)
      : RewritePattern(arith::SelectOp::getOperationName(), 3, context,
                       {LoadOp::getOperationName()}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto selectOp = llvm::dyn_cast<arith::SelectOp>(op);
    if (!selectOp)
      return failure();

    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();
    Value condSelect = selectOp.getCondition();

    auto loadOp = trueValue.getDefiningOp<LoadOp>();
    if (!loadOp)
      return failure();

    Value mask = loadOp.getMask();
    if (!mask)
      return failure();

    bool findTargetPattern = false;
    if (mask == condSelect) {
      // Case 1: The mask of the load and the cond of the select are the same
      // value.
      // ```mlir
      //   %mask : i1
      //   %true_val = tt.load %ptr, %mask, %other : !tt.ptr<f32>
      //   %res = arith.select %mask, %true_val, %false_val : f32
      // ```
      findTargetPattern = true;
    } else if (auto splatOp = mask.getDefiningOp<SplatOp>()) {
      // Case 2: The mask of the load is splatted from the cond of the select.
      // ```mlir
      //   %mask = tt.splat %cond : i1 -> tensor<8xi1>
      //   %true_val = tt.load %ptr, %mask : tensor<8x!tt.ptr<f32>>
      //   %res = arith.select %cond, %true_val, %false_val : tensor<8xf32>
      // ```
      if (splatOp.getSrc() == condSelect) {
        findTargetPattern = true;
      }
    }

    // Case 3: The condition of the select is a value extracted from a dense
    // tensor (the mask value).
    // ```mlir
    //   %mask : dense tensor(splatted from same value)
    //   %true_val = tt.load %ptr, %mask
    //   %cond = tensor.extract %mask[...]
    //   %res = arith.select %cond, %true_val, %false_val
    // ```
    if (auto extractOp = condSelect.getDefiningOp<tensor::ExtractOp>()) {
      auto tensor = extractOp.getTensor();
      if (tensor == mask) {
        if (llvm::all_of(
                llvm::cast<RankedTensorType>(tensor.getType()).getShape(),
                [](int64_t size) { return size == int64_t(1); })) {
          // The mask shape is a 0-rank tensor or a tensor with all unit
          // dimensions.
          // ```mlir
          //   %mask : tensor<1x1xi1>
          //   %cond = tensor.extract %mask[%c0, %c0] : tensor<1x1xi1>
          // ```
          findTargetPattern = true;
        } else {
          auto defineOp = tensor.getDefiningOp();
          if (llvm::isa_and_present<triton::SplatOp>(defineOp)) {
            // The mask value is splatted from a bool value.
            // ```mlir
            //   %mask = tt.splat %bool : i1 -> tensor<8x8xi1>
            //   %cond = tensor.extract %mask[%c0, %c0] : tensor<8x8xi1>
            // ```
            findTargetPattern = true;
          } else if (auto broadcastOp =
                         llvm::dyn_cast_if_present<triton::BroadcastOp>(
                             defineOp)) {
            // The mask value is broadcasted from an all unit-dimension tensor.
            // ```mlir
            //   %mask = tt.broadcast %bool : tensor<1x1xi1> -> tensor<8x8xi1>
            //   %cond = tensor.extract %mask[%c0, %c0] : tensor<8x8xi1>
            // ```
            if (llvm::all_of(
                    llvm::cast<RankedTensorType>(broadcastOp.getSrc().getType())
                        .getShape(),
                    [](int64_t size) { return size == int64_t(1); })) {
              findTargetPattern = true;
            }
          }
        }
      }
    }

    if (!findTargetPattern) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<LoadOp>(
        op, loadOp.getPtr(), loadOp.getMask(), /*other=*/falseValue,
        loadOp.getBoundaryCheckAttr(), loadOp.getPaddingAttr(),
        loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
    return success();
  }
};

// sum(x[:, :, None] * y[None, :, :], 1)
// -> dot(x, y)
class CombineBroadcastMulReducePattern : public RewritePattern {
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
  CombineBroadcastMulReducePattern(MLIRContext *context)
      : RewritePattern(ReduceOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const {
    auto reduceOp = llvm::dyn_cast<ReduceOp>(op);
    if (!reduceOp)
      return failure();
    // only support reduce with simple addition
    Region &combineOp = reduceOp.getCombineOp();
    bool isReduceAdd = combineOp.hasOneBlock() &&
                       combineOp.front().getOperations().size() == 2 &&
                       isAddF32(&*combineOp.front().getOperations().begin());
    if (!isReduceAdd)
      return failure();
    // operand of reduce has to be mul
    auto mulOp = reduceOp.getOperand(0).getDefiningOp<arith::MulFOp>();
    if (!mulOp)
      return failure();
    // mul operand has to be broadcast
    auto broadcastLhsOp = mulOp.getOperand(0).getDefiningOp<BroadcastOp>();
    if (!broadcastLhsOp)
      return failure();
    auto broadcastRhsOp = mulOp.getOperand(1).getDefiningOp<BroadcastOp>();
    if (!broadcastRhsOp)
      return failure();
    // broadcast operand is expand dims
    auto expandLhsOp = broadcastLhsOp.getSrc().getDefiningOp<ExpandDimsOp>();
    if (!expandLhsOp)
      return failure();
    auto expandRhsOp = broadcastRhsOp.getSrc().getDefiningOp<ExpandDimsOp>();
    if (!expandRhsOp)
      return failure();
    // get not-broadcast dimensions
    int expandLhsAxis = expandLhsOp.getAxis();
    int expandRhsAxis = expandRhsOp.getAxis();
    if (expandLhsAxis != 2 || expandRhsAxis != 0)
      return failure();
    auto broadcastLhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    auto broadcastRhsShape =
        cast<ShapedType>(broadcastLhsOp.getType()).getShape();
    if (broadcastLhsShape[2] < 16 || broadcastRhsShape[0] < 16)
      return failure();
    Type newAccType = RankedTensorType::get(
        {broadcastLhsShape[0], broadcastRhsShape[2]},
        cast<ShapedType>(broadcastLhsOp.getSrc().getType()).getElementType());
    rewriter.setInsertionPoint(op);
    auto newAcc = rewriter.create<SplatOp>(
        op->getLoc(), newAccType,
        rewriter.create<arith::ConstantOp>(op->getLoc(),
                                           rewriter.getF32FloatAttr(0)));
    rewriter.replaceOpWithNewOp<DotOp>(op, expandLhsOp.getSrc(),
                                       expandRhsOp.getSrc(), newAcc,
                                       InputPrecision::TF32, 0);
    return success();
  }
};

class CombineOpsPass : public TritonCombineOpsBase<CombineOpsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp m = getOperation();

    // Dot Add %{
    patterns.add<CombineDotAddIPattern>(context);
    patterns.add<CombineDotAddFPattern>(context);
    patterns.add<CombineDotAddIRevPattern>(context);
    patterns.add<CombineDotAddFRevPattern>(context);
    // %}
    patterns.add<CombineSelectMaskedLoadPattern>(context);
    patterns.add<CombineAddPtrPattern>(context);
    patterns.add<CombineBroadcastConstantPattern>(context);
    patterns.add<CombineBroadcastMulReducePattern>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> createCombineOpsPass() {
  return std::make_unique<CombineOpsPass>();
}

} // namespace mlir::triton
