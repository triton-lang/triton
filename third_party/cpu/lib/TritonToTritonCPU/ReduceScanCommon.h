#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include <numeric>

namespace mlir {
namespace triton {
namespace cpu {

// Base class for converting scans and reductions.
//
// It provides accumulation function that clones operations from the
// original combine region and applies them on provided vectors.
// Also, it handles multi-diumensional cases reducing them to two
// possible options: lowering for a 1-D vector inputs and lowering
// the operation over the leading dimension.
//
// Specialized pattern should implement lower1DInput to handle
// trailing dimension case (commonly through shuffles + accumulate)
// and lowerLeadingDimension to handle the leading dimension case
// through accumulation of sub-vectors.
template <typename OpT, typename ReturnOpT>
struct ReduceScanOpConversionBase : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpConversionPattern<OpT>::getTypeConverter;
  using typename OpConversionPattern<OpT>::OpAdaptor;

  virtual SmallVector<Value>
  lower1DInput(ValueRange inputs, OpT op,
               ConversionPatternRewriter &rewriter) const = 0;
  virtual SmallVector<Value>
  lowerLeadingDimension(ValueRange inputs, OpT op,
                        ConversionPatternRewriter &rewriter) const = 0;

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rank = cast<RankedTensorType>(op.getOperand(0).getType()).getRank();
    if (op.getAxis() == (rank - 1))
      return lowerTrailingDimension(op, rewriter);

    return lowerNonTrailingDimension(op, rewriter);
  }

  // To handle the trailing dimension case, we extract all input vectors
  // and process them through lower1DInput, then build the resulting
  // vector using inserts.
  LogicalResult
  lowerTrailingDimension(OpT op, ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    SmallVector<Value> inputs;
    if (failed(rewriter.getRemappedValues(op.getOperands(), inputs)))
      return failure();

    SmallVector<VectorType> inputTys(inputs.size());
    std::transform(inputs.begin(), inputs.end(), inputTys.begin(),
                   [](auto val) { return cast<VectorType>(val.getType()); });

    // 1-D input case.
    if (inputTys.front().getRank() == 1) {
      auto res = lower1DInput(inputs, op, rewriter);
      rewriter.replaceOp(op, res);
      return success();
    }

    SmallVector<Value> res =
        makeEmptyResults(loc, op.getResultTypes(), rewriter);
    auto shape = inputTys[0].getShape();
    int64_t numElems = inputTys[0].getNumElements();
    auto strides = computeStrides(shape);
    // Remove the last stride to produce sub-vector indices.
    strides.pop_back();
    for (int64_t idx = 0; idx < numElems; idx += shape.back()) {
      auto indices = delinearize(idx, strides);
      SmallVector<Value> subInputs(inputs.size());
      std::transform(
          inputs.begin(), inputs.end(), subInputs.begin(), [&](auto val) {
            return rewriter.create<vector::ExtractOp>(loc, val, indices);
          });

      auto resElems = lower1DInput(subInputs, op, rewriter);
      for (size_t i = 0; i < res.size(); ++i) {
        res[i] = rewriter.create<vector::InsertOp>(loc, resElems[i], res[i],
                                                   indices);
      }
    }

    rewriter.replaceOp(op, res);
    return success();
  }

  // In this case we either call lowerLeadingDimension to process the input
  // or extract sub-vectors, call lowerLeadingDimension, and then reconstruct
  // the result.
  LogicalResult
  lowerNonTrailingDimension(OpT op, ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    SmallVector<Value> inputs;
    if (failed(rewriter.getRemappedValues(op.getOperands(), inputs)))
      return failure();

    uint32_t axis = op.getAxis();
    if (axis == 0) {
      rewriter.replaceOp(op, lowerLeadingDimension(inputs, op, rewriter));
      return success();
    }

    SmallVector<Value> res =
        makeEmptyResults(loc, op.getResultTypes(), rewriter);
    auto vecTy = cast<VectorType>(inputs[0].getType());
    auto shape = vecTy.getShape();
    auto strides = computeStrides(shape);
    // Remove trailing elems to build indices of required rank.
    strides.erase(strides.begin() + axis, strides.end());
    int64_t numElems = vecTy.getNumElements();
    int64_t step = strides.back();
    for (int64_t idx = 0; idx < numElems; idx += step) {
      auto indices = delinearize(idx, strides);
      SmallVector<Value> subInputs(inputs.size());
      std::transform(
          inputs.begin(), inputs.end(), subInputs.begin(), [&](auto val) {
            return rewriter.create<vector::ExtractOp>(loc, val, indices);
          });
      auto resVecs = lowerLeadingDimension(subInputs, op, rewriter);
      for (size_t i = 0; i < res.size(); ++i) {
        res[i] =
            rewriter.create<vector::InsertOp>(loc, resVecs[i], res[i], indices);
      }
    }

    rewriter.replaceOp(op, res);
    return success();
  }

  // Accumulate inputs and existing accumulators into a new accumaltors
  // applying operations from the combine region.
  SmallVector<Value> accumulate(ValueRange inputs, ValueRange acc,
                                Region &combineOp,
                                ConversionPatternRewriter &rewriter) const {
    if (acc.empty())
      return inputs;

    auto shape = cast<VectorType>(inputs[0].getType()).getShape();
    auto &block = combineOp.getBlocks().front();
    IRMapping map;
    // Map block arguments to the current inputs and accumulators.
    for (unsigned i = 0; i < acc.size(); ++i) {
      map.map(block.getArgument(i), acc[i]);
      map.map(block.getArgument(acc.size() + i), inputs[i]);
    }
    for (auto &op : block.getOperations()) {
      // Returned values are a new accumulator.
      if (isa<ReturnOpT>(op)) {
        SmallVector<Value> res;
        for (auto operand : op.getOperands()) {
          res.push_back(map.lookup(operand));
        }
        return res;
      }

      // Clone operation mapping its inputs and building vector
      // result types using the input shape.
      OperationState newState(op.getLoc(), op.getName());
      for (auto operand : op.getOperands()) {
        newState.operands.push_back(
            lookupMappedValue(map, operand, shape, rewriter));
      }
      for (auto ty : op.getResultTypes()) {
        newState.types.push_back(VectorType::get(shape, ty));
      }
      newState.attributes = op.getAttrs();
      auto newOp = rewriter.create(newState);

      // Add new values to the map.
      for (auto [oldVal, newVal] :
           llvm::zip(op.getResults(), newOp->getResults())) {
        map.map(oldVal, newVal);
      }
    }
    llvm_unreachable("No return op found in scan/reduce region");
  }

  Value lookupMappedValue(IRMapping &localMap, Value val,
                          ArrayRef<int64_t> shape,
                          ConversionPatternRewriter &rewriter) const {

    Value res = localMap.lookupOrNull(val);
    if (!res) {
      // If value is not found then it's an invariant defined in the outer
      // region. We check if it has been already translated and add a splat
      // operation if it hasn't.
      res = invariantsMap.lookupOrNull(val);
      if (!res) {
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointAfterValue(val);
        res = rewriter.create<vector::SplatOp>(
            val.getLoc(), VectorType::get(shape, val.getType()), val);
        invariantsMap.map(val, res);
        rewriter.restoreInsertionPoint(ip);
      }
    }
    return res;
  }

  SmallVector<Value>
  makeEmptyResults(Location loc, TypeRange resTypes,
                   ConversionPatternRewriter &rewriter) const {
    // Initialize results to zero values.
    SmallVector<Value> res;
    for (auto ty : resTypes) {
      res.push_back(rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(getTypeConverter()->convertType(ty))));
    }
    return res;
  }

  // Dummy vectors are required for shuffles that cannot work on a single
  // vector.
  ArrayRef<Value>
  createShuffleDummies(Location loc, ValueRange inputs,
                       ConversionPatternRewriter &rewriter) const {
    if (shuffleDummies.empty()) {
      for (auto val : inputs) {
        auto ty = cast<VectorType>(val.getType());
        shuffleDummies.push_back(rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(ty.cloneWith(1, ty.getElementType()))));
      }
    }
    return shuffleDummies;
  }

private:
  mutable IRMapping invariantsMap;
  mutable SmallVector<Value> shuffleDummies;
};

} // namespace cpu
} // namespace triton
} // namespace mlir
