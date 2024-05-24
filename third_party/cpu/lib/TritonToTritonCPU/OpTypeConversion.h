#include "mlir/IR/OperationSupport.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

// Generic pattern to rewrite operation by converting types
// for operation operands and results using provided type
// converter.
template <typename OpT, typename ResOpT = OpT>
struct OpTypeConversion : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpConversionPattern<OpT>::getTypeConverter;
  using typename OpConversionPattern<OpT>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OperationState newState(op.getLoc(), ResOpT::getOperationName());
    // Convert operands.
    for (auto operand : op->getOperands()) {
      Value newOperand = rewriter.getRemappedValue(operand);
      newState.operands.push_back(newOperand);
    }
    // Convert result types.
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newState.types))) {
      return failure();
    }
    newState.attributes = op->getAttrs();

    auto newOp = rewriter.create(newState);
    rewriter.replaceOp(op, newOp);

    return success();
  }
};
