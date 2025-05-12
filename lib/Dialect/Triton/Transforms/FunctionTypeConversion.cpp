#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdlib>

namespace mlir::triton {

namespace {

SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> ret;
  for (const auto &vs : values) {
    llvm::append_range(ret, vs);
  }
  return ret;
}

struct CallOpConversion : public OpConversionPattern<CallOp> {
  using OpConversionPattern<CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CallOp callOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<std::size_t> resultReplacementGrouping;
    llvm::SmallVector<Type> convertedResults;

    for (auto type : callOp->getResultTypes()) {
      const auto oldNumFlattenedResults = convertedResults.size();
      if (failed(getTypeConverter()->convertTypes(type, convertedResults))) {
        return failure();
      }
      resultReplacementGrouping.push_back(convertedResults.size() -
                                          oldNumFlattenedResults);
    }

    auto newCallOp = rewriter.create<CallOp>(
        callOp->getLoc(), callOp.getCallee(), convertedResults,
        flattenValues(adaptor.getOperands()));
    // Preserve any additional attributes that may have been set on the op
    newCallOp->setAttrs(callOp->getAttrs());

    SmallVector<ValueRange> replacements;
    std::size_t offset = 0;
    for (auto groupSize : resultReplacementGrouping) {
      replacements.push_back(newCallOp->getResults().slice(offset, groupSize));
      offset += groupSize;
    }

    rewriter.replaceOpWithMultiple(callOp, replacements);
    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp returnOp, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newReturnOp = rewriter.create<ReturnOp>(
        returnOp->getLoc(), flattenValues(adaptor.getOperands()));
    // Preserve any additional attributes that may have been set on the op
    newReturnOp->setAttrs(returnOp->getAttrs());

    rewriter.replaceOp(returnOp, newReturnOp);
    return success();
  }
};

} // namespace

void populateFunctionTypeConversions(const TypeConverter &converter,
                                     RewritePatternSet &patterns) {
  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::triton::FuncOp>(
      patterns, converter);
  patterns.add<CallOpConversion, ReturnOpConversion>(converter,
                                                     patterns.getContext());
}

} // namespace mlir::triton
