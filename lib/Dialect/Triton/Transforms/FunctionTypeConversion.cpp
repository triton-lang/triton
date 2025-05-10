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
  matchAndRewrite(CallOp call_op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<std::size_t> result_replacement_grouping;
    llvm::SmallVector<Type> converted_results;

    for (auto type : call_op->getResultTypes()) {
      const auto old_num_flattened_results = converted_results.size();
      if (failed(getTypeConverter()->convertTypes(type, converted_results))) {
        return failure();
      }
      result_replacement_grouping.push_back(converted_results.size() -
                                            old_num_flattened_results);
    }

    auto new_call_op = rewriter.create<CallOp>(
        call_op->getLoc(), call_op.getCallee(), converted_results,
        flattenValues(adaptor.getOperands()));
    // Preserve any additional attributes that may have been set on the op
    new_call_op->setAttrs(call_op->getAttrs());

    SmallVector<ValueRange> replacements;
    std::size_t offset = 0;
    for (auto group_size : result_replacement_grouping) {
      replacements.push_back(
          new_call_op->getResults().slice(offset, group_size));
      offset += group_size;
    }

    rewriter.replaceOpWithMultiple(call_op, replacements);
    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp return_op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto new_return_op = rewriter.create<ReturnOp>(
        return_op->getLoc(), flattenValues(adaptor.getOperands()));
    // Preserve any additional attributes that may have been set on the op
    new_return_op->setAttrs(return_op->getAttrs());

    rewriter.replaceOp(return_op, new_return_op);
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
