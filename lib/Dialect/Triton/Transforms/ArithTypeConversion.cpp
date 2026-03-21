#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

struct RewriteArithSelectOp : mlir::OpConversionPattern<mlir::arith::SelectOp> {
  using mlir::OpConversionPattern<mlir::arith::SelectOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::arith::SelectOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // Note we're replacing the select op with an if op because we are
    // converting one value into many values.
    auto newIf = mlir::scf::IfOp::create(
        rewriter, op.getLoc(), mlir::TypeRange(adaptor.getTrueValue()),
        op.getCondition(), true);
    // We set the attributes from the op in case the op has any additional
    // attributes
    newIf->setAttrs(op->getAttrs());

    {
      mlir::ConversionPatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(newIf.thenBlock());
      mlir::scf::YieldOp::create(rewriter, op->getLoc(),
                                 adaptor.getTrueValue());
      rewriter.setInsertionPointToStart(newIf.elseBlock());
      mlir::scf::YieldOp::create(rewriter, op->getLoc(),
                                 adaptor.getFalseValue());
    }

    // Replace the old operation results
    rewriter.replaceOpWithMultiple(op, {newIf->getResults()});

    return mlir::success();
  }
};

} // namespace
namespace mlir::triton {

void populateArithTypeConversions(const TypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<RewriteArithSelectOp>(converter, patterns.getContext());
}

} // namespace mlir::triton
