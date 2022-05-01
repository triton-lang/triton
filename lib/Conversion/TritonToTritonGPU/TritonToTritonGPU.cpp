#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPU.h"
#include "../PassDetail.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

class ConvertArithmeticOp: public ConversionPattern {
public:
    ConvertArithmeticOp(TypeConverter &typeConverter, MLIRContext *context)
        : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                            context) {}

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter& rewriter) const override {
        Dialect* dialect = op->getDialect();
        if(dialect->getTypeID() != mlir::TypeID::get<arith::ArithmeticDialect>()) 
            return failure();
        // Arithmetic op to legalize here. Create layout conversion if necessary
        return success();
    }
};

void populateArithmeticPatternsAndLegality(
    TypeConverter& typeConverter, RewritePatternSet &patterns,
    ConversionTarget &target){
  // --------------
  // Add legality and rewrite pattern rules for operations
  // from the Arithmetic dialect. The basic premise is that
  // arithmetic operations require both inputs to have the same
  // non-null encoding
  // --------------
  MLIRContext *context = patterns.getContext();
  // Legality rule
  target.addDynamicallyLegalDialect<arith::ArithmeticDialect>(
      // TODO: check above rule here
      [](Operation *op){
          return false;
      }
  );
  // Rewrite rule
  patterns.add<ConvertArithmeticOp>(typeConverter, context);
}


class ConvertTritonToTritonGPU: 
    public ConvertTritonToTritonGPUBase<ConvertTritonToTritonGPU> {

public:
    void runOnOperation() override {
        MLIRContext *context = &getContext();
        ConversionTarget target(*context);
        // type converter
        TypeConverter typeConverter;
        // rewrite patterns
        RewritePatternSet patterns(context);
        // add rules
        populateArithmeticPatternsAndLegality(typeConverter, patterns, target);


        if(failed(applyPartialConversion(getOperation(), target, 
                                         std::move(patterns))))
            return signalPassFailure();
        
    }
};

}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToTritonGPUPass() {
  return std::make_unique<::ConvertTritonToTritonGPU>();
}
