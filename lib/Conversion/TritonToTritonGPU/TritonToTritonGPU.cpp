#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPU.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "../PassDetail.h"
#include <llvm-6.0/llvm/Support/ErrorHandling.h>

using namespace mlir;
using namespace mlir::triton;

namespace {

template<class Op>
class ArithGenericPattern : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type retType = this->getTypeConverter()->convertType(op.getType());
    Op res = rewriter.replaceOpWithNewOp<Op>(
      op, retType, adaptor.getOperands()
    );
    return success();
  }
};

template<class SrcOp, class DstOp>
class ArithCmpPattern : public OpConversionPattern<SrcOp> {
public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SrcOp op, typename SrcOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type retType = this->getTypeConverter()->convertType(op.getType());
    DstOp res = rewriter.replaceOpWithNewOp<DstOp>(
      op, retType, adaptor.getPredicate(), adaptor.getLhs(), adaptor.getRhs()
    );
    return success();
  }
};

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto value = adaptor.getValue().dyn_cast<DenseElementsAttr>();
    assert(value);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
      op, retType, value.reshape(retType) // This is a hack. We just want to add encoding
    );
    return success();
  }
};

class ConvertArithmeticOp: public ConversionPattern {
public:
    ConvertArithmeticOp(TritonGPUTypeConverter &typeConverter, MLIRContext *context)
        : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                            context) {}

    LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter& rewriter) const override {
        Dialect* dialect = op->getDialect();
        if(dialect->getTypeID() != mlir::TypeID::get<arith::ArithmeticDialect>()) 
            return failure();
        return success();
    }
};

void populateArithmeticPatternsAndLegality(
    TritonGPUTypeConverter& typeConverter, RewritePatternSet &patterns,
    TritonGPUConversionTarget &target){
  // --------------
  // Add legality and rewrite pattern rules for operations
  // from the Arithmetic dialect. The basic premise is that
  // arithmetic operations require both inputs to have the same
  // non-null encoding
  // --------------
  MLIRContext *context = patterns.getContext();
  // // Legality rule
  // target.addDynamicallyLegalDialect<arith::ArithmeticDialect>(
  //     // TODO: check above rule here
  //     [](Operation *op){
  //         return true;
  //     }
  // );
  // Rewrite rule
  // patterns.add<ConvertArithmeticOp>(typeConverter, context);
  patterns.add<ArithConstantPattern,
               ArithGenericPattern<arith::AddIOp>,
               ArithGenericPattern<arith::SubIOp>,
               ArithGenericPattern<arith::MulIOp>,
               ArithGenericPattern<arith::DivUIOp>,
               ArithGenericPattern<arith::DivSIOp>,
               ArithGenericPattern<arith::CeilDivUIOp>,
               ArithGenericPattern<arith::CeilDivSIOp>,
               ArithGenericPattern<arith::FloorDivSIOp>,
               ArithGenericPattern<arith::RemUIOp>,
               ArithGenericPattern<arith::RemSIOp>,
               ArithGenericPattern<arith::AndIOp>,
               ArithGenericPattern<arith::OrIOp>,
               ArithGenericPattern<arith::XOrIOp>,
               ArithGenericPattern<arith::ShLIOp>,
               ArithGenericPattern<arith::ShRUIOp>,
               ArithGenericPattern<arith::ShRSIOp>, // NegFOp
               // Floating point
               ArithGenericPattern<arith::AddFOp>,
               ArithGenericPattern<arith::SubFOp>,
               // MaxMin
               ArithGenericPattern<arith::MaxFOp>,
               ArithGenericPattern<arith::MaxSIOp>,
               ArithGenericPattern<arith::MaxUIOp>,
               ArithGenericPattern<arith::MinFOp>,
               ArithGenericPattern<arith::MinSIOp>,
               ArithGenericPattern<arith::MinUIOp>,
               // Floating point
               ArithGenericPattern<arith::MulFOp>,
               ArithGenericPattern<arith::DivFOp>,
               ArithGenericPattern<arith::RemFOp>,
               // Cmp
               ArithCmpPattern<arith::CmpIOp, triton::gpu::CmpIOp>,
               ArithCmpPattern<arith::CmpFOp, triton::gpu::CmpFOp>,
               // Cast Ops
               ArithGenericPattern<arith::TruncIOp>,
               ArithGenericPattern<arith::TruncFOp>
              >(typeConverter, context);
}

//
// Triton patterns
//
// TODO: Do we need to put them in anonymous namespace?
struct TritonMakeRangePattern : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<triton::MakeRangeOp>(
      op, retType, adaptor.start(), adaptor.end()
    );
    return success();
  }
};

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<triton::DotOp>(
      op, retType, adaptor.a(), adaptor.b(), adaptor.c(), adaptor.allowTF32()
    );
    return success();
  }
};

struct TritonLoadPattern : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<triton::LoadOp>(
      op, retType,
      adaptor.ptr(), adaptor.mask(), adaptor.other(),
      adaptor.cache(), adaptor.evict(), adaptor.isVolatile()
    );
    return success();
  }
};

struct TritonStorePattern : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<triton::StoreOp>(
      op, adaptor.ptr(), adaptor.value(), adaptor.mask()
    );
    return success();
  }
};

template <class Op>
struct TritonGenericPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type retType = this->getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<Op>(
      op, retType, adaptor.getOperands()
    );
    return success();
  }
};

void populateTritonPatterns(
  TritonGPUTypeConverter& typeConverter, RewritePatternSet &patterns
) {
  MLIRContext *context = patterns.getContext();
  patterns.add<TritonGenericPattern<triton::ReshapeOp>,
               TritonGenericPattern<triton::BroadcastOp>,
               TritonGenericPattern<triton::GEPOp>,
               TritonMakeRangePattern,
               TritonDotPattern,
               TritonLoadPattern,
               TritonStorePattern
              >(typeConverter, context);
}

//
// SCF patterns
//
struct SCFForPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto newFor = rewriter.replaceOpWithNewOp<scf::ForOp>(
      op, adaptor.getLowerBound(), adaptor.getUpperBound(), adaptor.getStep(),
      adaptor.getInitArgs()
    );
    // TODO: we need to copy (?) the body of ForOp
    llvm_unreachable("Not implemented");
    // newFor.getRegion().takeBody(adaptor.getRegion());
    return success();
  }
};

struct SCFYieldPattern : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(
      op, adaptor.getResults()
    );
    return success();
  }
};

void populateSCFPatterns(
  TritonGPUTypeConverter &typeConverter, RewritePatternSet &patterns
) {
  MLIRContext *context = patterns.getContext();
  patterns.add<SCFForPattern,
               SCFYieldPattern
              >(typeConverter, context);
}


class ConvertTritonToTritonGPU :
    public ConvertTritonToTritonGPUBase<ConvertTritonToTritonGPU> {

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // int numThreads = mod.getAttr();
    // type converter
    TritonGPUTypeConverter typeConverter(context, /*numThreads*/32);
    TritonGPUConversionTarget target(*context, typeConverter);
    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    populateArithmeticPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns);
    populateSCFPatterns(typeConverter, patterns);

    if(failed(applyPartialConversion(mod, target, 
                                      std::move(patterns))))
        return signalPassFailure();
  }
};

}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToTritonGPUPass() {
  return std::make_unique<::ConvertTritonToTritonGPU>();
}
