#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPU.h"
#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include <numeric>

using namespace mlir;
using namespace mlir::triton;

namespace {

template <class Op> class ArithGenericPattern : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = this->getTypeConverter()->convertType(op.getType());
    Op res =
        rewriter.replaceOpWithNewOp<Op>(op, retType, adaptor.getOperands());
    return success();
  }
};

template <class SrcOp, class DstOp>
class ArithCmpPattern : public OpConversionPattern<SrcOp> {
public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SrcOp op, typename SrcOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = this->getTypeConverter()->convertType(op.getType());
    DstOp res =
        rewriter.replaceOpWithNewOp<DstOp>(op, retType, adaptor.getPredicate(),
                                           adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto value = adaptor.getValue().dyn_cast<DenseElementsAttr>();
    assert(value);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        op, retType,
        value.reshape(retType) // This is a hack. We just want to add encoding
    );
    return success();
  }
};

class ConvertArithmeticOp : public ConversionPattern {
public:
  ConvertArithmeticOp(TritonGPUTypeConverter &typeConverter,
                      MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Dialect *dialect = op->getDialect();
    if (dialect->getTypeID() != mlir::TypeID::get<arith::ArithmeticDialect>())
      return failure();
    return success();
  }
};

void populateArithmeticPatternsAndLegality(
    TritonGPUTypeConverter &typeConverter, RewritePatternSet &patterns,
    TritonGPUConversionTarget &target) {
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
  patterns.add<
      ArithConstantPattern, ArithGenericPattern<arith::AddIOp>,
      ArithGenericPattern<arith::SubIOp>, ArithGenericPattern<arith::MulIOp>,
      ArithGenericPattern<arith::DivUIOp>, ArithGenericPattern<arith::DivSIOp>,
      ArithGenericPattern<arith::CeilDivUIOp>,
      ArithGenericPattern<arith::CeilDivSIOp>,
      ArithGenericPattern<arith::FloorDivSIOp>,
      ArithGenericPattern<arith::RemUIOp>, ArithGenericPattern<arith::RemSIOp>,
      ArithGenericPattern<arith::AndIOp>, ArithGenericPattern<arith::OrIOp>,
      ArithGenericPattern<arith::XOrIOp>, ArithGenericPattern<arith::ShLIOp>,
      ArithGenericPattern<arith::ShRUIOp>,
      ArithGenericPattern<arith::ShRSIOp>, // NegFOp
      // Floating point
      ArithGenericPattern<arith::AddFOp>, ArithGenericPattern<arith::SubFOp>,
      // MaxMin
      ArithGenericPattern<arith::MaxFOp>, ArithGenericPattern<arith::MaxSIOp>,
      ArithGenericPattern<arith::MaxUIOp>, ArithGenericPattern<arith::MinFOp>,
      ArithGenericPattern<arith::MinSIOp>, ArithGenericPattern<arith::MinUIOp>,
      // Floating point
      ArithGenericPattern<arith::MulFOp>, ArithGenericPattern<arith::DivFOp>,
      ArithGenericPattern<arith::RemFOp>,
      // Cmp
      ArithCmpPattern<arith::CmpIOp, triton::gpu::CmpIOp>,
      ArithCmpPattern<arith::CmpFOp, triton::gpu::CmpFOp>,
      // Cast Ops
      ArithGenericPattern<arith::TruncIOp>,
      ArithGenericPattern<arith::TruncFOp>>(typeConverter, context);
}

//
// Triton patterns
//
// TODO: Do we need to put them in anonymous namespace?
struct TritonMakeRangePattern
    : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<triton::MakeRangeOp>(
        op, retType, adaptor.start(), adaptor.end());
    return success();
  }
};

struct TritonExpandDimsPattern
    : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Type retType = op.getType());
    RankedTensorType argType = adaptor.src().getType().cast<RankedTensorType>();
    Attribute _argEncoding = argType.getEncoding();
    if (!_argEncoding)
      return failure();
    auto argEncoding =
        _argEncoding.cast<triton::gpu::TritonGPUBlockedEncodingAttr>();
    // return shape
    auto retShape = argType.getShape().vec();
    retShape.insert(retShape.begin() + op.axis(), 1);
    // return encoding
    auto retSizePerThread = argEncoding.getSizePerThread().vec();
    retSizePerThread.insert(retSizePerThread.begin() + op.axis(), 1);
    auto retThreadsPerWarp = argEncoding.getThreadsPerWarp().vec();
    retThreadsPerWarp.insert(retThreadsPerWarp.begin() + op.axis(), 1);
    auto retWarpsPerCTA = argEncoding.getWarpsPerCTA().vec();
    retWarpsPerCTA.insert(retWarpsPerCTA.begin() + op.axis(), 1);
    SmallVector<unsigned, 4> retOrder(retShape.size());
    std::iota(retOrder.begin(), retOrder.end(), 0);
    triton::gpu::TritonGPUBlockedEncodingAttr retEncoding =
        triton::gpu::TritonGPUBlockedEncodingAttr::get(
            getContext(), retSizePerThread, retThreadsPerWarp, retWarpsPerCTA,
            retOrder);
    // return type
    RankedTensorType retType =
        RankedTensorType::get(retShape, argType.getElementType(), retEncoding);
    // construct new op
    rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(
        op, retType, adaptor.src(), adaptor.axis());
    return success();
  }
};

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    // a & b must be of smem layout
    auto aType = adaptor.a().getType().cast<RankedTensorType>();
    auto bType = adaptor.b().getType().cast<RankedTensorType>();
    Attribute aEncoding = aType.getEncoding();
    Attribute bEncoding = bType.getEncoding();
    if (!aEncoding || !bEncoding)
      return failure();
    Value a = adaptor.a();
    Value b = adaptor.b();
    SmallVector<unsigned, 2> order{1, 0};
    if (!aEncoding.isa<triton::gpu::TritonGPUSharedEncodingAttr>()) {
      Attribute encoding = triton::gpu::TritonGPUSharedEncodingAttr::get(
          getContext(), 1, 1, 1, order);
      auto dstType = RankedTensorType::get(aType.getShape(),
                                           aType.getElementType(), encoding);
      a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), dstType, a);
    }
    if (!bEncoding.isa<triton::gpu::TritonGPUSharedEncodingAttr>()) {
      Attribute encoding = triton::gpu::TritonGPUSharedEncodingAttr::get(
          getContext(), 1, 1, 1, order);
      auto dstType = RankedTensorType::get(bType.getShape(),
                                           bType.getElementType(), encoding);
      b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), dstType, b);
    }
    auto newDot = rewriter.replaceOpWithNewOp<triton::DotOp>(
        op, retType, a, b, adaptor.c(), adaptor.allowTF32());
    return success();
  }
};

struct TritonLoadPattern : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, retType, adaptor.ptr(), adaptor.mask(), adaptor.other(),
        adaptor.cache(), adaptor.evict(), adaptor.isVolatile());
    return success();
  }
};

struct TritonStorePattern : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, adaptor.ptr(), adaptor.value(), adaptor.mask());
    return success();
  }
};

template <class Op>
struct TritonGenericPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = this->getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<Op>(op, retType, adaptor.getOperands());
    return success();
  }
};

struct TritonBroadcastPattern
    : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  // This creates a tensor with the new shape but the argument's layout
  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = adaptor.src().getType().cast<RankedTensorType>();
    auto srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    auto opType = op.getType().cast<RankedTensorType>();
    Type retType = RankedTensorType::get(opType.getShape(),
                                         opType.getElementType(), srcEncoding);
    // Type retType = this->getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<triton::BroadcastOp>(op, retType,
                                                     adaptor.getOperands());
    return success();
  }
};

struct TritonReducePattern : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = this->getTypeConverter()->convertType(op.getType());
    auto newOp = rewriter.replaceOpWithNewOp<triton::ReduceOp>(
        op, retType, adaptor.redOp(), adaptor.operand(), adaptor.axis());
    return success();
  }
};

void populateTritonPatterns(TritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add< // TODO: view should have custom pattern that views the layout
      TritonGenericPattern<triton::ViewOp>,
      TritonGenericPattern<triton::SplatOp>, TritonBroadcastPattern,
      TritonGenericPattern<triton::GEPOp>, TritonReducePattern,
      TritonExpandDimsPattern, TritonMakeRangePattern, TritonDotPattern,
      TritonLoadPattern, TritonStorePattern>(typeConverter, context);
}

//
// SCF patterns
//
// This is borrowed from ConvertForOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
struct SCFForPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;
  // Ref: ConvertForOpTypes
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getLoopBody(), newOp.getLoopBody(),
                                newOp.getLoopBody().end());

    // Now, update all the types.

    // Convert the types of block arguments within the given region. This
    // replaces each block with a new block containing the updated signature.
    // The entry block may have a special conversion if `entryConversion` is
    // provided. On success, the new entry block to the region is returned for
    // convenience. Otherwise, failure is returned.
    if (failed(rewriter.convertRegionTypes(&newOp.getLoopBody(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    // Change the clone to use the updated operands. We could have cloned with
    // a BlockAndValueMapping, but this seems a bit more direct.
    newOp->setOperands(adaptor.getOperands());
    // Update the result types to the new converted types.
    SmallVector<Type> newResultTypes;
    for (Type type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));

    rewriter.replaceOp(op, newOp.getResults());

    return success();
  }
};

struct SCFYieldPattern : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // rewriter.setInsertionPointToEnd(rewriter.getInsertionBlock());
    // rewriter.create<scf::YieldOp>(op.getLoc(), adaptor.getOperands());
    // op.erase();
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

void populateSCFPatterns(TritonGPUTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<SCFYieldPattern, SCFForPattern>(typeConverter, context);
}

class ConvertTritonToTritonGPU
    : public ConvertTritonToTritonGPUBase<ConvertTritonToTritonGPU> {
public:
  ConvertTritonToTritonGPU(int numWarps) { this->numWarps = numWarps; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // type converter
    TritonGPUTypeConverter typeConverter(context, numWarps);
    TritonGPUConversionTarget target(*context, typeConverter);
    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    populateArithmeticPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns);
    // TODO: can we use
    //    mlir::scf::populateSCFStructurealTypeConversionsAndLegality(...) here?
    populateSCFPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    // update layouts
    //  broadcast src => multicast, dst => broadcasted
    // if (failed(target.refineLayouts(mod, numWarps)))
    //   return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToTritonGPUPass(int numWarps) {
  return std::make_unique<::ConvertTritonToTritonGPU>(numWarps);
}
