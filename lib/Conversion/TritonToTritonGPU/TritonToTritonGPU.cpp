#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPU.h"
#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>
using namespace mlir;
using namespace mlir::triton;

namespace {

template <class Op> class GenericOpPattern : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = this->getTypeConverter()->convertType(op.getType());
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
  // TODO: there's probably a better way to avoid adding all ops one-by-one
  patterns.add<
      ArithConstantPattern, GenericOpPattern<arith::AddIOp>,
      GenericOpPattern<arith::SubIOp>, GenericOpPattern<arith::MulIOp>,
      GenericOpPattern<arith::DivUIOp>, GenericOpPattern<arith::DivSIOp>,
      GenericOpPattern<arith::CeilDivUIOp>,
      GenericOpPattern<arith::CeilDivSIOp>,
      GenericOpPattern<arith::FloorDivSIOp>, GenericOpPattern<arith::RemUIOp>,
      GenericOpPattern<arith::RemSIOp>, GenericOpPattern<arith::AndIOp>,
      GenericOpPattern<arith::OrIOp>, GenericOpPattern<arith::XOrIOp>,
      GenericOpPattern<arith::ShLIOp>, GenericOpPattern<arith::ShRUIOp>,
      GenericOpPattern<arith::ShRSIOp>, // NegFOp
      // Floating point
      GenericOpPattern<arith::AddFOp>, GenericOpPattern<arith::SubFOp>,
      // MaxMin
      GenericOpPattern<arith::MaxFOp>, GenericOpPattern<arith::MaxSIOp>,
      GenericOpPattern<arith::MaxUIOp>, GenericOpPattern<arith::MinFOp>,
      GenericOpPattern<arith::MinSIOp>, GenericOpPattern<arith::MinUIOp>,
      // Floating point
      GenericOpPattern<arith::MulFOp>, GenericOpPattern<arith::DivFOp>,
      GenericOpPattern<arith::RemFOp>,
      // Cmp
      ArithCmpPattern<arith::CmpIOp, triton::gpu::CmpIOp>,
      ArithCmpPattern<arith::CmpFOp, triton::gpu::CmpFOp>,
      // Cast Ops
      GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
      GenericOpPattern<arith::ExtUIOp>, GenericOpPattern<arith::ExtSIOp>,
      GenericOpPattern<arith::ExtFOp>, GenericOpPattern<arith::SIToFPOp>,
      GenericOpPattern<arith::UIToFPOp>>(typeConverter, context);
}

// this shouldn't exist if mlir's SelectOp checked encodings properly
class StdSelectPattern : public OpConversionPattern<SelectOp> {
public:
  using OpConversionPattern<SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SelectOp op, typename SelectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = this->getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<triton::gpu::SelectOp>(
        op, retType, adaptor.getCondition(), adaptor.getTrueValue(),
        adaptor.getFalseValue());
    return success();
  }
};

void populateStdPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    TritonGPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // Rewrite rule
  patterns.add<StdSelectPattern>(typeConverter, context);
  target.addLegalOp<ReturnOp>(); // this is ok because all functions are inlined
                                 // by the frontend
}

void populateMathPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     TritonGPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // Rewrite rule
  patterns.add<GenericOpPattern<math::ExpOp>, GenericOpPattern<math::CosOp>,
               GenericOpPattern<math::SinOp>, GenericOpPattern<math::LogOp>,
               GenericOpPattern<math::SqrtOp>>(typeConverter, context);
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
    auto argEncoding = _argEncoding.cast<triton::gpu::BlockedEncodingAttr>();
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
    triton::gpu::BlockedEncodingAttr retEncoding =
        triton::gpu::BlockedEncodingAttr::get(getContext(), retSizePerThread,
                                              retThreadsPerWarp, retWarpsPerCTA,
                                              retOrder);
    // convert operand to slice of return type
    Attribute newArgEncoding = triton::gpu::SliceEncodingAttr::get(
        getContext(), op.axis(), retEncoding);
    RankedTensorType newArgType = RankedTensorType::get(
        argType.getShape(), argType.getElementType(), newArgEncoding);
    // construct new op
    auto newSrc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        op.getLoc(), newArgType, adaptor.src());
    rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(op, newSrc,
                                                      adaptor.axis());
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
    if (!aEncoding.isa<triton::gpu::SharedEncodingAttr>()) {
      Attribute encoding =
          triton::gpu::SharedEncodingAttr::get(getContext(), 1, 1, 1, order);
      auto dstType = RankedTensorType::get(aType.getShape(),
                                           aType.getElementType(), encoding);
      a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), dstType, a);
    }
    if (!bEncoding.isa<triton::gpu::SharedEncodingAttr>()) {
      Attribute encoding =
          triton::gpu::SharedEncodingAttr::get(getContext(), 1, 1, 1, order);
      auto dstType = RankedTensorType::get(bType.getShape(),
                                           bType.getElementType(), encoding);
      b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), dstType, b);
    }
    rewriter.replaceOpWithNewOp<triton::DotOp>(
        op, retType, a, b, adaptor.c(), adaptor.allowTF32(), adaptor.transA(),
        adaptor.transB());
    return success();
  }
};

struct TritonLoadPattern : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, typeConverter->convertType(op.getType()), adaptor.ptr(),
        adaptor.mask(), adaptor.other(), adaptor.cache(), adaptor.evict(),
        adaptor.isVolatile());
    return success();
  }
};

struct TritonStorePattern : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, adaptor.ptr(), adaptor.value(), adaptor.mask());
    return success();
  }
};

struct TritonExtElemwisePattern
    : public OpConversionPattern<triton::ExtElemwiseOp> {
  using OpConversionPattern<triton::ExtElemwiseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExtElemwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<triton::ExtElemwiseOp>(
        op, typeConverter->convertType(op.getType()), adaptor.args(),
        adaptor.libname(), adaptor.libpath(), adaptor.symbol());
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
    rewriter.replaceOpWithNewOp<triton::ReduceOp>(
        op, adaptor.redOp(), adaptor.operand(), adaptor.axis());
    return success();
  }
};

void populateTritonPatterns(TritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add< // TODO: view should have custom pattern that views the layout
      TritonGenericPattern<triton::ViewOp>,
      TritonGenericPattern<triton::BitcastOp>,
      TritonGenericPattern<triton::IntToPtrOp>,
      TritonGenericPattern<triton::PtrToIntOp>,
      TritonGenericPattern<triton::SplatOp>, TritonBroadcastPattern,
      TritonGenericPattern<triton::AddPtrOp>, TritonReducePattern,
      TritonExpandDimsPattern, TritonMakeRangePattern, TritonDotPattern,
      TritonLoadPattern, TritonStorePattern, TritonExtElemwisePattern>(
      typeConverter, context);
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
  ConvertTritonToTritonGPU() = default;
  // constructor with some parameters set explicitly.
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
    populateStdPatternsAndLegality(typeConverter, patterns, target);
    populateArithmeticPatternsAndLegality(typeConverter, patterns, target);
    populateMathPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns);
    // TODO: can we use
    //    mlir::scf::populateSCFStructurealTypeConversionsAndLegality(...) here?
    populateSCFPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    auto inti = llvm::APSInt(32, false);
    auto i32_ty = IntegerType::get(mod->getContext(), 32);

    mod->setAttr(
        AttrNumWarpsName,
        IntegerAttr::get(i32_ty, llvm::APInt(32, numWarps.getValue())));

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

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToTritonGPUPass() {
  return std::make_unique<::ConvertTritonToTritonGPU>();
}
