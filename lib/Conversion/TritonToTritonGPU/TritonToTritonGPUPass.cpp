#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"

namespace {

// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

template <class Op> class GenericOpPattern : public OpConversionPattern<Op> {
public:
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(
        rewriter.replaceOpWithNewOp<Op>(op, retType, adaptor.getOperands()),
        adaptor.getAttributes());
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
    addNamedAttrs(
        rewriter.replaceOpWithNewOp<DstOp>(op, retType, adaptor.getPredicate(),
                                           adaptor.getLhs(), adaptor.getRhs()),
        adaptor.getAttributes());
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
    if (value.getElementType().isInteger(1) && value.isSplat())
      // Workaround until https://reviews.llvm.org/D133743 is included.
      value = DenseElementsAttr::get(retType, value.getSplatValue<bool>());
    else
      // This is a hack. We just want to add encoding
      value = value.reshape(retType);
    addNamedAttrs(
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, retType, value),
        adaptor.getAttributes());
    return success();
  }
};

class ConvertArithOp : public ConversionPattern {
public:
  ConvertArithOp(TritonGPUTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    Dialect *dialect = op->getDialect();
    if (dialect->getTypeID() != mlir::TypeID::get<arith::ArithDialect>())
      return failure();
    return success();
  }
};

void populateArithPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      TritonGPUConversionTarget &target) {
  // --------------
  // Add legality and rewrite pattern rules for operations
  // from the Arith dialect. The basic premise is that
  // Arith operations require both inputs to have the same
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
      GenericOpPattern<arith::FPToSIOp>, GenericOpPattern<arith::FPToUIOp>,
      GenericOpPattern<arith::UIToFPOp>>(typeConverter, context);
}

// this shouldn't exist if mlir's SelectOp checked encodings properly
class StdSelectPattern : public OpConversionPattern<arith::SelectOp> {
public:
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::gpu::SelectOp>(
                      op, retType, adaptor.getCondition(),
                      adaptor.getTrueValue(), adaptor.getFalseValue()),
                  adaptor.getAttributes());
    return success();
  }
};

void populateStdPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    TritonGPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // Rewrite rule
  patterns.add<StdSelectPattern>(typeConverter, context);
  target.addLegalOp<func::ReturnOp>(); // this is ok because all functions are
                                       // inlined by the frontend
}

void populateMathPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     TritonGPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // Rewrite rule
  patterns.add<GenericOpPattern<math::ExpOp>, GenericOpPattern<math::CosOp>,
               GenericOpPattern<math::SinOp>, GenericOpPattern<math::LogOp>,
               GenericOpPattern<math::AbsFOp>, GenericOpPattern<math::AbsIOp>,
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
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::MakeRangeOp>(
                      op, retType, adaptor.getStart(), adaptor.getEnd()),
                  adaptor.getAttributes());
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
    RankedTensorType argType =
        adaptor.getSrc().getType().cast<RankedTensorType>();
    Attribute _argEncoding = argType.getEncoding();
    if (!_argEncoding)
      return failure();
    auto argEncoding = _argEncoding.cast<triton::gpu::BlockedEncodingAttr>();
    // return shape
    auto retShape = argType.getShape().vec();
    retShape.insert(retShape.begin() + op.getAxis(), 1);
    // return encoding
    auto retSizePerThread = argEncoding.getSizePerThread().vec();
    retSizePerThread.insert(retSizePerThread.begin() + op.getAxis(), 1);
    auto retThreadsPerWarp = argEncoding.getThreadsPerWarp().vec();
    retThreadsPerWarp.insert(retThreadsPerWarp.begin() + op.getAxis(), 1);
    auto retWarpsPerCTA = argEncoding.getWarpsPerCTA().vec();
    retWarpsPerCTA.insert(retWarpsPerCTA.begin() + op.getAxis(), 1);
    SmallVector<unsigned, 4> retOrder(retShape.size());
    std::iota(retOrder.begin(), retOrder.end(), 0);
    triton::gpu::BlockedEncodingAttr retEncoding =
        triton::gpu::BlockedEncodingAttr::get(getContext(), retSizePerThread,
                                              retThreadsPerWarp, retWarpsPerCTA,
                                              retOrder);
    // convert operand to slice of return type
    Attribute newArgEncoding = triton::gpu::SliceEncodingAttr::get(
        getContext(), op.getAxis(), retEncoding);
    RankedTensorType newArgType = RankedTensorType::get(
        argType.getShape(), argType.getElementType(), newArgEncoding);
    // construct new op
    auto newSrc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        op.getLoc(), newArgType, adaptor.getSrc());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(
                      op, newSrc, adaptor.getAxis()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType origType = op.getType().cast<RankedTensorType>();
    auto origShape = origType.getShape();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();
    int numWarps = typeConverter->getNumWarps();

    SmallVector<unsigned> retSizePerThread = {1, 1};
    if (origShape[0] * origShape[1] / (numWarps * 32) >= 4)
      retSizePerThread = {2, 2};
    if (origShape[0] * origShape[1] / (numWarps * 32) >= 16)
      retSizePerThread = {4, 4};
    SmallVector<unsigned> retOrder = {1, 0};
    Attribute dEncoding = triton::gpu::BlockedEncodingAttr::get(
        getContext(), origShape, retSizePerThread, retOrder, numWarps);
    RankedTensorType retType =
        RankedTensorType::get(origShape, origType.getElementType(), dEncoding);
    // a & b must be of smem layout
    auto aType = adaptor.getA().getType().cast<RankedTensorType>();
    auto bType = adaptor.getB().getType().cast<RankedTensorType>();
    Attribute aEncoding = aType.getEncoding();
    Attribute bEncoding = bType.getEncoding();
    if (!aEncoding || !bEncoding)
      return failure();
    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
    if (!aEncoding.isa<triton::gpu::DotOperandEncodingAttr>()) {
      Attribute encoding =
          triton::gpu::DotOperandEncodingAttr::get(getContext(), 0, dEncoding);
      auto dstType = RankedTensorType::get(aType.getShape(),
                                           aType.getElementType(), encoding);
      a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), dstType, a);
    }
    if (!bEncoding.isa<triton::gpu::DotOperandEncodingAttr>()) {
      Attribute encoding =
          triton::gpu::DotOperandEncodingAttr::get(getContext(), 1, dEncoding);
      auto dstType = RankedTensorType::get(bType.getShape(),
                                           bType.getElementType(), encoding);
      b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), dstType, b);
    }
    c = rewriter.create<triton::gpu::ConvertLayoutOp>(c.getLoc(), retType, c);

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::DotOp>(
                      op, retType, a, b, c, adaptor.getAllowTF32()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonCatPattern : public OpConversionPattern<triton::CatOp> {

  using OpConversionPattern<triton::CatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // For now, this behaves like generic, but this will evolve when
    // we add support for `can_reorder=False`
    Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::CatOp>(
                      op, retType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonTransPattern : public OpConversionPattern<triton::TransOp> {

  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    auto srcType = src.getType().cast<RankedTensorType>();
    Attribute srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    if (!srcEncoding.isa<triton::gpu::SharedEncodingAttr>()) {
      // TODO: end-to-end correctness is broken if
      // the input is blocked and the output is shared
      // with different order. Maybe a backend issue in BlockedToShared?
      SmallVector<unsigned> order = {1, 0};
      if (auto srcBlockedEncoding =
              srcEncoding.dyn_cast<triton::gpu::BlockedEncodingAttr>())
        llvm::copy(srcBlockedEncoding.getOrder(), order.begin());
      srcEncoding =
          triton::gpu::SharedEncodingAttr::get(getContext(), 1, 1, 1, order);
      srcType = RankedTensorType::get(srcType.getShape(),
                                      srcType.getElementType(), srcEncoding);
      src = rewriter.create<triton::gpu::ConvertLayoutOp>(src.getLoc(), srcType,
                                                          src);
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::TransOp>(op, src),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonLoadPattern : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::LoadOp>(
                      op, typeConverter->convertType(op.getType()),
                      adaptor.getPtr(), adaptor.getMask(), adaptor.getOther(),
                      adaptor.getBoundaryCheckAttr(), adaptor.getPaddingAttr(),
                      adaptor.getCache(), adaptor.getEvict(),
                      adaptor.getIsVolatile()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonStorePattern : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::StoreOp>(
                      op, adaptor.getPtr(), adaptor.getValue(),
                      adaptor.getMask(), adaptor.getCache(),
                      adaptor.getEvict()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonAtomicCASPattern
    : public OpConversionPattern<triton::AtomicCASOp> {
  using OpConversionPattern<triton::AtomicCASOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::AtomicCASOp>(
                      op, typeConverter->convertType(op.getType()),
                      adaptor.getPtr(), adaptor.getCmp(), adaptor.getVal()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonAtomicRMWPattern
    : public OpConversionPattern<triton::AtomicRMWOp> {
  using OpConversionPattern<triton::AtomicRMWOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::AtomicRMWOp>(
                      op, typeConverter->convertType(op.getType()),
                      adaptor.getAtomicRmwOp(), adaptor.getPtr(),
                      adaptor.getVal(), adaptor.getMask()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonExtElemwisePattern
    : public OpConversionPattern<triton::ExtElemwiseOp> {
  using OpConversionPattern<triton::ExtElemwiseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExtElemwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::ExtElemwiseOp>(
                      op, typeConverter->convertType(op.getType()),
                      adaptor.getArgs(), adaptor.getLibname(),
                      adaptor.getLibpath(), adaptor.getSymbol()),
                  adaptor.getAttributes());
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
    addNamedAttrs(
        rewriter.replaceOpWithNewOp<Op>(op, retType, adaptor.getOperands()),
        adaptor.getAttributes());
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
    auto srcType = adaptor.getSrc().getType().cast<RankedTensorType>();
    auto srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    auto opType = op.getType().cast<RankedTensorType>();
    Type retType = RankedTensorType::get(opType.getShape(),
                                         opType.getElementType(), srcEncoding);
    // Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::BroadcastOp>(
                      op, retType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonReducePattern : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    addNamedAttrs(
        rewriter.replaceOpWithNewOp<triton::ReduceOp>(
            op, adaptor.getRedOp(), adaptor.getOperand(), adaptor.getAxis()),
        adaptor.getAttributes());
    return success();
  }
};

struct TritonPrintPattern : public OpConversionPattern<triton::PrintOp> {
  using OpConversionPattern<triton::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, typename triton::PrintOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::PrintOp>(
                      op, op.getPrefixAttr(), adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonAssertPattern : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern<triton::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op,
                  typename triton::AssertOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::AssertOp>(
                      op, adaptor.getCondition(), op.getMessageAttr(),
                      op.getFileAttr(), op.getFuncAttr(), op.getLineAttr()),
                  adaptor.getAttributes());
    return success();
  }
};

void populateTritonPatterns(TritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns
      .insert< // TODO: view should have custom pattern that views the layout
          TritonGenericPattern<triton::ViewOp>,
          TritonGenericPattern<triton::BitcastOp>,
          TritonGenericPattern<triton::FpToFpOp>,
          TritonGenericPattern<triton::IntToPtrOp>,
          TritonGenericPattern<triton::PtrToIntOp>,
          TritonGenericPattern<triton::SplatOp>, TritonBroadcastPattern,
          TritonGenericPattern<triton::AddPtrOp>, TritonCatPattern,
          TritonReducePattern, TritonTransPattern, TritonExpandDimsPattern,
          TritonMakeRangePattern, TritonDotPattern, TritonLoadPattern,
          TritonStorePattern, TritonExtElemwisePattern, TritonPrintPattern,
          TritonAssertPattern, TritonAtomicRMWPattern>(typeConverter, context);
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
    // a IRMapping, but this seems a bit more direct.
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
    addNamedAttrs(
        rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands()),
        adaptor.getAttributes());
    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFIfPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Generalize this to any type conversion, not just 1:1.
    //
    // We need to implement something more sophisticated here that tracks which
    // types convert to which other types and does the appropriate
    // materialization logic.
    // For example, it's possible that one result type converts to 0 types and
    // another to 2 types, so newResultTypes would at least be the right size to
    // not crash in the llvm::zip call below, but then we would set the the
    // wrong type on the SSA values! These edge cases are also why we cannot
    // safely use the TypeConverter::convertTypes helper here.
    SmallVector<Type> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // See comments in the ForOp pattern for why we clone without regions and
    // then inline.
    scf::IfOp newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    // Update the operands and types.
    newOp->setOperands(adaptor.getOperands());
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFWhilePattern : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern<scf::WhileOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    assert(converter);
    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();

    auto newOp = rewriter.create<scf::WhileOp>(op.getLoc(), newResultTypes,
                                               adaptor.getOperands());
    for (auto i : {0u, 1u}) {
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
      if (failed(rewriter.convertRegionTypes(&dstRegion, *converter)))
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFConditionPattern : public OpConversionPattern<scf::ConditionOp> {
public:
  using OpConversionPattern<scf::ConditionOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.updateRootInPlace(
        op, [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

void populateSCFPatterns(TritonGPUTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<SCFYieldPattern, SCFForPattern, SCFIfPattern, SCFWhilePattern,
               SCFConditionPattern>(typeConverter, context);
}

// CF

class CFBranchPattern : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern<cf::BranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        op, op.getSuccessor(), adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(newOp.getSuccessor()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

class CFCondBranchPattern : public OpConversionPattern<cf::CondBranchOp> {
public:
  using OpConversionPattern<cf::CondBranchOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());

    if (failed(rewriter.convertRegionTypes(newOp.getTrueDest()->getParent(),
                                           *converter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(newOp.getFalseDest()->getParent(),
                                           *converter)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

class FuncOpPattern : public OpConversionPattern<func::FuncOp> {
public:
  using OpConversionPattern<func::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<func::FuncOp>(
        op, op.getName(), op.getFunctionType());
    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *converter)))
      return failure();

    return success();
  }
};

void populateCFPatterns(TritonGPUTypeConverter &typeConverter,
                        RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<FuncOpPattern, CFCondBranchPattern, CFBranchPattern>(
      typeConverter, context);
}
//

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
    populateArithPatternsAndLegality(typeConverter, patterns, target);
    populateMathPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns);
    // TODO: can we use
    //    mlir::scf::populateSCFStructurealTypeConversionsAndLegality(...) here?
    populateSCFPatterns(typeConverter, patterns);
    populateCFPatterns(typeConverter, patterns);

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
