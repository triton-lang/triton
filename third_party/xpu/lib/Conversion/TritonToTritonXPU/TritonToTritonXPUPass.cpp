//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
// clang-format off
#include <numeric>
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h" // mlir::triton::op

#include "triton/Conversion/TritonToTritonXPU/Passes.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

#include "triton/Dialect/TritonXPU/Transforms/TritonXPUConversion.h" // TritonXPUTypeConverter + TritonXPUConversionTarget
#include "llvm/Support/ErrorHandling.h" // TODO[dyq]: Check All Pattern And Remove It

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonToTritonXPU/Passes.h.inc"
// clang-format on

namespace {
using namespace mlir;
using namespace mlir::triton;

// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

template <class Op> struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());

    return success();
  }
};

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto retShapedType = cast<ShapedType>(retType);
    auto value = dyn_cast<DenseElementsAttr>(adaptor.getValue());
    if (dyn_cast<RankedTensorType>(retShapedType)) {
      assert(value);
      if (value.getElementType().isInteger(1) && value.isSplat())
        // Workaround until https://reviews.llvm.org/D133743 is included.
        value =
            DenseElementsAttr::get(retShapedType, value.getSplatValue<bool>());
      else
        // This is a hack. We just want to add encoding
        value = value.reshape(retShapedType);
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                      op, retShapedType, value),
                  adaptor.getAttributes());
    return success();
  }
};

void populateArithPatternsAndLegality(TritonXPUTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      TritonXPUConversionTarget &target) {
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
      GenericOpPattern<arith::MaximumFOp>, GenericOpPattern<arith::MaxNumFOp>,
      GenericOpPattern<arith::MaxSIOp>, GenericOpPattern<arith::MaxUIOp>,
      GenericOpPattern<arith::MinimumFOp>, GenericOpPattern<arith::MinNumFOp>,
      GenericOpPattern<arith::MinSIOp>, GenericOpPattern<arith::MinUIOp>,
      // Floating point
      GenericOpPattern<arith::MulFOp>, GenericOpPattern<arith::DivFOp>,
      GenericOpPattern<arith::RemFOp>,
      // Cmp
      GenericOpPattern<arith::CmpIOp>, GenericOpPattern<arith::CmpFOp>,
      // Select
      GenericOpPattern<arith::SelectOp>,
      // Cast Ops
      GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
      GenericOpPattern<arith::ExtUIOp>, GenericOpPattern<arith::ExtSIOp>,
      GenericOpPattern<arith::ExtFOp>, GenericOpPattern<arith::SIToFPOp>,
      GenericOpPattern<arith::FPToSIOp>, GenericOpPattern<arith::FPToUIOp>,
      GenericOpPattern<arith::UIToFPOp>>(typeConverter, context);
}

void populateMathPatternsAndLegality(TritonXPUTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     TritonXPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // Rewrite rule
  patterns.add<GenericOpPattern<math::ExpOp>, GenericOpPattern<math::Exp2Op>,
               GenericOpPattern<math::FloorOp>, GenericOpPattern<math::CeilOp>,
               GenericOpPattern<math::CosOp>, GenericOpPattern<math::SinOp>,
               GenericOpPattern<math::LogOp>, GenericOpPattern<math::Log2Op>,
               GenericOpPattern<math::ErfOp>, GenericOpPattern<math::AbsFOp>,
               GenericOpPattern<math::AbsIOp>, GenericOpPattern<math::SqrtOp>,
               GenericOpPattern<math::RsqrtOp>, GenericOpPattern<math::FmaOp>>(
      typeConverter, context);
}

//
// Triton patterns
//
struct TritonExpandDimsPattern
    : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Type retType = op.getType());
    RankedTensorType argType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    Attribute _argEncoding = argType.getEncoding();
    if (!_argEncoding)
      return failure();
    auto argEncoding = cast<triton::xpu::ClusterLayoutAttr>(_argEncoding);
    // return shape
    auto retShape = argType.getShape().vec();
    retShape.insert(retShape.begin() + op.getAxis(), 1);
    // return encoding
    auto retSizePerCore = argEncoding.getSizePerCore().vec();
    retSizePerCore.insert(retSizePerCore.begin() + op.getAxis(), 1);
    auto retCoresPerGroup = argEncoding.getCoresPerGroup().vec();
    retCoresPerGroup.insert(retCoresPerGroup.begin() + op.getAxis(), 1);
    auto retGroupsPerCluster = argEncoding.getGroupsPerCluster().vec();
    retGroupsPerCluster.insert(retGroupsPerCluster.begin() + op.getAxis(), 1);
    SmallVector<unsigned, 4> retOrder(retShape.size());
    std::iota(retOrder.begin(), retOrder.end(), 0);

    bool isReduceOpt = argEncoding.getIsReduceOpt();

    triton::xpu::ClusterLayoutAttr retEncoding =
        triton::xpu::ClusterLayoutAttr::get(
            getContext(), retSizePerCore, retCoresPerGroup, retGroupsPerCluster,
            retOrder, isReduceOpt);

    // convert operand to slice of return type
    Attribute newArgEncoding = triton::gpu::SliceEncodingAttr::get(
        getContext(), op.getAxis(), retEncoding);
    RankedTensorType newArgType = RankedTensorType::get(
        argType.getShape(), argType.getElementType(), newArgEncoding);
    // construct new op
    auto newSrc = rewriter.create<triton::xpu::ConvertLayoutOp>(
        op.getLoc(), newArgType, adaptor.getSrc());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(
                      op, newSrc, adaptor.getAxis()),
                  adaptor.getAttributes());
    return success();
  }

private:
  template <typename T>
  SmallVector<T> insertOne(ArrayRef<T> vec, unsigned axis) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + axis, 1);
    return res;
  }

  // Example:    order = [   0, 2, 1, 3], dim = 2
  //          resOrder = [2, 0, 3, 1, 4]
  SmallVector<unsigned> insertOrder(ArrayRef<unsigned> order,
                                    unsigned axis) const {
    SmallVector<unsigned> resOrder(order.begin(), order.end());
    for (unsigned i = 0; i < resOrder.size(); ++i)
      if (resOrder[i] >= axis)
        ++resOrder[i];
    resOrder.insert(resOrder.begin(), axis);
    return resOrder;
  }
};

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm_unreachable("TODO[dyq]: XPUSDNN-CHECK Add "
                     "triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure();
  }
};

struct TritonCatPattern : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm_unreachable(
        "TODO[dyq]: Add triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure();
  }
};

struct TritonJoinOpPattern : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    llvm_unreachable("TODO[dyq]: Check Logic");
    // Simply rely on type inference for this op.  (Notably, GenericOpPattern
    // does not do this, instead it assigns the default layout to the ins and
    // outs.)
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::JoinOp>(
                      op, adaptor.getLhs(), adaptor.getRhs()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonSplitOpPattern : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    llvm_unreachable(
        "TODO[dyq]: Add triton::xpu::GlobalEncodingAttr Calculation Logic");
    return failure();
  }
};

struct TritonTransPattern : public OpConversionPattern<TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm_unreachable("TODO[dyq]: Check Logic");
    Value src = adaptor.getSrc();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcEnc = srcTy.getEncoding();
    if (!srcEnc)
      return failure();
    addNamedAttrs(rewriter.replaceOpWithNewOp<TransOp>(op, src, op.getOrder()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonBroadcastPattern
    : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  // This creates a tensor with the new shape but the argument's layout
  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(adaptor.getSrc().getType());
    auto srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    Type retType = RankedTensorType::get(
        op.getType().getShape(), op.getType().getElementType(), srcEncoding);
    // Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::xpu::BroadcastOp>(
                      op, retType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};
struct TritonReducePattern : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newReduce = rewriter.create<triton::ReduceOp>(
        op.getLoc(), adaptor.getOperands(), adaptor.getAxis());
    addNamedAttrs(newReduce, adaptor.getAttributes());

    auto &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newReduce.getResult());
    return success();
  }
};

struct TritonScanPattern : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm_unreachable("TODO[dyq]: Check Logic");
    auto newScan = rewriter.create<triton::ScanOp>(
        op.getLoc(), adaptor.getOperands(), adaptor.getAxis(), op.getReverse());
    addNamedAttrs(newScan, adaptor.getAttributes());

    auto &newCombineOp = newScan.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newScan.getResult());
    return success();
  }
};

class TritonFuncOpPattern : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm_unreachable("TODO[dyq]: Check Logic");
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<triton::FuncOp>(
        op, op.getName(), op.getFunctionType());
    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    if (failed(rewriter.convertRegionTypes(&newOp.getBody(), *converter)))
      return failure();

    return success();
  }
};

void populateTritonPatterns(TritonXPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.insert< // TODO: view should have custom pattern that views the
                   // layout
      GenericOpPattern<triton::AdvanceOp>,
      GenericOpPattern<triton::MakeTensorPtrOp>,
      GenericOpPattern<triton::ReshapeOp>, GenericOpPattern<triton::BitcastOp>,
      GenericOpPattern<triton::FpToFpOp>, GenericOpPattern<triton::IntToPtrOp>,
      GenericOpPattern<triton::PtrToIntOp>, GenericOpPattern<triton::SplatOp>,
      TritonBroadcastPattern, GenericOpPattern<triton::AddPtrOp>,
      TritonCatPattern, TritonJoinOpPattern, TritonSplitOpPattern,
      GenericOpPattern<triton::ClampFOp>,
      GenericOpPattern<triton::PreciseSqrtOp>,
      GenericOpPattern<triton::PreciseDivFOp>,
      GenericOpPattern<triton::MulhiUIOp>,
      GenericOpPattern<triton::ElementwiseInlineAsmOp>, TritonReducePattern,
      GenericOpPattern<triton::ReduceReturnOp>, TritonScanPattern,
      GenericOpPattern<triton::ScanReturnOp>,
      GenericOpPattern<triton::MakeRangeOp>, TritonExpandDimsPattern,
      TritonTransPattern, TritonDotPattern, GenericOpPattern<triton::LoadOp>,
      GenericOpPattern<triton::StoreOp>, GenericOpPattern<triton::HistogramOp>,
      GenericOpPattern<triton::ExternElementwiseOp>,
      GenericOpPattern<triton::PrintOp>, GenericOpPattern<triton::AssertOp>,
      GenericOpPattern<triton::AtomicCASOp>,
      GenericOpPattern<triton::AtomicRMWOp>, GenericOpPattern<ReturnOp>,
      GenericOpPattern<triton::ExperimentalDescriptorLoadOp>,
      GenericOpPattern<triton::ExperimentalDescriptorStoreOp>,
      GenericOpPattern<triton::CallOp>, TritonFuncOpPattern>(typeConverter,
                                                             context);
}

//
// SCF patterns
//
// This is borrowed from ConvertForOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
struct SCFForPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;
  // Ref: ConvertForOpTypes
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Now, update all the types.

    // Convert the types of block arguments within the given region. This
    // replaces each block with a new block containing the updated signature.
    // The entry block may have a special conversion if `entryConversion` is
    // provided. On success, the new entry block to the region is returned for
    // convenience. Otherwise, failure is returned.
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
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

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFIfPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
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
  using OpConversionPattern::OpConversionPattern;

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
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

void populateSCFPatterns(TritonXPUTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<scf::YieldOp>, SCFForPattern, SCFIfPattern,
               SCFWhilePattern, SCFConditionPattern>(typeConverter, context);
}

// CF

class CFBranchPattern : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm_unreachable("TODO[dyq]: Check Logic");
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
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm_unreachable("TODO[dyq]: Check Logic");
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
    return success();
  }
};

void populateCFPatterns(TritonXPUTypeConverter &typeConverter,
                        RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<CFCondBranchPattern, CFBranchPattern>(typeConverter, context);
}
//

class ConvertTritonToTritonXPU
    : public ConvertTritonToTritonXPUBase<ConvertTritonToTritonXPU> {
public:
  ConvertTritonToTritonXPU() = default;
  // constructor with some parameters set explicitly.
  ConvertTritonToTritonXPU(uint32_t xpu_arch, uint32_t buffer_size,
                           uint32_t core_num) {
    this->xpu_arch = xpu_arch;
    this->buffer_size = buffer_size;
    this->core_num = core_num;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // type converter. the reason that we cant use TT2TTGPass directly
    TritonXPUTypeConverter typeConverter(context, buffer_size, core_num);
    TritonXPUConversionTarget target(*context, typeConverter);
    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    populateArithPatternsAndLegality(typeConverter, patterns, target);
    populateMathPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns);
    // TODO: can we use
    //    mlir::scf::populateSCFStructurealTypeConversionsAndLegality(...) here?
    populateSCFPatterns(typeConverter, patterns);
    populateCFPatterns(typeConverter, patterns);

    auto inti = llvm::APSInt(32, false);
    auto i32_ty = IntegerType::get(mod->getContext(), 32);

    if (!this->xpu_arch.getValue()) {
      mod.emitError("expected target specification to attach to the module op");
      return signalPassFailure();
    }
    mod->setAttr(
        AttrXPUTargetName,
        StringAttr::get(context,
                        "xpu:" + std::to_string(this->xpu_arch.getValue())));

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
mlir::triton::createConvertTritonToTritonXPUPass(uint32_t xpu_arch,
                                                 uint32_t buffer_size,
                                                 uint32_t core_num) {
  return std::make_unique<::ConvertTritonToTritonXPU>(xpu_arch, buffer_size,
                                                      core_num);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonToTritonXPUPass() {
  return std::make_unique<::ConvertTritonToTritonXPU>();
}
