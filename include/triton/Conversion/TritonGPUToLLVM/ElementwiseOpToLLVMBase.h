#ifndef TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_OP_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {

namespace gpu {

Type getElementType(Value value);

class MultipleOperandsRange
    : public iterator_range<SmallVector<SmallVector<Value>>::iterator> {
  using ContainerT = SmallVector<SmallVector<Value>>;

public:
  using iterator_range<ContainerT::iterator>::iterator_range;
  ContainerT::reference operator[](ContainerT::size_type idx) {
    return begin()[idx];
  }
  ContainerT::const_reference operator[](ContainerT::size_type idx) const {
    return begin()[idx];
  }
  ContainerT::size_type size() const { return end() - begin(); }
};

// Base pattern for elementwise conversion using ConcreteT. Unpacks individual
// elements from a `!llvm.struct` via `llvm.extactvalue`, calls
// ConcreteT::createDestOps on each element, and packs them back into an
// `!llvm.struct` using `llvm.insertvalue`.
//
// Also supports processing the inputs in a vectorized form by consuming and
// producing multiple operand sets in ConcreteT::createDestOps.
template <typename SourceOp, typename ConcreteT>
class ElementwiseOpConversionBase : public ConvertOpToLLVMPattern<SourceOp> {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  explicit ElementwiseOpConversionBase(
      LLVMTypeConverter &typeConverter,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      PatternBenefit benefit = patternBenefitDefault)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        axisAnalysisPass(axisAnalysisPass) {}

  // Try to deduplicate the resultVals based on the
  // constancy properties of the result discovered by
  // the axis analysis pass. If possible, redundant
  // computation is eliminated.
  SmallVector<Value> maybeDeduplicate(SourceOp op,
                                      SmallVector<Value> resultVals) const {
    auto ctx = op.getContext();
    if (!isMemoryEffectFree(op))
      // the op has side effects: can't dedup
      return resultVals;
    SmallVector<Value> results = op->getResults();
    if (results.size() == 0 || results.size() > 1)
      // there must be exactly 1 result
      return resultVals;
    Value result = results[0];
    RankedTensorType rtType = dyn_cast<RankedTensorType>(result.getType());
    if (!rtType)
      // the result must be a tensor
      return resultVals;

    // Bail out if we don't have the constancy analysis
    AxisInfo *axisInfo = axisAnalysisPass.getAxisInfo(result);
    if (!axisInfo)
      return resultVals;
    SmallVector<int64_t> constancy = axisInfo->getConstancy();

    if (llvm::all_of(constancy, [](int64_t c) { return c == 1; }))
      return resultVals;

    // We zero out the bases that are constant
    auto kReg = StringAttr::get(ctx, "register");
    auto ll = toLinearLayout(rtType);
    auto dims = to_vector(ll.getOutDimNames());
    auto llReg = ll.sublayout({kReg}, dims);
    auto inv = ll.pseudoinvert();
    auto invReg = inv.sublayout(dims, {kReg});
    auto bases_inv = invReg.getBases();
    for (auto [c, d] : llvm::zip(constancy, dims)) {
      assert(llvm::isPowerOf2_32(c));
      for (int i = 0; i < llvm::Log2_32(c); i++) {
        bases_inv[d][i] = {0};
      }
    }
    auto invBroadcast = LinearLayout(std::move(bases_inv), invReg.getOutDims(),
                                     /*isSurjective=*/false);
    auto cvt = llReg.compose(invBroadcast);

    // Deduplicate the result values
    SmallVector<Value> outVals(resultVals.size());
    for (int i = 0; i < outVals.size(); i++) {
      auto srcIdx = cvt.apply({{kReg, i}}).begin()->second;
      outVals[i] = resultVals[srcIdx];
    }
    return outVals;
  }
  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = op.getType();
    Location loc = op->getLoc();
    // element type
    auto resultElementTy = getElementTypeOrSelf(resultTy);
    Type elemTy = this->getTypeConverter()->convertType(resultElementTy);
    SmallVector<SmallVector<Value>> allOperands;
    for (auto operand : adaptor.getOperands()) {
      auto argTy = op->getOperand(0).getType();
      auto subOperands = unpackLLElements(loc, operand, rewriter);
      allOperands.resize(subOperands.size());
      for (auto v : llvm::enumerate(subOperands))
        allOperands[v.index()].push_back(v.value());
    }
    if (allOperands.size() == 0)
      allOperands.push_back({});

    SmallVector<Value> resultVals;
    for (auto it = allOperands.begin(), end = allOperands.end(); it != end;) {
      auto curr = static_cast<const ConcreteT *>(this)->createDestOps(
          op, adaptor, rewriter, elemTy, MultipleOperandsRange(it, end), loc);
      if (curr.size() == 0)
        return failure();
      for (auto v : curr) {
        if (!static_cast<bool>(v))
          return failure();
        resultVals.push_back(v);
      }
      it += curr.size();
    }
    resultVals = maybeDeduplicate(op, resultVals);
    Value view = packLLElements(loc, this->getTypeConverter(), resultVals,
                                rewriter, resultTy);
    rewriter.replaceOp(op, view);

    return success();
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;
};

// Trivial case where we map elementwise to an existing LLVM operator
template <typename SourceOp, typename DestOp>
struct ElementwiseOpConversion
    : public ElementwiseOpConversionBase<
          SourceOp, ElementwiseOpConversion<SourceOp, DestOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp,
                                  ElementwiseOpConversion<SourceOp, DestOp>>;
  using Base::Base;
  using OpAdaptor = typename Base::OpAdaptor;

  // An interface to support variant DestOp builder.
  SmallVector<DestOp> createDestOps(SourceOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter,
                                    Type elemTy, MultipleOperandsRange operands,
                                    Location loc) const {
    return {DestOp::create(rewriter, loc, elemTy, operands[0],
                           adaptor.getAttributes().getValue())};
  }
};

template <typename SourceOp>
struct ElementwiseToIntrinsicOpConversion
    : public ElementwiseOpConversionBase<
          SourceOp, ElementwiseToIntrinsicOpConversion<SourceOp>> {
  using Base =
      ElementwiseOpConversionBase<SourceOp, ElementwiseToIntrinsicOpConversion>;
  using OpAdaptor = typename Base::OpAdaptor;

  using Base::Base;

  explicit ElementwiseToIntrinsicOpConversion(
      LLVMTypeConverter &typeConverter,
      ModuleAxisInfoAnalysis &axisAnalysisPass, StringRef intrinsic,
      PatternBenefit benefit = patternBenefitDefault)
      : Base(typeConverter, axisAnalysisPass, benefit), intrinsic(intrinsic) {}

  SmallVector<Value> createDestOps(SourceOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   Type elemTy, MultipleOperandsRange operands,
                                   Location loc) const {
    return {LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic, elemTy,
                                            operands[0])
                .getResult(0)};
  }

private:
  StringRef intrinsic;
};

} // namespace gpu

} // namespace mlir::triton
#endif
