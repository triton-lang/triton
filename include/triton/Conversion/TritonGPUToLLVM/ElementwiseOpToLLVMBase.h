#ifndef TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_OP_H

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton {

namespace gpu {

inline Type getElementType(Value value) {
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return tensorType.getElementType();
  return type;
}

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

  explicit ElementwiseOpConversionBase(LLVMTypeConverter &typeConverter,
                                       ModuleAxisInfoAnalysis &axisAnalysisPass,
                                       PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<SourceOp>(typeConverter, benefit),
        axisAnalysisPass(axisAnalysisPass) {}

  // Try to deduplicate the resultVals based on the
  // constancy properties of the result discovered by
  // the axis analysis pass. If possible, redundant
  // computation is eliminated.
  SmallVector<Value> maybeDeduplicate(SourceOp op,
                                      SmallVector<Value> resultVals) const;
  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override;

protected:
  ModuleAxisInfoAnalysis &axisAnalysisPass;

private:
  int computeCapability;
};

#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.cpp"
} // namespace gpu

} // namespace mlir::triton
#endif
