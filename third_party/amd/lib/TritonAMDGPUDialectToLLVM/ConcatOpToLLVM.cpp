#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/Utility/CommonUtils.h"
#include "third_party/amd/include/Utils/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

template <typename T> unsigned getNumElements(const ArrayRef<T> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

struct ConcatOpConversion : public ConvertOpToLLVMPattern<amdgpu::ConcatOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(amdgpu::ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    RankedTensorType resultType =
        cast<RankedTensorType>(op.getResult().getType());

    ArrayRef<int64_t> dstShape = resultType.getShape();
    Attribute dstEncoding = resultType.getEncoding();

    Value srcVal = op.getSources()[0];
    RankedTensorType srcType = cast<RankedTensorType>(srcVal.getType());
    ArrayRef<int64_t> srcShape = srcType.getShape();
    Attribute srcEncoding = srcType.getEncoding();

    MLIRContext *context = resultType.getContext();
    auto linearLayoutSrc = triton::gpu::toLinearLayout(srcType);
    auto outDimNames = llvm::to_vector(linearLayoutSrc.getOutDimNames());
    // Call transposeOuts, to ensure that order of input and output tensor
    // element coordinates are compatible on stage 8 in algorithm below.
    auto linearLayoutDst =
        triton::gpu::toLinearLayout(resultType).transposeOuts(outDimNames);
    auto rank = srcShape.size();
    // Default order is fastest to slowest varying dimension.
    std::vector<unsigned> defaultOrder(rank);
    std::iota(defaultOrder.rbegin(), defaultOrder.rend(), 0);
    auto srcToDstShape = LLVM::AMD::multiDimElementwise<int64_t, int64_t>(
        dstShape, srcShape, std::divides<unsigned>());

    auto sources = adaptor.getSources();

    llvm::SmallVector<Value> resultVals;
    llvm::SmallVector<SmallVector<Value>> unpackedSources;
    unpackedSources.reserve(sources.size());

    for (size_t i = 0; i < sources.size(); i++) {
      Value currSrc = sources[i];
      unpackedSources.push_back(unpackLLElements(loc, currSrc, rewriter));
    }

    // Algorithm:
    // 1. for all elements in dst tensor
    // 2.   get dst value location in tensor
    // 3.   find, which input tile holds the dst value
    // 4.   subtract dst coordinates and start coordinates of the tile
    // 5.   find source register number which holds dst value
    // 6.   copy dst element from computed tile and register
    auto ctx = rewriter.getContext();
    StringAttr kReg = StringAttr::get(ctx, "register");
    auto dstRegBases = linearLayoutDst.getBases().lookup(kReg);

    // for every output register get element coords,
    // find corresponding operand and copy src register
    int dstRegNum = 1 << dstRegBases.size();
    // 1. for all elements in dst tensor
    for (int regId = 0; regId < dstRegNum; ++regId) {
      // 2.   get dst value location in tensor
      auto elemCoords = mlir::triton::AMD::getElemCoordinatesFromRegisters(
          linearLayoutDst, regId, ctx);
      auto elemCoordsArray =
          llvm::to_vector(llvm::make_second_range(elemCoords));
      // The n-dim destination tensor is built by arranging n-dim source tensors
      // into a destination tensor shape. Determine which source tensor contains
      // the current CTA tile.
      auto multiDimOperandIdx =
          LLVM::AMD::multiDimElementwise<int32_t, int64_t>(
              elemCoordsArray, srcShape, std::divides<unsigned>());
      // Compute linear index of the current source tensor.
      // Concat operands are laid out in the destination tensor
      // in fastest  varying dimension order.
      // 3.   find, which input tile holds the dst value
      auto linearOperandIdx = mlir::LLVM::linearize(
          multiDimOperandIdx, srcToDstShape, defaultOrder);

      // 4.   subtract dst coordinates and start coordinates of the tile
      for (int dim = 0; dim < rank; ++dim)
        elemCoords[dim].second -= multiDimOperandIdx[dim] * srcShape[dim];

      // 5.   find source register number which holds dst value
      std::optional<int> srcReg = mlir::triton::AMD::getRegFromCoordinates(
          linearLayoutSrc, elemCoords, ctx);
      assert(srcReg.has_value());

      // 6.   copy dst element from found tile and register
      resultVals.push_back(unpackedSources[linearOperandIdx][srcReg.value()]);
    }

    Value packedResult = packLLElements(loc, this->getTypeConverter(),
                                        resultVals, rewriter, resultType);

    rewriter.replaceOp(op, packedResult);
    return success();
  }
};
} // namespace

namespace mlir::triton::AMD {
void populateConcatOpToLLVMPatterns(mlir::LLVMTypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns,
                                    mlir::PatternBenefit benefit) {
  patterns.add<ConcatOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::AMD
