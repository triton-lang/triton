#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

template <typename T, typename U, typename BinaryOp>
std::vector<unsigned> multiDimElementwise(ArrayRef<T> lhs, ArrayRef<U> rhs,
                                          BinaryOp op) {
  assert(lhs.size() == rhs.size() && "Input dimensions must match");
  std::vector<unsigned> result;
  result.reserve(lhs.size());
  for (size_t i = 0, n = lhs.size(); i < n; ++i) {
    unsigned a = static_cast<unsigned>(lhs[i]);
    unsigned b = static_cast<unsigned>(rhs[i]);
    result.push_back(op(a, b));
  }
  return result;
}

template <typename T> unsigned getNumElements(const ArrayRef<T> shape) {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

// Determine the order in which CTA tiles are laid out across the tensor.
// That is, create vector of dimensions from fastest to slowest varying.
SmallVector<unsigned> getCTATileOrder(MLIRContext *ctx,
                                      const LinearLayout &layout) {
  auto llEnc = triton::gpu::LinearEncodingAttr::get(ctx, layout);
  auto regDim = StringAttr::get(ctx, "register");
  auto &bases = layout.getBases().find(regDim)->second;

  // Compute number of CTA tiles in a layout.
  unsigned totalElems = layout.getTotalOutDimSize();
  auto ctaShape = llEnc.getShapePerCTATile();
  unsigned elemsPerCTA =
      std::accumulate(ctaShape.begin(), ctaShape.end(), 1, std::multiplies<>());
  assert((totalElems % elemsPerCTA) == 0 &&
         "Total elements must be divisible by elemsPerCTA");
  unsigned numCTAs = totalElems / elemsPerCTA;

  // To determine the CTA tile order, start by identifying the register basis
  // vector that corresponds to the first element of the second CTA tile. The
  // nonzero index in the logical tensor it maps to indicates the fastest
  // varying dimension. Then, for each subsequent basis register (first element
  // of some CTA tile), extract the next nonzero index to build the full
  // dimension order.
  unsigned registersPerThreadPerCTA =
      product(llEnc.basesPerDim(regDim, /*skipBroadcast=*/false)) / numCTAs;
  unsigned startIndex =
      static_cast<unsigned>(std::log2(registersPerThreadPerCTA));

  llvm::SmallSetVector<unsigned, 8> order;
  for (unsigned i = startIndex; i < bases.size(); ++i) {
    auto range = llvm::make_range(bases[i].begin(), bases[i].end());
    auto it = llvm::find_if(range, [](unsigned v) { return v != 0; });
    if (it != bases[i].end())
      order.insert(std::distance(bases[i].begin(), it));
  }

  // Append any dims missing from our default order.
  for (unsigned dim : llEnc.getOrder())
    order.insert(dim);

  return order.takeVector();
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
    auto linearLayoutSrc = triton::gpu::toLinearLayout(srcShape, srcEncoding);
    auto linearLayoutDst = triton::gpu::toLinearLayout(dstShape, dstEncoding);
    auto srcCTAOrder = getCTATileOrder(context, linearLayoutSrc);
    auto dstCTAOrder = getCTATileOrder(context, linearLayoutSrc);

    auto rank = srcShape.size();
    auto shapePerCTATile = triton::gpu::getShapePerCTATile(resultType);
    auto sources = adaptor.getSources();

    unsigned totalElems = ::getNumElements<int64_t>(dstShape);
    unsigned elemsPerTile = ::getNumElements<unsigned>(shapePerCTATile);
    unsigned numCTATiles = totalElems / elemsPerTile;

    // Default order is fastest to slowest varying dimension.
    std::vector<unsigned> defaultOrder(rank);
    std::iota(defaultOrder.rbegin(), defaultOrder.rend(), 0);

    auto dstCTAShape = multiDimElementwise<int64_t, unsigned>(
        dstShape, shapePerCTATile, std::divides<unsigned>());
    auto srcCTAShape = multiDimElementwise<int64_t, unsigned>(
        srcShape, shapePerCTATile, std::divides<unsigned>());
    auto srcToDstShape = multiDimElementwise<int64_t, int64_t>(
        dstShape, srcShape, std::divides<unsigned>());

    unsigned elemsPerThreadPerCTA =
        triton::gpu::getTotalElemsPerThread(srcType) /
        ::getNumElements<unsigned>(srcCTAShape);

    llvm::SmallVector<Value> resultVals;
    llvm::SmallVector<SmallVector<Value>> unpackedSources;
    unpackedSources.reserve(sources.size());

    for (size_t i = 0; i < sources.size(); i++) {
      Value currSrc = sources[i];
      unpackedSources.push_back(unpackLLElements(loc, currSrc, rewriter));
    }

    // Traverse CTA tiles in the result tensor
    for (int i = 0; i < numCTATiles; ++i) {
      auto currTileIdx = mlir::LLVM::delinearize(i, dstCTAShape, dstCTAOrder);
      // The n-dim destination tensor is built by arranging n-dim source tensors
      // into a destination tensor shape. Determine which source tensor contains
      // the current CTA tile.
      auto multiDimSrcIdx = multiDimElementwise<unsigned, unsigned>(
          currTileIdx, srcCTAShape, std::divides<unsigned>());
      // Compute linear index of the current source tensor.
      // Concat operands are laid out in the destination tensor
      // in fastest slowest varying dimension order.
      auto linearSrcIdx =
          mlir::LLVM::linearize(multiDimSrcIdx, srcToDstShape, defaultOrder);

      // After determining which source tensor the current CTA tile belongs to,
      // compute the index of this CTA tile within that source tensor,
      // considering the source tensors may include CTA tiles.
      auto multiDimSrcCTAIdx = multiDimElementwise<unsigned, unsigned>(
          currTileIdx, srcCTAShape, std::modulus<unsigned>());
      auto linearSrcCTAIdx =
          mlir::LLVM::linearize(multiDimSrcCTAIdx, srcCTAShape, srcCTAOrder);
      auto unpackedElements = unpackedSources[linearSrcIdx];

      auto startIt =
          unpackedElements.begin() + linearSrcCTAIdx * elemsPerThreadPerCTA;
      auto endIt = startIt + elemsPerThreadPerCTA;
      llvm::append_range(resultVals, llvm::make_range(startIt, endIt));
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
