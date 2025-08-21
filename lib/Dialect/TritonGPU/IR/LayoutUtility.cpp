#include <triton/Dialect/TritonGPU/IR/LayoutUtility.h>
#include <triton/Tools/LayoutUtils.h>

#include <llvm/ADT/SmallVector.h>
#include <triton/Dialect/Triton/IR/Utility.h>

namespace mlir::triton::gpu {

CTALayoutAttr permuteCTALayout(MLIRContext *ctx, CTALayoutAttr layout,
                               ArrayRef<int> order) {
  auto n = order.size();
  assert(n == layout.getRank() && "order and layout rank mismatch");

  auto invOrder = inversePermutation(order);
  llvm::SmallVector<unsigned> invOrderUnsigned(invOrder.begin(),
                                               invOrder.end());
  return CTALayoutAttr::get(
      ctx, applyPermutation(layout.getCTAsPerCGA(), order),
      applyPermutation(layout.getCTASplitNum(), order),
      applyPermutation(invOrderUnsigned, layout.getCTAOrder()));
}

LinearLayout getPaddedRegToSharedLayout(const LinearLayout &regLayout,
                                        PaddedSharedEncodingAttr paddedEnc) {
  auto *ctx = paddedEnc.getContext();
  auto kOffset = StringAttr::get(ctx, "offset");
  auto outNames = to_vector(regLayout.getOutDimNames());
  auto order = paddedEnc.getOrder();
  // transposeOuts just iterates over out dims so we order them based on the
  // order from the encoding
  auto inOrderRegLayout =
      regLayout.transposeOuts(triton::applyPermutation(outNames, order));
  return inOrderRegLayout.reshapeOuts(
      {{kOffset, inOrderRegLayout.getTotalOutDimSize()}});
}

LinearLayout getElemIndexToSharedLayout(PaddedSharedEncodingAttr paddedEnc,
                                        ArrayRef<int64_t> shape) {
  auto *ctx = paddedEnc.getContext();
  auto kOffset = StringAttr::get(ctx, "offset");

  auto shapePerCTA = getShapePerCTA(paddedEnc, shape);

  auto innerCtaLayout = identityStandardND(
      kOffset, llvm::to_vector_of<unsigned>(shapePerCTA), paddedEnc.getOrder());

  auto ctaLayoutAttr = paddedEnc.getCTALayout();

  // Part of makeCgaLayout from LinearLayoutConversion.cpp
  StringAttr kBlock = StringAttr::get(ctx, "block");

  int rank = ctaLayoutAttr.getCTAOrder().size();
  SmallVector<StringAttr> outDimNames = standardOutDimNames(ctx, rank);

  LinearLayout cgaLayout = LinearLayout::empty();
  for (int i = 0; i < rank; i++) {
    int dim = ctaLayoutAttr.getCTAOrder()[i];
    int split = ctaLayoutAttr.getCTASplitNum()[dim];
    int ctas = ctaLayoutAttr.getCTAsPerCGA()[dim];
    assert(ctas % split == 0);
    cgaLayout *= LinearLayout::identity1D(split, kBlock, outDimNames[dim]) *
                 LinearLayout::zeros1D(ctas / split, kBlock, outDimNames[dim]);
  }

  cgaLayout =
      cgaLayout.transposeOuts(llvm::to_vector(innerCtaLayout.getOutDimNames()));

  return innerCtaLayout * cgaLayout;
}

} // namespace mlir::triton::gpu
