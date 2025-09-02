#include <triton/Dialect/TritonGPU/IR/LayoutUtility.h>

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

} // namespace mlir::triton::gpu
