#include <triton/Dialect/TritonGPU/IR/LayoutUtility.h>

#include <llvm/ADT/SmallVector.h>
#include <triton/Dialect/Triton/IR/Utility.h>
#include <triton/Tools/LayoutUtils.h>

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
  if (paddedEnc.getLinearComponent().has_value()) {
    auto sharedLL = *paddedEnc.getLinearComponent();
    // linear_component is the smallest repeatable tile, so we extend it to the
    // required shape. Note that we cannot trim the shared layout or otherwise
    // we will get incorrect results when we rearange elements
    llvm::SmallDenseMap<StringAttr, int64_t> namedShape;
    auto regOutDims = regLayout.getOutDims();
    for (auto [name, size] : regLayout.getOutDims()) {
      namedShape[name] = size;
      assert(sharedLL.getOutDimSize(name) <= size);
    }
    sharedLL = ensureLayoutNotSmallerThan(sharedLL, namedShape);
    return regLayout.invertAndCompose(sharedLL);
  }
  // Otherwise just return a contiguous mapping from regs to shared
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
