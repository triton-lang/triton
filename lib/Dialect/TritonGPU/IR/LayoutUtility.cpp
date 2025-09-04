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

} // namespace mlir::triton::gpu
