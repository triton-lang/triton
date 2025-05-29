#include <triton/Dialect/TritonGPU/IR/LayoutUtility.h>

#include <llvm/ADT/SmallVector.h>
#include <triton/Dialect/Triton/IR/Utility.h>

namespace mlir::triton::gpu {

FailureOr<CTALayoutAttr>
permuteCTALayout(MLIRContext *ctx, CTALayoutAttr layout, ArrayRef<int> order) {
  auto n = order.size();
  if (layout.getCTAsPerCGA().size() != n ||
      layout.getCTASplitNum().size() != n || layout.getCTAOrder().size() != n) {
    return failure();
  }

  auto invOrder = inversePermutation(order);
  llvm::SmallVector<unsigned> invOrderUnsigned(invOrder.begin(),
                                               invOrder.end());
  return CTALayoutAttr::get(
      ctx, applyPermutation(layout.getCTAsPerCGA(), order),
      applyPermutation(layout.getCTASplitNum(), order),
      applyPermutation(invOrderUnsigned, layout.getCTAOrder()));
}

} // namespace mlir::triton::gpu
