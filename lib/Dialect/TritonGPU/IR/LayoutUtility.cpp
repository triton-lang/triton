#include <triton/Dialect/TritonGPU/IR/LayoutUtility.h>

#include <llvm/ADT/SmallVector.h>
#include <triton/Dialect/Triton/IR/Utility.h>
#include <triton/Tools/LayoutUtils.h>

namespace mlir::triton::gpu {

CTAEncodingAttr permuteCTALayout(MLIRContext *ctx, CTAEncodingAttr layout,
                                 ArrayRef<int> order) {
  auto ll = transposeLinearLayout(layout.getLinearLayout(), order);
  return CTAEncodingAttr::get(ctx, ll);
}

} // namespace mlir::triton::gpu
