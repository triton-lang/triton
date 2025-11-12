#include <llvm/Support/LogicalResult.h>
#include <triton/Dialect/TritonGPU/IR/Dialect.h>

namespace mlir::triton::gpu {

CTAEncodingAttr permuteCTALayout(MLIRContext *ctx, CTAEncodingAttr layout,
                                 ArrayRef<int> order);
} // namespace mlir::triton::gpu
