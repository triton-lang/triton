#include <llvm/Support/LogicalResult.h>
#include <triton/Dialect/TritonGPU/IR/Dialect.h>

namespace mlir::triton::gpu {

llvm::FailureOr<CTALayoutAttr>
permuteCTALayout(MLIRContext *ctx, CTALayoutAttr layout, ArrayRef<int> order);

}
