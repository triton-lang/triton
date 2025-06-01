#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>

namespace mlir::triton {
class LinearLayout;
}

namespace mlir::triton::gpu {
LinearLayout optimalSwizzling(const LinearLayout &src, const LinearLayout &dst,
                              int32_t bitwidth);

// Compute a basis of span(b1) ∩ span(b2)
llvm::SmallVector<int32_t> intersectionBasis(llvm::ArrayRef<int32_t> b1,
                                             llvm::ArrayRef<int32_t> b2,
                                             int32_t rank);
} // namespace mlir::triton::gpu
