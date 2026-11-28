#ifndef TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_UTILITY_H
#define TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_UTILITY_H

#include "mlir/IR/Operation.h"
#include "triton/Analysis/Allocation.h"

namespace mlir {
namespace triton {
namespace membar {
struct SMemAccess;
}
namespace NVIDIA {

/// Return true if this pair of accesses needs no barrier even when their
/// shared storage overlaps.
bool canSkipBarSync(const membar::SMemAccess &source,
                    const membar::SMemAccess &sink, Allocation &allocation);
} // namespace NVIDIA
} // namespace triton
} // namespace mlir

#endif // TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_UTILITY_H
