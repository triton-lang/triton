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

/// Return true if we can skip a barrier synchronization between two operations
/// even if they access the same shared memory.
bool canSkipBarSync(const membar::SMemAccess &source,
                    const membar::SMemAccess &sink, Allocation &allocation);
} // namespace NVIDIA
} // namespace triton
} // namespace mlir

#endif // TRITONGPU_CONVERSION_TRITONNVIDIAGPUTOLLVM_UTILITY_H
