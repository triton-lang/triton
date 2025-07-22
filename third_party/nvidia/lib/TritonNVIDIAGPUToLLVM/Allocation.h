#ifndef TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_ALLOCATION_H
#define TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_ALLOCATION_H

#include "mlir/IR/Operation.h"

#include <functional>

namespace mlir {
namespace triton {
class TargetInfoBase;

namespace nvidia_gpu {
std::function<unsigned(Operation *)>
getNvidiaAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo);

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
#endif // TRITON_CONVERSION_TRITONNVIDIAGPU_TO_LLVM_ALLOCATION_H
