#ifndef TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_
#define TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_

#include <functional>
#include <optional>
#include <tuple>

namespace mlir {
class OpBuilder;
class Operation;

namespace scf {
class ForOp;
}
namespace triton::nvidia_gpu {
class MMAv5OpInterface;
class TMEMAllocOp;
class TMEMLoadOp;

// Returns the TMEMAllocOp and TMEMLoadOp that are used to allocate and load the
// accumulator for the given MMA operation. The TMEMAllocOp and TMEMLoadOp must
// be in the same region as the MMA operation.
std::optional<std::pair<TMEMAllocOp, TMEMLoadOp>>
getTMemAllocAndLoad(MMAv5OpInterface mmaOp);
// Create a new TMEMAllocOp to use for the pipelined MMA operation. It is
// optionally multi-buffered based on the number of stages.
TMEMAllocOp createTMemAlloc(OpBuilder &builder, TMEMAllocOp oldTMemAllocOp,
                            bool multiBufferred, int numStages);

// Return true if operands of the MMA operation are/are going to be pipelined
// and multibuffered, enabling the MMA operation to be pipelined.
bool mmaHasPipelineableOperands(
    MMAv5OpInterface mma, scf::ForOp forOp,
    std::function<bool(Operation *)> isLoadPipelineable);

// Return true if the loop has a read-modify-write access to the accumulator.
bool hasAccReadModifyWrite(MMAv5OpInterface mma, scf::ForOp forOp);
} // namespace triton::nvidia_gpu
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_
