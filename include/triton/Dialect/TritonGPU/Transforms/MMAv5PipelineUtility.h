#ifndef TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_
#define TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_

#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {

class OpBuilder;
class DominanceInfo;

namespace scf {
class ForOp;
} // namespace scf
namespace triton::nvidia_gpu {

//===----------------------------------------------------------------------===//
// MMA Pipeline Analysis
//===----------------------------------------------------------------------===//

// Returns the TMEMAllocOp and TMEMLoadOp that are used to allocate and load the
// accumulator for the given MMA operation. The TMEMAllocOp and TMEMLoadOp must
// be in the same region as the MMA operation.
std::optional<std::pair<TMEMAllocOp, TMEMLoadOp>>
getTMemAllocAndLoad(MMAv5OpInterface mmaOp);
// Given an MMAv5 operation in a loop, determine if its accumulator can be
// multibuffered.
bool isAccMultibufferingPossible(MMAv5OpInterface mma, scf::ForOp forOp);
// Only pipeline the loops where the MMA happens before the tmem_load, or is in
// the same stage as the tmem_load. Lowering does not support the case where the
// MMA is in a different stage as the tmem_load and happens after it.
bool mmav5DominatesTmemLoads(
    scf::ForOp forOp, function_ref<bool(MMAv5OpInterface)> isMmaPipelineable);

//===----------------------------------------------------------------------===//
// MMA Pipeline Rewriters
//===----------------------------------------------------------------------===//

// Create a new TMEMAllocOp to use for the pipelined MMA operation. It is
// optionally multi-buffered based on the number of stages.
TMEMAllocOp createTMemAlloc(OpBuilder &builder, TMEMAllocOp oldTMemAllocOp,
                            bool multiBufferred, int numStages);

// Return true if operands of the MMA operation are/are going to be pipelined
// and multibuffered, enabling the MMA operation to be pipelined.
bool mmaHasPipelineableOperands(
    MMAv5OpInterface mma, scf::ForOp forOp,
    std::function<bool(Operation *)> isLoadPipelineable);

// Return true if the accumulator of an mma in subsequent iterations is either
// independent from the previous iteration (overwritten) or completely reused,
// without read-modify-write.
// Otherwise, we can not pipeline the MMA, as we need to insert a wait after the
// mma to read back the accumulator for RMW.
bool hasAccReadModifyWrite(MMAv5OpInterface mma, scf::ForOp forOp);

} // namespace triton::nvidia_gpu
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_
