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

// Given an MMAv5 operation in a loop, determine if its accumulator can be
// multibuffered.
bool isAccMultibufferingPossible(MMAv5OpInterface mma, scf::ForOp forOp);

// Returns true if the MMA operation requires acc multi-buffering when
// pipelined.
bool requiresAccMultiBuffering(MMAv5OpInterface mma, scf::ForOp forOp);

// Helper class to determine if the operands of an MMA operation are
// pipelineable.
class MMAv5PipelineableOperandsHelper {
public:
  MMAv5PipelineableOperandsHelper(
      MMAv5OpInterface mmaOp, scf::ForOp forOp,
      std::function<bool(Operation *)> isLoadPipelineable)
      : mmaOp(mmaOp), forOp(forOp), isLoadPipelineable(isLoadPipelineable) {
    run();
  }
  bool isPipelineable = false;
  bool isOperandsStateDetermined = false;
  SmallVector<Operation *> unpipelineableOperandLoads;

private:
  MMAv5OpInterface mmaOp;
  scf::ForOp forOp;
  std::function<bool(Operation *)> isLoadPipelineable;
  bool comesFromLoadOrOutsideLoop(Value v, Operation *&foundLoad);
  void run();
};

//===----------------------------------------------------------------------===//
// MMA Pipeline Rewriters
//===----------------------------------------------------------------------===//

// Create a new TMEMAllocOp to use for the pipelined MMA operation. It is
// optionally multi-buffered based on the number of stages.
TMEMAllocOp createTMemAlloc(OpBuilder &builder, TMEMAllocOp oldTMemAllocOp,
                            bool multiBufferred, int numStages);

// Return true if the accumulator of an mma in subsequent iterations is either
// independent from the previous iteration (overwritten) or completely reused,
// without read-modify-write.
// Otherwise, we can not pipeline the MMA, as we need to insert a wait after the
// mma to read back the accumulator for RMW.
bool hasAccReadModifyWrite(MMAv5OpInterface mma, scf::ForOp forOp);

} // namespace triton::nvidia_gpu
} // namespace mlir

#endif // TRITON_TRITONGPU_TRANSFORMS_MMAV5PIPELINEUTILITY_H_
