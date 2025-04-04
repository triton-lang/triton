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
// MMAInfo
//===----------------------------------------------------------------------===//

// This struct contains analysis information about an MMAv5 operation inside a
// loop used for pipelining MMA ops.
struct MMAInfo {
  // This struct contains information about when the MMA's accumulator is
  // overridden in the loop, if it is at all.
  struct AccOverridePoint {
    // The operation which overrides the accumulator.
    Operation *op;
    // The condition on which the accumulator is reset.
    Value condition = nullptr;
    // The initial value of the accumulator and the value after a reset.
    Value initValue = nullptr;
    // The number of loop iterations ago the accumulator was reset.
    int distance = 0;
    // Whether the accumulator is reset via setting the `useAcc` flag to false
    // or by clearing the accumulator tensor value.
    bool isFlag = false;
  };

  // The TMEM allocation of the accumuator, which directly precedes the dot op.
  TMEMAllocOp accAlloc;
  // The TMEM load of the accumulator value out of TMEM, which directly follows
  // the dot op.
  TMEMLoadOp accLoad;
  // The override point of the accumulator value, if it is overriden in the
  // loop. E.g. this is typically present for persistent kernels.
  std::optional<AccOverridePoint> accDef;
  // If the accumulator is used in future iterations of the loop, this is the
  // iter arg number.
  std::optional<int> yieldArgNo;
  // Whether the accumulator needs to be multibuffered.
  bool accIsMultiBuffered;

  Value phase = nullptr;
  Value barrierIdx = nullptr;
  Value accInsertIdx = nullptr;
  Value accExtractIdx = nullptr;
  Value barrierAlloc = nullptr;
};

//===----------------------------------------------------------------------===//
// MMA Pipeline Analysis
//===----------------------------------------------------------------------===//

// Returns the TMEMAllocOp and TMEMLoadOp that are used to allocate and load the
// accumulator for the given MMA operation. The TMEMAllocOp and TMEMLoadOp must
// be in the same region as the MMA operation.
std::optional<std::pair<TMEMAllocOp, TMEMLoadOp>>
getTMemAllocAndLoad(MMAv5OpInterface mmaOp);
// Get immediate users of the accumulator within the current loop iteration.
SmallVector<Operation *> getDirectAccUses(TMEMLoadOp accDef);
// Analyze an MMA op inside a loop to determine information about how it can be
// pipelined. Returns `std::nullopt` if it cannot be pipelined.
std::optional<MMAInfo> getMMAInfo(scf::ForOp forOp, MMAv5OpInterface mmaOp,
                                  DominanceInfo &domInfo);

//===----------------------------------------------------------------------===//
// MMA Pipeline Rewriters
//===----------------------------------------------------------------------===//

// Create a new TMEMAllocOp to use for the pipelined MMA operation. It is
// optionally multi-buffered based on the number of stages.
TMEMAllocOp createTMemAlloc(OpBuilder &builder, TMEMAllocOp oldTMemAllocOp,
                            bool multiBufferred, int numStages);

// Create a store op of the initial value of the accumulator into the
// potentially multi-buffered accumulator.
void createInitStore(OpBuilder &builder, TMEMAllocOp allocOp, Value initVal,
                     bool multiBufferred);

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
