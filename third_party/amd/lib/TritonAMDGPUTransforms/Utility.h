#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_UTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_UTILITY_H_

#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

inline void dumpScheduleDebug(triton::CoarseSchedule &schedule,
                              const char *debugType, llvm::StringRef msg) {
  DEBUG_WITH_TYPE(debugType, {
    llvm::dbgs() << "\n[" << debugType << "]: " << msg << "\n";
    schedule.dump();
  });
}

// DFS the def chain of 'defValue' starting from 'consumer' and will return the
// minimum found when accumulating countFunc(op) for all non control flow ops
// between value and the consumer. This function will traverse through for loop
// iterations and to the outside of the loop to find all its producers.
//    CountOp(Operation*) should return the value to accumulate for the
//    operation
// Returns 0 if there is an error traversing the def chain
int deduceMinCountOnDefChain(Value defValue, Operation *consumerOp,
                             llvm::function_ref<int(Operation *)> countFunc);

// Returns a padded shared encoding minimizing bank conflicts for a dot
// operand. Note the CDNA4 path requires both dotOpEnc (parent MFMA encoding's
// instruction shape) and useAsyncCopy.
triton::gpu::PaddedSharedEncodingAttr
composePaddedLayout(const triton::AMD::TargetInfo &targetInfo, int opIdx,
                    unsigned vecWidth, triton::gpu::TensorOrMemDesc srcTy,
                    ArrayRef<unsigned> sharedOrder,
                    triton::gpu::DotOperandEncodingAttr dotOpEnc = {},
                    bool useAsyncCopy = false);

triton::gpu::SharedEncodingTrait
getEncodingFromDescriptor(Operation *op, RankedTensorType tensorType,
                          Value desc);

// Returns the given |inputValue|'s dot user result encoding and updates |opIdx|
// and |vecSize| with which dot operand |inputValue| is fed into if possible.
template <class T>
T getDotEncoding(Value inputValue, unsigned *opIdx, unsigned *vecSize,
                 T *dummy = nullptr) {
  if (!llvm::hasSingleElement(inputValue.getUses()))
    return nullptr;

  Operation *user = *inputValue.getUsers().begin();
  if (user->getNumResults() != 1 ||
      user->getBlock() != inputValue.getParentBlock())
    return nullptr;

  if (auto dotOp = dyn_cast<triton::DotOpInterface>(user)) {
    OpOperand &use = *inputValue.getUses().begin();
    *opIdx = use.getOperandNumber();
    auto operandType = cast<RankedTensorType>(inputValue.getType());
    *vecSize =
        triton::gpu::toLinearLayout(operandType).getNumConsecutiveInOut();
    auto dotType = cast<RankedTensorType>(dotOp->getResult(0).getType());
    return dyn_cast<T>(dotType.getEncoding());
  }
  return getDotEncoding<T>(user->getResult(0), opIdx, vecSize);
}

#endif
