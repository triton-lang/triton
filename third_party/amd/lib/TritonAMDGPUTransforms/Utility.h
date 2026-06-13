#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_UTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_UTILITY_H_

#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "mlir/IR/Builders.h"
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
composePaddedLayout(const triton::amdgpu::TargetFeatures &targetFeatures,
                    int opIdx, unsigned vecWidth,
                    triton::gpu::TensorOrMemDesc srcTy,
                    ArrayRef<unsigned> sharedOrder,
                    triton::gpu::DotOperandEncodingAttr dotOpEnc = {},
                    bool useAsyncCopy = false);

// Returns null when useAsyncCopy is false or when the operand's MFMA shape,
// kWidth, or element bitwidth fall outside the supported set.
triton::gpu::PaddedSharedEncodingAttr composePaddedLayoutForAsyncCopyCDNA4(
    triton::gpu::DotOperandEncodingAttr dotOpEnc,
    triton::gpu::TensorOrMemDesc srcTy, ArrayRef<unsigned> sharedOrder,
    bool useAsyncCopy, unsigned warpSize);

triton::gpu::SharedEncodingTrait
getEncodingFromDescriptor(Operation *op, RankedTensorType tensorType,
                          Value desc);

// Build the index encoding for TDM gather/scatter.
//
// Layout: BlockedLayout([1, M], [threadsPerWarp, 1], [1, numWarps], [0, 1])
// sliced along dim 0 to produce a 1D encoding. M is the max number of row
// indices per TDM instruction (256 bits / index element bitwidth). The
// freeVarMasks mechanism in the LLVM lowering adapts the number of active
// warps and gathers per warp to the actual problem size.
triton::gpu::SliceEncodingAttr
getTDMGatherScatterIndexEncoding(Operation *op, RankedTensorType indicesType);

// Emit an amdg.update_tensor_descriptor that advances `desc` by `addOffsets`
// (with clamp_bounds) and sets `pred` when non-null.  Returns the new
// descriptor.
Value createUpdateTDMDescriptorOp(OpBuilder &builder, Location loc, Value desc,
                                  ValueRange addOffsets, Value pred);

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
