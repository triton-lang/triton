#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_UTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_UTILITY_H_

#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;

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

#endif
