#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

using mlir::triton::AMD::TargetInfo;

namespace mlir::LLVM::AMD {

// Create a TDM descriptor, divided into 2 group vectors.
std::pair<SmallVector<Value>, SmallVector<Value>>
createTDMDescriptor(RewriterBase &rewriter, Location loc,
                    const LLVMTypeConverter *typeConverter, Type elementType,
                    SmallVector<int64_t> blockShape, int numWarps,
                    unsigned padInterval, unsigned padAmount,
                    SmallVector<Value> tensorShape,
                    SmallVector<Value> tensorStride, Value srcPtr);

// Fill a TDM descriptor with offset, shared memory address, and pred.
void fillTDMDescriptor(RewriterBase &rewriter, Location loc,
                       const LLVMTypeConverter *typeConverter, Type elementType,
                       SmallVector<int64_t> blockShape, int numWarps,
                       unsigned padInterval, unsigned padAmount,
                       SmallVector<Value> &group0, SmallVector<Value> &group1,
                       SmallVector<Value> offset, Value dstPtr, Value pred);

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
