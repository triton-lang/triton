#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

using mlir::triton::AMD::TargetInfo;

namespace mlir::LLVM::AMD {

// Create a TDM descriptor, divided into 2 groups.
std::pair<Value, Value> createTDMDescriptor(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, SmallVector<Value> tensorShape,
    SmallVector<Value> tensorStride, SmallVector<Value> tensorOffset,
    Value srcPtr, Value dstPtr, Value pred, int numWarps, unsigned padInterval,
    unsigned padAmount);

// Pack base pointer, shape, and stride from a tensor descriptor into a single
// llvm struct value.
Value packTensorDesc(RewriterBase &rewriter, Location loc,
                     const LLVMTypeConverter *typeConverter, Value base,
                     ValueRange tensorShape, ValueRange tensorStride,
                     Type resultTy);

// Unpack a tensor descriptor from a single llvm struct value into
// (base, [shape0, shape1, ...], [stride0, stride1, ...]).
std::tuple<Value, SmallVector<Value>, SmallVector<Value>>
unpackTensorDesc(RewriterBase &rewriter, Location loc, Value desc);

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
