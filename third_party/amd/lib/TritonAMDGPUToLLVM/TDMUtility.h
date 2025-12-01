#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include <optional>

using mlir::triton::AMD::TargetInfo;

namespace mlir::LLVM::AMD {

// Structure to hold TDM descriptor groups
struct TDMDescriptor {
  SmallVector<Value> group0;
  SmallVector<Value> group1;
  std::optional<SmallVector<Value>> group2;
  std::optional<SmallVector<Value>> group3;

  // Get all groups as a flat vector (for compatibility)
  SmallVector<Value> getAllGroups() const;
};

// Create a TDM descriptor. This creates a partially filled descriptor, with
// shared memory address and pred set to zero. User of the descriptor is
// expected to fill these fields later.
// For 1D-2D tensors: returns TDMDescriptor with only group0 and group1
// For 3D-5D tensors: returns TDMDescriptor with all groups populated
TDMDescriptor createTDMDescriptor(RewriterBase &rewriter, Location loc,
                                  const LLVMTypeConverter *typeConverter,
                                  Type elementType,
                                  SmallVector<int64_t> blockShape, int numWarps,
                                  unsigned padInterval, unsigned padAmount,
                                  SmallVector<Value> tensorShape,
                                  SmallVector<Value> tensorStride,
                                  Value srcPtr);

// Update the global memory address with offset, and fill the shared memory
// address and pred in a given TDM descriptor for >2D tensors.
void fillTDMDescriptor(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, int numWarps, unsigned padInterval,
    unsigned padAmount, SmallVector<Value> &group0, SmallVector<Value> &group1,
    std::optional<std::reference_wrapper<SmallVector<Value>>> group2,
    std::optional<std::reference_wrapper<SmallVector<Value>>> group3,
    SmallVector<Value> offset, Value dstPtr, Value pred, Value barrierPtr);

// Helper function to handle TDM operations for both load and store
void emitTDMOperation(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter,
                      ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                      int numWarps, unsigned padInterval, unsigned padAmount,
                      ArrayRef<Value> offset, Value dstPtr, Value pred,
                      Value multicastMask, Type elementType, Value barrierPtr,
                      bool isLoad, const triton::LinearLayout &cgaLayout,
                      Value ctaId);

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
