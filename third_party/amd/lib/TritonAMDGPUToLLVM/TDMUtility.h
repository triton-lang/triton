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
// address and pred in a given TDM descriptor for regular load/store (1D-5D).
void fillTDMDescriptor(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, int numWarps, unsigned padInterval,
    unsigned padAmount, SmallVector<Value> &group0, SmallVector<Value> &group1,
    std::optional<std::reference_wrapper<SmallVector<Value>>> group2,
    std::optional<std::reference_wrapper<SmallVector<Value>>> group3,
    SmallVector<Value> offset, Value dstPtr, Value pred, Value multicastMask,
    Value barrierPtr, const triton::LinearLayout &cgaLayout, Value ctaId);

// Fill TDM descriptor for scatter operation (2D only).
// Scatter writes data from LDS to non-contiguous rows in global memory.
// - rowIndices: which global rows to write to
// - ldsRowOffset: starting row within shared memory
// - globalColOffset: starting column in global memory
// - use32BitIndices: true for 32-bit indices (max 8 rows), false for 16-bit
// (max 16 rows)
void fillTDMDescriptorForScatter(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, SmallVector<Value> &group0,
    SmallVector<Value> &group1, SmallVector<Value> &group2,
    SmallVector<Value> &group3, Value ldsRowOffset, Value globalColOffset,
    Value ldsPtr, Value pred, Value barrierPtr,
    const triton::LinearLayout &cgaLayout, Value ctaId,
    ArrayRef<Value> rowIndices, bool use32BitIndices);

// Emit a TDM load or store operation for regular (non-scatter) transfers.
// Supports 1D-5D tensors with contiguous access patterns.
// - offset: the starting position in global memory for each dimension
// - dstPtr: pointer to shared memory for load, or source pointer for store
// - isLoad: true for global->LDS, false for LDS->global
void emitTDMLoadStore(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter,
                      ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                      int numWarps, unsigned padInterval, unsigned padAmount,
                      ArrayRef<Value> offset, Value dstPtr, Value pred,
                      Value multicastMask, Type elementType, Value barrierPtr,
                      bool isLoad, const triton::LinearLayout &cgaLayout,
                      Value ctaId);

// Emit a TDM scatter operation to write non-contiguous rows from LDS to global.
// - srcPtr: pointer to shared memory containing the source data
// - rowIndices: which global rows to write to
// - colOffset: starting column offset in global memory
// - use32BitIndices: true for 32-bit indices (max 8 rows/instr), false for
//   16-bit (max 16 rows/instr)
// Multiple TDM instructions are issued automatically if more rows are needed.
void emitTDMScatter(RewriterBase &rewriter, Location loc,
                    const LLVMTypeConverter *typeConverter,
                    ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                    Value srcPtr, Value pred, Type elementType,
                    Value barrierPtr, const triton::LinearLayout &cgaLayout,
                    Value ctaId, ArrayRef<Value> rowIndices, Value colOffset,
                    bool use32BitIndices);

// Emit prefetches for a TDM tile to make it available for an actual load in
// the future. Data is prefetched cooperatively across all CTAs, warps, and
// lanes to cover the entire TDM tile.
// Returns the prefetched memory offsets. This should only be used for testing
// purposes.
SmallVector<Value> emitTDMPrefetch(RewriterBase &rewriter, Location loc,
                                   ArrayRef<Value> desc,
                                   ArrayRef<int64_t> blockShape, int numLanes,
                                   int numWarps, int numCTAs,
                                   ArrayRef<Value> offset, Value pred,
                                   Type elementType, Value laneId, Value warpId,
                                   Value ctaId, bool isSpeculative);

} // namespace mlir::LLVM::AMD

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
