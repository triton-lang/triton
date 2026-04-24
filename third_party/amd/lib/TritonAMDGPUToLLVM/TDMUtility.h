#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <optional>

using mlir::triton::AMD::TargetInfo;
using PartitionedSharedEncodingAttr =
    mlir::triton::gpu::PartitionedSharedEncodingAttr;

namespace mlir::LLVM::AMD {

// Internal vector-grouped representation of a TDM descriptor.
//   group0: <4 x i32>
//   group1: <8 x i32>
//   group2: <4 x i32>  (only for 3D-5D)
//   group3: <4 x i32>  (only for 3D-5D)
// The MLIR-visible struct type remains flat {i32 × N} so the in-memory ABI
// matches the host-side TDMDescriptor struct in third_party/amd/backend/
// driver.c; the grouping here is a lowering-pass-internal convenience.
struct TDMDescriptor {
  Value group0;
  Value group1;
  std::optional<Value> group2;
  std::optional<Value> group3;

  // Return the group vectors as a flat list.  For 2D: {group0, group1};
  // for 3D-5D: {group0, group1, group2, group3}.
  SmallVector<Value> getAllGroups() const;
};

// Unpack the flat `{i32 × 12/20}` descriptor struct that
// `convertTensorDescType` returns (and that the host-side `TDMDescriptor` in
// driver.c serializes) into 2 or 4 vector groups suitable for
// `emitTDMLoadStore` / `emitTDMGatherScatter` / `emitTDMPrefetch`.
SmallVector<Value> unpackTDMDescriptor(RewriterBase &rewriter, Location loc,
                                       Value descStruct);

// Inverse of unpackTDMDescriptor for packing via `packLLElements`.  Flattens
// 2/4 vector groups back into 12/20 scalars.
SmallVector<Value> scalarizeTDMDescriptor(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> vectors);

// Create a TDM descriptor. This creates a partially filled descriptor, with
// shared memory address and pred set to zero. User of the descriptor is
// expected to fill these fields later.
// For 1D-2D tensors: returns TDMDescriptor with only group0 and group1
// For 3D-5D tensors: returns TDMDescriptor with all groups populated
// When encoding is a PartitionedSharedEncodingAttr, the warp distribution is
// adjusted so each wave's chunk fits in one LDS partition.
TDMDescriptor createTDMDescriptor(RewriterBase &rewriter, Location loc,
                                  const LLVMTypeConverter *typeConverter,
                                  Type elementType,
                                  SmallVector<int64_t> blockShape, int numWarps,
                                  unsigned padInterval, unsigned padAmount,
                                  SmallVector<Value> tensorShape,
                                  SmallVector<Value> tensorStride, Value srcPtr,
                                  Attribute sharedEncoding);

// Update the global memory address with offset, and fill the shared memory
// address and pred in a given TDM descriptor for regular load/store (1D-5D).
// For partitioned shared memory, dstPtrs contains multiple base pointers and
// the correct one is selected based on sharedLayout's partition dimension.
void fillTDMDescriptor(RewriterBase &rewriter, Location loc,
                       const LLVMTypeConverter *typeConverter, Type elementType,
                       SmallVector<int64_t> blockShape, int numWarps,
                       unsigned padInterval, unsigned padAmount, Value &group0,
                       Value &group1,
                       std::optional<std::reference_wrapper<Value>> group2,
                       std::optional<std::reference_wrapper<Value>> group3,
                       SmallVector<Value> offset, ArrayRef<Value> dstPtrs,
                       Value pred, Value multicastMask, Value barrierPtr,
                       const triton::LinearLayout &sharedLayout, Value ctaId,
                       bool isStore, ArrayRef<unsigned> warpsPerCTA);

// Fill TDM descriptor for gather/scatter operations (2D only).
// Gather reads from non-contiguous rows in global memory to LDS.
// Scatter writes from LDS to non-contiguous rows in global memory.
// - rowIndices: which global rows to read from (gather) or write to (scatter)
// - ldsRowOffset: starting row within shared memory
// - globalColOffset: starting column in global memory
// - use32BitIndices: true for 32-bit indices (max 8 rows), false for 16-bit
// (max 16 rows)
void fillTDMDescriptorForGatherScatter(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, unsigned padInterval, unsigned padAmount,
    Value &group0, Value &group1, Value &group2, Value &group3,
    Value ldsRowOffset, Value globalColOffset, Value ldsPtr, Value pred,
    Value barrierPtr, const triton::LinearLayout &cgaLayout, Value ctaId,
    ArrayRef<Value> rowIndices, bool use32BitIndices, bool isGather);

// Emit a TDM load or store for regular (non-scatter) contiguous transfers.
//
// Supports 1D-5D tensors.  When the encoding is a PartitionedSharedEncoding,
// the warp distribution is adjusted so each wave's tile fits within a single
// LDS partition.  If the warps cannot cover all logical pieces in one
// instruction, the operation is automatically split into multiple sequential
// TDM instructions.
//
// - offset: starting position in global memory for each dimension
// - dstPtrs: base pointers to LDS (multiple for partitioned encoding)
// - sharedLayout: linear layout for the full allocation shape (pre-CGA-split)
// - encoding: shared memory encoding (used for partition constraints when
//   splitting into multiple TDM instructions)
void emitTDMLoadStore(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter,
                      ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                      int numWarps, unsigned padInterval, unsigned padAmount,
                      ArrayRef<Value> offset, ArrayRef<Value> dstPtrs,
                      Value pred, Value multicastMask, Type elementType,
                      Value barrierPtr, bool isLoad,
                      const triton::LinearLayout &sharedLayout,
                      Attribute encoding, Value ctaId);

// Returns (warpsPerCTA, numTDMInstructions) for a given shared encoding.
// For PartitionedSharedEncodingAttr, computes a partition-aligned warp
// distribution.  For all other encodings, falls back to the default TDM warp
// distribution with numTDMInstructions = 1.
std::pair<SmallVector<unsigned>, unsigned>
distributeTDMWarpsAlignToPartition(ArrayRef<int64_t> blockShape, int numWarps,
                                   Attribute encoding);

// Calculate the number of TDM gather/scatter instructions needed using the
// same LinearLayout analysis as emitTDMGatherScatter: broadcasts are removed
// and contiguity is considered when batching indices per instruction.
size_t getTDMGatherScatterInstrinsicCount(RankedTensorType indicesType);

// Emit a TDM gather or scatter operation for non-contiguous row access.
// Gather: reads from non-contiguous global rows into LDS
// Scatter: writes from LDS to non-contiguous global rows
// - ldsPtr: pointer to shared memory (destination for gather, source for
// scatter)
// - rowIndices: which global rows to read from (gather) or write to (scatter)
// - colOffset: starting column offset in global memory
// - isGather: true for gather (global->LDS), false for scatter (LDS->global)
// - numWarps: number of warps in the CTA (used for warp predication)
// - indicesType: the RankedTensorType of the index tensor. Used to derive
//   whether indices are 32-bit or 16-bit, detect redundant warps (via
//   getFreeVariableMasks), and compute per-warp LDS offsets.
// Multiple TDM instructions are issued automatically if more rows are needed.
void emitTDMGatherScatter(RewriterBase &rewriter, Location loc,
                          const LLVMTypeConverter *typeConverter,
                          ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                          unsigned padInterval, unsigned padAmount,
                          Value ldsPtr, Value pred, Type elementType,
                          Value barrierPtr,
                          const triton::LinearLayout &cgaLayout, Value ctaId,
                          ArrayRef<Value> rowIndices, Value colOffset,
                          bool isGather, int numWarps,
                          RankedTensorType indicesType);

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
