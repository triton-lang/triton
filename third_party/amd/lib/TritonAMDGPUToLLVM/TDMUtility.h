#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>

using mlir::triton::AMD::TargetInfo;
using PartitionedSharedEncodingAttr =
    mlir::triton::gpu::PartitionedSharedEncodingAttr;

namespace mlir::LLVM::AMD {

// Decoded form of a verifier-validated axis-aligned `warp_used_hint`.
// Active set = { i0 ^ x : x in span(basisBits) }.
struct WarpHintInfo {
  unsigned K = 0;            // popcount(hint), power of two
  uint32_t i0 = 0;           // smallest active warp index
  SmallVector<int32_t, 5> basisBits; // log2(K) distinct positions in [0, log2(numWarps))
};

// Decode a verifier-validated `warp_used_hint`.
WarpHintInfo extractWarpHintInfo(uint32_t hint, int numWarps);

// Internal vector-grouped TDM descriptor (lowering-pass-only; the MLIR-
// visible struct stays flat {i32 x N} to match the host-side TDMDescriptor
// in third_party/amd/backend/driver.c).
//   group0/1: <4 x i32> / <8 x i32> (always)
//   group2/3: <4 x i32> each (3D-5D only)
struct TDMDescriptor {
  Value group0;
  Value group1;
  std::optional<Value> group2;
  std::optional<Value> group3;

  // Flatten to {g0,g1} (2D) or {g0,g1,g2,g3} (3D-5D).
  SmallVector<Value> getAllGroups() const;
};

// Unpack the flat {i32 x 12/20} descriptor struct (from convertTensorDescType
// / host-side TDMDescriptor in driver.c) into 2 or 4 vector groups for
// emitTDMLoadStore / emitTDMGatherScatter / emitTDMPrefetch.
SmallVector<Value> unpackTDMDescriptor(RewriterBase &rewriter, Location loc,
                                       Value descStruct);

// Inverse of unpackTDMDescriptor: 2/4 vector groups back to 12/20 scalars.
SmallVector<Value> scalarizeTDMDescriptor(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> vectors);

// Create a TDM descriptor. This creates a partially filled descriptor, with
// shared memory address and pred set to zero. User of the descriptor is
// expected to fill these fields later.
// For 1D-2D tensors: returns TDMDescriptor with only group0 and group1
// For 3D-5D tensors: returns TDMDescriptor with all groups populated
TDMDescriptor createTDMDescriptor(RewriterBase &rewriter, Location loc,
                                  const LLVMTypeConverter *typeConverter,
                                  Type elementType, size_t numDims,
                                  unsigned padInterval, unsigned padAmount,
                                  SmallVector<Value> tensorShape,
                                  SmallVector<Value> tensorStride,
                                  Value srcPtr);

// Fill the dst/pred fields of a TDM descriptor for regular load/store (1D-5D).
// Partitioned dst: `dstPtrs` holds per-partition bases, picked by partitionDim.
// With `warpHint`, K identity rows are placed at `warpHint->basisBits` and
// `warpId` is XOR-anchored by `warpHint->i0`; otherwise basis = {0..log2K-1}.
void fillTDMDescriptor(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> blockShape, int numWarps, unsigned padInterval,
    unsigned padAmount, Value &group0, Value &group1,
    std::optional<std::reference_wrapper<Value>> group2,
    std::optional<std::reference_wrapper<Value>> group3,
    SmallVector<Value> offset, ArrayRef<Value> dstPtrs, Value pred,
    Value multicastMask, Value barrierPtr,
    const triton::LinearLayout &sharedLayout, Value ctaId, bool isStore,
    ArrayRef<unsigned> warpsPerCTA,
    const std::optional<WarpHintInfo> &warpHint = std::nullopt);

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

// Emit a TDM load/store for regular contiguous transfers (1D-5D).
// PartitionedSharedEncoding aligns warps to LDS partitions; without a hint
// the op auto-splits into multiple instructions when warps don't cover all
// pieces, while a hint guarantees single-instruction emission (verifier).
// `warpUsedHint`: see TritonAMDGPUOps.td.
void emitTDMLoadStore(RewriterBase &rewriter, Location loc,
                      const LLVMTypeConverter *typeConverter,
                      ArrayRef<Value> desc, ArrayRef<int64_t> blockShape,
                      int numWarps, unsigned padInterval, unsigned padAmount,
                      ArrayRef<Value> offset, ArrayRef<Value> dstPtrs,
                      Value pred, Value multicastMask, Type elementType,
                      Value barrierPtr, bool isLoad,
                      const triton::LinearLayout &sharedLayout,
                      Attribute encoding, Value ctaId, int32_t auxBits,
                      std::optional<uint32_t> warpUsedHint = std::nullopt);

// ---------------------------------------------------------------------------
// Implicit op-merging support.
//
// `computeTDMMergeGroups` runs once at TDM->LLVM entry and builds a side-
// table from each merging `async_tdm_copy_global_to_local` op to its group
// info (IR unchanged).  The conversion pattern dispatches on it: the first
// visited member emits a fused intrinsic via `emitTDMLoadStoreMerged` and
// erases the whole group; singletons fall back to `emitTDMLoadStore`.
//
// Mergeability rules (v1; all required):
//   1. Every member has a verifier-legal `warp_used_hint`; hint-less ops
//      flush the in-flight batch.
//   2. No member has an `mbarrier` (fused intrinsic can't encode it);
//      mbarrier-carrying ops flush.
//   3. Members share K = popcount(hint), have pairwise-disjoint hints,
//      and their union is itself an axis-aligned coset.
//   4. Group size N is a power of two >= 2.
//   5. Members are consecutive in one block; pure ops thread through,
//      side-effecting non-TDM ops flush.
//   6. Results have structurally equal MemDescType and pairwise-distinct
//      SSA destinations.
//   7. Members share the same `cache` modifier (one auxBits on the fused
//      intrinsic).
//
// To pair an mbarrier with hinted partial copies, bracket the fusable
// batch with an mbarrier-carrying singleton.
struct TDMMergeGroupInfo {
  SmallVector<Operation *> members;  // program order; |members| = N
  uint32_t unionHint = 0;            // bit-OR of member hints; itself a coset
  WarpHintInfo unionInfo;            // decoded `unionHint` vs module numWarps
};

// Walk `mod` and identify all merge groups; ops not in any group are
// absent from the result.
llvm::DenseMap<Operation *, TDMMergeGroupInfo>
computeTDMMergeGroups(ModuleOp mod);

// Emit one fused TDM intrinsic for a merge group: build per-member
// descriptors via `fillTDMDescriptor` (each with its own hint/dst/idx/
// pred), then `select` between them on an SGPR-uniform per-wave selector
// derived from the union hint.  Shared `sharedLayout`/`encoding`/
// `auxBits` come from any member (mergeability makes them uniform);
// no barrier (rule 2).
void emitTDMLoadStoreMerged(RewriterBase &rewriter, Location loc,
                            const LLVMTypeConverter *typeConverter,
                            ArrayRef<SmallVector<Value>> descPerMember,
                            ArrayRef<int64_t> blockShape, int numWarps,
                            unsigned padInterval, unsigned padAmount,
                            ArrayRef<SmallVector<Value>> offsetPerMember,
                            ArrayRef<SmallVector<Value>> dstPtrsPerMember,
                            ArrayRef<Value> predPerMember, Value multicastMask,
                            Type elementType, bool isLoad,
                            const triton::LinearLayout &sharedLayout,
                            Attribute encoding, Value ctaId, int32_t auxBits,
                            const TDMMergeGroupInfo &groupInfo);

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
