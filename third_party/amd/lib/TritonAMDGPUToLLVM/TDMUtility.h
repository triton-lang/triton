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

// Decoded form of an axis-aligned `warp_used_hint` bitmask.  Populated
// by `extractWarpHintInfo` from a verifier-validated hint, so all
// invariants are assumed to hold.
struct WarpHintInfo {
  // popcount(hint), always a power of two.
  unsigned K = 0;
  // Anchor: smallest active warp index.  Active set = { i0 ^ x : x in
  // span(basisBits) }.
  uint32_t i0 = 0;
  // Bit positions of the log2(K) basis vectors; distinct, all in
  // [0, log2(numWarps)).
  SmallVector<int32_t, 5> basisBits;
};

// Decode a verifier-validated axis-aligned `warp_used_hint`.  Drives
// `getTDMLinearLayout`, `fillTDMDescriptor`, and per-wave predication.
WarpHintInfo extractWarpHintInfo(uint32_t hint, int numWarps);

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
TDMDescriptor createTDMDescriptor(RewriterBase &rewriter, Location loc,
                                  const LLVMTypeConverter *typeConverter,
                                  Type elementType, size_t numDims,
                                  unsigned padInterval, unsigned padAmount,
                                  SmallVector<Value> tensorShape,
                                  SmallVector<Value> tensorStride,
                                  Value srcPtr);

// Update the global memory address with offset, and fill the shared memory
// address and pred in a given TDM descriptor for regular load/store (1D-5D).
// For partitioned shared memory, dstPtrs contains multiple base pointers and
// the correct one is selected based on sharedLayout's partition dimension.
//
// With `warpHint`, the warp sublayout's K identity rows are placed at
// `warpHint->basisBits` and `warpId` is XOR-anchored by `warpHint->i0`
// before the free-variable-mask predication.  Without a hint, basis
// bits default to {0, ..., log2K - 1} (the pre-hint placement).
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

// Emit a TDM load or store for regular (non-scatter) contiguous transfers
// (1D-5D).  For PartitionedSharedEncoding, the warp distribution is aligned
// to LDS partitions; without a hint the op auto-splits into multiple TDM
// instructions when warps can't cover all logical pieces.  With a hint the
// verifier guarantees a single instruction suffices.
//
// - offset / dstPtrs: per-dim global offsets / per-partition LDS bases
// - sharedLayout: linear layout for the full allocation (pre-CGA-split)
// - encoding: shared encoding, used for partition constraints
// - warpUsedHint: see TritonAMDGPUOps.td; `K = popcount(hint)` replaces
//   `numWarps`, inactive warps become no-ops via free-variable masking
//   plus an XOR by `i0`.
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
// The merge analysis runs once at TDM->LLVM conversion entry and returns
// a side-table from each `async_tdm_copy_global_to_local` op in a group
// to its group info; the IR itself is unchanged.  The conversion pattern
// dispatches on this map: the first visited member emits a fused
// intrinsic via `emitTDMLoadStoreMerged` and erases the whole group;
// singletons fall back to `emitTDMLoadStore`.
//
// Mergeability rules (v1; all required):
//   1. Every member carries a verifier-legal `warp_used_hint`.  Hint-less
//      ops aren't candidates and act as a flush boundary.
//   2. No member carries an `mbarrier` operand (the fused intrinsic
//      doesn't encode the barrier semantic).  Mbarrier-carrying ops are
//      flush boundaries.
//   3. Members share the same K = popcount(hint), have pairwise-disjoint
//      hints, and their union is itself an axis-aligned coset.
//   4. Group size N is a power of two with N >= 2.
//   5. Members are consecutive in the same block; pure (memory-effect-
//      free) ops thread through, any side-effecting non-TDM op flushes.
//   6. Results have structurally equal MemDescType (same encoding +
//      shapePerCTA) and pairwise-distinct SSA destinations.
//   7. Members share the same `cache` modifier (single auxBits on the
//      fused intrinsic).
//
// To combine an mbarrier with hinted partial copies, separate them: one
// mbarrier-carrying singleton bracketing a fusable batch.
struct TDMMergeGroupInfo {
  // Members in program order; size N >= 2, N a power of two.
  SmallVector<Operation *> members;
  // Bit-OR of every member's `warp_used_hint`.  Itself a verifier-legal
  // axis-aligned coset.
  uint32_t unionHint = 0;
  // Decoded form of `unionHint` against the module's numWarps.
  WarpHintInfo unionInfo;
};

// Walk `mod` and identify all merge groups (rules above).  Every member
// of a group maps to the same `TDMMergeGroupInfo` value; ops not in any
// group are absent from the map.
llvm::DenseMap<Operation *, TDMMergeGroupInfo>
computeTDMMergeGroups(ModuleOp mod);

// Emit a single fused TDM intrinsic for a merge group.  Per-member
// descriptors are built via `fillTDMDescriptor` (each with its own
// hint/dst/indices/pred), then combined via SGPR-uniform `select`
// chains keyed on a per-wave selector derived from the union hint.
//
// `descPerMember[i]` / `dstPtrsPerMember[i]` are the lowered descriptor
// and LDS bases for member i; per-member hints are read off
// `groupInfo.members[i]`.  The shared `sharedLayout`/`encoding`/
// `auxBits` come from any member (mergeability makes them uniform), and
// no per-member barrier is needed (rule 2).
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
