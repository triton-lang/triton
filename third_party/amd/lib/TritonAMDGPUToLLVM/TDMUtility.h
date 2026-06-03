#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTOLLVM_TDMUTILITY_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include <memory>
#include <optional>

using mlir::triton::AMD::TargetInfo;
using PartitionedSharedEncodingAttr =
    mlir::triton::gpu::PartitionedSharedEncodingAttr;

namespace mlir::LLVM::AMD {

// TDM descriptor groups (lowering-pass-only; the MLIR-visible struct stays
// flat {i32 x N} to match the host-side TDMDescriptor in driver.c):
//   groups[0]/[1]: <4 x i32> / <8 x i32> (always)
//   groups[2]/[3]: <4 x i32> each (3D-5D only)
// Size is 2 (1D-2D) or 4 (3D-5D); see unpackTDMDescriptor.

// Unpack the flat {i32 x 12/20} descriptor struct (from convertTensorDescType
// / host-side TDMDescriptor in driver.c) into 2 or 4 vector groups for
// emitTDMLoadStore / emitTDMGatherScatter / emitTDMPrefetch.
SmallVector<Value> unpackTDMDescriptor(RewriterBase &rewriter, Location loc,
                                       Value descStruct);

// Inverse of unpackTDMDescriptor: 2/4 vector groups back to 12/20 scalars.
SmallVector<Value> scalarizeTDMDescriptor(RewriterBase &rewriter, Location loc,
                                          ArrayRef<Value> vectors);

// Updates TDM descriptor fields in place.
// Mirrors the parameter semantics of `amdg.update_tensor_descriptor`.
//
// - addOffsets (incremental): bumps global_addr by
//   sum(addOffsets[i]*stride[i]) scaled by element size.  Empty = skip.
// - setBounds  (rewrite):     overwrites tensor_dim absolutely.  Empty = skip.
// - dest       (rewrite):     overwrites lds_addr.  Null = skip.
// - pred       (rewrite):     overwrites pred.  Null = skip.
// - barrier    (rewrite):     enables barrier signaling and writes barrier
//                             addr.  Null = skip.
//
// Currently 2D-only; 3D-5D support TBD.
void updateTensorDescriptor(RewriterBase &rewriter, Location loc,
                            Type elementType, ArrayRef<int64_t> blockShape,
                            Value &group0, Value &group1,
                            ArrayRef<Value> addOffsets,
                            ArrayRef<Value> setBounds, Value dest, Value pred,
                            Value barrier);

// Create the base TDM descriptor from tensor metadata: global base pointer,
// tensor shape, strides, and padding.  Fields that depend on a particular TDM
// operation (pred, LDS address, barrier, tile_dim*) are filled later by the
// per-op descriptor fillers.  Returns 2 vector groups for 1D-2D, 4 for 3D-5D.
SmallVector<Value> createTDMDescriptor(RewriterBase &rewriter, Location loc,
                                       const LLVMTypeConverter *typeConverter,
                                       Type elementType, size_t numDims,
                                       unsigned padInterval, unsigned padAmount,
                                       SmallVector<Value> tensorShape,
                                       SmallVector<Value> tensorStride,
                                       Value srcPtr);

// Fill the dst/pred fields of a TDM descriptor for regular load/store (1D-5D).
// `groups` is 2 (1D-2D) or 4 (3D-5D) vector entries, updated in place.
// Partitioned dst: `dstPtrs` holds per-partition bases, picked by partitionDim.
// `warpUsedHint`: see TritonAMDGPUOps.td for the axis-aligned hint rule.
void fillTDMDescriptor(RewriterBase &rewriter, Location loc,
                       const LLVMTypeConverter *typeConverter, Type elementType,
                       SmallVector<int64_t> blockShape, int numWarps,
                       unsigned padInterval, unsigned padAmount,
                       MutableArrayRef<Value> groups, SmallVector<Value> offset,
                       ArrayRef<Value> dstPtrs, Value pred, Value multicastMask,
                       Value barrierPtr,
                       const triton::LinearLayout &sharedLayout, Value ctaId,
                       bool isStore, ArrayRef<unsigned> warpsPerCTA,
                       std::optional<uint32_t> warpUsedHint = std::nullopt);

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
// TDM copy merging support.
//
// TDM-to-LLVM lowering can merge adjacent compatible copies whenever the copies
// already carry verifier-legal `warp_used_hint` masks.  This includes
// user-authored hints and hints created by `prepareGeneratedTDMMergeHints`.
//
// TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS (default off; set to 1 / "on" /
// "true" to disable auto-generation) gates ONLY the automatic generation of
// hints (`prepareGeneratedTDMMergeHints`).  When set, the compiler stops
// synthesizing hints for adjacent unhinted copies, but `computeTDMMergeGroups`
// is NOT gated: copies that already carry compatible `warp_used_hint` masks
// (user-authored or previously generated) still merge.  To keep a hinted copy
// standalone, make its hint overlap its neighbor's hint.
//
// `prepareGeneratedTDMMergeHints` is a narrower pre-pass: it mutates only the
// canonical adjacent unhinted indexed-destination form with non-partitioned
// destinations into hinted merge candidates.  It is not the full mergeability
// contract; it only creates hints for one safe IR shape.  Then
// `computeTDMMergeGroups` builds a map from each merging
// `async_tdm_copy_global_to_local` op to its group info (IR unchanged).  The
// conversion pattern dispatches on it: the first visited member emits a fused
// intrinsic via `emitTDMLoadStoreMerged` and erases the whole group; singletons
// fall back to `emitTDMLoadStore`.
//
// Mergeability rules (v1; all required):
//   1. Every member has a verifier-legal `warp_used_hint`; unhinted copies end
//      the current run.
//   2. No member carries an `mbarrier` (the fused intrinsic cannot encode one);
//      mbarrier-carrying copies end the current run.
//   3. Members have pairwise-disjoint hints.  Their union does not need to be a
//      valid `warp_used_hint`.  Members may have different K = popcount(hint).
//   4. Group size N is 2, 3, or 4.
//   5. Members are strictly consecutive in one block; any intervening op (TDM
//      or not) ends the current run.
//   6. Members have same-rank descriptors representable by a compatible
//      hardware descriptor group form for the fused intrinsic.
//   7. Members share the same `cache` modifier (one auxBits on the fused
//      intrinsic).
struct TDMMergeGroupInfo {
  // Members in program order. |members| = |memberHints| = N.
  SmallVector<Operation *> members;
  SmallVector<uint32_t> memberHints;
  // Last member in program order; anchors the fused intrinsic's insertion
  // point so all members dominate it.
  Operation *lastInProgramOrder = nullptr;
};

struct TDMMergeMemberInfo {
  SmallVector<int64_t> shapePerCTA;
  unsigned padInterval = 0;
  unsigned padAmount = 0;
  Type elementType;
  triton::LinearLayout sharedLayout;
  Attribute encoding;
  Value multicastMask;
  size_t numGroups = 0;
};

// Enabled by default (set TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS=1 to disable):
// mutate only canonical adjacent unhinted copies with indexed destinations of
// the form
//   memdesc_index A; async_tdm_copy A; memdesc_index B; async_tdm_copy B; ...
// into hinted merge candidates by moving the destination memdesc_index ops
// before the copy group and assigning disjoint warp_used_hint masks.  Partitioned
// destinations are skipped because their extra hint legality rule is verified
// before this pass runs.
// No-op when the env var disables auto-merge.
void prepareGeneratedTDMMergeHints(ModuleOp mod);

// Walk `mod` and identify all merge groups from copies that already carry
// compatible hints; ops not in any group are absent from the result.  Not
// gated by the auto-merge env var (it only controls hint generation),
// so user-authored compatible hints still merge when it is disabled.
//
// Each group's info is stored once and shared by all its members (the map value
// is a shared_ptr), so an N-member group keeps a single TDMMergeGroupInfo.
llvm::DenseMap<Operation *, std::shared_ptr<TDMMergeGroupInfo>>
computeTDMMergeGroups(ModuleOp mod);

// Emit one fused TDM intrinsic for a merge group. The conversion builds
// per-member descriptors and `select`s between them on an SGPR-uniform per-wave
// selector. `auxBits` comes from any member (rule 7 makes it uniform); no
// mbarrier (rule 2).
void emitTDMLoadStoreMerged(RewriterBase &rewriter, Location loc,
                            const LLVMTypeConverter *typeConverter,
                            ArrayRef<SmallVector<Value>> descPerMember,
                            ArrayRef<TDMMergeMemberInfo> memberInfo,
                            int numWarps,
                            ArrayRef<SmallVector<Value>> offsetPerMember,
                            ArrayRef<SmallVector<Value>> dstPtrsPerMember,
                            ArrayRef<Value> predPerMember, bool isLoad,
                            Value ctaId, int32_t auxBits,
                            const TDMMergeGroupInfo &groupInfo);

// Effective warp count that drives TDM warp distribution and the resulting
// physical instruction count.  A `warp_used_hint` restricts emission to
// K = popcount(hint) warps (the rest become hardware no-ops), so the
// distribution must be sized by K, not num_warps.  Both the lowering
// (`emitTDMLoadStore`) and the wait-count pass call this so the counted
// intrinsics cannot drift from the emitted ones.
int getTDMEffectiveWarps(int numWarps, std::optional<uint32_t> warpUsedHint);

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
