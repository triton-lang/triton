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
  // Active warp count (popcount(hint), always a power of two).
  unsigned K = 0;
  // Anchor offset: lsb(hint).  Active set is { i0 ^ x : x subset of basis
  // span }.
  uint32_t i0 = 0;
  // Bit positions of the basis vectors (each a power of two).  Size =
  // log2(K), all entries < log2(numWarps), all distinct.
  SmallVector<int32_t, 5> basisBits;
};

// Decode an axis-aligned `warp_used_hint` (as validated by
// `AsyncTDMCopyGlobalToLocalOp::validateWarpUsedHint`) for the given
// `numWarps`.  The returned info is sufficient to drive
// `getTDMLinearLayout`, `fillTDMDescriptor`, and the per-wave
// predication / piece mapping in the lowering.
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
// When `warpHint` is provided (i.e. the op carried a `warp_used_hint`),
// the warp sublayout's identity rows are placed at `warpHint->basisBits`
// and `warpId` is XOR-anchored by `warpHint->i0` before applying the
// layout's free-variable-mask predication.  Without a hint the layout uses
// the canonical-prefix placement (lowest log2K bits), matching the
// pre-hint behavior.
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

// Emit a TDM load or store for regular (non-scatter) contiguous transfers.
//
// Supports 1D-5D tensors.  When the encoding is a PartitionedSharedEncoding,
// the warp distribution is adjusted so each wave's tile fits within a single
// LDS partition.  Without `warpUsedHint`, if the warps cannot cover all
// logical pieces in one instruction the op is automatically split into
// multiple sequential TDM instructions.  With `warpUsedHint`, the verifier
// requires the resulting warpsPerCTA[partitionDim] >= numLogicalPieces, so
// a single instruction always suffices (multi-instruction slicing is not
// supported in that path).
//
// - offset: starting position in global memory for each dimension
// - dstPtrs: base pointers to LDS (multiple for partitioned encoding)
// - sharedLayout: linear layout for the full allocation shape (pre-CGA-split)
// - encoding: shared memory encoding (used for partition constraints when
//   splitting into multiple TDM instructions)
// - warpUsedHint: when present, an i32 bitmask whose active set is an
//   axis-aligned coset (verifier-checked, see TritonAMDGPUOps.td) with
//   `K = popcount` warps replacing `numWarps` for the warp distribution.
//   Inactive warps issue hardware no-ops via the layout's free-variable
//   mask plus an XOR by `i0`.
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
// The merge analysis runs once at the start of TDM->LLVM conversion and
// returns a side-table mapping each `amdgpu.async_tdm_copy_global_to_local`
// op that participates in a merge group to its group descriptor.  The IR
// is *not* modified.  The conversion pattern then queries this map: the
// first visited member emits a single fused intrinsic via
// `emitTDMLoadStoreMerged` and erases the whole group.  Singletons (not in
// the map) lower as today via `emitTDMLoadStore`.
//
// Mergeability rules (v1) — every clause is required:
//   1. Each candidate member carries a verifier-legal `warp_used_hint`
//      (an axis-aligned coset).  An op without a hint is *not* a
//      candidate; it also acts as a side-effecting boundary that
//      flushes any in-flight candidate run.
//   2. *No member carries an `mbarrier` operand.*  An mbarrier-carrying
//      TDM copy implies a hardware barrier semantic that the merged
//      lowering does not encode; such ops are excluded from candidates
//      and act as a flush boundary just like rule 1.
//   3. Every member has the same active-warp count `K = popcount(hint)`,
//      the per-member hints are pairwise disjoint, and their union is
//      itself a verifier-legal axis-aligned coset.
//   4. Group size `N` is a power of two with `N >= 2`.
//   5. Members are consecutive in the same basic block, modulo
//      memory-effect-free ops (arith/index math) which thread through.
//      Any side-effecting non-TDM op (async_wait, barrier, memref ops,
//      ...) flushes the run.
//   6. Every member's *result* `MemDescType` is structurally equal
//      (same shared-memory encoding, same shapePerCTA) and the SSA
//      destination values are pairwise distinct.
//   7. Every member carries the same `cache` modifier.  Differing cache
//      modifiers flush the run.
//
// Front-ends that need a hardware barrier alongside a hinted partial
// copy must therefore separate the two: one mbarrier-carrying TDM copy
// (lowered as a singleton) bracketing a run of hint-only copies that
// can fuse.
struct TDMMergeGroupInfo {
  // Members in program order; size N >= 2, N a power of two.
  SmallVector<Operation *> members;
  // Bit-OR of every member's `warp_used_hint`.  Itself a verifier-legal
  // axis-aligned coset.
  uint32_t unionHint = 0;
  // Decoded form of `unionHint` against the module's numWarps.
  WarpHintInfo unionInfo;
};

// Walk `mod` and identify all merge groups according to the rules
// documented above.
//
// Returned map is keyed on op pointer; every op in a group maps to the
// *same* `TDMMergeGroupInfo` instance via shared_ptr-style ownership
// (here: each entry holds the same value-type info).  Ops not in any
// group are absent from the map.
llvm::DenseMap<Operation *, TDMMergeGroupInfo>
computeTDMMergeGroups(ModuleOp mod);

// Emit a single fused TDM intrinsic for a merge group.  Each member's
// descriptor is built independently (`fillTDMDescriptor` per member,
// using the member's own hint, dst, indices, pred), then the resulting
// per-member packed-descriptor lanes are combined via SGPR-uniform
// `select` chains keyed on a per-wave selector derived from the union
// hint.  The intrinsic is emitted at the rewriter's current insertion
// point (which the caller should set to immediately before the first
// member).
//
// `descPerMember[i]` is the lowered TDM descriptor for member i (the
// per-tensor part returned by `createTDMDescriptor`); `dstPtrsPerMember[i]`
// likewise.  All other parameters mirror `emitTDMLoadStore` and must be
// consistent across the group (mergeability enforces this for
// shared-memory encoding, padding, shapePerCTA, and cache modifier; the
// caller provides one shared `sharedLayout`/`encoding`/`auxBits` for the
// group).  Per-member `warp_used_hint` values are read directly from
// `groupInfo.members`, and per-member barrier pointers are *not* a
// parameter: mergeability guarantees no member carries an mbarrier, so
// each `fillTDMDescriptor` is invoked with a null barrier.
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
