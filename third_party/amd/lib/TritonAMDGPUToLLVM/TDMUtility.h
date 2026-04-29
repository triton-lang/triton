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
  // Mask of the warpId bits that are NOT in the basis span and are
  // within the live num_warps range.  An active-warp test is
  // `((warpId ^ i0) & freeMask) == 0`.
  uint32_t freeMask = 0;
};

// Decode an axis-aligned `warp_used_hint` (as validated by
// `AsyncTDMCopyGlobalToLocalOp::validateWarpUsedHint`) for the given
// `numWarps`.  The returned info is sufficient to drive
// `getTDMLinearLayout`, `fillTDMDescriptor`, and the per-wave
// predication / piece mapping in the lowering.
WarpHintInfo extractWarpHintInfo(uint32_t hint, int numWarps);

// Structure to hold TDM descriptor groups
struct TDMDescriptor {
  SmallVector<Value> group0;
  SmallVector<Value> group1;
  std::optional<SmallVector<Value>> group2;
  std::optional<SmallVector<Value>> group3;

  // Get all groups as a flat vector (for compatibility)
  SmallVector<Value> getAllGroups() const;
};

// Create a TDM descriptor.  This builds the per-tensor portion of the
// descriptor (base ptr, tensor shape, stride, padding, data size) shared by
// every TDM op against the same `tensor_descriptor`.  pred, lds address, and
// the per-instruction `tile_dim*` fields are intentionally not encoded here
// and are filled in by `fillTDMDescriptor` (or
// `fillTDMDescriptorForGatherScatter`) at the actual load/store site, where
// the per-op warp distribution is known.
//
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
// address, pred, and per-instruction tile_dim* fields in a given TDM
// descriptor for regular load/store (1D-5D).  For partitioned shared memory,
// dstPtrs contains multiple base pointers and the correct one is selected
// based on sharedLayout's partition dimension.
//
// The per-warp tile shape is computed as `shapePerCTA / warpsPerCTA`.
// When `warpHint` is provided (i.e. the op carried a `warp_used_hint`),
// the warp sublayout's identity rows are placed at `warpHint->basisBits`
// and the active-warp test is `((warpId ^ i0) & freeMask) == 0`.  Without
// a hint the layout uses the canonical-prefix placement (lowest log2K
// bits) and predicates on `(warpId & freeMask) == 0`, matching the
// pre-hint behavior.
void fillTDMDescriptor(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter, Type elementType,
    SmallVector<int64_t> shapePerCTA, unsigned padInterval, unsigned padAmount,
    SmallVector<Value> &group0, SmallVector<Value> &group1,
    std::optional<std::reference_wrapper<SmallVector<Value>>> group2,
    std::optional<std::reference_wrapper<SmallVector<Value>>> group3,
    SmallVector<Value> offset, ArrayRef<Value> dstPtrs, Value pred,
    Value multicastMask, Value barrierPtr,
    const triton::LinearLayout &sharedLayout, Value ctaId, bool isStore,
    ArrayRef<unsigned> warpsPerCTA, int numWarps,
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
    SmallVector<Value> &group0, SmallVector<Value> &group1,
    SmallVector<Value> &group2, SmallVector<Value> &group3, Value ldsRowOffset,
    Value globalColOffset, Value ldsPtr, Value pred, Value barrierPtr,
    const triton::LinearLayout &cgaLayout, Value ctaId,
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
                      Attribute encoding, Value ctaId,
                      std::optional<uint32_t> warpUsedHint = std::nullopt);

// ---------------------------------------------------------------------------
// Implicit op-merging support.
//
// The merge analysis runs once at the start of TDM->LLVM conversion and
// returns a side-table mapping each `amdgpu.async_tdm_copy_global_to_local`
// op that participates in a merge group to its group descriptor.  The IR
// is *not* modified.  The conversion pattern then queries this map: the
// first member emits a single fused intrinsic via `emitTDMLoadStoreMerged`,
// while non-first members emit no intrinsic and are erased by the rewriter.
// Singletons (not in the map) lower as today via `emitTDMLoadStore`.
struct TDMMergeGroupInfo {
  // Members in program order; size N >= 2, N a power of two.
  SmallVector<Operation *> members;
  // Bit-OR of every member's `warp_used_hint`.  Itself a verifier-legal
  // axis-aligned coset.
  uint32_t unionHint = 0;
  // Decoded form of `unionHint` against the module's numWarps.
  WarpHintInfo unionInfo;
};

// Walk `mod` and identify all merge groups according to the mergeability
// rules: axis-aligned closure of every member, pairwise disjoint hints,
// union legal, N power of two, no `mbarrier` operand, consecutive in
// program order, non-overlapping destinations, same shared-memory
// encoding + same shapePerCTA.
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
// shared-memory encoding, padding, and shapePerCTA; the caller provides
// one shared `sharedLayout`/`encoding` for the group).
void emitTDMLoadStoreMerged(
    RewriterBase &rewriter, Location loc,
    const LLVMTypeConverter *typeConverter,
    ArrayRef<SmallVector<Value>> descPerMember,
    ArrayRef<int64_t> blockShape, int numWarps, unsigned padInterval,
    unsigned padAmount,
    ArrayRef<SmallVector<Value>> offsetPerMember,
    ArrayRef<SmallVector<Value>> dstPtrsPerMember,
    ArrayRef<Value> predPerMember, Value multicastMask, Type elementType,
    ArrayRef<Value> barrierPtrPerMember, bool isLoad,
    const triton::LinearLayout &sharedLayout, Attribute encoding,
    Value ctaId, ArrayRef<uint32_t> hintPerMember,
    const TDMMergeGroupInfo &groupInfo);

// Returns (warpsPerCTA, numTDMInstructions) for a given shared encoding.
// For PartitionedSharedEncodingAttr, computes a partition-aligned warp
// distribution.  For all other encodings, falls back to the default TDM warp
// distribution with numTDMInstructions = 1.
std::pair<SmallVector<unsigned>, unsigned>
distributeTDMWarpsAlignToPartition(ArrayRef<int64_t> blockShape, int numWarps,
                                   Attribute encoding);

// Calculate the number of TDM gather/scatter instructions needed.
// - numIndices: number of row indices
// - use32BitIndices: true for 32-bit indices (max 8 rows/instr), false for
//   16-bit (max 16 rows/instr)
// Returns: the number of TDM instructions that will be emitted
size_t getTDMGatherScatterInstrinsicCount(size_t numIndices,
                                          bool use32BitIndices);

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
