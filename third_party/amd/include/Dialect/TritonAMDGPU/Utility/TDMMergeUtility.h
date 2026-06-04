#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMMERGEUTILITY_H
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMMERGEUTILITY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <memory>

namespace mlir::triton::AMD {

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
// The grouping is decided once, early, by the `tritonamdgpu-prepare-tdm-merge`
// pass (`assignTDMMergeGroupIds`): it stamps hints on runs of already-adjacent
// unhinted, non-partitioned copies (gated by the env var), runs
// `computeTDMMergeGroups`, and freezes each member's group id/index as
// attributes.  Both later consumers -- the wait-count pass and the LLVM
// conversion -- call `readTDMMergeGroups` to recover that same map (IR
// unchanged), so they share one grouping.
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
//      or not) ends the current run.  Evaluated when the grouping is frozen;
//      later adjacency changes do not split or grow a group.
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

// Each group's info is stored once and shared by all its members, so an
// N-member group keeps a single TDMMergeGroupInfo.
using TDMMergeGroupMap =
    llvm::DenseMap<Operation *, std::shared_ptr<TDMMergeGroupInfo>>;

// Enabled by default (set TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS=1 to disable):
// stamp disjoint `warp_used_hint` masks on runs of already-adjacent unhinted
// copies so they fuse later.  Only attributes are added -- copies separated by
// any other op (including their own `memdesc_index` destinations) are left
// alone.  Partitioned destinations are skipped because their extra hint
// legality rule is verified before hint generation runs.  No-op when the env
// var disables auto-merge.  Invoked by `assignTDMMergeGroupIds`.
void prepareGeneratedTDMMergeHints(ModuleOp mod);

// Walk `mod` and identify all merge groups from copies that already carry
// compatible hints; ops not in any group are absent from the result.  Not
// gated by the auto-merge env var (it only controls hint generation),
// so user-authored compatible hints still merge when it is disabled.
TDMMergeGroupMap computeTDMMergeGroups(ModuleOp mod);

// Runs early, from the `tritonamdgpu-prepare-tdm-merge` pass: generate hints
// (gated by the env var), compute the merge groups, and freeze the decision by
// stamping each member with `amdgpu.tdm_merge_id` (a module-unique group id) and
// `amdgpu.tdm_merge_index` (its position in the group, program order).  The
// wait-count pass and the LLVM conversion both read this frozen grouping, so the
// counted intrinsics match the emitted ones even if a later pass perturbs copy
// adjacency.
void assignTDMMergeGroupIds(ModuleOp mod);

// Rebuild the merge-group map from the attributes stamped by
// `assignTDMMergeGroupIds`.  The wait-count pass and the LLVM conversion both
// call this to consume the one frozen grouping.  Each reconstructed group is
// validated (2..4 members, one block, contiguous indices, disjoint hints,
// hinted/mbarrier-free members, uniform rank/cache) to catch a stale or aliased
// `amdgpu.tdm_merge_id` before it turns into a bogus fused intrinsic.
TDMMergeGroupMap readTDMMergeGroups(ModuleOp mod);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMMERGEUTILITY_H
