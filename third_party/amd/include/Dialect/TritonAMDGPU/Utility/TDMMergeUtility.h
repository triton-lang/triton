#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMMERGEUTILITY_H
#define TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMMERGEUTILITY_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <memory>

namespace mlir::triton::AMD {

// TDM copy merge groups are decided once by
// `tritonamdgpu-prepare-tdm-merge`, then stored as attributes so wait-count
// and LLVM lowering use the same grouping.  The pass may generate
// `warp_used_hint` masks for adjacent unhinted copies unless
// TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS is set.  Copies that already have
// compatible hints can still merge when auto-generation is disabled.
//
// Mergeability rules:
//   1. Members must have verifier-legal `warp_used_hint` masks.
//   2. Members must not carry an `mbarrier`.
//   3. Hints must be pairwise-disjoint; their union need not be a valid hint.
//   4. Group size must be 2, 3, or 4.
//   5. Members must be consecutive in one block when the grouping is frozen;
//      later passes do not split or grow a frozen group.
//   6. Members must have same-rank descriptors.
//   7. Members must share the same `cache` modifier.
struct TDMMergeGroupInfo {
  // Members in program order. |members| = |memberHints| = N.
  SmallVector<Operation *> members;
  SmallVector<uint32_t> memberHints;
  // Insertion anchor for the fused intrinsic.
  Operation *lastInProgramOrder = nullptr;
};

// Each group's info is stored once and shared by all its members.
using TDMMergeGroupMap =
    llvm::DenseMap<Operation *, std::shared_ptr<TDMMergeGroupInfo>>;

// Add generated hints to runs of adjacent unhinted copies when auto-merge is
// enabled.  Only attributes are added; copy order is unchanged.
void prepareGeneratedTDMMergeHints(ModuleOp mod);

// Identify merge groups from copies that already carry compatible hints.  Ops
// that do not belong to a group are absent from the result.
TDMMergeGroupMap computeTDMMergeGroups(ModuleOp mod);

// Generate hints, compute groups, and stamp `amdgpu.tdm_merge_id` plus
// `amdgpu.tdm_merge_index` on each group member.  The index is the member's
// position in program order.
void assignTDMMergeGroupIds(ModuleOp mod);

// Rebuild merge groups from the stamped attributes and validate that each group
// still satisfies the merge rules.
TDMMergeGroupMap readTDMMergeGroups(ModuleOp mod);

} // namespace mlir::triton::AMD

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_DIALECT_TRITONAMDGPU_UTILITY_TDMMERGEUTILITY_H
