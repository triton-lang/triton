#include "Dialect/TritonAMDGPU/Utility/TDMMergeUtility.h"

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <cstdint>

#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "tdm-merge"

namespace mlir::triton::AMD {

namespace {

using TDMCopyGlobalToLocalOp =
    ::mlir::triton::amdgpu::AsyncTDMCopyGlobalToLocalOp;
using TDMFusedCopyGlobalToLocalOp =
    ::mlir::triton::amdgpu::AsyncTDMFusedCopyGlobalToLocalOp;

// Mergeability rules:
//   1. Members must have verifier-legal `warp_used_hint` masks.
//   2. Members must not carry an `mbarrier`.
//   3. Hints must be pairwise-disjoint; their union need not be a valid hint.
//   4. Group size must be 2, 3, or 4.
//   5. Members must be consecutive in one block when materialized.
//   6. Members must have same-rank descriptors.
//   7. Members must share the same `cache` modifier.
struct TDMMergeGroup {
  SmallVector<TDMCopyGlobalToLocalOp> members;
  SmallVector<uint32_t> memberHints;
};

SmallVector<Block *> collectBlocksWithTDMCopies(ModuleOp mod) {
  llvm::SmallSetVector<Block *, 8> blocks;
  mod->walk([&](TDMCopyGlobalToLocalOp copy) {
    blocks.insert(copy->getBlock());
  });
  return SmallVector<Block *>(blocks.begin(), blocks.end());
}

// Generated hint for one member of an automatically hinted group.
uint32_t getGeneratedHint(unsigned memberIdx, unsigned groupSize,
                          unsigned numWarps) {
  assert(memberIdx < groupSize && "memberIdx out of range");
  if (groupSize == 3) {
    // numWarps is 4 or 8 (the greedy splitter caps auto-merge at 8). The 3-way
    // split is half + quarter + quarter to keep each hint an axis-aligned coset.
    static constexpr uint32_t kHints4[3] = {0b0011, 0b0100, 0b1000};
    static constexpr uint32_t kHints8[3] = {0b00001111, 0b00110000, 0b11000000};
    return (numWarps == 4 ? kHints4 : kHints8)[memberIdx];
  }

  // Every groupSize-th warp starting at memberIdx (e.g. groupSize=2 -> 0b...0101,
  // then shifted). Gives one set bit per stride.
  uint32_t stridePattern =
      ((uint32_t{1} << numWarps) - 1) / ((uint32_t{1} << groupSize) - 1);
  return stridePattern << memberIdx;
}

bool canReceiveGeneratedHint(TDMCopyGlobalToLocalOp copy) {
  if (copy.getWarpUsedHintAttr() || copy.getBarrier())
    return false;

  // Partitioned destinations need encoding-specific hint constraints, so leave
  // them to user-provided hints instead of generating masks implicitly.
  return !isa<triton::gpu::PartitionedSharedEncodingAttr>(
      copy.getResult().getType().getEncoding());
}

bool canJoinMaterializedGroup(TDMCopyGlobalToLocalOp copy) {
  return copy.getWarpUsedHintAttr() && !copy.getBarrier();
}

size_t getDescriptorRank(TDMCopyGlobalToLocalOp copy) {
  return copy.getDesc().getType().getShape().size();
}

void assignGeneratedHints(MutableArrayRef<TDMCopyGlobalToLocalOp> copies,
                          unsigned numWarps) {
  auto groupSize = static_cast<unsigned>(copies.size());
  auto hintTy = IntegerType::get(copies.front().getContext(), 32);
  for (auto [idx, copy] : llvm::enumerate(copies)) {
    uint32_t hint = getGeneratedHint(idx, groupSize, numWarps);
    copy.setWarpUsedHintAttr(IntegerAttr::get(hintTy, hint));
  }
}

// Split an adjacent unhinted run into largest supported groups and assign hints.
void assignGeneratedHintsGreedily(
    MutableArrayRef<TDMCopyGlobalToLocalOp> run, unsigned numWarps) {
  // Generated 3-way hints only support 4 or 8 warps, so cap auto-merge at 8.
  if (numWarps > 8)
    return;

  for (size_t i = 0; i < run.size();) {
    size_t remaining = run.size() - i;
    // Take the largest group that fits (<= 4 members and <= numWarps).  A 3-way
    // group is only reachable for an exact trailing triple.
    size_t chosen = 0;
    for (size_t size : {size_t{4}, size_t{3}, size_t{2}}) {
      if (size <= remaining && size <= numWarps) {
        chosen = size;
        break;
      }
    }
    if (chosen < 2)
      break;
    assignGeneratedHints(run.slice(i, chosen), numWarps);
    i += chosen;
  }
}

void assignGeneratedHintsInBlock(Block &block) {
  SmallVector<TDMCopyGlobalToLocalOp, 8> run;
  auto flush = [&]() {
    if (run.empty())
      return;
    assignGeneratedHintsGreedily(run, triton::gpu::lookupNumWarps(run.front()));
    run.clear();
  };

  for (Operation &op : block) {
    auto copy = dyn_cast<TDMCopyGlobalToLocalOp>(&op);
    if (copy && canReceiveGeneratedHint(copy)) {
      run.push_back(copy);
      continue;
    }
    flush();
  }
  flush();
}

// Pairwise group checks other than hint disjointness.
bool isCompatibleWithGroup(ArrayRef<TDMCopyGlobalToLocalOp> members,
                           TDMCopyGlobalToLocalOp candidate) {
  auto first = members.front();
  // Equal rank implies equal descriptor group count (groupCount = rank>2?4:2),
  // so the rank check subsumes the group-count check.
  if (getDescriptorRank(first) != getDescriptorRank(candidate))
    return false;
  if (first.getCache() != candidate.getCache())
    return false;
  return true;
}

uint32_t getWarpUsedHint(TDMCopyGlobalToLocalOp copy) {
  return static_cast<uint32_t>(copy.getWarpUsedHintAttr().getInt());
}

void appendMergeGroupsFromRun(MutableArrayRef<TDMCopyGlobalToLocalOp> run,
                              SmallVectorImpl<TDMMergeGroup> &groups) {
  while (run.size() >= 2) {
    SmallVector<TDMCopyGlobalToLocalOp, 4> members{run.front()};
    SmallVector<uint32_t, 4> hints{getWarpUsedHint(run.front())};
    uint32_t orSoFar = hints.front();
    for (size_t i = 1; i < run.size() && members.size() < 4; ++i) {
      auto copy = run[i];
      if (!isCompatibleWithGroup(members, copy))
        break;
      uint32_t hint = getWarpUsedHint(copy);
      if (orSoFar & hint)
        break;
      orSoFar |= hint;
      hints.push_back(hint);
      members.push_back(run[i]);
    }

    size_t groupSize = members.size();
    if (groupSize >= 2) {
      LLVM_DEBUG({
        llvm::dbgs() << "[tdm-merge] group of " << groupSize << " ops\n";
        for (auto [idx, copy] : llvm::enumerate(members))
          llvm::dbgs() << "  hint=0x" << llvm::Twine::utohexstr(hints[idx])
                       << " " << *copy << "\n";
      });

      TDMMergeGroup group;
      group.members.assign(members.begin(), members.end());
      group.memberHints.assign(hints.begin(), hints.end());
      groups.push_back(std::move(group));
      run = run.drop_front(groupSize);
    } else {
      run = run.drop_front(1);
    }
  }
}

// Gate only auto-generation of hints, not grouping of existing hints.
bool autoHintGenerationEnabled() {
  auto disabled = mlir::triton::tools::isEnvValueBool(
      mlir::triton::tools::getStrEnv("TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS"));
  return !disabled.value_or(false);
}

void collectMergeGroupsInBlock(Block &block,
                               SmallVectorImpl<TDMMergeGroup> &groups) {
  SmallVector<TDMCopyGlobalToLocalOp, 8> run;
  auto flush = [&]() {
    if (run.size() >= 2)
      appendMergeGroupsFromRun(run, groups);
    run.clear();
  };

  for (Operation &op : block) {
    auto copy = dyn_cast<TDMCopyGlobalToLocalOp>(&op);
    if (copy && canJoinMaterializedGroup(copy)) {
      run.push_back(copy);
      continue;
    }
    flush();
  }
  flush();
}

void materializeGroup(const TDMMergeGroup &group) {
  assert(group.members.size() == group.memberHints.size());
  auto first = group.members.front();
  OpBuilder builder(first);

  SmallVector<Value, 4> descs;
  SmallVector<Value, 4> dests;
  SmallVector<int32_t, 4> hints;
  auto cache = first.getCache();
  Type tokenType = first.getToken().getType();
  for (auto [idx, hint] : llvm::enumerate(group.memberHints)) {
    TDMCopyGlobalToLocalOp copy = group.members[idx];
    descs.push_back(copy.getDesc());
    dests.push_back(copy.getResult());
    hints.push_back(static_cast<int32_t>(hint));
  }

  auto fused = TDMFusedCopyGlobalToLocalOp::create(
      builder, first.getLoc(), tokenType, descs, dests,
      builder.getDenseI32ArrayAttr(hints), cache);

  for (TDMCopyGlobalToLocalOp copy : group.members)
    copy.getToken().replaceAllUsesWith(fused.getToken());

  for (TDMCopyGlobalToLocalOp copy : llvm::reverse(group.members))
    copy.erase();
}

} // namespace

void materializeTDMMergeGroups(ModuleOp mod) {
  SmallVector<Block *> blocks = collectBlocksWithTDMCopies(mod);

  if (autoHintGenerationEnabled())
    for (Block *block : blocks)
      assignGeneratedHintsInBlock(*block);

  SmallVector<TDMMergeGroup> groups;
  for (Block *block : blocks)
    collectMergeGroupsInBlock(*block, groups);

  for (const auto &group : groups)
    materializeGroup(group);
}

} // namespace mlir::triton::AMD
