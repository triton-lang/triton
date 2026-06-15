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

// Mergeability rules:
//   1. Members must have verifier-legal `warp_used_hint` masks.
//   2. Members must not carry an `mbarrier`.
//   3. Hints must be pairwise-disjoint; their union need not be a valid hint.
//   4. Group size must be 2, 3, or 4.
//   5. Members must be consecutive in one block when materialized.
//   6. Members must have same-rank descriptors.
//   7. Members must share the same `cache` modifier.
struct TDMMergeGroupInfo {
  SmallVector<Operation *> members;
  SmallVector<uint32_t> memberHints;
};

// Generated hint for one member of an auto-merge group.
uint32_t getGeneratedMergeHint(unsigned groupIdx, unsigned groupSize,
                               unsigned numWarps) {
  assert(groupIdx < groupSize && "groupIdx out of range");
  if (groupSize == 3) {
    // numWarps is 4 or 8 (the greedy splitter caps auto-merge at 8). The 3-way
    // split is half + quarter + quarter to keep each hint an axis-aligned coset.
    static constexpr uint32_t kHints4[3] = {0b0011, 0b0100, 0b1000};
    static constexpr uint32_t kHints8[3] = {0b00001111, 0b00110000, 0b11000000};
    return (numWarps == 4 ? kHints4 : kHints8)[groupIdx];
  }

  // Every groupSize-th warp starting at groupIdx (e.g. groupSize=2 -> 0b...0101,
  // then shifted). Gives one set bit per stride.
  uint32_t stridePattern =
      ((uint32_t{1} << numWarps) - 1) / ((uint32_t{1} << groupSize) - 1);
  return stridePattern << groupIdx;
}

// Check if hint generation can apply to this copy.
bool isGeneratedMergeHintCandidate(TDMCopyGlobalToLocalOp op) {
  if (op.getWarpUsedHintAttr() || op.getBarrier())
    return false;
  return !isa<triton::gpu::PartitionedSharedEncodingAttr>(
      op.getResult().getType().getEncoding());
}

// Check if this copy can join a merge group.
bool isMergeableTDMCopy(TDMCopyGlobalToLocalOp op) {
  return op.getWarpUsedHintAttr() && !op.getBarrier();
}

// Rank of the copy's tensor descriptor.
size_t getTDMDescriptorRank(TDMCopyGlobalToLocalOp op) {
  return op.getDesc().getType().getShape().size();
}

// Assign each member of an adjacent group a disjoint generated hint.
void assignGeneratedMergeHints(MutableArrayRef<TDMCopyGlobalToLocalOp> group,
                               unsigned numWarps) {
  auto groupSize = static_cast<unsigned>(group.size());
  auto hintTy = IntegerType::get(group.front().getContext(), 32);
  for (auto [idx, copyOp] : llvm::enumerate(group)) {
    uint32_t hint = getGeneratedMergeHint(idx, groupSize, numWarps);
    copyOp.setWarpUsedHintAttr(IntegerAttr::get(hintTy, hint));
  }
}

// Split an adjacent run into largest supported groups and assign hints.
void assignGeneratedMergeHintsGreedily(
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
    assignGeneratedMergeHints(run.slice(i, chosen), numWarps);
    i += chosen;
  }
}

// Assign generated hints to already-adjacent unhinted copy runs.
void prepareGeneratedTDMMergeHintsImpl(ModuleOp mod) {
  llvm::SmallSetVector<Block *, 8> blocks;
  mod->walk(
      [&](TDMCopyGlobalToLocalOp tdm) { blocks.insert(tdm->getBlock()); });

  for (Block *block : blocks) {
    SmallVector<TDMCopyGlobalToLocalOp, 8> run;
    auto flush = [&]() {
      if (run.empty())
        return;
      unsigned numWarps = triton::gpu::lookupNumWarps(run.front());
      assignGeneratedMergeHintsGreedily(run, numWarps);
      run.clear();
    };

    for (Operation &op : *block) {
      auto tdm = dyn_cast<TDMCopyGlobalToLocalOp>(&op);
      if (tdm && isGeneratedMergeHintCandidate(tdm))
        run.push_back(tdm);
      else
        flush();
    }
    flush();
  }
}

// Pairwise merge checks other than hint disjointness.
bool canMergeWith(ArrayRef<Operation *> members,
                  TDMCopyGlobalToLocalOp candidate) {
  auto first = cast<TDMCopyGlobalToLocalOp>(members.front());
  // Equal rank implies equal descriptor group count (groupCount = rank>2?4:2),
  // so the rank check subsumes the group-count check.
  if (getTDMDescriptorRank(first) != getTDMDescriptorRank(candidate))
    return false;
  if (first.getCache() != candidate.getCache())
    return false;
  return true;
}

// Build merge groups from an adjacent run of hinted TDM copies.
void emitMergeGroup(MutableArrayRef<Operation *> run,
                    SmallVectorImpl<TDMMergeGroupInfo> &result) {
  auto hintOf = [](Operation *op) {
    return static_cast<uint32_t>(
        cast<TDMCopyGlobalToLocalOp>(op).getWarpUsedHintAttr().getInt());
  };

  while (run.size() >= 2) {
    SmallVector<Operation *, 4> members{run.front()};
    SmallVector<uint32_t, 4> hints{hintOf(run.front())};
    uint32_t orSoFar = hints.front();
    for (size_t i = 1; i < run.size() && members.size() < 4; ++i) {
      auto op = cast<TDMCopyGlobalToLocalOp>(run[i]);
      if (!canMergeWith(members, op))
        break;
      uint32_t hint = hintOf(op);
      if (orSoFar & hint)
        break;
      orSoFar |= hint;
      hints.push_back(hint);
      members.push_back(run[i]);
    }

    size_t groupSize = members.size();
    if (groupSize >= 2) {
      TDMMergeGroupInfo info;
      info.members.assign(members.begin(), members.end());
      info.memberHints.assign(hints.begin(), hints.end());
      LLVM_DEBUG({
        llvm::dbgs() << "[tdm-merge] group of " << groupSize << " ops\n";
        for (auto [idx, op] : llvm::enumerate(info.members))
          llvm::dbgs() << "  hint=0x"
                       << llvm::Twine::utohexstr(info.memberHints[idx]) << " "
                       << *op << "\n";
      });
      result.push_back(std::move(info));
      run = run.drop_front(groupSize);
    } else {
      run = run.drop_front(1);
    }
  }
}

// Gate only auto-generation of hints, not grouping of existing hints.
bool tdmAutoMergeEnabled() {
  auto disabled = mlir::triton::tools::isEnvValueBool(
      mlir::triton::tools::getStrEnv("TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS"));
  return !disabled.value_or(false);
}

// Generate hints for adjacent unhinted copies when the env knob allows it.
void prepareGeneratedTDMMergeHints(ModuleOp mod) {
  if (!tdmAutoMergeEnabled())
    return;
  prepareGeneratedTDMMergeHintsImpl(mod);
}

// Find groups of consecutive, hinted TDM copies.
SmallVector<TDMMergeGroupInfo> computeTDMMergeGroups(ModuleOp mod) {
  SmallVector<TDMMergeGroupInfo> result;

  llvm::SmallSetVector<Block *, 8> blocks;
  mod->walk(
      [&](TDMCopyGlobalToLocalOp tdm) { blocks.insert(tdm->getBlock()); });
  for (Block *block : blocks) {
    SmallVector<Operation *> candidates;
    auto flush = [&]() {
      if (candidates.size() >= 2)
        emitMergeGroup(candidates, result);
      candidates.clear();
    };

    for (Operation &op : *block) {
      if (auto tdm = dyn_cast<TDMCopyGlobalToLocalOp>(&op))
        if (isMergeableTDMCopy(tdm)) {
          candidates.push_back(&op);
          continue;
        }
      flush();
    }
    flush();
  }

  return result;
}

static void materializeTDMMergeGroup(const TDMMergeGroupInfo &group) {
  assert(group.members.size() == group.memberHints.size());
  OpBuilder builder(group.members.front());

  SmallVector<Value, 4> descs;
  SmallVector<Value, 4> dests;
  SmallVector<int32_t, 4> hints;
  auto cache = cast<TDMCopyGlobalToLocalOp>(group.members.front()).getCache();
  Type tokenType = cast<TDMCopyGlobalToLocalOp>(group.members.front())
                       .getToken()
                       .getType();
  for (auto [member, hint] : llvm::zip_equal(group.members, group.memberHints)) {
    auto copy = cast<TDMCopyGlobalToLocalOp>(member);
    descs.push_back(copy.getDesc());
    dests.push_back(copy.getResult());
    hints.push_back(static_cast<int32_t>(hint));
  }

  auto fused = triton::amdgpu::AsyncTDMFusedCopyGlobalToLocalOp::create(
      builder, group.members.front()->getLoc(), tokenType, descs, dests,
      builder.getDenseI32ArrayAttr(hints), cache);

  for (Operation *member : group.members)
    cast<TDMCopyGlobalToLocalOp>(member).getToken().replaceAllUsesWith(
        fused.getToken());

  for (Operation *member : llvm::reverse(group.members))
    member->erase();
}

} // namespace

void materializeTDMMergeGroups(ModuleOp mod) {
  prepareGeneratedTDMMergeHints(mod);
  auto groups = computeTDMMergeGroups(mod);

  for (const auto &group : groups)
    materializeTDMMergeGroup(group);
}

} // namespace mlir::triton::AMD
