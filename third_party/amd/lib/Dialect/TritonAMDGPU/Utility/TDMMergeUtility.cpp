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

    auto group = run.slice(i, chosen);
    auto hintTy = IntegerType::get(group.front().getContext(), 32);
    for (auto [idx, copy] : llvm::enumerate(group)) {
      uint32_t hint =
          getGeneratedHint(static_cast<unsigned>(idx), chosen, numWarps);
      copy.setWarpUsedHintAttr(IntegerAttr::get(hintTy, hint));
    }
    i += chosen;
  }
}

uint32_t getWarpUsedHint(TDMCopyGlobalToLocalOp copy) {
  return static_cast<uint32_t>(copy.getWarpUsedHintAttr().getInt());
}

bool haveSameRankAndCache(TDMCopyGlobalToLocalOp lhs,
                          TDMCopyGlobalToLocalOp rhs) {
  // Equal rank implies equal descriptor group count (groupCount = rank>2?4:2),
  // so the rank check subsumes the group-count check.
  return lhs.getDesc().getType().getShape().size() ==
             rhs.getDesc().getType().getShape().size() &&
         lhs.getCache() == rhs.getCache();
}

bool autoHintGenerationEnabled() {
  auto disabled = mlir::triton::tools::isEnvValueBool(
      mlir::triton::tools::getStrEnv("TRITON_AMD_DISABLE_TDM_AUTO_MERGE_HINTS"));
  return !disabled.value_or(false);
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
  // Mergeability rules:
  //   1. Members must have verifier-legal `warp_used_hint` masks.
  //   2. Members must not carry an `mbarrier`.
  //   3. Hints must be pairwise-disjoint; their union need not be a valid hint.
  //   4. Group size must be 2, 3, or 4.
  //   5. Members must be consecutive in one block when materialized.
  //   6. Members must have same-rank descriptors.
  //   7. Members must share the same `cache` modifier.
  SmallVector<Block *> blocks = collectBlocksWithTDMCopies(mod);

  // Phase 1: optionally assign hints to adjacent unhinted copy runs. This only
  // enables auto-merge; existing manual hints are grouped regardless of the env
  // knob below.
  if (autoHintGenerationEnabled()) {
    for (Block *block : blocks) {
      SmallVector<TDMCopyGlobalToLocalOp, 8> run;
      auto flush = [&]() {
        if (run.empty())
          return;
        assignGeneratedHintsGreedily(
            run, triton::gpu::lookupNumWarps(run.front()));
        run.clear();
      };

      for (Operation &op : *block) {
        auto copy = dyn_cast<TDMCopyGlobalToLocalOp>(&op);
        bool canAutoHint =
            copy && !copy.getWarpUsedHintAttr() && !copy.getBarrier() &&
            !isa<triton::gpu::PartitionedSharedEncodingAttr>(
                copy.getResult().getType().getEncoding());
        if (canAutoHint) {
          run.push_back(copy);
          continue;
        }
        flush();
      }
      flush();
    }
  }

  // Phase 2: find adjacent hinted copy groups.
  SmallVector<TDMMergeGroup> groups;
  for (Block *block : blocks) {
    SmallVector<TDMCopyGlobalToLocalOp, 8> run;
    auto flush = [&]() {
      MutableArrayRef<TDMCopyGlobalToLocalOp> remaining(run);
      while (remaining.size() >= 2) {
        TDMMergeGroup group;
        group.members.push_back(remaining.front());
        group.memberHints.push_back(getWarpUsedHint(remaining.front()));

        uint32_t usedHints = group.memberHints.front();
        for (size_t i = 1; i < remaining.size() && group.members.size() < 4;
             ++i) {
          TDMCopyGlobalToLocalOp copy = remaining[i];
          if (!haveSameRankAndCache(group.members.front(), copy))
            break;

          uint32_t hint = getWarpUsedHint(copy);
          if (usedHints & hint)
            break;

          usedHints |= hint;
          group.members.push_back(copy);
          group.memberHints.push_back(hint);
        }

        if (group.members.size() < 2) {
          remaining = remaining.drop_front(1);
          continue;
        }

        LLVM_DEBUG({
          llvm::dbgs() << "[tdm-merge] group of " << group.members.size()
                       << " ops\n";
          for (auto [idx, copy] : llvm::enumerate(group.members))
            llvm::dbgs() << "  hint=0x"
                         << llvm::Twine::utohexstr(group.memberHints[idx])
                         << " " << *copy << "\n";
        });
        remaining = remaining.drop_front(group.members.size());
        groups.push_back(std::move(group));
      }
      run.clear();
    };

    for (Operation &op : *block) {
      auto copy = dyn_cast<TDMCopyGlobalToLocalOp>(&op);
      if (copy && copy.getWarpUsedHintAttr() && !copy.getBarrier()) {
        run.push_back(copy);
        continue;
      }
      flush();
    }
    flush();
  }

  // Phase 3: replace each discovered group with one explicit fused op.
  for (const auto &group : groups)
    materializeGroup(group);
}

} // namespace mlir::triton::AMD
