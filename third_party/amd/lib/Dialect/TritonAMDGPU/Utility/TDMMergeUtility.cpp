#include "Dialect/TritonAMDGPU/Utility/TDMMergeUtility.h"

#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
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

bool haveSameRankAndCache(TDMCopyGlobalToLocalOp lhs,
                          TDMCopyGlobalToLocalOp rhs) {
  // Equal rank implies equal descriptor group count (groupCount = rank>2?4:2),
  // so the rank check subsumes the group-count check.
  return lhs.getDesc().getType().getShape().size() ==
             rhs.getDesc().getType().getShape().size() &&
         lhs.getCache() == rhs.getCache();
}

bool isAutoMergeCandidate(TDMCopyGlobalToLocalOp copy) {
  if (copy.getWarpUsedHintAttr() || copy.getBarrier())
    return false;
  // Partitioned destinations need encoding-specific hint constraints, so leave
  // them to explicit fused loads instead of generating masks implicitly.
  return !isa<triton::gpu::PartitionedSharedEncodingAttr>(
      copy.getResult().getType().getEncoding());
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
  //   1. Only unhinted copies are auto-materialized. User-provided hints stay as
  //      standalone copies; users should use the explicit fused op for manual
  //      merges.
  //   2. Members must not carry an `mbarrier`.
  //   3. Members must be consecutive in one block.
  //   4. Group size must be 2, 3, or 4.
  //   5. Members must have same-rank descriptors.
  //   6. Members must share the same `cache` modifier.
  if (!autoHintGenerationEnabled())
    return;

  SmallVector<Block *> blocks = collectBlocksWithTDMCopies(mod);

  // Scan each adjacent unhinted run and immediately materialize compatible auto
  // groups. Manual hinted copies are not considered here.
  for (Block *block : blocks) {
    SmallVector<TDMCopyGlobalToLocalOp, 8> run;
    auto flush = [&]() {
      unsigned numWarps =
          run.empty() ? 0 : triton::gpu::lookupNumWarps(run.front());
      if (numWarps <= 8) {
        MutableArrayRef<TDMCopyGlobalToLocalOp> remaining(run);
        while (remaining.size() >= 2) {
          size_t maxGroupSize = std::min<size_t>(4, remaining.size());
          maxGroupSize =
              std::min<size_t>(maxGroupSize, static_cast<size_t>(numWarps));
          if (maxGroupSize < 2)
            break;

          // Take the largest compatible prefix. A trailing 3-way group is
          // allowed; otherwise the greedy choice naturally prefers 4 then 2.
          size_t chosen = 1;
          for (size_t i = 1; i < maxGroupSize; ++i) {
            if (!haveSameRankAndCache(remaining.front(), remaining[i]))
              break;
            ++chosen;
          }

          if (chosen < 2) {
            remaining = remaining.drop_front(1);
            continue;
          }

          TDMMergeGroup group;
          for (auto [idx, copy] :
               llvm::enumerate(remaining.take_front(chosen))) {
            group.members.push_back(copy);
            group.memberHints.push_back(
                getGeneratedHint(static_cast<unsigned>(idx),
                                 static_cast<unsigned>(chosen), numWarps));
          }

          LLVM_DEBUG({
            llvm::dbgs() << "[tdm-merge] auto group of "
                         << group.members.size() << " ops\n";
            for (auto [idx, copy] : llvm::enumerate(group.members))
              llvm::dbgs() << "  hint=0x"
                           << llvm::Twine::utohexstr(group.memberHints[idx])
                           << " " << *copy << "\n";
          });

          size_t groupSize = group.members.size();
          remaining = remaining.drop_front(groupSize);
          materializeGroup(group);
        }
      }
      run.clear();
    };

    for (Operation &op : llvm::make_early_inc_range(*block)) {
      auto copy = dyn_cast<TDMCopyGlobalToLocalOp>(&op);
      if (copy && isAutoMergeCandidate(copy)) {
        run.push_back(copy);
        continue;
      }
      flush();
    }
    flush();
  }
}

} // namespace mlir::triton::AMD
