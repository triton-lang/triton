#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>

#ifdef DEBUG_TYPE
#undef DEBUG_TYPE
#endif
#define DEBUG_TYPE "tdm-copy-fuse"

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUAUTOFUSETDMCOPY
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

using TDMCopyGlobalToLocalOp =
    ::mlir::triton::amdgpu::AsyncTDMCopyGlobalToLocalOp;
using TDMFusedCopyGlobalToLocalOp =
    ::mlir::triton::amdgpu::AsyncTDMFusedCopyGlobalToLocalOp;

SmallVector<Block *, 8> collectBlocksWithTDMCopies(ModuleOp mod) {
  llvm::SmallSetVector<Block *, 8> blocks;
  mod->walk(
      [&](TDMCopyGlobalToLocalOp copy) { blocks.insert(copy->getBlock()); });
  return blocks.takeVector();
}

// Generate verifier-legal, pairwise-disjoint warp masks for an auto-fused
// group. 2-way and 4-way groups use a regular stride pattern across all warps;
// 3-way groups use a half/quarter/quarter split because equal thirds are not
// axis-aligned.
uint32_t getGeneratedHint(unsigned memberIdx, unsigned groupSize,
                          unsigned numWarps) {
  assert(memberIdx < groupSize && "memberIdx out of range");
  if (groupSize == 3) {
    assert((numWarps == 4 || numWarps == 8) &&
           "3-way generated hints require 4 or 8 warps");
    static constexpr uint32_t kHints4[3] = {0b0011, 0b0100, 0b1000};
    static constexpr uint32_t kHints8[3] = {0b00001111, 0b00110000, 0b11000000};
    return (numWarps == 4 ? kHints4 : kHints8)[memberIdx];
  }

  uint32_t stridePattern =
      ((uint32_t{1} << numWarps) - 1) / ((uint32_t{1} << groupSize) - 1);
  return stridePattern << memberIdx;
}

bool haveSameRankAndCache(TDMCopyGlobalToLocalOp lhs,
                          TDMCopyGlobalToLocalOp rhs) {
  return lhs.getDesc().getType().getShape().size() ==
             rhs.getDesc().getType().getShape().size() &&
         lhs.getCache() == rhs.getCache();
}

bool isAutoFuseCandidate(TDMCopyGlobalToLocalOp copy) {
  // TODO: Relax this conservative mbarrier exclusion; mbarriers are descriptor
  // state and HW can attach multiple mbarriers to one TDM instruction.
  if (copy.getWarpUsedHintAttr() || copy.getBarrier())
    return false;
  // Partitioned destinations need encoding-specific hint constraints, so leave
  // them to explicit fused loads instead of generating masks implicitly.
  return !isa<triton::gpu::PartitionedSharedEncodingAttr>(
      copy.getResult().getType().getEncoding());
}

bool isTransparentBetweenAutoFuseCandidates(Operation *op) {
  return isa<triton::gpu::MemDescIndexOp>(op);
}

// Replace one compatible run of TDM copies with a single fused TDM copy.
void fuseGroup(MutableArrayRef<TDMCopyGlobalToLocalOp> group,
               unsigned numWarps) {
  assert(group.size() >= 2 && group.size() <= 4);
  auto first = group.front();
  OpBuilder builder(group.back());

  SmallVector<Value, 4> descs;
  SmallVector<Value, 4> dests;
  SmallVector<int32_t, 4> hints;
  auto cache = first.getCache();
  Type tokenType = first.getToken().getType();
  for (auto [idx, copy] : llvm::enumerate(group)) {
    descs.push_back(copy.getDesc());
    dests.push_back(copy.getResult());
    hints.push_back(static_cast<int32_t>(
        getGeneratedHint(static_cast<unsigned>(idx),
                         static_cast<unsigned>(group.size()), numWarps)));
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[tdm-copy-fuse] auto group of " << group.size()
                 << " ops\n";
    for (auto [idx, copy] : llvm::enumerate(group))
      llvm::dbgs() << "  hint=0x"
                   << llvm::Twine::utohexstr(static_cast<uint32_t>(hints[idx]))
                   << " " << *copy << "\n";
  });

  auto fused = TDMFusedCopyGlobalToLocalOp::create(
      builder, first.getLoc(), tokenType, descs, dests,
      builder.getDenseI32ArrayAttr(hints), cache);

  for (TDMCopyGlobalToLocalOp copy : group)
    copy.getToken().replaceAllUsesWith(fused.getToken());

  for (TDMCopyGlobalToLocalOp copy : llvm::reverse(group))
    copy.erase();
}

// Find auto-fuse candidates separated only by transparent view ops, split them
// into greedy compatible groups, and materialize each group as one explicit
// fused copy op. The fused op is inserted at the last copy in a group so any
// transparent view results between member copies dominate the new op.
void autoFuseTDMCopies(ModuleOp mod) {
  auto blocks = collectBlocksWithTDMCopies(mod);

  for (Block *block : blocks) {
    SmallVector<TDMCopyGlobalToLocalOp, 8> run;
    auto flush = [&]() {
      if (run.empty())
        return;

      unsigned numWarps = triton::gpu::lookupNumWarps(run.front());

      MutableArrayRef<TDMCopyGlobalToLocalOp> remaining(run);
      while (remaining.size() >= 2) {
        std::array<size_t, 3> groupLimits = {
            remaining.size(), static_cast<size_t>(numWarps), size_t(4)};
        size_t maxGroupSize = *llvm::min_element(groupLimits);
        if (maxGroupSize < 2)
          break;

        size_t groupSize = 1;
        for (size_t i = 1; i < maxGroupSize; ++i) {
          if (!haveSameRankAndCache(remaining.front(), remaining[i]))
            break;
          ++groupSize;
        }

        if (groupSize < 2) {
          remaining = remaining.drop_front(1);
          continue;
        }

        auto group = remaining.take_front(groupSize);
        remaining = remaining.drop_front(groupSize);
        fuseGroup(group, numWarps);
      }

      run.clear();
    };

    for (Operation &op : llvm::make_early_inc_range(*block)) {
      auto copy = dyn_cast<TDMCopyGlobalToLocalOp>(&op);
      if (copy && isAutoFuseCandidate(copy)) {
        // Generated hint patterns currently cover up to 8 warps.
        if (triton::gpu::lookupNumWarps(copy) > 8) {
          flush();
          continue;
        }
        run.push_back(copy);
        continue;
      }
      if (isTransparentBetweenAutoFuseCandidates(&op))
        continue;
      flush();
    }
    flush();
  }
}

struct TritonAMDGPUAutoFuseTDMCopyPass
    : impl::TritonAMDGPUAutoFuseTDMCopyBase<TritonAMDGPUAutoFuseTDMCopyPass> {
  void runOnOperation() override { autoFuseTDMCopies(getOperation()); }
};

} // namespace
} // namespace mlir
