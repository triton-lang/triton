#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/TDMUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include <limits>

// This pass computes, for each AsyncWait, the number of outstanding async
// intrinsics that must be waited on. An AsyncWait can specify its wait target
// either via AsyncToken operands or via an explicit count (num) of outstanding
// async operations, with tokens taking precedence. To preserve correctness, the
// pass must never overestimate the wait count; underestimation only impacts
// performance by waiting more conservatively. The wait count represents the
// number of hardware instructions/intrinsics corresponding to the outstanding
// async operations. For waits that carry async tokens, the pass walks the
// def-use chains of each token and sums the number of async intrinsics
// oustanding excluding the producer of the async token. Tokens may be copied
// across loop boundaries (e.g., passed as loop initial arguments and yielded
// from the loop body); in such cases, the pass takes the minimum count across
// the possible paths. The final wait count is the minimum over all tokens and
// their paths. For waits without tokens the count represent the number of
// outstanding ttg.async_commit_groups (inclusive). The pass scans the IR
// backward to find the specified num async commit groups and computes the
// number of outstanding async intrinsics from async operations. Note that we
// walk until we find n+1 commit groups to include all async ops of the n'th
// commit group. Again, when multiple paths are possible, the pass takes the
// minimum count across all paths needed to reach num async operations. For
// ttg.async_wait we count:
// - On GFX9 the number of direct-to-lds instructions. We ignore loads to
//   registers since we do not control the vectorization (llvm can change it).
//   Therefore interleaving direct-to-lds and loads to registers will produce
//   conservative waits.
// - On GFX1250 the number of (multicast) async_load and async_stores. On
//   GFX1250 those are out of order with register loads so we will not get
//   conservative waits.
// For amdg.tdm_async_wait we only count TDM ops. Each tdm_load/store will
// produce exactly one instruction so it directly correlates with OP at TGGIR
// level.

namespace tt = triton;
namespace ttg = triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUUPDATEASYNCWAITCOUNT
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Returns the number of async copy instructions for global↔shared transfers.
// Works for both load (global→shared) and store (shared→global) operations.
// The calculation is based on data contiguity, mask alignment, and the layout
// mapping between global and shared memory addresses.
int getNumberOfAsyncCopyInstructions(RankedTensorType globalType,
                                     ttg::MemDescType sharedType, Value mask,
                                     int contig,
                                     ModuleAxisInfoAnalysis &axisInfo) {
  LinearLayout globalLayout = tt::gpu::toLinearLayout(globalType);
  LinearLayout sharedLayout;
  if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
          sharedType.getEncoding())) {
    sharedLayout = paddedEnc.getLinearComponent();
  } else {
    sharedLayout = triton::gpu::toLinearLayout(sharedType);
  }
  LinearLayout globalToSharedLayout =
      globalLayout.invertAndCompose(sharedLayout);

  // If the warp dimension has free variables, not all warps execute this async
  // copy — the lowering will predicate execution to canonical warps only (via
  // emitRedundantThreadPredicate + emitBranch). Non-canonical warps branch over
  // the buffer_load instructions entirely, so their vmcnt is not incremented.
  auto kWarp = StringAttr::get(globalType.getContext(), "warp");
  if (globalToSharedLayout.getFreeVariableMasks().lookup(kWarp) != 0) {
    return 0;
  }
  contig = std::min(contig, globalToSharedLayout.getNumConsecutiveInOut());

  if (mask)
    contig = std::min<int>(contig, axisInfo.getMaskAlignment(mask));

  // Divide number of registers by contig to get the number of async intrinsics.
  // Strip zero bases from the register dimension first — a zero basis means
  // multiple register indices map to the same offset, so no additional load
  // instruction is generated.
  auto kReg = StringAttr::get(globalType.getContext(), "register");
  int numberOfRegisters =
      globalToSharedLayout.removeZeroBasesAlongDim(kReg).getInDimSize(kReg);
  return std::max(1, numberOfRegisters / contig);
}

// Return the number of generated intrinsics for async ops; 0 otherwise
// If emitRemarkOnNonAsyncOp is set for any non async op having a side effect on
// GlobalMemory an performance remark will be emitted
int getOpNumberOfAsyncCopyInstructions(Operation *op,
                                       AMD::TargetInfo targetInfo,
                                       ModuleAxisInfoAnalysis &axisInfo,
                                       bool emitRemarkOnNonAsyncOp) {
  if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
    int contig = LLVM::AMD::getVectorSize(copyOp.getSrc(), axisInfo);
    return getNumberOfAsyncCopyInstructions(copyOp.getSrc().getType(),
                                            copyOp.getResult().getType(),
                                            copyOp.getMask(), contig, axisInfo);
  } else if (auto bufferOp = dyn_cast<amdgpu::BufferLoadToLocalOp>(op)) {
    auto ptrType = cast<RankedTensorType>(LLVM::AMD::getPointerTypeWithShape(
        bufferOp.getPtr(), bufferOp.getOffsets()));
    int contig = LLVM::AMD::getVectorSize(bufferOp.getPtr(),
                                          bufferOp.getOffsets(), axisInfo);
    return getNumberOfAsyncCopyInstructions(
        ptrType, bufferOp.getDest().getType(), bufferOp.getMask(), contig,
        axisInfo);
  } else if (auto copyOp = dyn_cast<amdgpu::AsyncCopyLocalToGlobalOp>(op)) {
    int contig = LLVM::AMD::getVectorSize(copyOp.getDst(), axisInfo);
    return getNumberOfAsyncCopyInstructions(copyOp.getDst().getType(),
                                            copyOp.getSrc().getType(),
                                            copyOp.getMask(), contig, axisInfo);
  } else if (emitRemarkOnNonAsyncOp) {
    SmallVector<mlir::MemoryEffects::EffectInstance> effects;
    if (auto memEffectIface = dyn_cast<MemoryEffectOpInterface>(op))
      memEffectIface.getEffectsOnResource(triton::GlobalMemory::get(), effects);
    if (!effects.empty()) {
      op->emitRemark("Global memory operation between async wait and "
                     "async_loads. This will hinder the interleaving of memory "
                     "operations and might impact performance.");
    }
  }
  return 0;
}

// Walks the IR backwards and accumulates countFunc(op) until we find
// numOustanding ops returning a non zero value. For control flow all possible
// paths are walked in a recursive DFS way and the minimum number found along
// all paths is returned. For unsupported ops with subregions it will return a
// conservative wait count to avoid incorrect waits. Parameters:
// - `cursor`: the operation we walk backwards from
// - `cameFrom`: tracks the operation we most recently stepped from as we
//      walk backwards, so we can disambiguate how to traverse multi-block ops
// - `numOutstanding`: remaining countFunc(op) > 0 to visit before acc stops
// - `pathSum`: accumulated result along the current path
// - `bestPath`: current found minimum when reaching numOutstanding or start of
//               the kernel
// - `branchStateCache`: memoization cache to stop walking multi blocks
//      ops already visited with the same number of outstanding ops. This
//      prevents infinite recursion depths for loops without ops contributing
// - `countFunc`: called on ops to determine if they contribute to the pathSum
// - `countCommitGroups`: if true, decrement on AsyncCommitGroupOp (for commit
//      group counting); if false, decrement numOutstanding when countFunc
//      returns non-zero (for instruction counting).
// TODO: walk static loops correctly to avoid conservative loops. (static loops
// from Gluon are unrolled right now)
using MemoCache = llvm::DenseSet<std::tuple<Operation *, int, int>>;
template <bool countCommitGroups>
int computeMinCountBackward(Operation *cursor, Operation *cameFrom,
                            int numOutstanding, int pathSum, int bestPath,
                            MemoCache &branchStateCache,
                            llvm::function_ref<int(Operation *)> countFunc) {
  assert(cameFrom != nullptr);
  // Step to the previous op within the current block; if none, step to
  // the parent op. Stop at the module since it asserts on ->getPrevNode().
  auto getPredecessor = [](Operation *op) {
    auto prevOp = op->getPrevNode();
    if (!prevOp) {
      prevOp = op->getParentOp();
      if (isa<ModuleOp>(prevOp)) {
        prevOp = nullptr;
      }
    }

    return prevOp;
  };

  // Continues the walk and updates bestPath to stop exploration early for paths
  // leading to a higher sum; repeated calls will return monotonically
  // decreasing values
  auto continueWalkFrom = [&](Operation *newCursor) {
    auto pathResult = computeMinCountBackward<countCommitGroups>(
        newCursor, cursor, numOutstanding, pathSum, bestPath, branchStateCache,
        countFunc);
    bestPath = std::min(bestPath, pathResult);
    return pathResult;
  };

  // Walk backwards through the IR
  while (cursor) {
    // numOutstanding is inclusive so we have to walk until < 0 to include the
    // async ops from the last outstanding commit group. Also prune path if the
    // current path cannot beat the known minimum.
    if (numOutstanding < 0 || pathSum >= bestPath) {
      return std::min(bestPath, pathSum);
    }

    // Handle operations with subregions.
    if (auto ifOp = dyn_cast<scf::IfOp>(cursor)) {
      // Traversal depends on where we came from:
      // If cameFrom is the successor of the ifOp, we walk the then and else
      // blocks. If there is no else block we continue upwards instead since we
      // could skip the if in case the condition is false.
      // If cameFrom is from then/else regions continue upwards
      bool cameFromThenOrElse = cameFrom->getParentOp() == ifOp;
      if (cameFromThenOrElse) {
        continueWalkFrom(getPredecessor(ifOp));
      } else {
        continueWalkFrom(ifOp.getThenRegion().front().getTerminator());
        if (!ifOp.getElseRegion().empty()) {
          continueWalkFrom(ifOp.getElseRegion().front().getTerminator());
        } else {
          continueWalkFrom(getPredecessor(ifOp));
        }
      }
      return bestPath;
    } else if (auto forOp = dyn_cast<scf::ForOp>(cursor)) {
      // We walk upwards (skip/escape for body) and walk the body
      continueWalkFrom(getPredecessor(forOp));

      // If we came from the body only walk it again if it's not in the cache
      auto cameFromBody = cameFrom->getBlock() == forOp.getBody();
      auto cacheKey = std::make_tuple(cursor, numOutstanding, pathSum);
      if (!cameFromBody || branchStateCache.insert(cacheKey).second) {
        continueWalkFrom(forOp.getBody()->getTerminator());
      }
      return bestPath;
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(cursor)) {
      // Traversal depends on which region we came from:
      //  - Came from successor -> before-body
      //  - Came from before-body -> after-body and upwards
      //  - Came from after-body -> before-body.
      Block *lastBlock = cameFrom->getBlock();
      bool cameFromBefore = lastBlock == whileOp.getBeforeBody();
      bool cameFromAfter = lastBlock == whileOp.getAfterBody();
      bool cameFromSuccessor = !cameFromAfter && !cameFromBefore;

      if (cameFromAfter || cameFromSuccessor) {
        // Walk before body
        continueWalkFrom(whileOp.getBeforeBody()->getTerminator());
      } else if (cameFromBefore) {
        // Walk upwards
        continueWalkFrom(getPredecessor(whileOp));
        // Do not walk the after-block if we already visited it with a lower
        // num outstanding because we already walked an identical path
        auto cacheKey = std::make_tuple(cursor, numOutstanding, pathSum);
        if (branchStateCache.insert(cacheKey).second)
          continueWalkFrom(whileOp.getAfterBody()->getTerminator());
      }
      return bestPath;
    } else if (auto executeRegionOp = dyn_cast<scf::ExecuteRegionOp>(cursor)) {
      auto &blocks = executeRegionOp.getRegion().getBlocks();
      // Warp pipelining only requires a single block per execute region
      if (blocks.size() > 1) {
        cursor->emitRemark(
            "ExecuteRegion with multiple blocks is not supported; falling back "
            "to conservative wait count");
        return 0;
      }
      // Traverse upwards if we came from the first block; else walk the body.
      // This assumes a single block per execute region.
      auto body = &blocks.front();
      if (cameFrom->getBlock() == body) {
        continueWalkFrom(getPredecessor(executeRegionOp));
      } else {
        continueWalkFrom(body->getTerminator());
      }
      return bestPath;
    } else if (isa<triton::FuncOp>(cursor)) {
      // Reached function boundary; return current sum (conservative)
      return std::min(bestPath, pathSum);
    } else if (cursor->getNumRegions() > 0 && !isa<triton::ReduceOp>(cursor)) {
      // For unhandled ops with subregions we conservatively bail out.
      // We ignore triton.reduce because it cannot contain async ops
      cursor->emitRemark(
          "has subregions but is not analyzed when determining async "
          "wait count; this yields conservative waits");
      return 0;
    }

    // Non-control-flow ops: keep walking and accumulate via countFunc
    int count = countFunc(cursor);
    pathSum += count;

    // Decrement numOutstanding based on counting mode
    if constexpr (countCommitGroups) {
      if (isa<ttg::AsyncCommitGroupOp>(cursor))
        numOutstanding--;
    } else {
      // For instruction counting: decrement when we find an operation that
      // emits instructions
      if (count > 0)
        numOutstanding--;
    }

    cameFrom = cursor;
    cursor = getPredecessor(cursor);
  }
  // No more ops or parents to traverse; return the accumulated count.
  return std::min(pathSum, bestPath);
}

// Overload for ease of use with AsyncWait (counts commit groups)
int computeMinCountBackward(ttg::AsyncWaitOp waitOp,
                            llvm::function_ref<int(Operation *)> countFunc) {
  MemoCache memoCache;
  return computeMinCountBackward</*countCommitGroups=*/true>(
      waitOp, waitOp, waitOp.getNum(), 0, std::numeric_limits<int>::max(),
      memoCache, countFunc);
}

// Overload for ease of use with AsyncTDMWait (counts TDM instructions)
int computeMinCountBackward(triton::amdgpu::AsyncTDMWait waitOp,
                            llvm::function_ref<int(Operation *)> countFunc) {
  MemoCache memoCache;
  // `computeMinCountBackward` expects `numOutstanding` to be the number of
  // outstanding ops to wait on. Since `AsyncTDMWait` counts the number of TDM
  // ops to wait on, we have to subtract 1 to stop before the n'th op is
  // reached.
  int numOutstanding = static_cast<int>(waitOp.getNum()) - 1;
  return computeMinCountBackward</*countCommitGroups=*/false>(
      waitOp, waitOp, numOutstanding, 0, std::numeric_limits<int>::max(),
      memoCache, countFunc);
}

// Follows the tokens of waitOp or walks the IR backwards from waitOp and
// modifies the waitCnt in place based on the accumulated result of
// computeCountForOp on interleaved instructions. See the file header for more
// details.
template <typename WaitType>
void updateWaitCount(WaitType waitOp,
                     llvm::function_ref<int(Operation *)> computeCountForOp,
                     RewriterBase &rewriter) {
  int waitCnt = std::numeric_limits<int>::max();

  if (waitOp.getNumOperands() > 0) {
    // AsyncWait can await multiple tokens so we get the minimum from all
    // tokens
    for (auto token : waitOp.getOperands()) {
      // Traverse def chain from waitOp to the producer of the token and count
      // the minumum number of vmcnt instructions
      auto tokenWaitCnt =
          deduceMinCountOnDefChain(token, waitOp, computeCountForOp);
      waitCnt = std::min(waitCnt, tokenWaitCnt);
    }
  } else {
    // For tokenless waits we treat `num` as the number:
    // - ttg.async_wait: number of outstanding commit groups
    // - amdg.async_tdm_wait: number of outstanding TDM operations
    waitCnt = computeMinCountBackward(waitOp, computeCountForOp);
  }

  if (waitCnt == std::numeric_limits<int>::max()) {
    // Could not determine wait count, emit conservative waitCnt=0
    waitCnt = 0;
  }

  if (std::is_same_v<WaitType, ttg::AsyncWaitOp>) {
    // Replace ttg.async_wait which counts outstanding commit groups with
    // amdg.async_wait which counts the number of outstanding intrinsics
    auto tokens = waitOp.getAsyncToken();
    auto origNum = waitOp.getNum();
    rewriter.setInsertionPointAfter(waitOp);
    auto newOp = rewriter.replaceOpWithNewOp<amdgpu::AsyncWaitOp>(
        waitOp, tokens, waitCnt);
    // Preserve the original commit-group count so downstream passes
    // (e.g. ConcurrencySanitizer) that reason about commit groups can
    // still access it after this lowering.
    newOp->setAttr("ttg.num_commit_groups",
                   rewriter.getI32IntegerAttr(origNum));
  } else if (std::is_same_v<WaitType, triton::amdgpu::AsyncTDMWait>) {
    // Replace amdg.async_tdm_wait (counts TDM operations) with
    // amdg.async_tdm_intrinsic_wait (counts TDM intrinsics)
    auto tokens = waitOp.getAsyncToken();
    auto origNum = waitOp.getNum();
    rewriter.setInsertionPointAfter(waitOp);
    auto newOp = rewriter.replaceOpWithNewOp<amdgpu::AsyncTDMIntrinsicWait>(
        waitOp, tokens, waitCnt);
    // Preserve the original TDM operation count so downstream passes
    // (e.g. ConcurrencySanitizer) can still access it after this lowering.
    newOp->setAttr("ttg.num_tdm_ops", rewriter.getI32IntegerAttr(origNum));
  } else {
    assert(false && "Unsupported wait type");
  }
}

} // anonymous namespace

struct TritonAMDGPUUpdateAsyncWaitCountPass
    : impl::TritonAMDGPUUpdateAsyncWaitCountBase<
          TritonAMDGPUUpdateAsyncWaitCountPass> {
  using Base::Base;

  void runOnOperation() override {
    tt::AMD::TargetInfo targetInfo(archGenerationName);
    if (!isCDNA(targetInfo.getISAFamily()) &&
        targetInfo.getISAFamily() != tt::AMD::ISAFamily::GFX1250) {
      return;
    }

    ModuleOp m = getOperation();

    // With asyncmark/wait_asyncmark, LLVM handles vmcnt computation —
    // Triton no longer needs to walk the IR and count outstanding async
    // intrinsics. Keep the ttg.async_wait ops unchanged (they track
    // commit groups) and lower them directly to wait_asyncmark later.
    if (!targetInfo.useAsyncMarks()) {
      // GFX1250 (and future arches without asyncmark) use instruction counting.
      SmallVector<ttg::AsyncWaitOp> waitOps;
      getOperation()->walk(
          [&](ttg::AsyncWaitOp waitOp) { waitOps.push_back(waitOp); });

      ModuleAxisInfoAnalysis axisInfo(m);
      DenseMap<Operation *, int> intrinsicCountCache;
      auto countAsyncLoadInstructions = [&](Operation *op) {
        auto found = intrinsicCountCache.find(op);
        if (found != intrinsicCountCache.end()) {
          return found->second;
        }
        auto v = getOpNumberOfAsyncCopyInstructions(
            op, targetInfo, axisInfo,
            /*emitRemarkOnNonAsyncOp=*/false);
        intrinsicCountCache[op] = v;
        return v;
      };

      for (auto waitOp : waitOps) {
        IRRewriter builder(waitOp->getContext());
        updateWaitCount(waitOp, countAsyncLoadInstructions, builder);
      }
    }

    // amdgpu.AsyncTDMWait should only count async tdm ops
    SmallVector<triton::amdgpu::AsyncTDMWait> waitTDMOps;
    getOperation()->walk([&](triton::amdgpu::AsyncTDMWait waitOp) {
      waitTDMOps.push_back(waitOp);
    });

    DenseMap<Operation *, int> tdmIntrinsicCountCache;
    auto countTDMInstructions = [&](Operation *op) -> int {
      auto found = tdmIntrinsicCountCache.find(op);
      if (found != tdmIntrinsicCountCache.end()) {
        return found->second;
      }

      auto v = [&]() -> int {
        using namespace triton::amdgpu;
        if (auto copyOp = dyn_cast<AsyncTDMCopyGlobalToLocalOp>(op)) {
          auto smemTy = copyOp.getResult().getType();
          int numWarps = ttg::lookupNumWarps(op);
          auto [_, numInstr] =
              mlir::LLVM::AMD::distributeTDMWarpsAlignToPartition(
                  smemTy.getShape(), numWarps, smemTy.getEncoding());
          return numInstr;
        } else if (auto copyOp = dyn_cast<AsyncTDMCopyLocalToGlobalOp>(op)) {
          auto smemTy = copyOp.getSrc().getType();
          int numWarps = ttg::lookupNumWarps(op);
          auto [_, numInstr] =
              mlir::LLVM::AMD::distributeTDMWarpsAlignToPartition(
                  smemTy.getShape(), numWarps, smemTy.getEncoding());
          return numInstr;
        } else if (isa<AsyncTDMScatterOp, AsyncTDMGatherOp>(op)) {
          // For scatter and gather we need to get the count of TDM intrinsics
          // based on the row indices tensor type
          auto rowIndicesType =
              cast<RankedTensorType>(op->getOperandTypes()[1]);
          bool use32BitIndices =
              rowIndicesType.getElementType().getIntOrFloatBitWidth() == 32;
          size_t numIndices = rowIndicesType.getNumElements();
          return mlir::LLVM::AMD::getTDMGatherScatterInstrinsicCount(
              numIndices, use32BitIndices);
        } else {
          return 0;
        }
      }();
      tdmIntrinsicCountCache[op] = v;
      return v;
    };

    for (auto waitOp : waitTDMOps) {
      IRRewriter builder(waitOp->getContext());
      updateWaitCount(waitOp, countTDMInstructions, builder);
    }
  }
};

} // namespace mlir
