#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/TDMUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include <limits>

// Lower async waits to hardware instruction counts and annotate their
// completion scope for AMD Membar. On asyncmark targets, retain ordinary
// commit-group waits and let LLVM compute their hardware counts; TDM waits
// still need lowering.
//
// For instruction-counted waits, token operands take precedence over `num`.
// Follow each token's definition and count the instructions issued after it,
// taking the minimum across tokens and possible control-flow paths. Tokenless
// waits instead walk backward over `num` outstanding commit groups (ordinary
// copies) or TDM operations, counting the instructions emitted by lowering.
// Never overestimate the number allowed to remain outstanding: a smaller count
// waits more conservatively, while a larger count could leave required work
// incomplete.
//
// After count lowering, a direct forward scan tracks possible async global
// writes through the existing IR's control flow. Scope is evaluated before a
// full hardware wait clears its queue's bit; partial waits conservatively
// retain it.

namespace tt = triton;
namespace ttg = triton::gpu;
using mlir::triton::amdgpu::TargetFeatures;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUUPDATEASYNCWAITCOUNT
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

static bool hasAsyncToken(TypeRange types) {
  return llvm::any_of(types,
                      [](Type type) { return isa<ttg::AsyncTokenType>(type); });
}

static bool isAsyncWait(Operation *op) {
  return isa<ttg::AsyncWaitOp, amdgpu::AsyncWaitOp, amdgpu::AsyncTDMWait,
             amdgpu::AsyncTDMIntrinsicWait>(op);
}

// One global-write bit per queue is the cheapest model for load/store phases:
// loads need no state, joins use OR, and no producer queue or token history is
// needed. A hardware waitcnt(0) clears the bit after classifying the wait,
// restoring Shared scope after stores. Partial waits keep the bit; producer
// order/counts could refine retirement if needed.
class AsyncCompletionAnalysis {
  enum GlobalWriteQueue : unsigned {
    CopyGlobalWrite = 1,
    TDMGlobalWrite = 2,
  };

  struct EntryState {
    // Each bit marks a queue that may still have async global writes pending.
    unsigned pendingGlobalWrites;
    bool queued = false;
  };
  DenseMap<Operation *, EntryState> entries;
  DenseMap<Operation *, unsigned> waitScopes;
  SmallVector<Operation *> worklist;
  bool unsupported = false;

  // Schedule first visits even with zero bits, and revisit only if state grows.
  // Already queued entries pick up merged state when processed.
  void enqueue(Operation *op, unsigned pendingGlobalWrites) {
    if (!op)
      return;
    auto [it, inserted] = entries.try_emplace(op, EntryState{0});
    auto &entry = it->second;
    unsigned previous = entry.pendingGlobalWrites;
    entry.pendingGlobalWrites |= pendingGlobalWrites;
    if (!entry.queued && (inserted || entry.pendingGlobalWrites != previous)) {
      entry.queued = true;
      worklist.push_back(op);
    }
  }

  // Track only explicit async global writes completed by the supported waits.
  static unsigned classify(Operation *op) {
    return llvm::TypeSwitch<Operation *, unsigned>(op)
        .Case<amdgpu::AsyncCopyLocalToGlobalOp>(
            [](auto) { return CopyGlobalWrite; })
        .Case<amdgpu::AsyncTDMCopyLocalToGlobalOp, amdgpu::AsyncTDMScatterOp>(
            [](auto) { return TDMGlobalWrite; })
        .Default([](Operation *) { return 0; });
  }

  // Use final hardware counts. An asyncmark wait may leave uncommitted copies
  // outstanding even with num=0, so it cannot clear the ordinary queue's bit.
  static unsigned getDrainedQueue(Operation *op) {
    if (auto wait = dyn_cast<amdgpu::AsyncWaitOp>(op))
      return wait.getNumInst() == 0 ? CopyGlobalWrite : 0;
    if (auto wait = dyn_cast<amdgpu::AsyncTDMIntrinsicWait>(op))
      return wait.getCount() == 0 ? TDMGlobalWrite : 0;
    return 0;
  }

  // Follow existing block successors or return to the enclosing structured op.
  // Loop backedges resume at the header, without rescanning pre-loop producers.
  void leaveBlock(Operation *terminator, unsigned pendingGlobalWrites) {
    if (terminator->getNumSuccessors()) {
      for (Block *block : terminator->getSuccessors())
        enqueue(&block->front(), pendingGlobalWrites);
      return;
    }
    Operation *parent = terminator->getParentOp();
    if (isa<scf::IfOp, scf::ExecuteRegionOp>(parent)) {
      enqueue(parent->getNextNode(), pendingGlobalWrites);
    } else if (isa<scf::ForOp>(parent)) {
      enqueue(parent, pendingGlobalWrites);
    } else if (auto loop = dyn_cast<scf::WhileOp>(parent)) {
      bool fromBefore = terminator->getParentRegion() == &loop.getBefore();
      Block *next = fromBefore ? loop.getAfterBody() : loop.getBeforeBody();
      enqueue(&next->front(), pendingGlobalWrites);
      // The condition can also exit; the after-region only loops back.
      if (fromBefore)
        enqueue(parent->getNextNode(), pendingGlobalWrites);
    } else if (!isa<FunctionOpInterface>(parent)) {
      unsupported = true;
    }
  }

  // Scan straight-line IR directly; only control-flow transfers need worklist
  // entries. Accumulate wait scope before a full wait clears its queue's bit.
  void scan(Operation *op, unsigned pendingGlobalWrites) {
    for (; op; op = op->getNextNode()) {
      if (isAsyncWait(op)) {
        waitScopes[op] |= pendingGlobalWrites;
        pendingGlobalWrites &= ~getDrainedQueue(op);
        continue;
      }
      if (op->hasTrait<OpTrait::IsTerminator>()) {
        leaveBlock(op, pendingGlobalWrites);
        return;
      }
      if (!op->getNumRegions()) {
        pendingGlobalWrites |= classify(op);
        continue;
      }

      // Add the alternative path for an if or a possibly zero-trip for.
      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        if (branch.getElseRegion().empty())
          enqueue(op->getNextNode(), pendingGlobalWrites);
        else
          enqueue(&branch.getElseRegion().front().front(), pendingGlobalWrites);
      } else if (isa<scf::ForOp>(op)) {
        enqueue(op->getNextNode(), pendingGlobalWrites);
      } else if (isa<tt::ReduceOp, tt::ScanOp>(op)) {
        // Pure combiners cannot add async queue work.
        continue;
      } else if (!isa<scf::WhileOp, scf::ExecuteRegionOp>(op)) {
        // Unsupported regions may execute concurrently. Keep the whole
        // function conservative rather than treating them as sequential.
        unsupported = true;
        return;
      }
      // All supported ops enter their first region, including while's condition
      // region. Its terminator schedules the continuation or loop backedge.
      enqueue(&op->getRegion(0).front().front(), pendingGlobalWrites);
      return;
    }
  }

public:
  // Finish the forward fixed point before annotating waits. Unvisited waits,
  // unsupported regions and unknown entry state retain All scope.
  void annotate(FunctionOpInterface function, bool unknownEntry,
                StringAttr name, StringAttr shared, StringAttr all) {
    if (!unknownEntry) {
      for (Region &region : function->getRegions())
        if (!region.empty())
          enqueue(&region.front().front(), 0);
      while (!worklist.empty() && !unsupported) {
        Operation *op = worklist.pop_back_val();
        auto &entry = entries.find(op)->second;
        entry.queued = false;
        scan(op, entry.pendingGlobalWrites);
      }
    }
    function.walk([&](Operation *op) {
      if (!isAsyncWait(op))
        return;
      auto found = waitScopes.find(op);
      unsigned mask =
          isa<amdgpu::AsyncTDMWait, amdgpu::AsyncTDMIntrinsicWait>(op)
              ? TDMGlobalWrite
              : CopyGlobalWrite;
      // An empty queue also permits Shared; no shared producer is required.
      bool sharedOnly = !unsupported && !unknownEntry &&
                        found != waitScopes.end() && !(found->second & mask);
      op->setAttr(name, sharedOnly ? shared : all);
    });
  }
};

// Scan final waits after count lowering. Calls retain conservative
// module-wide scope because a callee can wait on work issued by its caller.
static void annotateCompletionScopes(ModuleOp module) {
  bool hasCalls =
      module.walk([](CallOpInterface) { return WalkResult::interrupt(); })
          .wasInterrupted();
  auto *ctx = module.getContext();
  auto name = StringAttr::get(ctx, amdgpu::kAsyncCompletionScopeAttrName);
  auto shared = StringAttr::get(ctx, "shared");
  auto all = StringAttr::get(ctx, "all");
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto function = dyn_cast<FunctionOpInterface>(op)) {
      AsyncCompletionAnalysis analysis;
      analysis.annotate(function,
                        hasCalls || hasAsyncToken(function.getArgumentTypes()),
                        name, shared, all);
      return WalkResult::skip();
    }
    if (isAsyncWait(op)) {
      op->setAttr(name, all);
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
}

// Returns the number of async copy instructions for global↔shared transfers.
// Works for both load (global→shared) and store (shared→global) operations.
// The calculation is based on data contiguity, mask alignment, and the layout
// mapping between global and shared memory addresses.
int getNumberOfAsyncCopyInstructions(RankedTensorType globalType,
                                     ttg::MemDescType sharedType, Value mask,
                                     int ptrContig, int contigHint,
                                     ModuleAxisInfoAnalysis &axisInfo,
                                     const TargetFeatures &targetFeatures,
                                     bool isStore) {
  LinearLayout globalLayout = tt::gpu::toLinearLayout(globalType);
  triton::LinearLayout sharedLayout =
      triton::gpu::toLinearLayoutIgnoringPadding(sharedType);
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

  // We progressively tighten the pointer contiguity to mirror the lowering of
  // async ops, starting with the mask alignment.
  if (mask)
    ptrContig = std::min<int>(ptrContig, axisInfo.getMaskAlignment(mask));

  // Ops may carry a contiguity hint that raises the contiguity beyond what
  // the pointer and mask axis info analysis can prove.
  ptrContig = std::max<int>(ptrContig, contigHint);

  // The global-to-shared layout limits the consecutive elements which can be
  // transferred by a single async intrinsic.
  ptrContig =
      std::min(ptrContig, globalToSharedLayout.getNumConsecutiveInOut());

  // For padded layouts the padding interval limits the vectorization.
  auto srcEnc = sharedType.getEncoding();
  if (auto padEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(srcEnc)) {
    ptrContig = std::min<int>(ptrContig, padEnc.getMinInterval());
  }

  // Divide number of registers by contig to get the number of async intrinsics.
  // Strip zero bases from the register dimension first — a zero basis means
  // multiple register indices map to the same offset, so no additional load
  // instruction is generated.
  auto kReg = StringAttr::get(globalType.getContext(), "register");
  int numberOfRegisters =
      globalToSharedLayout.removeZeroBasesAlongDim(kReg).getInDimSize(kReg);
  int numInstructions = std::max(1, numberOfRegisters / ptrContig);

  // When a given vector width is unsupported but half of it is, the store
  // lowering splits each async store into two half-width stores.
  if (isStore) {
    int elemBitWidth = sharedType.getElementType().getIntOrFloatBitWidth();
    int vecBits = ptrContig * elemBitWidth;
    if (!targetFeatures.supportsDirectFromLdsStoreBitWidth(vecBits) &&
        targetFeatures.supportsDirectFromLdsStoreBitWidth(vecBits / 2)) {
      numInstructions *= 2;
    }
  }

  return numInstructions;
}

// Return the number of generated intrinsics for async ops; 0 otherwise
// If emitRemarkOnNonAsyncOp is set for any non async op having a side effect on
// GlobalMemory an performance remark will be emitted
int getOpNumberOfAsyncCopyInstructions(Operation *op,
                                       const TargetFeatures &targetFeatures,
                                       ModuleAxisInfoAnalysis &axisInfo,
                                       bool emitRemarkOnNonAsyncOp) {
  if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
    int contig = LLVM::AMD::getVectorSize(copyOp.getSrc(), axisInfo);
    return getNumberOfAsyncCopyInstructions(
        copyOp.getSrc().getType(), copyOp.getResult().getType(),
        copyOp.getMask(), contig, copyOp.getContiguity(), axisInfo,
        targetFeatures, /*isStore=*/false);
  } else if (auto bufferOp = dyn_cast<amdgpu::BufferLoadToLocalOp>(op)) {
    auto ptrType = cast<RankedTensorType>(LLVM::AMD::getPointerTypeWithShape(
        bufferOp.getPtr(), bufferOp.getOffsets()));
    int contig = LLVM::AMD::getVectorSize(bufferOp.getPtr(),
                                          bufferOp.getOffsets(), axisInfo);
    return getNumberOfAsyncCopyInstructions(
        ptrType, bufferOp.getDest().getType(), bufferOp.getMask(), contig,
        bufferOp.getContiguity(), axisInfo, targetFeatures,
        /*isStore=*/false);
  } else if (auto copyOp = dyn_cast<amdgpu::AsyncCopyLocalToGlobalOp>(op)) {
    int contig = LLVM::AMD::getVectorSize(copyOp.getDst(), axisInfo);
    return getNumberOfAsyncCopyInstructions(
        copyOp.getDst().getType(), copyOp.getSrc().getType(), copyOp.getMask(),
        contig, copyOp.getContiguity(), axisInfo, targetFeatures,
        /*isStore=*/true);
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

// Whether any op nested in `op`'s regions is counted by `countFunc` or, when
// counting commit groups, is an async commit group, i.e. whether `op` can
// change the outstanding count. Commit groups do not affect TDM waits.
template <bool countCommitGroups>
static bool
regionsContainAsyncOps(Operation *op,
                       llvm::function_ref<int(Operation *)> countFunc) {
  for (Region &region : op->getRegions()) {
    WalkResult result = region.walk([&](Operation *inner) {
      if (countFunc(inner) > 0 ||
          (countCommitGroups && isa<ttg::AsyncCommitGroupOp>(inner)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return true;
  }
  return false;
}

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
    } else if (cursor->getNumRegions() > 0 &&
               regionsContainAsyncOps<countCommitGroups>(cursor, countFunc)) {
      // For unhandled ops with subregions that issue async ops we
      // conservatively bail out. Ops whose regions issue none (tt.reduce,
      // tt.map_elementwise, ...) are walked past like any other op.
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

// Lower the wait using a hardware count inferred from tokens or a backward
// scan.
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
      // the minimum number of vmcnt instructions
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
    llvm_unreachable("Unsupported wait type");
  }
}

} // anonymous namespace

struct TritonAMDGPUUpdateAsyncWaitCountPass
    : impl::TritonAMDGPUUpdateAsyncWaitCountBase<
          TritonAMDGPUUpdateAsyncWaitCountPass> {
  using Base::Base;

  void runOnOperation() override {
    TargetFeatures targetFeatures{llvm::StringRef(gfxArch)};
    if (!targetFeatures.isCDNA() && !targetFeatures.isCDNA5()) {
      return;
    }

    ModuleOp m = getOperation();

    // With asyncmark/wait_asyncmark, LLVM handles vmcnt computation —
    // Triton no longer needs to walk the IR and count outstanding async
    // intrinsics. Preserve ttg.async_wait and its commit-group count for
    // lowering to wait_asyncmark later; scope annotation still runs below.
    if (!targetFeatures.useAsyncMarks()) {
      // CDNA5 (and future arches without asyncmark) use instruction counting.
      SmallVector<ttg::AsyncWaitOp> waitOps;
      getOperation()->walk(
          [&](ttg::AsyncWaitOp waitOp) { waitOps.push_back(waitOp); });

      ModuleAxisInfoAnalysis axisInfo(m);
      DenseMap<Operation *, int> intrinsicCountCache;
      auto countAsyncLoadInstructions = [&](Operation *op) -> int {
        auto found = intrinsicCountCache.find(op);
        if (found != intrinsicCountCache.end()) {
          return found->second;
        }
        auto v = getOpNumberOfAsyncCopyInstructions(
            op, targetFeatures, axisInfo,
            /*emitRemarkOnNonAsyncOp=*/false);
        intrinsicCountCache[op] = v;
        return v;
      };

      for (auto waitOp : waitOps) {
        IRRewriter builder(waitOp->getContext());
        updateWaitCount(waitOp, countAsyncLoadInstructions, builder);
      }
    }

    // amdgpu.AsyncTDMWait should only count async tdm ops. Fused TDM copies are
    // explicit IR operations by this point, so the count mirrors LLVM lowering
    // directly.
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
          // warp_used_hint changes descriptor layout, not the number of static
          // TDM instructions. Count the sequence sized by numWarps.
          auto [_, numInstr] =
              mlir::LLVM::AMD::distributeTDMWarpsAlignToPartition(
                  smemTy.getShape(), numWarps, smemTy.getEncoding());
          return numInstr;
        } else if (isa<AsyncTDMFusedCopyGlobalToLocalOp>(op)) {
          return 1;
        } else if (auto copyOp = dyn_cast<AsyncTDMCopyLocalToGlobalOp>(op)) {
          auto smemTy = copyOp.getSrc().getType();
          int numWarps = ttg::lookupNumWarps(op);
          auto [_, numInstr] =
              mlir::LLVM::AMD::distributeTDMWarpsAlignToPartition(
                  smemTy.getShape(), numWarps, smemTy.getEncoding());
          return numInstr;
        } else if (isa<AsyncTDMScatterOp, AsyncTDMGatherOp>(op)) {
          auto rowIndicesType =
              cast<RankedTensorType>(op->getOperandTypes()[1]);
          return mlir::LLVM::AMD::getTDMGatherScatterIntrinsicCount(
              rowIndicesType);
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
    annotateCompletionScopes(m);
  }
};

} // namespace mlir
