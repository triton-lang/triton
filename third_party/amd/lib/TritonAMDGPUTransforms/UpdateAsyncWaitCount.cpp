#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/TDMUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include <limits>
#include <memory>

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
// After count lowering, a compact graph tracks possible async global writes
// through control flow. Scope is evaluated before a full hardware wait clears
// its queue's bit; partial waits conservatively retain it.

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

// Opaque effects may hide queue work; known synchronous memory effects do not.
// Inspect only this op's effects; the graph builder handles regions separately.
static bool hasOpaqueEffects(Operation *op) {
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!interface)
    return !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  SmallVector<MemoryEffects::EffectInstance> effects;
  interface.getEffects(effects);
  return llvm::any_of(effects, [](const auto &effect) {
    return isa<SideEffects::DefaultResource>(effect.getResource());
  });
}

// One global-write bit per queue is the cheapest model for load/store phases:
// loads need no state, joins use OR, and no producer queue or token history is
// needed. A hardware waitcnt(0) clears the bit after classifying the wait,
// restoring Shared scope after stores when no unknown work remains. Partial
// waits keep the bit; producer order/counts could refine retirement if needed.
class AsyncCompletionGraph {
  enum Work : unsigned {
    CopyGlobalWrite = 1,
    TDMGlobalWrite = 2,
    // Opaque effects may involve unmodeled queues; known queue drains cannot
    // establish that this work has completed.
    Unknown = 4,
  };

  // Each node summarizes producers before an optional wait. Classify the wait
  // before applying its drain to the outgoing state.
  struct Node {
    unsigned generated = 0;
    unsigned incoming = 0;
    unsigned drained = 0;
    Operation *wait = nullptr;
    SmallVector<unsigned, 2> successors;
    bool reached = false;
    bool queued = false;
  };
  SmallVector<Node> nodes;
  SmallVector<unsigned> waits;
  bool unsupported = false;

  unsigned addNode() {
    nodes.emplace_back();
    return nodes.size() - 1;
  }

  void addEdge(unsigned from, unsigned to) {
    nodes[from].successors.push_back(to);
  }

  // Synchronous operations disappear; consecutive producers share one summary.
  static unsigned classify(Operation *op) {
    // Barriers introduce no async work. Keep pending queue bits until a full
    // hardware wait proves completion, even though barriers have opaque effects.
    if (isa<ttg::BarrierOp>(op))
      return 0;
    if (!hasAsyncToken(op->getResultTypes()) &&
        !isa<amdgpu::TDMOpInterface>(op))
      return hasOpaqueEffects(op) ? Unknown : 0;
    return llvm::TypeSwitch<Operation *, unsigned>(op)
        .Case<ttg::AsyncCopyGlobalToLocalOp, amdgpu::BufferLoadToLocalOp,
              amdgpu::AsyncTDMCopyGlobalToLocalOp,
              amdgpu::AsyncTDMFusedCopyGlobalToLocalOp,
              amdgpu::AsyncTDMGatherOp, ttg::AsyncCommitGroupOp>(
            [](auto) { return 0; })
        .Case<amdgpu::AsyncCopyLocalToGlobalOp>(
            [](auto) { return CopyGlobalWrite; })
        .Case<amdgpu::TDMOpInterface>([](auto) { return TDMGlobalWrite; })
        .Case<BranchOpInterface, RegionBranchTerminatorOpInterface>(
            [](auto op) { return hasOpaqueEffects(op) ? Unknown : 0; })
        .Default([](Operation *) { return Unknown; });
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

  // Retain block edges as well as structured branches and loop backedges.
  // Unsupported regions may execute concurrently, so do not model them as
  // sequential code. Keep their waits, but disable narrowing for the function.
  // Pure reduce/scan combiners do not contribute queue work.
  void buildRegion(Region &region, unsigned entry, unsigned exit) {
    if (region.empty()) {
      addEdge(entry, exit);
      return;
    }
    DenseMap<Block *, unsigned> blocks;
    for (Block &block : region)
      blocks[&block] = addNode();
    addEdge(entry, blocks.lookup(&region.front()));
    for (Block &block : region) {
      unsigned current = blocks.lookup(&block);
      for (Operation &op : block) {
        if (isAsyncWait(&op)) {
          nodes[current].wait = &op;
          nodes[current].drained = getDrainedQueue(&op);
          waits.push_back(current);
          unsigned next = addNode();
          addEdge(current, next);
          current = next;
        } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          unsigned next = addNode();
          buildRegion(ifOp.getThenRegion(), current, next);
          buildRegion(ifOp.getElseRegion(), current, next);
          current = next;
        } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
          // Do not regenerate pre-loop work on the backedge.
          unsigned header = addNode(), next = addNode();
          addEdge(current, header);
          addEdge(header, next);
          buildRegion(forOp.getRegion(), header, header);
          current = next;
        } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
          // The before-region also executes on the zero-trip path.
          unsigned header = addNode(), condition = addNode(), next = addNode();
          addEdge(current, header);
          buildRegion(whileOp.getBefore(), header, condition);
          addEdge(condition, next);
          buildRegion(whileOp.getAfter(), condition, header);
          current = next;
        } else if (auto execute = dyn_cast<scf::ExecuteRegionOp>(op)) {
          unsigned next = addNode();
          buildRegion(execute.getRegion(), current, next);
          current = next;
        } else if (op.getNumRegions() && !isa<tt::ReduceOp, tt::ScanOp>(op)) {
          unsupported = true;
          op.walk([&](Operation *nested) {
            if (isAsyncWait(nested)) {
              unsigned index = addNode();
              nodes[index].wait = nested;
              waits.push_back(index);
            }
          });
        } else {
          nodes[current].generated |= classify(&op);
        }
      }
      auto successors = block.getSuccessors();
      if (successors.empty())
        addEdge(current, exit);
      else
        for (Block *successor : successors)
          addEdge(current, blocks.lookup(successor));
    }
  }

public:
  explicit AsyncCompletionGraph(FunctionOpInterface function) {
    unsigned entry = addNode(), exit = addNode();
    for (Region &region : function->getRegions())
      buildRegion(region, entry, exit);
  }

  // Union states at joins and revisit successors only when their input grows.
  // Backedges reach a fixed point before any wait receives its final scope.
  void annotate(bool unknownEntry, StringAttr name, StringAttr shared,
                StringAttr all) {
    SmallVector<unsigned> worklist{0};
    nodes[0].reached = nodes[0].queued = true;
    while (!worklist.empty() && !unsupported && !unknownEntry) {
      unsigned index = worklist.pop_back_val();
      Node &node = nodes[index];
      node.queued = false;
      unsigned outgoing = (node.incoming | node.generated) & ~node.drained;
      for (unsigned successor : node.successors) {
        Node &next = nodes[successor];
        unsigned merged = next.incoming | outgoing;
        if (next.reached && merged == next.incoming)
          continue;
        next.incoming = merged;
        next.reached = true;
        if (!next.queued) {
          next.queued = true;
          worklist.push_back(successor);
        }
      }
    }
    for (unsigned index : waits) {
      const Node &node = nodes[index];
      bool isTDM = isa<amdgpu::AsyncTDMIntrinsicWait>(node.wait);
      unsigned work = (node.incoming | node.generated) &
                      ((isTDM ? TDMGlobalWrite : CopyGlobalWrite) | Unknown);
      // No pending work also permits Shared; a shared producer is not required.
      bool sharedOnly =
          !unsupported && !unknownEntry && node.reached && work == 0;
      node.wait->setAttr(name, sharedOnly ? shared : all);
    }
  }
};

// Analyze final waits without changing the IR graph. Calls retain conservative
// module-wide scope because a callee can wait on work issued by its caller.
static void annotateCompletionScopes(ModuleOp module, bool hasCalls) {
  auto *ctx = module.getContext();
  auto name = StringAttr::get(ctx, amdgpu::kAsyncCompletionScopeAttrName);
  auto shared = StringAttr::get(ctx, "shared");
  auto all = StringAttr::get(ctx, "all");
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto function = dyn_cast<FunctionOpInterface>(op)) {
      AsyncCompletionGraph graph(function);
      graph.annotate(hasCalls || hasAsyncToken(function.getArgumentTypes()),
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

// Lower the wait using a hardware count inferred from tokens or a backward scan.
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

    // Asyncmark targets retain commit-group waits and do not need axis info.
    std::unique_ptr<ModuleAxisInfoAnalysis> axisInfo;
    if (!targetFeatures.useAsyncMarks())
      axisInfo = std::make_unique<ModuleAxisInfoAnalysis>(m);
    DenseMap<Operation *, int> intrinsicCountCache;
    auto countAsyncInstructions = [&](Operation *op) -> int {
      auto found = intrinsicCountCache.find(op);
      if (found != intrinsicCountCache.end())
        return found->second;
      int count = getOpNumberOfAsyncCopyInstructions(
          op, targetFeatures, *axisInfo, /*emitRemarkOnNonAsyncOp=*/false);
      intrinsicCountCache[op] = count;
      return count;
    };

    // Fused and partitioned TDM operations use the same counts as lowering.
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

    // Finish count lowering before building the scope graph. Collect first so
    // replacing waits cannot invalidate the walk; preserve their IR order.
    SmallVector<Operation *> waits;
    bool hasCalls = false;
    m.walk([&](Operation *op) {
      hasCalls |= isa<CallOpInterface>(op);
      if (isa<ttg::AsyncWaitOp, amdgpu::AsyncTDMWait>(op))
        waits.push_back(op);
    });
    IRRewriter builder(m.getContext());
    for (Operation *op : waits) {
      if (auto wait = dyn_cast<ttg::AsyncWaitOp>(op)) {
        if (!targetFeatures.useAsyncMarks())
          updateWaitCount(wait, countAsyncInstructions, builder);
      } else {
        updateWaitCount(cast<amdgpu::AsyncTDMWait>(op), countTDMInstructions,
                        builder);
      }
    }
    annotateCompletionScopes(m, hasCalls);
  }
};

} // namespace mlir
