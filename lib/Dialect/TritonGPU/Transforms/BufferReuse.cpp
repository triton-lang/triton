#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/AddressRanges.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// RegionPredecessorAnalysis
//===----------------------------------------------------------------------===//

using VirtualBlock = std::pair<Block *, Block::iterator>;

struct RegionPredecessorAnalysis {
  RegionPredecessorAnalysis(Operation *op);

  DenseMap<llvm::PointerUnion<Operation *, Region *>, SetVector<VirtualBlock>>
      predecessors;
};

RegionPredecessorAnalysis::RegionPredecessorAnalysis(Operation *op) {
  op->walk([&](Operation *op) {
    if (auto br = dyn_cast<RegionBranchOpInterface>(op)) {
      SmallVector<RegionSuccessor> successors;
      br.getSuccessorRegions(RegionBranchPoint::parent(), successors);
      VirtualBlock it(op->getBlock(), op->getIterator());
      for (RegionSuccessor &successor : successors) {
        if (successor.isParent())
          predecessors[op].insert(it);
        else
          predecessors[successor.getSuccessor()].insert(it);
      }
      return WalkResult::advance();
    }

    // FIXME: `ReturnLike` adds `RegionBranchTerminatorOpInterface` for some
    // reason. Check that the parent is actually a `RegionBranchOpInterface`.
    auto br = dyn_cast<RegionBranchTerminatorOpInterface>(op);
    if (br && isa<RegionBranchOpInterface>(br->getParentOp())) {
      SmallVector<Attribute> operands(br->getNumOperands());
      SmallVector<RegionSuccessor> regions;
      br.getSuccessorRegions(operands, regions);
      VirtualBlock it(br->getBlock(), br->getBlock()->end());
      for (RegionSuccessor &successor : regions) {
        if (successor.isParent())
          predecessors[br->getParentOp()].insert(it);
        else
          predecessors[successor.getSuccessor()].insert(it);
      }
      return WalkResult::advance();
    }

    return WalkResult::advance();
  });
}

//===----------------------------------------------------------------------===//
// BufferRange
//===----------------------------------------------------------------------===//

namespace {
struct BufferRange {
  Value alloc;
  bool isFullRange;
};
} // namespace

static BufferRange getBufferAccess(Value value) {
  if (auto trans = value.getDefiningOp<MemDescTransOp>()) {
    // The full memdesc is accessed.
    return getBufferAccess(trans.getSrc());
  }

  if (auto reshape = value.getDefiningOp<MemDescReshapeOp>()) {
    // The full memdesc is accessed.
    return getBufferAccess(reshape.getSrc());
  }

  if (auto subview = value.getDefiningOp<MemDescSubviewOp>()) {
    BufferRange srcAccess = getBufferAccess(subview.getSrc());
    // If the access range is unknown, propagate that.
    if (!srcAccess.isFullRange)
      return srcAccess;
    // FIXME: Only full buffer accesses are supported for now.
    for (Value offset : subview.getOffsets()) {
      APInt value;
      if (!matchPattern(offset, m_ConstantInt(&value)) || !value.isZero())
        return {srcAccess.alloc, /*isFullRange=*/false};
    }
    if (subview.getType().getNumElements() !=
        subview.getSrc().getType().getNumElements())
      return {srcAccess.alloc, /*isFullRange=*/false};
    // The full buffer is accessed.
    return {srcAccess.alloc, /*isFullRange=*/true};
  }

  if (auto alloc = value.getDefiningOp<LocalAllocOp>()) {
    uint64_t accessSize = product(alloc.getType().getShape());
    return BufferRange{value, /*isFullRange=*/true};
  }
  if (auto alloc = value.getDefiningOp<ttng::TMEMAllocOp>()) {
    uint64_t accessSize = product(alloc.getType().getShape());
    return BufferRange{value, /*isFullRange=*/true};
  }

  // TODO: Could handle joins, like `scf.if` or `arith.select`, if necessary.
  return {Value(), /*isFullRange=*/false};
}

// Get tracked buffers accessed by this operation and the memory effect on them.
static SmallVector<std::pair<BufferRange, MemoryEffects::Effect *>>
getAccessedBuffersAndEffects(Operation *op) {
  SmallVector<std::pair<BufferRange, MemoryEffects::Effect *>> result;
  auto itf = dyn_cast<MemoryEffectOpInterface>(op);
  if (!itf)
    return result;

  SmallVector<MemoryEffects::EffectInstance> effects;
  itf.getEffects(effects);
  // Consider the effects in reverse order because this is a backwards
  // analysis. E.g. a read then write should be treated as a write then read
  // by the analysis.
  for (const MemoryEffects::EffectInstance &effect : llvm::reverse(effects)) {
    Value mem = effect.getValue();
    if (!mem)
      continue;
    BufferRange access = getBufferAccess(mem);
    if (!access.alloc)
      continue;
    result.push_back({access, effect.getEffect()});
  }
  return result;
}

// Determine whether all accesses to the buffer can be tracked by the analysis.
// This function implements the inverse of `getBufferAccess` and
// `getAccessedBuffersAndEffects`.
static bool canTrackBufferAccesses(Value value) {
  for (Operation *user : value.getUsers()) {
    // Look through views.
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      if (!canTrackBufferAccesses(user->getResult(0)))
        return false;
      continue;
    }
    // Consider any known memory-effecting operation as a sink.
    if (isa<MemoryEffectOpInterface>(user))
      continue;
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// BufferLiveRangeAnalysis
//===----------------------------------------------------------------------===//

using llvm::BitVector;
using BufferStates = std::optional<BitVector>;

namespace {
struct BufferLiveRangeAnalysis {
  static bool join(BufferStates &lhs, const BitVector &rhs);
  void initialize(FuncOp func);
  void run(FuncOp func, RegionPredecessorAnalysis &preds);

  SmallVector<Operation *> buffers;
  llvm::MapVector<Operation *, size_t> bufferIds;
  llvm::MapVector<VirtualBlock, BufferStates> bufferStates;
};
} // namespace

bool BufferLiveRangeAnalysis::join(BufferStates &lhs, const BitVector &rhs) {
  if (!lhs) {
    lhs = rhs;
    return true;
  }
  assert(lhs->size() == rhs.size());
  if (*lhs == rhs)
    return false;
  *lhs |= rhs;
  return true;
}

void BufferLiveRangeAnalysis::initialize(FuncOp func) {
  // First find all the buffers to track.
  func.walk([&](Operation *op) {
    if (!isa<ttng::TMEMAllocOp, LocalAllocOp>(op) ||
        !canTrackBufferAccesses(op->getResult(0)))
      return;
    auto memdesc = cast<MemDescType>(op->getResult(0).getType());
    // Don't bother tracking buffers that are too small, such as mbarriers.
    if (memdesc.getNumElements() < 16)
      return;
    bufferIds.insert({op, bufferIds.size()});
    buffers.push_back(op);
  });
}

void BufferLiveRangeAnalysis::run(FuncOp func,
                                  RegionPredecessorAnalysis &preds) {
  SmallVector<VirtualBlock> worklist;

  // Root the analysis at function exits.
  for (auto root : func.getOps<ReturnOp>()) {
    Block *block = root->getBlock();
    worklist.push_back({block, block->end()});
    // All buffers are initially dead.
    bufferStates.insert({worklist.back(), BitVector(bufferIds.size(), false)});
  }

  // Fixed-point propagation.
  while (!worklist.empty()) {
    auto [block, iter] = worklist.pop_back_val();
    VirtualBlock afterIt(block, iter);
    BitVector stateAfter = *bufferStates.find(afterIt)->second;

    // Check for block entry.
    if (iter == block->begin()) {
      // Check for region entry.
      if (block->isEntryBlock()) {
        // Check for function entry.
        if (block->getParentOp() == func) {
          assert(stateAfter.none() && "all buffers should be dead on entry");
          continue;
        }

        // The parent must be a region branch op.
        assert(isa<RegionBranchOpInterface>(block->getParentOp()));
        for (VirtualBlock pred : preds.predecessors.at(block->getParent())) {
          if (join(bufferStates[pred], stateAfter))
            worklist.push_back(pred);
        }
        continue;
      }
      // Check for predecessors.
      for (Block *pred : block->getPredecessors()) {
        VirtualBlock beforeIt(pred, pred->end());
        if (join(bufferStates[beforeIt], stateAfter))
          worklist.push_back(beforeIt);
      }
      continue;
    }

    // We are visiting an operation.
    Operation &op = *std::prev(iter);

    // Check for a region branch op.
    if (isa<RegionBranchOpInterface>(op)) {
      for (VirtualBlock pred : preds.predecessors.at(&op)) {
        if (join(bufferStates[pred], stateAfter))
          worklist.push_back(pred);
      }
      continue;
    }

    // HACK: Because `ttng.tmem_alloc` and `ttg.local_alloc` don't indicate an
    // `Allocate` effect before memory allocation, we have to check them
    // separately.
    BitVector curState = stateAfter;
    if (auto it = bufferIds.find(&op); it != bufferIds.end()) {
      // Buffers are definitely dead before they are allocated.
      curState.reset(it->second);
    }

    // Check the operation for effects on tracked buffers.
    for (auto [access, effect] : getAccessedBuffersAndEffects(&op)) {
      auto it = bufferIds.find(access.alloc.getDefiningOp());
      if (it == bufferIds.end())
        continue;
      size_t bufferId = it->second;
      if (isa<MemoryEffects::Read>(effect)) {
        // Read always require the buffer to be live.
        curState.set(bufferId);
      } else if (isa<MemoryEffects::Write>(effect)) {
        // Writes mark the buffer as dead only if we know it writes the whole
        // buffer.
        if (access.isFullRange)
          curState.reset(bufferId);
      } else if (isa<MemoryEffects::Allocate>(effect)) {
        // Buffers are definitely dead before they are allocated.
        curState.reset(bufferId);
      } else {
        assert(isa<MemoryEffects::Free>(effect));
      }
    }

    // Enqueue the next virtual block if anything changed.
    VirtualBlock beforeIt(block, op.getIterator());
    if (join(bufferStates[beforeIt], curState))
      worklist.push_back(beforeIt);
  }
}

//===----------------------------------------------------------------------===//
// reuseBuffers
//===----------------------------------------------------------------------===//

// Determine whether two buffers with independent liveranges can be merged by
// checking if one trivially fits into the other.
static bool canMergeBuffers(Value a, Value b) {
  auto aType = cast<MemDescType>(a.getType());
  auto bType = cast<MemDescType>(b.getType());
  // Obviously, both have to be in the same address space.
  if (aType.getMemorySpace() != bType.getMemorySpace())
    return false;
  Attribute memorySpace = aType.getMemorySpace();

  if (isa<ttng::TensorMemorySpaceAttr>(memorySpace)) {
    // TMEM allocations are in squares, which means one has to fit into the
    // other in both dimensions.
    auto fits = [](auto &a, auto &b) {
      return a.numRows <= b.numRows && a.numCols <= b.numCols;
    };
    ttng::TMemAllocation aAllocSize = ttng::getTmemAllocSizes(aType);
    ttng::TMemAllocation bAllocSize = ttng::getTmemAllocSizes(bType);
    if (fits(aAllocSize, bAllocSize) || fits(bAllocSize, aAllocSize))
      return true;
    // Technically, we could replace both with a new allocation whose dimensions
    // are the max among the two, but for now just reject the merge.
    return false;
  }

  // Shared memory allocations are flat, so one buffer is always at least the
  // size of the other.
  assert(isa<SharedMemorySpaceAttr>(memorySpace) && "unknown memory space");
  return true;
}

// Given two buffers that can be merged, determine if they should be merged.
static bool shouldMergeBuffers(Value a, Value b) {
  // The buffers should not be merged if they do not have overlapping extents.
  // In this case, the memory allocator(s) could have placed them in the same
  // memory location anyways. Merging them early muddles the IR and could block
  // optimizations among other things.

  // CURRENTLY A HACK:
  auto usedInAcc = [&](OpOperand &use) {
    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(use.getOwner())) {
      if (use.get() == mmaOp.getAccumulator())
        return true;
    }
    return false;
  };

  auto hasDirectSSADependency = [](Value a, Value b) {
    SetVector<Operation *> slice;
    getForwardSlice(a, &slice);
    if (llvm::any_of(b.getUsers(),
                     [&](Operation *op) { return slice.contains(op); }))
      return true;
    slice.clear();
    BackwardSliceOptions opts;
    opts.omitUsesFromAbove = false;
    getBackwardSlice(b, &slice, opts);
    if (llvm::any_of(a.getUsers(),
                     [&](Operation *op) { return slice.contains(op); }))
      return true;
    return false;
  };

  auto getUserLoopParent = [](Value a) -> scf::ForOp {
    for (Operation *user : a.getUsers()) {
      if (auto loop = user->getParentOfType<scf::ForOp>())
        return loop;
    }
    return nullptr;
  };

  bool aUsedByMMAAcc = llvm::any_of(a.getUses(), usedInAcc);
  bool bUsedByMMAAcc = llvm::any_of(b.getUses(), usedInAcc);
  if (aUsedByMMAAcc == bUsedByMMAAcc)
    return false;
  if (getUserLoopParent(a) != getUserLoopParent(b))
    return false;

  return hasDirectSSADependency(a, b) || hasDirectSSADependency(b, a);
}

static void reuseBuffers(FuncOp func) {
  // First determine when buffers are live at every point in the function.
  RegionPredecessorAnalysis preds(func);
  BufferLiveRangeAnalysis analysis;
  analysis.initialize(func);
  analysis.run(func, preds);

  // There are likely a bunch of identical states, so compress them.
  SetVector<BitVector> uniqueStates;
  for (const BufferStates &state :
       llvm::make_second_range(analysis.bufferStates))
    uniqueStates.insert(*state);

  // Idealy, buffer merging runs after warp specialization and pipelining,
  // because the goal of both of these passes is to overlap execution of code.
  // Merging buffers too early can block this overlapping by creating a
  // dependency between what otherwise would have been independent regions of
  // code. E.g. this pass may determine that buffers A and B are never live at
  // the same time inside a loop body, but the pipeliner may want to overlap the
  // code in which they are live.
  //
  // But buffer merging cannot run before warp specialization because afterwards
  // it is difficult to determine which regions of code can actually execute
  // independently, based on the mbarrier waits and arrives in the code. This
  // consequently makes it difficult to determine whether two buffers are ever
  // live at the same time.
  //
  // To balance this, the pass runs before warp specialization but is careful to
  // not merge buffers in a way that blocks it and the pipeliner. This means
  // buffers that could be multi-buffered (e.g. MMAv5 accumulators) or whose
  // users don't have a direct SSA dependency on each other will not be merged.

  struct MergedBuffer {
    SmallVector<Operation *> ops;
    SmallVector<size_t> ids;
  };
  llvm::SpecificBumpPtrAllocator<MergedBuffer> allocator;
  SmallVector<MergedBuffer *> mergedBuffers;
  for (auto [buffer, id] : analysis.bufferIds) {
    MergedBuffer *mergedBuffer = allocator.Allocate();
    ::new (mergedBuffer) MergedBuffer{{buffer}, {id}};
    mergedBuffers.push_back(mergedBuffer);
  }

  // Two buffer sets can be merged if all of the buffers in one set can be
  // merged into any buffer in the other.
  auto canMerge = [](MergedBuffer *a, MergedBuffer *b) {
    return llvm::all_of(a->ops, [&](Operation *aOp) {
      return llvm::any_of(b->ops, [&](Operation *bOp) {
        return canMergeBuffers(aOp->getResult(0), bOp->getResult(0));
      });
    });
  };
  // Two buffer sets should be merged if all of the buffers in one set should be
  // merged with all of the buffers in the other.
  auto shouldMerge = [](MergedBuffer *a, MergedBuffer *b) {
    return llvm::all_of(a->ops, [&](Operation *aOp) {
      return llvm::all_of(b->ops, [&](Operation *bOp) {
        return shouldMergeBuffers(aOp->getResult(0), bOp->getResult(0));
      });
    });
  };

  // For a given buffer, find all the buffers that at any point are live at the
  // same time as the given buffer. We can do this by joining all the masks for
  // when the buffer is live.
  size_t numBuffers = analysis.buffers.size();
  bool changed;
  do {
    changed = false;

    for (MergedBuffer *mergedBuffer : mergedBuffers) {
      if (mergedBuffer->ops.empty())
        continue;
      BitVector liveMask(numBuffers, false);
      for (const BitVector &state : uniqueStates) {
        for (size_t id : mergedBuffer->ids) {
          if (state.test(id))
            liveMask |= state;
        }
      }
      // Greedily merge buffers.
      for (MergedBuffer *otherBuffer : mergedBuffers) {
        if (mergedBuffer->ops.empty())
          continue;
        if (llvm::any_of(otherBuffer->ids,
                         [&](size_t id) { return liveMask.test(id); }))
          continue;
        assert(mergedBuffer != otherBuffer);
        if (!canMerge(mergedBuffer, otherBuffer) ||
            !shouldMerge(mergedBuffer, otherBuffer))
          continue;
        // Merge the two sets.
        llvm::append_range(mergedBuffer->ops, otherBuffer->ops);
        llvm::append_range(mergedBuffer->ids, otherBuffer->ids);
        otherBuffer->ops.clear();
        otherBuffer->ids.clear();
        changed = true;
        break;
      }
      if (changed)
        break;
    }
  } while (changed);

  auto bufferLess = [](Operation *lhs, Operation *rhs) {
    auto aType = cast<MemDescType>(lhs->getResult(0).getType());
    auto bType = cast<MemDescType>(rhs->getResult(0).getType());
    assert(aType.getMemorySpace() == bType.getMemorySpace());

    if (isa<ttng::TensorMemorySpaceAttr>(aType.getMemorySpace())) {
      ttng::TMemAllocation aAllocSize = ttng::getTmemAllocSizes(aType);
      ttng::TMemAllocation bAllocSize = ttng::getTmemAllocSizes(bType);
      return aAllocSize.numRows <= bAllocSize.numRows &&
             aAllocSize.numCols <= bAllocSize.numCols;
    }

    size_t abits = product(getAllocationShapePerCTA(aType)) *
                   aType.getElementTypeBitWidth();
    size_t bbits = product(getAllocationShapePerCTA(bType)) *
                   bType.getElementTypeBitWidth();
    return abits <= bbits;
  };

  DominanceInfo domInfo(func);
  for (MergedBuffer *mergedBuffer : mergedBuffers) {
    if (mergedBuffer->ops.size() < 2)
      continue;
    Operation *leader = *llvm::max_element(mergedBuffer->ops, bufferLess);
    Operation *domOp = findNearestCommonDominator(mergedBuffer->ops, domInfo);
    if (leader != domOp)
      leader->moveBefore(domOp);
    Value leaderBuffer = leader->getResult(0);

    auto leaderType = cast<MemDescType>(leaderBuffer.getType());
    for (Operation *op : mergedBuffer->ops) {
      if (op == leader)
        continue;
      Value buffer = op->getResult(0);
      auto type = cast<MemDescType>(buffer.getType());

      Value src;
      if (auto alloc = dyn_cast<LocalAllocOp>(op))
        src = alloc.getSrc();
      else if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(op))
        src = tmemAlloc.getSrc();
      ImplicitLocOpBuilder b(op->getLoc(), op);
      Value replBuffer =
          b.create<MemDescReinterpretOp>(type, leaderBuffer, src);
      buffer.replaceAllUsesWith(replBuffer);
      op->erase();
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUBUFFERREUSE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct BufferReuse
    : public triton::gpu::impl::TritonGPUBufferReuseBase<BufferReuse> {
  using TritonGPUBufferReuseBase::TritonGPUBufferReuseBase;

  void runOnOperation() override {
    for (auto func : getOperation().getOps<FuncOp>())
      reuseBuffers(func);
  }
};
} // namespace
