#include "triton/Analysis/LiveBuffers.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

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
// LiveBuffersAnalysis
//===----------------------------------------------------------------------===//

bool LiveBuffersAnalysis::join(BufferStates &lhs, const BitVector &rhs) {
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

void LiveBuffersAnalysis::initialize(FuncOp func) {
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

void LiveBuffersAnalysis::run(FuncOp func, RegionPredecessorAnalysis &preds) {
  SmallVector<BlockIter> worklist;

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
    BlockIter afterIt(block, iter);
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
        for (BlockIter pred : preds.getPredecessors(block->getParent())) {
          if (join(bufferStates[pred], stateAfter))
            worklist.push_back(pred);
        }
        continue;
      }
      // Check for predecessors.
      for (Block *pred : block->getPredecessors()) {
        BlockIter beforeIt(pred, pred->end());
        if (join(bufferStates[beforeIt], stateAfter))
          worklist.push_back(beforeIt);
      }
      continue;
    }

    // We are visiting an operation.
    Operation &op = *std::prev(iter);

    // Check for a region branch op.
    if (isa<RegionBranchOpInterface>(op)) {
      for (BlockIter pred : preds.getPredecessors(&op)) {
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
        curState.set(bufferId);
      }
    }

    // Enqueue the next virtual block if anything changed.
    BlockIter beforeIt(block, op.getIterator());
    if (join(bufferStates[beforeIt], curState))
      worklist.push_back(beforeIt);
  }
}

LiveBuffersAnalysis::LiveBuffersAnalysis(FuncOp func,
                                         RegionPredecessorAnalysis &preds) {
  initialize(func);
  run(func, preds);

  // There are likely a bunch of identical states, so compress them.
  for (const BufferStates &state : llvm::make_second_range(bufferStates))
    uniqueStates.insert(*state);
  // For a given buffer, find all the buffers that at any point are live at the
  // same time as the given buffer. We can do this by joining all the masks for
  // when the buffer is live.
  for (auto [bufferId, bufferOp] : llvm::enumerate(buffers)) {
    BitVector &liveMask = liveBufferMasks.emplace_back(buffers.size(), false);
    for (const BitVector &state : uniqueStates) {
      if (state.test(bufferId))
        liveMask |= state;
    }
  }
}

size_t LiveBuffersAnalysis::getBufferId(Operation *op) const {
  auto it = bufferIds.find(op);
  assert(it != bufferIds.end() && "operation is not a tracked buffer");
  return it->second;
}

Operation *LiveBuffersAnalysis::getBufferOp(size_t id) const {
  assert(id < buffers.size() && "buffer ID out of range");
  return buffers[id];
}

const BitVector &
LiveBuffersAnalysis::getLiveBuffersBefore(Operation *op) const {
  BlockIter it(op->getBlock(), op->getIterator());
  return getLiveBuffersBefore(it);
}

const BitVector &LiveBuffersAnalysis::getLiveBuffersAfter(Operation *op) const {
  BlockIter it(op->getBlock(), std::next(op->getIterator()));
  return getLiveBuffersBefore(it);
}

const BitVector &LiveBuffersAnalysis::getLiveBuffersBefore(BlockIter it) const {
  auto stateIt = bufferStates.find(it);
  assert(stateIt != bufferStates.end() && stateIt->second &&
         "operation does not have a live buffer state");
  return *stateIt->second;
}

const BitVector &LiveBuffersAnalysis::getLiveBufferMask(size_t id) const {
  assert(id < buffers.size() && "buffer ID out of range");
  return liveBufferMasks[id];
}

BitVector LiveBuffersAnalysis::getLiveBufferMask(ArrayRef<size_t> ids) const {
  BitVector result(buffers.size(), false);
  for (size_t id : ids)
    result |= getLiveBufferMask(id);
  return result;
}
