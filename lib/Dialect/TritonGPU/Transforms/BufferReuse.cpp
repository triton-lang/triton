#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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

    auto br = dyn_cast<RegionBranchTerminatorOpInterface>(op);
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

//===----------------------------------------------------------------------===//
// BufferLiveRangeAnalysis
//===----------------------------------------------------------------------===//

namespace {
enum class BufferState : unsigned char { Dead, Live };

using BufferStates = SmallVector<BufferState>;

struct BufferLiveRangeAnalysis {
  static bool join(BufferStates &lhs, const BufferStates &rhs);
  void initialize(FuncOp func);
  void run(FuncOp func, RegionPredecessorAnalysis &preds);

  DenseMap<Operation *, size_t> bufferIds;
  DenseMap<VirtualBlock, BufferStates> bufferStates;
};
} // namespace

bool BufferLiveRangeAnalysis::join(BufferStates &lhs, const BufferStates &rhs) {
  if (lhs.empty() && rhs.empty())
    return false;
  if (lhs.empty()) {
    lhs = rhs;
    return true;
  }
  assert(lhs.size() == rhs.size());
  bool changed = false;
  for (auto [lhsState, rhsState] : llvm::zip(lhs, rhs)) {
    BufferState joined = std::max(lhsState, rhsState);
    if (joined != lhsState) {
      lhsState = joined;
      changed = true;
    }
  }
  return changed;
}

void BufferLiveRangeAnalysis::initialize(FuncOp func) {
  // First find all the buffers to track.
  func.walk([&](Operation *op) {
    if (auto alloc = dyn_cast<LocalAllocOp>(op)) {
      bufferIds.insert({op, bufferIds.size()});
      return WalkResult::advance();
    }
    if (auto alloc = dyn_cast<ttng::TMEMAllocOp>(op)) {
      bufferIds.insert({op, bufferIds.size()});
      return WalkResult::advance();
    }
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
    bufferStates[worklist.back()].resize(bufferIds.size(), BufferState::Dead);
  }

  // Fixed-point propagation.
  while (!worklist.empty()) {
    auto [block, iter] = worklist.pop_back_val();
    VirtualBlock afterIt(block, iter);
    const BufferStates &stateAfter = bufferStates[afterIt];

    // Check for block entry.
    if (iter == block->begin()) {
      // Check for region entry.
      if (block->isEntryBlock()) {
        // Check for function entry.
        if (block->getParentOp() == func) {
          assert(llvm::all_of(stateAfter,
                              [](BufferState state) {
                                return state == BufferState::Dead;
                              }) &&
                 "all buffers should be dead on entry");
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
    BufferStates curState = stateAfter;
    if (auto it = bufferIds.find(&op); it != bufferIds.end()) {
      // Buffers are definitely dead before they are allocated.
      curState[it->second] = BufferState::Dead;

    } else if (auto itf = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      itf.getEffects(effects);
      // Consider the effects in reverse order because this is a backwards
      // analysis. E.g. a read then write should be treated as a write then read
      // by the analysis.
      for (const MemoryEffects::EffectInstance &effect :
           llvm::reverse(effects)) {
        Value mem = effect.getValue();
        if (!mem)
          continue;
        BufferRange access = getBufferAccess(mem);
        if (!access.alloc)
          continue;
        size_t bufferId = bufferIds.at(access.alloc.getDefiningOp());
        if (isa<MemoryEffects::Read>(effect.getEffect())) {
          // Read always require the buffer to be live.
          curState[bufferId] = BufferState::Live;
        } else if (isa<MemoryEffects::Write>(effect.getEffect())) {
          // Writes mark the buffer as dead only if we know it writes the whole
          // buffer.
          if (access.isFullRange)
            curState[bufferId] = BufferState::Dead;
        } else if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
          // Buffers are definitely dead before they are allocated.
          curState[bufferId] = BufferState::Dead;
        } else {
          assert(isa<MemoryEffects::Free>(effect.getEffect()));
        }
      }
    }

    // Enqueue the next virtual block if anything changed.
    VirtualBlock beforeIt(block, op.getIterator());
    if (join(bufferStates[beforeIt], curState))
      worklist.push_back(beforeIt);
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

  void runOnOperation() override;
};
} // namespace

void BufferReuse::runOnOperation() {}
