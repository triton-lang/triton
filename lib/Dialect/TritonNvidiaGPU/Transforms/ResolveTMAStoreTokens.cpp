#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include <functional>
#include <iterator>
#include <queue>
#include <vector>

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPURESOLVETMASTORETOKENSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

static int countTMAStoreLikeOps(Block::iterator begin, Block::iterator end) {
  int count = 0;
  for (auto it = begin; it != end; ++it) {
    if (isa<TMAStoreLikeOpInterface>(&*it))
      ++count;
  }
  return count;
}

static int countTMAStoreLikeOpsBefore(Block *block, Operation *endOp) {
  return countTMAStoreLikeOps(block->begin(), Block::iterator(endOp));
}

static int countTMAStoreLikeOpsInBlock(Block *block) {
  return countTMAStoreLikeOps(block->begin(), block->end());
}

// Compute the minimum number of TMA store-like ops between `producer` and
// `waitOp` across all CFG paths. Minimum is correct because wait_group N
// means "wait until at most N remain pending"; using the shortest path
// guarantees the producer has completed on every path (over-synchronizing on
// longer paths is safe, under-synchronizing on shorter paths is not).
static FailureOr<int> computePendingCount(Operation *producer,
                                          Operation *waitOp) {
  Block *producerBlock = producer->getBlock();
  Block *waitBlock = waitOp->getBlock();
  if (producerBlock == waitBlock) {
    auto begin = std::next(Block::iterator(producer));
    return countTMAStoreLikeOps(begin, Block::iterator(waitOp));
  }

  llvm::DenseMap<Block *, int> dist;
  using QueueItem = std::pair<int, Block *>;
  std::priority_queue<QueueItem, std::vector<QueueItem>,
                      std::greater<QueueItem>>
      worklist;

  int initial = countTMAStoreLikeOps(std::next(Block::iterator(producer)),
                                     producerBlock->end());
  for (Block *succ : producerBlock->getSuccessors()) {
    int cost = initial;
    cost += succ == waitBlock ? countTMAStoreLikeOpsBefore(succ, waitOp)
                              : countTMAStoreLikeOpsInBlock(succ);
    auto it = dist.find(succ);
    if (it == dist.end() || cost < it->second) {
      dist[succ] = cost;
      worklist.push({cost, succ});
    }
  }

  while (!worklist.empty()) {
    auto [cost, block] = worklist.top();
    worklist.pop();
    if (dist.lookup(block) != cost)
      continue;
    if (block == waitBlock)
      return cost;
    for (Block *succ : block->getSuccessors()) {
      int nextCost =
          cost + (succ == waitBlock ? countTMAStoreLikeOpsBefore(succ, waitOp)
                                    : countTMAStoreLikeOpsInBlock(succ));
      auto it = dist.find(succ);
      if (it == dist.end() || nextCost < it->second) {
        dist[succ] = nextCost;
        worklist.push({nextCost, succ});
      }
    }
  }
  return failure();
}

static FailureOr<int> computePendingCountFromLoopResult(scf::ForOp forOp,
                                                        unsigned resultIdx,
                                                        Operation *waitOp) {
  Value yielded = forOp.getYieldedValues()[resultIdx];
  Operation *producer = yielded.getDefiningOp();
  if (!producer || !isa<TMAStoreLikeOpInterface>(producer))
    return failure();

  int count = countTMAStoreLikeOps(std::next(Block::iterator(producer)),
                                   producer->getBlock()->end());
  FailureOr<int> afterLoop = computePendingCount(forOp, waitOp);
  if (failed(afterLoop))
    return failure();
  return count + *afterLoop;
}

static FailureOr<int> computePendingCount(Value token, Operation *waitOp) {
  Operation *producer = token.getDefiningOp();
  if (!producer)
    return failure();
  if (isa<TMAStoreLikeOpInterface>(producer))
    return computePendingCount(producer, waitOp);
  if (auto forOp = dyn_cast<scf::ForOp>(producer)) {
    unsigned resultIdx = cast<OpResult>(token).getResultNumber();
    return computePendingCountFromLoopResult(forOp, resultIdx, waitOp);
  }
  return failure();
}

} // anonymous namespace

class TritonNvidiaGPUResolveTMAStoreTokensPass
    : public impl::TritonNvidiaGPUResolveTMAStoreTokensPassBase<
          TritonNvidiaGPUResolveTMAStoreTokensPass> {
public:
  using TritonNvidiaGPUResolveTMAStoreTokensPassBase::
      TritonNvidiaGPUResolveTMAStoreTokensPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool failed = false;
    mod.walk([&](TMAStoreWaitOp waitOp) {
      Value token = waitOp.getToken();
      if (!token)
        return;
      FailureOr<int> pendingCount = computePendingCount(token, waitOp);
      if (::mlir::failed(pendingCount)) {
        waitOp.emitError("could not resolve token wait pending count");
        failed = true;
        return;
      }
      OpBuilder b(waitOp);
      auto newWait = TMAStoreWaitOp::create(b, waitOp.getLoc(), *pendingCount,
                                            waitOp.getReadOnly());
      waitOp.erase();
    });
    if (failed)
      return signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
