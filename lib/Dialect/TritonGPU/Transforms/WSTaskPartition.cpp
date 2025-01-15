#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

#define DEBUG_TYPE "tritongpu-warp-task-partition"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define GEN_PASS_DEF_TRITONGPUWSTASKPARTITION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct TaskSchedule {
  unsigned numTasks = 0;
  DenseMap<Operation *, unsigned> opToTaskId;
};

// Compute a partition schedule for later passes to actually partition the
// program into async tasks.
void doPartition(triton::FuncOp &funcOp, unsigned numConsumerGroups) {

  // Bail out in the presence of user annotations.
  DenseSet<int> allAsyncTasks;
  funcOp->walk([&](Operation *op) {
    auto asyncTasks = getAsyncTaskIds(op);
    allAsyncTasks.insert(asyncTasks.begin(), asyncTasks.end());
  });

  if (!allAsyncTasks.empty())
    return;

  SmallVector<scf::ForOp> loops;
  SmallVector<Operation *> loads;
  SmallVector<Operation *> dots;

  funcOp.walk([&](Operation *op) {
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op))
      loops.push_back(forOp);
    else if (isa<nvidia_gpu::WarpGroupDotOp>(op))
      dots.push_back(op);
    else if (isa<triton::LoadOp, ExperimentalDescriptorLoadOp>(op))
      loads.push_back(op);
  });

  if (loops.empty() || loads.empty() || dots.empty())
    return;

  auto getLoopLevel = [&](Operation *op) {
    // Compute loop depth
    unsigned depth = 0;
    Operation *parent = op->getParentOp();
    while (parent) {
      if (isa<scf::ForOp>(parent)) {
        ++depth;
      }
      parent = parent->getParentOp();
    }
    return depth;
  };

  // Step 1. Select loads into the first task, which is the producer task by
  // default. Place dots into the second task, which is the consumer.
  // Only consider loads that are connected to a dot op in a loop.
  SmallVector<Operation *> producerOps;
  SmallVector<Operation *> consumerOps;
  for (auto op : dots) {
    if (getLoopLevel(op) == 0)
      continue;
    consumerOps.push_back(op);
    auto dotOp = dyn_cast<nvidia_gpu::WarpGroupDotOp>(op);
    if (!dotOp)
      continue;
    SetVector<Operation *> backwardSlice;
    getBackwardSlice(dotOp.getA(), &backwardSlice);
    getBackwardSlice(dotOp.getB(), &backwardSlice);

    for (auto depOp : backwardSlice) {
      if (isa<triton::LoadOp, ExperimentalDescriptorLoadOp>(depOp)) {
        producerOps.push_back(depOp);
      }
    }
  }

  LLVM_DEBUG({
    LDBG("Producer ops:\n");
    for (auto op : producerOps) {
      op->dump();
    }

    LDBG("\n");
    LDBG("Consumer ops:\n");
    for (auto op : consumerOps) {
      op->dump();
    }

    LDBG("\n");
  });

  if (consumerOps.empty() || producerOps.empty())
    return;

  // Annoate the program with task ids
  SmallVector<AsyncTaskId, 1> producerTaskIds{0};
  SmallVector<AsyncTaskId, 2> consumerTaskIds;
  for (unsigned i = 0; i < numConsumerGroups; ++i) {
    consumerTaskIds.push_back(i + producerTaskIds.size());
  }

  for (auto op : producerOps) {
    setAsyncTaskIds(op, producerTaskIds);
  }

  for (auto op : consumerOps) {
    setAsyncTaskIds(op, consumerTaskIds);
  }

  LLVM_DEBUG({
    LDBG("After task partition");
    funcOp.dump();
    LDBG("\n");
  });
}

class TritonGPUWSTaskPartitionPass
    : public impl::TritonGPUWSTaskPartitionBase<TritonGPUWSTaskPartitionPass> {
public:
  using impl::TritonGPUWSTaskPartitionBase<
      TritonGPUWSTaskPartitionPass>::TritonGPUWSTaskPartitionBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numConsumerGroups == 0)
      return;
    doPartition(funcOp, numConsumerGroups);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
