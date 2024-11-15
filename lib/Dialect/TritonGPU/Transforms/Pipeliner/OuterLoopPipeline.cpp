#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

// create the schedule for a matmul loop. This is ad hoc based on how we know
// matmul loops should be pipelined and is not a generic scheduler.
static std::vector<std::pair<Operation *, unsigned>>
createSchedule(scf::ForOp forOp, int numStages) {
  SmallVector<Operation *> insertOps;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::AsyncCopyGlobalToLocalOp, ttg::AsyncCommitGroupOp>(op))
      insertOps.emplace_back(&op);
  }
  DenseSet<Operation *> insertAndDeps;
  for (Operation *op : insertOps) {
    tt::addDep(op, insertAndDeps, true);
  }

  DenseSet<Operation *> epilogue;
  bool foundLoop = false;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (insertAndDeps.count(&op))
      continue;
    if (isa<scf::ForOp>(op))
      foundLoop = true;
    if (isa<scf::ForOp, ttg::AsyncWaitOp>(op))
      continue;
    if (foundLoop)
      epilogue.insert(&op);
  }

  std::vector<std::pair<Operation *, unsigned>> schedule;
  // Schedule stage 1 first.
  tt::addOps(forOp, 1, schedule, [&](Operation *op) {
    return insertAndDeps.count(op) == 0 && epilogue.count(op) == 0;
  });

  // Then Schedule stage 0.
  tt::addOps(forOp, 0, schedule,
             [&](Operation *op) { return insertAndDeps.count(op); });

  // Then schedule the epilogue in stage 1
  tt::addOps(forOp, 1, schedule,
             [&](Operation *op) { return epilogue.count(op); });
  return schedule;
}

// pre-process the loop by hosting allocations/deallocation out of the
// loop.
static void hoistAllocAndConst(scf::ForOp forOp) {
  SmallVector<Operation *> toHoist;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto allocOp = dyn_cast<ttg::LocalAllocOp>(op)) {
      // We hoist the allocOp only if it is created by the inner loop
      // pipelining.
      if (!allocOp.getSrc())
        toHoist.push_back(&op);
    } else if (isa<arith::ConstantOp>(op)) {
      toHoist.push_back(&op);
    }
  }
  for (Operation *op : toHoist) {
    op->moveBefore(forOp);
    auto allocOp = dyn_cast<ttg::LocalAllocOp>(op);
    if (!allocOp)
      continue;
    for (Operation *user : allocOp->getUsers()) {
      if (auto dealloc = dyn_cast<ttg::LocalDeallocOp>(user)) {
        dealloc->moveAfter(forOp);
      }
    }
  }
}

// TODO: Revert name change when possible
static bool preConditionOuter(scf::ForOp forOp) {
  // Check if there is a dependency from the loop to the async copy op. In this
  // case we cannot pipeline the async copy.
  SmallVector<Operation *> insertOps;
  int numForOps = 0;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::AsyncCopyGlobalToLocalOp, ttg::AsyncCommitGroupOp>(op))
      insertOps.emplace_back(&op);
    if (isa<scf::ForOp>(op))
      numForOps++;
  }
  if (insertOps.empty() || numForOps != 1)
    return false;
  DenseSet<Operation *> insertAndDeps;
  for (Operation *op : insertOps) {
    tt::addDep(op, insertAndDeps, true);
  }
  // If there is a recurrence containing both the async and the for op we cannot
  // pipeline.
  for (Operation *op : insertAndDeps) {
    if (isa<scf::ForOp>(op))
      return false;
  }
  return true;
}

bool mlir::triton::getOuterLoopSchedule(
    scf::ForOp &forOp, int numStages, mlir::triton::PipeliningOption &options) {
  assert(numStages == 2 && "only support 2 stage pipelining for now");
  // 1. Check precondition, we cannot have a recurrence involving async cp ops
  if (!preConditionOuter(forOp))
    return false;

  // 2. pre-process the loop by hosting allocations.
  hoistAllocAndConst(forOp);

  // 3. Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      createSchedule(forOp, numStages);

  // 4. Fill out the pipeline options.
  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = mlir::triton::predicateOp;
  options.supportDynamicLoops = true;
  return true;
}
