#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Get the earliest user of a value, assuming all users are in the same block.
static Operation *getEarliestUser(ArrayRef<OpOperand *> uses) {
  OpOperand *use = *llvm::min_element(uses, [](OpOperand *lhs, OpOperand *rhs) {
    return lhs->getOwner()->isBeforeInBlock(rhs->getOwner());
  });
  return use->getOwner();
}

//===----------------------------------------------------------------------===//
// UseInfo
//===----------------------------------------------------------------------===//

namespace {
// Use information for a partition SSA output.
struct UseInfo {
  // Get the maximum distance to a use, according for stage and iteration,
  // given the partition where the value is defined.
  int getMaxUseDistance(const Partition &partitition);

  // Map from partition and distance to the uses in that partition.
  llvm::MapVector<std::pair<Partition *, unsigned>, SmallVector<OpOperand *>>
      consumers;
};
} // namespace

int UseInfo::getMaxUseDistance(const Partition &partition) {
  int maxDistance = 0;
  for (auto [usePartition, distance] : llvm::make_first_range(consumers)) {
    int dist = 1 + distance;
    maxDistance = std::max(maxDistance, dist);
  }
  return maxDistance;
}
//===----------------------------------------------------------------------===//
// AsyncRef
//===----------------------------------------------------------------------===//

namespace {
struct AsyncRef {
  auto putView(PartitionBuilder &b, Partition &partition,
               StageCluster srcStageCluster) {
    auto zero = b.create<arith::ConstantOp>(b.getI32IntegerAttr(0));
    auto enterOp = b.createInto<triton::nvws::ArefPutEnterOp>(
        partition, srcStageCluster, viewType, tokenType, aref, zero, zero);
    auto token = enterOp.getToken();

    auto exitOp = [this, &partition, srcStageCluster,
                   token](PartitionBuilder &b) {
      auto zero = b.create<arith::ConstantOp>(b.getI32IntegerAttr(0));
      auto exitOp = b.createInto<triton::nvws::ArefPutExitOp>(
          partition, srcStageCluster, aref, token, zero,
          b.getArrayAttr(SmallVector<Attribute>{triton::nvws::AsyncOpAttr::get(
              aref.getContext(), triton::nvws::AsyncOp::NONE)}));
    };
    return std::make_tuple(enterOp.getResult(0), exitOp);
  }

  auto getView(PartitionBuilder &b, Partition &partition,
               StageCluster srcStageCluster) {
    auto zero = b.create<arith::ConstantOp>(b.getI32IntegerAttr(0));
    auto enterOp = b.createInto<triton::nvws::ArefGetEnterOp>(
        partition, srcStageCluster, viewType, tokenType, aref, zero, zero);
    auto token = enterOp.getToken();

    auto exitOp = [this, &partition, srcStageCluster,
                   token](PartitionBuilder &b) {
      auto zero = b.create<arith::ConstantOp>(b.getI32IntegerAttr(0));
      auto exitOp = b.createInto<triton::nvws::ArefGetExitOp>(
          partition, srcStageCluster, aref, token, zero,
          b.getArrayAttr(SmallVector<Attribute>{triton::nvws::AsyncOpAttr::get(
              aref.getContext(), triton::nvws::AsyncOp::NONE)}));
    };
    return std::make_tuple(enterOp.getResult(0), exitOp);
  }

  Value aref;
  MemDescType viewType;
  AsyncTokenType tokenType;
};

//===----------------------------------------------------------------------===//
// DependencyRewriter
//===----------------------------------------------------------------------===//

// Helper class for dependency rewriting.
class DependencyRewriter {
public:
  DependencyRewriter(WarpSchedule &schedule, scf::ForOp &loop)
      : schedule(schedule), loop(loop), b(loop.getLoc(), loop),
        endBuilder(loop.getLoc(), loop->getNextNode()) {}

  // Partition the loop.
  LogicalResult run();

private:
  AsyncRef allocateAsyncValue(RankedTensorType tensorType,
                              unsigned maxDistance);

  // The schedule to apply.
  WarpSchedule &schedule;
  // The loop to partition.
  scf::ForOp &loop;
  // The builders to use.
  PartitionBuilder b, endBuilder;
};
} // namespace

AsyncRef DependencyRewriter::allocateAsyncValue(RankedTensorType tensorType,
                                                unsigned maxDistance) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);
  unsigned numBars = maxDistance;
  Value alloc = createAlloc(loop, tensorType, b.getLoc(),
                            getSharedEncoding(tensorType), numBars);
  auto allocType = cast<MemDescType>(alloc.getType());
  auto arefTy = triton::nvws::ArefType::get(
      b.getContext(),
      triton::nvws::TypeArrayAttr::get(b.getContext(), alloc.getType()));
  auto aref = b.create<triton::nvws::ArefCreateOp>(b.getLoc(), arefTy, alloc);

  return AsyncRef{aref, getBufferViewType(allocType),
                  b.getType<AsyncTokenType>()};
}

LogicalResult DependencyRewriter::run() {
  SmallVector<llvm::MapVector<Value, UseInfo>> partitionUseInfo;

  for (const Partition &partition : schedule.getPartitions()) {
    // Find all consumers of all outputs of this partition, tracking the
    // specific Partition
    auto &useInfo = partitionUseInfo.emplace_back();
    SmallVector<std::tuple<Value, OpOperand *, unsigned>> uses;

    std::function<void(OpOperand &)> collectUses;
    collectUses = [&](OpOperand &use) {
      Operation *owner = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      if (isa<scf::YieldOp>(owner)) {
        // This value is used in a subsequent iteration.
        // collect the uses of the appropriate loop arg
        for (auto &newUse : loop.getBody()
                                ->getArgument(use.getOperandNumber() + 1)
                                .getUses()) {
          collectUses(newUse);
        }
      } else if (schedule.getPartition(owner) != &partition) {
        // This value is used in a different partition in the same iteration.
        uses.emplace_back(use.get(), &use, 0);
      }
    };
    for (Operation *op : partition.getOps()) {
      for (OpOperand &use : op->getUses()) {
        collectUses(use);
      }
    }

    auto callback = [&](Value output, OpOperand &use, unsigned distance) {
      Operation *user = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      Partition *usePartition = schedule.getPartition(user);
      // Ignore uses in the same partition in the future.
      if (usePartition == &partition) {
        assert(distance > 0 && "self-recursion must occur in the future");
        return;
      }
      Operation *owner = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      UseInfo &info = useInfo[output];
      info.consumers[{usePartition, distance}].push_back(&use);
    };

    while (!uses.empty()) {
      auto [output, use, distance] = uses.pop_back_val();
      Operation *owner =
          loop.getBody()->findAncestorOpInBlock(*use->getOwner());
      if (!isa<scf::YieldOp>(owner))
        callback(output, *use, distance);
    }
  }

  // Cut all SSA dependencies by passing outputs through shared memory.
  for (auto [partition, useInfo] :
       llvm::zip(schedule.getPartitions(), partitionUseInfo)) {
    // The amount of buffering is based on the longest distance to a user.
    for (auto &[output, info] : useInfo) {
      // FIXME: No IR support for passing simple scalars through shared
      // memory.
      auto tensorType = dyn_cast<RankedTensorType>(output.getType());
      if (!tensorType) {
        return mlir::emitWarning(output.getLoc(),
                                 "FIXME: only tensor SSA dependencies between "
                                 "partitions are supported");
      }

      Operation *defOp;
      Value tmp = output;
      while (true) {
        if (auto arg = dyn_cast<BlockArgument>(tmp)) {
          tmp = loop.getBody()->getTerminator()->getOperand(arg.getArgNumber() -
                                                            1);
          continue;
        }
        defOp = tmp.getDefiningOp();
        break;
      }

      // Buffer the value based on the greatest distance to a consumer
      // partition.
      int maxDistance = info.getMaxUseDistance(partition);

      // Allocate buffers for the value and its associated barriers.
      b.setLoc(output.getLoc());
      ImplicitLocOpBuilder endBuilder(b.getLoc(), loop->getNextNode());
      AsyncRef aref = allocateAsyncValue(tensorType, maxDistance);

      unsigned numConsumers = info.consumers.size();

      for (auto &[key, uses] : info.consumers) {
        assert(!uses.empty() && "expected at least one use");
        Operation *earliestUser = getEarliestUser(uses);
        b.setInsertionPoint(
            loop.getBody()->findAncestorOpInBlock(*earliestUser));

        auto [usePartition, distance] = key;

        // Wait for the value to be available.
        StageCluster sinkSrcCluster = getStageCluster(earliestUser);
        auto [view, exitOp] = aref.getView(b, *usePartition, sinkSrcCluster);
        // Load the value at the current index and replace uses in this
        // partition with it.
        Value value = b.createInto<LocalLoadOp>(*usePartition, sinkSrcCluster,
                                                tensorType, view);
        for (OpOperand *use : uses)
          use->set(value);
        exitOp(b);
      }

      // Set up production of the value
      if (isa<BlockArgument>(output))
        b.setInsertionPointToStart(loop.getBody());
      else
        b.setInsertionPointAfter(defOp);

      StageCluster srcStageCluster = getStageCluster(defOp);
      auto [view, exitOp] = aref.putView(b, partition, srcStageCluster);
      b.createInto<LocalStoreOp>(partition, srcStageCluster, output, view);
      exitOp(b);
    }
  }

  // Rewrite the loop to add the new results. Calling this function with no
  // indices set will just resize the results.
  eraseLoopCarriedValues(loop, {});
  // Update the schedule.
  schedule.serialize(loop);
  return success();
}

//===----------------------------------------------------------------------===//
// rewritePartitionDependenies
//===----------------------------------------------------------------------===//

LogicalResult triton::gpu::rewritePartitionDependencies(scf::ForOp &loop) {
  FailureOr<WarpSchedule> scheduleOr = WarpSchedule::deserialize(loop);
  if (failed(scheduleOr))
    return failure();
  WarpSchedule schedule = std::move(*scheduleOr);
  if (failed(schedule.verify(loop)))
    return failure();
  DependencyRewriter rewriter(schedule, loop);
  if (failed(rewriter.run()))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUREWRITEPARTITIONDEPENDENCIES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct RewritePartitionDependencies
    : triton::gpu::impl::TritonGPURewritePartitionDependenciesBase<
          RewritePartitionDependencies> {
  using TritonGPURewritePartitionDependenciesBase::
      TritonGPURewritePartitionDependenciesBase;

  void runOnOperation() override;
};
} // namespace

void RewritePartitionDependencies::runOnOperation() {
  // Collect for loops to warp specialize. This pass expects the loop to
  // already be scheduled.
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttrOfType<ArrayAttr>(kPartitionStagesAttrName))
      loops.push_back(loop);
  });

  for (scf::ForOp loop : loops) {
    if (failed(rewritePartitionDependencies(loop)))
      return signalPassFailure();
  }
}
