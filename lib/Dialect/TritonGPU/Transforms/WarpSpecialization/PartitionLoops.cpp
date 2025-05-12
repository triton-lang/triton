#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

using Partition = WarpSchedule::Partition;

//===----------------------------------------------------------------------===//
// slicePartition
//===----------------------------------------------------------------------===//

// Given a loop, erase ops and loop iter args that are not part of the root
// partition or the provided `partition`.
static void eraseOtherPartitions(scf::ForOp &loop, const WarpSchedule &schedule,
                                 const Partition *partition) {
  auto inPartition = [&](Operation *op) {
    const Partition *opPartition =
        schedule.getPartition(loop.getBody()->findAncestorOpInBlock(*op));
    return llvm::is_contained({partition, schedule.getRootPartition()},
                              opPartition);
  };
  llvm::BitVector toErase(loop.getNumRegionIterArgs(), true);
  for (Operation &op :
       llvm::make_early_inc_range(loop.getBody()->without_terminator())) {
    if (!inPartition(&op)) {
      op.dropAllUses();
      op.erase();
      continue;
    }
    // Trace uses into the `scf.yield` to mark the corresponding iter arg as
    // used.
    for (OpOperand &use : op.getUses()) {
      if (use.getOwner() == loop.getBody()->getTerminator())
        toErase.reset(use.getOperandNumber());
    }
  }
  for (auto [i, arg, result] :
       llvm::enumerate(loop.getRegionIterArgs(), loop.getResults())) {
    if (llvm::any_of(arg.getUsers(), inPartition) || !result.use_empty())
      toErase.reset(i);
    else if (toErase.test(i))
      arg.dropAllUses();
  }
  eraseLoopCarriedValues(loop, std::move(toErase));
  // Erase the schedule attributes.
  WarpSchedule::eraseFrom(loop);
}

// Given a loop and a scheduled partition, slice a copy of the partition into a
// new loop. This returns the block containing the new loop.
static FailureOr<std::unique_ptr<Block>>
slicePartition(scf::ForOp baseLoop, const WarpSchedule &baseSchedule,
               const Partition *slicePartition) {
  // Generate the partition loop by cloning the whole loop and deleting anything
  // that doesn't belong to the partition and the root partition. This is easier
  // than trying to generate a new loop from scratch while keeping the
  // operations in the same order.
  scf::ForOp loop = baseLoop.clone();
  std::unique_ptr<Block> block = std::make_unique<Block>();
  block->push_back(loop);
  WarpSchedule schedule = *WarpSchedule::deserialize(loop);
  Partition *partition = schedule.getPartition(slicePartition->getIndex());

  // Check for results that need to be passed back into the default warp group.
  SmallVector<unsigned> resultIndices;
  baseSchedule.iterateOutputs(
      baseLoop, slicePartition, [&](Operation *op, OpOperand &use) {
        unsigned idx = use.getOperandNumber();
        // Ignore results with no uses.
        if (isa<scf::YieldOp>(op) && !baseLoop.getResult(idx).use_empty())
          resultIndices.push_back(idx);
      });

  // Pass these results through shared memory.
  for (unsigned resultIdx : resultIndices) {
    Value result = loop.getResult(resultIdx);
    auto ty = dyn_cast<RankedTensorType>(result.getType());
    if (!ty) {
      return mlir::emitWarning(
          result.getLoc(),
          "FIXME: only tensor partition results are supported");
    }
    // Store the result after the loop and load it after the base loop.
    ImplicitLocOpBuilder b(result.getLoc(), baseLoop);
    Value alloc = b.create<LocalAllocOp>(MemDescType::get(
        ty.getShape(), ty.getElementType(), getSharedEncoding(ty),
        SharedMemorySpaceAttr::get(ty.getContext()), /*mutable=*/true));
    b.setInsertionPointAfter(loop);
    b.create<LocalStoreOp>(result, alloc);
    b.setInsertionPointAfter(baseLoop);
    Value output = b.create<LocalLoadOp>(ty, alloc);
    b.create<LocalDeallocOp>(alloc);
    baseLoop.getResult(resultIdx).replaceAllUsesWith(output);
  }

  // Delete everything else.
  eraseOtherPartitions(loop, schedule, partition);

  return std::move(block);
}

//===----------------------------------------------------------------------===//
// partitionLoop
//===----------------------------------------------------------------------===//

LogicalResult triton::gpu::partitionLoop(scf::ForOp loop) {
  FailureOr<WarpSchedule> scheduleOr = WarpSchedule::deserialize(loop);
  if (failed(scheduleOr))
    return failure();
  WarpSchedule schedule = std::move(*scheduleOr);
  if (failed(schedule.verify(loop)))
    return failure();

  // Only the root node should have consumers at this point.
  for (const Partition &partition : schedule.getPartitions()) {
    bool failed = false;
    auto callback = [&](OpResult output, OpOperand &use, unsigned distance) {
      Operation *owner = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      const Partition *usePartition = schedule.getPartition(owner);
      if (usePartition == schedule.getRootPartition() ||
          usePartition == &partition)
        return;
      failed = true;
      InFlightDiagnostic diag =
          mlir::emitWarning(output.getLoc(), "non-root partition #")
          << partition.getIndex() << " has direct SSA consumer";
      diag.attachNote(use.getOwner()->getLoc())
          << "use at distance " << distance << " in partition #"
          << usePartition->getIndex() << " here";
    };
    schedule.iterateUses(loop, &partition, callback);
    if (failed)
      return failure();
  }

  // There is nothing to do if the loop has 1 or fewer partitions.
  if (llvm::size(schedule.getPartitions()) <= 1)
    return success();

  // Always assign the first partition to the default warp group.
  const Partition &defaultPartition = *schedule.getPartition(0u);
  SmallVector<std::unique_ptr<Block>> partitionBlocks;
  for (const Partition &partition :
       llvm::drop_begin(schedule.getPartitions())) {
    FailureOr<std::unique_ptr<Block>> blockOr =
        slicePartition(loop, schedule, &partition);
    if (failed(blockOr))
      return failure();
    partitionBlocks.push_back(std::move(*blockOr));
  }

  // Now delete everything except the default and root partitions from the base
  // loop.
  eraseOtherPartitions(loop, schedule, &defaultPartition);

  // Create the warp specialize op and move in the partition blocks.
  ImplicitLocOpBuilder b(loop.getLoc(), loop);
  int32_t functionNumWarps = lookupNumWarps(loop);
  SmallVector<int32_t> partitionNumWarps(partitionBlocks.size(),
                                         functionNumWarps);
  auto wsOp = b.create<WarpSpecializeOp>(
      loop.getResultTypes(), partitionNumWarps, partitionBlocks.size());
  loop.replaceAllUsesWith(wsOp);
  Block *defaultBlock = b.createBlock(&wsOp.getDefaultRegion());
  loop->moveBefore(defaultBlock, defaultBlock->end());
  b.setInsertionPointAfter(loop);
  b.create<WarpYieldOp>(loop.getResults());
  for (auto [region, block] :
       llvm::zip(wsOp.getPartitionRegions(), partitionBlocks)) {
    region->push_back(block.release());
    b.setInsertionPointToEnd(&region->front());
    b.create<WarpReturnOp>();
  }

  // The capture set is the same for every partition region, so now find the
  // captures and thread them in to the regions.
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(wsOp.getPartitionOpHolder(), captures);

  // Find the subgraph that should be cloned into the partition regions. The
  // explicit captures are the leaves of the subgraph.
  SetVector<Operation *> opsToClone;
  SmallVector<Value> explicitCaptures;
  for (unsigned i = 0; i < captures.size(); ++i) {
    Value capture = captures[i];

    // Rematerialize constants and also pure tensor ops to get around the
    // restriction below on capturing tensors.
    Operation *defOp = capture.getDefiningOp();
    if (defOp && isPure(defOp) &&
        (defOp->hasTrait<OpTrait::ConstantLike>() ||
         isa<RankedTensorType>(capture.getType()))) {
      captures.insert(defOp->operand_begin(), defOp->operand_end());
      opsToClone.insert(defOp);
      continue;
    }

    // Explicitly pass tensor captures through shared memory.
    auto tensorTy = dyn_cast<RankedTensorType>(capture.getType());
    if (tensorTy) {
      SharedEncodingTrait sharedEnc = getSharedEncoding(tensorTy);
      ImplicitLocOpBuilder b(capture.getLoc(), wsOp);
      auto memdescTy = MemDescType::get(
          tensorTy.getShape(), tensorTy.getElementType(), sharedEnc,
          SharedMemorySpaceAttr::get(tensorTy.getContext()));
      auto alloc = b.create<LocalAllocOp>(memdescTy, capture);
      for (Region *region : wsOp.getPartitionRegions()) {
        b.setInsertionPointToStart(&region->front());
        Value value = b.create<LocalLoadOp>(tensorTy, alloc);
        replaceAllUsesInRegionWith(capture, value, *region);
      }
      capture = alloc;
    }

    explicitCaptures.push_back(capture);
  }

  // Clone the ops into each region in topological order.
  opsToClone = topologicalSort(opsToClone);
  for (Region *region : wsOp.getPartitionRegions()) {
    b.setInsertionPointToStart(&region->front());
    IRMapping mapping;
    for (Operation *op : opsToClone) {
      Value copy = b.clone(*op, mapping)->getResult(0);
      mapping.map(op->getResult(0), copy);
      replaceAllUsesInRegionWith(op->getResult(0), copy, *region);
    }
  }

  // Replace the leaves with explicit captures.
  wsOp->insertOperands(wsOp.getNumOperands(), explicitCaptures);
  for (Region *region : wsOp.getPartitionRegions()) {
    for (Value capture : explicitCaptures) {
      BlockArgument arg =
          region->addArgument(capture.getType(), capture.getLoc());
      replaceAllUsesInRegionWith(capture, arg, *region);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUPARTITIONLOOPS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct PartitionLoops
    : triton::gpu::impl::TritonGPUPartitionLoopsBase<PartitionLoops> {
  using TritonGPUPartitionLoopsBase::TritonGPUPartitionLoopsBase;

  void runOnOperation() override;
};
} // namespace

void PartitionLoops::runOnOperation() {
  // Collect for loops to warp specialize. This pass expects the loop to already
  // be scheduled.
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttrOfType<ArrayAttr>(kPartitionStagesAttrName))
      loops.push_back(loop);
  });

  for (scf::ForOp loop : loops) {
    if (failed(partitionLoop(loop)))
      return signalPassFailure();
  }
}
