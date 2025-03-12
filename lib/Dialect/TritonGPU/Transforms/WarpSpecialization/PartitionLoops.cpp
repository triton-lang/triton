#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

using Partition = WarpSchedule::Partition;

//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

static void getBackwardSlice(Value value, scf::ForOp loop,
                             SetVector<Operation *> &slice,
                             SetVector<BlockArgument> &args) {
  auto arg = dyn_cast<BlockArgument>(value);
  if (arg && arg.getOwner() == loop.getBody()) {
    args.insert(arg);
    return;
  }

  Operation *op = value.getDefiningOp();
  op->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      if (operand.getParentRegion() == &loop.getBodyRegion())
        getBackwardSlice(operand, loop, slice, args);
    }
  });
  slice.insert(op);
}

static FailureOr<Block *> slicePartition(scf::ForOp baseLoop,
                                         const WarpSchedule &baseSchedule,
                                         const Partition *slicePartition) {
  // Generate the partition loop by cloning the whole loop and deleting anything
  // that doesn't belong to the partition and its backward slice. This is easier
  // than trying to generate a new loop from scratch while keeping the
  // operations in the same order.
  scf::ForOp loop = baseLoop.clone();
  WarpSchedule schedule = *WarpSchedule::deserialize(loop);
  Partition *partition = schedule.getPartition(slicePartition->getIndex());

  // Add the ops in the partition to the slice and take the backward slice of
  // their dependencies.
  SetVector<Operation *> iterSlice;
  iterSlice.insert(partition->getOps().begin(), partition->getOps().end());
  SetVector<BlockArgument> iterArgs;
  schedule.iterateInputs(loop, partition, [&](OpOperand &operand) {
    getBackwardSlice(operand.get(), loop, iterSlice, iterArgs);
  });

  // Recurse on the block arguments.
  for (unsigned i = 0; i < iterArgs.size(); ++i) {
    BlockArgument arg = iterArgs[i];
    if (arg == loop.getInductionVar())
      continue;
    unsigned idx = arg.getArgNumber() - 1;
    getBackwardSlice(loop.getYieldedValues()[idx], loop, iterSlice, iterArgs);
  }

  // Check that we did this right. If the schedule is valid, all ops in the
  // slice should belong to either the root partition or the current partition.
  assert(llvm::all_of(iterSlice, [&](Operation *op) {
    Partition *opPartition = schedule.getPartition(op);
    return opPartition == schedule.getRootPartition() ||
           opPartition == partition;
  }));

  // Check for results that need to be passed back into the default warp group.
  llvm::MapVector<unsigned, SmallVector<OpOperand *>> resultUses;
  baseSchedule.iterateUses(
      baseLoop, slicePartition,
      [&](OpResult output, OpOperand &use, unsigned distance) {
        if (distance != -1)
          return;
        auto result = cast<OpResult>(use.get());
        assert(result.getDefiningOp() == baseLoop);
        iterArgs.insert(loop.getRegionIterArg(result.getResultNumber()));
        resultUses[result.getResultNumber()].push_back(&use);
      },
      /*includeResults=*/true);

  // Pass these results through shared memory.
  Block *block = new Block;
  block->push_back(loop);
  for (auto &[resultIdx, uses] : resultUses) {
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
  for (Operation &op :
       llvm::make_early_inc_range(loop.getBody()->without_terminator())) {
    if (iterSlice.contains(&op))
      continue;
    op.dropAllUses();
    op.erase();
  }
  llvm::BitVector toErase(loop.getNumRegionIterArgs());
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    if (iterArgs.contains(arg))
      continue;
    toErase.set(i);
    arg.dropAllUses();
  }
  eraseLoopCarriedValues(loop, std::move(toErase));

  return block;
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
      const Partition *usePartition = schedule.getPartition(use.getOwner());
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

  // This loop has no partitions.
  if (schedule.getPartitions().begin() == schedule.getPartitions().end())
    return success();

  // Always assign the first partition to the default warp group.
  const Partition &defaultPartition = *schedule.getPartition(0u);
  SmallVector<Block *> partitionBlocks;
  for (const Partition &partition :
       llvm::drop_begin(schedule.getPartitions())) {
    FailureOr<Block *> blockOr = slicePartition(loop, schedule, &partition);
    if (failed(blockOr))
      return failure();
    partitionBlocks.push_back(*blockOr);
  }

  // Now delete everything except the default and root partitions from the base
  // loop.
  DenseSet<BlockArgument> argsToKeep;
  for (Operation &op :
       llvm::make_early_inc_range(loop.getBody()->without_terminator())) {
    const Partition *opPartition = schedule.getPartition(&op);
    if (opPartition != &defaultPartition &&
        opPartition != schedule.getRootPartition()) {
      op.dropAllUses();
      op.erase();
      continue;
    }
    for (OpOperand &use : op.getUses()) {
      if (use.getOwner() == loop.getBody()->getTerminator())
        argsToKeep.insert(loop.getRegionIterArg(use.getOperandNumber()));
    }
  }
  llvm::BitVector toErase(loop.getNumRegionIterArgs());
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    if (argsToKeep.contains(arg))
      continue;
    toErase.set(i);
    arg.dropAllUses();
  }
  eraseLoopCarriedValues(loop, std::move(toErase));

  // Figure out how many warps each partition needs. For now, this is either 1
  // or the number of warps.
  SmallVector<int32_t> partitionNumWarps;
  int32_t functionNumWarps = lookupNumWarps(loop);
  auto isTensor = [](Type t) { return isa<RankedTensorType>(t); };
  for (Block *block : partitionBlocks) {
    int32_t numWarps = 1;
    if (llvm::any_of(*block, [&](Operation &op) {
          return llvm::any_of(op.getOperandTypes(), isTensor) ||
                 llvm::any_of(op.getResultTypes(), isTensor);
        }))
      numWarps = functionNumWarps;
    partitionNumWarps.push_back(numWarps);
  }

  ImplicitLocOpBuilder b(loop.getLoc(), loop);
  auto wsOp = b.create<WarpSpecializeOp>(
      loop.getResultTypes(), partitionNumWarps, partitionBlocks.size());
  loop.replaceAllUsesWith(wsOp);
  Block *defaultBlock = b.createBlock(&wsOp.getDefaultRegion());
  loop->moveBefore(defaultBlock, defaultBlock->end());
  b.setInsertionPointAfter(loop);
  b.create<WarpYieldOp>(loop.getResults());
  for (auto [region, block] :
       llvm::zip(wsOp.getPartitionRegions(), partitionBlocks)) {
    region->push_back(block);
    b.setInsertionPointToEnd(block);
    b.create<WarpReturnOp>();
  }

  SetVector<Value> captures;
  getUsedValuesDefinedAbove(wsOp.getPartitionOpHolder(), captures);
  for (Value capture : captures) {
    // Rematerialize constants.
    if (capture.getDefiningOp() &&
        capture.getDefiningOp()->hasTrait<OpTrait::ConstantLike>()) {
      for (Region *region : wsOp.getPartitionRegions()) {
        b.setInsertionPointToStart(&region->front());
        Value copy = b.clone(*capture.getDefiningOp())->getResult(0);
        replaceAllUsesInRegionWith(capture, copy, *region);
      }
      continue;
    }

    if (isa<RankedTensorType>(capture.getType())) {
      return mlir::emitWarning(capture.getLoc(),
                               "FIXME: capturing tensor values into warp "
                               "partitions is not supported");
    }
    wsOp->insertOperands(wsOp.getNumOperands(), capture);
    for (Region *region : wsOp.getPartitionRegions()) {
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
      continue;
  }
}
