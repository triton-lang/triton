#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
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
  llvm::BitVector toErase(loop.getNumRegionIterArgs(), true);
  for (Operation &op :
       llvm::make_early_inc_range(loop.getBody()->without_terminator())) {
    const Partition *opPartition = schedule.getPartition(&op);
    if (!llvm::is_contained({partition, schedule.getRootPartition()},
                            opPartition)) {
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
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    if (toErase.test(i))
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
// shrinkWarps
//===----------------------------------------------------------------------===//

static LogicalResult shrinkWarps(Region *region, unsigned numWarps) {
  OpBuilder b(region->getContext());
  OwningOpRef<ModuleOp> container = ModuleOp::create(region->getLoc());
  Block &containerBlock = container->getBodyRegion().front();

  b.setInsertionPointToStart(&containerBlock);
  auto containerFunc =
      b.create<FuncOp>(region->getLoc(), "container",
                       b.getFunctionType(region->getArgumentTypes(), {}));
  containerFunc.getBody().takeBody(*region);

  mlir::AttrTypeReplacer wipeout;
  wipeout.addReplacement([](RankedTensorType ty) {
    return RankedTensorType::get(ty.getShape(), ty.getElementType());
  });
  wipeout.addReplacement([](TensorDescType ty) -> std::pair<Type, WalkResult> {
    return {ty, WalkResult::skip()};
  });
  wipeout.recursivelyReplaceElementsIn(*container, /*replaceAttrs=*/false,
                                       /*replaceLocs=*/false,
                                       /*replaceTypes=*/true);

  PassManager pm(region->getContext());
  pm.enableVerifier(false);

  pm.addPass(createConvertTritonToTritonGPUPass("cuda:100", numWarps));
  pm.addPass(createTritonGPUCoalesce());
  pm.addPass(createTritonGPURemoveLayoutConversions());
  pm.addPass(createTritonGPUOptimizeThreadLocality());
  pm.addPass(createTritonGPURemoveLayoutConversions());
  if (failed(pm.run(*container)))
    return failure();

  region->takeBody(containerFunc.getBody());
  return success();
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
  auto wsOp = b.create<WarpSpecializeOp>(loop.getResultTypes(), std::nullopt,
                                         partitionBlocks.size());
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
  for (unsigned i = 0; i < captures.size(); ++i) {
    Value capture = captures[i];

    // Rematerialize constants and also pure tensor ops to get around the
    // restriction below on capturing tensors.
    Operation *defOp = capture.getDefiningOp();
    if (defOp && isPure(defOp) &&
        (defOp->hasTrait<OpTrait::ConstantLike>() ||
         isa<RankedTensorType>(capture.getType()))) {
      captures.insert(defOp->operand_begin(), defOp->operand_end());
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

  // Run DCE on the isolated regions to clean up the IR before determining the
  // number of warps and registers.
  MLIRContext *ctx = loop.getContext();
  RewritePatternSet patterns(ctx);
  populateForOpDeadArgumentElimination(patterns);
  scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
  scf::IfOp::getCanonicalizationPatterns(patterns, ctx);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  // Figure out how many warps each partition needs. For now, this is either 1
  // or the number of warps.
  SmallVector<int32_t> partitionNumWarps;
  int32_t functionNumWarps = lookupNumWarps(loop);
  auto isTensor = [](Type t) { return isa<RankedTensorType>(t); };
  for (Region *region : wsOp.getPartitionRegions()) {
    if (failed(applyPatternsGreedily(*region, frozenPatterns)))
      return failure();
    WalkResult result = region->walk([&](Operation *op) {
      if (llvm::any_of(op->getOperandTypes(), isTensor) ||
          llvm::any_of(op->getResultTypes(), isTensor))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    partitionNumWarps.push_back(result.wasInterrupted() ? functionNumWarps : 1);
  }
  wsOp.setPartitionNumWarps(partitionNumWarps);

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
  // SmallVector<scf::ForOp> loops;
  // getOperation().walk([&](scf::ForOp loop) {
  //  if (loop->hasAttrOfType<ArrayAttr>(kPartitionStagesAttrName))
  //    loops.push_back(loop);
  //});

  // for (scf::ForOp loop : loops) {
  //   if (failed(partitionLoop(loop)))
  //     continue;
  // }
  getOperation().walk([](WarpSpecializeOp op) {
    SmallVector<int32_t> newNumWarps;
    for (auto [numWarps, region] :
         llvm::zip(op.getPartitionNumWarps(), op.getPartitionRegions())) {
      if (numWarps == 8) {
        (void)shrinkWarps(region, 1);
        newNumWarps.push_back(1);
      } else {
        newNumWarps.push_back(numWarps);
      }
    }
    op.setPartitionNumWarps(newNumWarps);
  });
}
