#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

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
  SmallVector<IRMapping> mappings(wsOp.getPartitionNumWarps().size());
  SmallVector<OpBuilder> builders;
  for (Region *region : wsOp.getPartitionRegions())
    builders.push_back(OpBuilder::atBlockBegin(&region->front()));
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
      for (auto [i, region] : llvm::enumerate(wsOp.getPartitionRegions())) {
        Value value =
            builders[i].create<LocalLoadOp>(capture.getLoc(), tensorTy, alloc);
        replaceAllUsesInRegionWith(capture, value, *region);
        mappings[i].map(capture, value);
      }
      capture = alloc;
    }

    explicitCaptures.push_back(capture);
  }

  // Clone the ops into each region in topological order.
  opsToClone = topologicalSort(opsToClone);
  for (auto [i, region] : llvm::enumerate(wsOp.getPartitionRegions())) {
    OpBuilder &b = builders[i];
    IRMapping &mapping = mappings[i];
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

struct WgBuilder {
  OpBuilder builder;
  IRMapping mapping;
  size_t partitionId;
};

bool isUsedBy(Value arg, const Partition *partition, scf::ForOp loop,
              const WarpSchedule &schedule) {
  auto inPartition = [&](Operation *op) {
    const Partition *opPartition =
        schedule.getPartition(loop.getBody()->findAncestorOpInBlock(*op));
    return llvm::is_contained({partition, schedule.getRootPartition()},
                              opPartition);
  };
  return llvm::any_of(arg.getUsers(), inPartition);
}

void cloneOpsInBlock(Block *block, SmallVector<WgBuilder> &builders,
                     const WarpSchedule &schedule);

void cloneForOp(scf::ForOp forOp, SmallVector<WgBuilder> &builders,
                const WarpSchedule &schedule) {
  SmallVector<scf::ForOp> newForOps;
  for (size_t i = 0; i < builders.size(); ++i) {
    auto &b = builders[i];
    auto partition = schedule.getPartition(i);
    // TODO: Mapping?
    auto lb = forOp.getLowerBound();
    auto ub = forOp.getUpperBound();
    auto step = forOp.getStep();
    SmallVector<Value> initArgs;
    for (auto [idx, arg] : llvm::enumerate(forOp.getInitArgs())) {
      if (isUsedBy(forOp.getRegionIterArgs()[idx], partition, forOp,
                   schedule) ||
          (i == 0 && !forOp.getResult(idx).use_empty())) {
        initArgs.push_back(arg);
      }
    }
    auto newForOp =
        b.builder.create<scf::ForOp>(forOp.getLoc(), lb, ub, step, initArgs);
    newForOp->setAttrs(forOp->getAttrs());
    newForOps.push_back(newForOp);

    b.mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    // map the results of the forOp to the newForOp
    auto oldIterArgs = forOp.getRegionIterArgs();
    auto newIterArgs = newForOp.getRegionIterArgs();
    for (int oldIdx = 0, newIdx = 0; oldIdx < oldIterArgs.size(); ++oldIdx) {
      auto oldArg = oldIterArgs[oldIdx];
      if (isUsedBy(oldArg, partition, forOp, schedule) ||
          (i == 0 && !forOp.getResult(oldIdx).use_empty())) {
        auto newArg = newIterArgs[newIdx++];
        b.mapping.map(oldArg, newArg);
      }
    }

    for (int oldIdx = 0, newIdx = 0; oldIdx < forOp.getResults().size();
         ++oldIdx) {
      if (isUsedBy(forOp.getRegionIterArgs()[oldIdx], partition, forOp,
                   schedule) ||
          (i == 0 && !forOp.getResult(oldIdx).use_empty())) {
	auto oldArg = forOp.getResult(oldIdx);
        auto newArg = newForOp.getResult(newIdx++);
        b.mapping.map(oldArg, newArg);
      }
    }
    // set builder insertion point to the start of the newForOp body
    b.builder.setInsertionPointToStart(newForOp.getBody());
  }

  // resursive clone ops in the forOp body
  cloneOpsInBlock(forOp.getBody(), builders, schedule);

  for (auto newForOp: newForOps) {
    WarpSchedule::eraseFrom(newForOp);
  }
}

void cloneIfOp(scf::IfOp ifOp, SmallVector<WgBuilder> &builders,
               const WarpSchedule &schedule) {
  auto partition = schedule.getPartition(ifOp);
  SmallVector<size_t> builderIndices;

  if (partition == schedule.getRootPartition()) {
    for (size_t i = 0; i < builders.size(); ++i) {
      builderIndices.push_back(i);
    }
  } else {
    builderIndices.push_back(partition->getIndex());
  }

  SmallVector<scf::IfOp> newIfOps;
  for (size_t idx : builderIndices) {
    auto& b = builders[idx];
    SmallVector<Type> newIfResultTypes;
    for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
      newIfResultTypes.push_back(result.getType());
    }
    auto cond = b.mapping.lookupOrDefault(ifOp.getCondition());
    auto newIfOp = b.builder.create<scf::IfOp>(
        ifOp.getLoc(), newIfResultTypes, cond, ifOp.elseBlock() ? true : false);
    newIfOp->setAttrs(ifOp->getAttrs());
    newIfOps.push_back(newIfOp);

    // map results
    for (int oldIdx = 0, newIdx = 0; oldIdx < ifOp.getResults().size();
         ++oldIdx) {
      auto oldArg = ifOp.getResult(oldIdx);
      auto newArg = newIfOp.getResult(newIdx++);
      b.mapping.map(oldArg, newArg);
    }
    // map block args
    for (auto [oldArg, newArg] :
         llvm::zip(ifOp.thenBlock()->getArguments(),
                   newIfOp.thenBlock()->getArguments())) {
      b.mapping.map(oldArg, newArg);
    }
    if (ifOp.elseBlock()) {
      for (auto [oldArg, newArg] :
           llvm::zip(ifOp.elseBlock()->getArguments(),
                     newIfOp.elseBlock()->getArguments())) {
        b.mapping.map(oldArg, newArg);
      }
    }
    b.builder.setInsertionPointToStart(newIfOp.thenBlock());
  }

  cloneOpsInBlock(ifOp.thenBlock(), builders, schedule);

  if (auto elseBlock = ifOp.elseBlock()) {
    for (auto [builder, newIfOp] : llvm::zip(builders, newIfOps)) {
      builder.builder.setInsertionPointToStart(newIfOp.elseBlock());
    }
    cloneOpsInBlock(elseBlock, builders, schedule);
  }

  for (auto [idx, newIfOp] : llvm::zip(builderIndices, newIfOps)) {
    builders[idx].builder.setInsertionPointAfter(newIfOp);
  }
}

void cloneReduceOp(triton::ReduceOp reduceOp, SmallVector<WgBuilder> &builders,
                   const WarpSchedule &schedule) {
  auto partition = schedule.getPartition(reduceOp);
  assert(partition);
  auto& b = builders[partition->getIndex()];

  SmallVector<Value> srcs;
  for (auto src : reduceOp.getSrcs())
    srcs.push_back(b.mapping.lookupOrDefault(src));
  auto axis = reduceOp.getAxis();
  auto newReduceOp =
      b.builder.create<triton::ReduceOp>(reduceOp.getLoc(), srcs, axis);

  for (auto [oldResult, newResult] :
       llvm::zip(reduceOp.getResults(), newReduceOp.getResults())) {
    b.mapping.map(oldResult, newResult);
  }

  auto &region = newReduceOp.getRegion();
  Block *block = &region.emplaceBlock();
  for (auto args : reduceOp.getRegion().getBlocks().front().getArguments()) {
    auto newArg = block->addArgument(args.getType(), args.getLoc());
    b.mapping.map(args, newArg);
  }

  b.builder.setInsertionPointToStart(block);

  cloneOpsInBlock(reduceOp.getBody(), builders, schedule);

  b.builder.setInsertionPointAfter(newReduceOp);
}

const Partition *getPartition(Operation *op, const WarpSchedule &schedule) {
  while (op && !schedule.getPartition(op)) {
    op = op->getParentOp();
  }
  if (op) {
    return schedule.getPartition(op);
  }
  return nullptr;
}

void cloneOpsInBlock(Block *block, SmallVector<WgBuilder> &builders,
                     const WarpSchedule &schedule) {
  for (auto &op_ : *block) {
    auto op = &op_;
    auto partition = getPartition(op, schedule);

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      cloneForOp(forOp, builders, schedule);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      cloneIfOp(ifOp, builders, schedule);
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      cloneReduceOp(reduceOp, builders, schedule);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      if (yieldOp.getOperands().empty()) {
	continue;
      }

      auto yieldParent = yieldOp->getParentOp();

      auto doClone = [&](WgBuilder &builder) {
        SmallVector<Value> newYieldOperands;
        for (auto [i, operand] : llvm::enumerate(yieldOp.getOperands())) {
          bool keep = false;
          if (auto forOp = dyn_cast<scf::ForOp>(yieldParent)) {
            keep =
                isUsedBy(forOp.getRegionIterArgs()[i],
                         schedule.getPartition(builder.partitionId), forOp,
                         schedule) ||
                (builder.partitionId == 0 && !forOp.getResult(i).use_empty());
          } else if (auto ifOp = dyn_cast<scf::IfOp>(yieldParent)) {
            keep = true; // TODO
          } else {
            llvm_unreachable("NYI");
          }

          if (keep) {
            newYieldOperands.push_back(
                builder.mapping.lookupOrDefault(operand));
          }
        }
        builder.builder.create<scf::YieldOp>(op->getLoc(), newYieldOperands);
      };

      if (!partition || partition == schedule.getRootPartition()) {
        for (auto &builder : builders) {
          doClone(builder);
        }
      } else {
        auto &builder = builders[partition->getIndex()];
        doClone(builder);
      }
    } else {
      // all remaining ops are expected to be regionless
      assert(op->getNumRegions() == 0);

      if (!partition) {
	// TODO: Is this possible?
	llvm_unreachable("No partition");
      }

      auto doClone = [&](WgBuilder &builder, Operation *op) {
        auto newOp = builder.builder.clone(*op, builder.mapping);
        for (auto [oldResult, newResult] :
             llvm::zip(op->getResults(), newOp->getResults())) {
          builder.mapping.map(oldResult, newResult);
        }
      };

      if (partition == schedule.getRootPartition()) {
        for (auto &builder : builders) {
          doClone(builder, op);
        }
      } else {
        auto &builder = builders[partition->getIndex()];
        doClone(builder, op);
      }
    }
  }
}

LogicalResult partitionLoopV2(scf::ForOp loop) {
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

  SmallVector<Type> resultTypes;
  auto defaultPartition = schedule.getPartition((int)0);
  SmallVector<int> newResultIdx(loop.getNumRegionIterArgs(), -1);
  for (auto [i, iterArg, result, resultTy] :
       llvm::enumerate(loop.getRegionIterArgs(), loop.getResults(),
                       loop.getResultTypes())) {
    if (isUsedBy(iterArg, defaultPartition, loop, schedule) ||
        !result.use_empty()) {
      newResultIdx[i] = resultTypes.size();
      resultTypes.push_back(resultTy);
    }
  }

  auto numPartitions = schedule.getNumPartitions();
  SmallVector<int32_t> numWarps(numPartitions, lookupNumWarps(loop));
  ImplicitLocOpBuilder b(loop.getLoc(), loop);
  auto wgOp = b.create<nvws::WarpGroupOp>(resultTypes, numWarps, numPartitions);

  SmallVector<WgBuilder> builders;
  for (Region &region : wgOp.getPartitionRegions()) {
    OpBuilder builder = OpBuilder::atBlockEnd(&region.emplaceBlock());
    auto partitionId = builders.size();
    builders.push_back({builder, IRMapping(), partitionId});
  }

  cloneForOp(loop, builders, schedule);

  for (size_t i = 0; i < numPartitions; ++i) {
    auto builder = builders[i].builder;
    auto &region = wgOp.getPartitionRegions()[i];
    auto newForOp = cast<scf::ForOp>(region.front().front());
    builder.setInsertionPointAfter(newForOp);

    if (i == 0) {
      auto outputs = newForOp.getResults();
      builder.create<nvws::WarpGroupYieldOp>(wgOp.getLoc(), outputs);
    } else {
      builder.create<nvws::WarpGroupReturnOp>(wgOp.getLoc());
    }
  }

  for (auto [i, res] : llvm::enumerate(loop.getResults())) {
    if (!res.use_empty()) {
      res.replaceAllUsesWith(wgOp.getResult(newResultIdx[i]));
    }
  }

  loop->erase();

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
    if (failed(partitionLoopV2(loop)))
      return signalPassFailure();
  }

  OpPassManager pm;
  pm.addPass(mlir::triton::createNVWSLowerWarpGroup());

  if (failed(runPipeline(pm, getOperation())))
    return signalPassFailure();
}
