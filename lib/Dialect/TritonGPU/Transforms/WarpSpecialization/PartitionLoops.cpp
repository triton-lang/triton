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

namespace {

struct WarpGroupBuilder : public OpBuilder {
  WarpGroupBuilder(Block *block, Block::iterator insertPoint,
                   size_t partitionId)
      : OpBuilder(block, insertPoint), partitionId(partitionId) {}

  IRMapping mapping;
  size_t partitionId;
};

// This is computed per loop and partition
enum class LoopVarCategory {
  // The given loop variable is not used by the given partition. For example,
  // the use-D flag for MMA is only used by the MMA partition, and thus
  // is `Unused` for any other partition.
  Unused,
  // The given loop variable is used by the given partition. For example, a loop
  // index might be used to compute a relevant stage or phase value for the
  // given partition.
  Used,
  // The results of warp_group op are defined to be those of the first
  // partition. If the original loop results include a tensor which is computed
  // only by a non-default partition, such tensor cannot be returned from the
  // first partition and and must be passed through shared memory. The
  // corresponding loop variable falls into this category.
  // Recognizing this category is necessary for the first partition. For other
  // partitions, some loop variables might be assigned this category, but that
  // information is not used.
  TensorResultFromOtherPartition,
};

bool isTensorResultComputedBy(scf::ForOp loop, size_t resultIdx,
                              const Partition *partition,
                              const PartitionSet &partitions) {
  bool ret = false;
  iterateOutputs(loop, partition, [&](Operation *op, OpOperand &use) {
    if (isa<scf::YieldOp>(op) && use.getOperandNumber() == resultIdx &&
        isa<RankedTensorType>(loop.getResult(resultIdx).getType())) {
      ret = true;
    }
  });
  return ret;
}

SmallVector<size_t> getPartitionIds(Operation *op, size_t numPartitions) {
  auto partitionIds = triton::gpu::getPartitionIds(op);
  if (!partitionIds) {
    SmallVector<size_t> ret(numPartitions);
    std::iota(ret.begin(), ret.end(), 0);
    return ret;
  }
  SmallVector<size_t> ret(partitionIds->begin(), partitionIds->end());
  return ret;
}

SmallVector<LoopVarCategory> classifyLoopVars(scf::ForOp loop,
                                              const Partition *partition,
                                              const PartitionSet &partitions) {
  auto inPartition = [&](Operation *op) {
    auto opPartitionIds = getPartitionIds(op, partitions.getNumPartitions());
    return llvm::is_contained(opPartitionIds, partition->getIndex());
  };
  auto isTensorResultFromOtherPartition = [&](int i) {
    for (auto otherPartition : partitions.getPartitions()) {
      if (&otherPartition == partition) {
        continue;
      }
      if (isTensorResultComputedBy(loop, i, &otherPartition, partitions)) {
        return true;
      }
    }
    return false;
  };

  SmallVector<LoopVarCategory> categories(loop.getNumRegionIterArgs());
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    if (llvm::any_of(arg.getUsers(), inPartition)) {
      categories[i] = LoopVarCategory::Used;
    } else if (isTensorResultFromOtherPartition(i)) {
      categories[i] = LoopVarCategory::TensorResultFromOtherPartition;
    } else {
      categories[i] = LoopVarCategory::Unused;
    }
  }

  return categories;
}

std::pair<SmallVector<size_t>, SmallVector<std::optional<size_t>>>
getLoopVarIndicesToKeep(scf::ForOp loop, const Partition *partition,
                        ArrayRef<LoopVarCategory> loopVarCategories) {
  SmallVector<size_t> indices;
  // The null index means an invalid index, the corresponding loop variable in
  // the original loop is removed in the cloned loop
  SmallVector<std::optional<size_t>> reverseIndices(loop.getNumRegionIterArgs(),
                                                    std::nullopt);
  for (auto [i, arg] : llvm::enumerate(loop.getRegionIterArgs())) {
    // For the default partition, keep non-tensor results used outside of the
    // loop even if the corresponding loop variable is not used in that
    // partition.
    if (loopVarCategories[i] == LoopVarCategory::Used ||
        (partition->getIndex() == 0 && !loop.getResult(i).use_empty() &&
         loopVarCategories[i] !=
             LoopVarCategory::TensorResultFromOtherPartition)) {
      reverseIndices[i] = indices.size();
      indices.push_back(i);
    }
  }
  return std::make_pair(indices, reverseIndices);
}

std::pair<SmallVector<size_t>, SmallVector<std::optional<size_t>>>
getLoopVarIndicesToKeep(scf::ForOp loop, const Partition *partition,
                        const PartitionSet &partitions) {
  auto loopVarCategories = classifyLoopVars(loop, partition, partitions);
  return getLoopVarIndicesToKeep(loop, partition, loopVarCategories);
}

void mapRange(ValueRange fromRange, ValueRange toRange, IRMapping &mapping) {
  for (auto [from, to] : llvm::zip(fromRange, toRange)) {
    mapping.map(from, to);
  }
}

// TODO: remove this
int getPartitionIndex(Operation *op) {
  if (isa<nvws::WarpGroupOp>(op->getParentOp()))
    return op->getParentRegion()->getRegionNumber();
  return getPartitionIndex(op->getParentOp());
}

void cloneOpsInBlock(Block *block, SmallVector<WarpGroupBuilder> &builders,
                     const PartitionSet &partitions);

void cloneForOp(scf::ForOp forOp, SmallVector<WarpGroupBuilder> &builders,
                const PartitionSet &partitions) {
  SmallVector<scf::ForOp> newForOps;
  for (auto [b, partition] : llvm::zip(builders, partitions.getPartitions())) {
    auto [newLoopIndices, _] =
        getLoopVarIndicesToKeep(forOp, &partition, partitions);
    auto lb = b.mapping.lookupOrDefault(forOp.getLowerBound());
    auto ub = b.mapping.lookupOrDefault(forOp.getUpperBound());
    auto step = b.mapping.lookupOrDefault(forOp.getStep());
    SmallVector<Value> initArgs;
    for (auto idx : newLoopIndices) {
      initArgs.push_back(forOp.getInitArgs()[idx]);
    }
    auto newForOp =
        b.create<scf::ForOp>(forOp.getLoc(), lb, ub, step, initArgs);
    newForOp->setAttrs(forOp->getAttrs());
    newForOps.push_back(newForOp);

    b.mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    auto oldIterArgs = forOp.getRegionIterArgs();
    auto newIterArgs = newForOp.getRegionIterArgs();
    for (auto [newIdx, oldIdx] : llvm::enumerate(newLoopIndices)) {
      b.mapping.map(oldIterArgs[oldIdx], newIterArgs[newIdx]);
      b.mapping.map(forOp.getResult(oldIdx), newForOp.getResult(newIdx));
    }

    b.setInsertionPointToStart(newForOp.getBody());
  }

  cloneOpsInBlock(forOp.getBody(), builders, partitions);

  for (auto newForOp : newForOps) {
    builders[getPartitionIndex(newForOp)].setInsertionPointAfter(newForOp);
    newForOp.walk([&](Operation *op) { op->removeAttr(kPartitionAttrName); });
    newForOp->removeAttr(kPartitionStagesAttrName);
  }
}

void cloneIfOp(scf::IfOp ifOp, SmallVector<WarpGroupBuilder> &builders,
               const PartitionSet &partitions) {
  auto partitionIndices = getPartitionIds(ifOp, partitions.getNumPartitions());

  SmallVector<scf::IfOp> newIfOps;
  for (size_t idx : partitionIndices) {
    auto &b = builders[idx];
    auto cond = b.mapping.lookupOrDefault(ifOp.getCondition());
    auto newIfOp = b.create<scf::IfOp>(ifOp.getLoc(), ifOp.getResultTypes(),
                                       cond, ifOp.elseBlock() ? true : false);
    newIfOp->setAttrs(ifOp->getAttrs());
    newIfOps.push_back(newIfOp);

    mapRange(ifOp.getResults(), newIfOp.getResults(), b.mapping);
    mapRange(ifOp.thenBlock()->getArguments(),
             newIfOp.thenBlock()->getArguments(), b.mapping);

    if (ifOp.elseBlock()) {
      mapRange(ifOp.elseBlock()->getArguments(),
               newIfOp.elseBlock()->getArguments(), b.mapping);
    }

    b.setInsertionPointToStart(newIfOp.thenBlock());
  }

  cloneOpsInBlock(ifOp.thenBlock(), builders, partitions);

  if (auto elseBlock = ifOp.elseBlock()) {
    for (auto [builder, newIfOp] : llvm::zip(builders, newIfOps)) {
      builder.setInsertionPointToStart(newIfOp.elseBlock());
    }
    cloneOpsInBlock(elseBlock, builders, partitions);
  }

  for (auto [idx, newIfOp] : llvm::zip(partitionIndices, newIfOps)) {
    builders[idx].setInsertionPointAfter(newIfOp);
  }
}

void cloneReduceOp(triton::ReduceOp reduceOp,
                   SmallVector<WarpGroupBuilder> &builders,
                   const PartitionSet &partitions) {
  auto partitionIndices =
      getPartitionIds(reduceOp, partitions.getNumPartitions());

  SmallVector<ReduceOp> newReduceOps;
  for (size_t idx : partitionIndices) {
    auto &b = builders[idx];

    SmallVector<Value> srcs;
    for (auto src : reduceOp.getSrcs()) {
      srcs.push_back(b.mapping.lookupOrDefault(src));
    }
    auto axis = reduceOp.getAxis();
    auto newReduceOp =
        b.create<triton::ReduceOp>(reduceOp.getLoc(), srcs, axis);
    newReduceOp->setAttrs(reduceOp->getAttrs());
    newReduceOps.push_back(newReduceOp);

    mapRange(reduceOp.getResults(), newReduceOp.getResults(), b.mapping);

    auto &region = newReduceOp.getRegion();
    Block *block = &region.emplaceBlock();
    for (auto arg : reduceOp.getRegion().getBlocks().front().getArguments()) {
      auto newArg = block->addArgument(arg.getType(), arg.getLoc());
      b.mapping.map(arg, newArg);
    }

    b.setInsertionPointToStart(block);
  }

  cloneOpsInBlock(reduceOp.getBody(), builders, partitions);

  for (auto [idx, newReduceOp] : llvm::zip(partitionIndices, newReduceOps)) {
    builders[idx].setInsertionPointAfter(newReduceOp);
  }
}

void cloneOp(Operation *op, SmallVector<WarpGroupBuilder> &builders,
             ArrayRef<size_t> partitionIndices) {
  if (op->getNumRegions() != 0) {
    llvm::report_fatal_error(
        "Ops are expected to be regionless at this point.");
  }

  for (size_t idx : partitionIndices) {
    auto &builder = builders[idx];
    auto newOp = builder.clone(*op, builder.mapping);
    mapRange(op->getResults(), newOp->getResults(), builder.mapping);
  }
}

void cloneOpsInBlock(Block *block, SmallVector<WarpGroupBuilder> &builders,
                     const PartitionSet &partitions) {
  for (auto &op_ : *block) {
    auto op = &op_;
    auto partitionIndices = getPartitionIds(op, partitions.getNumPartitions());

    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      cloneForOp(forOp, builders, partitions);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      cloneIfOp(ifOp, builders, partitions);
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      cloneReduceOp(reduceOp, builders, partitions);
    } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      if (yieldOp.getOperands().empty()) {
        continue;
      }

      for (size_t idx : partitionIndices) {
        auto &builder = builders[idx];
        SmallVector<size_t> newOperandIndices;
        if (auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp())) {
          newOperandIndices =
              getLoopVarIndicesToKeep(
                  forOp, partitions.getPartition(builder.partitionId), partitions)
                  .first;
        } else {
          for (size_t i = 0; i < yieldOp.getOperands().size(); ++i) {
            newOperandIndices.push_back(i);
          }
        }

        SmallVector<Value> newYieldOperands;
        for (size_t i : newOperandIndices) {
          newYieldOperands.push_back(
              builder.mapping.lookupOrDefault(yieldOp.getOperand(i)));
        }

        if (!newYieldOperands.empty()) {
          builder.create<scf::YieldOp>(op->getLoc(), newYieldOperands);
        }
      }
    } else {
      cloneOp(op, builders, partitionIndices);
    }
  }
}

void assignRootPartition(scf::ForOp loop, int numPartitions) {
  auto ctx = loop.getContext();
  Builder b(ctx);
  SmallVector<int> ids;
  for (int i = 0; i < numPartitions; ++i) {
    ids.push_back(i);
  }
  auto partitionAttr = b.getDenseI32ArrayAttr(ids);

  for (Operation &op : loop.getBody()->without_terminator()) {
    if (!hasPartition(&op)) {
      op.setAttr(kPartitionAttrName, partitionAttr);
    }
  }
}

void assignRegionBodyPartition(scf::ForOp loop) {
  loop->walk([&](Operation *op) {
    if (!isa<scf::ForOp>(op) && !hasPartition(op)) {
      auto parentOp = loop.getBody()->findAncestorOpInBlock(*op);
      if (hasPartition(parentOp)) {
        op->setAttr(kPartitionAttrName, parentOp->getAttr(kPartitionAttrName));
      }
    }
  });
}

} // namespace

LogicalResult triton::gpu::partitionLoop(scf::ForOp loop) {
  FailureOr<PartitionSet> partitionsOr = PartitionSet::deserialize(loop);
  if (failed(partitionsOr))
    return failure();
  PartitionSet partitions = std::move(*partitionsOr);

  assignRootPartition(loop, partitions.getNumPartitions());
  assignRegionBodyPartition(loop);

  //  loop->getParentOfType<ModuleOp>().dump();

  // Only the root node should have consumers at this point.
  for (const Partition &partition : partitions.getPartitions()) {
    bool failed = false;
    auto callback = [&](OpResult output, OpOperand &use, unsigned distance) {
      auto defOpPartitionIds = getPartitionIds(output.getDefiningOp());
      if (!defOpPartitionIds ||
          defOpPartitionIds->size() == partitions.getNumPartitions()) {
        return;
      }
      auto partitionIds = getPartitionIds(use.getOwner());
      if (!partitionIds ||
          partitionIds->size() == partitions.getNumPartitions() ||
          llvm::is_contained(*partitionIds, partition.getIndex()))
        return;
      failed = true;
      InFlightDiagnostic diag =
          mlir::emitWarning(output.getLoc(), "non-root partition #")
          << partition.getIndex() << " has direct SSA consumer";

      for (auto partitionId : *partitionIds) {
        diag.attachNote(use.getOwner()->getLoc())
            << "use at distance " << distance << " in partition #"
            << partitionId << " here";
      }
    };
    iterateUses(loop, &partition, callback);
    if (failed)
      return failure();
  }

  // There is nothing to do if the loop has 1 or fewer partitions.
  if (llvm::size(partitions.getPartitions()) <= 1)
    return success();

  auto numPartitions = partitions.getNumPartitions();
  auto defaultPartition = partitions.getPartition((int)0);
  auto loopVarCategories = classifyLoopVars(loop, defaultPartition, partitions);
  auto [loopVarIndices, newResultIndices] =
      getLoopVarIndicesToKeep(loop, defaultPartition, loopVarCategories);

  ImplicitLocOpBuilder topBuilder(loop.getLoc(), loop);
  SmallVector<Value> tensorResultAllocs(loop.getNumRegionIterArgs());
  for (auto [i, res] : llvm::enumerate(loop.getResults())) {
    if (loopVarCategories[i] ==
        LoopVarCategory::TensorResultFromOtherPartition) {
      auto ty = cast<RankedTensorType>(res.getType());
      auto memdesc = MemDescType::get(
          ty.getShape(), ty.getElementType(), getSharedEncoding(ty),
          SharedMemorySpaceAttr::get(ty.getContext()), /*mutable=*/true);
      tensorResultAllocs[i] = topBuilder.create<LocalAllocOp>(memdesc);
    }
  }

  SmallVector<Type> resultTypes;
  for (auto i : loopVarIndices) {
    resultTypes.push_back(loop.getResultTypes()[i]);
  }

  SmallVector<int32_t> numWarps(numPartitions, lookupNumWarps(loop));
  auto wgOp = topBuilder.create<nvws::WarpGroupOp>(resultTypes, numWarps,
                                                   numPartitions);

  SmallVector<WarpGroupBuilder> builders;
  for (Region &region : wgOp.getPartitionRegions()) {
    auto partitionId = builders.size();
    auto &block = region.emplaceBlock();
    builders.push_back(WarpGroupBuilder(&block, block.end(), partitionId));
  }

  SmallVector<Operation *> opsToErase;
  for (auto &op_ : *loop->getBlock()) {
    auto op = &op_;
    auto wsTag = op->getAttrOfType<IntegerAttr>(kWarpSpecializeTagAttrName);
    if (!wsTag || wsTag.getInt() != partitions.getTag())
      continue;
    if (auto partitionIds = triton::gpu::getPartitionIds(op)) {
      SmallVector<size_t> tmp(partitionIds->begin(), partitionIds->end());
      cloneOp(op, builders, tmp);
      opsToErase.push_back(op);
    } else {
      assert(loop.getOperation() == op && "Unexpected op");
      cloneForOp(loop, builders, partitions);
      opsToErase.push_back(loop);
    }
  }

  for (auto [b, region, partition] : llvm::zip(
           builders, wgOp.getPartitionRegions(), partitions.getPartitions())) {
    auto newForOp = *region.front().getOps<scf::ForOp>().begin();
    auto outputs = newForOp.getResults();

    if (b.partitionId == 0) {
      b.create<nvws::WarpGroupYieldOp>(wgOp.getLoc(), outputs);
    } else {
      // Tensor results computed by non-default partitions are communicated back
      // via SMEM.
      // The calls to getLoopVarIndicesToKeep and isTensorResultComputedBy
      // below are unnecessary if we can encode the partition index and the
      // corresponding result tensor index of newForOp in
      // LoopVarCategory::TensorResultFromOtherPartition. In the absence of such
      // language support, we end up computing the same information multiple
      // times.
      auto [_, reverseIndices] =
          getLoopVarIndicesToKeep(loop, &partition, partitions);
      for (size_t i = 0; i < loop.getNumRegionIterArgs(); ++i) {
        if (loopVarCategories[i] ==
                LoopVarCategory::TensorResultFromOtherPartition &&
            isTensorResultComputedBy(loop, i, &partition, partitions)) {
          assert(reverseIndices[i] && "A valid index is expected.");
          auto result = newForOp.getResult(*reverseIndices[i]);
          b.create<LocalStoreOp>(wgOp.getLoc(), result, tensorResultAllocs[i]);
        }
      }
      b.create<nvws::WarpGroupReturnOp>(wgOp.getLoc());
    }
  }

  topBuilder.setInsertionPointAfter(wgOp);

  for (auto [i, res] : llvm::enumerate(loop.getResults())) {
    if (res.use_empty())
      continue;

    if (loopVarCategories[i] ==
        LoopVarCategory::TensorResultFromOtherPartition) {
      auto ty = cast<RankedTensorType>(loop.getResult(i).getType());
      auto output = topBuilder.create<LocalLoadOp>(ty, tensorResultAllocs[i]);
      topBuilder.create<LocalDeallocOp>(tensorResultAllocs[i]);
      res.replaceAllUsesWith(output);
    } else {
      assert(newResultIndices[i] && "A valid index is expected.");
      res.replaceAllUsesWith(wgOp.getResult(*newResultIndices[i]));
    }
  }

  //  loop->getParentOfType<ModuleOp>().dump();
  for (auto op : llvm::reverse(opsToErase))
    op->erase();

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
