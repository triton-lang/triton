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

SetVector<int> getIfOpResultPartitionIds(scf::IfOp ifOp, int pos) {
  auto arrayAttr = ifOp->getAttrOfType<ArrayAttr>(kPartitionOutputsAttrName);
  assert(arrayAttr.size() == ifOp.getResultTypes().size());
  auto partitionIdsRef = cast<DenseI32ArrayAttr>(arrayAttr[pos]).asArrayRef();
  return {partitionIdsRef.begin(), partitionIdsRef.end()};
}

SetVector<int> getIfOpResultPartitionIds(scf::IfOp ifOp, Value value) {
  for (auto result : ifOp.getResults()) {
    if (result == value) {
      auto pos = result.getResultNumber();
      return getIfOpResultPartitionIds(ifOp, pos);
    }
  }
  llvm_unreachable("value is not a result of if-stmt");
}

bool isTensorResultComputedBy(scf::ForOp loop, size_t resultIdx,
                              const Partition *partition,
                              const PartitionSet &partitions) {
  auto value = loop.getYieldedValues()[resultIdx];
  if (!isa<RankedTensorType>(value.getType()))
    return false;
  auto defOp = value.getDefiningOp();
  auto partitionIds = *getPartitionIds(defOp);
  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    partitionIds = getIfOpResultPartitionIds(ifOp, value);
  }
  return llvm::is_contained(partitionIds, partition->getIndex());
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
  auto inPartition = [&](OpOperand &opnd) {
    auto op = opnd.getOwner();
    auto partitionIds = getPartitionIds(op, partitions.getNumPartitions());
    if (auto ifOp = dyn_cast<scf::IfOp>(op->getParentOp());
        ifOp && isa<scf::YieldOp>(op)) {
      auto ids = getIfOpResultPartitionIds(ifOp, opnd.getOperandNumber());
      partitionIds = SmallVector<size_t>(ids.begin(), ids.end());
    }
    return llvm::is_contained(partitionIds, partition->getIndex());
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
    if (llvm::any_of(arg.getUses(), inPartition)) {
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

  for (auto [i, newForOp] : enumerate(newForOps)) {
    builders[i].setInsertionPointAfter(newForOp);
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
    SmallVector<Type> newIfResultTypes;
    SmallVector<int> newIfResultIndices;
    for (auto pos = 0; pos < ifOp.getResultTypes().size(); ++pos) {
      auto partitionIds = getIfOpResultPartitionIds(ifOp, pos);
      if (llvm::is_contained(partitionIds, b.partitionId)) {
        newIfResultTypes.push_back(ifOp.getResult(pos).getType());
        newIfResultIndices.push_back(pos);
      }
    }
    auto newIfOp = b.create<scf::IfOp>(ifOp.getLoc(), newIfResultTypes, cond,
                                       ifOp.elseBlock() ? true : false);
    newIfOp->setAttrs(ifOp->getAttrs());
    newIfOps.push_back(newIfOp);

    for (auto [newIdx, oldIdx] : llvm::enumerate(newIfResultIndices)) {
      b.mapping.map(ifOp.getResult(oldIdx), newIfOp.getResult(newIdx));
    }
    assert(ifOp.thenBlock()->getNumArguments() == 0);

    b.setInsertionPointToStart(newIfOp.thenBlock());
  }

  cloneOpsInBlock(ifOp.thenBlock(), builders, partitions);

  if (auto elseBlock = ifOp.elseBlock()) {
    for (auto [idx, newIfOp] : llvm::zip(partitionIndices, newIfOps)) {
      builders[idx].setInsertionPointToStart(newIfOp.elseBlock());
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
                  forOp, partitions.getPartition(builder.partitionId),
                  partitions)
                  .first;
        } else {
          auto ifOp = cast<scf::IfOp>(yieldOp->getParentOp());
          for (size_t i = 0; i < yieldOp.getOperands().size(); ++i) {
            auto ids = getIfOpResultPartitionIds(ifOp, i);
            if (llvm::is_contained(ids, builder.partitionId)) {
              newOperandIndices.push_back(i);
            }
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

} // namespace

LogicalResult triton::gpu::partitionLoop(scf::ForOp loop) {
  FailureOr<PartitionSet> partitionsOr = PartitionSet::fromLoop(loop);
  if (failed(partitionsOr))
    return failure();
  PartitionSet partitions = std::move(*partitionsOr);

  // Only the root node should have consumers at this point.
  for (const Partition &partition : partitions.getPartitions()) {
    bool failed = false;
    auto callback = [&](OpResult output, OpOperand &use, unsigned distance) {
      if (partitions.isInRootPartition(output.getDefiningOp())) {
        return;
      }
      auto partitionIds = getPartitionIds(use.getOwner());
      if (partitions.isInRootPartition(use.getOwner()) ||
          llvm::is_contained(*partitionIds, partition.getIndex()))
        return;

      // check if consumer partition set is a subset of the producer partitions
      auto defOpPartitionIds = getPartitionIds(output.getDefiningOp());
      bool isValidSubset = std::all_of(
          partitionIds->begin(), partitionIds->end(), [&](int consumerId) {
            return llvm::is_contained(*defOpPartitionIds, consumerId);
          });

      if (isValidSubset)
        return; // Valid: consumer ⊆ producer

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
    partition.iterateUses(loop, callback);
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
      cloneOp(op, builders,
              SmallVector<size_t>{partitionIds->begin(), partitionIds->end()});
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

  for (auto op : llvm::reverse(opsToErase))
    op->erase();

  return success();
}

LogicalResult inferIfOpPartitions(scf::IfOp ifOp) {
  using PartitionSet = SetVector<int>;
  PartitionSet ifOpPartitions;
  SmallVector<std::optional<PartitionSet>> partitionIndices(
      ifOp.getResultTypes().size());

  auto processBlock = [&](Block *block) -> LogicalResult {
    if (!block)
      return success();

    // for yield op, get partition of each result of if-stmt
    auto yieldOp = cast<scf::YieldOp>(block->getTerminator());
    for (auto &opnd : yieldOp->getOpOperands()) {
      auto partitionIds = getPartitionIds(opnd.get().getDefiningOp());
      if (!partitionIds)
        continue;
      auto idx = opnd.getOperandNumber();
      if (partitionIndices[idx] && partitionIndices[idx] != partitionIds) {
        return emitError(yieldOp.getLoc(),
                         "inconsistent partitions for then/else "
                         "branches yield operand");
      }
      partitionIndices[idx] = partitionIds;
    }

    // if-op partition set is the union of all op partitions in the block
    for (auto &op : block->without_terminator()) {
      auto opPartitions = getPartitionIds(&op);
      for (auto p : *opPartitions) {
        ifOpPartitions.insert(p);
      }
    }
    return success();
  };

  if (failed(processBlock(ifOp.thenBlock())))
    return failure();
  if (failed(processBlock(ifOp.elseBlock())))
    return failure();

  llvm::SmallVector<Attribute> partitionAttrs;
  for (auto [idx, partition] : llvm::enumerate(partitionIndices)) {
    if (!partition) {
      if (!isa<AsyncTokenType>(ifOp.getResult(idx).getType())) {
        return emitError(ifOp.getLoc(),
                         "partition not found for if-stmt result");
      }
      partition = SetVector<int>();
      partition->insert(0);
    }
    ArrayRef<int> ids(partition->begin(), partition->end());
    OpBuilder b(ifOp);
    partitionAttrs.push_back(b.getDenseI32ArrayAttr(ids));
  }
  ifOp->setAttr(kPartitionOutputsAttrName,
                ArrayAttr::get(ifOp.getContext(), partitionAttrs));
  setPartition(ifOp, ifOpPartitions);
  return success();
}

LogicalResult inferReduceOpPartitions(triton::ReduceOp reduceOp) {
  auto terminator = reduceOp.getRegion().getBlocks().front().getTerminator();
  auto partitionIds = getPartitionIds(terminator);
  if (!partitionIds)
    return emitError(reduceOp.getLoc(), "reduce op has no partition ids");
  setPartition(reduceOp, *partitionIds);
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
  // be annotated with partitions.
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttrOfType<ArrayAttr>(kPartitionStagesAttrName))
      loops.push_back(loop);
  });

  for (scf::ForOp loop : loops) {
    loop.walk([&](triton::ReduceOp reduceOp) {
      if (failed(inferReduceOpPartitions(reduceOp)))
        signalPassFailure();
    });
    loop.walk([&](scf::IfOp ifOp) {
      if (failed(inferIfOpPartitions(ifOp)))
        signalPassFailure();
    });
    if (failed(partitionLoop(loop)))
      return signalPassFailure();
  }
}
