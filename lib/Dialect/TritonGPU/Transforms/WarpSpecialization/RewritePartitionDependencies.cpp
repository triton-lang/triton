#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
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
    auto enterOp = b.createInto<triton::nvws::ArefPutEnterOp>(
        partition, srcStageCluster, aref, TypeRange{viewType}, tokenType);
    auto token = enterOp.getToken();

    auto exitOp = [this, &partition, srcStageCluster,
                   token](PartitionBuilder &b) {
      auto exitOp = b.createInto<triton::nvws::ArefPutExitOp>(
          partition, srcStageCluster, aref, token,
          b.getArrayAttr(SmallVector<Attribute>{triton::nvws::AsyncOpAttr::get(
              aref.getContext(), triton::nvws::AsyncOp::NONE)}));
    };
    return std::make_tuple(enterOp.getResult(0), exitOp);
  }

  auto getView(PartitionBuilder &b, Partition &partition,
               StageCluster srcStageCluster) {
    auto enterOp = b.createInto<triton::nvws::ArefGetEnterOp>(
        partition, srcStageCluster, aref, TypeRange{viewType}, tokenType);
    auto token = enterOp.getToken();

    auto exitOp = [this, &partition, srcStageCluster,
                   token](PartitionBuilder &b) {
      auto exitOp = b.createInto<triton::nvws::ArefGetExitOp>(
          partition, srcStageCluster, aref, token,
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
  DependencyRewriter(PartitionSet &partitions, scf::ForOp &loop)
      : partitions(partitions), loop(loop), b(loop.getLoc(), loop),
        endBuilder(loop.getLoc(), loop->getNextNode()) {}

  // Partition the loop.
  LogicalResult run();

private:
  AsyncRef allocateAsyncValue(RankedTensorType tensorType,
                              unsigned maxDistance);

  // The partition set to apply.
  PartitionSet &partitions;
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
  auto aref = triton::nvws::ArefCreateOp::create(b, b.getLoc(), arefTy, alloc);

  return AsyncRef{aref, getBufferViewType(allocType),
                  b.getType<AsyncTokenType>()};
}

LogicalResult DependencyRewriter::run() {
  SmallVector<llvm::MapVector<Value, UseInfo>> partitionUseInfo;

  for (const Partition &partition : partitions.getPartitions()) {
    // Find all consumers of all outputs of this partition, tracking the
    // specific Partition
    auto &useInfo = partitionUseInfo.emplace_back();
    SmallVector<std::tuple<Value, OpOperand *, unsigned>> uses;
    llvm::DenseMap<Value, llvm::DenseSet<int>> consumerPartitions;

    std::function<void(OpOperand &, llvm::DenseSet<int>)> collectUses;
    collectUses = [&](OpOperand &use, llvm::DenseSet<int> partitionsToSearch) {
      Operation *owner = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      auto partitionIds = getPartitionIds(owner);
      if (isa<scf::YieldOp>(owner)) {
        // Filter out partitions that have already consumed this value.
        // This is a performance optimization to avoid duplicate loads for
        // loop-carried values.
        if (consumerPartitions.contains(use.get())) {
          for (auto id : consumerPartitions[use.get()]) {
            partitionsToSearch.erase(id);
          }
          if (partitionsToSearch.empty()) {
            return;
          }
        }

        // This value is used in a subsequent iteration.
        // collect the uses of the appropriate loop arg
        for (auto &newUse : loop.getBody()
                                ->getArgument(use.getOperandNumber() + 1)
                                .getUses()) {
          collectUses(newUse, partitionsToSearch);
        }
      } else if (partitionIds) {
        // Record the partitions that consume this value.
        consumerPartitions[use.get()].insert(partitionIds->begin(),
                                             partitionIds->end());
        for (auto id : *partitionIds) {
          if (id != partition.getIndex() && partitionsToSearch.contains(id)) {
            // This value is used in a different partition in the same
            // iteration.
            uses.emplace_back(use.get(), &use, 0);
          }
        }
      }
    };
    for (Operation *op : partition.getOps()) {
      if (partitions.isInRootPartition(op)) {
        // skip ops in the root partition
        continue;
      }

      // Only collect cross partition uses.
      llvm::DenseSet<int> partitionsToSearch;
      for (const Partition &p : partitions.getPartitions()) {
        if (p.getIndex() != partition.getIndex()) {
          partitionsToSearch.insert(p.getIndex());
        }
      }

      // Collect non-yield uses first in order to detect redundancy.
      SmallVector<OpOperand *> yieldUses;
      for (OpOperand &use : op->getUses()) {
        if (isa<scf::YieldOp>(use.getOwner()))
          yieldUses.push_back(&use);
        else
          collectUses(use, partitionsToSearch);
      }
      for (OpOperand *use : yieldUses)
        collectUses(*use, partitionsToSearch);
    }

    auto callback = [&](Value output, OpOperand &use, unsigned distance) {
      Operation *user = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      Partition *usePartition = partitions.getPartition(user);
      // Ignore uses in the same partition in the future.
      if (usePartition == &partition) {
        assert(distance > 0 && "self-recursion must occur in the future");
        return;
      }
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
       llvm::zip(partitions.getPartitions(), partitionUseInfo)) {
    // The amount of buffering is based on the longest distance to a user.
    for (auto &[output, info] : useInfo) {
      // Skip AsyncTokenType outputs - they are handled correctly by design
      // in a previous pass and should not be passed through shared memory
      if (isa<AsyncTokenType>(output.getType())) {
        continue;
      }

      b.setLoc(output.getLoc());
      ImplicitLocOpBuilder endBuilder(b.getLoc(), loop->getNextNode());

      bool isScalar = false;
      Value tmp = output;
      Operation *defOp;
      while (true) {
        if (auto arg = dyn_cast<BlockArgument>(tmp)) {
          tmp = loop.getBody()->getTerminator()->getOperand(arg.getArgNumber() -
                                                            1);
          continue;
        }
        defOp = tmp.getDefiningOp();
        break;
      }
      Value val = output;
      auto tensorType = dyn_cast<RankedTensorType>(output.getType());
      if (!tensorType) {
        isScalar = true;
        b.setInsertionPointAfterValue(output);
        auto mod = output.getParentRegion()->getParentOfType<ModuleOp>();
        auto nWarps = lookupNumWarps(mod);
        auto threadsPerWarp =
            triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
        int CTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
        Attribute encoding = getDefaultBlockedEncoding(
            b.getContext(), {1}, nWarps, threadsPerWarp, CTAs);
        tensorType = RankedTensorType::get({1}, output.getType(), encoding);
        StageCluster srcStageCluster = getStageCluster(defOp);

        defOp = b.createInto<triton::SplatOp>(partition, srcStageCluster,
                                              tensorType, output);
        val = defOp->getResult(0);
      }

      // Buffer the value based on the greatest distance to a consumer
      // partition.
      int maxDistance = info.getMaxUseDistance(partition);

      // Allocate buffers for the value and its associated barriers.
      AsyncRef aref = allocateAsyncValue(tensorType, maxDistance);

      unsigned numConsumers = info.consumers.size();

      for (auto &[key, uses] : info.consumers) {
        assert(!uses.empty() && "expected at least one use");
        Operation *earliestUser = getEarliestUser(uses);
        b.setInsertionPoint(
            loop.getBody()->findAncestorOpInBlock(*earliestUser));

        Partition *usePartition = key.first;

        // Wait for the value to be available.
        StageCluster sinkSrcCluster = getStageCluster(earliestUser);
        auto [view, exitOp] = aref.getView(b, *usePartition, sinkSrcCluster);
        // Load the value at the current index and replace uses in this
        // partition with it.
        Value value = b.createInto<LocalLoadOp>(*usePartition, sinkSrcCluster,
                                                tensorType, view);
        if (isScalar) {
          value = b.createInto<triton::UnsplatOp>(*usePartition, sinkSrcCluster,
                                                  value);
        }
        for (OpOperand *use : uses)
          use->set(value);

        // Check for any loop carried usages to replace.
        for (OpOperand &use : output.getUses()) {
          Operation *user =
              loop.getBody()->findAncestorOpInBlock(*use.getOwner());
          if (isa<scf::YieldOp>(user)) {
            int yieldArgIndex = use.getOperandNumber();
            SmallVector<OpOperand *> crossPartitionUsages;
            for (OpOperand &carriedUse :
                 loop.getBody()->getArgument(yieldArgIndex + 1).getUses()) {
              Operation *carriedUser =
                  loop.getBody()->findAncestorOpInBlock(*carriedUse.getOwner());
              auto partitionIds = getPartitionIds(carriedUser);
              // Ensure consumed by current partition.
              if (partitionIds &&
                  llvm::is_contained(*partitionIds, usePartition->getIndex())) {
                crossPartitionUsages.push_back(&carriedUse);
              }
            }

            // Add new loop arguments for cross-partition loop carried values.
            if (!crossPartitionUsages.empty()) {
              // Use the same init value for the new loop argument.
              OpBuilder rewriter(loop.getContext());
              loop = addIterArgsToLoop(
                  rewriter, loop,
                  ValueRange{loop.getInitArgs()[yieldArgIndex]});

              // Use the same output value in the new position of the yield op.
              appendToForOpYield(loop, {value});

              // Replace all uses of the old loop argument in this partition
              // with the new loop argument.
              for (OpOperand *use : crossPartitionUsages) {
                use->set(loop.getBody()->getArguments().back());
              }
            }
          }
        }

        exitOp(b);
      }

      // Set up production of the value
      if (isa<BlockArgument>(val))
        b.setInsertionPointToStart(loop.getBody());
      else
        b.setInsertionPointAfter(defOp);

      StageCluster srcStageCluster = getStageCluster(defOp);
      auto [view, exitOp] = aref.putView(b, partition, srcStageCluster);
      b.createInto<LocalStoreOp>(partition, srcStageCluster, val, view);
      exitOp(b);
    }
  }

  // Rewrite the loop to add the new results. Calling this function with no
  // indices set will just resize the results.
  eraseLoopCarriedValues(loop, {});
  return success();
}

//===----------------------------------------------------------------------===//
// rewritePartitionDependenies
//===----------------------------------------------------------------------===//

LogicalResult triton::gpu::rewritePartitionDependencies(scf::ForOp &loop) {
  FailureOr<PartitionSet> partitionsOr = PartitionSet::fromLoop(loop);
  if (failed(partitionsOr))
    return failure();
  PartitionSet partitions = std::move(*partitionsOr);
  DependencyRewriter rewriter(partitions, loop);
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
  // already be annotated with partitions.
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
