#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/WalkResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <optional>
#include <utility>

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// assignPartitions
//===----------------------------------------------------------------------===//

bool trySetPartition(Operation *op, Partition *partition) {
  if (hasPartition(op)) {
    return false;
  }
  setPartition(op, partition);
  return true;
}

// Find the last operation in the loop body that defined this value, with a
// maximum of distance 1.
static Operation *findDefOpInLoop(scf::ForOp loop, Value value,
                                  int distance = 0) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getParentBlock() != loop.getBody())
      return {};
    // Don't look back more than distance 1.
    if (distance == 1)
      return {};
    return findDefOpInLoop(
        loop, loop.getYieldedValues()[arg.getArgNumber() - 1], distance + 1);
  }
  Operation *defOp = value.getDefiningOp();
  if (!loop.getBodyRegion().isAncestor(defOp->getParentRegion()))
    return {};
  return defOp;
}

// For `op`, invoke `callback` on all the definitions of its inputs from within
// `loop`, which might not be in the same iteration.
static void iterateDefs(scf::ForOp loop, Operation *op,
                        function_ref<void(OpResult)> callback) {
  visitNestedOperands(op, [&](OpOperand &operand) {
    Value value = operand.get();
    if (value.getParentBlock() != loop.getBody())
      return;
    auto arg = dyn_cast<BlockArgument>(value);
    if (arg == loop.getInductionVar())
      return;
    auto [def, distance] = getDefinitionAndDistance(loop, operand.get());
    if (def && def.getParentBlock() == loop.getBody())
      callback(def);
  });
}

// For `op`, invoke `callback` on all its transitive users within `loop`, which
// may be in a future iteration.
static void iterateUsers(scf::ForOp loop, Operation *op,
                         function_ref<void(Operation *)> callback) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : op->getUses())
    uses.push_back(&use);
  while (!uses.empty()) {
    OpOperand *use = uses.pop_back_val();
    Operation *owner = loop.getBody()->findAncestorOpInBlock(*use->getOwner());
    if (!isa<scf::YieldOp>(owner)) {
      callback(owner);
      continue;
    }
    BlockArgument arg = loop.getRegionIterArg(use->getOperandNumber());
    for (OpOperand &use : arg.getUses())
      uses.emplace_back(&use);
  }
}

// Check if any of the inputs to `op` are reachable from a non-null partition.
static bool hasDefPartition(scf::ForOp loop, Operation *op,
                            PartitionSet &partitions) {
  SmallVector<Operation *> worklist{op};
  DenseSet<Operation *> seen;
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!seen.insert(op).second)
      continue;
    std::optional<SetVector<int>> partitionIds;
    if (hasPartition(op))
      partitionIds = getPartitionIds(op);
    if (partitionIds && partitionIds->size() != partitions.getNumPartitions())
      return true;
    iterateDefs(loop, op,
                [&](OpResult def) { worklist.push_back(def.getDefiningOp()); });
  }
  return false;
}

// Recursively schedule the dependencies of an operation, stopping when
// encountering an operation that is already assigned.
static void scheduleDependencies(scf::ForOp loop, PartitionSet &partitions,
                                 Partition *partition, Operation *op) {
  SmallVector<Value> deps;
  for (Value value : getNestedOperands(op)) {
    if (isa<RankedTensorType, MemDescType>(value.getType()))
      deps.push_back(value);
  }

  while (!deps.empty()) {
    Value dep = deps.pop_back_val();

    if (auto arg = dyn_cast<BlockArgument>(dep)) {
      if (arg.getOwner() == loop.getBody() && arg != loop.getInductionVar())
        deps.push_back(loop.getYieldedValues()[arg.getArgNumber() - 1]);
      continue;
    }

    Operation *defOp =
        loop.getBody()->findAncestorOpInBlock(*dep.getDefiningOp());
    if (!defOp || !hasDefPartition(loop, defOp, partitions) ||
        !trySetPartition(defOp, partition))
      continue;
    llvm::append_range(deps, getNestedOperands(defOp));
  }
}

// Recursively schedule the users of an operation, stopping when
// encountering an operation that is already assigned.
static void scheduleUsers(scf::ForOp loop, PartitionSet &partitions,
                          Partition *partition, Operation *op) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : op->getUses())
    uses.push_back(&use);
  while (!uses.empty()) {
    OpOperand *use = uses.pop_back_val();
    Operation *user = loop.getBody()->findAncestorOpInBlock(*use->getOwner());

    if (user == loop.getBody()->getTerminator()) {
      for (OpOperand &use :
           loop.getRegionIterArg(use->getOperandNumber()).getUses())
        uses.push_back(&use);
      continue;
    }

    if (!trySetPartition(user, partition))
      continue;
    for (OpOperand &use : user->getUses())
      uses.push_back(&use);
  }
}

SetVector<Partition *> getInitialPartitions(scf::ForOp loop,
                                            PartitionSet &partitions,
                                            Partition *defaultPartition,
                                            Partition *mmaPartition,
                                            Partition *loadPartition) {
  SmallVector<Operation *> loadsAndAllocs;
  SmallVector<ttng::MMAv5OpInterface> mmas;
  SetVector<Partition *> userPartitions;
  userPartitions.insert(defaultPartition);

  for (Operation &op : loop.getOps()) {
    if (auto innerFor = dyn_cast<scf::ForOp>(op)) {
      for (auto userPartition :
           getInitialPartitions(innerFor, partitions, defaultPartition,
                                mmaPartition, loadPartition)) {
        userPartitions.insert(userPartition);
      }
    } else if (isa<DescriptorLoadOp, DescriptorGatherOp>(op)) {
      setPartition(&op, loadPartition);
      loadsAndAllocs.push_back(&op);
      // Local alloc users of the load with matching encoding will cause the
      // underlying buffer to be pass through. Keep track of them.
      SharedEncodingTrait sharedEnc = getSharedEncoding(&op);
      for (Operation *user : op.getUsers()) {
        if (auto alloc = dyn_cast<LocalAllocOp>(user)) {
          if (sharedEnc == alloc.getType().getEncoding()) {
            setPartition(alloc, loadPartition);
            loadsAndAllocs.push_back(alloc);
          }
        } else if (isa<ttng::TMEMAllocOp>(user)) {
          setPartition(user, loadPartition);
          loadsAndAllocs.push_back(user);
        }
      }
    } else if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(op)) {
      setPartition(mmaOp, mmaPartition);
      mmas.push_back(mmaOp);

      // If the store is unrelated to the use of the MMA, then it gets placed in
      // the MMA partition.
      auto storeOp = dyn_cast_or_null<ttng::TMEMStoreOp>(
          findDefOpInLoop(loop, mmaOp.getAccDep()));
      if (!ttng::hasAccReadModifyWrite(mmaOp, loop) && storeOp &&
          loop.isDefinedOutsideOfLoop(storeOp.getSrc()))
        setPartition(storeOp, mmaPartition);

      // Look for views into the operands.
      SmallVector<Operation *> operandViews;
      for (Value operand : mmaOp->getOperands()) {
        if (Operation *defOp = operand.getDefiningOp())
          operandViews.push_back(defOp);
      }
      while (!operandViews.empty()) {
        Operation *op = operandViews.pop_back_val();
        if (!op->hasTrait<OpTrait::MemDescViewTrait>())
          continue;

        // Duplicate the op if necessary to ensure that the MMA partition is the
        // only user.
        if (!llvm::all_of(op->getUsers(), [&](Operation *user) {
              return mmaPartition->hasOp(user);
            })) {
          Operation *newOp = OpBuilder(op).clone(*op);
          op->replaceUsesWithIf(newOp->getResults(), [&](OpOperand &use) {
            return mmaPartition->hasOp(use.getOwner());
          });
          op = newOp;
        }

        setPartition(op, mmaPartition);
        if (Operation *defOp = op->getOperand(0).getDefiningOp())
          operandViews.push_back(defOp);
      }
    }
  }

  if (loadPartition->empty() && mmaPartition->empty()) {
    return {};
  }

  // Propagate defs of exp.
  for (Operation &op : loop.getOps()) {
    if (!isa<math::Exp2Op, ElementwiseInlineAsmOp>(op))
      continue;
    int elementCount = 0;
    for (Type type : op.getResultTypes()) {
      if (auto tensorTy = dyn_cast<RankedTensorType>(type))
        elementCount += tensorTy.getNumElements();
    }
    if (elementCount > 256) {
      setPartition(&op, defaultPartition);
      scheduleDependencies(loop, partitions, defaultPartition, &op);
    }
  }

  // Propagate users of loads and MMAs.
  for (Operation *loadOrAlloc : loadsAndAllocs)
    scheduleUsers(loop, partitions, defaultPartition, loadOrAlloc);

  while (userPartitions.size() < mmas.size()) {
    userPartitions.insert(partitions.addPartition(userPartitions.size()));
  }
  for (auto [mmaOp, userPartition] :
       llvm::reverse(llvm::zip(mmas, userPartitions))) {
    scheduleUsers(loop, partitions, userPartition, mmaOp);
  }

  // Annotate remaining unannotated tmem loads, for example those outside of the
  // inner loop
  for (ttng::TMEMLoadOp tmemLoad : loop.getOps<ttng::TMEMLoadOp>()) {
    if (hasPartition(tmemLoad)) {
      continue;
    }

    if (userPartitions.size() == 1) {
      setPartition(tmemLoad, defaultPartition);
    } else {
      auto tmem = tmemLoad.getSrc();
      SetVector<Partition *> tmemUserPartitions;
      for (auto user : tmem.getUsers()) {
        if (!hasPartition(user)) {
          continue;
        }
        if (auto partition = partitions.getPartition(user);
            partition != mmaPartition) {
          tmemUserPartitions.insert(partition);
        }
      }
      // TMEM should only used by MMA and one user partition
      assert(tmemUserPartitions.size() == 1);
      setPartition(tmemLoad, tmemUserPartitions.front());
    }
  }

  // Annotate the inner loop with its body partitions
  if (!loop->hasAttr(kWarpSpecializeAttrName)) {
    SetVector<Partition *> bodyPartitons;
    for (Operation &op : loop.getOps()) {
      if (hasPartition(&op)) {
        for (auto id : getPartitionIds(&op)) {
          bodyPartitons.insert(partitions.getPartition(id));
        }
      }
    }

    setPartition(loop, bodyPartitons);
  }

  return userPartitions;
}

// Given a partitioning scheme, determine an initial schedule by performing a
// first-order partition assignment to the operations in the scheme and its
// users and/or dependencies. This sets up the initial partitioning of the ops.
static std::optional<PartitionSet> getInitialPartitions(scf::ForOp loop) {
  // Check for an existing partition set.
  if (FailureOr<PartitionSet> partitionsOr = PartitionSet::fromLoop(loop);
      succeeded(partitionsOr))
    return {std::move(*partitionsOr)};
  // Start by creating the default partition, a partition for for all loads, and
  // a partition for all MMAs.
  PartitionSet partitions;
  Partition *defaultPartition = partitions.addPartition(0);
  Partition *mmaPartition = partitions.addPartition(1);
  Partition *loadPartition = partitions.addPartition(0);

  getInitialPartitions(loop, partitions, defaultPartition, mmaPartition,
                       loadPartition);

  // If there are no loads or MMAs, don't warp specialize.
  if (loadPartition->empty() && mmaPartition->empty()) {
    return std::nullopt;
  }

  return partitions;
}

namespace {
// This data structure represents a cluster of operations that have not been
// assigned to a stage. Operations form a cluster when:
//
// - they are adjacent in the SSA use def graph
// - they are not already assigned to a partition
// - at least one of their inputs is reachable from a definition partition
//
struct OpCluster {
  // These are the operations in the cluster.
  SetVector<Operation *> ops;
  // The definition partitions are the partitions from which inputs of the
  // operation are reachable. When the cluster is fully formed, the defining op
  // in the loop of any input to any operation in the cluster is either in the
  // root partition or one of these partitions.
  SetVector<Partition *> defPartitions;
  // The sink partitions which consume the outputs of operations in this
  // cluster. When the cluster is fully formed, all uses in the loop of outputs
  // of any operation in the cluster belong to one of these partitions.
  SetVector<Partition *> sinkPartitions;
};

// Owning class for a bunch of clusters. This class manages the lifetimes of the
// clusters and has some helper functions.
struct OpClusters : public llvm::MapVector<Operation *, OpCluster *> {
  using MapVector::MapVector;

  // Create a new cluster that contains only the given operation, a return a
  // cluster that already contains the operation.
  OpCluster *getOrCreate(Operation *op) {
    OpCluster *&cluster = (*this)[op];
    if (!cluster) {
      cluster = clusters.emplace_back(new OpCluster).get();
      cluster->ops.insert(op);
    }
    return cluster;
  }
  // Merge two clusters by merging their sets and clearing the other cluster,
  // marking it as dead.
  void merge(OpCluster *dst, OpCluster *src) {
    dst->ops.insert_range(src->ops);
    dst->defPartitions.insert_range(src->defPartitions);
    dst->sinkPartitions.insert_range(src->sinkPartitions);
    for (Operation *op : src->ops)
      (*this)[op] = dst;
    src->ops.clear();
    src->defPartitions.clear();
    src->sinkPartitions.clear();
  }

  SmallVector<std::unique_ptr<OpCluster>> clusters;
};
} // namespace

// Operations that require partition assignment are those reachable from an
// operation in a partition. This function propagates partitions by first
// forming contiguous clusters from the unassigned operations and then deciding
// what to do with the operations in that cluster.
void propagatePartitions(scf::ForOp loop, PartitionSet &partitions) {
  for (Operation &op : loop.getOps()) {
    if (auto innerFor = dyn_cast<scf::ForOp>(op)) {
      propagatePartitions(innerFor, partitions);
    }
  }

  OpClusters opClusters;

  for (Partition &partition : partitions.getPartitions()) {
    // For each partition, check if any of their inputs are reachable from
    // another partition and spawn a single cluster at that operation.
    auto defCallback = [&](OpResult result, unsigned distance) {
      Operation *defOp = result.getDefiningOp();
      if (!hasPartition(defOp) && hasDefPartition(loop, defOp, partitions)) {
        // Add the current partition as a sink to the cluster.
        opClusters.getOrCreate(defOp)->sinkPartitions.insert(&partition);
      }
    };
    partition.iterateDefs(loop, defCallback);

    // For each partition, place users of its outputs in a cluster if it is not
    // already assigned to a partition.
    auto useCallback = [&](OpResult result, OpOperand &use, unsigned distance) {
      Operation *user = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      if (!hasPartition(user)) {
        // Add the current partition as a def to the cluster.
        opClusters.getOrCreate(user)->defPartitions.insert(&partition);
      }
    };
    partition.iterateUses(loop, useCallback);
  }

  // Now we have a pile of single-operation clusters directly adjacent to the
  // operations in a partition. Grow the clusters by adding adjacent operations
  // clusters and merging clusters when possible.
  SmallVector<Operation *> worklist =
      llvm::to_vector(llvm::make_first_range(opClusters));
  while (!worklist.empty()) {
    // Grab an op off the worklist. We know it has a cluster already.
    Operation *op = worklist.pop_back_val();
    OpCluster *cluster = opClusters.find(op)->second;
    // Look at the definitions directly feeding into this operation.
    iterateDefs(loop, op, [&](OpResult def) {
      Operation *defOp = def.getDefiningOp();
      if (hasPartition(defOp)) {
        auto partitionIds = getPartitionIds(defOp);
        // The input originates from an operation already assigned to a
        // partition. Add this as a def partition.
        for (auto id : partitionIds) {
          cluster->defPartitions.insert(partitions.getPartition(id));
        }
      } else {
        // If the input is not reachable from a partition, ignore it.
        if (!hasDefPartition(loop, defOp, partitions))
          return;
        // This operation is not assigned to a partition.
        OpCluster *&defCluster = opClusters[defOp];
        if (!defCluster) {
          // This operation has not yet been added to a cluster. Add it to the
          // current cluster and recurse on it.
          defCluster = cluster;
          cluster->ops.insert(defOp);
          worklist.push_back(defOp);
        } else if (defCluster != cluster) {
          // This operation is part of another cluster. Merge the two clusters
          // together and continue.
          opClusters.merge(cluster, defCluster);
        }
      }
    });
    // Check the users of the operation.
    iterateUsers(loop, op, [&](Operation *user) {
      if (hasPartition(user)) {
        auto partitionIds = getPartitionIds(user);
        // If the user is already assigned to a partition, add that partition as
        // one of the sink partitions.
        for (auto id : partitionIds) {
          cluster->sinkPartitions.insert(partitions.getPartition(id));
        }
        return;
      }
      // If the user does not already have a cluster, add it to the current
      // cluster. We don't have to handle merging here because when the user
      // visits the current op, it will trigger the merge.
      OpCluster *&userCluster = opClusters[user];
      if (userCluster)
        return;
      userCluster = cluster;
      cluster->ops.insert(user);
      worklist.push_back(user);
    });
  }

  // We have clustered unassigned ops in the liveouts of ops in assigned
  // partitions and in the critical paths between ops in different partitions.
  // Ops that are next to each other are placed in the same cluster. Now the
  // task is to figure out how to assign partitions to the ops in each cluster
  // based on the def and sink partitions, which is very non-trivial.
  for (OpCluster &cluster : llvm::make_pointee_range(opClusters.clusters)) {
    // Skip dead clusters.
    if (cluster.ops.empty())
      continue;
    assert(!cluster.defPartitions.empty());
    assert(llvm::all_of(cluster.ops,
                        [&](Operation *op) { return !hasPartition(op); }));

    // If there is no sink partition, this means there is a backedge somewhere,
    // for now assign the cluster to the def partition.
    Partition *defPartition = cluster.defPartitions.front();
    if (cluster.sinkPartitions.empty()) {
      for (Operation *op : cluster.ops)
        setPartition(op, defPartition);
      continue;
    }

    // Find the critical path between the def partition and sink partition.
    Partition *sinkPartition = cluster.sinkPartitions.front();
    SetVector<Operation *> critPath;
    DenseSet<Operation *> opsInCluster(cluster.ops.begin(), cluster.ops.end());
    auto callback = [&](OpResult result, unsigned distance) {
      Operation *defOp = result.getDefiningOp();
      if (opsInCluster.contains(defOp))
        critPath.insert(defOp);
    };
    sinkPartition->iterateDefs(loop, callback);
    for (unsigned i = 0; i < critPath.size(); ++i) {
      Operation *op = critPath[i];
      iterateDefs(loop, op, [&](OpResult def) {
        Operation *defOp = def.getDefiningOp();
        if (opsInCluster.contains(defOp))
          critPath.insert(defOp);
      });
    }

    // If all ops are on the critical path, assign them to the def partition.
    if (critPath.size() == cluster.ops.size()) {
      for (Operation *op : cluster.ops)
        setPartition(op, defPartition);
      continue;
    }

    // Some ops are on the critical path, and there is also a backedge.
    // Rematerialize the critical path ops into the sink partition. Leave the
    // rest in the def partition and rely on DCE to remove them.
    critPath = topologicalSort(critPath);
    DenseSet<Operation *> sinkOps(sinkPartition->getOps().begin(),
                                  sinkPartition->getOps().end());
    for (Operation *op : llvm::reverse(critPath)) {
      OpBuilder b(op);
      Operation *clone = b.clone(*op);
      op->replaceUsesWithIf(clone->getResults(), [&](OpOperand &use) {
        return sinkOps.contains(use.getOwner());
      });
      sinkOps.insert(clone);
      setPartition(clone, sinkPartition);
    }
    for (Operation *op : cluster.ops)
      setPartition(op, defPartition);
  }
}

// Rematerialize chains of broadcasts where the user is in a different partition
// than the broadcast to reduce the amount of data that needs to be transferred.
void rematerializeBroadcasts(PartitionSet &partitions, OpOperand *use) {
  static_assert(
      std::is_base_of_v<OpTrait::OneResult<BroadcastOp>, BroadcastOp> &&
      std::is_base_of_v<OpTrait::OneResult<ExpandDimsOp>, ExpandDimsOp>);

  Operation *defOp = use->get().getDefiningOp();
  while (isa_and_nonnull<BroadcastOp, ExpandDimsOp>(defOp)) {
    Operation *clone = OpBuilder(defOp).clone(*defOp);
    assert(hasPartition(use->getOwner()) && "user not scheduled");
    auto userPartitionIds = getPartitionIds(use->getOwner());
    for (auto id : userPartitionIds) {
      Partition *userPartition = partitions.getPartition(id);
      setPartition(clone, userPartition);
    }
    use->set(clone->getResult(0));

    defOp = clone->getOperand(0).getDefiningOp();
    use = &clone->getOpOperand(0);
  }
}

void optimizePartitions(scf::ForOp loop, PartitionSet &partitions) {
  for (Operation &op : loop.getOps()) {
    if (auto innerFor = dyn_cast<scf::ForOp>(op)) {
      optimizePartitions(innerFor, partitions);
    }
  }

  for (Partition &partition : partitions.getPartitions()) {
    SmallVector<OpOperand *> uses;
    partition.iterateOutputs(loop, [&](Operation *defOp, OpOperand &use) {
      if (!isa<scf::YieldOp>(use.getOwner()))
        uses.push_back(&use);
    });
    for (OpOperand *use : uses)
      rematerializeBroadcasts(partitions, use);
  }
}

void getUseOps(Value value, SetVector<Operation *> &useOps,
               DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return;
  for (auto &use : value.getUses()) {
    auto useOp = use.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(useOp)) {
      if (use.getOperandNumber() < forOp.getNumControlOperands()) {
        useOps.insert(forOp);
      } else {
        auto pos = use.getOperandNumber() - forOp.getNumControlOperands();
        auto arg = forOp.getRegionIterArg(pos);
        getUseOps(arg, useOps, visited);
      }
    } else if (isa<scf::YieldOp>(useOp)) {
      auto parentOp = useOp->getParentOp();
      Value arg;
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        arg = forOp.getRegionIterArg(use.getOperandNumber());
      } else {
        auto ifOp = cast<scf::IfOp>(parentOp);
        arg = ifOp.getResults()[use.getOperandNumber()];
      }
      getUseOps(arg, useOps, visited);
    } else {
      useOps.insert(useOp);
    }
  }
}
// TODO: Implement a mutually-recursive traversal that can handle
//       nested control flow structures (if/reduce/for operations).
//       While we don't currently have use cases requiring this,
//       implementing it would prepare for when it is needed.
LogicalResult assignMissingPartitions(scf::ForOp loop,
                                      PartitionSet &partitions) {
  // For operations that have no partitions assigned, assign a partition set
  // that is the union of all partition sets of its direct users.
  auto isScalarOp = [](Operation *op) {
    return llvm::all_of(op->getResultTypes(), [](Type type) {
      return isa<FloatType, IntegerType>(type);
    });
  };

  loop.walk([&](ttng::TMEMAllocOp allocOp) {
    std::optional<int> mmaPartitionId, loadPartitionId, storePartitionId;
    bool hasSIMT = false;
    for (auto users : allocOp.getResult().getUsers()) {
      if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(users)) {
        if (hasPartition(mma)) {
          mmaPartitionId = getPartitionIds(mma).front();
        }
      } else if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(users)) {
        hasSIMT = true;
        if (hasPartition(storeOp)) {
          storePartitionId = getPartitionIds(storeOp).front();
        }
      } else {
        auto loadOp = cast<ttng::TMEMLoadOp>(users);
        hasSIMT = true;
        if (hasPartition(loadOp)) {
          loadPartitionId = getPartitionIds(loadOp).front();
        }
      }
    }

    assert(mmaPartitionId && "mma must have a partition");
    if (!hasSIMT)
      return WalkResult::advance();

    assert((loadPartitionId || storePartitionId) &&
           "at least one of load or store must have a partition");
    if (loadPartitionId && storePartitionId) {
      assert(loadPartitionId == storePartitionId &&
             "load and store partitions must be in the same partition");
    }
    int simtPartitionId;
    if (loadPartitionId) {
      simtPartitionId = *loadPartitionId;
    } else {
      simtPartitionId = *storePartitionId;
    }

    for (auto user : allocOp->getUsers()) {
      if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp>(user)) {
        if (!hasPartition(user)) {
          SetVector<int> simtPartitionIds;
          simtPartitionIds.insert(simtPartitionId);
          setPartition(user, simtPartitionIds);
        }
      }
    }
    return WalkResult::advance();
  });

  llvm::MapVector<Operation *, SetVector<Operation *>> opsMap;
  DenseMap<Operation *, DenseSet<int>> partitionMap;

  loop.walk([&](Operation *op) {
    if (op->getNumRegions() > 0)
      return WalkResult::advance();

    DenseSet<int> ids;
    if (hasPartition(op)) {
      auto partitionIds = getPartitionIds(op);
      ids.insert(partitionIds.begin(), partitionIds.end());
    }
    partitionMap[op] = ids;

    if (hasPartition(op) || isa<scf::YieldOp>(op))
      return WalkResult::advance();

    SetVector<Operation *> useOps;
    DenseSet<Value> visited;
    for (auto &use : op->getUses()) {
      getUseOps(use.get(), useOps, visited);
    }

    opsMap[op] = useOps;
    return WalkResult::advance();
  });

  std::function<void(Operation *, DenseSet<int> &)> getOpPartitionIds =
      [&](Operation *op, DenseSet<int> &opPartitionIds) {
        for (auto &region : op->getRegions()) {
          for (auto &block : region.getBlocks()) {
            for (auto &op_ : block.without_terminator()) {
              auto op = &op_;
              getOpPartitionIds(op, opPartitionIds);
            }
          }
        }
        auto partitionIds = partitionMap[op];
        opPartitionIds.insert(partitionIds.begin(), partitionIds.end());
      };

  auto iteratePartitions = [&]() {
    int maxIter = 100;
    while (maxIter-- > 0) {
      bool converged = true;
      for (auto [op, useOps] : opsMap) {
        auto oldPartitionIds = partitionMap[op];
        auto newPartitionIds = oldPartitionIds;
        for (auto useOp : useOps) {
          getOpPartitionIds(useOp, newPartitionIds);
        }
        converged = converged && oldPartitionIds == newPartitionIds;
        partitionMap[op] = newPartitionIds;
      }
      if (converged)
        break;
    }
    if (maxIter <= 0) {
      emitError(loop.getLoc(), "assignMissingPartitions failed to converge");
      return failure();
    }

    for (auto [op, partitionIds] : partitionMap) {
      if (partitionIds.empty())
        continue;
      setPartition(op,
                   SetVector<int>(partitionIds.begin(), partitionIds.end()));
    }
    return success();
  };
  if (failed(iteratePartitions())) {
    return failure();
  }

  // Work-around for use cases where the partitioner doesn't assign partitions
  // to scalar operations. This handles remaining scalars that have no partition
  // assignments by propagating partitions forward through the def-use chain.
  // Example scenario:
  //    %46 = scalar_op ..  @2     // has partition assignment
  //    %47 = scalar_op %46        // no partition assignment
  //    llvm.intr.assume %47: i1   // terminal use, no further uses
  std::function<void(Operation *, SetVector<Operation *> &,
                     DenseSet<Operation *> &)>
      getDefOps = [&](Operation *op, SetVector<Operation *> &defOps,
                      DenseSet<Operation *> &visited) {
        if (!visited.insert(op).second)
          return;
        for (auto value : op->getOperands()) {
          if (auto defOp = value.getDefiningOp()) {
            defOps.insert(defOp);
          }
        }
      };
  opsMap.clear();
  loop.walk([&](Operation *op) {
    if (hasPartition(op))
      return WalkResult::advance();
    // skip region ops and their terminators
    if (op->getNumRegions() > 0 ||
        isa<scf::YieldOp, triton::ReduceReturnOp>(op))
      return WalkResult::advance();

    // skip non-scalar ops that return value
    if (op->getNumResults() > 0 && !isScalarOp(op))
      return WalkResult::advance();

    SetVector<Operation *> defOps;
    DenseSet<Operation *> visited;
    getDefOps(op, defOps, visited);

    opsMap[op] = defOps;

    return WalkResult::advance();
  });

  if (failed(iteratePartitions())) {
    return failure();
  }

  return success();
}

void verifyPartitions(scf::ForOp loop, PartitionSet &partitions) {
  loop.walk([&](Operation *op) {
    if (hasPartition(op))
      return WalkResult::advance();
    if (op->hasAttr(kWarpSpecializeAttrName))
      return WalkResult::advance();
    if (isa<scf::YieldOp, triton::ReduceReturnOp>(op))
      return WalkResult::advance();
    llvm_unreachable("no partition");
  });
}

SetVector<int> getBlockPartitions(Block *block);
SmallVector<SetVector<int>> getYieldPartitions(Block *block) {
  auto terminator = block->getTerminator();
  SmallVector<SetVector<int>> yieldPartitions(terminator->getNumOperands());
  for (auto &opnd : terminator->getOpOperands()) {
    auto op = opnd.get().getDefiningOp();
    if (auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());
        forOp && isa<AsyncTokenType>(opnd.get().getType())) {
      // Heuristic: when for-op yields an async-token, the output partition of
      //            the token is that of its user.
      // At the moment token must have only one use
      auto arg = forOp.getRegionIterArg(opnd.getOperandNumber());
      assert(arg.hasOneUse());
      op = arg.getUses().begin()->getOwner();
      assert(op);
    }
    if (!op)
      continue;
    std::optional<SetVector<int>> partitionIds;
    if (hasPartition(op)) {
      partitionIds = getPartitionIds(op);
    }
    if (op->getNumRegions() > 0) {
      auto it = llvm::find(op->getResults(), opnd.get());
      assert(it != op->getResults().end());
      auto pos = it - op->getResults().begin();
      partitionIds = getPartitionOutputs(op)[pos];
    }
    if (!partitionIds) {
      // inherit from uses
      partitionIds = SetVector<int>();
      for (auto user : op->getUsers()) {
        if (auto op1 = block->findAncestorOpInBlock(*user);
            op1 && hasPartition(op1)) {
          auto ids = getPartitionIds(op1);
          partitionIds->insert(ids.begin(), ids.end());
        }
      }
    }
    yieldPartitions[opnd.getOperandNumber()] = *partitionIds;
  }
  return yieldPartitions;
}

SetVector<int>
setOutputPartitions(Operation *op, SetVector<int> opPartitions,
                    SmallVector<SetVector<int>> outputPartitions) {
  for (auto ids : outputPartitions) {
    opPartitions.insert(ids.begin(), ids.end());
  }
  setPartition(op, opPartitions);
  setPartitionOutputs(op, outputPartitions);
  return opPartitions;
}

SetVector<int> assignIfOpPartitions(scf::IfOp ifOp) {
  auto ifOpPartitions = getBlockPartitions(ifOp.thenBlock());
  auto thenYieldPartitions = getYieldPartitions(ifOp.thenBlock());
  if (!ifOp.elseBlock()) {
    return setOutputPartitions(ifOp, ifOpPartitions, thenYieldPartitions);
  }

  auto elsePartitions = getBlockPartitions(ifOp.elseBlock());
  ifOpPartitions.insert(elsePartitions.begin(), elsePartitions.end());

  auto elseYieldPartitions = getYieldPartitions(ifOp.elseBlock());
  assert(thenYieldPartitions.size() == elseYieldPartitions.size());
  SmallVector<SetVector<int>> outputPartitions;
  for (int i = 0; i < thenYieldPartitions.size(); ++i) {
    auto &thenIds = thenYieldPartitions[i];
    auto &elseIds = elseYieldPartitions[i];
    auto thenYieldOpnd = ifOp.thenYield()->getOperand(i);
    auto elseYieldOpnd = ifOp.elseYield()->getOperand(i);
    auto thenYieldOpndDefOp = thenYieldOpnd.getDefiningOp();
    auto elseYieldOpndDefOp = elseYieldOpnd.getDefiningOp();

    if (isa<AsyncTokenType>(thenYieldOpnd.getType())) {
      // Heuristic: when if-op yields an async-token, the output partition of
      //            the token is that of its producer
      if (ifOp.thenBlock()->findAncestorOpInBlock(
              *thenYieldOpnd.getDefiningOp())) {
        outputPartitions.push_back(elseIds);
      } else {
        outputPartitions.push_back(thenIds);
      }
    } else if (thenYieldOpndDefOp &&
               thenYieldOpndDefOp->getBlock() == ifOp.thenBlock()) {
      // Heuristic: if yield operand is defined in then block, use its Ids
      outputPartitions.push_back(thenIds);
    } else if (elseYieldOpndDefOp &&
               elseYieldOpndDefOp->getBlock() == ifOp.elseBlock()) {
      // same for else block
      outputPartitions.push_back(elseIds);
    } else {
      // otherwise pick thenIds if avaialble, otherwise elseIds
      outputPartitions.push_back(!thenIds.empty() ? thenIds : elseIds);
    }
  }
  return setOutputPartitions(ifOp, ifOpPartitions, outputPartitions);
}

SetVector<int> assignSingleRegionOpPartition(Operation *op) {
  auto block = &op->getRegion(0).getBlocks().front();
  auto blockPartitions = getBlockPartitions(block);
  return setOutputPartitions(op, blockPartitions, getYieldPartitions(block));
}

SetVector<int> getBlockPartitions(Block *block) {
  SetVector<int> blockPartitions;
  for (auto &op_ : block->without_terminator()) {
    auto op = &op_;
    SetVector<int> partitionIds;
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      partitionIds = assignIfOpPartitions(ifOp);
    } else if (isa<scf::ForOp, triton::ReduceOp>(op)) {
      partitionIds = assignSingleRegionOpPartition(op);
    } else if (hasPartition(op)) {
      auto ids = getPartitionIds(op);
      partitionIds.insert(ids.begin(), ids.end());
    }
    blockPartitions.insert(partitionIds.begin(), partitionIds.end());
  }
  return blockPartitions;
}

void assignRegionBodyPartition(scf::ForOp loop, PartitionSet &partitions) {
  loop->walk([&](Operation *op) {
    if (isa<scf::YieldOp, scf::ForOp>(op) || hasPartition(op))
      return WalkResult::advance();

    auto parentOp =
        op->getParentOfType<scf::ForOp>().getBody()->findAncestorOpInBlock(*op);
    if (!hasPartition(parentOp))
      return WalkResult::advance();

    auto partitionIds = getPartitionIds(parentOp);
    SetVector<Partition *> parentPartitions;
    for (auto id : partitionIds) {
      parentPartitions.insert(partitions.getPartition(id));
    }
    setPartition(op, parentPartitions);
    return WalkResult::advance();
  });

  loop->walk([&](Operation *op) {
    // remove partition attribute in ops that have regions
    // such op's partition set will be inferred from regions
    // in partition-loops pass
    if (!isa<scf::ForOp>(op) && hasPartition(op) && op->getNumRegions() > 0) {
      op->removeAttr(kPartitionAttrName);
    }
  });
}

void assignRegionOpPartitions(scf::ForOp loop) {
  assignSingleRegionOpPartition(loop);

  // Work-around for operations that don't produce results, nor use operands
  // from inside ws-loop, but need partition assignments. These operations
  // inherit partitions from their parent operation.
  //   %a = ...
  //   scf.for ... {
  //     scf.if ... {
  //       ...
  //       llvm.intr.assume %a : i1  // inherits partition from scf.if
  //       ...
  //     } {ttg.partition = [2]}
  //   } {ttg.ws}
  loop.walk([&](Operation *op) {
    if (op->getNumResults() > 0 || hasPartition(op))
      return WalkResult::advance();
    if (op->getNumRegions() > 0 ||
        isa<scf::YieldOp, triton::ReduceReturnOp>(op))
      return WalkResult::advance();
    auto parentOp = op->getParentOp();
    auto parentPartitionIds = getPartitionIds(parentOp);
    setPartition(op, parentPartitionIds);
    return WalkResult::advance();
  });
}

bool underWSLoop(Operation *op) {
  scf::ForOp topLevelFor = op->getParentOfType<scf::ForOp>();
  if (!topLevelFor) {
    return false;
  }

  if (topLevelFor->hasAttr(kWarpSpecializeAttrName)) {
    return true;
  } else {
    while (auto outer = topLevelFor->getParentOfType<scf::ForOp>()) {
      topLevelFor = outer;
      if (outer->hasAttr(kWarpSpecializeAttrName)) {
        return true;
      }
    }
  }

  return false;
}

class FoldTmemStoreIntoAlloc : public OpRewritePattern<ttng::TMEMAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMAllocOp alloc,
                                PatternRewriter &rewriter) const override {
    if (alloc.getSrc() || !underWSLoop(alloc)) {
      return failure();
    }

    for (auto user : alloc->getUsers()) {
      if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
        auto storeSrc = store.getSrc();
        if (auto storeSrcDef = storeSrc.getDefiningOp()) {
          DominanceInfo dom(storeSrcDef);
          if (dom.dominates(storeSrcDef, alloc)) {
            auto newAlloc = rewriter.create<ttng::TMEMAllocOp>(
                alloc.getLoc(), alloc.getResultTypes()[0],
                rewriter.getType<AsyncTokenType>(), storeSrc);

            if (auto allocTok = alloc.getToken()) {
              allocTok.replaceAllUsesWith(newAlloc.getToken());
            }
            if (auto storeTok = store.getToken()) {
              storeTok.replaceAllUsesWith(newAlloc.getToken());
            }
            if (hasPartition(store)) {
              setPartition(newAlloc, getPartitionIds(store));
            }
            rewriter.eraseOp(store);
            rewriter.replaceOp(alloc, newAlloc);
            return success();
          }
        }
      }
    }

    return failure();
  }
};

std::optional<std::pair<scf::ForOp, ttng::MMAv5OpInterface>>
getUniqueUserLoopAndMMA(ttng::TMEMAllocOp tmemAlloc) {
  auto tok = tmemAlloc.getToken();
  if (!tok || !tok.hasOneUse())
    return std::nullopt;
  auto loop = dyn_cast<scf::ForOp>(*tok.getUsers().begin());
  if (!loop)
    return std::nullopt;
  auto loopTok = loop.getBody()->getArgument(
      tok.getUses().begin()->getOperandNumber() - 2);
  if (!loopTok.hasOneUse())
    return std::nullopt;
  auto mma = dyn_cast<ttng::MMAv5OpInterface>(*loopTok.getUsers().begin());
  if (mma)
    return std::make_pair(loop, mma);
  return std::nullopt;
}

bool canRemoveTmemStore(ttng::TMEMAllocOp tmemAlloc) {
  auto opt = getUniqueUserLoopAndMMA(tmemAlloc);
  if (!opt)
    return false;
  auto [loop, mma] = *opt;
  auto useD = dyn_cast<BlockArgument>(mma.useAccumulator());
  if (!useD)
    return false;
  auto parent = useD.getParentBlock()->getParentOp();
  if (parent != loop)
    return false;
  auto loopInit = loop.getInitArgs()[useD.getArgNumber() - 1];
  auto val = getBoolFromConstant(loopInit);
  return val && val.value() == false;
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUPARTITIONSCHEDULING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct PartitionScheduling
    : public triton::gpu::impl::TritonGPUPartitionSchedulingBase<
          PartitionScheduling> {
  using TritonGPUPartitionSchedulingBase::TritonGPUPartitionSchedulingBase;

  void runOnOperation() override;
};
} // namespace

std::optional<ConstantIntRanges> getBoundFromCmpOp(arith::CmpIOp cmpOp,
                                                   Value anchor) {
  // The following was taken from third_party/amd/lib/Analysis/
  bool anchorIsLhs = cmpOp.getLhs() == anchor;
  unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(anchor.getType());
  APInt min = APInt::getSignedMinValue(bitWidth);
  APInt max = APInt::getSignedMaxValue(bitWidth);

  auto fold = getAsOpFoldResult(anchorIsLhs ? cmpOp.getRhs() : cmpOp.getLhs());
  if (auto constValue = getConstantIntValue(fold)) {
    bool isSigned = true;
    APInt apVal = {bitWidth, static_cast<uint64_t>(*constValue), isSigned};
    switch (cmpOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      return mlir::ConstantIntRanges::constant(apVal);
    case arith::CmpIPredicate::sge: {
      // K >= apVal implies K ∈ [apVal, max]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(apVal, max, isSigned);
      // apVal >= K implies K ∈ [min, apVal]
      return mlir::ConstantIntRanges::range(min, apVal, isSigned);
    }
    case arith::CmpIPredicate::sgt: {
      // K > apVal implies K >= apVal + 1 implies K ∈ [apVal + 1, max]
      if (anchorIsLhs) {
        return mlir::ConstantIntRanges::range(apVal + 1, max, isSigned);
      }
      // apVal > K implies apVal - 1 >= K implies K ∈ [min, apVal - 1]
      return mlir::ConstantIntRanges::range(min, apVal - 1, isSigned);
    }
    case arith::CmpIPredicate::sle: {
      // K <= apVal implies K ∈ [min, apVal]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(min, apVal, isSigned);
      // apVal <= K implies K ∈ [apVal, max]
      return mlir::ConstantIntRanges::range(apVal, max, isSigned);
    }
    case arith::CmpIPredicate::slt: {
      // K < apVal implies K <= apVal -1 implies K ∈ [min, apVal - 1]
      if (anchorIsLhs)
        return mlir::ConstantIntRanges::range(min, apVal - 1, isSigned);
      // apVal < K implies apVal + 1 <= K implies K ∈ [apVal + 1, max]
      return mlir::ConstantIntRanges::range(apVal + 1, max, isSigned);
    }
    default:
      break;
    }
  }
  return std::nullopt;
}

bool canProveExecuteOnce(scf::ForOp forOp) {
  auto getAssumedBound = [&](Value v) -> std::optional<ConstantIntRanges> {
    mlir::ForwardSliceOptions opt;
    SetVector<Operation *> slice;
    (void)getForwardSlice(v, &slice, opt);

    // For simplicity, we only handle an assume op directly operating on v. It's
    // possible to support more general cases, but they require a range
    // analysis.
    for (auto op : slice) {
      if (auto assumeOp = dyn_cast<LLVM::AssumeOp>(op)) {
        auto cond = assumeOp.getCond();
        if (auto cmpOp = cond.getDefiningOp<arith::CmpIOp>();
            cmpOp && (cmpOp.getLhs() == v || cmpOp.getRhs() == v)) {
          if (auto bound = getBoundFromCmpOp(cmpOp, v)) {
            return *bound;
          }
        }
      }
    }
    return std::nullopt;
  };

  auto getConstIntBound = [&](Value v) {
    unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(v.getType());
    if (auto cst = getConstantIntValue(getAsOpFoldResult(v))) {
      APInt apVal = {bitWidth, static_cast<uint64_t>(*cst), /*signed*/ true};
      return mlir::ConstantIntRanges::constant(apVal);
    } else if (auto assumedBound = getAssumedBound(v)) {
      return *assumedBound;
    } else {
      APInt min = APInt::getSignedMinValue(bitWidth);
      APInt max = APInt::getSignedMaxValue(bitWidth);
      return mlir::ConstantIntRanges::range(min, max, true);
    }
  };

  auto lbBound = getConstIntBound(forOp.getLowerBound());
  auto ubBound = getConstIntBound(forOp.getUpperBound());
  return mlir::intrange::evaluatePred(mlir::intrange::CmpPredicate::slt,
                                      lbBound, ubBound)
      .value_or(false);
}

bool hoistTmemAlloc(ttng::TMEMAllocOp allocToHoist) {
  // extra loop nest
  SmallVector<scf::ForOp> loopNest;
  auto currentForOp = allocToHoist->getParentOfType<scf::ForOp>();
  while (currentForOp && !currentForOp->hasAttr(kWarpSpecializeAttrName)) {
    loopNest.push_back(currentForOp);
    currentForOp = currentForOp->getParentOfType<scf::ForOp>();
  }

  if (!currentForOp) {
    return false;
  }
  loopNest.push_back(currentForOp);

  {
    // Check if hoisting across all loop nests is valid. Hoisting is invalid
    // when the inner loop that does MMA executes variable number of times
    // depending on the outer loop variables, and some instances of the inner
    // loops never execute while others do. So we hoist across loop nests only
    // in the following cases:
    // 1. The loop iteration counts for all loops do not depend on their outer
    // loop variables.
    // 2. If there is a loop whose iteration count depends on outer loop
    // varaibles, there is an llvm.intr.assume op from which we can prove that
    // the number of iteration is greater than zero.
    auto opt = getUniqueUserLoopAndMMA(allocToHoist);
    if (!opt) {
      return false;
    }
    auto mmaLoop = opt->first;
    SmallVector<scf::ForOp> innerLoopNest{mmaLoop};
    innerLoopNest.insert(innerLoopNest.begin(), loopNest.begin(),
                         loopNest.end() - 1);

    // Does the expression x depend on y?
    auto dependOn = [](Value x, Value y) {
      mlir::BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      SetVector<Operation *> slice;
      (void)getBackwardSlice(x, &slice, opt);
      for (auto user : y.getUsers()) {
        if (x.getDefiningOp() == user || slice.count(user)) {
          return true;
        }
      }
      return false;
    };

    for (auto [i, innerFor] : llvm::enumerate(innerLoopNest)) {
      for (int j = i; j < loopNest.size(); ++j) {
        auto outerForIter = loopNest[j].getInductionVar();
        if ((dependOn(innerFor.getLowerBound(), outerForIter) ||
             dependOn(innerFor.getUpperBound(), outerForIter)) &&
            !canProveExecuteOnce(innerFor)) {
          // Cannot hoist this tmem alloc across the outer loop loopNest[j]
          return false;
        }
      }
    }
  }

  // hoist to outside tt.warp_specialized loop
  allocToHoist->moveBefore(currentForOp);
  allocToHoist->removeAttr(kPartitionAttrName);

  Value token = allocToHoist.getToken();
  assert(token.hasOneUse());
  auto &tokenUse = *token.getUses().begin();
  auto tokenPos =
      tokenUse.getOperandNumber() - currentForOp.getNumControlOperands();
  auto tokenPartition = getPartitionOutputs(tokenUse.getOwner())[tokenPos];

  // thread token to for-op init/iter args from outer-to inner
  std::reverse(loopNest.begin(), loopNest.end());
  for (auto &forOp : loopNest) {
    OpBuilder b(forOp);
    int nArgs = forOp.getRegionIterArgs().size();
    forOp = addIterArgsToLoop(b, forOp, {token});

    // update partitions for the forOp
    if (forOp->hasAttr(kPartitionOutputsAttrName)) {
      auto partitionOuputs = getPartitionOutputs(forOp);
      partitionOuputs.push_back(tokenPartition);
      setPartitionOutputs(forOp, partitionOuputs);
    } else {
      setPartitionOutputs(forOp, {tokenPartition});
    }
    auto partitions = getPartitionIds(forOp);
    partitions.insert(tokenPartition.begin(), tokenPartition.end());
    setPartition(forOp, partitions);

    token = forOp.getRegionIterArg(nArgs);
  }

  // set inner loop init_args with updated token
  tokenUse.set(token);

  // get last produced token, the one w/o use
  token = tokenUse.getOwner()->getResult(tokenPos);
  while (!token.use_empty()) {
    assert(token.hasOneUse());
    auto tokenUser = *token.getUsers().begin();
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(tokenUser)) {
      token = load.getToken();
    } else if (auto store = dyn_cast<ttng::TMEMStoreOp>(tokenUser)) {
      token = store.getToken();
    } else {
      auto mma = cast<ttng::MMAv5OpInterface>(tokenUser);
      token = mma.getToken();
    }
  }

  // append token to yield, from inner to outer loop
  std::reverse(loopNest.begin(), loopNest.end());
  for (auto forOp : loopNest) {
    appendToForOpYield(forOp, {token});
    setPartition(forOp.getBody()->getTerminator(), getPartitionIds(forOp));
    token = forOp->getResults().back();
  }

  return true;
}

void PartitionScheduling::runOnOperation() {
  ModuleOp m = getOperation();
  SmallVector<scf::ForOp> loops;
  m.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName)) {
      loops.push_back(loop);
    }
  });

  for (auto [idx, loop] : llvm::enumerate(loops)) {
    if (std::optional<PartitionSet> partitions = getInitialPartitions(loop)) {
      propagatePartitions(loop, *partitions);
      optimizePartitions(loop, *partitions);
      assignRegionBodyPartition(loop, *partitions);
      if (failed(assignMissingPartitions(loop, *partitions)))
        return signalPassFailure();

      assignRegionOpPartitions(loop);
      verifyPartitions(loop, *partitions);
      loop->setAttr(
          kWarpSpecializeTagAttrName,
          IntegerAttr::get(IntegerType::get(loop.getContext(), 32), idx));

      SmallVector<Attribute> stages;
      Builder b(loop.getContext());
      for (Partition &partition : partitions->getPartitions())
        stages.push_back(b.getI32IntegerAttr(partition.getStage()));
      loop->setAttr(kPartitionStagesAttrName, b.getArrayAttr(stages));
    }
  }

  MLIRContext *context = &getContext();
  OpPassManager pm;
  mlir::RewritePatternSet patterns(context);
  patterns.add<FoldTmemStoreIntoAlloc>(context);
  ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
  if (failed(applyPatternsGreedily(m, std::move(patterns))))
    signalPassFailure();

  m.walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName)) {
      SmallVector<ttng::TMEMAllocOp> tmemAllocToHoist;
      loop.walk([&](ttng::TMEMAllocOp tmemAlloc) {
        if (tmemAlloc.getSrc() && canRemoveTmemStore(tmemAlloc)) {
          tmemAllocToHoist.push_back(tmemAlloc);
        }
      });

      for (auto alloc : tmemAllocToHoist) {
        if (!hoistTmemAlloc(alloc)) {
          SetVector<int> mmaPartition;
          mmaPartition.insert(1);
          // tmem store remaining in the outer loop must belong to the MMA
          // partition. This is necessary for correctly double buffering this
          // accumulator.
          setPartition(alloc, mmaPartition);
        }
      }
    }
  });
}
