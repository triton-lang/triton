#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

using Partition = WarpSchedule::Partition;

//===----------------------------------------------------------------------===//
// getPartitionScheme
//===----------------------------------------------------------------------===//

namespace {
struct PipelinedLoad {
  PipelinedLoad(Operation *loadOp)
      : loadOp(loadOp), type(getResult().getType()),
        sharedEnc(getSharedEncoding(loadOp)) {}

  TypedValue<RankedTensorType> getResult() const {
    return cast<TypedValue<RankedTensorType>>(loadOp->getResult(0));
  }
  unsigned getLoadSizeInBytes() const {
    return type.getNumElements() * type.getElementTypeBitWidth() / 8;
  }
  LogicalResult determineLiveRange(Block &container, DominanceInfo &domInfo,
                                   PostDominanceInfo &postDomInfo,
                                   WarpSchedule &schedule);

  Operation *loadOp;
  RankedTensorType type;
  SharedEncodingTrait sharedEnc;

  SmallVector<Operation *, 1> allocOps;
  SmallVector<Operation *, 1> liveBeforeOps;
  SmallVector<Operation *, 0> liveUntilOps;
  SmallVector<Operation *, 1> asyncUsers;
};

struct PipelinedMMA {
  PipelinedMMA(ttng::MMAv5OpInterface mmaOp) : mmaOp(mmaOp) {}

  ttng::MMAv5OpInterface mmaOp;
  ttng::TMEMStoreOp storeOp;
  SmallVector<Operation *> operandViews;
};

struct PartitionScheme {
  SmallVector<PipelinedLoad> loads;
  SmallVector<PipelinedMMA> mmas;
  SmallVector<math::Exp2Op> exps;
};
} // namespace

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

// Analyze the loop to find operations that should be outlined by warp
// specialization to overlap latencies.
static PartitionScheme getPartitionScheme(scf::ForOp loop) {
  // Find loads to pipeline.
  SmallVector<PipelinedLoad> loads;
  for (Operation &loadOp : loop.getOps()) {
    // Only TMA loads are supported at the moment.
    if (!isa<DescriptorLoadOp, DescriptorGatherOp>(loadOp))
      continue;

    PipelinedLoad &load = loads.emplace_back(&loadOp);
    // Local alloc users of the load with matching encoding will cause the
    // underlying buffer to be pass through. Keep track of them.
    for (Operation *user : loadOp.getUsers()) {
      if (auto alloc = dyn_cast<LocalAllocOp>(user)) {
        if (load.sharedEnc == alloc.getType().getEncoding())
          load.allocOps.push_back(alloc);
      } else if (isa<ttng::TMEMAllocOp>(user)) {
        load.allocOps.push_back(user);
      }
    }
  }

  // Find MMAs to pipeline.
  SmallVector<PipelinedMMA> mmas;
  for (auto mmaOp : loop.getOps<ttng::MMAv5OpInterface>()) {
    PipelinedMMA &mma = mmas.emplace_back(mmaOp);

    // If the store is unrelated to the use of the MMA, then it gets placed in
    // the MMA partition.
    auto storeOp = dyn_cast_or_null<ttng::TMEMStoreOp>(
        findDefOpInLoop(loop, mmaOp.getAccDep()));
    if (!ttng::hasAccReadModifyWrite(mmaOp, loop) && storeOp)
      mma.storeOp = storeOp;

    // Look for views into the operands.
    SmallVector<Operation *> operandViews;
    for (Value operand : mmaOp->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp())
        operandViews.push_back(defOp);
    }
    while (!operandViews.empty()) {
      Operation *op = operandViews.pop_back_val();
      if (!op->hasOneUse() || !op->hasTrait<OpTrait::MemDescViewTrait>())
        continue;
      mma.operandViews.push_back(op);
      if (Operation *defOp = op->getOperand(0).getDefiningOp())
        operandViews.push_back(defOp);
    }
  }

  // Look for large exp ops that will have significant MFU latency.
  SmallVector<math::Exp2Op> exps;
  for (auto expOp : loop.getOps<math::Exp2Op>()) {
    auto tensorTy = dyn_cast<RankedTensorType>(expOp.getType());
    if (tensorTy && tensorTy.getNumElements() > 256)
      exps.push_back(expOp);
  }

  return PartitionScheme{std::move(loads), std::move(mmas), std::move(exps)};
}

//===----------------------------------------------------------------------===//
// assignPartitions
//===----------------------------------------------------------------------===//

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
                            WarpSchedule &schedule) {
  SmallVector<Operation *> worklist{op};
  DenseSet<Operation *> seen;
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (!seen.insert(op).second)
      continue;
    Partition *p = schedule.getPartition(op);
    if (p && p != schedule.getRootPartition())
      return true;
    iterateDefs(loop, op,
                [&](OpResult def) { worklist.push_back(def.getDefiningOp()); });
  }
  return false;
}

// Recursively schedule the dependencies of an operation, stopping when
// encountering an operation that is already assigned.
static void scheduleDependencies(scf::ForOp loop, WarpSchedule &schedule,
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
    if (!defOp || !hasDefPartition(loop, defOp, schedule) ||
        !schedule.trySchedule(partition, defOp))
      continue;
    llvm::append_range(deps, getNestedOperands(defOp));
  }
}

// Recursively schedule the users of an operation, stopping when
// encountering an operation that is already assigned.
static void scheduleUsers(scf::ForOp loop, WarpSchedule &schedule,
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

    if (!schedule.trySchedule(partition, user))
      continue;
    for (OpOperand &use : user->getUses())
      uses.push_back(&use);
  }
}

// Given a partitioning scheme, determine an initial schedule by performing a
// first-order partition assignment to the operations in the scheme and its
// users and/or dependencies. This sets up the initial partitioning of the ops.
static WarpSchedule getInitialSchedule(const PartitionScheme &scheme,
                                       scf::ForOp loop) {
  WarpSchedule schedule;

  // Start by creating the default partition, a partition for for all loads, and
  // a partition for all MMAs.
  Partition *defaultPartition = schedule.addPartition(0);
  Partition *mmaPartition = schedule.addPartition(1);
  Partition *loadPartition = schedule.addPartition(0);

  for (const PipelinedLoad &load : scheme.loads) {
    schedule.trySchedule(loadPartition, load.loadOp);
    for (Operation *allocOp : load.allocOps)
      schedule.trySchedule(loadPartition, allocOp);
  }

  for (const PipelinedMMA &mma : scheme.mmas) {
    schedule.trySchedule(mmaPartition, mma.mmaOp);
    if (mma.storeOp)
      schedule.trySchedule(mmaPartition, mma.storeOp);
    for (Operation *viewOp : mma.operandViews)
      schedule.trySchedule(mmaPartition, viewOp);
  }

  // Propagate defs of exp.
  for (math::Exp2Op exp : scheme.exps) {
    schedule.trySchedule(defaultPartition, exp);
    scheduleDependencies(loop, schedule, defaultPartition, exp);
  }

  // Propagate users of loads and MMAs.
  for (const PipelinedLoad &load : scheme.loads) {
    scheduleUsers(loop, schedule, defaultPartition, load.loadOp);
    for (Operation *allocOp : load.allocOps)
      scheduleUsers(loop, schedule, defaultPartition, allocOp);
  }

  SmallVector<Partition *> userPartitions{defaultPartition};
  while (userPartitions.size() < scheme.mmas.size()) {
    userPartitions.push_back(schedule.addPartition(userPartitions.size()));
  }
  for (auto [mma, userPartition] : llvm::zip(scheme.mmas, userPartitions)) {
    scheduleUsers(loop, schedule, userPartition, mma.mmaOp);
  }
  for (const PipelinedMMA &mma : scheme.mmas) {
    scheduleDependencies(loop, schedule, defaultPartition, mma.mmaOp);
  }

  schedule.updatePartitions();
  return schedule;
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
void propagatePartitions(scf::ForOp loop, WarpSchedule &schedule) {
  OpClusters opClusters;

  for (Partition &partition : schedule.getPartitions()) {
    // For each partition, check if any of their inputs are reachable from
    // another partition and spawn a single cluster at that operation.
    auto defCallback = [&](OpResult result, unsigned distance) {
      Operation *defOp = result.getDefiningOp();
      if (!schedule.isScheduled(defOp) &&
          hasDefPartition(loop, defOp, schedule)) {
        // Add the current partition as a sink to the cluster.
        opClusters.getOrCreate(defOp)->sinkPartitions.insert(&partition);
      }
    };
    schedule.iterateDefs(loop, &partition, defCallback);

    // For each partition, place users of its outputs in a cluster if it is not
    // already assigned to a partition.
    auto useCallback = [&](OpResult result, OpOperand &use, unsigned distance) {
      Operation *user = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      if (!schedule.isScheduled(user)) {
        // Add the current partition as a def to the cluster.
        opClusters.getOrCreate(user)->defPartitions.insert(&partition);
      }
    };
    schedule.iterateUses(loop, &partition, useCallback);
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
      if (schedule.isScheduled(defOp)) {
        // The input originates from an operation already assigned to a
        // partition. Add this as a def partition.
        cluster->defPartitions.insert(schedule.getPartition(defOp));
      } else {
        // If the input is not reachable from a partition, ignore it.
        if (!hasDefPartition(loop, defOp, schedule))
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
      if (schedule.isScheduled(user)) {
        // If the user is already assigned to a partition, add that partition as
        // one of the sink partitions.
        Partition *userPartition = schedule.getPartition(user);
        cluster->sinkPartitions.insert(userPartition);
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
    assert(llvm::all_of(
        cluster.ops, [&](Operation *op) { return !schedule.isScheduled(op); }));

    // If there are multiple def or sink partitions, don't know what to do.
    // Assign the whole cluster to its own partition.
    if (cluster.defPartitions.size() > 1 || cluster.sinkPartitions.size() > 1) {
      Partition *newPartition = schedule.addPartition(0);
      for (Operation *op : cluster.ops)
        schedule.insert(newPartition, op);
      continue;
    }

    // If there is no sink partition, this means there is a backedge somewhere,
    // for now assign the cluster to the def partition.
    Partition *defPartition = cluster.defPartitions.front();
    if (cluster.sinkPartitions.empty()) {
      for (Operation *op : cluster.ops)
        schedule.insert(defPartition, op);
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
    schedule.iterateDefs(loop, sinkPartition, callback);
    for (unsigned i = 0; i < critPath.size(); ++i) {
      Operation *op = critPath[i];
      iterateDefs(loop, op, [&](OpResult def) {
        Operation *defOp = def.getDefiningOp();
        if (opsInCluster.contains(defOp))
          critPath.insert(defOp);
      });
    }

    // If all ops are on the critical path, assign them to the sink partition.
    if (critPath.size() == cluster.ops.size()) {
      for (Operation *op : cluster.ops)
        schedule.insert(sinkPartition, op);
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
      schedule.insert(sinkPartition, clone);
    }
    for (Operation *op : cluster.ops)
      schedule.insert(defPartition, op);
  }

  schedule.updatePartitions();
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

struct PartitionBuilder : public ImplicitLocOpBuilder {
  using ImplicitLocOpBuilder::ImplicitLocOpBuilder;

  Value intCst(int value, unsigned width = 32) {
    return create<arith::ConstantIntOp>(value, width);
  }
  Value boolCst(bool value) { return intCst(value, /*width=*/1); }

  void assignStage(Operation *op, std::optional<unsigned> stage) {
    if (stage)
      op->setAttr(kAssignedStageAttrName, getI32IntegerAttr(*stage));
  }

  template <typename OpT, typename... Args>
  auto createInto(Partition &partition, std::optional<unsigned> stage,
                  Args &&...args) {
    auto op = create<OpT>(std::forward<Args>(args)...);
    op->setAttr(kPartitionAttrName, getI32IntegerAttr(partition.getIndex()));
    assignStage(op, stage);
    partition.insert(op);
    return op;
  }
};

using StageMap = DenseMap<Operation *, std::optional<unsigned>>;

static void replaceAllUsesDominatedBy(Operation *domOp, Value newValue,
                                      Value oldValue, DominanceInfo &domInfo) {
  if (newValue == oldValue)
    return;
  oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
    return domInfo.properlyDominates(domOp, use.getOwner());
  });
}

static std::pair<Value, Value> postIncrementModulo(ImplicitLocOpBuilder &b,
                                                   Value index, Value phase,
                                                   unsigned numStages) {
  auto intCst = [&](int value) {
    return b.create<arith::ConstantIntOp>(value, 32);
  };
  Value nextIndex = b.create<arith::AddIOp>(index, intCst(1));
  Value nextPhase = b.create<arith::XOrIOp>(phase, intCst(1));

  Value rollover = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, nextIndex,
                                           intCst(numStages));
  nextIndex = b.create<arith::SelectOp>(rollover, intCst(0), nextIndex);
  nextPhase = b.create<arith::SelectOp>(rollover, nextPhase, phase);

  return {nextIndex, nextPhase};
}

static std::pair<BlockArgument, BlockArgument>
addIndexAndPhase(PartitionBuilder &b, scf::ForOp &loop, unsigned numStages,
                 Value epilogue = {}) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);

  // Index and phase both start at 0.
  unsigned curArgIdx = loop.getNumRegionIterArgs();
  auto newArgs = addIterArgsToLoop(b, loop, {b.intCst(0), b.intCst(0)});
  BlockArgument index = newArgs[0];
  BlockArgument phase = newArgs[1];

  // Post-increment the index and phase.
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  b.setInsertionPoint(yield);

  auto [nextIndex, nextPhase] = postIncrementModulo(b, index, phase, numStages);
  if (epilogue) {
    nextIndex = b.create<arith::SelectOp>(epilogue, nextIndex, index);
    nextPhase = b.create<arith::SelectOp>(epilogue, nextPhase, phase);
  }
  yield->insertOperands(yield.getNumOperands(), {nextIndex, nextPhase});

  return {index, phase};
}

static std::pair<Value, Operation *>
getUserPrecondition(ImplicitLocOpBuilder &b, scf::ForOp loop, Operation *domOp,
                    Value initialValue = {}) {
  // If the use is inside a loop besides the actual loop being pipelined, we
  // have to hoist the use up to that loop, otherwise the barriers will be
  // inserted in the loop.
  for (Operation *userLoop;
       loop != (userLoop = domOp->getParentOfType<LoopLikeOpInterface>());)
    domOp = userLoop;
  assert(loop->isProperAncestor(domOp));

  Value trueVal = b.create<arith::ConstantOp>(b.getBoolAttr(true));
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop.getBody()->findAncestorOpInBlock(*domOp));

  Value precondition = initialValue ? initialValue : trueVal;
  Operation *parentOp = domOp;
  while (loop != (parentOp = parentOp->getParentOp())) {
    assert(!isa<LoopLikeOpInterface>(parentOp));
    auto ifOp = dyn_cast<scf::IfOp>(parentOp);
    if (!ifOp) {
      llvm::report_fatal_error(
          "FIXME: unsupported parent operation for MMA user");
    }
    Value cond = ifOp.getCondition();
    if (domOp->getParentRegion() == &ifOp.getElseRegion())
      cond = b.create<arith::XOrIOp>(cond, trueVal);
    precondition = b.create<arith::AndIOp>(precondition, cond);
  }

  return {precondition, domOp};
}

static MemDescType getAsMutable(MemDescType type) {
  return MemDescType::get(type.getShape(), type.getElementType(),
                          type.getEncoding(), type.getMemorySpace(),
                          /*mutableMemory=*/true);
}

//===----------------------------------------------------------------------===//
// Load Pipelining
//===----------------------------------------------------------------------===//

// Find the last operation that consumes the in-memory result of a load. This
// only looks at the current loop iteration.
static LogicalResult
findSharedMemorySinkOps(Value value, SmallVectorImpl<Operation *> &sinkOps) {
  for (Operation *user : value.getUsers()) {
    if (isa<ttng::MMAv5OpInterface, LocalLoadOp>(user)) {
      sinkOps.push_back(user);
    } else if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      if (failed(findSharedMemorySinkOps(user->getResult(0), sinkOps)))
        return failure();
    } else {
      return mlir::emitWarning(user->getLoc(),
                               "failed to warp specialize: cannot handle sink "
                               "of in-memory load operation");
    }
  }
  return success();
}

LogicalResult PipelinedLoad::determineLiveRange(Block &container,
                                                DominanceInfo &domInfo,
                                                PostDominanceInfo &postDomInfo,
                                                WarpSchedule &schedule) {
  // Find the liveBefore and liveUntil operations of the load.
  llvm::MapVector<Partition *, SmallVector<Operation *>> regSinks, shmemSinks;
  for (Operation *user : loadOp->getUsers()) {
    auto it = llvm::find(allocOps, user);
    if (it == allocOps.end()) {
      // This is an in-register use of the load. The result must be live before
      // the op. Since it will be loaded out of shared memory, it only needs to
      // be live until the op as well.
      regSinks[schedule.getPartition(user)].push_back(user);
      continue;
    }
    SmallVector<Operation *> sinkOps;
    if (failed(findSharedMemorySinkOps((*it)->getResult(0), sinkOps)))
      return failure();
    for (Operation *sinkOp : sinkOps)
      shmemSinks[schedule.getPartition(sinkOp)].push_back(sinkOp);
  }
  SetVector<Partition *> userPartitions;
  userPartitions.insert_range(llvm::make_first_range(regSinks));
  userPartitions.insert_range(llvm::make_first_range(shmemSinks));

  // The result must be live before all the sinks in each partition.
  for (Partition *userPartition : userPartitions) {
    SmallVector<Operation *> regSink = regSinks.lookup(userPartition);
    SmallVector<Operation *> shmemSink = shmemSinks.lookup(userPartition);

    auto sinks = llvm::to_vector(llvm::concat<Operation *>(regSink, shmemSink));
    Operation *liveBeforeOp = findNearestCommonDominator(sinks, domInfo);
    liveBeforeOp = container.findAncestorOpInBlock(*liveBeforeOp);
    liveBeforeOps.push_back(liveBeforeOp);

    SmallVector<Operation *> shmemTerminals;
    for (Operation *sinkOp : shmemSink) {
      sinkOp = container.findAncestorOpInBlock(*sinkOp);
      // Async operations require the memory to be live as long as the operation
      // is in-flight. Each async operation is treated as a separate consumer.
      if (isa<ttng::MMAv5OpInterface>(sinkOp)) {
        asyncUsers.push_back(sinkOp);
        continue;
      }
      // The sink operation is synchronous and the memory is released after the
      // operation.
      shmemTerminals.push_back(sinkOp);
    }

    // Normalize the sink op to be one immediately under the loop. Then, the
    // memory must be live until after this operation.
    Operation *lastShmemSink =
        findNearestCommonPostDominator(shmemTerminals, postDomInfo);
    if (lastShmemSink)
      lastShmemSink = lastShmemSink->getNextNode();

    // The memory only needs to be live until before the first register user.
    Operation *liveUntilReg = findNearestCommonDominator(regSink, domInfo);
    if (liveUntilReg)
      liveUntilReg = container.findAncestorOpInBlock(*liveUntilReg);

    // The memory is live until before the first register user or after the last
    // shmem terminal, whichever is later.
    Operation *liveUntilOp;
    if (lastShmemSink && liveUntilReg) {
      liveUntilOp = liveUntilReg->isBeforeInBlock(lastShmemSink) ? lastShmemSink
                                                                 : liveUntilReg;
    } else if (liveUntilReg) {
      liveUntilOp = liveUntilReg;
    } else {
      liveUntilOp = lastShmemSink;
    }
    liveUntilOps.push_back(liveUntilOp);
  }

  return success();
}

namespace {

struct PipelinedLoadGroup {
  Location getLoc();
  void allocateAref(scf::ForOp &loop, int numStages);
  LogicalResult lowerLoads(WarpSchedule &schedule, DominanceInfo &domInfo,
                           PostDominanceInfo &postDomInfo,
                           std::optional<unsigned> stage,
                           const StageMap &stages);

  SmallVector<PipelinedLoad> loads;

  SmallVector<Value> loadBuffers;
  Value emptyBars;
  Value readyBars;
  BlockArgument index;
  BlockArgument phase;
};
} // namespace

Location PipelinedLoadGroup::getLoc() {
  SmallVector<Location> locs = llvm::map_to_vector(
      loads, [](PipelinedLoad &load) { return load.loadOp->getLoc(); });
  return FusedLoc::get(locs.front().getContext(), locs);
}

void PipelinedLoadGroup::allocateAref(scf::ForOp &loop, int numStages) {
  assert(loadBuffers.empty() && "already allocated");

  // Create buffers for each the loads.
  for (PipelinedLoad &load : loads) {
    loadBuffers.push_back(createAlloc(loop, load.type, load.loadOp->getLoc(),
                                      load.sharedEnc, numStages));
  }

  // Determine how many distinct consumers of the result there are.
  int maxLiveUntil = 0;
  DenseSet<Operation *> distinctAsyncUsers;
  for (PipelinedLoad &load : loads) {
    distinctAsyncUsers.insert(load.asyncUsers.begin(), load.asyncUsers.end());
    int numLiveUntil =
        llvm::count_if(load.liveUntilOps, [](Operation *op) { return !!op; });
    maxLiveUntil = std::max(maxLiveUntil, numLiveUntil);
  }
  int arriveCount = distinctAsyncUsers.size() + maxLiveUntil;

  // Share the same set of barriers all loads in the group.
  emptyBars = createBarrierAlloc(loop, numStages, arriveCount);
  readyBars = createBarrierAlloc(loop, numStages, /*arriveCount=*/1);
  // All buffers are initially in the empty state.
  PartitionBuilder b(getLoc(), loop);
  for (auto i : llvm::seq(numStages)) {
    Value emptyBar = createSingleBufferView(b, emptyBars, i);
    b.create<ttng::ArriveBarrierOp>(emptyBar, arriveCount);
  }

  std::tie(index, phase) = addIndexAndPhase(b, loop, numStages);
}

static void lowerTMACopy(PartitionBuilder &b, Partition &loadPartition,
                         std::optional<unsigned> stage, Operation *op,
                         Value barrier, Value view) {
  Value truePred = b.create<arith::ConstantIntOp>(true, /*width=*/1);
  if (auto load = dyn_cast<DescriptorLoadOp>(op)) {
    Value tmaPtr = b.createInto<ttng::TensorDescToTMAPtrOp>(
        loadPartition, stage, load.getDesc());
    auto indices = ttng::translateTMAIndices(
        b, load.getLoc(), load.getDesc().getType().getBlockType().getEncoding(),
        load.getIndices());
    b.createInto<ttng::AsyncTMACopyGlobalToLocalOp>(
        loadPartition, stage, tmaPtr, indices, barrier, view, truePred);
  } else {
    auto gather = cast<DescriptorGatherOp>(op);
    Value tmaPtr = b.createInto<ttng::TensorDescToTMAPtrOp>(
        loadPartition, stage, gather.getDesc());
    b.createInto<ttng::AsyncTMAGatherOp>(
        loadPartition, stage, tmaPtr, gather.getXOffsets(), gather.getYOffset(),
        barrier, view, truePred);
  }
}

LogicalResult PipelinedLoadGroup::lowerLoads(WarpSchedule &schedule,
                                             DominanceInfo &domInfo,
                                             PostDominanceInfo &postDomInfo,
                                             std::optional<unsigned> stage,
                                             const StageMap &stages) {
  // Insert before the group of loads.
  auto firstLoad = llvm::min_element(loads, [&](auto &lhs, auto &rhs) {
    return domInfo.properlyDominates(lhs.loadOp, rhs.loadOp);
  });
  Partition &loadPartition = *schedule.getPartition(firstLoad->loadOp);
  PartitionBuilder b(getLoc(), firstLoad->loadOp);

  // Producer acquire.
  Value curEmptyBar = createSingleBufferView(b, emptyBars, index);
  b.createInto<ttng::WaitBarrierOp>(loadPartition, stage, curEmptyBar, phase);

  // Indicate the expected size of the loads.
  unsigned loadSizeInBytes = 0;
  for (const PipelinedLoad &load : loads)
    loadSizeInBytes += load.getLoadSizeInBytes();
  Value curLoadBar = createSingleBufferView(b, readyBars, index);
  b.createInto<ttng::BarrierExpectOp>(loadPartition, stage, curLoadBar,
                                      loadSizeInBytes, b.boolCst(true));

  // Set up the consumer wait. We know the live before ops are the same for all
  // loads since that's how they were grouped.
  SetVector<Operation *> distinctAsyncUsers;
  DenseMap<Partition *, ttng::ArriveBarrierOp> arriveOps;
  for (auto [i, liveBeforeOp] : llvm::enumerate(firstLoad->liveBeforeOps)) {
    b.setInsertionPoint(liveBeforeOp);
    Partition &userPartition = *schedule.getPartition(liveBeforeOp);
    auto userStage = stages.lookup(liveBeforeOp);
    b.createInto<ttng::WaitBarrierOp>(userPartition, userStage, curLoadBar,
                                      phase);

    SmallVector<Operation *> liveUntilOps;
    for (PipelinedLoad &load : loads) {
      if (Operation *liveUntilOp = load.liveUntilOps[i])
        liveUntilOps.push_back(liveUntilOp);
    }
    if (!liveUntilOps.empty()) {
      Operation *liveUntilOp =
          findNearestCommonPostDominator(liveUntilOps, postDomInfo);
      b.setInsertionPoint(liveUntilOp);
      auto arriveOp = b.createInto<ttng::ArriveBarrierOp>(
          userPartition, userStage, curEmptyBar, 1);
      arriveOps[schedule.getPartition(liveUntilOp)] = arriveOp;
    }
  }

  // Handle async users distinct to the whole load group.
  for (PipelinedLoad &load : loads)
    distinctAsyncUsers.insert(load.asyncUsers.begin(), load.asyncUsers.end());
  for (Operation *asyncUser : distinctAsyncUsers) {
    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(asyncUser)) {
      mmaOp.addCompletionBarrier(curEmptyBar, b.boolCst(true));
      continue;
    }
    llvm::report_fatal_error("FIXME: unhandled async user of pipelined load: " +
                             asyncUser->getName().getStringRef());
  }

  // Now create the async loads.
  for (auto [load, buffer] : llvm::zip(loads, loadBuffers)) {
    b.setInsertionPoint(load.loadOp);
    Value view = createSingleBufferView(b, buffer, index);
    lowerTMACopy(b, loadPartition, stage, load.loadOp, curLoadBar, view);
    // Propagate through shared memory uses.
    for (Operation *allocOp : load.allocOps) {
      replaceUsesAndPropagateType(b, allocOp, view);
      allocOp->erase();
    }
    // If there are remaining users, they must be in-register.
    llvm::MapVector<Partition *, SmallVector<OpOperand *>> regUses;
    for (OpOperand &use : load.loadOp->getUses())
      regUses[schedule.getPartition(use.getOwner())].push_back(&use);
    for (auto &[partition, uses] : regUses) {
      auto users = llvm::to_vector(llvm::map_range(
          uses, [](OpOperand *use) { return use->getOwner(); }));
      if (Operation *arriveOp = arriveOps.lookup(partition))
        users.push_back(arriveOp);
      Operation *loadBeforeOp = findNearestCommonDominator(users, domInfo);
      b.setInsertionPoint(loadBeforeOp);
      Value loaded = b.createInto<LocalLoadOp>(
          *partition, stages.lookup(loadBeforeOp), load.type, view);
      for (OpOperand *use : uses)
        use->set(loaded);
    }
    load.loadOp->erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// MMA Pipelining
//===----------------------------------------------------------------------===//

static Value getLastInductionValue(PartitionBuilder &b, scf::ForOp loop) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);
  // (ub - lb -1) // step * step + lb
  Value diff =
      b.create<arith::SubIOp>(loop.getUpperBound(), loop.getLowerBound());
  diff = b.create<arith::SubIOp>(diff, b.intCst(1));
  Value ceilStep = b.create<arith::MulIOp>(
      b.create<arith::DivSIOp>(diff, loop.getStep()), loop.getStep());
  return b.create<arith::AddIOp>(ceilStep, loop.getLowerBound());
}

static LogicalResult pipelineMMA(scf::ForOp &loop, PipelinedMMA &mma,
                                 WarpSchedule &schedule, DominanceInfo &domInfo,
                                 PostDominanceInfo &postDomInfo,
                                 std::optional<unsigned> stage,
                                 const StageMap &stages) {
  ttng::MMAv5OpInterface mmaOp = mma.mmaOp;
  auto fail = [&](StringRef msg) { return emitWarning(mmaOp.getLoc(), msg); };
  Block &body = *loop.getBody();
  auto inBody = [&](Operation *op) { return body.findAncestorOpInBlock(*op); };

  // Determine if the MMA accumulator can be multibuffered.
  bool accIsMultiBuffered =
      // MMAs in subsequent iterations can be overlapped.
      !ttng::hasAccReadModifyWrite(mmaOp, loop) &&
      // The accumulator is reset at some point, thus allowing multibuffering.
      ttng::isAccMultibufferingPossible(mmaOp, loop) &&
      // The user didn't disable it with a flag.
      !getDisallowAccMultiBuffer(loop);

  // Check that the accumulator can be multi-buffered.
  ttng::TMEMAllocOp oldAllocOp =
      mmaOp.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!oldAllocOp)
    return fail("accumulator is not a TMEM alloc");
  for (Operation *user : oldAllocOp.getResult().getUsers()) {
    if (!loop->getParentRegion()->isAncestor(user->getParentRegion()))
      return fail("cannot track accumulator uses");
  }

  PartitionBuilder b(mmaOp.getLoc(), oldAllocOp);
  int numMmaStages = 1 + accIsMultiBuffered;
  ttng::TMEMAllocOp allocOp =
      createTMemAlloc(b, oldAllocOp, /*multiBuffered=*/true, numMmaStages);

  // Use placeholder values for the indices in the loop.
  auto indexPhase = addIterArgsToLoop(b, loop, {b.intCst(0), b.intCst(0)});
  BlockArgument index = indexPhase[0];
  BlockArgument phase = indexPhase[1];

  // Replace uses of the accumulator before the loop with buffer 0, and replace
  // those after the loop with the last buffer.
  Value firstView = createSingleBufferView(b, allocOp, b.intCst(0));
  b.setInsertionPointAfter(loop);
  Value lastIndex = loop.getResult(index.getArgNumber() - 1);
  Value lastPhase = loop.getResult(phase.getArgNumber() - 1);
  Value lastView = createSingleBufferView(b, allocOp, lastIndex);

  // Find users of the accumulator in the loop and sort them by program order.
  SmallVector<Operation *> usersInLoop;
  for (OpOperand &use :
       llvm::make_early_inc_range(oldAllocOp.getResult().getUses())) {
    Operation *user = use.getOwner();
    if (user->getParentRegion() == loop->getParentRegion()) {
      if (loop->isBeforeInBlock(user))
        use.set(lastView);
      else
        use.set(firstView);
    } else if (loop.getBodyRegion().isAncestor(user->getParentRegion())) {
      usersInLoop.push_back(user);
    } else {
      return fail("cannot trace accumulator use");
    }
  }
  llvm::sort(usersInLoop, [&](Operation *lhs, Operation *rhs) {
    return inBody(lhs)->isBeforeInBlock(inBody(rhs));
  });

  // Find the read and overwrite points.
  Operation *overwriteOp = nullptr, *readOp = nullptr;
  for (Operation *user : usersInLoop) {
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(user)) {
      overwriteOp = storeOp;
    } else if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user)) {
      if (!matchPattern(mmaOp.useAccumulator(), m_One()))
        overwriteOp = mmaOp;
    } else if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(user)) {
      readOp = loadOp;
    } else {
      llvm::report_fatal_error("FIXME: unhandled MMA accumulator user");
    }
  }

  if (!overwriteOp)
    overwriteOp = mmaOp;
  if (!readOp)
    readOp = overwriteOp;

  struct Node {
    Operation *op;
    Partition *partition;
    Value barPrev;
    Value barNext;
    Value index;
    Value phase;
  };

  SmallVector<Node, 3> nodes{Node{overwriteOp}, Node{mmaOp}, Node{readOp}};
  llvm::sort(nodes, [&](Node &lhs, Node &rhs) {
    return inBody(lhs.op)->isBeforeInBlock(inBody(rhs.op));
  });

  for (int i = 0; i < nodes.size(); ++i) {
    Node &cur = nodes[i];
    Node &next = nodes[(i + 1) % nodes.size()];
    if (schedule.getPartition(inBody(cur.op)) !=
        schedule.getPartition(inBody(next.op))) {
      cur.barNext = createBarrierAlloc(loop, numMmaStages);
      next.barPrev = cur.barNext;
    }
  }

  Value firstBar;
  for (int i = nodes.size(); i > 0; --i) {
    if ((firstBar = nodes[i % nodes.size()].barPrev))
      break;
  }
  if (firstBar) {
    for (auto i : llvm::seq(numMmaStages)) {
      b.setInsertionPoint(loop);
      Value bar = createSingleBufferView(b, firstBar, i);
      b.create<ttng::ArriveBarrierOp>(bar, /*arriveCount=*/1);
    }
  }
  Value userPred = b.boolCst(true);
  if (readOp == mmaOp) {
    PartitionBuilder b(mmaOp.getLoc(), mmaOp);
    Value lastInductionValue = getLastInductionValue(b, loop);
    userPred = b.create<arith::CmpIOp>(
        arith::CmpIPredicate::eq, loop.getInductionVar(), lastInductionValue);
    nodes.back().barNext = createBarrierAlloc(loop, /*numBarriers=*/1);
  }

  Value curIndex = index, curPhase = phase;
  b.setInsertionPoint(loop);
  Value replTok = b.create<ub::PoisonOp>(b.getType<AsyncTokenType>());
  DenseSet<Operation *> seen;
  std::optional<OpBuilder::InsertPoint> incrementPt;
  for (Node &node : nodes) {
    node.index = curIndex;
    node.phase = curPhase;
    if (incrementPt && node.barPrev && node.barPrev != firstBar) {
      b.setInsertionPoint(loop);
      b.create<ttng::ArriveBarrierOp>(
          createSingleBufferView(b, node.barPrev, 0), /*arriveCount=*/1);
    }
    if (!seen.insert(node.op).second)
      continue;
    b.setInsertionPoint(node.op);
    Value view = createSingleBufferView(b, allocOp, node.index);
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(node.op)) {
      storeOp.getDstMutable().assign(view);
      storeOp.getDepMutable().clear();
      storeOp.getToken().replaceAllUsesWith(replTok);
    } else if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(node.op)) {
      loadOp.getSrcMutable().assign(view);
      loadOp.getDepMutable().clear();
      loadOp.getToken().replaceAllUsesWith(replTok);
    } else {
      assert(node.op == mmaOp);
      mmaOp.setAccumulator(view);
      mmaOp.getAccDepMutable().clear();
      mmaOp.getToken().replaceAllUsesWith(replTok);
    }
    if (node.op == dyn_cast<ttng::TMEMLoadOp>(readOp)) {
      ImplicitLocOpBuilder b(readOp->getLoc(), loop);
      userPred = getUserPrecondition(b, loop, node.op).first;
      b.setInsertionPointAfter(inBody(readOp));
      auto [nextIndex, nextPhase] =
          postIncrementModulo(b, index, phase, numMmaStages);
      curIndex = b.create<arith::SelectOp>(userPred, nextIndex, index);
      curPhase = b.create<arith::SelectOp>(userPred, nextPhase, phase);
      incrementPt = b.saveInsertionPoint();
    }
  }
  oldAllocOp.getToken().replaceAllUsesWith(allocOp.getToken());
  oldAllocOp.erase();
  cast<scf::YieldOp>(loop.getBody()->getTerminator())
      .getResultsMutable()
      .append({curIndex, curPhase});

  // Find operands that need to be pipelined through shmem.
  SmallVector<std::pair<Operation *, Partition *>> operandDefs;
  for (Value operand : mma.mmaOp->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp || !loop.getBodyRegion().isAncestor(defOp->getParentRegion()))
      continue;
    defOp = inBody(defOp);
    Partition *defPartition = schedule.getPartition(defOp);
    if (!defPartition)
      continue;
    if (auto allocOp = operand.getDefiningOp<LocalAllocOp>()) {
      PartitionBuilder b(allocOp.getLoc(), allocOp);
      auto store = b.createInto<LocalStoreOp>(*defPartition, std::nullopt,
                                              allocOp.getSrc(), allocOp);
      operandDefs.emplace_back(body.findAncestorOpInBlock(*store),
                               defPartition);
      allocOp->moveBefore(loop);
      allocOp->removeAttr(kPartitionAttrName);
      allocOp.getSrcMutable().clear();
      allocOp.getResult().setType(getAsMutable(allocOp.getType()));
    } else if (auto tmemAllocOp = operand.getDefiningOp<ttng::TMEMAllocOp>()) {
      PartitionBuilder b(tmemAllocOp.getLoc(), tmemAllocOp);
      auto store = b.createInto<ttng::TMEMStoreOp>(
          *defPartition, std::nullopt, Type(), tmemAllocOp.getResult(), Value(),
          tmemAllocOp.getSrc(), b.boolCst(true));
      operandDefs.emplace_back(body.findAncestorOpInBlock(*store),
                               defPartition);
      tmemAllocOp->moveBefore(loop);
      tmemAllocOp->removeAttr(kPartitionAttrName);
      tmemAllocOp.getSrcMutable().clear();
      tmemAllocOp.getResult().setType(getAsMutable(tmemAllocOp.getType()));
    }
  }

  for (Node &node : nodes) {
    Partition *partition = schedule.getPartition(inBody(node.op));
    PartitionBuilder b(node.op->getLoc(), loop);

    SmallVector<Operation *> defs;
    defs.push_back(node.op);

    // Find operand defs that come from the same partition and incorporate them
    // in this synchronization edge.
    decltype(operandDefs) nextOperandDefs;
    for (auto &[defOp, defPartition] : operandDefs) {
      if (defPartition == partition && inBody(node.op)->isBeforeInBlock(mmaOp))
        defs.push_back(defOp);
      else
        nextOperandDefs.emplace_back(defOp, defPartition);
    }
    operandDefs = std::move(nextOperandDefs);

    Operation *domOp = findNearestCommonDominator(defs, domInfo);
    Operation *lastOp = findNearestCommonPostDominator(defs, postDomInfo);

    if (node.barPrev) {
      if (!isa<ttng::TMEMLoadOp>(node.op)) {
        if (incrementPt && domOp->isBeforeInBlock(&*incrementPt->getPoint()))
          b.restoreInsertionPoint(*incrementPt);
        else
          b.setInsertionPoint(domOp);
        Value bar = createSingleBufferView(b, node.barPrev, curIndex);
        b.createInto<ttng::WaitBarrierOp>(*partition, stages.lookup(node.op),
                                          bar, curPhase, userPred);
      } else {
        b.setInsertionPoint(domOp);
        if (isa<scf::IfOp>(domOp->getParentOp()))
          b.setInsertionPointToStart(domOp->getBlock());
        Value bar = createSingleBufferView(b, node.barPrev, node.index);
        b.createInto<ttng::WaitBarrierOp>(*partition, stages.lookup(node.op),
                                          bar, node.phase);
      }
    }
    if (node.barNext) {
      if (mmaOp == node.op) {
        b.setInsertionPoint(mmaOp);
        Value bar = createSingleBufferView(b, node.barNext, node.index);
        mmaOp.addCompletionBarrier(bar, userPred);
        b.assignStage(mmaOp, stage);
      } else {
        b.setInsertionPointAfter(lastOp);
        if (isa<scf::IfOp>(lastOp->getParentOp()))
          b.setInsertionPoint(lastOp->getBlock()->getTerminator());
        Value bar = createSingleBufferView(b, node.barNext, node.index);
        b.createInto<ttng::ArriveBarrierOp>(*partition, stages.lookup(lastOp),
                                            bar, 1);
      }
    }
  }

  // Handle leftover operand defs.
  llvm::MapVector<Partition *, SmallVector<Operation *>> operandDefsMap;
  for (auto &[defOp, defPartition] : operandDefs)
    operandDefsMap[defPartition].push_back(defOp);
  for (auto &[partition, defs] : operandDefsMap) {
    Value emptyBar = createBarrierAlloc(loop, /*numBarriers=*/1);
    Value readyBar = createBarrierAlloc(loop, /*numBarriers=*/1);
    PartitionBuilder b(defs.front()->getLoc(), loop);
    b.create<ttng::ArriveBarrierOp>(emptyBar, /*arriveCount=*/1);

    Operation *domOp = findNearestCommonDominator(defs, domInfo);
    Operation *lastOp = findNearestCommonPostDominator(defs, postDomInfo);

    auto [index, phase] = addIndexAndPhase(b, loop, /*numStages=*/1);
    auto srcStage = stages.lookup(domOp);
    b.setInsertionPoint(domOp);
    b.createInto<ttng::WaitBarrierOp>(*partition, srcStage, emptyBar, phase);

    b.setInsertionPointAfter(lastOp);
    b.createInto<ttng::ArriveBarrierOp>(*partition, srcStage, readyBar, 1);

    b.setInsertionPoint(mmaOp);
    b.createInto<ttng::WaitBarrierOp>(*schedule.getPartition(mmaOp), stage,
                                      readyBar, phase);
    mmaOp.addCompletionBarrier(emptyBar, b.boolCst(true));
  }

  if (nodes.back().barNext) {
    b.setInsertionPointAfter(loop);
    // Re-acquire loop results as they may have been invalidated.
    Value lastIndex = loop.getResult(index.getArgNumber() - 1);
    Value lastPhase = loop.getResult(phase.getArgNumber() - 1);
    Value lastBar = createSingleBufferView(b, nodes.back().barNext, lastIndex);
    b.create<ttng::WaitBarrierOp>(lastBar, lastPhase);
  }

  llvm::SetVector<Operation *> predOps;
  Operation *hoistPt =
      findNearestCommonDominator(llvm::to_vector(userPred.getUsers()), domInfo);
  if (!hoistPt)
    return success();
  if (!getDominatingValueSetOpsToHoist(
          domInfo, body.findAncestorOpInBlock(*hoistPt), userPred, predOps))
    return fail("failed to hoist predicate ops above MMA");
  hoistOpsBefore(hoistPt, predOps);
  return success();
}

//===----------------------------------------------------------------------===//
// lowerLoops
//===----------------------------------------------------------------------===//

LogicalResult lowerLoops(scf::ForOp &loop, PartitionScheme &scheme,
                         WarpSchedule &schedule, int numLoadStages) {
  Block &body = *loop.getBody();
  DominanceInfo domInfo(loop);
  PostDominanceInfo postDomInfo(loop);

  // Group loads by common first user operations. This ensures, for example,
  // that multiple loads feeding into the same MMA op are placed together.
  llvm::MapVector<ArrayRef<Operation *>, SmallVector<PipelinedLoad>>
      liveBeforeGroups;
  for (PipelinedLoad &load : scheme.loads) {
    if (failed(load.determineLiveRange(body, domInfo, postDomInfo, schedule)))
      return failure();
    liveBeforeGroups[load.liveBeforeOps].push_back(std::move(load));
  }
  SmallVector<PipelinedLoadGroup> loadGroups;
  for (auto &loads : llvm::make_second_range(liveBeforeGroups))
    loadGroups.push_back({std::move(loads)});

  // Assign stages to ops when there are multiple ops in the same partition.
  StageMap stages;
  if (loadGroups.size() > 1) {
    unsigned curLoadStage = 0;
    for (const PipelinedLoadGroup &group : loadGroups) {
      stages.insert({group.loads.front().loadOp, curLoadStage});
      curLoadStage += 2;
    }
  }
  if (scheme.mmas.size() > 1) {
    unsigned curMMAStage = 0;
    for (const PipelinedMMA &mma : scheme.mmas) {
      stages.insert({mma.mmaOp, curMMAStage});
      curMMAStage += 2;
    }
  }

  // Multi-buffer and lower the loads.
  for (PipelinedLoadGroup &group : loadGroups)
    group.allocateAref(loop, numLoadStages);

  for (PipelinedLoadGroup &group : loadGroups) {
    if (failed(group.lowerLoads(schedule, domInfo, postDomInfo,
                                stages.lookup(group.loads.front().loadOp),
                                stages)))
      return failure();
  }

  // Multi-buffer and lower the MMAs.
  for (PipelinedMMA &mma : scheme.mmas) {
    if (failed(pipelineMMA(loop, mma, schedule, domInfo, postDomInfo,
                           stages.lookup(mma.mmaOp), stages)))
      return failure();
  }

  schedule.updatePartitions();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPULOADMMASPECIALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct LoadMMASpecialization
    : triton::gpu::impl::TritonGPULoadMMASpecializationBase<
          LoadMMASpecialization> {
  using TritonGPULoadMMASpecializationBase::TritonGPULoadMMASpecializationBase;

  void runOnOperation() override;
};
} // namespace

void LoadMMASpecialization::runOnOperation() {
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttr(kWarpSpecializeAttrName))
      loops.push_back(loop);
  });
  for (scf::ForOp loop : loops) {
    PartitionScheme scheme = getPartitionScheme(loop);
    if (scheme.loads.empty() && scheme.mmas.empty())
      continue;
    WarpSchedule schedule = getInitialSchedule(scheme, loop);
    propagatePartitions(loop, schedule);
    schedule.serialize(loop);
    int loopNumStages = getNumStagesOrDefault(loop, numStages);
    if (failed(lowerLoops(loop, scheme, schedule, loopNumStages)))
      continue;
    // HACK: Set this attribute so that LowerLoops will multi-buffer TMA
    // descriptors.
    loop->setAttr(kScheduledMaxStageAttrName,
                  Builder(&getContext()).getI32IntegerAttr(loopNumStages));
  }
}
