#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPULOOPSCHEDULING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

bool hasGpuBarriers(scf::ForOp forOp) {
  WalkResult result = forOp.walk(
      [&](mlir::gpu::BarrierOp barrier) { return WalkResult::interrupt(); });
  return result.wasInterrupted();
}

// Return true if the preconditions for pipelining the loop are met.
bool isSafeToPipeline(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (loopHasDistGreaterThanOne(forOp))
    return false;
  // Don't pipeline outer loops.
  if (isOuterLoop(forOp))
    return false;
  // Skip loops with barriers.
  if (hasGpuBarriers(forOp))
    return false;
  return true;
}

bool hasLatenciesAssigned(scf::ForOp forOp,
                          const DenseMap<Operation *, int> &opLatency) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opLatency.count(&op))
      return true;
  }
  return false;
}

CoarseSchedule scheduleKeyOps(scf::ForOp forOp,
                              const DenseMap<Operation *, int> &opLatency) {
  llvm::MapVector<Operation *, int> opToStage;
  // Find terminator for later reference
  auto terminator = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  // Determine all operations that have a non-zero latency
  SmallVector<Operation *> latOps;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opLatency.count(&op))
      latOps.push_back(&op);
  }
  // If no latency ops, nothing to schedule
  if (latOps.empty())
    return CoarseSchedule(0);

  // Compute the longest path to the yield for each operation reachable
  // from any latency operation.
  DenseMap<Operation *, int> distance;
  std::function<int(Operation *)> computeDistance = [&](Operation *op) -> int {
    auto it = distance.find(op);
    if (it != distance.end())
      return it->second;
    // Compute max distance among all users that are inside the loop body
    int maxDist = -1;
    for (Operation *user : op->getUsers()) {
      // Only consider users inside the same block and not the terminator
      Operation *inBlockUser = forOp.getBody()->findAncestorOpInBlock(*user);
      if (!inBlockUser || inBlockUser == terminator)
        continue;
      int distUser = computeDistance(inBlockUser);
      if (distUser > maxDist)
        maxDist = distUser;
    }
    int lat = 0;
    if (opLatency.count(op))
      lat = opLatency.lookup(op);
    // If an op has no users (maxDist == -1) but has latency, we include its
    // latency otherwise it contributes 0 to the distance.
    int d = lat + (maxDist < 0 ? 0 : maxDist);
    distance[op] = d;
    return d;
  };

  // Compute distances for all latency-starting ops
  int maxDistance = 0;
  for (Operation *latOp : latOps) {
    int d = computeDistance(latOp);
    if (d > maxDistance)
      maxDistance = d;
  }

  // Assign stage to each op reachable from a latency op
  for (auto &kv : distance) {
    Operation *op = kv.first;
    int dist = kv.second;
    // We only schedule ops that are downstream of a latency op
    // (had a non-negative distance due to a latency op).
    if (dist >= 0)
      opToStage[op] = maxDistance - dist;
  }

  auto stages = llvm::make_second_range(opToStage);
  int maxStage = *llvm::max_element(stages);
  CoarseSchedule schedule(maxStage + 1);
  SmallVector<CoarseSchedule::Cluster> clusters(maxStage + 1);
  for (int i = 0; i <= maxStage; i++) {
    clusters[i] = schedule.clusters.newAtBack();
  }
  CoarseSchedule::Cluster epilogue = schedule.clusters.newAtBack();
  // Assign ops to the clusters in reverse-stage order;
  // ops with higher stage numbers are assigned first. This way we will
  // end up with roughly reverse program order in the clusters.
  for (auto [op, stage] : opToStage) {
    if (isa<scf::IfOp>(op)) {
      schedule.insert(op, stage, epilogue);
      continue;
    }
    schedule.insert(op, stage, clusters[maxStage - stage]);
  }

  return schedule;
}

// Find dependencies with distance of 1. They will go to the next stage,
// but in the cluster before the current op.
void scheduleDistanceOneDependencies(scf::ForOp forOp,
                                     CoarseSchedule &schedule) {
  int numStages = schedule.getNumStages();
  auto getNestedOperands = [](Operation *op) -> SmallVector<Value> {
    SmallVector<Value> operands;
    op->walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (operand.getParentBlock()->getParentOp()->isAncestor(nestedOp))
          operands.push_back(operand);
      }
    });
    return operands;
  };

  // Mapping from the cluster to the cluster before it.
  DenseMap<CoarseSchedule::Cluster *, CoarseSchedule::Cluster> dist1Cluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      continue;
    auto [stage, cluster] = schedule[&op];
    // Can't schedule past the last stage.
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op.getBlock()) {
          auto yieldOp = op.getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp && schedule.count(defOp) == 0) {
            if (isa<tt::LoadOp>(defOp)) {
              // Exception: Schedule loads with a distance of 1 together
              // with the current op.
              schedule.insertIfAbsent(defOp, stage, cluster);
              schedule.insertDepsOfOp(defOp, stage, cluster, true);
            } else {
              if (dist1Cluster.count(&cluster) == 0) {
                dist1Cluster[&cluster] = schedule.clusters.newBefore(cluster);
              }
              schedule.insertIfAbsent(defOp, stage + 1, dist1Cluster[&cluster]);
              schedule.insertDepsOfOp(defOp, stage + 1, dist1Cluster[&cluster],
                                      true);
            }
          }
        }
      }
    }
  }
}

// Schedule the prologue and epilogue `if` ops in the loop, pushing them as
// close to the loop boundaries as possible. Return the cluster after the
// prologue (or the beginning of the loop if there is no prologue).
CoarseSchedule::Cluster schedulePrologueAndEpilogue(scf::ForOp forOp,
                                                    CoarseSchedule &schedule) {
  int numStages = schedule.getNumStages();
  CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  // Look for the IfOp that is in the backward slice any of the currently
  // scheduled ops and put it at the beginning of the loop.
  DenseMap<scf::IfOp, int> ifsToStage;
  // Go stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : schedule.getOpsInOrder(forOp)) {
      if (stage_ != stage)
        continue;
      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      getBackwardSlice((Operation *)op, &backwardSlice, opt);

      for (auto op : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          ifsToStage.insert({ifOp, stage});
        }
      }
    }
  }
  if (!ifsToStage.empty()) {
    CoarseSchedule::Cluster prologueCluster = schedule.clusters.newAtFront();
    for (auto [ifOp, stage] : ifsToStage) {
      schedule.insert(ifOp, stage, prologueCluster);
    }
  }

  // Other IfOps should be pushed to the end.
  CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (ifsToStage.count(ifOp) == 0) {
        schedule.insertIfAbsent(ifOp, numStages - 1,
                                epilogueCluster); // after prefetch extracts
      }
    }
  }
  return afterPrologue;
}

void scheduleRemainingToLastStage(scf::ForOp forOp, CoarseSchedule &schedule,
                                  CoarseSchedule::Cluster afterPrologue) {
  int numStages = schedule.getNumStages();
  // Assign the rest of the ops to the last stage.
  // Take care of the ordering of the ops - uses cannot be scheduled to the
  // cluster before the definition.
  DenseMap<Operation *, CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0) {
      opToCluster[&op] = afterPrologue;
    }
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    // We really only care about the producers from the last stage.
    // Others will be scheduled before these ops anyway.
    if (stage == numStages - 1) {
      queue.push_back(op);
    }
  }
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (auto user : op->getUsers()) {
      if (opToCluster.count(user)) {
        CoarseSchedule::Cluster userCluster = opToCluster[user];
        CoarseSchedule::Cluster opCluster;
        if (schedule.count(op))
          opCluster = schedule[op].second;
        else
          opCluster = opToCluster[op];
        if (*userCluster < *opCluster) {
          opToCluster[user] = opCluster;
          queue.push_back(user);
        }
      }
    }
  }
  for (auto [op, cluster] : opToCluster) {
    schedule.insert(op, numStages - 1, cluster);
  }
}

void scheduleLoop(scf::ForOp forOp,
                  const DenseMap<Operation *, int> &opLatency) {
  if (!hasLatenciesAssigned(forOp, opLatency) || !isSafeToPipeline(forOp))
    return;
  // Based on the latencies, schedule the key ops to the stages.
  CoarseSchedule schedule = scheduleKeyOps(forOp, opLatency);
  if (schedule.empty())
    return;
  LLVM_DEBUG({
    LDBG("Initial coarse schedule:");
    schedule.dump();
  });
  // Schedule the dependencies
  CoarseSchedule::Cluster afterPrologue =
      schedulePrologueAndEpilogue(forOp, schedule);
  LLVM_DEBUG({
    LDBG("Coarse schedule with prologue and epilogue:");
    schedule.dump();
  });
  scheduleDependencies(forOp, schedule);
  LLVM_DEBUG({
    LDBG("Coarse schedule with dependencies:");
    schedule.dump();
  });
  scheduleDistanceOneDependencies(forOp, schedule);
  LLVM_DEBUG({
    LDBG("Coarse schedule with dist 1:");
    schedule.dump();
  });
  scheduleRemainingToLastStage(forOp, schedule, afterPrologue);
  LLVM_DEBUG({
    LDBG("Final coarse schedule:");
    schedule.dump();
  });

  // Write the schedule to the IR
  schedule.serialize(forOp);
}

/////////////////////////////////////////////////////////////////////////////////////
// LOWERING
/////////////////////////////////////////////////////////////////////////////////////

class OpBuilderForStage : public OpBuilder {
  std::optional<int> _stage;
  std::optional<CoarseSchedule::Cluster> _cluster;
  CoarseSchedule &_schedule;

public:
  explicit OpBuilderForStage(Operation *op, CoarseSchedule &schedule, int stage,
                             CoarseSchedule::Cluster cluster)
      : OpBuilder(op, nullptr), _schedule(schedule), _stage(stage),
        _cluster(cluster) {}
  explicit OpBuilderForStage(Operation *op, CoarseSchedule &schedule)
      : OpBuilder(op, nullptr), _schedule(schedule) {
    if (_schedule.count(op)) {
      auto sc = _schedule[op];
      _stage = sc.first;
      _cluster = sc.second;
    }
  }
  void setStageCluster(std::pair<int, CoarseSchedule::Cluster> stageCluster) {
    _stage = stageCluster.first;
    _cluster = stageCluster.second;
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    OpTy op = OpBuilder::create<OpTy>(std::forward<Args>(args)...);
    if (_stage && _cluster) {
      _schedule.insertIfAbsent(op, *_stage, *_cluster);
    }
    return op;
  }
};

Operation *getFirstUseOfPipelinedOp(Operation *op, CoarseSchedule &schedule) {
  Operation *firstUser = nullptr;
  for (Operation *user : op->getUsers()) {
    Operation *topLevelUser = op->getBlock()->findAncestorOpInBlock(*user);
    assert(schedule.count(topLevelUser) && "op user not found in the schedule");
    auto [_useStage, _useCluster] = schedule[topLevelUser];
    if (!firstUser) {
      firstUser = topLevelUser;
    } else {
      auto [_firstUserStage, _firstUserCluster] = schedule[firstUser];
      if (_useStage < _firstUserStage ||
          (_useStage == _firstUserStage &&
           schedule.clusters.isBefore(_useCluster, _firstUserCluster))) {
        firstUser = topLevelUser;
      }
    }
  }
  return firstUser;
}

int getDefUseStageDiff(Operation *op, scf::ForOp forOp,
                       CoarseSchedule &schedule) {
  assert(schedule.count(op) && "LoadOp not found in the schedule");
  auto [defStage, _] = schedule[op];
  int useStage = INT_MAX;
  for (auto user : op->getUsers()) {
    Operation *topLevelUser = forOp.getBody()->findAncestorOpInBlock(*user);
    assert(schedule.count(topLevelUser) && "op user not found in the schedule");
    auto [_useStage, _] = schedule[topLevelUser];
    useStage = std::min(_useStage, useStage);
  }
  assert(useStage >= defStage && "LoadOp used before defined");
  return useStage - defStage;
}

ttg::SharedEncodingTrait getSharedEncoding(tt::LoadOp loadOp) {
  // Try to use local alloc encoding if possible.
  ttg::SharedEncodingTrait localAllocEnc;
  if (llvm::any_of(loadOp->getUsers(), [&](Operation *user) {
        return isa<ttg::LocalAllocOp>(user);
      })) {
    for (auto user : loadOp->getUsers()) {
      auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
      if (!localAlloc)
        continue;
      auto enc = mlir::cast<ttg::SharedEncodingTrait>(
          localAlloc.getType().getEncoding());
      if (!localAllocEnc) {
        localAllocEnc = enc;
      }
      if (enc != localAllocEnc) {
        // Some users have different encoding than others.
        // Use one of the encodings, and warn about the performance issue.
        // TODO pawel: report the warning.
        return localAllocEnc;
      }
    }
  }

  // TODO pawel: Add the case for TMA loads.

  // Try to use dot encoding if possible.
  bool incompatible = false;
  localAllocEnc =
      getSharedEncIfAllUsersAreDotEnc(loadOp.getResult(), incompatible)
          .value_or(nullptr);

  if (!localAllocEnc) {
    // Use generic layout. This won't be optimal for 2D tensors.
    auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
    auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
    auto blockedOrder = ttg::getOrder(ty.getEncoding());
    SmallVector<unsigned> order;
    if (blockedOrder.size() == 3) {
      for (unsigned i = 0; i < blockedOrder.size(); ++i) {
        if (blockedOrder[i] == 0)
          continue;
        order.push_back(blockedOrder[i]);
      }
      order.push_back(0);
    } else {
      order = blockedOrder;
    }
    localAllocEnc = ttg::SwizzledSharedEncodingAttr::get(ty.getContext(), 1, 1,
                                                         1, order, ctaLayout);
  }
  return localAllocEnc;
}

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, Operation *loadOp,
                         ttg::SharedEncodingTrait sharedEnc,
                         unsigned distance) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type memdescType = ttg::MemDescType::get(bufferShape, ty.getElementType(),
                                           sharedEnc, sharedMemorySpace,
                                           /*mutableMemory=*/true);
  Value alloc =
      builder.create<ttg::LocalAllocOp>(loadOp->getLoc(), memdescType);
  return alloc;
}

template <typename BuilderT, typename... Args>
Operation *createWithStage(BuilderT &builder, Location loc, int stage,
                           CoarseSchedule::Cluster cluster, Args &&...args) {
  Operation *op = builder.template create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, std::forward<Args>(args)...);

  return op;
}

// Check if the load can be pipelined entirely in shared memory, with user
// consuming directly the shared memory, without going through registers.
bool canBeShmemPipelined(tt::LoadOp loadOp) {
  // TODO pawel: think about it.
  return true;
}

void createAsyncCopy(scf::ForOp forOp, tt::LoadOp loadOp, Value alloc,
                     Value insertIdx, Value extractIdx,
                     CoarseSchedule &schedule) {
  OpBuilderForStage builder(forOp, schedule);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);

  Operation *firstUse = getFirstUseOfPipelinedOp(loadOp, schedule);
  // Replace the load with async copy, wait and loal_load.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loadOp);
  builder.setStageCluster(schedule[loadOp]);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();
  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());

  // TODO pawel: think about optimizing the blocked layout of indirect loads
  // here

  // Create async copy
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  ttg::MemDescType subviewTy = ttg::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true,
      /*allocShape=*/allocTy.getAllocShape());
  auto view =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);
  Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, src, view, mask, other, loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile());
  Operation *commit =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copy->getResult(0));

  // Create wait and local load
  builder.setStageCluster(schedule[firstUse]);
  Operation *wait =
      builder.create<ttg::AsyncWaitOp>(loc, commit->getResult(0), 0);
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  auto localLoad = builder.create<ttg::LocalLoadOp>(
      loc, loadOp.getType(), viewLoad, wait->getResult(0));
  auto result = localLoad->getResults();

  // Create select for other values as they are not handled by
  // AsyncCopyGlobalToLocalOp for now.
  if (other && !isZeroConst(other)) {
    auto select = builder.create<arith::SelectOp>(
        loc, loadOp.getType(),
        // Use the mask operand from the original load, not the one with a
        // potentially transformed layout.
        loadOp.getMask(), localLoad.getResult(), other);
    result = select->getResults();
  }

  // TODO: Think about prefetching the load for MMAv2

  loadOp->replaceAllUsesWith(result);
  loadOp->erase();
}

scf::ForOp lowerLoads(scf::ForOp forOp, CoarseSchedule &schedule) {
  struct AsyncLoad {
    int stageDiff;
    Value alloc;
    Value barrier;
    SharedEncodingTrait sharedEncoding;
  };
  struct LoadGroupInfo {
    Value insertIdx;
    Value extractIdx;
    Value phase;
    bool hasTMALoad = false;
  };
  llvm::MapVector<tt::LoadOp, AsyncLoad> asyncLoads;
  llvm::MapVector<int, LoadGroupInfo> loadGroups;
  forOp.getBody()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      int stageDiff = getDefUseStageDiff(loadOp, forOp, schedule);
      SharedEncodingTrait sharedEncoding = getSharedEncoding(loadOp);
      // Do not create async loads for small loads (cp.async requires at least 4
      // bytes)
      int copyVecBytes = getCopyVecBytes(
          cast<RankedTensorType>(loadOp.getType()), sharedEncoding);
      if (stageDiff > 0 && copyVecBytes >= 4) {
        asyncLoads[loadOp] = {.stageDiff = stageDiff,
                              .sharedEncoding = sharedEncoding};
      }
    } else if (isa<scf::ForOp>(op)) {
      // Skip nested loops.
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  for (auto &[loadOp, asyncLoad] : asyncLoads) {
    if (!isa<RankedTensorType>(loadOp.getType())) {
      continue;
    }

    Value alloc = createAlloc(forOp, loadOp, asyncLoad.sharedEncoding,
                              asyncLoad.stageDiff);
    asyncLoad.alloc = alloc;
    loadGroups.insert({asyncLoad.stageDiff, {}});
  }

  IRRewriter builder(forOp);
  builder.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  // Create a counter to index into the allocations per loop iteration.
  // NOTE: We create two duplicates values, insertIdx and extractIdx so that the
  // pipeliner will re-materialize the value in later stages of the pipeline
  // instead of carrying it as a dependency across multiple iterations.
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  SmallVector<Value> newOperands;
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  for (auto [_, loadGroup] : loadGroups) {
    newOperands.push_back(minusOne); // insertIdx
    newOperands.push_back(minusOne); // extractIdx
    // TODO: Add tma support / phase
  }

  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;

  // Update yield op with temporary yield values
  auto forYield = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  for (unsigned i = 0; i < newOperands.size(); ++i) {
    forYield.getResultsMutable().append(newOperands[i]);
  }

  builder.setInsertionPoint(forOp);
  loc = forOp.getLoc();
  int argIdx = newOperandIndex;
  for (auto &[numBuffers, loadGroup] : loadGroups) {
    Value insertIdx = newForOp.getBody()->getArgument(argIdx);
    argIdx++;
    Value extractIdx = newForOp.getBody()->getArgument(argIdx);
    argIdx++;
    // TODO: Add tma support / phase

    // Create two counters for the insert and extract indices to avoid creating
    // long liverange.
    builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());

    Value numBuffersVal =
        builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
    insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
    Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 insertIdx, numBuffersVal);
    insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);
    loadGroup.insertIdx = insertIdx;

    extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
    // Duplicate the constant to keep it from being carried across loops.
    numBuffersVal = builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
    Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 extractIdx, numBuffersVal);
    extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);
    loadGroup.extractIdx = extractIdx;
  }

  for (auto [loadOp, asyncLoad] : asyncLoads) {
    auto [insertIdx, extractIdx, phase, _] = loadGroups[asyncLoad.stageDiff];
    createAsyncCopy(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                    schedule);
  }
  // Patch the yield with the updated counters. Subtract to account for the loop
  // counter.
  argIdx = newOperandIndex - 1;
  for (auto &[numBuffers, loadGroup] : loadGroups) {
    forYield.setOperand(argIdx++, loadGroup.insertIdx);
    forYield.setOperand(argIdx++, loadGroup.extractIdx);
  }

  // TODO: Maybe we can annotate all of the new ops manually instead of
  // discoveringDependencies?
  scheduleDependencies(forOp, schedule);

  // Make sure all ops have attributes.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    assert(schedule.count(&op) && "op not found in the schedule");
  }
  return forOp;
}

void lowerLoop(scf::ForOp forOp) {
  CoarseSchedule schedule;
  schedule.deSerialize(forOp);
  scf::ForOp newForOp = lowerLoads(forOp, schedule);
  schedule.serialize(newForOp);
}

}; // namespace

void scheduleLoops(ModuleOp moduleOp,
                   const DenseMap<Operation *, int> &opLatency) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    scheduleLoop(forOp, opLatency);
  }
}

void lowerLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    lowerLoop(forOp);
  }
}

class TritonGPULoopSchedulingPass
    : public impl::TritonGPULoopSchedulingBase<TritonGPULoopSchedulingPass> {
public:
  using impl::TritonGPULoopSchedulingBase<
      TritonGPULoopSchedulingPass>::TritonGPULoopSchedulingBase;

  void runOnOperation() override {
    // Go over the interesting ops and assign latencies (based on the
    // numStages) to the them, trying to populate the allowed stages. This
    // step will be at some point extracted to separate pass that will be run
    // only for loops missing the latency information.
    DenseMap<Operation *, int> opLatency =
        assignLatencies(getOperation(), numStages);
    // numStages should not be used below this point. We should know everything
    // based on the assigned stages

    // Schedule the loops
    scheduleLoops(getOperation(), opLatency);

    // Transform the loop by introducing async operations to prepare it for
    // pipeline expansion.
    if (enableLowering) {
      lowerLoops(getOperation());
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
