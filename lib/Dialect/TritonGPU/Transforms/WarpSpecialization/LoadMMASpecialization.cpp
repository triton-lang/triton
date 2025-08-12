#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
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
  SmallVector<std::pair<Operation *, bool>, 0> liveUntilOps;
  SmallVector<Operation *, 1> asyncUsers;
};

struct PipelinedMMA {
  PipelinedMMA(ttng::MMAv5OpInterface mmaOp) : mmaOp(mmaOp) {}

  ttng::MMAv5OpInterface mmaOp;
};
} // namespace

static std::pair<SmallVector<PipelinedLoad>, SmallVector<PipelinedMMA>>
getPartitionScheme(scf::ForOp loop, const WarpSchedule &schedule) {
  SmallVector<PipelinedLoad> loads;
  SmallVector<PipelinedMMA> mmas;

  for (Operation &op : loop.getOps()) {
    if (!isa<DescriptorLoadOp, DescriptorGatherOp>(op))
      continue;
    auto &load = loads.emplace_back(&op);
    for (Operation *user : op.getUsers()) {
      if (schedule.getPartition(user) == schedule.getPartition(&op) &&
          isa<LocalAllocOp, ttng::TMEMAllocOp>(user))
        load.allocOps.push_back(user);
    }
  }

  for (auto mmaOp : loop.getOps<ttng::MMAv5OpInterface>()) {
    mmas.emplace_back(mmaOp);
  }

  return {std::move(loads), std::move(mmas)};
}

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

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
  loop = addIterArgsToLoop(b, loop, {b.intCst(0), b.intCst(0)});
  auto newArgs = loop.getRegionIterArgs().take_back(2);
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

static Value getUserPrecondition(ImplicitLocOpBuilder &b, scf::ForOp loop,
                                 Operation *domOp) {
  // If the use is inside a loop besides the actual loop being pipelined, we
  // have to hoist the use up to that loop, otherwise the barriers will be
  // inserted in the loop.
  for (Operation *userLoop;
       loop != (userLoop = domOp->getParentOfType<LoopLikeOpInterface>());)
    domOp = userLoop;
  assert(loop->isProperAncestor(domOp));

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);
  Value trueVal = b.create<arith::ConstantOp>(b.getBoolAttr(true));

  Value userPred = trueVal;
  Operation *parentOp = domOp;
  b.setInsertionPoint(loop.getBody()->findAncestorOpInBlock(*domOp));
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
    userPred = b.create<arith::AndIOp>(userPred, cond);
  }

  return userPred;
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

    // The memory only needs to be live until before the first register user.
    Operation *liveUntilReg = findNearestCommonDominator(regSink, domInfo);
    if (liveUntilReg)
      liveUntilReg = container.findAncestorOpInBlock(*liveUntilReg);

    // The memory is live until before the first register user or after the last
    // shmem terminal, whichever is later.
    std::pair<Operation *, bool> liveUntilOp{nullptr, false};
    if (lastShmemSink && liveUntilReg) {
      if (liveUntilReg->isBeforeInBlock(lastShmemSink))
        liveUntilOp = {lastShmemSink, /*after=*/true};
      else
        liveUntilOp = {liveUntilReg, /*after=*/false};
    } else if (liveUntilReg) {
      liveUntilOp = {liveUntilReg, /*after=*/false};
    } else {
      liveUntilOp = {lastShmemSink, /*after=*/true};
    }
    liveUntilOps.push_back(liveUntilOp);
  }

  return success();
}

static void propagateMutability(Value value) {
  for (Operation *user : value.getUsers()) {
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      user->getResult(0).setType(
          getAsMutable(cast<MemDescType>(user->getResult(0).getType())));
      propagateMutability(user->getResult(0));
    }
  }
}

namespace {

struct PipelinedLoadGroup {
  Location getLoc();
  void allocateAref(scf::ForOp &loop, int numStages);
  LogicalResult lowerLoads(WarpSchedule &schedule, DominanceInfo &domInfo,
                           PostDominanceInfo &postDomInfo);

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
        llvm::count_if(load.liveUntilOps, [](auto p) { return !!p.first; });
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
                         StageCluster stageCluster, Operation *op,
                         Value barrier, Value view) {
  Value truePred = b.boolCst(true);
  if (auto load = dyn_cast<DescriptorLoadOp>(op)) {
    auto indices = ttng::translateTMAIndices(
        b, load.getLoc(), load.getDesc().getType().getBlockType().getEncoding(),
        load.getIndices());
    b.createInto<ttng::AsyncTMACopyGlobalToLocalOp>(loadPartition, stageCluster,
                                                    load.getDesc(), indices,
                                                    barrier, view, truePred);
  } else {
    auto gather = cast<DescriptorGatherOp>(op);
    b.createInto<ttng::AsyncTMAGatherOp>(
        loadPartition, stageCluster, gather.getDesc(), gather.getXOffsets(),
        gather.getYOffset(), barrier, view, truePred);
  }
}

LogicalResult PipelinedLoadGroup::lowerLoads(WarpSchedule &schedule,
                                             DominanceInfo &domInfo,
                                             PostDominanceInfo &postDomInfo) {
  // Insert before the group of loads.
  auto firstLoad = llvm::min_element(loads, [&](auto &lhs, auto &rhs) {
    return domInfo.properlyDominates(lhs.loadOp, rhs.loadOp);
  });
  Partition &loadPartition = *schedule.getPartition(firstLoad->loadOp);
  PartitionBuilder b(getLoc(), firstLoad->loadOp);
  StageCluster stageCluster = getStageCluster(firstLoad->loadOp);

  // Producer acquire.
  Value curEmptyBar = createSingleBufferView(b, emptyBars, index);
  b.createInto<ttng::WaitBarrierOp>(loadPartition, stageCluster, curEmptyBar,
                                    phase);

  // Indicate the expected size of the loads.
  unsigned loadSizeInBytes = 0;
  for (const PipelinedLoad &load : loads)
    loadSizeInBytes += load.getLoadSizeInBytes();
  Value curLoadBar = createSingleBufferView(b, readyBars, index);
  b.createInto<ttng::BarrierExpectOp>(loadPartition, stageCluster, curLoadBar,
                                      loadSizeInBytes, b.boolCst(true));

  // Set up the consumer wait. We know the live before ops are the same for all
  // loads since that's how they were grouped.
  SetVector<Operation *> distinctAsyncUsers;
  DenseMap<Partition *, ttng::ArriveBarrierOp> arriveOps;
  for (auto [i, liveBeforeOp] : llvm::enumerate(firstLoad->liveBeforeOps)) {
    b.setInsertionPoint(liveBeforeOp);
    Partition &userPartition = *schedule.getPartition(liveBeforeOp);
    StageCluster userStageCluster = getStageCluster(liveBeforeOp);
    b.createInto<ttng::WaitBarrierOp>(userPartition, userStageCluster,
                                      curLoadBar, phase);

    SmallVector<Operation *> liveUntilOps;
    for (PipelinedLoad &load : loads) {
      auto [liveUntilOp, after] = load.liveUntilOps[i];
      if (liveUntilOp) {
        liveUntilOps.push_back(after ? liveUntilOp->getNextNode()
                                     : liveUntilOp);
      }
    }
    if (!liveUntilOps.empty()) {
      Operation *liveUntilOp =
          findNearestCommonPostDominator(liveUntilOps, postDomInfo);
      b.setInsertionPoint(liveUntilOp);
      auto arriveOp = b.createInto<ttng::ArriveBarrierOp>(
          userPartition, userStageCluster, curEmptyBar, 1);
      arriveOps[schedule.getPartition(liveUntilOp)] = arriveOp;
    }
  }

  // Handle async users distinct to the whole load group.
  for (PipelinedLoad &load : loads)
    distinctAsyncUsers.insert(load.asyncUsers.begin(), load.asyncUsers.end());
  for (Operation *asyncUser : distinctAsyncUsers) {
    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(asyncUser)) {
      mmaOp.addCompletionBarrier(curEmptyBar, b.boolCst(true));
      mmaOp.setIsAsync(true);
      continue;
    }
    llvm::report_fatal_error("FIXME: unhandled async user of pipelined load: " +
                             asyncUser->getName().getStringRef());
  }

  // Now create the async loads.
  for (auto [load, buffer] : llvm::zip(loads, loadBuffers)) {
    b.setInsertionPoint(load.loadOp);
    Value view = createSingleBufferView(b, buffer, index);
    lowerTMACopy(b, loadPartition, stageCluster, load.loadOp, curLoadBar, view);
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
      StageCluster userStageCluster = getStageCluster(loadBeforeOp);
      Value loaded = b.createInto<LocalLoadOp>(*partition, userStageCluster,
                                               load.type, view);
      b.createInto<ttng::FenceAsyncSharedOp>(*partition, userStageCluster,
                                             /*bCluster=*/false);
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

static LogicalResult pipelineMMA(scf::ForOp &loop, PipelinedMMA &mma,
                                 WarpSchedule &schedule, DominanceInfo &domInfo,
                                 PostDominanceInfo &postDomInfo) {
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
  loop = addIterArgsToLoop(b, loop, {b.intCst(0), b.intCst(0)});
  auto indexPhase = loop.getRegionIterArgs().take_back(2);
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

  // If the first node has a barrier, fully initialize it to let it run.
  if (nodes.front().barPrev) {
    for (auto i : llvm::seq(numMmaStages)) {
      b.setInsertionPoint(loop);
      Value bar = createSingleBufferView(b, nodes.front().barPrev, i);
      b.create<ttng::ArriveBarrierOp>(bar, /*arriveCount=*/1);
    }
  }

  Value userPred = b.boolCst(true);
  if (readOp == mmaOp) {
    PartitionBuilder b(mmaOp.getLoc(), mmaOp);
    Value lastInductionValue = [&]() {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPoint(loop);
      return getLastInductionValue(b, loop);
    }();
    userPred = b.create<arith::CmpIOp>(
        arith::CmpIPredicate::eq, loop.getInductionVar(), lastInductionValue);
    nodes.back().barNext = createBarrierAlloc(loop, /*numBarriers=*/1);
  }

  Value curIndex = index, curPhase = phase;
  b.setInsertionPoint(loop);
  Value replTok = b.create<ub::PoisonOp>(b.getType<AsyncTokenType>());
  DenseSet<Operation *> seen;
  std::optional<OpBuilder::InsertPoint> incrementPt;
  Node *firstAfterInc = nullptr;
  for (Node &node : nodes) {
    node.index = curIndex;
    node.phase = curPhase;
    if (incrementPt && node.barPrev && !firstAfterInc)
      firstAfterInc = &node;
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
      userPred = getUserPrecondition(b, loop, node.op);
      b.setInsertionPointAfter(inBody(readOp));
      auto [nextIndex, nextPhase] =
          postIncrementModulo(b, index, phase, numMmaStages);
      curIndex = b.create<arith::SelectOp>(userPred, nextIndex, index);
      curPhase = b.create<arith::SelectOp>(userPred, nextPhase, phase);
      incrementPt = b.saveInsertionPoint();
    }
  }
  if (firstAfterInc) {
    b.setInsertionPoint(loop);
    if (firstAfterInc->op == mmaOp) {
      Value firstBar = createSingleBufferView(b, firstAfterInc->barPrev, 0);
      b.create<ttng::ArriveBarrierOp>(firstBar, /*arriveCount=*/1);
    } else {
      assert(firstAfterInc->op == dyn_cast<ttng::TMEMStoreOp>(overwriteOp));
      for (auto i : llvm::seq(numMmaStages)) {
        Value firstBar = createSingleBufferView(b, firstAfterInc->barPrev, i);
        b.create<ttng::ArriveBarrierOp>(firstBar, /*arriveCount=*/1);
      }
    }
  }
  oldAllocOp.getToken().replaceAllUsesWith(allocOp.getToken());
  oldAllocOp.erase();
  cast<scf::YieldOp>(loop.getBody()->getTerminator())
      .getResultsMutable()
      .append({curIndex, curPhase});

  // Find operands that need to be pipelined through shmem.
  SmallVector<Value> incomingOperands;
  llvm::append_range(incomingOperands, mmaOp->getOperands());
  SmallVector<std::pair<Operation *, Partition *>> operandDefs;
  while (!incomingOperands.empty()) {
    Value operand = incomingOperands.pop_back_val();
    if (!isa<MemDescType>(operand.getType()))
      continue;
    Operation *defOp = operand.getDefiningOp();
    if (!defOp || loop.isDefinedOutsideOfLoop(operand))
      continue;
    defOp = inBody(defOp);
    Partition *defPartition = schedule.getPartition(defOp);

    if (!defPartition || defPartition == schedule.getRootPartition()) {
      // If the MMA operand is coming from outside the loop, move the alloc out.
      auto allocOp = dyn_cast<LocalAllocOp>(defOp);
      if (allocOp && loop.isDefinedOutsideOfLoop(allocOp.getSrc()))
        allocOp->moveBefore(loop);
      continue;
    }

    if (auto allocOp = operand.getDefiningOp<LocalAllocOp>()) {
      PartitionBuilder b(allocOp.getLoc(), allocOp);
      StageCluster stageCluster = getStageCluster(allocOp);
      auto store = b.createInto<LocalStoreOp>(*defPartition, stageCluster,
                                              allocOp.getSrc(), allocOp);
      auto fence = b.createInto<ttng::FenceAsyncSharedOp>(
          *defPartition, stageCluster, /*bCluster=*/false);
      operandDefs.emplace_back(body.findAncestorOpInBlock(*store),
                               defPartition);
      operandDefs.emplace_back(body.findAncestorOpInBlock(*fence),
                               defPartition);
      allocOp->moveBefore(loop);
      allocOp->removeAttr(kPartitionAttrName);
      allocOp.getSrcMutable().clear();
      allocOp.getResult().setType(getAsMutable(allocOp.getType()));
      propagateMutability(allocOp.getResult());
    } else if (auto tmemAllocOp = operand.getDefiningOp<ttng::TMEMAllocOp>()) {
      PartitionBuilder b(tmemAllocOp.getLoc(), tmemAllocOp);
      StageCluster stageCluster = getStageCluster(tmemAllocOp);
      auto store = b.createInto<ttng::TMEMStoreOp>(
          *defPartition, stageCluster, Type(), tmemAllocOp.getResult(), Value(),
          tmemAllocOp.getSrc(), b.boolCst(true));
      operandDefs.emplace_back(body.findAncestorOpInBlock(*store),
                               defPartition);
      tmemAllocOp->moveBefore(loop);
      tmemAllocOp->removeAttr(kPartitionAttrName);
      tmemAllocOp.getSrcMutable().clear();
      tmemAllocOp.getResult().setType(getAsMutable(tmemAllocOp.getType()));
    } else if (defOp->hasTrait<OpTrait::MemDescViewTrait>()) {
      incomingOperands.push_back(defOp->getOperand(0));
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

    StageCluster nodeStageCluster = getStageCluster(node.op);
    if (node.barPrev) {
      if (!isa<ttng::TMEMLoadOp>(node.op)) {
        // If the user precondition is defined after the MMA, we need to peel
        // the wait for the user.
        if (incrementPt && domOp->isBeforeInBlock(&*incrementPt->getPoint()) &&
            domInfo.properlyDominates(mmaOp, userPred.getDefiningOp())) {
          b.restoreInsertionPoint(*incrementPt);
          Value bar = createSingleBufferView(b, node.barPrev, curIndex);
          b.createInto<ttng::WaitBarrierOp>(*partition, nodeStageCluster, bar,
                                            curPhase, userPred);
        } else {
          b.setInsertionPoint(domOp);
          Value bar = createSingleBufferView(b, node.barPrev, node.index);
          b.createInto<ttng::WaitBarrierOp>(*partition, nodeStageCluster, bar,
                                            node.phase, userPred);
        }
      } else {
        b.setInsertionPoint(domOp);
        if (isa<scf::IfOp>(domOp->getParentOp()) && accIsMultiBuffered)
          b.setInsertionPointToStart(domOp->getBlock());
        Value bar = createSingleBufferView(b, node.barPrev, node.index);
        b.createInto<ttng::WaitBarrierOp>(*partition, nodeStageCluster, bar,
                                          node.phase);
      }
    }
    if (node.barNext) {
      if (mmaOp == node.op) {
        b.setInsertionPoint(mmaOp);
        Value bar = createSingleBufferView(b, node.barNext, node.index);
        mmaOp.addCompletionBarrier(bar, userPred);
        mmaOp.setIsAsync(true);
      } else {
        b.setInsertionPointAfter(lastOp);
        if (isa<scf::IfOp>(lastOp->getParentOp()) && accIsMultiBuffered)
          b.setInsertionPoint(lastOp->getBlock()->getTerminator());
        Value bar = createSingleBufferView(b, node.barNext, node.index);
        b.createInto<ttng::ArriveBarrierOp>(*partition, nodeStageCluster, bar,
                                            1);
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
    // For Nx1 barrier allocations, pass a 1D view into barrier ops.
    Value emptyView0 = createSingleBufferView(b, emptyBar, b.intCst(0));
    b.create<ttng::ArriveBarrierOp>(emptyView0, /*arriveCount=*/1);

    Operation *domOp = findNearestCommonDominator(defs, domInfo);
    Operation *lastOp = findNearestCommonPostDominator(defs, postDomInfo);

    auto [index, phase] = addIndexAndPhase(b, loop, /*numStages=*/1);
    StageCluster srcStageCluster = getStageCluster(domOp);
    b.setInsertionPoint(domOp);
    Value emptyView = createSingleBufferView(b, emptyBar, index);
    b.createInto<ttng::WaitBarrierOp>(*partition, srcStageCluster, emptyView,
                                      phase);

    b.setInsertionPointAfter(lastOp);
    Value readyView = createSingleBufferView(b, readyBar, index);
    b.createInto<ttng::ArriveBarrierOp>(*partition, srcStageCluster, readyView,
                                        1);

    b.setInsertionPoint(mmaOp);
    Value readyView2 = createSingleBufferView(b, readyBar, index);
    b.createInto<ttng::WaitBarrierOp>(*schedule.getPartition(mmaOp),
                                      getStageCluster(mmaOp), readyView2,
                                      phase);
    Value emptyView2 = createSingleBufferView(b, emptyBar, index);
    mmaOp.addCompletionBarrier(emptyView2, b.boolCst(true));
    mmaOp.setIsAsync(true);
  }

  if (nodes.back().barNext) {
    b.setInsertionPointAfter(loop);
    // Re-acquire loop results as they may have been invalidated.
    Value lastIndex = loop.getResult(index.getArgNumber() - 1);
    Value lastPhase = loop.getResult(phase.getArgNumber() - 1);
    Value lastBar = createSingleBufferView(b, nodes.back().barNext, lastIndex);
    auto waitBarrierOp = b.create<ttng::WaitBarrierOp>(lastBar, lastPhase);
    auto node_front = nodes.front();
    auto partition = schedule.getPartition(inBody(node_front.op));
    PartitionBuilder b(waitBarrierOp->getLoc(), waitBarrierOp);
    lastBar.getDefiningOp()->setAttr(kWarpSpecializeTagAttrName,
                                     b.getI32IntegerAttr(schedule.getTag()));
    waitBarrierOp->setAttr(kWarpSpecializeTagAttrName,
                           b.getI32IntegerAttr(schedule.getTag()));
    b.assignPartition(lastBar.getDefiningOp(), *partition);
    b.assignPartition(waitBarrierOp, *partition);
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

LogicalResult lowerLoops(scf::ForOp &loop, MutableArrayRef<PipelinedLoad> loads,
                         MutableArrayRef<PipelinedMMA> mmas,
                         WarpSchedule &schedule, int numLoadStages) {
  Block &body = *loop.getBody();
  DominanceInfo domInfo(loop);
  PostDominanceInfo postDomInfo(loop);

  // Group loads by common first user operations. This ensures, for example,
  // that multiple loads feeding into the same MMA op are placed together.
  llvm::MapVector<ArrayRef<Operation *>, SmallVector<PipelinedLoad>>
      liveBeforeGroups;
  for (PipelinedLoad &load : loads) {
    if (failed(load.determineLiveRange(body, domInfo, postDomInfo, schedule)))
      return failure();
    liveBeforeGroups[load.liveBeforeOps].push_back(std::move(load));
  }
  SmallVector<PipelinedLoadGroup> loadGroups;
  for (auto &loads : llvm::make_second_range(liveBeforeGroups))
    loadGroups.push_back({std::move(loads)});

  // Multi-buffer and lower the loads.
  for (PipelinedLoadGroup &group : loadGroups)
    group.allocateAref(loop, numLoadStages);

  for (PipelinedLoadGroup &group : loadGroups) {
    if (failed(group.lowerLoads(schedule, domInfo, postDomInfo)))
      return failure();
  }

  // Multi-buffer and lower the MMAs.
  for (PipelinedMMA &mma : mmas) {
    if (failed(pipelineMMA(loop, mma, schedule, domInfo, postDomInfo)))
      return failure();
  }

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
    : public triton::gpu::impl::TritonGPULoadMMASpecializationBase<
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
    FailureOr<WarpSchedule> schedule = WarpSchedule::deserialize(loop);
    if (failed(schedule))
      continue;
    auto [loads, mmas] = getPartitionScheme(loop, *schedule);
    if (loads.empty() && mmas.empty())
      continue;
    int loopNumStages = getNumStagesOrDefault(loop, numStages);
    if (failed(lowerLoops(loop, loads, mmas, *schedule, loopNumStages)))
      continue;
  }
}
