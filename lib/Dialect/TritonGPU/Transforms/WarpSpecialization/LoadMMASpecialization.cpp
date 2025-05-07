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
// assignPartitions
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
  SetVector<Operation *> userOps;
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

// Assign load and MMAs to partitions and figure out where the user partition
// is.
static PartitionScheme assignPartitions(scf::ForOp loop) {
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
      if (!op->hasOneUse() || !isa<MemDescSubviewOp, MemDescTransOp>(op))
        continue;
      mma.operandViews.push_back(op);
      if (Operation *defOp = op->getOperand(0).getDefiningOp())
        operandViews.push_back(defOp);
    }
  }

  // Assign initial partitions.
  Builder b(loop.getContext());
  SmallVector<Operation *> transitiveUsers;

  DenseSet<Operation *> scheduled;
  for (PipelinedLoad &load : loads) {
    for (Operation *allocOp : load.allocOps) {
      scheduled.insert(allocOp);
      transitiveUsers.push_back(allocOp);
    }
    scheduled.insert(load.loadOp);
    transitiveUsers.push_back(load.loadOp);
  }

  for (PipelinedMMA &mma : mmas) {
    scheduled.insert(mma.mmaOp);
    if (mma.storeOp)
      scheduled.insert(mma.storeOp);
    for (Operation *view : mma.operandViews)
      scheduled.insert(view);
    transitiveUsers.push_back(mma.mmaOp);
  }

  // Recursively propagate partitions to the users.
  SetVector<Operation *> userOps;
  while (!transitiveUsers.empty()) {
    Operation *op = transitiveUsers.pop_back_val();

    SmallVector<OpOperand *> uses;
    for (OpOperand &use : op->getUses())
      uses.push_back(&use);
    for (unsigned i = 0; i < uses.size(); ++i) {
      OpOperand *use = uses[i];
      Operation *user = use->getOwner();
      if (user == loop.getBody()->getTerminator()) {
        for (OpOperand &use :
             loop.getRegionIterArg(use->getOperandNumber()).getUses())
          uses.push_back(&use);
      } else {
        if (!scheduled.insert(user).second)
          continue;
        user = loop.getBody()->findAncestorOpInBlock(*user);
        userOps.insert(user);
        transitiveUsers.push_back(user);
      }
    }
  }

  return PartitionScheme{std::move(loads), std::move(mmas), std::move(userOps)};
}

static WarpSchedule getInitialSchedule(const PartitionScheme &scheme) {
  WarpSchedule schedule;

  Partition *loadPartition = schedule.addPartition(0);
  for (const PipelinedLoad &load : scheme.loads) {
    loadPartition->insert(load.loadOp);
    for (Operation *allocOp : load.allocOps)
      loadPartition->insert(allocOp);
  }

  Partition *mmaPartition = schedule.addPartition(1);
  for (const PipelinedMMA &mma : scheme.mmas) {
    mmaPartition->insert(mma.mmaOp);
    if (mma.storeOp)
      mmaPartition->insert(mma.storeOp);
    for (Operation *viewOp : mma.operandViews)
      mmaPartition->insert(viewOp);
  }

  if (!scheme.userOps.empty()) {
    Partition *userPartition = schedule.addPartition(2);
    for (Operation *userOp : scheme.userOps)
      userPartition->insert(userOp);
    // Place the epilogue partition in the default warpgroup. The MMA and load
    // partitions shouldn't have tensor computations in them, which means they
    // will get assigned just 1 warp each.
    schedule.reorderPartitions({2, 1, 0});
  }

  schedule.updatePartitions();
  return schedule;
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

  template <typename OpT, typename... Args>
  auto createInPartition(Partition &partition, Args &&...args) {
    auto op = create<OpT>(std::forward<Args>(args)...);
    op->setAttr(kPartitionAttrName, getI32IntegerAttr(partition.getIndex()));
    partition.insert(op);
    return op;
  }
};

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
    } else if (isa<MemDescTransOp, MemDescSubviewOp>(user)) {
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
                         Operation *op, Value barrier, Value view) {
  Value truePred = b.create<arith::ConstantIntOp>(true, /*width=*/1);
  if (auto load = dyn_cast<DescriptorLoadOp>(op)) {
    Value tmaPtr = b.createInPartition<ttng::TensorDescToTMAPtrOp>(
        loadPartition, load.getDesc());
    auto indices = ttng::translateTMAIndices(
        b, load.getLoc(), load.getDesc().getType().getBlockType().getEncoding(),
        load.getIndices());
    b.createInPartition<ttng::AsyncTMACopyGlobalToLocalOp>(
        loadPartition, tmaPtr, indices, barrier, view, truePred);
  } else {
    auto gather = cast<DescriptorGatherOp>(op);
    Value tmaPtr = b.createInPartition<ttng::TensorDescToTMAPtrOp>(
        loadPartition, gather.getDesc());
    b.createInPartition<ttng::AsyncTMAGatherOp>(
        loadPartition, tmaPtr, gather.getXOffsets(), gather.getYOffset(),
        barrier, view, truePred);
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

  // Producer acquire.
  Value curEmptyBar = createSingleBufferView(b, emptyBars, index);
  b.createInPartition<ttng::WaitBarrierOp>(loadPartition, curEmptyBar, phase);

  // Indicate the expected size of the loads.
  unsigned loadSizeInBytes = 0;
  for (const PipelinedLoad &load : loads)
    loadSizeInBytes += load.getLoadSizeInBytes();
  Value curLoadBar = createSingleBufferView(b, readyBars, index);
  b.createInPartition<ttng::BarrierExpectOp>(loadPartition, curLoadBar,
                                             loadSizeInBytes, b.boolCst(true));

  // Set up the consumer wait. We know the live before ops are the same for all
  // loads since that's how they were grouped.
  SetVector<Operation *> distinctAsyncUsers;
  DenseMap<Partition *, ttng::ArriveBarrierOp> arriveOps;
  for (auto [i, liveBeforeOp] : llvm::enumerate(firstLoad->liveBeforeOps)) {
    b.setInsertionPoint(liveBeforeOp);
    Partition &userPartition = *schedule.getPartition(liveBeforeOp);
    b.createInPartition<ttng::WaitBarrierOp>(userPartition, curLoadBar, phase);

    SmallVector<Operation *> liveUntilOps;
    for (PipelinedLoad &load : loads) {
      if (Operation *liveUntilOp = load.liveUntilOps[i])
        liveUntilOps.push_back(liveUntilOp);
    }
    if (!liveUntilOps.empty()) {
      Operation *liveUntilOp =
          findNearestCommonPostDominator(liveUntilOps, postDomInfo);
      b.setInsertionPoint(liveUntilOp);
      arriveOps[schedule.getPartition(liveUntilOp)] =
          b.createInPartition<ttng::ArriveBarrierOp>(userPartition, curEmptyBar,
                                                     /*arriveCount=*/1);
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
    lowerTMACopy(b, loadPartition, load.loadOp, curLoadBar, view);
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
      Value loaded =
          b.createInPartition<LocalLoadOp>(*partition, load.type, view);
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
    userPred = b.create<arith::CmpIOp>(
        arith::CmpIPredicate::eq, loop.getInductionVar(),
        b.create<arith::SubIOp>(loop.getUpperBound(), b.intCst(1)));
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
      auto store = b.createInPartition<LocalStoreOp>(*defPartition,
                                                     allocOp.getSrc(), allocOp);
      operandDefs.emplace_back(body.findAncestorOpInBlock(*store),
                               defPartition);
      allocOp->moveBefore(loop);
      allocOp.getSrcMutable().clear();
      allocOp.getResult().setType(getAsMutable(allocOp.getType()));
    } else if (auto tmemAllocOp = operand.getDefiningOp<ttng::TMEMAllocOp>()) {
      PartitionBuilder b(tmemAllocOp.getLoc(), tmemAllocOp);
      auto store = b.createInPartition<ttng::TMEMStoreOp>(
          *defPartition, Type(), tmemAllocOp.getResult(), Value(),
          tmemAllocOp.getSrc(), b.boolCst(true));
      operandDefs.emplace_back(body.findAncestorOpInBlock(*store),
                               defPartition);
      tmemAllocOp->moveBefore(loop);
      tmemAllocOp.getSrcMutable().clear();
      tmemAllocOp.getResult().setType(getAsMutable(tmemAllocOp.getType()));
    }
  }

  for (Node &node : nodes) {
    Partition *partition = schedule.getPartition(inBody(node.op));
    PartitionBuilder b(node.op->getLoc(), loop);

    SmallVector<Operation *> defs;
    defs.push_back(node.op);
    for (auto &[defOp, defPartition] : operandDefs) {
      if (defPartition == partition)
        defs.push_back(defOp);
    }
    Operation *domOp = findNearestCommonDominator(defs, domInfo);
    Operation *lastOp = findNearestCommonPostDominator(defs, postDomInfo);

    if (node.barPrev) {
      if (!isa<ttng::TMEMLoadOp>(node.op)) {
        if (incrementPt && domOp->isBeforeInBlock(&*incrementPt->getPoint()))
          b.restoreInsertionPoint(*incrementPt);
        else
          b.setInsertionPoint(domOp);
        Value bar = createSingleBufferView(b, node.barPrev, curIndex);
        b.createInPartition<ttng::WaitBarrierOp>(*partition, bar, curPhase,
                                                 userPred);
      } else {
        b.setInsertionPoint(domOp);
        if (isa<scf::IfOp>(domOp->getParentOp()))
          b.setInsertionPointToStart(domOp->getBlock());
        Value bar = createSingleBufferView(b, node.barPrev, node.index);
        b.createInPartition<ttng::WaitBarrierOp>(*partition, bar, node.phase);
      }
    }
    if (node.barNext) {
      if (mmaOp == node.op) {
        b.setInsertionPoint(mmaOp);
        Value bar = createSingleBufferView(b, node.barNext, node.index);
        mmaOp.addCompletionBarrier(bar, userPred);
      } else {
        b.setInsertionPointAfter(lastOp);
        if (isa<scf::IfOp>(lastOp->getParentOp()))
          b.setInsertionPoint(lastOp->getBlock()->getTerminator());
        Value bar = createSingleBufferView(b, node.barNext, node.index);
        b.createInPartition<ttng::ArriveBarrierOp>(*partition, bar, 1);
      }
    }
  }

  if (nodes.back().barNext) {
    b.setInsertionPointAfter(loop);
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

  // Multi-buffer and lower the loads.
  for (PipelinedLoadGroup &group : loadGroups)
    group.allocateAref(loop, numLoadStages);
  for (PipelinedLoadGroup &group : loadGroups) {
    if (failed(group.lowerLoads(schedule, domInfo, postDomInfo)))
      return failure();
  }

  // Multi-buffer and lower the MMAs.
  for (PipelinedMMA &mma : scheme.mmas) {
    if (failed(pipelineMMA(loop, mma, schedule, domInfo, postDomInfo)))
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
    PartitionScheme scheme = assignPartitions(loop);
    if (scheme.loads.empty() && scheme.mmas.empty())
      continue;
    WarpSchedule schedule = getInitialSchedule(scheme);
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
