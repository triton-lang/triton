#include "mlir/Analysis/SliceAnalysis.h"
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

  SmallVector<LocalAllocOp, 1> allocOps;
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
    for (LocalAllocOp allocOp : load.allocOps) {
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
    for (LocalAllocOp allocOp : load.allocOps)
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
    // will get assigned just 1 warp each. Add an extra partition to pad the
    // number of warps to the nearest warpgroup.
    schedule.addPartition(0);
    schedule.reorderPartitions({2, 1, 0, 3});
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
    auto it = llvm::find(allocOps, dyn_cast_or_null<LocalAllocOp>(user));
    if (it == allocOps.end()) {
      // This is an in-register use of the load. The result must be live before
      // the op. Since it will be loaded out of shared memory, it only needs to
      // be live until the op as well.
      regSinks[schedule.getPartition(user)].push_back(user);
      continue;
    }
    SmallVector<Operation *> sinkOps;
    if (failed(findSharedMemorySinkOps(it->getResult(), sinkOps)))
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
    b.createInPartition<ttng::AsyncTMACopyGlobalToLocalOp>(
        loadPartition, tmaPtr, load.getIndices(), barrier, view, truePred);
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
  SmallVector<ttng::ArriveBarrierOp> arriveOps;
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
      arriveOps.push_back(
          b.createInPartition<ttng::ArriveBarrierOp>(userPartition, curEmptyBar,
                                                     /*arriveCount=*/1));
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
    for (LocalAllocOp allocOp : load.allocOps) {
      replaceUsesAndPropagateType(b, allocOp, view);
      allocOp->erase();
    }
    // If there are remaining users, they must be in-register.
    if (!load.loadOp->use_empty()) {
      SmallVector<Operation *> regUsers =
          llvm::to_vector(load.loadOp->getUsers());
      llvm::append_range(regUsers, arriveOps);
      Operation *firstRegUser = findNearestCommonDominator(regUsers, domInfo);
      b.setInsertionPoint(firstRegUser);
      Value loaded = b.create<LocalLoadOp>(load.type, view);
      load.loadOp->replaceAllUsesWith(ValueRange{loaded});
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
  auto fail = [&](StringRef msg) {
    return emitWarning(mma.mmaOp.getLoc(), msg);
  };
  Block &body = *loop.getBody();

  // Determine if the MMA accumulator can be multibuffered.
  bool accIsMultiBuffered =
      // MMAs in subsequent iterations can be overlapped.
      !ttng::hasAccReadModifyWrite(mma.mmaOp, loop) &&
      // The accumulator is reset at some point, thus allowing multibuffering.
      ttng::isAccMultibufferingPossible(mma.mmaOp, loop) &&
      // The user didn't disable it with a flag.
      !getDisallowAccMultiBuffer(loop);

  // Check that the accumulator can be multi-buffered.
  ttng::TMEMAllocOp oldAllocOp =
      mma.mmaOp.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!oldAllocOp)
    return fail("accumulator is not a TMEM alloc");
  for (Operation *user : oldAllocOp.getResult().getUsers()) {
    if (!loop->getParentRegion()->isAncestor(user->getParentRegion()))
      return fail("cannot track accumulator uses");
  }

  PartitionBuilder b(mma.mmaOp.getLoc(), oldAllocOp);
  int numMmaStages = 1 + accIsMultiBuffered;
  ttng::TMEMAllocOp allocOp =
      createTMemAlloc(b, oldAllocOp, /*multiBuffered=*/true, numMmaStages);

  // Use placeholder values for the indices in the loop.
  auto indexPhase = addIterArgsToLoop(b, loop, {b.intCst(0), b.intCst(0)});
  BlockArgument index = indexPhase[0];
  BlockArgument phase = indexPhase[1];
  cast<scf::YieldOp>(loop.getBody()->getTerminator())
      .getResultsMutable()
      .append({index, phase});

  // Replace uses of the accumulator before the loop with buffer 0, and replace
  // those after the loop with the last buffer.
  Value firstView = createSingleBufferView(b, allocOp, b.intCst(0));
  b.setInsertionPointAfter(loop);
  Value lastIndex = loop.getResult(index.getArgNumber() - 1);
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
    return body.findAncestorOpInBlock(*lhs)->isBeforeInBlock(
        body.findAncestorOpInBlock(*rhs));
  });

  // Replace uses of the accumulator in the loop.
  b.setInsertionPoint(loop);
  Value replTok = b.create<ub::PoisonOp>(b.getType<AsyncTokenType>());
  b.setInsertionPointToStart(&body);
  Operation *overwriteOp = nullptr, *readOp = nullptr;
  for (Operation *user : usersInLoop) {
    b.setInsertionPoint(user);
    if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(user)) {
      overwriteOp = storeOp;
      storeOp.getDepMutable().clear();
      storeOp.getToken().replaceAllUsesWith(replTok);
      Value view = createSingleBufferView(b, allocOp, index);
      storeOp.getDstMutable().assign(view);
    } else if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user)) {
      if (!matchPattern(mmaOp.useAccumulator(), m_One()))
        overwriteOp = mmaOp;
      mmaOp.getAccDepMutable().clear();
      mmaOp.getToken().replaceAllUsesWith(replTok);
      Value view = createSingleBufferView(b, allocOp, index);
      mmaOp.setAccumulator(view);
    } else if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(user)) {
      readOp = loadOp;
      loadOp.getDepMutable().clear();
      loadOp.getToken().replaceAllUsesWith(replTok);
      Value view = createSingleBufferView(b, allocOp, index);
      loadOp.getSrcMutable().assign(view);
    } else {
      llvm::report_fatal_error("FIXME: unhandled MMA accumulator user");
    }
  }
  oldAllocOp.getToken().replaceAllUsesWith(allocOp.getToken());
  oldAllocOp.erase();

  // Case 1: The MMA result is only read after the loop.
  if (!readOp) {
    b.setInsertionPoint(loop);
    Value doneBar = createBarrierAlloc(loop, /*numBarriers=*/1);
    doneBar = createSingleBufferView(b, doneBar, 0);
    b.setInsertionPoint(mma.mmaOp);
    Value pred = b.create<arith::CmpIOp>(
        arith::CmpIPredicate::eq, loop.getInductionVar(),
        b.create<arith::SubIOp>(loop.getUpperBound(), b.intCst(1)));
    mma.mmaOp.addCompletionBarrier(doneBar, pred);
    b.setInsertionPointAfter(loop);
    b.create<ttng::WaitBarrierOp>(doneBar, b.intCst(0));
    return success();
  }

  // Allocate producer and consumer barriers.
  b.setInsertionPoint(loop);
  Value userPred = getUserPrecondition(b, loop, readOp).first;
  Value emptyBars = createBarrierAlloc(loop, numMmaStages);
  Value readyBars = createBarrierAlloc(loop, numMmaStages);
  for (auto i : llvm::seq(numMmaStages)) {
    Value emptyBar = createSingleBufferView(b, emptyBars, i);
    // Mark all empty barriers as phase 0 complete.
    b.create<ttng::ArriveBarrierOp>(emptyBar, /*arriveCount=*/1);
  }

  bool startReady =
      body.findAncestorOpInBlock(*readOp)->isBeforeInBlock(mma.mmaOp);
  if (startReady) {
    // Mark the first ready barrier as phase 0 complete.
    Value firstReadyBar = createSingleBufferView(b, readyBars, 0);
    b.create<ttng::ArriveBarrierOp>(firstReadyBar, /*arriveCount=*/1);
  }

  Partition &mmaPartition = *schedule.getPartition(mma.mmaOp);
  Partition &readPartition =
      *schedule.getPartition(body.findAncestorOpInBlock(*readOp));

  // Find operands that need to be pipelined through shmem.
  SmallVector<Operation *> operandDefs;
  for (Value operand : mma.mmaOp->getOperands()) {
    Operation *defOp = operand.getDefiningOp();
    if (!defOp || !loop.getBodyRegion().isAncestor(defOp->getParentRegion()))
      continue;
    defOp = body.findAncestorOpInBlock(*defOp);
    if (schedule.getPartition(defOp) != &readPartition)
      continue;
    if (auto allocOp = operand.getDefiningOp<LocalAllocOp>()) {
      PartitionBuilder b(allocOp.getLoc(), allocOp);
      auto store = b.createInPartition<LocalStoreOp>(readPartition,
                                                     allocOp.getSrc(), allocOp);
      operandDefs.push_back(body.findAncestorOpInBlock(*store));
      allocOp->moveBefore(loop);
      allocOp.getSrcMutable().clear();
      allocOp.getResult().setType(getAsMutable(allocOp.getType()));
    } else if (auto tmemAllocOp = operand.getDefiningOp<ttng::TMEMAllocOp>()) {
      PartitionBuilder b(tmemAllocOp.getLoc(), tmemAllocOp);
      auto store = b.createInPartition<ttng::TMEMStoreOp>(
          readPartition, Type(), tmemAllocOp.getResult(), Value(),
          tmemAllocOp.getSrc(), b.boolCst(true));
      operandDefs.push_back(body.findAncestorOpInBlock(*store));
      tmemAllocOp->moveBefore(loop);
      tmemAllocOp.getSrcMutable().clear();
      tmemAllocOp.getResult().setType(getAsMutable(tmemAllocOp.getType()));
    }
  }

  // Producer commit.
  b.setInsertionPoint(mma.mmaOp);
  Value readyBar = createSingleBufferView(b, readyBars, index);
  mma.mmaOp.addCompletionBarrier(readyBar, userPred);

  // Place the consumer wait before the read.
  operandDefs.push_back(readOp);
  Operation *consumerWait = findNearestCommonDominator(operandDefs, domInfo);
  b.setInsertionPoint(consumerWait);
  // If the consumer is inside a conditional, have it acquire the buffer for the
  // whole of the conditional block to improve instruction scheduling.
  if (isa<scf::IfOp>(consumerWait->getParentOp()))
    b.setInsertionPointToStart(consumerWait->getBlock());
  readyBar = createSingleBufferView(b, readyBars, index);
  b.createInPartition<ttng::WaitBarrierOp>(readPartition, readyBar, phase);

  // Place the consumer release after the read, and after the write as well if
  // it is in the user partition.
  if (overwriteOp != mma.mmaOp && !mma.storeOp)
    operandDefs.push_back(overwriteOp);
  Operation *consumerRelease =
      findNearestCommonPostDominator(operandDefs, postDomInfo);
  b.setInsertionPointAfter(consumerRelease);
  if (isa<scf::IfOp>(consumerRelease->getParentOp()))
    b.setInsertionPoint(consumerRelease->getBlock()->getTerminator());
  Value emptyBar = createSingleBufferView(b, emptyBars, index);
  auto releaseOp = b.createInPartition<ttng::ArriveBarrierOp>(
      readPartition, emptyBar, /*arriveCount=*/1);

  // Always place the producer acquire after the consumer release.
  b.setInsertionPointAfter(body.findAncestorOpInBlock(*releaseOp));
  emptyBar = createSingleBufferView(b, emptyBars, index);
  b.createInPartition<ttng::WaitBarrierOp>(mmaPartition, emptyBar, phase,
                                           userPred);

  // Increment after the read, but also after the mbarrier arrive if it is after
  // the read.
  Operation *afterRead = readOp;
  if (readOp->getNextNode() == releaseOp)
    afterRead = releaseOp;
  afterRead = body.findAncestorOpInBlock(*afterRead);
  b.setInsertionPointAfter(afterRead);
  auto [nextIndex, nextPhase] =
      postIncrementModulo(b, index, phase, numMmaStages);
  nextIndex = b.create<arith::SelectOp>(userPred, nextIndex, index);
  nextPhase = b.create<arith::SelectOp>(userPred, nextPhase, phase);
  replaceAllUsesDominatedBy(nextIndex.getDefiningOp(), nextIndex, index,
                            domInfo);
  replaceAllUsesDominatedBy(nextPhase.getDefiningOp(), nextPhase, phase,
                            domInfo);

  llvm::SetVector<Operation *> predOps;
  Operation *hoistPt =
      findNearestCommonDominator(llvm::to_vector(userPred.getUsers()), domInfo);
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
