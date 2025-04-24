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

  SmallVector<LocalAllocOp> allocOps;
  Operation *liveBeforeOp = nullptr;
  Operation *liveUntilOp = nullptr;
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
    if (loadOp.use_empty())
      continue;

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
    mmas.push_back(std::move(mma));
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
  }

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

  // Create an operation inside a partition.
  template <typename OpT, typename... Args>
  auto createInPartition(Partition &partition, Args &&...args) {
    auto op = create<OpT>(std::forward<Args>(args)...);
    partition.insert(op);
    return op;
  }
};

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
  SmallVector<Operation *> regSinks, shmemSinks;
  for (Operation *user : loadOp->getUsers()) {
    auto it = llvm::find(allocOps, dyn_cast_or_null<LocalAllocOp>(user));
    if (it == allocOps.end()) {
      // This is an in-register use of the load. The result must be live before
      // the op. Since it will be loaded out of shared memory, it only needs to
      // be live until the op as well.
      regSinks.push_back(user);
      continue;
    }
    if (failed(findSharedMemorySinkOps(it->getResult(), shmemSinks)))
      return failure();
  }

  // The result must be live before all the sinks.
  auto liveBeforeOps =
      llvm::to_vector(llvm::concat<Operation *>(regSinks, shmemSinks));
  liveBeforeOp = findNearestCommonDominator(liveBeforeOps, domInfo);
  assert(liveBeforeOp && "expected a common dominator");
  liveBeforeOp = container.findAncestorOpInBlock(*liveBeforeOp);

  // The result must be live until the earliest register sink but after all the
  // shared memory sinks.
  Operation *liveUntilReg = findNearestCommonDominator(regSinks, domInfo);
  assert(liveUntilReg && "expected a common dominator");
  Operation *liveUntilMem =
      findNearestCommonPostDominator(shmemSinks, postDomInfo);
  assert(liveUntilMem && "expected a common post-dominator");

  liveUntilReg = container.findAncestorOpInBlock(*liveUntilReg);
  liveUntilMem = container.findAncestorOpInBlock(*liveUntilMem);
  liveUntilMem = liveUntilMem->getNextNode();

  liveUntilOp =
      findNearestCommonPostDominator({liveUntilReg, liveUntilMem}, postDomInfo);
  assert(liveUntilOp && "expected a common post-dominator");

  // Require that all the consumers are in the same partition.
  auto userPartitions = llvm::map_to_vector(liveBeforeOps, [&](Operation *op) {
    return schedule.getPartition(container.findAncestorOpInBlock(*op));
  });
  if (!llvm::all_equal(userPartitions)) {
    return mlir::emitWarning(loadOp->getLoc(),
                             "failed to warp specialize: multiple load "
                             "consumer partitions not supported");
  }
}

namespace {

struct PipelinedLoadGroup {
  Location getLoc();
  void allocateAref(scf::ForOp loop, int numStages);
  LogicalResult lowerLoads(WarpSchedule &schedule, DominanceInfo &domInfo,
                           PostDominanceInfo &postDomInfo);

  SmallVector<PipelinedLoad> loads;
  Operation *liveBeforeOp;

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

void PipelinedLoadGroup::allocateAref(scf::ForOp loop, int numStages) {
  assert(loadBuffers.empty() && "already allocated");

  for (PipelinedLoad &load : loads) {
    loadBuffers.push_back(createAlloc(loop, load.type, load.loadOp->getLoc(),
                                      load.sharedEnc, numStages));
  }

  // Share the same set of barriers all loads in the group.
  emptyBars = createBarrierAlloc(loop, numStages);
  readyBars = createBarrierAlloc(loop, numStages);
  // All buffers are initially in the empty state.
  PartitionBuilder b(getLoc(), loop);
  for (auto i : llvm::seq(numStages)) {
    Value emptyBar = createSingleBufferView(b, emptyBars, i);
    b.create<ttng::ArriveBarrierOp>(emptyBar, 1);
  }

  std::tie(index, phase) = addIndexAndPhase(b, loop, numStages);
}

LogicalResult PipelinedLoadGroup::lowerLoads(WarpSchedule &schedule,
                                             DominanceInfo &domInfo,
                                             PostDominanceInfo &postDomInfo) {
  // Insert before the group of loads.
  auto firstLoadIt = llvm::min_element(loads, [&](auto &lhs, auto &rhs) {
    return domInfo.properlyDominates(lhs.loadOp, rhs.loadOp);
  });
  Operation *firstLoad = firstLoadIt->loadOp;
  Partition &loadPartition = *schedule.getPartition(firstLoad);
  Partition &userPartition = *schedule.getPartition(liveBeforeOp);
  PartitionBuilder b(getLoc(), firstLoad);

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
  OpBuilder::InsertPoint asyncPt = b.saveInsertionPoint();

  // Set up the consumer wait.
  b.setInsertionPoint(liveBeforeOp);
  Operation *liveAfter = b.createInPartition<ttng::WaitBarrierOp>(
      userPartition, curLoadBar, phase);

  // Set up the consumer release.
  SmallVector<Operation *> liveUntilOps = llvm::map_to_vector(
      loads, [](PipelinedLoad &load) { return load.liveUntilOp; });
  Operation *liveUntilOp =
      findNearestCommonPostDominator(liveUntilOps, postDomInfo);
  assert(liveUntilOp && "expected a common post-dominator");
}

// Pattern match a simple `tma_load -> ... -> tl.dot` single-user chain. This
// ensures there are no extraneous users of the load or intermediate values and
// that a valid partition schedule can be formed.
static LogicalResult findSingleChainToLoad(scf::ForOp loop, Value value,
                                           SmallVectorImpl<Operation *> &ops) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || !value.hasOneUse() || defOp->getParentOp() != loop)
    return failure();

  // This only works on TMA loads because they directly use the mbarrier
  // mechanism. Since async groups are per-thread, commit groups cannot be used
  // to synchronize across warp groups. We have to wait on the async group in
  // the same partition as the loads and arrive an mbarrier to synchronize with
  // the MMA partition, and then software pipeline the load partition.
  //
  // Triple-buffered example:
  //
  //   cp.async %a_ptrs[0], %a_buf[0]
  //   cp.async %b_ptrs[0], %b_buf[0]
  //   cp.async.commit_group
  //
  //   cp.async %a_ptrs[1], %a_buf[1]
  //   cp.async %b_ptrs[1], %b_buf[1]
  //   cp.async.commit_group
  //
  //   for i in range(2, N+2):
  //     @i<N mbarrier.wait %empty_mbars[i%3]
  //     @i<N cp.async %a_ptrs[i], %a_buf[i%3]
  //     @i<N cp.async %b_ptrs[i], %b_buf[i%3]
  //     @i<N cp.async.commit_group
  //
  //     cp.async.wait_group 2 # the i-2 load group is complete
  //     mbarrier.arrive %load_mbars[(i-2)%3]
  if (isa<DescriptorLoadOp, DescriptorGatherOp>(defOp)) {
    ops.push_back(defOp);
    return success();
  }

  // See through allocations and layout conversions.
  if (isa<LocalAllocOp, MemDescTransOp>(defOp)) {
    assert(llvm::is_contained({0, 1}, defOp->getNumOperands()));
    // Alloc ops have an optional source operand.
    if (defOp->getNumOperands() != 1)
      return failure();
    ops.push_back(defOp);
    return findSingleChainToLoad(loop, defOp->getOperand(0), ops);
  }

  return failure();
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

namespace {
struct PipelineableLoad {
  PipelineableLoad(OpOperand &mmaUse, SmallVector<Operation *> useChain)
      : mmaUse(&mmaUse), loadOp(useChain.pop_back_val()),
        allocOp(cast<LocalAllocOp>(useChain.pop_back_val())),
        viewOps(std::move(useChain)),
        type(cast<RankedTensorType>(loadOp->getResult(0).getType())),
        sharedEnc(getSharedEncoding(loadOp)) {}

  Value allocate(scf::ForOp loop, unsigned numStages) const {
    return createAlloc(loop, type, loadOp->getLoc(), sharedEnc, numStages);
  }
  unsigned getLoadSizeInBytes() const {
    return type.getNumElements() * type.getElementTypeBitWidth() / 8;
  }
  void lowerLoadAndPropagate(PartitionBuilder &b, Value alloc, Value barrier,
                             Value loadIndex, Partition &loadPartition,
                             scf::ForOp loop,
                             SmallVectorImpl<Operation *> &loadUsers) && {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(loadOp);
    Value view = createSingleBufferView(b, alloc, loadIndex);
    lowerTMACopy(b, loadPartition, loadOp, barrier, view);
    replaceUsesAndPropagateType(b, allocOp, view);
    allocOp->erase();
    loadOp->erase();
    (void)findSingleChainToLoad(loop, mmaUse->get(), loadUsers);
  }

  OpOperand *mmaUse;
  Operation *loadOp;
  LocalAllocOp allocOp;
  SmallVector<Operation *> viewOps;
  RankedTensorType type;
  SharedEncodingTrait sharedEnc;
};
} // namespace

static std::optional<PipelineableLoad>
findPipelineableLoad(scf::ForOp loop, OpOperand &operand) {
  SmallVector<Operation *> ops;
  if (failed(findSingleChainToLoad(loop, operand.get(), ops)))
    return std::nullopt;
  return PipelineableLoad(operand, std::move(ops));
}

//===----------------------------------------------------------------------===//
// specializeLoadMMADependencies
//===----------------------------------------------------------------------===//

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

namespace {
struct LoadGroup {
  LoadGroup(ttng::MMAv5OpInterface mmaOp) : mmaOp(mmaOp) {}

  void lowerLoads(DominanceInfo &domInfo, PartitionBuilder &b, scf::ForOp &loop,
                  int numStages, Partition &loadPartition,
                  Partition &mmaPartition);

  ttng::MMAv5OpInterface mmaOp;
  SmallVector<PipelineableLoad> loads;
  SmallVector<Value> loadBuffers;
  Value emptyBars;
  Value readyBars;
  BlockArgument index;
  BlockArgument phase;
};
} // namespace

void LoadGroup::lowerLoads(DominanceInfo &domInfo, PartitionBuilder &b,
                           scf::ForOp &loop, int numStages,
                           Partition &loadPartition, Partition &mmaPartition) {
  for (const PipelineableLoad &load : loads)
    loadBuffers.push_back(load.allocate(loop, numStages));

  // Share the same set of barriers all loads in the group.
  emptyBars = createBarrierAlloc(loop, numStages);
  readyBars = createBarrierAlloc(loop, numStages);

  // Mark the empty barriers as initially ready.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);
  for (auto i : llvm::seq(numStages)) {
    Value emptyBar = createSingleBufferView(b, emptyBars, i);
    b.create<ttng::ArriveBarrierOp>(emptyBar, 1);
  }

  unsigned loadSizeInBytes = 0;
  for (const PipelineableLoad &load : loads)
    loadSizeInBytes += load.getLoadSizeInBytes();

  // Insert before the group of loads.
  std::sort(loads.begin(), loads.end(), [](auto &lhs, auto &rhs) {
    return lhs.loadOp->isBeforeInBlock(rhs.loadOp);
  });
  b.setInsertionPoint(loads.front().loadOp);

  // Multi-buffer the loads.
  std::tie(index, phase) = addIndexAndPhase(b, loop, numStages);

  // Wait for the buffer to be empty and the corresponding barrier to be
  // exhausted.
  Value curEmptyBar = createSingleBufferView(b, emptyBars, index);
  b.createInPartition<ttng::WaitBarrierOp>(loadPartition, curEmptyBar, phase);
  // Indicate the expected size of the loads.
  Value curLoadBar = createSingleBufferView(b, readyBars, index);
  b.createInPartition<ttng::BarrierExpectOp>(
      loadPartition, curLoadBar, loadSizeInBytes, b.intCst(true, 1));

  // Replace the loads with async copies and place the remaining users in the
  // MMA partition. Re-acquire the use chain because some ops were invalidated
  // by `replaceUsesAndPropagateType`.
  SmallVector<Operation *> loadUsers{mmaOp};
  for (auto [load, alloc] : llvm::zip(loads, loadBuffers)) {
    std::move(load).lowerLoadAndPropagate(b, alloc, curLoadBar, index,
                                          loadPartition, loop, loadUsers);
  }

  // Place users in the MMA partition.
  for (Operation *user : loadUsers)
    mmaPartition.insert(user);

  // Insert the load wait before the first user.
  Operation *minOp = findNearestCommonDominator(loadUsers, domInfo);
  b.setInsertionPoint(minOp);
  b.createInPartition<ttng::WaitBarrierOp>(mmaPartition, curLoadBar, phase);

  // Add a completion on the MMA to signal the load empty barrier.
  mmaOp.addCompletionBarrier(curEmptyBar, b.boolCst(true));
}

LogicalResult lowerLoops(scf::ForOp &loop, PartitionScheme &scheme,
                         WarpSchedule &schedule, int numLoadStages) {
  Block &body = *loop.getBody();
  DominanceInfo domInfo(loop);
  PostDominanceInfo postDomInfo(loop);

  // Group loads by common first user operations. This ensures, for example,
  // that multiple loads feeding into the same MMA op are placed together.
  llvm::MapVector<Operation *, SmallVector<PipelinedLoad>> liveBeforeGroups;
  for (PipelinedLoad &load : scheme.loads) {
    if (failed(load.determineLiveRange(body, domInfo, postDomInfo, schedule)))
      return failure();
    liveBeforeGroups[load.liveBeforeOp].push_back(std::move(load));
  }
  SmallVector<PipelinedLoadGroup> loadGroups;
  for (auto &[liveBeforeOp, loads] : liveBeforeGroups)
    loadGroups.push_back({std::move(loads), liveBeforeOp});

  // Multi-buffer and lower the loads.
  for (PipelinedLoadGroup &group : loadGroups) {
    group.allocateAref(loop, numLoadStages);
  }

  return success();
}

LogicalResult triton::gpu::specializeLoadMMADependencies(scf::ForOp &loop,
                                                         int defaultNumStages) {
  auto ops = llvm::to_vector(loop.getOps<ttng::MMAv5OpInterface>());
  if (ops.empty())
    return success();
  // Support only 1 MMA op.
  // if (ops.size() > 1) {
  //  return mlir::emitWarning(
  //      loop.getLoc(),
  //      "failed to warp specialize: more than one `tt.dot` found in the
  //      loop");
  //}
  ttng::MMAv5OpInterface mmaOp = ops.front();

  // Look for the loads that feed into the MMA operands.
  SmallVector<LoadGroup> loadGroups;
  DenseSet<Operation *> allLoadOps;
  for (ttng::MMAv5OpInterface mmaOp : ops) {
    LoadGroup group(mmaOp);
    for (OpOperand &operand : mmaOp->getOpOperands()) {
      if (std::optional<PipelineableLoad> load =
              findPipelineableLoad(loop, operand)) {
        group.loads.push_back(std::move(*load));
        allLoadOps.insert(group.loads.back().loadOp);
      }
    }
    if (!group.loads.empty())
      loadGroups.push_back(std::move(group));
  }

  ttng::TMEMAllocOp oldAccAlloc =
      mmaOp.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!oldAccAlloc)
    return mlir::emitWarning(mmaOp.getLoc(), "accumulator is not a TMEM alloc");
  auto accUsersInLoop = llvm::to_vector(
      llvm::make_filter_range(oldAccAlloc->getUsers(), [&](Operation *user) {
        return loop.getBody()->findAncestorOpInBlock(*user);
      }));

  // Determine if the MMA accumulator can be multibuffered.
  auto isLoadPipelineable = [&](Operation *op) {
<<<<<<< HEAD
    return allLoadOps.contains(op);
=======
    return llvm::is_contained({aChain.back(), bChain.back()}, op);
>>>>>>> parent of 7e608e3b0 ([Bench][Blackwell] Support optional scale TMAs in warp specialization for tl.dot_scaled)
  };
  bool accIsMultiBuffered =
      // All operand feeds are pipelineable.
      ttng::mmaHasPipelineableOperands(mmaOp, loop, isLoadPipelineable) &&
      // MMAs in subsequent iterations can be overlapped.
      !ttng::hasAccReadModifyWrite(mmaOp, loop) &&
      // The accumulator is reset at some point, thus allowing multibuffering.
      ttng::isAccMultibufferingPossible(mmaOp, loop) &&
      // The user didn't disable it with a flag.
      !getDisallowAccMultiBuffer(loop);

  // Uses of the accumulator inside the loop must occur after the MMA op as they
  // will be placed in a user partition.
  // TODO: We can support uses prior to the MMA op by rotating the user loop.
  DominanceInfo domInfo(loop);
  for (Operation *user : accUsersInLoop) {
    if (domInfo.dominates(mmaOp, user))
      continue;
    return mlir::emitWarning(loop.getLoc(),
                             "failed to warp specialize: accumulator user does "
                             "not occur after the `tt.dot`");
  }

  PartitionBuilder b(mmaOp.getLoc(), loop);

  // Collect a condition that fires whenever the accumulator value is reset in
  // the loop.
  Value overridePred;
  for (Operation *user : accUsersInLoop) {
    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user)) {
      Value flag = mmaOp.useAccumulator();
      if (matchPattern(flag, m_Zero())) {
        overridePred = b.boolCst(true);
      } else if (!matchPattern(flag, m_One())) {
        if (auto arg = dyn_cast<BlockArgument>(flag)) {
          auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
          overridePred = yield.getOperand(arg.getArgNumber() - 1);
          b.setInsertionPoint(yield);
          overridePred = b.create<arith::XOrIOp>(overridePred, b.boolCst(true));
        } else {
          return mlir::emitWarning(flag.getLoc(), "acc use flag is not an arg");
        }
      }
    } else if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(user)) {
      if (!matchPattern(storeOp.getPred(), m_Zero()))
        overridePred = storeOp.getPred();
    } else if (!isa<ttng::TMEMLoadOp>(user)) {
      return mlir::emitWarning(user->getLoc(), "unexpected accumulator user");
    }
  }

  // Pattern match succeeded. Now rewrite the loads and MMA ops to pass tensor
  // values through buffers.
  int numStages = getNumStagesOrDefault(loop, defaultNumStages);
  int numMmaStages = 1 + accIsMultiBuffered;
  WarpSchedule schedule;
  Partition *loadPartition = schedule.addPartition(0);
  Partition *mmaPartition = schedule.addPartition(numStages);

  Partition *userPartition = nullptr;
  if (loadGroups.size() > 1)
    userPartition = schedule.addPartition(numStages + numMmaStages);
  for (auto [i, group] : llvm::enumerate(loadGroups)) {
    group.lowerLoads(domInfo, b, loop, numStages, *loadPartition,
                     *(i == 0 ? mmaPartition : userPartition));
  }

  // Now rewrite the MMA by multi-buffering the accumulator if necessary.
  // However, the TMEM multi-buffering may be with respect to the outer loop.
  b.setInsertionPointAfter(mmaOp);
  OpBuilder::InsertPoint donePt = b.saveInsertionPoint();

  // Now handle the accumulator, which is the tricky bit. The accumulator value
  // may be conditionally reset in the MMA partition before the MMA op, and it
  // may be conditionally used in a user partition.
  b.setInsertionPoint(oldAccAlloc);
  ttng::TMEMAllocOp accAlloc =
      createTMemAlloc(b, oldAccAlloc, /*multiBuffered=*/true, numMmaStages);

  // If the accumulator is multibuffered, the buffer changes when the
  // accumulator is reset.
  auto [accIndex, accPhase] =
      addIndexAndPhase(b, loop, numMmaStages, overridePred);

  // Replace uses of the original accumulator with the right subview before,
  // inside, and after the loop.
  SmallVector<Operation *> loadsInLoop;
  b.setInsertionPoint(loop);
  Value replTok = b.create<ub::PoisonOp>(b.getType<AsyncTokenType>());
  for (OpOperand &use :
       llvm::make_early_inc_range(oldAccAlloc.getResult().getUses())) {
    Operation *user = use.getOwner();
    b.setInsertionPoint(user);
    Value bufIdx;
    if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
      if (loop->isAncestor(store)) {
        store.getDepMutable().clear();
        store.getToken().replaceAllUsesWith(replTok);
        mmaPartition->insert(store);
        bufIdx = b.create<arith::AddIOp>(accIndex, b.intCst(1));
        bufIdx = b.create<arith::RemUIOp>(bufIdx, b.intCst(numMmaStages));
      } else {
        if (!store->isBeforeInBlock(loop))
          return mlir::emitWarning(store.getLoc(), "store not before loop?");
        bufIdx = b.intCst(0);
      }
    } else if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (loop->isAncestor(load)) {
        load.getDepMutable().clear();
        load.getToken().replaceAllUsesWith(replTok);
        loadsInLoop.push_back(load);
        bufIdx = accIndex;
      } else {
        if (!loop->isBeforeInBlock(load))
          return mlir::emitWarning(load.getLoc(), "load not after loop?");
        bufIdx = loop.getResult(accIndex.getArgNumber() - 1);
      }
    } else if (user == mmaOp) {
      mmaOp.getAccDepMutable().clear();
      mmaOp.getToken().replaceAllUsesWith(replTok);
      bufIdx = accIndex;
    } else {
      return mlir::emitWarning(user->getLoc(), "unknown acc user");
    }
    Value buf = createSingleBufferView(b, accAlloc, bufIdx);
    use.set(buf);
  }
  oldAccAlloc.getToken().replaceAllUsesWith(accAlloc.getToken());
  oldAccAlloc->erase();

  // Replace uses of the accumulator inside the loop with a value loaded from
  // the buffer. Place these in a new user partition.
  if (!loadsInLoop.empty()) {
    Value accEmptyBars = createBarrierAlloc(loop, numMmaStages);
    Value accReadyBars = createBarrierAlloc(loop, numMmaStages);
    b.setInsertionPoint(loop);
    // Because the accumulator reset occurs after the MMA op, we have to place
    // the wait on the empty barrier after the MMA op as well. This is OK since
    // we know all buffers are empty upon entry to the loop. However, this means
    // the last mbarrier is guarding the first buffer. Thus, initialize all but
    // the last mbarrier.
    for (auto i : llvm::drop_end(llvm::seq(numMmaStages))) {
      Value emptyBar = createSingleBufferView(b, accEmptyBars, i);
      b.create<ttng::ArriveBarrierOp>(emptyBar, 1);
    }
    b.setInsertionPointToStart(loop.getBody());
    Value curAccEmptyBar = createSingleBufferView(b, accEmptyBars, accIndex);
    Value curAccReadyBar = createSingleBufferView(b, accReadyBars, accIndex);

    Operation *domOp = findNearestCommonDominator(loadsInLoop, domInfo);
    assert(domOp && "could not find common dominator for accumulator uses");
    Value pred;
    b.restoreInsertionPoint(donePt);
    std::tie(pred, domOp) = getUserPrecondition(b, loop, domOp);

    // We have to hoist the predicate above the MMA op to add the barrier.
    b.setInsertionPointAfter(pred.getDefiningOp());
    llvm::SetVector<Operation *> predOps;
    if (!getDominatingValueSetOpsToHoist(domInfo, mmaOp, pred, predOps)) {
      return mlir::emitWarning(pred.getLoc(),
                               "failed to hoist user predicate above MMA op");
    }
    hoistOpsBefore(mmaOp, predOps);

    // Set up production of the accumulator result.
    mmaOp.addCompletionBarrier(curAccReadyBar, pred);
    b.createInPartition<ttng::WaitBarrierOp>(*mmaPartition, curAccEmptyBar,
                                             accPhase, pred);
    assert(donePt.getPoint() == b.getInsertionPoint() ||
           donePt.getPoint()->isBeforeInBlock(&*b.getInsertionPoint()));

    if (!userPartition)
      userPartition = schedule.addPartition(numStages + numMmaStages);
    // Acquire and get the accumulator result. Normally, we want to acquire the
    // accumulator for as small of a critical section as possible to unblock
    // dependents, but if the most dominating user is inside a conditional,
    // acquire the accumulator for the whole branch. This will improve
    // instruction scheduling and interleaving of the TMEM load.
    bool userInConditional = isa<scf::IfOp>(domOp->getParentOp());
    b.setInsertionPoint(domOp);
    if (userInConditional)
      b.setInsertionPointToStart(domOp->getBlock());
    b.createInPartition<ttng::WaitBarrierOp>(*userPartition, curAccReadyBar,
                                             accPhase);

    b.setInsertionPoint(domOp);

    // Signal the accumulator buffer is ready for the next iteration. Because
    // the mbarriers got shifted over by 1, we have to signal the next mbarrier.
    if (userInConditional) {
      b.setInsertionPoint(domOp->getBlock()->getTerminator());
    } else {
      PostDominanceInfo postDomInfo(loop);
      b.setInsertionPointAfter(
          findNearestCommonPostDominator(loadsInLoop, postDomInfo));
    }
    Value prevIndex =
        b.create<arith::AddIOp>(accIndex, b.intCst(numMmaStages - 1));
    prevIndex = b.create<arith::RemUIOp>(prevIndex, b.intCst(numMmaStages));
    Value nextAccEmptyBar = createSingleBufferView(b, accEmptyBars, prevIndex);
    b.createInPartition<ttng::ArriveBarrierOp>(*userPartition, nextAccEmptyBar,
                                               1);

    // Propagate the partition to transitive users. If this happens to create a
    // cycle, subsequent warp specialization steps will fail.
    SmallVector<Operation *> transitiveUsers(loadsInLoop.begin(),
                                             loadsInLoop.end());
    DenseSet<Operation *> seen;
    while (!transitiveUsers.empty()) {
      Operation *op = transitiveUsers.pop_back_val();
      op = loop.getBody()->findAncestorOpInBlock(*op);
      if (!seen.insert(op).second)
        continue;
      userPartition->insert(op);
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : op->getUses())
        uses.push_back(&use);
      for (unsigned i = 0; i < uses.size(); ++i) {
        Operation *user = uses[i]->getOwner();
        if (user == loop.getBody()->getTerminator()) {
          for (OpOperand &use :
               loop.getRegionIterArgs()[uses[i]->getOperandNumber()].getUses())
            uses.push_back(&use);
        } else {
          transitiveUsers.push_back(user);
        }
      }
    }

    // Place the epilogue partition in the default warpgroup. The MMA and load
    // partitions shouldn't have tensor computations in them, which means they
    // will get assigned just 1 warp each. Add an extra partition to pad the
    // number of warps to the nearest warpgroup.
    schedule.addPartition(0);
    schedule.reorderPartitions({2, 1, 0, 3});

  } else if (!loadGroups.empty()) {
    b.setInsertionPointAfter(loop);
    // The MMA has no direct use in the loop, so we have to drain the pipeline
    // of MMA waits.
    LoadGroup &group = loadGroups.front();
    Value lastIdx = loop.getResult(group.index.getArgNumber() - 1);
    Value lastPhase = loop.getResult(group.phase.getArgNumber() - 1);
    for (auto i : llvm::seq(numStages)) {
      Value emptyBar = createSingleBufferView(b, group.emptyBars, lastIdx);
      b.create<ttng::WaitBarrierOp>(emptyBar, lastPhase);
      std::tie(lastIdx, lastPhase) =
          postIncrementModulo(b, lastIdx, lastPhase, numStages);
    }
  }

  schedule.serialize(loop);

  // HACK: Set this attribute so that LowerLoops will multi-buffer TMA
  // descriptors.
  loop->setAttr(kScheduledMaxStageAttrName, b.getI32IntegerAttr(numStages));
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
    assignPartitions(loop);
    // if (failed(specializeLoadMMADependencies(loop, numStages)))
    //   continue;
  }
}
