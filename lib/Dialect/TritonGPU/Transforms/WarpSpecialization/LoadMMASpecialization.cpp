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
// specializeLoadMMADependencies
//===----------------------------------------------------------------------===//

// Pattern match a simple `tma_load -> ... -> tl.dot` single-user chain. This
// ensures there are extraneous users of the load or intermediate values and
// that a valid partition schedule can be formed.
//
// TODO: Expand partioning scheme to support arbitrary DAG of loads and MMAs.
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
  if (isa<ttng::TMEMAllocOp, LocalAllocOp, MemDescTransOp, ConvertLayoutOp>(
          defOp)) {
    assert(llvm::is_contained({0, 1}, defOp->getNumOperands()));
    // Alloc ops have an optional source operand.
    if (defOp->getNumOperands() != 1)
      return failure();
    ops.push_back(defOp);
    return findSingleChainToLoad(loop, defOp->getOperand(0), ops);
  }

  return failure();
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
addIndexAndPhase(ImplicitLocOpBuilder &b, scf::ForOp &loop, unsigned numStages,
                 Value epilogue = {}) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);
  auto intCst = [&](int value) {
    return b.create<arith::ConstantIntOp>(value, 32);
  };

  // Index and phase both start at 0.
  unsigned curArgIdx = loop.getNumRegionIterArgs();
  auto newArgs = addIterArgsToLoop(b, loop, {intCst(0), intCst(0)});
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

// Create an operation inside a partition.
template <typename OpT, typename... Args>
static auto createInPartition(ImplicitLocOpBuilder &b, Partition &partition,
                              Args &&...args) {
  auto op = b.create<OpT>(std::forward<Args>(args)...);
  partition.insert(op);
  return op;
}

static void lowerTMACopy(ImplicitLocOpBuilder &b, Partition &partition,
                         Operation *op, Value barrier, Value view) {
  Value truePred = b.create<arith::ConstantIntOp>(true, /*width=*/1);
  if (auto load = dyn_cast<DescriptorLoadOp>(op)) {
    Value tmaPtr = createInPartition<ttng::TensorDescToTMAPtrOp>(
        b, partition, load.getDesc());
    createInPartition<ttng::AsyncTMACopyGlobalToLocalOp>(
        b, partition, tmaPtr, load.getIndices(), barrier, view, truePred);
  } else {
    auto gather = cast<DescriptorGatherOp>(op);
    Value tmaPtr = createInPartition<ttng::TensorDescToTMAPtrOp>(
        b, partition, gather.getDesc());
    createInPartition<ttng::AsyncTMAGatherOp>(
        b, partition, tmaPtr, gather.getXOffsets(), gather.getYOffset(),
        barrier, view, truePred);
  }
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

LogicalResult triton::gpu::specializeLoadMMADependencies(scf::ForOp &loop,
                                                         int defaultNumStages) {
  auto ops = llvm::to_vector(loop.getOps<ttng::MMAv5OpInterface>());
  if (ops.empty())
    return success();
  // Support only 1 MMA op.
  if (ops.size() > 1) {
    return mlir::emitWarning(
        loop.getLoc(),
        "failed to warp specialize: more than one `tt.dot` found in the loop");
  }
  ttng::MMAv5OpInterface mmaOp = ops.front();
  auto dot = cast<DotOpInterface>(*mmaOp);

  // Look for the loads that feed the A and B operands.
  SmallVector<Operation *> aChain, bChain;
  if (failed(findSingleChainToLoad(loop, dot.getA(), aChain)) ||
      failed(findSingleChainToLoad(loop, dot.getB(), bChain))) {
    return mlir::emitWarning(loop.getLoc(),
                             "failed to warp specialize: could not find TMA "
                             "loads for `tt.dot` operands");
  }

  SmallVector<Operation *> aScaleChain, bScaleChain;
  auto scaledMMAOp = dyn_cast<ttng::TCGen5MMAScaledOp>(mmaOp.getOperation());
  if (scaledMMAOp) {
    if (failed(
            findSingleChainToLoad(loop, scaledMMAOp.getAScale(), aScaleChain)))
      aScaleChain.clear();
    if (failed(
            findSingleChainToLoad(loop, scaledMMAOp.getBScale(), bScaleChain)))
      bScaleChain.clear();
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
    return llvm::is_contained(llvm::to_vector(llvm::concat<Operation *>(
                                  aChain, bChain, aScaleChain, bScaleChain)),
                              op);
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

  ImplicitLocOpBuilder b(mmaOp.getLoc(), loop);
  auto intCst = [&](int value, unsigned width = 32) {
    return b.create<arith::ConstantIntOp>(value, width);
  };

  // Collect a condition that fires whenever the accumulator value is reset in
  // the loop.
  Value overridePred;
  for (Operation *user : accUsersInLoop) {
    if (auto mmaOp = dyn_cast<ttng::MMAv5OpInterface>(user)) {
      Value flag = mmaOp.useAccumulator();
      if (!matchPattern(flag, m_One())) {
        if (auto arg = dyn_cast<BlockArgument>(flag)) {
          auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
          overridePred = yield.getOperand(arg.getArgNumber() - 1);
          b.setInsertionPoint(yield);
          overridePred = b.create<arith::XOrIOp>(overridePred, intCst(true, 1));
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

  // Multi-buffer the loads.
  BlockArgument loadIndex;
  BlockArgument loadPhase;
  std::tie(loadIndex, loadPhase) = addIndexAndPhase(b, loop, numStages);

  auto allocate = [&](const SmallVector<Operation *> &chain)
      -> std::tuple<Operation *, RankedTensorType, SharedEncodingTrait, Value> {
    if (chain.empty())
      return {nullptr, RankedTensorType(), SharedEncodingTrait(), Value()};

    Operation *load = chain.back();
    auto type = cast<RankedTensorType>(load->getResult(0).getType());
    SharedEncodingTrait enc = getSharedEncoding(chain.back());
    Value alloc = createAlloc(loop, type, load->getLoc(), enc, numStages);

    return {load, type, enc, alloc};
  };

  auto [aLoad, aType, aEnc, aAlloc] = allocate(aChain);
  auto [bLoad, bType, bEnc, bAlloc] = allocate(bChain);
  auto [aScaleLoad, aScaleType, aScaleEnc, aScaleAlloc] = allocate(aScaleChain);
  auto [bScaleLoad, bScaleType, bScaleEnc, bScaleAlloc] = allocate(bScaleChain);

  // Share the same set of barriers for both.
  Value emptyBars = createBarrierAlloc(loop, numStages);
  Value readyBars = createBarrierAlloc(loop, numStages);
  // Mark the empty barriers as initially ready.
  b.setInsertionPoint(loop);
  for (auto i : llvm::seq(numStages)) {
    Value emptyBar = createSingleBufferView(b, emptyBars, i);
    b.create<ttng::ArriveBarrierOp>(emptyBar, 1);
  }

  int loadSizeInBytes =
      product(aType.getShape()) * aType.getElementTypeBitWidth() / 8 +
      product(bType.getShape()) * bType.getElementTypeBitWidth() / 8;
  if (aScaleLoad)
    loadSizeInBytes += product(aScaleType.getShape()) *
                       aScaleType.getElementTypeBitWidth() / 8;
  if (bScaleLoad)
    loadSizeInBytes += product(bScaleType.getShape()) *
                       bScaleType.getElementTypeBitWidth() / 8;

  // Insert before the group of loads.
  SmallVector<Operation *> allLoads{aLoad, bLoad};
  if (aScaleLoad)
    allLoads.push_back(aScaleLoad);
  if (bScaleLoad)
    allLoads.push_back(bScaleLoad);
  std::sort(allLoads.begin(), allLoads.end(),
            [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  b.setInsertionPoint(allLoads.front());

  // Wait for the buffer to be empty and the corresponding barrier to be
  // exhausted.
  Value curEmptyBar = createSingleBufferView(b, emptyBars, loadIndex);
  createInPartition<ttng::WaitBarrierOp>(b, *loadPartition, curEmptyBar,
                                         loadPhase);
  // Indicate the expected size of the loads.
  Value curLoadBar = createSingleBufferView(b, readyBars, loadIndex);
  createInPartition<ttng::BarrierExpectOp>(b, *loadPartition, curLoadBar,
                                           loadSizeInBytes, intCst(true, 1));

  // Replace the loads with async copies.
  auto lowerLoadAndPropagate = [&](Operation *load, Value alloc,
                                   Value barrier) {
    b.setInsertionPoint(load);
    Value view = createSingleBufferView(b, alloc, loadIndex);
    lowerTMACopy(b, *loadPartition, load, barrier, view);
    replaceUsesAndPropagateType(b, *load->user_begin(), view);
    load->user_begin()->erase();
    load->erase();
  };
  lowerLoadAndPropagate(aLoad, aAlloc, curLoadBar);
  lowerLoadAndPropagate(bLoad, bAlloc, curLoadBar);
  if (aScaleLoad)
    lowerLoadAndPropagate(aScaleLoad, aScaleAlloc, curLoadBar);
  if (bScaleLoad)
    lowerLoadAndPropagate(bScaleLoad, bScaleAlloc, curLoadBar);

  // Place the remaining users in the MMA partition. Re-acquire the use chain
  // because some ops were invalidated by `replaceUsesAndPropagateType`.
  aChain.clear();
  bChain.clear();
  aChain.push_back(mmaOp);
  (void)findSingleChainToLoad(loop, dot.getA(), aChain);
  (void)findSingleChainToLoad(loop, dot.getB(), bChain);
  if (aScaleLoad) {
    aScaleChain.clear();
    (void)findSingleChainToLoad(loop, scaledMMAOp.getAScale(), aScaleChain);
  }
  if (bScaleLoad) {
    bScaleChain.clear();
    (void)findSingleChainToLoad(loop, scaledMMAOp.getBScale(), bScaleChain);
  }

  // Place users in the MMA partition.
  auto allUsers = llvm::to_vector(
      llvm::concat<Operation *>(aChain, bChain, aScaleChain, bScaleChain));
  for (Operation *user : allUsers)
    mmaPartition->insert(user);

  // Insert the load wait before the first user.
  Operation *minOp = findNearestCommonDominator(allUsers, domInfo);
  b.setInsertionPoint(minOp);
  createInPartition<ttng::WaitBarrierOp>(b, *mmaPartition, curLoadBar,
                                         loadPhase);

  // Now rewrite the MMA by multi-buffering the accumulator if necessary.
  // However, the TMEM multi-buffering may be with respect to the outer loop.
  b.setInsertionPoint(mmaOp);
  mmaOp.addCompletionBarrier(curEmptyBar, intCst(true, 1));

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
  for (OpOperand &use : llvm::make_early_inc_range(oldAccAlloc->getUses())) {
    Operation *user = use.getOwner();
    b.setInsertionPoint(user);
    Value bufIdx;
    if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
      if (loop->isAncestor(store)) {
        mmaPartition->insert(store);
        bufIdx = b.create<arith::AddIOp>(accIndex, intCst(1));
        bufIdx = b.create<arith::RemUIOp>(bufIdx, intCst(numMmaStages));
      } else {
        if (!store->isBeforeInBlock(loop))
          return mlir::emitWarning(store.getLoc(), "store not before loop?");
        bufIdx = intCst(0);
      }
    } else if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (loop->isAncestor(load)) {
        loadsInLoop.push_back(load);
        bufIdx = accIndex;
      } else {
        if (!loop->isBeforeInBlock(load))
          return mlir::emitWarning(load.getLoc(), "load not after loop?");
        bufIdx = loop.getResult(accIndex.getArgNumber() - 1);
      }
    } else if (user == mmaOp) {
      bufIdx = accIndex;
    } else {
      return mlir::emitWarning(user->getLoc(), "unknown acc user");
    }
    Value buf = createSingleBufferView(b, accAlloc, bufIdx);
    use.set(buf);
  }
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
    createInPartition<ttng::WaitBarrierOp>(b, *mmaPartition, curAccEmptyBar,
                                           accPhase, pred);
    assert(donePt.getPoint() == b.getInsertionPoint() ||
           donePt.getPoint()->isBeforeInBlock(&*b.getInsertionPoint()));

    Partition *userPartition = schedule.addPartition(numStages + numMmaStages);
    // Acquire and get the accumulator result. Normally, we want to acquire the
    // accumulator for as small of a critical section as possible to unblock
    // dependents, but if the most dominating user is inside a conditional,
    // acquire the accumulator for the whole branch. This will improve
    // instruction scheduling and interleaving of the TMEM load.
    bool userInConditional = isa<scf::IfOp>(domOp->getParentOp());
    b.setInsertionPoint(domOp);
    if (userInConditional)
      b.setInsertionPointToStart(domOp->getBlock());
    createInPartition<ttng::WaitBarrierOp>(b, *userPartition, curAccReadyBar,
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
        b.create<arith::AddIOp>(accIndex, intCst(numMmaStages - 1));
    prevIndex = b.create<arith::RemUIOp>(prevIndex, intCst(numMmaStages));
    Value nextAccEmptyBar = createSingleBufferView(b, accEmptyBars, prevIndex);
    createInPartition<ttng::ArriveBarrierOp>(b, *userPartition, nextAccEmptyBar,
                                             1);

    // Propagate the partition to transitive users. If this happens to create a
    // cycle, subsequent warp specialization steps will fail.
    SmallVector<Operation *> transitiveUsers(loadsInLoop.begin(),
                                             loadsInLoop.end());
    while (!transitiveUsers.empty()) {
      Operation *op = transitiveUsers.pop_back_val();
      if (isa<scf::YieldOp>(op))
        continue;
      op = loop.getBody()->findAncestorOpInBlock(*op);
      userPartition->insert(op);
      llvm::append_range(transitiveUsers, op->getUsers());
    }

    // Place the epilogue partition in the default warpgroup. The MMA and load
    // partitions shouldn't have tensor computations in them, which means they
    // will get assigned just 1 warp each. Add an extra partition to pad the
    // number of warps to the nearest warpgroup.
    schedule.addPartition(0);
    schedule.reorderPartitions({2, 1, 0, 3});

  } else {
    b.setInsertionPointAfter(loop);
    // The MMA has no direct use in the loop, so we have to drain the pipeline
    // of MMA waits.
    Value lastIdx = loop.getResult(loadIndex.getArgNumber() - 1);
    Value lastPhase = loop.getResult(loadPhase.getArgNumber() - 1);
    for (auto i : llvm::seq(numStages)) {
      Value emptyBar = createSingleBufferView(b, emptyBars, lastIdx);
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
    if (failed(specializeLoadMMADependencies(loop, numStages)))
      continue;
  }
}
