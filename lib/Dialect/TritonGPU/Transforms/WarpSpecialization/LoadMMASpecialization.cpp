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

using MMAInfo = ttng::MMAInfo;
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

static std::pair<Value, Value> addIndexAndPhase(ImplicitLocOpBuilder &b,
                                                scf::ForOp &loop,
                                                unsigned numStages,
                                                Value epilogue = {}) {
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(loop);
  auto intCst = [&](int value) {
    return b.create<arith::ConstantIntOp>(value, 32);
  };

  // Index and phase both start at 0.
  unsigned curArgIdx = loop.getNumRegionIterArgs();
  auto newArgs = addIterArgsToLoop(b, loop, {intCst(0), intCst(0)});
  Value index = newArgs[0];
  Value phase = newArgs[1];

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  b.setInsertionPoint(yield);

  // Post-increment the index and phase.
  Value nextIndex = b.create<arith::AddIOp>(index, intCst(1));
  Value nextPhase = b.create<arith::XOrIOp>(phase, intCst(1));

  Value rollover = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, nextIndex,
                                           intCst(numStages));
  nextIndex = b.create<arith::SelectOp>(rollover, intCst(0), nextIndex);
  nextPhase = b.create<arith::SelectOp>(rollover, nextPhase, phase);

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
getUserPrecondition(ImplicitLocOpBuilder &b, scf::ForOp loop,
                    Operation *domOp) {
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

  Value precondition = trueVal;
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

  // Determine if the MMA can be pipelined according to some specific rules.
  DominanceInfo domInfo(loop);
  std::optional<MMAInfo> mmaInfoOr = getMMAInfo(loop, mmaOp, domInfo);
  if (!mmaInfoOr) {
    return mlir::emitWarning(loop.getLoc(),
                             "failed to warp specialize: could not determine "
                             "if the MMA op can be pipelined");
  }
  MMAInfo info = std::move(*mmaInfoOr);

  // FIXME: We rely on the reset point of the accumulator to indicate where the
  // epilogue once was if the loop was flattened.
  if (info.accIsMultiBuffered) {
    MMAInfo::AccOverridePoint def = *info.accDef;
    Operation *defOp = def.condition ? def.condition.getDefiningOp() : def.op;
    if (defOp->getBlock() != loop.getBody() || defOp->isBeforeInBlock(mmaOp)) {
      return mlir::emitWarning(loop.getLoc(),
                               "failed to warp specialize: accumulator reset "
                               "does not occur after the `tt.dot`");
    }
  }

  // Pattern match succeeded. Now rewrite the loads and MMA ops to pass tensor
  // values through buffers.
  int numStages = getNumStagesOrDefault(loop, defaultNumStages);
  int numMmaStages = 1 + info.accIsMultiBuffered;
  WarpSchedule schedule;
  Partition *loadPartition = schedule.addPartition(0);
  Partition *mmaPartition = schedule.addPartition(numStages);
  Partition *waiterPartition = schedule.addPartition(numStages + numMmaStages);

  ImplicitLocOpBuilder b(mmaOp.getLoc(), loop);
  auto intCst = [&](int value, unsigned width = 32) {
    return b.create<arith::ConstantIntOp>(value, width);
  };

  // Multi-buffer the loads.
  auto [loadIndex, loadPhase] = addIndexAndPhase(b, loop, numStages);

  Operation *aLoad = aChain.back();
  Operation *bLoad = bChain.back();
  auto aType = cast<RankedTensorType>(aLoad->getResult(0).getType());
  auto bType = cast<RankedTensorType>(bLoad->getResult(0).getType());
  SharedEncodingTrait aEnc = getSharedEncoding(aChain.back());
  SharedEncodingTrait bEnc = getSharedEncoding(bChain.back());
  Value aAlloc = createAlloc(loop, aType, aLoad->getLoc(), aEnc, numStages);
  Value bAlloc = createAlloc(loop, bType, bLoad->getLoc(), bEnc, numStages);

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

  // Insert before the group of loads.
  b.setInsertionPoint(aLoad->isBeforeInBlock(bLoad) ? aLoad : bLoad);
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
  b.setInsertionPoint(aLoad);
  Value aView = createSingleBufferView(b, aAlloc, loadIndex);
  lowerTMACopy(b, *loadPartition, aLoad, curLoadBar, aView);
  replaceUsesAndPropagateType(b, *aLoad->user_begin(), aView);
  aLoad->user_begin()->erase();
  aLoad->erase();

  b.setInsertionPoint(bLoad);
  Value bView = createSingleBufferView(b, bAlloc, loadIndex);
  lowerTMACopy(b, *loadPartition, bLoad, curLoadBar, bView);
  replaceUsesAndPropagateType(b, *bLoad->user_begin(), bView);
  bLoad->user_begin()->erase();
  bLoad->erase();

  // Place the remaining users in the MMA partition. Re-acquire the use chain
  // because some ops were invalidated by `replaceUsesAndPropagateType`.
  aChain.clear();
  bChain.clear();
  aChain.push_back(mmaOp);
  (void)findSingleChainToLoad(loop, dot.getA(), aChain);
  (void)findSingleChainToLoad(loop, dot.getB(), bChain);

  // Place users in the MMA partition.
  auto allUsers = llvm::concat<Operation *>(aChain, bChain);
  for (Operation *user : allUsers)
    mmaPartition->insert(user);

  // Insert the load wait before the first user.
  auto minIt = llvm::min_element(allUsers, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  b.setInsertionPoint(*minIt);
  createInPartition<ttng::WaitBarrierOp>(b, *mmaPartition, curLoadBar,
                                         loadPhase);

  // Now rewrite the MMA by hoisting the TMEM allocation out of the loop and
  // multi-buffering it if necessary. However, the TMEM multi-buffering may be
  // with respect to the outer loop.
  auto [mmaIndex, mmaPhase] = addIndexAndPhase(b, loop, numStages);
  Value mmaBars = createBarrierAlloc(loop, numStages);

  b.setInsertionPoint(mmaOp);
  Value curMmaBar = createSingleBufferView(b, mmaBars, mmaIndex);
  mmaOp.setBarrier(curMmaBar);

  b.setInsertionPointAfter(mmaOp);
  createInPartition<ttng::WaitBarrierOp>(b, *waiterPartition, curMmaBar,
                                         mmaPhase);
  createInPartition<ttng::ArriveBarrierOp>(b, *waiterPartition, curEmptyBar, 1);
  OpBuilder::InsertPoint donePt = b.saveInsertionPoint();

  // Now handle the accumulator, which is the tricky bit. The accumulator value
  // may be conditionally reset in the MMA partition before the MMA op, and it
  // may be conditionally used in a user partition.
  b.setInsertionPoint(loop);
  ttng::TMEMAllocOp accAlloc =
      createTMemAlloc(b, info.accAlloc, /*multiBuffered=*/true, numMmaStages);
  auto accInitArg = cast<BlockArgument>(info.accAlloc.getSrc());
  Value accInitValue = loop.getInitArgs()[accInitArg.getArgNumber() - 1];
  ttng::createInitStore(b, accAlloc, accInitValue, /*multiBuffered=*/true);

  // If the accumulator is multibuffered, the buffer changes when the
  // accumulator is reset.
  auto [accIndex, accPhase] = addIndexAndPhase(
      b, loop, numMmaStages,
      info.accDef.value_or(MMAInfo::AccOverridePoint{}).condition);
  b.setInsertionPoint(mmaOp);
  Value curAccBuf = createSingleBufferView(b, accAlloc, accIndex);
  mmaOp.setAccumulator(curAccBuf);

  // Replace uses of the accumulator inside the loop with a value loaded from
  // the buffer. Place these in a new user partition.
  SmallVector<Operation *> accUses = getDirectAccUses(info.accLoad);
  if (!accUses.empty()) {
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

    Operation *domOp = findNearestCommonDominator(accUses, domInfo);
    assert(domOp && "could not find common dominator for accumulator uses");
    Value pred;
    b.restoreInsertionPoint(donePt);
    std::tie(pred, domOp) = getUserPrecondition(b, loop, domOp);

    // Set up production of the accumulator result.
    b.setInsertionPointAfter(pred.getDefiningOp());
    createInPartition<ttng::ArriveBarrierOp>(b, *waiterPartition,
                                             curAccReadyBar, 1, pred);
    createInPartition<ttng::WaitBarrierOp>(b, *mmaPartition, curAccEmptyBar,
                                           accPhase, pred);
    assert(donePt.getPoint() == b.getInsertionPoint() ||
           donePt.getPoint()->isBeforeInBlock(&*b.getInsertionPoint()));
    donePt = b.saveInsertionPoint();

    // Acquire and get the accumulator result.
    b.setInsertionPoint(domOp);
    Partition *userPartition = schedule.addPartition(numStages + numMmaStages);
    createInPartition<ttng::WaitBarrierOp>(b, *userPartition, curAccReadyBar,
                                           accPhase);
    Value acc = createInPartition<ttng::TMEMLoadOp>(
        b, *userPartition, info.accLoad.getType(), curAccBuf);
    for (Operation *user : accUses)
      user->replaceUsesOfWith(info.accLoad, acc);
    // Signal the accumulator buffer is ready for the next iteration. Because
    // the mbarriers got shifted over by 1, we have to signal the next mbarrier.
    Value nextIndex =
        b.create<arith::AddIOp>(accIndex, intCst(numMmaStages - 1));
    nextIndex = b.create<arith::RemUIOp>(nextIndex, intCst(numMmaStages));
    Value nextAccEmptyBar = createSingleBufferView(b, accEmptyBars, nextIndex);
    createInPartition<ttng::ArriveBarrierOp>(b, *userPartition, nextAccEmptyBar,
                                             1);

    // Propagate the partition to transitive users. If this happens to create a
    // cycle, subsequent warp specialization steps will fail.
    while (!accUses.empty()) {
      Operation *op = accUses.pop_back_val();
      if (isa<scf::YieldOp>(op))
        continue;
      op = loop.getBody()->findAncestorOpInBlock(*op);
      userPartition->insert(op);
      llvm::append_range(accUses, op->getUsers());
    }

    // Place the epilogue partition in the default warpgroup. The MMA and load
    // partitions shouldn't have tensor computations in them, which means they
    // will get assigned just 1 warp each.
    schedule.reorderPartitions({2, 1, 3, 0});
  }

  // Update the reset of the accumulator in the loop if it is multi-buffered.
  if (info.accIsMultiBuffered && info.accDef->initValue) {
    MMAInfo::AccOverridePoint def = *info.accDef;
    b.setInsertionPointAfter(def.condition ? def.condition.getDefiningOp()
                                           : def.op);
    assert(b.getInsertionBlock() == loop.getBody());
    if (b.getInsertionPoint()->isBeforeInBlock(&*donePt.getPoint()))
      b.restoreInsertionPoint(donePt);
    Value pred = def.condition ? def.condition : intCst(true, 1);

    // Write the initial value for the next accumulator buffer.
    Value nextIndex = b.create<arith::AddIOp>(accIndex, intCst(1));
    nextIndex = b.create<arith::RemUIOp>(nextIndex, intCst(numMmaStages));
    Value nextAccBuf = createSingleBufferView(b, accAlloc, nextIndex);
    createInPartition<ttng::TMEMStoreOp>(b, *mmaPartition, nextAccBuf,
                                         def.initValue, pred);
    if (def.condition) {
      def.op->dropAllUses();
      def.op->erase();
    }
  }

  // Replace uses of the accumulator outside the loop.
  llvm::BitVector toErase(loop.getNumRegionIterArgs());
  if (info.yieldArgNo) {
    b.setInsertionPointAfter(loop);
    Value accBuf =
        createSingleBufferView(b, accAlloc, loop.getResults().end()[-2]);
    Value acc = b.create<ttng::TMEMLoadOp>(info.accLoad.getType(), accBuf);
    loop.getResult(*info.yieldArgNo).replaceAllUsesWith(acc);
    toErase.set(*info.yieldArgNo);
  }

  info.accAlloc->dropAllUses();
  info.accLoad->dropAllUses();
  info.accAlloc.erase();
  info.accLoad.erase();
  eraseLoopCarriedValues(loop, toErase);

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
