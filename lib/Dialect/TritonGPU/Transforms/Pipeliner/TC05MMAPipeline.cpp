#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

using MMAInfo = ttng::MMAInfo;

namespace {

const char *kPipelineStageAttrName = "triton.pipeline_stage";
const char *kPipelineAttrName = "triton.pipeline";

// Utils:
void replaceAllUsesDominatedBy(Operation *domOp, Value newValue, Value oldValue,
                               DominanceInfo &domInfo) {
  oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
    return domInfo.properlyDominates(domOp, use.getOwner());
  });
}

void annotateWithPipelineStage(IRRewriter &builder, Operation *op, int stage) {
  op->setAttr(kPipelineStageAttrName,
              IntegerAttr::get(builder.getI32Type(), stage));
}

int getPipelineStage(Operation *op) {
  return op->getAttrOfType<IntegerAttr>(kPipelineStageAttrName).getInt();
}

void updateAccUsesInLoop(IRRewriter &builder, scf::ForOp forOp, MMAInfo &info,
                         ttng::TMEMAllocOp newAlloc, int numStages) {
  DominanceInfo domInfo(forOp);
  SmallVector<Operation *> directUses = ttng::getDirectAccUses(info.accLoad);
  if (!directUses.empty()) {
    Operation *domOp = findNearestCommonDominator(directUses, domInfo);
    assert(domOp != nullptr && "Could not find a common dominator");
    builder.setInsertionPoint(domOp);
    Value extractSlice = newAlloc;
    if (info.accIsMultiBuffered) {
      extractSlice =
          triton::createSingleBufferView(builder, newAlloc, info.accExtractIdx);
    }
    auto load = builder.create<ttng::TMEMLoadOp>(
        domOp->getLoc(), info.accLoad.getType(), extractSlice);
    // If accumulator is multi-buffered, it is implicit that we put the load
    // in the last stage.
    int pipelineStage = info.accIsMultiBuffered ? numStages - 1 : 0;
    annotateWithPipelineStage(
        builder, forOp.getBody()->findAncestorOpInBlock(*load.getOperation()),
        pipelineStage);
    for (auto user : directUses) {
      user->replaceUsesOfWith(info.accLoad, load);
    }
  }
}

void updateAccUsesOutsideLoop(IRRewriter &builder, scf::ForOp forOp,
                              const MMAInfo &info, ttng::TMEMAllocOp newAlloc,
                              int extractIdxArgNo) {
  builder.setInsertionPointAfter(forOp);
  if (!info.yieldArgNo.has_value()) {
    return;
  }
  if (forOp.getResult(info.yieldArgNo.value()).getUsers().empty()) {
    return;
  }
  Value bufferSlice = newAlloc;
  if (info.accIsMultiBuffered) {
    Value extractIdxVal = forOp.getResult(extractIdxArgNo);
    bufferSlice =
        triton::createSingleBufferView(builder, newAlloc, extractIdxVal);
  }
  auto load = builder.create<ttng::TMEMLoadOp>(
      forOp.getLoc(), forOp.getResult(info.yieldArgNo.value()).getType(),
      bufferSlice);
  forOp.getResult(info.yieldArgNo.value()).replaceAllUsesWith(load);
}

void updateAccDefsInLoop(IRRewriter &builder, scf::ForOp forOp, MMAInfo &info,
                         ttng::TMEMAllocOp newAlloc, int numStages,
                         DominanceInfo &domInfo) {
  assert(info.accDef.has_value());
  Operation *def = info.accDef->op;
  Value condition = info.accDef->condition;
  Location loc = def->getLoc();

  builder.setInsertionPointAfter(def);
  if (condition && condition.getDefiningOp()) {
    builder.setInsertionPointAfter(condition.getDefiningOp());
  }
  // if insertion point is outside the loop body, move it inside
  if (builder.getBlock() != forOp.getBody()) {
    builder.setInsertionPointAfter(&forOp.getBody()->front());
  }
  Value numStagesVal = builder.create<arith::ConstantIntOp>(loc, numStages, 32);

  Value newInsertIdx = builder.create<arith::AddIOp>(
      loc, info.accInsertIdx, builder.create<arith::ConstantIntOp>(loc, 1, 32));
  Value insWrap = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                newInsertIdx, numStagesVal);
  newInsertIdx = builder.create<arith::SelectOp>(
      loc, insWrap, builder.create<arith::ConstantIntOp>(loc, 0, 32),
      newInsertIdx);
  if (condition) {
    newInsertIdx = builder.create<arith::SelectOp>(loc, condition, newInsertIdx,
                                                   info.accInsertIdx);
  }
  annotateWithPipelineStage(builder, newInsertIdx.getDefiningOp(), 0);

  Value newExtractIdx = builder.create<arith::AddIOp>(
      loc, info.accExtractIdx,
      builder.create<arith::ConstantIntOp>(loc, 1, 32));
  auto extWrap = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               newExtractIdx, numStagesVal);
  newExtractIdx = builder.create<arith::SelectOp>(
      loc, extWrap, builder.create<arith::ConstantIntOp>(loc, 0, 32),
      newExtractIdx);
  if (info.accDef->condition) {
    newExtractIdx = builder.create<arith::SelectOp>(
        loc, info.accDef->condition, newExtractIdx, info.accExtractIdx);
  }
  annotateWithPipelineStage(builder, newExtractIdx.getDefiningOp(), 1);

  if (info.accDef->initValue) {
    Value bufferSlice =
        triton::createSingleBufferView(builder, newAlloc, newInsertIdx);
    Value vTrue = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    auto tmemStore = builder.create<ttng::TMEMStoreOp>(
        loc, bufferSlice, info.accDef->initValue,
        condition ? condition : vTrue);
    annotateWithPipelineStage(builder, tmemStore, 0);
  }

  // Always update the for yield with the new insert and extract indices
  auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  forYield->replaceUsesOfWith(info.accInsertIdx, newInsertIdx);
  forYield->replaceUsesOfWith(info.accExtractIdx, newExtractIdx);

  // Only update rest of the uses if the override is dist 0 (the same
  // loop iteration)
  if (info.accDef->distance == 0) {
    replaceAllUsesDominatedBy(newInsertIdx.getDefiningOp(), newInsertIdx,
                              info.accInsertIdx, domInfo);
    replaceAllUsesDominatedBy(newExtractIdx.getDefiningOp(), newExtractIdx,
                              info.accExtractIdx, domInfo);
  }

  if (info.accDef->initValue && condition) {
    assert(isa<arith::SelectOp>(info.accDef->op));
    info.accDef->op->erase();
  }

  info.accInsertIdx = newInsertIdx;
  info.accExtractIdx = newExtractIdx;
}

// Hoist tmem_allocs outside of the loop and update the mma ops to use the
// hoisted tmem allocs. Also, update the acc loads and stores to use the new
// tmem allocs.
void hoistAndUseTMemAlloc(IRRewriter &builder, scf::ForOp forOp,
                          ttng::MMAv5OpInterface mmaOp, MMAInfo &info,
                          int numStages, DominanceInfo &domInfo) {
  builder.setInsertionPoint(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
  Value numStagesVal =
      builder.create<arith::ConstantIntOp>(forOp.getLoc(), numStages, 32);
  Value vTrue = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 1);

  builder.setInsertionPoint(forOp);
  ttng::TMEMAllocOp newAlloc = createTMemAlloc(
      builder, info.accAlloc, info.accIsMultiBuffered, numStages);
  bool chainedAcc = info.yieldArgNo.has_value();
  if (chainedAcc) {
    Value accInitValue = forOp.getInitArgs()[info.yieldArgNo.value()];
    ttng::createInitStore(builder, newAlloc, accInitValue,
                          info.accIsMultiBuffered);
  }

  // Update mma ops to use the hoisted tmem allocs
  Value insertSlice = newAlloc;
  if (info.accIsMultiBuffered) {
    builder.setInsertionPoint(mmaOp);
    insertSlice =
        triton::createSingleBufferView(builder, insertSlice, info.accInsertIdx);
  }

  mmaOp.setAccumulator(insertSlice);

  updateAccUsesInLoop(builder, forOp, info, newAlloc, numStages);
  assert(isa<BlockArgument>(info.accExtractIdx));
  int extractIdxArgNo =
      cast<BlockArgument>(info.accExtractIdx).getArgNumber() - 1;
  updateAccUsesOutsideLoop(builder, forOp, info, newAlloc, extractIdxArgNo);

  // Short circuit loop carry value that was holding the accumulator value,
  // removing the last reference to the loaded accumulator.
  if (info.yieldArgNo.has_value()) {
    forOp.getBody()->getTerminator()->setOperand(
        info.yieldArgNo.value(), forOp.getInitArgs()[info.yieldArgNo.value()]);
  }

  if (info.accIsMultiBuffered) {
    updateAccDefsInLoop(builder, forOp, info, newAlloc, numStages, domInfo);
  }

  info.accLoad.erase();
  info.accAlloc.erase();
  info.accAlloc = newAlloc;
}

// Create multi-buffered barrier allocs and lower the MMA to MMA + wait barrier
void createBarrierAndWaitOps(IRRewriter &builder, scf::ForOp forOp,
                             ttng::MMAv5OpInterface mmaOp, MMAInfo &info,
                             int numStages, DominanceInfo &domInfo) {
  builder.setInsertionPoint(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
  Value numStagesVal =
      builder.create<arith::ConstantIntOp>(forOp.getLoc(), numStages, 32);

  info.barrierAlloc = triton::createBarrierAlloc(forOp, numStages);

  Location loc = mmaOp->getLoc();
  builder.setInsertionPoint(mmaOp);

  Value barrierSlice = triton::createSingleBufferView(
      builder, info.barrierAlloc, info.barrierIdx);
  mmaOp.setBarrier(barrierSlice);

  builder.setInsertionPointAfter(mmaOp);
  auto waitOp =
      builder.create<ttng::WaitBarrierOp>(loc, barrierSlice, info.phase);
  annotateWithPipelineStage(builder, waitOp, numStages - 1);

  Value newBarrierIdx =
      builder.create<arith::AddIOp>(loc, info.barrierIdx, one);
  auto barWrap = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               newBarrierIdx, numStagesVal);

  // New barrierIdx and phase are in the first stage, so they can be used by
  // the ops that are ahead of them in either order or stages.
  newBarrierIdx =
      builder.create<arith::SelectOp>(loc, barWrap, zero, newBarrierIdx);
  replaceAllUsesDominatedBy(newBarrierIdx.getDefiningOp(), newBarrierIdx,
                            info.barrierIdx, domInfo);
  info.barrierIdx = newBarrierIdx;
  annotateWithPipelineStage(builder, info.barrierIdx.getDefiningOp(), 0);

  Value originalPhase = info.phase;
  Value newPhase = builder.create<arith::SelectOp>(
      loc, barWrap, builder.create<arith::XOrIOp>(loc, info.phase, one),
      info.phase);
  replaceAllUsesDominatedBy(newPhase.getDefiningOp(), newPhase, info.phase,
                            domInfo);
  info.phase = newPhase;
  annotateWithPipelineStage(builder, info.phase.getDefiningOp(), 0);

  // We need to add a barrier before load from the accumulator, if it is in the
  // same stage as the dot.
  ttng::TMEMLoadOp tmemLoad = nullptr;
  SmallVector<Operation *> users = {info.accAlloc->getUsers().begin(),
                                    info.accAlloc->getUsers().end()};
  while (!users.empty()) {
    auto user = users.pop_back_val();
    if (isa<ttg::MemDescSubviewOp>(user)) {
      users.append(user->getUsers().begin(), user->getUsers().end());
    }
    if (isa<ttng::TMEMLoadOp>(user) && forOp->isAncestor(user)) {
      if (tmemLoad) {
        assert(tmemLoad == cast<ttng::TMEMLoadOp>(user) &&
               "Should have only one tmem load from the accumulator");
      }
      tmemLoad = cast<ttng::TMEMLoadOp>(user);
    }
  }
  if (tmemLoad) {
    int loadStage =
        getPipelineStage(forOp.getBody()->findAncestorOpInBlock(*tmemLoad));
    int mmaOpStage = getPipelineStage(mmaOp);
    if (loadStage == mmaOpStage) {
      builder.setInsertionPoint(tmemLoad);
      auto barrier =
          builder.create<ttng::WaitBarrierOp>(loc, barrierSlice, originalPhase);
      annotateWithPipelineStage(
          builder, forOp.getBody()->findAncestorOpInBlock(*barrier),
          mmaOpStage);
    }
  }
}

bool isSafeToPipeline(ttng::TCGen5MMAScaledOp scaledDot, scf::ForOp forOp) {
  // MMAv5 scaled dot (tcgen05.mma mxf8f6f4) is safe to be pipelined only
  // when its scales in TMEM are stored by the TMEMCopy op (tcgen05.cp).
  // That condition is equivalent to scale arguments of
  // ttng::TCGen5MMAScaledOp being in SMEM during SWP in our convention.
  auto isInvariantOrCopiedByTMEMCopy = [&](Value scale) {
    if (forOp.isDefinedOutsideOfLoop(scale))
      return true;
    if (auto tmemAlloc = scale.getDefiningOp<ttng::TMEMAllocOp>()) {
      Value tmemAllocSrc = tmemAlloc.getSrc();
      if (tmemAllocSrc && forOp.isDefinedOutsideOfLoop(tmemAllocSrc))
        return true;
    }
    auto scaleAlloc = findShmemAlloc(scale);
    if (!scaleAlloc || !forOp.isDefinedOutsideOfLoop(scaleAlloc))
      return false;
    return true;
  };

  return isInvariantOrCopiedByTMEMCopy(scaledDot.getAScale()) &&
         isInvariantOrCopiedByTMEMCopy(scaledDot.getBScale());
}

// Find MMAs eligible for pipelining and lower them by:
// 1. Hoisting the accumulator allocation outside of the loop.
// 2. Creating a barrier alloc and lowering the MMA to MMA + wait barrier.
// 3. Updating the uses of the accumulator in the loop to use the new tmem
// alloc.
FailureOr<scf::ForOp> preProcessLoopForTC05MMAPipelining(scf::ForOp forOp,
                                                         int numStages) {
  SmallVector<Operation *> mmaOps;
  forOp.walk([&](Operation *op) {
    // Skip MMA nested in another forOp
    if (op->getParentOfType<scf::ForOp>() == forOp) {
      if (isa<ttng::TCGen5MMAOp>(op)) {
        mmaOps.push_back(op);
      } else if (auto scaledDot = dyn_cast<ttng::TCGen5MMAScaledOp>(op)) {
        if (isSafeToPipeline(scaledDot, forOp)) {
          mmaOps.push_back(op);
        } else {
          op->emitWarning("Skipping pipelining of an MMAv5 scaled op because "
                          "TMEM copy is not used.");
        }
      }
    }
  });

  // Temporarily disable mma pipelining if there are more than one mmaOp in the
  // loop. This is a workaround for difficult to solve scheduling issues with
  // loads feeding into non-0 stage ops.
  if (mmaOps.empty() || mmaOps.size() > 1) {
    return failure();
  }

  mmaOps = getMMAsWithMultiBufferredOperands(forOp, mmaOps);

  if (mmaOps.empty()) {
    return failure();
  }

  IRRewriter builder(forOp->getContext());
  DominanceInfo domInfo(forOp);
  for (auto op : mmaOps) {
    // Avoid pipelining if in the backward slice of the mmaOp there is an
    // operation that is already assigned a stage, as it would make the pipeline
    // deeper than we are prepared for.
    auto mmaOp = cast<ttng::MMAv5OpInterface>(op);
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    getBackwardSlice(mmaOp, &backwardSlice, opt);
    if (llvm::any_of(backwardSlice, [&](Operation *op) {
          return op->hasAttr(kPipelineStageAttrName);
        })) {
      continue;
    }

    std::optional<MMAInfo> mmaInfoOr = ttng::getMMAInfo(forOp, mmaOp, domInfo);
    if (!mmaInfoOr)
      continue;
    MMAInfo mmaInfo = std::move(*mmaInfoOr);

    builder.setInsertionPoint(forOp);
    Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);

    // Update for loop with new arguments
    SmallVector<Value> newOperands;
    const int argsPerMMA = 4;
    newOperands.push_back(zero); // phase
    newOperands.push_back(zero); // barrierIdx
    newOperands.push_back(zero); // accInsertIdx
    newOperands.push_back(zero); // accExtractIdx
    assert(newOperands.size() == argsPerMMA);

    int firstNewOperandIndex = forOp.getInitArgs().size();
    (void)addIterArgsToLoop(builder, forOp, newOperands);

    mmaInfo.phase = forOp.getRegionIterArg(firstNewOperandIndex + 0);
    mmaInfo.barrierIdx = forOp.getRegionIterArg(firstNewOperandIndex + 1);
    mmaInfo.accInsertIdx = forOp.getRegionIterArg(firstNewOperandIndex + 2);
    mmaInfo.accExtractIdx = forOp.getRegionIterArg(firstNewOperandIndex + 3);

    SmallVector<Value> newYieldOperands;
    newYieldOperands.push_back(mmaInfo.phase);
    newYieldOperands.push_back(mmaInfo.barrierIdx);
    newYieldOperands.push_back(mmaInfo.accInsertIdx);
    newYieldOperands.push_back(mmaInfo.accExtractIdx);

    appendToForOpYield(forOp, newYieldOperands);

    annotateWithPipelineStage(builder, mmaOp, 0);
    hoistAndUseTMemAlloc(builder, forOp, mmaOp, mmaInfo, numStages, domInfo);
    createBarrierAndWaitOps(builder, forOp, mmaOp, mmaInfo, numStages, domInfo);
  }

  return forOp;
}

bool insertUsersOfOp(tt::CoarseSchedule &coarseSchedule, Operation *op,
                     int stage, tt::CoarseSchedule::Cluster cluster) {
  bool changed = false;
  for (auto user : op->getUsers()) {
    // Let wait barriers be scheduled based on the stage of async op it waits
    // for.
    if (!isa<ttng::WaitBarrierOp>(user) && coarseSchedule.count(user) == 0) {
      changed = true;
      coarseSchedule.insert(user, stage, cluster);
      insertUsersOfOp(coarseSchedule, user, stage, cluster);
    }
  }
  return changed;
}

bool getTC05MMASchedule(scf::ForOp &forOp, int numStages,
                        tt::PipeliningOption &options) {
  tt::CoarseSchedule coarseSchedule(numStages);
  tt::CoarseSchedule::Cluster cluster = coarseSchedule.clusters.newAtFront();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (op.hasAttr(kPipelineStageAttrName)) {
      int stage =
          op.getAttrOfType<IntegerAttr>(kPipelineStageAttrName).getInt();
      coarseSchedule.insert(&op, stage, cluster);
    }
  }

  auto scheduleDependencies = [&]() {
    bool fixedPoint = false;
    while (!fixedPoint) {
      fixedPoint = true;
      // Schedule upstream dependencies
      for (int stage = 0; stage < numStages; stage++) {
        for (auto &op : forOp.getBody()->without_terminator()) {
          if (coarseSchedule.count(&op) && coarseSchedule[&op].first == stage) {
            bool changed = coarseSchedule.insertDepsOfOp(&op, stage, cluster,
                                                         /*includeArg=*/false);
            fixedPoint &= !changed;
          }
        }
      }
      // Schedule downstream dependencies
      for (int stage = numStages - 1; stage >= 0; stage--) {
        for (auto &op : forOp.getBody()->without_terminator()) {
          if (coarseSchedule.count(&op) && coarseSchedule[&op].first == stage) {
            bool changed = insertUsersOfOp(coarseSchedule, &op, stage, cluster);
            fixedPoint &= !changed;
          }
        }
      }
    }
  };

  scheduleDependencies();

  // Make sure that async loads are scheduled in the same stage they are used.
  DenseMap<ttg::LocalAllocOp, int> allocToStage;
  DenseMap<ttg::LocalAllocOp, ttng::WaitBarrierOp> allocToBarrierWait;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto barrierWait = dyn_cast<ttng::WaitBarrierOp>(op)) {
      auto localAlloc = findShmemAlloc(barrierWait.getAlloc());
      assert(localAlloc);
      assert(allocToBarrierWait.count(localAlloc) == 0);
      allocToBarrierWait[localAlloc] = barrierWait;
      continue;
    }
    if (!coarseSchedule.count(&op))
      continue;

    auto [stage, cluster] = coarseSchedule[&op];
    for (auto arg : op.getOperands()) {
      auto memDescTy = dyn_cast<ttg::MemDescType>(arg.getType());
      if (!memDescTy)
        continue;

      auto localAlloc = findShmemAlloc(arg);
      if (!localAlloc)
        continue;

      allocToStage[localAlloc] = stage;
    }
  }

  for (auto &op : forOp.getBody()->without_terminator()) {
    Value memDesc;
    Value barrier;
    if (auto copyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      memDesc = copyOp.getResult();
    } else if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
      memDesc = copyOp.getResult();
      barrier = copyOp.getBarrier();
    } else if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
      memDesc = gatherOp.getResult();
      barrier = gatherOp.getBarrier();
    } else if (auto storeOp = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      memDesc = storeOp.getSrc();
    } else if (auto scatterOp = dyn_cast<ttng::AsyncTMAScatterOp>(op)) {
      memDesc = scatterOp.getSrc();
    } else {
      continue;
    }
    auto localAlloc = findShmemAlloc(memDesc);
    assert(localAlloc);
    int stage = allocToStage[localAlloc];
    coarseSchedule.insert(&op, stage, cluster);

    // Schedule any barrier wait in the same stage as well, otherwise we will
    // change the loop distance to the wait.
    if (!barrier)
      continue;
    auto barrierAlloc = findShmemAlloc(barrier);
    assert(barrierAlloc);
    auto waitOp = allocToBarrierWait[barrierAlloc];
    // NOTE: barriers can be grouped onto multiple loads, so schedule into the
    // eariest stage where the result is used. This means we reduce the distance
    // between the tma issue and wait, but it is at least correct.
    coarseSchedule.insertMinimum(waitOp, stage, cluster);
  }

  scheduleDependencies();

  // Schedule everything else to stage 0
  for (auto &op : forOp.getBody()->without_terminator()) {
    op.removeAttr(kPipelineStageAttrName);
    if (coarseSchedule.count(&op) == 0) {
      coarseSchedule.insert(&op, 0, cluster);
    }
  }

  std::vector<std::pair<Operation *, unsigned>> schedule =
      coarseSchedule.createFinalSchedule(forOp);

  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = tt::predicateOp;
  options.supportDynamicLoops = true;

  return true;
}

} // namespace

void mlir::triton::pipelineTC05MMALoops(ModuleOp module, int numStages,
                                        bool disableExpander) {
  SmallVector<scf::ForOp> forOps;
  module->walk([&](scf::ForOp forOp) { forOps.push_back(forOp); });

  for (auto forOp : forOps) {
    FailureOr<scf::ForOp> newForOp =
        preProcessLoopForTC05MMAPipelining(forOp, numStages);
    if (succeeded(newForOp)) {
      (*newForOp)->setAttr(kPipelineAttrName,
                           UnitAttr::get(module.getContext()));
    }
  }
  // Run canonicalization to clean up the short-circuited loop carried values.
  mlir::RewritePatternSet patterns(module.getContext());
  scf::ForOp::getCanonicalizationPatterns(patterns, module.getContext());
  if (applyPatternsGreedily(module, std::move(patterns)).failed()) {
    llvm::errs() << "Failed to canonicalize the module\n";
    return;
  }

  if (!disableExpander) {
    SmallVector<scf::ForOp> loops;
    module->walk([&](scf::ForOp forOp) {
      if (forOp->getAttr(kPipelineAttrName))
        loops.push_back(forOp);
    });

    for (auto forOp : loops) {
      mlir::triton::PipeliningOption options;
      bool foundSchedule = getTC05MMASchedule(forOp, /*numStages=*/2, options);
      assert(foundSchedule && "Failed to find a schedule for TC05MMA");

      IRRewriter rewriter(forOp->getContext());
      rewriter.setInsertionPoint(forOp);
      FailureOr<scf::ForOp> newForOp =
          mlir::triton::pipelineForLoop(rewriter, forOp, options);
    }
  }
}
