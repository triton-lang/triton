#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "mlir/IR/Dominance.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// MMA Pipeline Analysis
//===----------------------------------------------------------------------===//

std::optional<std::pair<ttng::TMEMAllocOp, ttng::TMEMLoadOp>>
ttng::getTMemAllocAndLoad(ttng::MMAv5OpInterface mmaOp) {
  auto acc = mmaOp->getOperand(2).getDefiningOp<ttng::TMEMAllocOp>();
  if (!acc || acc->getParentRegion() != mmaOp->getParentRegion()) {
    return std::nullopt;
  }
  for (auto user : acc->getUsers()) {
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (load->getParentRegion() == mmaOp->getParentRegion()) {
        return std::make_pair(acc, load);
      }
    }
  }
  return std::nullopt;
}

SmallVector<Operation *> ttng::getDirectAccUses(ttng::TMEMLoadOp accDef) {
  SmallVector<Operation *> accUses;
  for (auto user : accDef.getResult().getUsers()) {
    if (!isa<arith::SelectOp>(user) && !isa<scf::YieldOp>(user)) {
      accUses.push_back(user);
    }
  }
  return accUses;
}

//===----------------------------------------------------------------------===//
// getMMAInfo

// Check if the accumulator is being used by the same MMA in the next iteration.
// If so, return the yield argument number that the accumulator is being used
// as. Also, check if accumulator has runtime divergent uses - uses that may not
// be known at the compile time.
static std::optional<int> trackAccChain(scf::ForOp forOp,
                                        ttng::TMEMLoadOp accDef,
                                        ttng::TMEMAllocOp accAlloc,
                                        bool &hasDivergentUses) {
  hasDivergentUses = false;
  struct UseInfo {
    Value value = nullptr;
    std::optional<int> yieldArgNo = std::nullopt;
    bool divergentUse = false;
  };
  SmallVector<UseInfo> queue;
  std::optional<int> yieldArgNo = std::nullopt;
  queue.push_back({accDef.getResult(), std::nullopt, false});
  while (!queue.empty()) {
    UseInfo info = queue.pop_back_val();
    for (auto &use : info.value.getUses()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
        if (yieldOp->getParentOp() == forOp) {
          queue.push_back({forOp.getRegionIterArg(use.getOperandNumber()),
                           use.getOperandNumber(), true}); // divergent use
          continue;
        }
        if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
          queue.push_back({ifOp.getResult(use.getOperandNumber()),
                           info.yieldArgNo, true}); // divergent use
          continue;
        }
        assert(0 && "Unexpected use of accumulator");
      } else if (auto selectOp = dyn_cast<arith::SelectOp>(use.getOwner())) {
        queue.push_back({selectOp.getResult(), info.yieldArgNo, true});
      } else if (use.getOwner() == accAlloc) {
        yieldArgNo = info.yieldArgNo;
      } else {
        // Op other than yield or accAlloc. Mark as divergent use if
        // we had to go through selectOp or ifOp.
        hasDivergentUses = info.divergentUse;
      }
    }
  }
  return yieldArgNo;
}

static std::optional<ttng::MMAInfo::AccOverridePoint>
getAccOverridePointInLoop(scf::ForOp forOp, ttng::TMEMAllocOp accUse,
                          ttng::TMEMLoadOp accDef) {
  ttng::MMAInfo::AccOverridePoint accOverridePoint;
  accOverridePoint.isFlag = false;
  DenseSet<Value> seen;
  Value v = accUse.getSrc();
  if (v == nullptr) {
    // Uninitialized accumulator means unused accumulator
    accOverridePoint.op = accUse;
    return accOverridePoint;
  }
  int dist = 0;
  while (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (!seen.insert(v).second) {
      return std::nullopt;
    }
    assert(blockArg.getOwner() == forOp.getBody());
    auto yieldOp = cast<scf::YieldOp>(blockArg.getOwner()->getTerminator());
    v = yieldOp.getOperand(blockArg.getArgNumber() - 1);
    dist++;
  }
  if (!v.getDefiningOp()) {
    return std::nullopt;
  }
  accOverridePoint.distance = dist;
  bool thenOverrides = false;
  if (auto selectOp = dyn_cast<arith::SelectOp>(v.getDefiningOp())) {
    accOverridePoint.op = selectOp;
    bool trueIsConst =
        (selectOp.getTrueValue().getDefiningOp<arith::ConstantOp>() != nullptr);
    bool falseIsConst =
        (selectOp.getFalseValue().getDefiningOp<arith::ConstantOp>() !=
         nullptr);
    if (trueIsConst && falseIsConst) {
      // Both values are constant, so the select overrides unconditionally
      accOverridePoint.initValue = v;
      return accOverridePoint;
    } else if (trueIsConst) {
      accOverridePoint.initValue = selectOp.getTrueValue();
      thenOverrides = true;
    } else if (falseIsConst) {
      accOverridePoint.initValue = selectOp.getFalseValue();
      thenOverrides = false;
    } else {
      return std::nullopt;
    }
    accOverridePoint.condition = selectOp.getCondition();
    if (!thenOverrides) {
      IRRewriter builder(selectOp);
      Value vTrue = builder.create<arith::ConstantOp>(
          selectOp.getLoc(), builder.getBoolAttr(true));
      accOverridePoint.condition = builder.create<arith::XOrIOp>(
          selectOp.getLoc(), accOverridePoint.condition, vTrue);
    }
  } else if (v.getDefiningOp() != accDef) {
    assert(!isa<scf::IfOp>(v.getDefiningOp()) &&
           "Expected unconditional override op");
    accOverridePoint.op = v.getDefiningOp();
    accOverridePoint.initValue = v;
  } else {
    return std::nullopt;
  }

  return accOverridePoint;
}

static std::optional<ttng::MMAInfo::AccOverridePoint>
getAccUseFlagFalseInLoop(scf::ForOp forOp, Value useAccFlagUse) {
  DenseSet<Value> seen;
  Value v = useAccFlagUse;
  int dist = 0;
  while (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (!seen.insert(v).second) {
      return {};
    }
    assert(blockArg.getOwner() == forOp.getBody());
    auto yieldOp = cast<scf::YieldOp>(blockArg.getOwner()->getTerminator());
    v = yieldOp.getOperand(blockArg.getArgNumber() - 1);
    dist++;
  }
  if (!v.getDefiningOp() || !forOp->isAncestor(v.getDefiningOp())) {
    return std::nullopt;
  }
  assert(v.getType().isInteger(1));

  IRRewriter builder(v.getDefiningOp()->getNextNode());
  ttng::MMAInfo::AccOverridePoint accOverridePoint;
  accOverridePoint.isFlag = true;
  accOverridePoint.distance = dist;
  Location loc = v.getDefiningOp()->getLoc();
  auto vTrue =
      builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true));
  accOverridePoint.op = v.getDefiningOp();
  accOverridePoint.condition = builder.create<arith::XOrIOp>(loc, v, vTrue);

  return accOverridePoint;
}

static std::optional<ttng::MMAInfo::AccOverridePoint>
getAccOverrideOrFlagFalseInLoop(scf::ForOp forOp,
                                ttng::MMAv5OpInterface mmaOp) {
  auto tmemAllocAndLoad = getTMemAllocAndLoad(mmaOp);
  assert(tmemAllocAndLoad.has_value() && "Expected tmem alloc and load");
  auto [accAlloc, accLoad] = tmemAllocAndLoad.value();
  auto accOverridePoint = getAccOverridePointInLoop(forOp, accAlloc, accLoad);

  if (!accOverridePoint.has_value()) {
    auto useAccFlag = mmaOp.useAccumulator();
    accOverridePoint = getAccUseFlagFalseInLoop(forOp, useAccFlag);
  }

  return accOverridePoint;
}

std::optional<ttng::MMAInfo> ttng::getMMAInfo(scf::ForOp forOp,
                                              ttng::MMAv5OpInterface mmaOp,
                                              DominanceInfo &domInfo) {
  auto allocAndLoadOpt = getTMemAllocAndLoad(mmaOp);
  if (!allocAndLoadOpt) {
    return {};
  }
  auto [accAlloc, accLoad] = allocAndLoadOpt.value();

  bool hasDivergentUses = false;
  std::optional<int> yieldArgNo =
      trackAccChain(forOp, accLoad, accAlloc, hasDivergentUses);
  if (hasDivergentUses) {
    // If we can't tell for sure that the value is coming from the mma
    // accumulator, skip.
    return {};
  }
  assert(!yieldArgNo || cast<BlockArgument>(accAlloc.getSrc()).getArgNumber() ==
                            *yieldArgNo + 1);

  std::optional<ttng::MMAInfo::AccOverridePoint> accOverridePoint =
      getAccOverrideOrFlagFalseInLoop(forOp, mmaOp);
  if (accOverridePoint.has_value() && accOverridePoint->distance > 1) {
    // We only support an override up to 1 iteration back.
    return {};
  }

  SmallVector<Operation *> accUses = getDirectAccUses(accLoad);
  DominanceInfo domOpInfo(forOp);
  Operation *newAccLoadInsertPoint =
      findNearestCommonDominator(accUses, domOpInfo);
  // Check pipelining and multi-buffering constraints
  // 1. Really needs multibuffering - if the acc is used unconditionally in
  // the loop, or under different conditions. If we cannot multibuffer in this
  // case, we may as well not pipeline at all, as we will have to wait after
  // the dot in every loop iteration.
  scf::IfOp topLevelIf =
      newAccLoadInsertPoint
          ? dyn_cast<scf::IfOp>(
                forOp.getBody()->findAncestorOpInBlock(*newAccLoadInsertPoint))
          : nullptr;
  bool requiresMultiBuffer = accUses.size() > 0 && !topLevelIf;
  // If we override the acc in the loop, it is generally hard to handle it
  // without multibuffering. We make an exception if it not a physical
  // override of a value, but just setting a flag that acc is not used. In
  // this case we don't need different buffer to store init value.
  requiresMultiBuffer |=
      accOverridePoint.has_value() && !accOverridePoint->isFlag;

  // 2. If the acc is not owerwritten in the loop (by op other than the dot),
  // it cannot be multi-buffered. This is because the overwrite is the only
  // way to initialize next buffer without incurring a copy.
  bool canMultiBuffer = accOverridePoint.has_value() &&
                        !mlir::triton::getDisallowAccMultiBuffer(forOp);
  if (requiresMultiBuffer && !canMultiBuffer) {
    return {};
  }

  return ttng::MMAInfo{.accAlloc = accAlloc,
                       .accLoad = accLoad,
                       .accDef = accOverridePoint,
                       .yieldArgNo = yieldArgNo,
                       .accIsMultiBuffered = canMultiBuffer};
}

bool ttng::mmaHasPipelineableOperands(
    ttng::MMAv5OpInterface mmaOp, scf::ForOp forOp,
    std::function<bool(Operation *)> isLoadPipelineable) {
  // Accumulator alloc must be outside the loop.
  auto tmemAlloc = mmaOp.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!tmemAlloc) {
    return false;
  }
  if (!forOp.isDefinedOutsideOfLoop(tmemAlloc)) {
    return false;
  }
  // Operands of the MMA op must come from the (to be pipelined) load, or
  // from outside the loop.
  auto comesFromLoadOrOutsideLoop = [&](Value v) {
    if (forOp.isDefinedOutsideOfLoop(v)) {
      return true;
    }
    // Do not walk through the Block Arguments.
    if (!v.getDefiningOp()) {
      return false;
    }
    while (isa<ttg::MemDescTransOp>(v.getDefiningOp())) {
      v = v.getDefiningOp()->getOperand(0);
    }
    if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(v.getDefiningOp())) {
      if (!localAlloc.getSrc()) {
        return false;
      }
      if (forOp.isDefinedOutsideOfLoop(localAlloc.getSrc())) {
        return true;
      }
      if (isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(
              localAlloc.getSrc().getDefiningOp())) {
        return isLoadPipelineable(localAlloc.getSrc().getDefiningOp());
      }
    }
    return false;
  };
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(mmaOp.getOperation())) {
    if (!comesFromLoadOrOutsideLoop(dotOp.getA()) ||
        !comesFromLoadOrOutsideLoop(dotOp.getB())) {
      return false;
    }
  }

  // For scaled MMA check if the scales are passed through shared memory, and
  // also coming from load or outside the loop.
  if (auto scaledOp = dyn_cast<ttng::TCGen5MMAScaledOp>(mmaOp.getOperation())) {
    if (!isa<ttg::SharedEncodingTrait>(
            scaledOp.getAScale().getType().getEncoding()) &&
            !forOp.isDefinedOutsideOfLoop(scaledOp.getAScale()) ||
        !isa<ttg::SharedEncodingTrait>(
            scaledOp.getBScale().getType().getEncoding()) &&
            !forOp.isDefinedOutsideOfLoop(scaledOp.getBScale()))
      return false;
    if (!comesFromLoadOrOutsideLoop(scaledOp.getAScale()) ||
        !comesFromLoadOrOutsideLoop(scaledOp.getBScale()))
      return false;
  }
  return true;
}

bool ttng::hasAccReadModifyWrite(ttng::MMAv5OpInterface mma, scf::ForOp forOp) {
  auto tmemAlloc = mma.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!tmemAlloc || !forOp.isDefinedOutsideOfLoop(tmemAlloc)) {
    // Alloc not hoisted, or IR is not canonicalized. Pessimistically assume
    // the accumulator is read-modify-written.
    return true;
  }
  SmallVector<Operation *> stores;
  SmallVector<Operation *> loads;
  for (auto user : tmemAlloc->getUsers()) {
    if (isa<ttng::TMEMStoreOp>(user) &&
        forOp->isAncestor(user->getParentOp())) {
      stores.push_back(cast<ttng::TMEMStoreOp>(user));
    }
    if (isa<ttng::TMEMLoadOp>(user) && forOp->isAncestor(user->getParentOp())) {
      loads.push_back(cast<ttng::TMEMLoadOp>(user));
    }
  }
  if (stores.empty() || loads.empty()) {
    return false;
  }
  SmallVector<Value> readValues;
  DenseSet<Value> seen;
  llvm::SetVector<Value> modifiedValues;
  for (auto load : loads) {
    readValues.push_back(load->getResult(0));
  }
  while (!readValues.empty()) {
    Value v = readValues.pop_back_val();
    if (!seen.insert(v).second) {
      continue;
    }
    for (auto &use : v.getUses()) {
      if (llvm::is_contained(stores, use.getOwner())) {
        continue; // R-W, not midified, this is safe
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
        if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
          readValues.push_back(ifOp.getResult(use.getOperandNumber()));
        }
        if (forOp == yieldOp->getParentOp()) {
          readValues.push_back(forOp.getRegionIterArg(use.getOperandNumber()));
        }
      } else {
        modifiedValues.insert(use.getOwner()->getResults().begin(),
                              use.getOwner()->getResults().end());
      }
    }
  }
  while (!modifiedValues.empty()) {
    Value v = modifiedValues.pop_back_val();
    if (!seen.insert(v).second) {
      continue;
    }
    for (auto &use : v.getUses()) {
      if (llvm::is_contained(stores, use.getOwner())) {
        return true; // RMW!
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
        if (auto ifOp = dyn_cast<scf::IfOp>(yieldOp->getParentOp())) {
          modifiedValues.insert(ifOp.getResult(use.getOperandNumber()));
        }
        if (forOp == yieldOp->getParentOp()) {
          modifiedValues.insert(forOp.getRegionIterArg(use.getOperandNumber()));
        }
      } else {
        modifiedValues.insert(use.getOwner()->getResults().begin(),
                              use.getOwner()->getResults().end());
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// MMA Pipeline Rewriters
//===----------------------------------------------------------------------===//

ttng::TMEMAllocOp ttng::createTMemAlloc(OpBuilder &builder,
                                        ttng::TMEMAllocOp oldTMemAllocOp,
                                        bool multiBufferred, int numStages) {
  Location loc = oldTMemAllocOp.getLoc();
  auto oldRetType = oldTMemAllocOp.getType();
  SmallVector<int64_t> shape = {oldRetType.getShape().begin(),
                                oldRetType.getShape().end()};
  if (multiBufferred) {
    shape.insert(shape.begin(), numStages);
  }
  Type accMemDescType = triton::gpu::MemDescType::get(
      shape, oldRetType.getElementType(), oldRetType.getEncoding(),
      oldRetType.getMemorySpace(), /*mutableMemory=*/true);
  return builder.create<ttng::TMEMAllocOp>(oldTMemAllocOp.getLoc(),
                                           accMemDescType, nullptr);
}

void ttng::createInitStore(OpBuilder &builder, ttng::TMEMAllocOp allocOp,
                           Value initVal, bool multiBufferred) {
  Value bufferSlice = allocOp;
  if (multiBufferred) {
    bufferSlice = triton::createSingleBufferView(builder, allocOp, 0);
  }
  Value vTrue = builder.create<arith::ConstantIntOp>(allocOp.getLoc(), 1, 1);
  builder.create<ttng::TMEMStoreOp>(allocOp.getLoc(), bufferSlice, initVal,
                                    vTrue);
}
