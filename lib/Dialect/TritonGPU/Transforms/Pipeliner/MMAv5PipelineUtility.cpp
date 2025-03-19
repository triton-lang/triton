#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "mlir/IR/Dominance.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

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
