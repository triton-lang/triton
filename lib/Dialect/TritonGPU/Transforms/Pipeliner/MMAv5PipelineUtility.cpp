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
  auto acc = mmaOp.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
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

static bool accUseFlagSetToFalse(ttng::MMAv5OpInterface mma, scf::ForOp forOp) {
  Value accUseFlag = mma.useAccumulator();
  if (matchPattern(accUseFlag, m_Zero())) {
    return true;
  }
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  while (auto blockArg = dyn_cast<BlockArgument>(accUseFlag)) {
    accUseFlag = yieldOp.getOperand(blockArg.getArgNumber() - 1);
  }
  // If the accUseFlag is overwritten in the loop, we treat it as a 'false'
  // with condition being ~accUseFlag.
  return accUseFlag.getDefiningOp() &&
         forOp->isAncestor(accUseFlag.getDefiningOp());
}

static bool accOverwrittenInLoop(ttng::MMAv5OpInterface mma, scf::ForOp forOp) {
  auto tmemAlloc = mma.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!tmemAlloc || !forOp.isDefinedOutsideOfLoop(tmemAlloc)) {
    return false;
  }
  for (auto user : tmemAlloc->getUsers()) {
    if (isa<ttng::TMEMStoreOp>(user) &&
        forOp->isAncestor(user->getParentOp())) {
      return true;
    }
  }
  return false;
}

bool ttng::isAccMultibufferingPossible(ttng::MMAv5OpInterface mma,
                                       scf::ForOp forOp) {
  // If the accumulator is never overwritten in the loop, we can't multibuffer
  // it, as the overwrite point is the only place where we can swap the
  // buffer.
  return accUseFlagSetToFalse(mma, forOp) || accOverwrittenInLoop(mma, forOp);
}

bool ttng::mmav5DominatesTmemLoads(
    scf::ForOp forOp, function_ref<bool(MMAv5OpInterface)> isMmaPipelineable) {
  DominanceInfo domInfo(forOp);
  WalkResult result = forOp.walk([&](ttng::MMAv5OpInterface mma) {
    auto tmemAlloc = mma.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
    if (!tmemAlloc || !forOp.isDefinedOutsideOfLoop(tmemAlloc)) {
      return WalkResult::advance();
    }
    for (auto user : tmemAlloc->getUsers()) {
      if (isa<ttng::TMEMLoadOp>(user) && forOp->isAncestor(user) &&
          !domInfo.properlyDominates(mma, user) && isMmaPipelineable(mma)) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
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
