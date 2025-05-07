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

bool ttng::MMAv5PipelineableOperandsHelper::comesFromLoadOrOutsideLoop(
    Value v, Operation *&foundLoad) {
  if (forOp.isDefinedOutsideOfLoop(v)) {
    return true;
  }
  if (!v.getDefiningOp()) {
    return false;
  }
  while (isa<ttg::MemDescTransOp, ttg::MemDescReshapeOp>(v.getDefiningOp())) {
    v = v.getDefiningOp()->getOperand(0);
  }
  auto localAlloc = dyn_cast<ttg::LocalAllocOp>(v.getDefiningOp());
  if (!localAlloc) {
    return false;
  }
  if (!localAlloc.getSrc()) {
    return false;
  }
  if (forOp.isDefinedOutsideOfLoop(localAlloc.getSrc())) {
    return true;
  }
  auto localAllocSrc = localAlloc.getSrc().getDefiningOp();
  if (!isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(
          localAllocSrc)) {
    return false;
  }
  foundLoad = localAllocSrc;
  if (!isLoadToBePipelined(foundLoad)) {
    return false;
  }
  if (canBeAsyncLoad(foundLoad)) {
    return true;
  }
  return false;
}

void ttng::MMAv5PipelineableOperandsHelper::run() {
  isOperandsStateDetermined = true;
  // Accumulator alloc must be outside the loop.
  auto tmemAlloc = mmaOp.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!tmemAlloc) {
    return;
  }
  if (!forOp.isDefinedOutsideOfLoop(tmemAlloc)) {
    return;
  }
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(mmaOp.getOperation())) {
    Operation *foundLoad = nullptr;
    if (!comesFromLoadOrOutsideLoop(dotOp.getA(), foundLoad)) {
      if (foundLoad) {
        unpipelineableOperandLoads.push_back(foundLoad);
      } else {
        isOperandsStateDetermined = false;
      }
    }
    if (!comesFromLoadOrOutsideLoop(dotOp.getB(), foundLoad)) {
      if (foundLoad) {
        unpipelineableOperandLoads.push_back(foundLoad);
      } else {
        isOperandsStateDetermined = false;
      }
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
            !forOp.isDefinedOutsideOfLoop(scaledOp.getBScale())) {
      // Undecidable, we could follow the tmem use-def chain to find the first
      // tmem_load.
      isOperandsStateDetermined = false;
      return;
    }
    Operation *foundLoad = nullptr;
    if (!comesFromLoadOrOutsideLoop(scaledOp.getAScale(), foundLoad)) {
      if (foundLoad) {
        unpipelineableOperandLoads.push_back(foundLoad);
      } else {
        isOperandsStateDetermined = false;
      }
    }
    if (!comesFromLoadOrOutsideLoop(scaledOp.getBScale(), foundLoad)) {
      if (foundLoad) {
        unpipelineableOperandLoads.push_back(foundLoad);
      } else {
        isOperandsStateDetermined = false;
      }
    }
  }
  isPipelineable =
      isOperandsStateDetermined && unpipelineableOperandLoads.empty();
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

bool ttng::requiresAccMultiBuffering(ttng::MMAv5OpInterface mma,
                                     scf::ForOp forOp) {
  auto tmemAlloc = mma.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!tmemAlloc || !forOp.isDefinedOutsideOfLoop(tmemAlloc)) {
    return true; // Pessimistically assume the accumulator requires
                 // multi-buffering.
  }
  // If the accumulator is being read in the loop, we will need to multibuffer
  // when pipelining.
  for (auto user : tmemAlloc->getUsers()) {
    if (isa<ttng::TMEMLoadOp>(user) && forOp->isAncestor(user->getParentOp())) {
      return true;
    }
  }
  return false;
}

bool ttng::hasLoadsAfterMMA(ttng::MMAv5OpInterface mma, scf::ForOp forOp) {
  auto tmemAlloc = mma.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!tmemAlloc || !forOp.isDefinedOutsideOfLoop(tmemAlloc)) {
    return false;
  }
  for (auto user : tmemAlloc->getUsers()) {
    if (isa<ttng::TMEMLoadOp>(user)) {
      auto ancestorOp = forOp.getBody()->findAncestorOpInBlock(*user);
      if (ancestorOp && mma->isBeforeInBlock(ancestorOp)) {
        return true;
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
  return builder.create<ttng::TMEMAllocOp>(
      oldTMemAllocOp.getLoc(), accMemDescType,
      builder.getType<gpu::AsyncTokenType>(), /*src=*/Value());
}
