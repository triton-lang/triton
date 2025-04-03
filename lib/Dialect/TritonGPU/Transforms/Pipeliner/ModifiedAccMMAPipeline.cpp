#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

ttng::TMEMAllocOp createTMemAlloc(IRRewriter &rewriter,
                                  ttng::TMEMAllocOp oldTMemAllocOp) {
  Location loc = oldTMemAllocOp.getLoc();
  auto oldRetType = oldTMemAllocOp.getType();
  SmallVector<int64_t> shape = {oldRetType.getShape().begin(),
                                oldRetType.getShape().end()};
  Type accMemDescType = triton::gpu::MemDescType::get(
      shape, oldRetType.getElementType(), oldRetType.getEncoding(),
      oldRetType.getMemorySpace(), /*mutableMemory=*/true);
  return rewriter.create<ttng::TMEMAllocOp>(oldTMemAllocOp.getLoc(),
                                            accMemDescType, nullptr);
}

Value createBarrierAlloc(IRRewriter &rewriter, scf::ForOp forOp) {
  MLIRContext *ctx = forOp.getContext();
  Location loc = forOp.getLoc();
  unsigned numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(
      forOp->getParentOfType<ModuleOp>());
  Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
  auto barrierCTALayout = ttg::CTALayoutAttr::get(
      /*context=*/ctx, /*CTAsPerCGA=*/{numCTAs},
      /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding =
      ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCTALayout);
  ttg::MemDescType barrierMemDescType =
      ttg::MemDescType::get({1}, rewriter.getI64Type(), barrierEncoding,
                            sharedMemorySpace, /*mutableMemory=*/true);
  Value barrierAlloc =
      rewriter.create<ttg::LocalAllocOp>(loc, barrierMemDescType, Value());
  rewriter.create<ttng::InitBarrierOp>(forOp->getLoc(), barrierAlloc, 1);
  return barrierAlloc;
}

scf::ForOp pipelineDot(scf::ForOp forOp, ttng::TCGen5MMAOp dotOp,
                       ttng::TMEMLoadOp loadOp, ttng::TMEMStoreOp storeOp,
                       ttng::TMEMAllocOp allocOp, Operation *accModOp,
                       int yieldArgNo) {
  bool tmemHoisted = forOp.isDefinedOutsideOfLoop(allocOp.getResult());
  if (tmemHoisted) {
    assert(storeOp);
  }
  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  Value vTrue = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  Value accInitValue = forOp.getInitArgs()[yieldArgNo];
  auto newAlloc = tmemHoisted ? allocOp : createTMemAlloc(rewriter, allocOp);
  newAlloc.getSrcMutable().assign(accInitValue);
  Value barrier = createBarrierAlloc(rewriter, forOp);
  Value phase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  Value notZerothIter = rewriter.create<arith::ConstantIntOp>(loc, 0, 1);
  Type loadTy = loadOp.getType();
  (void)addIterArgsToLoop(rewriter, forOp, {phase, notZerothIter});
  phase = forOp.getRegionIterArg(forOp.getNumRegionIterArgs() - 2);
  notZerothIter = forOp.getRegionIterArg(forOp.getNumRegionIterArgs() - 1);
  Value oldAccValue = forOp.getRegionIterArg(yieldArgNo);

  rewriter.setInsertionPoint(accModOp ? accModOp : dotOp);
  loc = dotOp.getLoc();
  rewriter.create<ttng::WaitBarrierOp>(loc, barrier, phase, notZerothIter);
  auto flag = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI32Type(),
                                              notZerothIter);
  phase = rewriter.create<arith::XOrIOp>(loc, phase, flag);
  if (accModOp) {
    // Update the source of the modifying op
    loc = accModOp->getLoc();
    auto accValue = rewriter.create<ttng::TMEMLoadOp>(loc, loadTy, newAlloc);
    accModOp->replaceUsesOfWith(oldAccValue, accValue);
    rewriter.setInsertionPointAfter(accModOp);
    rewriter.create<ttng::TMEMStoreOp>(loc, newAlloc, accModOp->getResult(0),
                                       vTrue);
  }

  // Update the dot op
  dotOp.getDMutable().assign(newAlloc);
  dotOp.getBarrierMutable().assign(barrier);

  // Update the yield
  appendToForOpYield(forOp, {phase, vTrue});

  rewriter.setInsertionPointAfter(forOp);
  // Short-circuit the loop carry value that was holding the accumulator value,
  // removing the last reference to the loaded accumulator.
  forOp.getBody()->getTerminator()->setOperand(yieldArgNo, oldAccValue);
  // Update the uses outside the loop
  phase = forOp.getResult(forOp.getNumResults() - 2);
  notZerothIter = forOp.getResult(forOp.getNumResults() - 1);
  rewriter.create<ttng::WaitBarrierOp>(dotOp.getLoc(), barrier, phase,
                                       notZerothIter);
  auto afterLoopLoad =
      rewriter.create<ttng::TMEMLoadOp>(forOp.getLoc(), loadTy, newAlloc);
  forOp->getResult(yieldArgNo).replaceAllUsesWith(afterLoopLoad.getResult());

  rewriter.create<ttng::InvalBarrierOp>(dotOp.getLoc(), barrier);
  rewriter.create<ttg::LocalDeallocOp>(dotOp.getLoc(), barrier);

  // Remove the old alloc and load
  if (storeOp) {
    storeOp.erase();
  }
  loadOp.erase();
  if (!tmemHoisted) {
    allocOp.erase();
  }

  return forOp;
}

// TODO: Integrate with the new pipeliner
scf::ForOp mlir::triton::pipelineMMAWithScaledAcc(scf::ForOp forOp) {
  // Look for chained mmas for which the tmem access is not pipelined yet, with
  // an operation modifying the acc value before the mma.
  SmallVector<Operation *> dotOps;
  forOp.walk([&](ttng::TCGen5MMAOp mmaOp) {
    if (mmaOp->getBlock() != forOp.getBody()) {
      return;
    }
    dotOps.push_back(mmaOp);
  });

  dotOps = getMMAsWithMultiBufferredOperands(forOp, dotOps);

  for (auto op : dotOps) {
    auto dotOp = llvm::cast<ttng::TCGen5MMAOp>(op);
    auto tmemAlloc = dotOp.getD().getDefiningOp<ttng::TMEMAllocOp>();
    if (!tmemAlloc) {
      continue;
    }
    if (tmemAlloc->getParentOp() == forOp && tmemAlloc.getSrc() == nullptr) {
      continue;
    }
    ttng::TMEMLoadOp tmemLoad = nullptr;
    ttng::TMEMStoreOp tmemStore = nullptr;
    for (auto user : tmemAlloc.getResult().getUsers()) {
      if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
        if (load->getBlock() == dotOp->getBlock()) {
          if (tmemLoad) {
            // Multiple loads in the loop, bail out
            return forOp;
          }
          tmemLoad = load;
        }
      }
      if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
        if (store->getBlock() == dotOp->getBlock()) {
          if (tmemStore) {
            // Multiple stores in the loop, bail out
            return forOp;
          }
          tmemStore = store;
        }
      }
    }
    if (!tmemLoad || !tmemLoad.getResult().hasOneUse()) {
      continue;
    }
    if (tmemStore) {
      // If there is a store in the loop, alloc should be already hoisted
      assert(forOp.isDefinedOutsideOfLoop(tmemAlloc));
    }
    OpOperand &tmemLoadUse = *tmemLoad.getResult().getUses().begin();
    auto yieldOp = dyn_cast<scf::YieldOp>(tmemLoadUse.getOwner());
    if (!yieldOp || yieldOp->getParentOfType<scf::ForOp>() != forOp) {
      continue;
    }
    int yieldArgNo = tmemLoadUse.getOperandNumber();
    if (!forOp.getRegionIterArg(yieldArgNo).hasOneUse()) {
      continue;
    }
    Operation *accModOp =
        *forOp.getRegionIterArg(yieldArgNo).getUsers().begin();
    if (accModOp == tmemAlloc) {
      accModOp = nullptr; // not really an acc modification
    } else {
      if (!accModOp->hasOneUse() ||
          (*accModOp->getUsers().begin() != tmemAlloc &&
           *accModOp->getUsers().begin() != tmemStore)) {
        continue;
      }
    }

    forOp = pipelineDot(forOp, dotOp, tmemLoad, tmemStore, tmemAlloc, accModOp,
                        yieldArgNo);
  }
  return forOp;
};
