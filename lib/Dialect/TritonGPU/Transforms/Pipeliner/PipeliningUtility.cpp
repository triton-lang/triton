#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Casting.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

bool mlir::triton::loopHasDistGreaterThanOne(scf::ForOp forOp) {
  return llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                      [](Value operand) {
                        Operation *def = operand.getDefiningOp();
                        return !def;
                      });
}

bool mlir::triton::isOuterLoop(scf::ForOp forOp) {
  return llvm::any_of(forOp.getBody()->getOperations(), [](Operation &op) {
    return isa<scf::ForOp, scf::WhileOp>(op);
  });
}

// Combine the current mask with the given predicate.
static Value getPredMask(RewriterBase &rewriter, Type typeLike,
                         Value currentMask, Value pred) {
  Type maskType = tt::getI1SameShape(typeLike);
  Location loc = pred.getLoc();
  Value mask = pred;
  if (isa<RankedTensorType>(maskType)) {
    mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);
  }
  if (currentMask) {
    mask = rewriter.create<arith::AndIOp>(loc, mask, currentMask);
  }
  return mask;
}

// Function to mask operations during scheduling.
Operation *mlir::triton::predicateOp(RewriterBase &rewriter, Operation *op,
                                     Value pred) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (mlir::isMemoryEffectFree(op))
    return op;
  if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op))
    return op;
  if (isa<ttg::LocalLoadOp, ttg::LocalStoreOp>(op))
    return op;
  if (isa<ttng::TMEMAllocOp>(op))
    return op;
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    rewriter.setInsertionPoint(op);
    Value cnd = getPredMask(rewriter, ifOp.getCondition().getType(),
                            ifOp.getCondition(), pred);
    ifOp.getConditionMutable().assign(cnd);
    return op;
  }
  if (auto asyncCopyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
    rewriter.setInsertionPoint(asyncCopyOp);
    Value mask = getPredMask(rewriter, asyncCopyOp.getSrc().getType(),
                             asyncCopyOp.getMask(), pred);
    asyncCopyOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    rewriter.setInsertionPoint(loadOp);
    Value mask = getPredMask(rewriter, loadOp.getPtr().getType(),
                             loadOp.getMask(), pred);
    loadOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto copyOp = dyn_cast<ttng::AsyncTMACopyGlobalToLocalOp>(op)) {
    rewriter.setInsertionPoint(copyOp);
    Value mask = getPredMask(rewriter, copyOp.getPred().getType(),
                             copyOp.getPred(), pred);
    copyOp.getPredMutable().assign(mask);
    return op;
  }
  if (auto gatherOp = dyn_cast<ttng::AsyncTMAGatherOp>(op)) {
    rewriter.setInsertionPoint(gatherOp);
    Value mask = getPredMask(rewriter, gatherOp.getPred().getType(),
                             gatherOp.getPred(), pred);
    gatherOp.getPredMutable().assign(mask);
    return op;
  }
  if (auto expectOp = dyn_cast<ttng::BarrierExpectOp>(op)) {
    rewriter.setInsertionPoint(expectOp);
    Value mask = getPredMask(rewriter, expectOp.getPred().getType(),
                             expectOp.getPred(), pred);
    expectOp.getPredMutable().assign(mask);
    return op;
  }
  if (auto mmav5Op = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    rewriter.setInsertionPoint(mmav5Op);
    auto currPred = mmav5Op.getPredicate();
    Value mask = getPredMask(rewriter, currPred.getType(), currPred, pred);
    mmav5Op.setPredicate(mask);
    return op;
  }
  if (auto tmemStoreOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
    rewriter.setInsertionPoint(tmemStoreOp);
    Value mask = getPredMask(rewriter, tmemStoreOp.getPred().getType(),
                             tmemStoreOp.getPred(), pred);
    tmemStoreOp.getPredMutable().assign(mask);
    return op;
  }
  if (auto waitBarrier = dyn_cast<ttng::WaitBarrierOp>(op)) {
    rewriter.setInsertionPoint(waitBarrier);
    Value mask = pred;
    Value currentPred = waitBarrier.getPred();
    if (currentPred) {
      mask = getPredMask(rewriter, currentPred.getType(), currentPred, pred);
    }
    waitBarrier.getPredMutable().assign(mask);
    return op;
  }
  if (auto storeOp = dyn_cast<tt::StoreOp>(op)) {
    rewriter.setInsertionPoint(storeOp);
    Value mask = getPredMask(rewriter, storeOp.getPtr().getType(),
                             storeOp.getMask(), pred);
    storeOp.getMaskMutable().assign(mask);
    return op;
  }
  if (auto atomicRMWOp = dyn_cast<tt::AtomicRMWOp>(op)) {
    rewriter.setInsertionPoint(atomicRMWOp);
    Value mask = getPredMask(rewriter, atomicRMWOp.getPtr().getType(),
                             atomicRMWOp.getMask(), pred);
    atomicRMWOp.getMaskMutable().assign(mask);
    return op;
  }

  op->emitError("pipeliner doesn't know how to predicate this op.");
  llvm::report_fatal_error("Fatal pipeliner error");
  return op;
}

void mlir::triton::replaceUsesAndPropagateType(OpBuilder &builder,
                                               Operation *oldUse, Value val) {
  SmallVector<Operation *> opsToDelete;
  SmallVector<OpOperand *> operandsToReplace;

  // Save the operand to replace / delete later (avoid iterator invalidation).
  // TODO: can we use an early_inc iterator?
  for (OpOperand &use : oldUse->getUses()) {
    // Non-subview/trans ops will be replaced by `val`.
    if (!isa<triton::gpu::MemDescTransOp, triton::gpu::MemDescSubviewOp>(
            use.getOwner())) {
      operandsToReplace.push_back(&use);
      continue;
    }
    Operation *user = use.getOwner();
    // `subview(old_op)` is replaced by a new `subview(val)`.
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPoint(user);
    Value newVal;
    if (auto subview = dyn_cast<triton::gpu::MemDescSubviewOp>(user)) {
      triton::gpu::MemDescType oldType = subview.getType();
      bool isMutable =
          cast<triton::gpu::MemDescType>(val.getType()).getMutableMemory();
      Type newDstType = triton::gpu::MemDescType::get(
          oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
          oldType.getMemorySpace(), isMutable);
      newVal = builder.create<triton::gpu::MemDescSubviewOp>(
          subview.getLoc(), newDstType, val, subview.getOffsets());
      newVal.getDefiningOp()->setAttrs(user->getAttrs());
    } else if (auto trans = dyn_cast<triton::gpu::MemDescTransOp>(user)) {
      newVal = builder.create<triton::gpu::MemDescTransOp>(trans.getLoc(), val,
                                                           trans.getOrder());
      newVal.getDefiningOp()->setAttrs(user->getAttrs());
    }
    assert(newVal);
    replaceUsesAndPropagateType(builder, user, newVal);
    opsToDelete.push_back(use.getOwner());
  }

  // Perform late replacement.
  for (OpOperand *operand : operandsToReplace) {
    Operation *op = operand->getOwner();
    operand->set(val);
  }

  // Perform late op erasure.
  for (Operation *op : opsToDelete)
    op->erase();
}

// Return true if the given ForOp has the attribute
// `tt.disallow_acc_multi_buffer` set to true.
bool mlir::triton::getDisallowAccMultiBuffer(scf::ForOp forOp) {
  return forOp->hasAttr(mlir::triton::kDisallowAccMultiBufferAttrName);
}

void mlir::triton::visitNestedOperands(Operation *op,
                                       function_ref<void(Value)> visitor) {
  op->walk([&](Operation *nestedOp) {
    for (Value operand : nestedOp->getOperands()) {
      if (operand.getParentBlock()->getParentOp()->isProperAncestor(op))
        visitor(operand);
    }
  });
}

SetVector<Value> mlir::triton::getNestedOperands(Operation *op) {
  SetVector<Value> result;
  visitNestedOperands(op, [&](Value operand) { result.insert(operand); });
  return result;
}

int mlir::triton::getCopyVecBytes(RankedTensorType registerTy,
                                  ttg::SharedEncodingTrait sharedEnc) {
  auto regLayout = triton::gpu::toLinearLayout(registerTy.getShape(),
                                               registerTy.getEncoding());
  auto sharedLayout =
      triton::gpu::toLinearLayout(registerTy.getShape(), sharedEnc);
  auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);
  const int vecElems = regToSharedLayout.getNumConsecutiveInOut();
  return vecElems * registerTy.getElementTypeBitWidth() / 8;
}

void mlir::triton::serializeLatencies(ModuleOp module,
                                      DenseMap<Operation *, int> &opLatency) {
  for (auto &[op, latency] : opLatency) {
    op->setAttr(
        kLatencyAttrName,
        IntegerAttr::get(IntegerType::get(module.getContext(), 32), latency));
  }
}

DenseMap<Operation *, int> mlir::triton::deserializeLatencies(ModuleOp module) {
  DenseMap<Operation *, int> opLatency;
  module.walk([&](Operation *op) {
    if (op->hasAttr(kLatencyAttrName)) {
      opLatency[op] = op->getAttrOfType<IntegerAttr>(kLatencyAttrName).getInt();
      op->removeAttr(kLatencyAttrName);
    }
  });
  return opLatency;
}

// Create an allocation and init the mbarriers.
Value mlir::triton::createBarrierAlloc(scf::ForOp forOp, int numBarriers) {
  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
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
  ttg::MemDescType barrierMemDescType = ttg::MemDescType::get(
      {numBarriers}, rewriter.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  Value barrierAlloc =
      rewriter.create<ttg::LocalAllocOp>(loc, barrierMemDescType, Value());
  for (unsigned i = 0; i < numBarriers; i++) {
    Value barrierView = createSingleBufferView(rewriter, barrierAlloc, i);
    rewriter.create<ttng::InitBarrierOp>(forOp->getLoc(), barrierView, 1);
  }
  return barrierAlloc;
}

void mlir::triton::createBarrierDealloc(scf::ForOp forOp, Value barrierAlloc,
                                        int numBarriers) {
  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPointAfter(forOp);
  for (unsigned i = 0; i < numBarriers; i++) {
    Value barrierView = createSingleBufferView(rewriter, barrierAlloc, i);
    rewriter.create<ttng::InvalBarrierOp>(forOp->getLoc(), barrierView);
  }
  rewriter.create<ttg::LocalDeallocOp>(forOp->getLoc(), barrierAlloc);
}

Value mlir::triton::createSingleBufferView(OpBuilder &builder, Value alloc,
                                           Value idx) {
  assert(isa<triton::gpu::MemDescType>(alloc.getType()) &&
         "Expected MemDescType");
  auto allocDescType = cast<triton::gpu::MemDescType>(alloc.getType());
  SmallVector<int64_t> shape;
  if (allocDescType.getShape().size() > 1) {
    shape.insert(shape.end(), allocDescType.getShape().begin() + 1,
                 allocDescType.getShape().end());
  } else {
    shape.push_back(1);
  }
  auto viewDescType = triton::gpu::MemDescType::get(
      shape, allocDescType.getElementType(), allocDescType.getEncoding(),
      allocDescType.getMemorySpace(), allocDescType.getMutableMemory(),
      /*allocShape=*/allocDescType.getAllocShape());
  SmallVector<Value> idxs = {idx};
  if (allocDescType.getShape().size() > 1) {
    Value zero =
        builder.template create<arith::ConstantIntOp>(alloc.getLoc(), 0, 32);
    for (unsigned i = 1; i < allocDescType.getShape().size(); i++) {
      idxs.push_back(zero);
    }
  }
  return builder.template create<triton::gpu::MemDescSubviewOp>(
      alloc.getLoc(), viewDescType, alloc, idxs);
}

Value mlir::triton::createSingleBufferView(OpBuilder &builder, Value alloc,
                                           int idx) {
  return mlir::triton::createSingleBufferView(
      builder, alloc,
      builder.create<arith::ConstantIntOp>(alloc.getLoc(), idx, 32));
}

bool mlir::triton::mmaHasPipelineableOperands(
    Operation *op, scf::ForOp forOp,
    std::function<bool(Operation *)> isLoadPipelineable) {
  assert((isa<ttng::MMAv5OpInterface>(op)) && "Only MMA ops are supported");
  // Accumulator alloc must be outside the loop.
  auto mmaOp = cast<ttng::MMAv5OpInterface>(op);
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
    if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(v.getDefiningOp())) {
      if (!localAlloc.getSrc()) {
        return false;
      }
      if (forOp.isDefinedOutsideOfLoop(localAlloc.getSrc())) {
        return true;
      }
      if (auto loadOp =
              dyn_cast<tt::LoadOp>(localAlloc.getSrc().getDefiningOp())) {
        return isLoadPipelineable(loadOp);
      }
    }
    return false;
  };
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(op)) {
    if (!comesFromLoadOrOutsideLoop(dotOp.getA()) ||
        !comesFromLoadOrOutsideLoop(dotOp.getB())) {
      return false;
    }
  }

  // For scaled MMA check if the scales are passed through shared memory, and
  // also coming from load or outside the loop.
  if (auto scaledOp = dyn_cast<ttng::TCGen5MMAScaledOp>(op)) {
    if (!isa<ttg::SharedEncodingTrait>(
            scaledOp.getAScale().getType().getEncoding()) ||
        !isa<ttg::SharedEncodingTrait>(
            scaledOp.getBScale().getType().getEncoding()))
      return false;
    if (!comesFromLoadOrOutsideLoop(scaledOp.getAScale()) ||
        !comesFromLoadOrOutsideLoop(scaledOp.getBScale()))
      return false;
  }
  return true;
}

// Return true if the accumulator of an mma in subsequent iterations is either
// independent from the previous iteration (overwritten) or completely reused,
// without read-modify-write.
// Otherwise, we can not pipeline the MMA, as we need to insert a wait after the
// mma to read back the accumulator for RMW.
bool mlir::triton::hasAccReadModifyWrite(Operation *op, scf::ForOp forOp) {
  auto mma = cast<ttng::MMAv5OpInterface>(op);
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
  llvm::SetVector<Value> modifiedValues;
  for (auto load : loads) {
    readValues.push_back(load->getResult(0));
  }
  while (!readValues.empty()) {
    Value v = readValues.pop_back_val();
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
