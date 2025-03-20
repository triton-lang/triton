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
  if (isa<ttng::TMEMAllocOp, ttng::TMEMCopyOp>(op))
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
    newVal.getDefiningOp()->setAttrs(user->getAttrs());
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
