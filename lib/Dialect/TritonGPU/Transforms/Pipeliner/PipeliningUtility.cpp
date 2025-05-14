#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/Support/Casting.h"
#include <queue>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

//===----------------------------------------------------------------------===//
// Hoisting Utilities
//===----------------------------------------------------------------------===//

bool triton::isPureScalarOp(Operation *op) {
  auto isScalar = [](Type type) { return type.isIntOrIndexOrFloat(); };
  return isPure(op) && llvm::all_of(op->getOperandTypes(), isScalar) &&
         llvm::all_of(op->getResultTypes(), isScalar);
}

bool triton::getDominatingValueSetOpsToHoist(
    DominanceInfo &domInfo, Operation *refOp, ArrayRef<Value> valueSet,
    llvm::SetVector<Operation *> &toHoist,
    function_ref<bool(Operation *)> canHoist) {
  // The set of operations below `refOp` that are being checked if they can be
  // hoisted. This set prevents checking operations twice but also if the
  // computation can be hoisted, this becomes the set of operations to hoist.
  llvm::SetVector<Operation *> visited;

  // Climb the use-def chain breadth-first so that operations can be hoisted in
  // the reverse visitation order.
  std::queue<Value> queue;
  for (Value value : valueSet)
    queue.push(value);

  while (!queue.empty()) {
    Value value = queue.front();
    queue.pop();

    // If the value properly dominates the outer loop, then it must be invariant
    // to it.
    if (domInfo.properlyDominates(value, refOp))
      continue;
    // If the value is a block argument, it cannot be hoisted.
    if (auto arg = dyn_cast<BlockArgument>(value))
      return false;

    Operation *op = value.getDefiningOp();
    // Check if the op was already visited.
    if (visited.contains(op))
      continue;
    // If the defining op cannot be hoisted, then the value cannot be made loop
    // invariant.
    if (!canHoist(op))
      return false;
    visited.insert(op);
    // Recurse on the operands of the op.
    for (Value operand : op->getOperands())
      queue.push(operand);
  }

  // The operations in `visited` must be hoisted. Note that operations are not
  // added to `toHoist` unless all of `values` can be hoisted. This is to avoid
  // hoisting operations for loops that don't end up getting fused if one of
  // their bounds operands cannot be hoisted.
  toHoist.insert(visited.begin(), visited.end());

  return true;
}

void triton::hoistOpsBefore(Operation *refOp,
                            const llvm::SetVector<Operation *> &toHoist) {
  return hoistOpsBefore(refOp->getBlock(), refOp->getIterator(), toHoist);
}
void triton::hoistOpsBefore(Block *block, Block::iterator it,
                            const llvm::SetVector<Operation *> &toHoist) {
  for (Operation *op : topologicalSort(toHoist)) {
    op->moveBefore(block, it);
  }
}

//===----------------------------------------------------------------------===//
// Sinking Utilities
//===----------------------------------------------------------------------===//

Value triton::sinkValueRedefinition(RewriterBase &rewriter, Value in, Value out,
                                    Block *block) {
  OpBuilder::InsertionGuard guard(rewriter);
  for (; block != in.getParentBlock();
       block = block->getParentOp()->getBlock()) {
    Operation *op = block->getParentOp();
    rewriter.setInsertionPoint(op);

    // `in` is live into the loop body. `out` becomes the live-out if the
    // loop executes at least once.
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      (void)addIterArgsToLoop(rewriter, forOp, in);
      appendToForOpYield(forOp, out);
      out = forOp.getResults().back();
      continue;
    }

    // `in` is live into both branches. `out` becomes the live-out if the
    // particular branch is taken.
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      scf::IfOp newIfOp =
          replaceIfOpWithNewSignature(rewriter, ifOp, out.getType());
      scf::YieldOp taken = newIfOp.thenYield();
      scf::YieldOp other = newIfOp.elseYield();
      if (block == newIfOp.elseBlock())
        std::swap(taken, other);
      taken->insertOperands(taken.getNumOperands(), out);
      other->insertOperands(other.getNumOperands(), in);
      out = newIfOp.getResults().back();
      rewriter.eraseOp(ifOp);
      continue;
    }

    // TODO: Handle `scf.while`, etc.
    llvm::report_fatal_error("FIXME: sinking into unhandled control flow op: " +
                             op->getName().getStringRef());
  }

  return out;
}

//===----------------------------------------------------------------------===//
// Loop Pipelining Utilities
//===----------------------------------------------------------------------===//

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

// Function to mask operations during scheduling.
Operation *mlir::triton::predicateOp(RewriterBase &rewriter, Operation *op,
                                     Value pred) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (mlir::isMemoryEffectFree(op))
    return op;
  if (isa<LLVM::AssumeOp>(op))
    return op;
  if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op))
    return op;
  if (isa<ttg::LocalLoadOp, ttg::LocalStoreOp>(op))
    return op;
  if (isa<ttng::TMEMAllocOp, ttng::TMEMLoadOp>(op))
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
  if (auto arriveBarrier = dyn_cast<ttng::ArriveBarrierOp>(op)) {
    rewriter.setInsertionPoint(arriveBarrier);
    Value mask = pred;
    Value currentPred = arriveBarrier.getPred();
    if (currentPred) {
      mask = getPredMask(rewriter, currentPred.getType(), currentPred, pred);
    }
    arriveBarrier.getPredMutable().assign(mask);
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
  if (!op->isRegistered()) {
    // Skip ops from unregistered dialects to make writing lit tests easier.
    return op;
  }

  op->emitError("pipeliner doesn't know how to predicate this op.");
  llvm::report_fatal_error("Fatal pipeliner error");
  return op;
}

// Return true if the given ForOp has the attribute
// `tt.disallow_acc_multi_buffer` set to true.
bool mlir::triton::getDisallowAccMultiBuffer(scf::ForOp forOp) {
  return forOp->hasAttr(mlir::triton::kDisallowAccMultiBufferAttrName);
}

std::pair<OpResult, int64_t>
mlir::triton::getDefinitionAndDistance(scf::ForOp forOp, Value value) {
  int64_t distance = 0;
  DenseSet<Value> seen;
  while (auto arg = dyn_cast<BlockArgument>(value)) {
    // Ignore implicit captures.
    if (arg.getOwner() != forOp.getBody())
      return {nullptr, 0};
    // Ignore induction variable.
    if (arg.getArgNumber() == 0)
      return {nullptr, 0};
    ++distance;
    value = forOp.getYieldedValues()[arg.getArgNumber() - 1];
    if (!seen.insert(value).second)
      return {nullptr, 0};
  }
  return {cast<OpResult>(value), distance};
}

std::pair<Operation *, int64_t>
mlir::triton::getDefiningOpAndDistance(scf::ForOp forOp, Value value) {
  auto [definition, distance] = getDefinitionAndDistance(forOp, value);
  return {definition ? definition.getDefiningOp() : nullptr, distance};
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
  auto helper = TritonDialect::getLoaded(module)->getLatencyAttrHelper();
  auto builder = Builder(module);
  for (auto &[op, latency] : opLatency) {
    helper.setAttr(op, builder.getI32IntegerAttr(latency));
  }
}

void mlir::triton::serializeSelfLatencies(
    ModuleOp module, DenseMap<Operation *, int> &opSelfLatency) {
  auto helper = TritonDialect::getLoaded(module)->getSelfLatencyAttrHelper();
  auto builder = Builder(module);
  for (auto &[op, latency] : opSelfLatency) {
    helper.setAttr(op, builder.getI32IntegerAttr(latency));
  }
}

DenseMap<Operation *, int> mlir::triton::deserializeLatencies(Operation *op) {
  DenseMap<Operation *, int> opLatency;
  auto latencyHelper = TritonDialect::getLoaded(op)->getLatencyAttrHelper();
  op->walk([&](Operation *op) {
    if (auto attr = latencyHelper.getAttr(op)) {
      opLatency[op] = attr.getInt();
      latencyHelper.removeAttr(op);
    }
  });
  return opLatency;
}

Value mlir::triton::createScalarAlloc(ImplicitLocOpBuilder &rewriter, Type type,
                                      unsigned numBuffers) {
  MLIRContext *ctx = rewriter.getContext();
  unsigned numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(
      rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>());
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(rewriter.getContext());
  auto barrierCTALayout =
      ttg::CTALayoutAttr::get(/*context=*/ctx, /*CTAsPerCGA=*/{numCTAs},
                              /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding =
      ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCTALayout);
  ttg::MemDescType memDescType = ttg::MemDescType::get(
      {numBuffers}, type, barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  return rewriter.create<ttg::LocalAllocOp>(memDescType, Value());
}

// Create an allocation and init the mbarriers.
Value mlir::triton::createBarrierAlloc(scf::ForOp forOp, int numBarriers,
                                       int arriveCount) {
  ImplicitLocOpBuilder rewriter(forOp.getLoc(), forOp);

  Value barrierAlloc =
      createScalarAlloc(rewriter, rewriter.getI64Type(), numBarriers);
  for (unsigned i = 0; i < numBarriers; i++) {
    Value barrierView = createSingleBufferView(rewriter, barrierAlloc, i);
    rewriter.create<ttng::InitBarrierOp>(barrierView, arriveCount);
  }
  // Invalidate and deallocate the barriers.
  rewriter.setInsertionPointAfter(forOp);
  for (unsigned i = 0; i < numBarriers; i++) {
    Value barrierView = createSingleBufferView(rewriter, barrierAlloc, i);
    rewriter.create<ttng::InvalBarrierOp>(barrierView);
  }
  rewriter.create<ttg::LocalDeallocOp>(barrierAlloc);
  return barrierAlloc;
}

Value mlir::triton::createAlloc(Operation *insertBefore, RankedTensorType ty,
                                Location loc,
                                gpu::SharedEncodingTrait sharedEnc,
                                unsigned distance) {
  OpBuilder builder(insertBefore);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(insertBefore->getContext());
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type memdescType = ttg::MemDescType::get(bufferShape, ty.getElementType(),
                                           sharedEnc, sharedMemorySpace,
                                           /*mutableMemory=*/true);
  Value alloc = builder.create<ttg::LocalAllocOp>(loc, memdescType);

  builder.setInsertionPointAfter(insertBefore);
  builder.create<ttg::LocalDeallocOp>(insertBefore->getLoc(), alloc);
  return alloc;
}

bool mlir::triton::isTMALoad(Operation *op) {
  return isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op);
}

bool mlir::triton::canBeAsyncLoad(Operation *op) {
  if (mlir::triton::isTMALoad(op)) {
    return true;
  }
  assert(isa<tt::LoadOp>(op));
  ttg::SharedEncodingTrait sharedEncoding = mlir::triton::getSharedEncoding(op);
  // Do not create async loads for small loads (cp.async requires at least 4
  // bytes)
  int copyVecBytes = mlir::triton::getCopyVecBytes(
      cast<RankedTensorType>(op->getResultTypes()[0]), sharedEncoding);
  if (copyVecBytes >= 4) {
    return true;
  }
  return false;
}

void mlir::triton::combineRedundantWaitOps(
    llvm::SmallSetVector<ttg::AsyncWaitOp, 8> &waitOps) {
  llvm::MapVector<ttg::AsyncWaitOp, ttg::AsyncWaitOp> toDelete;
  for (auto waitOp : waitOps) {
    if (toDelete.count(waitOp))
      continue;
    SmallVector<ttg::AsyncWaitOp> waitGroup = {waitOp};
    SmallVector<Value> depTokens = waitOp.getOperands();
    unsigned minWaitNumber = waitOp.getNum();
    Operation *next = waitOp->getNextNode();
    while (next && !isa<ttg::AsyncCommitGroupOp>(next)) {
      if (auto nextWait = dyn_cast<ttg::AsyncWaitOp>(next)) {
        waitGroup.push_back(nextWait);
        minWaitNumber = std::min(minWaitNumber, nextWait.getNum());
        depTokens.append(nextWait.getOperands().begin(),
                         nextWait.getOperands().end());
      }
      next = next->getNextNode();
    }
    if (waitGroup.size() == 1)
      continue;
    OpBuilder builder(waitGroup.front());
    auto newWaitOp = builder.create<ttg::AsyncWaitOp>(waitOp.getLoc(),
                                                      depTokens, minWaitNumber);
    for (auto waitOp : waitGroup) {
      toDelete[waitOp] = newWaitOp;
    }
  }
  for (auto waitOp : toDelete) {
    waitOp.first->replaceAllUsesWith(waitOp.second);
    waitOp.first->erase();
  }
}

ttg::MemDescType mlir::triton::getBufferViewType(ttg::MemDescType allocTy) {
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(allocTy.getContext());
  return ttg::MemDescType::get(allocTy.getShape().drop_front(),
                               allocTy.getElementType(), allocTy.getEncoding(),
                               sharedMemorySpace, /*mutableMemory=*/true,
                               /*allocShape=*/allocTy.getAllocShape());
}

ttg::SharedEncodingTrait mlir::triton::getSharedEncoding(RankedTensorType ty) {
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  auto order = ttg::getOrder(ty);
  // Use generic layout. This won't be optimal for 2D tensors.
  return ttg::SwizzledSharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order,
                                              ctaLayout);
}

ttg::SharedEncodingTrait mlir::triton::getSharedEncoding(Operation *op) {
  // Try to use local alloc encoding if possible.
  ttg::SharedEncodingTrait localAllocEnc;
  if (llvm::any_of(op->getUsers(), [&](Operation *user) {
        return isa<ttg::LocalAllocOp>(user);
      })) {
    for (auto user : op->getUsers()) {
      auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
      if (!localAlloc)
        continue;
      auto enc = mlir::cast<ttg::SharedEncodingTrait>(
          localAlloc.getType().getEncoding());
      if (!localAllocEnc) {
        localAllocEnc = enc;
      }
      if (enc != localAllocEnc) {
        // Some users have different encoding than others.
        // Use one of the encodings, and warn about the performance issue.
        op->emitRemark()
            << "Pipelining load with different use encodings. This will lead "
               "to layout conversions and performance degradation.";
        continue;
      }
    }
  }

  auto ty = cast<RankedTensorType>(op->getResultTypes()[0]);
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  auto order = ttg::getOrder(ty);
  if (isTMALoad(op)) {
    // TMA encoding is set on the descriptor type
    TypedValue<tt::TensorDescType> desc;
    if (auto load = dyn_cast<tt::DescriptorLoadOp>(op)) {
      desc = load.getDesc();
    } else if (auto gather = dyn_cast<tt::DescriptorGatherOp>(op)) {
      desc = gather.getDesc();
    } else {
      op->emitError() << "unrecognized tma load type";
      llvm::report_fatal_error("unrecognized tma load type");
    }
    return ttng::getEncodingFromDescriptor(op, ty, desc);
  }

  if (localAllocEnc)
    return localAllocEnc;

  // Try to use dot encoding if possible.
  bool incompatible = false;
  localAllocEnc =
      getSharedEncIfAllUsersAreDotEnc(op->getResult(0), incompatible)
          .value_or(nullptr);

  if (localAllocEnc)
    return localAllocEnc;

  // Use generic layout. This won't be optimal for 2D tensors.
  return ttg::SwizzledSharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order,
                                              ctaLayout);
}

int mlir::triton::getNumStagesOrDefault(scf::ForOp forOp,
                                        int defaultNumStages) {
  // Use the attribute attached to the loop if it exists otherwise use the
  // global control.
  auto helper = TritonDialect::getLoaded(forOp)->getNumStagesAttrHelper();
  if (auto attr = helper.getAttr(forOp))
    return attr.getInt();
  return defaultNumStages;
}
