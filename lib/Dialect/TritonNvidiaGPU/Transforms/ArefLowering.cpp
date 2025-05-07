#include "../../TritonGPU/Transforms/WSUtility.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

#include <memory>

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritonnvidiagpu-aref-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;

bool useAref(ArefCreateOp op, Operation *user) {
  while (!isa<ArefCreateOp>(user) && user->getOperands().size() > 0) {
    user = user->getOperand(0).getDefiningOp();
  }
  return user == op.getOperation();
}

auto collectArefOps(ArefCreateOp arefOp,
                    const SmallVector<std::string> &groups) {
  SmallVector<ArefPutOp> putOps;
  SmallVector<ArefPutEnterOp> putEnterOps;
  SmallVector<ArefPutExitOp> putExitOps;
  SmallVector<ArefGetEnterOp> getOps;
  SmallVector<ArefGetExitOp> getExitOps;
  auto moduleOp = arefOp->getParentOfType<ModuleOp>();

  auto isOpInGroups = [](Operation *op,
                         const SmallVector<std::string> &groups) {
    for (const auto &group : groups) {
      if (isOpInGroup(op, group))
        return true;
    }
    return false;
  };

  for (auto func : moduleOp.getOps<triton::FuncOp>()) {
    func.walk([&](Operation *op) {
      if (!isOpInGroups(op, groups))
        return WalkResult::advance();

      if (auto putOp = dyn_cast<ArefPutOp>(op)) {
        Operation *defOp = putOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(arefOp, defOp)) {
          putOps.push_back(putOp);
        }
      } else if (auto getOp = dyn_cast<ArefGetEnterOp>(op)) {
        Operation *defOp = getOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(arefOp, defOp)) {
          getOps.push_back(getOp);
        }
      } else if (auto getExitOp = dyn_cast<ArefGetExitOp>(op)) {
        Operation *defOp =
            getExitOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(arefOp, defOp)) {
          getExitOps.push_back(getExitOp);
        }
      } else if (auto putOp = dyn_cast<ArefPutEnterOp>(op)) {
        Operation *defOp = putOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(arefOp, defOp)) {
          putEnterOps.push_back(putOp);
        }
      } else if (auto putExitOp = dyn_cast<ArefPutExitOp>(op)) {
        Operation *defOp =
            putExitOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(arefOp, defOp)) {
          putExitOps.push_back(putExitOp);
        }
      }

      return WalkResult::advance();
    });
  }
  return std::make_tuple(putOps, putEnterOps, putExitOps, getOps, getExitOps);
}

SmallVector<scf::ForOp> collectForOpsInWg(ModuleOp m) {
  // Note: The returned forOps includes the one for releasing TMA empty barriers
  // after the MMA loop. Such for loop appears for non-persistent matmul and
  // nested-loop persistent matmul kernels.
  SmallVector<scf::ForOp> forOps;
  SmallVector<WarpGroupOp> wgOps = findWarpGroupOps(m);

  for (auto wgOp : wgOps) {
    auto wgRegion = &wgOp->getRegion(0).front();
    for (auto &op : wgRegion->getOperations()) {
      if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
        forOps.push_back(forOp);
      } else if (auto ifOp2 = dyn_cast<scf::IfOp>(&op)) {
        for (const auto &thenOp2 : ifOp2.thenBlock()->getOperations()) {
          if (auto forOp2 = dyn_cast<scf::ForOp>(thenOp2)) {
            forOps.push_back(forOp2);
          }
        }
      }
    }
  }
  return forOps;
}

MemDescType getBarrierMemDesc(MLIRContext *ctx, PatternRewriter &rewriter,
                              llvm::ArrayRef<int64_t> shape) {
  Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  auto barrierCTALayout = CTALayoutAttr::get(
      /*context=*/ctx, /*CTAsPerCGA=*/{1},
      /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding =
      SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCTALayout);
  return MemDescType::get(shape, rewriter.getI64Type(), barrierEncoding,
                          sharedMemorySpace, /*mutableMemory=*/true);
}

struct ArefValue {
  Value emptyMbars;
  Value fullMbars;
  int depth;
  SmallVector<Value> buffers;
};

Value getBarrierAt(MLIRContext *ctx, Location loc, PatternRewriter &rewriter,
                   Value mbars, Value stage) {
  SmallVector<Value> offsetsVal{stage};
  auto memDesc = getBarrierMemDesc(ctx, rewriter, {1});
  return rewriter.create<triton::gpu::MemDescSubviewOp>(loc, memDesc, mbars,
                                                        offsetsVal);
}

Value getEmptyBarrierAt(MLIRContext *ctx, Location loc,
                        PatternRewriter &rewriter, ArefValue aref,
                        Value arefIdx) {
  // stage= arefIdx % depth
  auto stage = rewriter.create<arith::RemSIOp>(
      loc, arefIdx, rewriter.create<arith::ConstantIntOp>(loc, aref.depth, 32));
  return getBarrierAt(ctx, loc, rewriter, aref.emptyMbars, stage);
}

Value getFullBarrierAt(MLIRContext *ctx, Location loc,
                       PatternRewriter &rewriter, ArefValue aref,
                       Value arefIdx) {
  auto stage = rewriter.create<arith::RemSIOp>(
      loc, arefIdx, rewriter.create<arith::ConstantIntOp>(loc, aref.depth, 32));
  return getBarrierAt(ctx, loc, rewriter, aref.fullMbars, stage);
}

Value getNextPhase(Value k, Value phase, int numStage, Location loc,
                   PatternRewriter &rewriter) {
  // parity = (k+1) % numStage == 0 ? parity ^ 1 : parity
  Value D = rewriter.create<arith::ConstantIntOp>(loc, numStage, 32);
  Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  Value k1 = rewriter.create<arith::AddIOp>(loc, k, one);
  Value k1modD = rewriter.create<arith::RemSIOp>(loc, k1, D);
  Value pred = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, k1modD,
      rewriter.create<arith::ConstantIntOp>(loc, 0, 32));
  Value parityXor = rewriter.create<arith::XOrIOp>(loc, phase, one);
  return rewriter.create<arith::SelectOp>(loc, pred, parityXor, phase);
}

Value getPhaseInitValue(scf::ForOp &forOp, int numStage,
                        PatternRewriter &rewriter, Location loc) {
  if (auto outerFor = forOp->getParentOfType<scf::ForOp>()) {
    auto one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    auto stage = rewriter.create<arith::ConstantIntOp>(loc, numStage, 32);
    auto normIdx = forOp.getInitArgs().back();
    return rewriter.create<arith::AndIOp>(
        loc, rewriter.create<arith::DivSIOp>(loc, normIdx, stage), one);
  } else {
    return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  }
}

Value getPhaseInitValue(Operation *op, int numStage, bool put,
                        PatternRewriter &rewriter, Location loc) {
  Value phase;

  if (auto forOp = op->getParentOfType<scf::ForOp>()) {
    rewriter.setInsertionPoint(forOp);
    phase = getPhaseInitValue(forOp, numStage, rewriter, loc);
  } else {
    rewriter.setInsertionPoint(op);
    phase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  }

  if (put) {
    Value one = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
    return rewriter.create<arith::XOrIOp>(loc, phase, one);
  }
  return phase;
}

scf::ForOp cloneForOp(scf::ForOp forOp, PatternRewriter &rewriter,
                      const SmallVector<Value> &newInitArgs,
                      bool addNewBlockArg = false) {
  SmallVector<Value> initArgs(forOp.getInitArgs().begin(),
                              forOp.getInitArgs().end());
  initArgs.insert(initArgs.end(), newInitArgs.begin(), newInitArgs.end());

  scf::ForOp newLoop = rewriter.create<mlir::scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), initArgs);

  auto oldResults = forOp.getResults();
  auto newResults = newLoop.getResults();
  for (unsigned j = 0; j < oldResults.size(); ++j) {
    oldResults[j].replaceAllUsesWith(newResults[j]);
  }
  newLoop.getRegion().takeBody(forOp.getRegion());

  if (addNewBlockArg) {
    for (auto arg : newInitArgs) {
      auto argVar = newLoop.getBody()->addArgument(arg.getType(), arg.getLoc());
      rewriter.replaceOpUsesWithinBlock(arg.getDefiningOp(), argVar,
                                        newLoop.getBody());
    }
  }

  return newLoop;
}

scf::YieldOp cloneYieldOp(scf::YieldOp yieldOp, PatternRewriter &rewriter,
                          const SmallVector<Value> &newYieldVals) {
  SmallVector<Value> yieldVals(yieldOp.getOperands().begin(),
                               yieldOp.getOperands().end());
  yieldVals.insert(yieldVals.end(), newYieldVals.begin(), newYieldVals.end());
  return rewriter.create<scf::YieldOp>(yieldOp.getLoc(), yieldVals);
  ;
}

ArefValue createAndInitMbar(ArefCreateOp op, PatternRewriter &rewriter,
                            int consumerArrivCount) {
  MLIRContext *ctx = op.getContext();
  auto loc = op.getLoc();
  auto arefTy = op.getType();
  auto baseType = arefTy.getBaseType();
  auto arefBufTypes = llvm::to_vector(llvm::map_range(
      arefTy.getBaseType(), [](Type type) { return cast<MemDescType>(type); }));
  auto shape = arefBufTypes[0].getShape();
  auto depth = shape[0];

  MemDescType barrierMemDescType = getBarrierMemDesc(ctx, rewriter, {depth});
  // Create two mbarriers
  auto emptyMbars =
      rewriter.create<LocalAllocOp>(loc, barrierMemDescType, Value());
  emptyMbars->setAttr("aref_empty_mbarriers", rewriter.getUnitAttr());
  auto fullMbars =
      rewriter.create<LocalAllocOp>(loc, barrierMemDescType, Value());
  fullMbars->setAttr("aref_full_mbarriers", rewriter.getUnitAttr());
  auto lb = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  auto ub = rewriter.create<arith::ConstantIntOp>(loc, depth, 32);
  auto step = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  auto dLoop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
  rewriter.setInsertionPointToStart(dLoop.getBody());

  for (int i = 0; i < 2; ++i) {
    auto singleBarrier =
        getBarrierAt(ctx, loc, rewriter, (i == 0 ? emptyMbars : fullMbars),
                     dLoop.getInductionVar());
    int count = i == 0 ? consumerArrivCount : 1;
    rewriter.create<InitBarrierOp>(loc, singleBarrier, count);
  }

  return ArefValue{emptyMbars, fullMbars, static_cast<int>(depth),
                   op.getOperands()};
}

SmallVector<Value> getSubViews(ArefValue arefVal, Value stage, Location loc,
                               PatternRewriter &rewriter) {
  SmallVector<Value> views;
  for (auto buffer : arefVal.buffers) {
    SmallVector<Value> offsetsVal{stage};
    auto memDescType = cast<MemDescType>(buffer.getType());
    auto shape = memDescType.getShape();
    auto rank = shape.size() - 1;

    for (int i = 0; i < rank; ++i) {
      offsetsVal.push_back(rewriter.create<arith::ConstantIntOp>(
          loc, 0, rewriter.getIntegerType(32)));
    }
    SmallVector<int64_t> tensorShape(shape.begin() + 1, shape.end());
    auto memDescTypeNew = MemDescType::get(
        tensorShape, memDescType.getElementType(), memDescType.getEncoding(),
        memDescType.getMemorySpace(), true);
    Value singleBuffer = rewriter.create<triton::gpu::MemDescSubviewOp>(
        loc, memDescTypeNew, buffer, offsetsVal);
    views.push_back(singleBuffer);
  }

  return views;
}

Operation *getMBarAllocInsertPoint(ModuleOp mod) {
  NVVM::Barrier0Op initBarrierSyncOp;
  mod->walk([&](NVVM::Barrier0Op op) {
    if (op->hasAttr(ATTR_WS_INIT_BARRIER_SYNC)) {
      assert(!initBarrierSyncOp);
      initBarrierSyncOp = op;
    }
  });
  assert(initBarrierSyncOp);
  return initBarrierSyncOp;
}

Value rewritePersistentKernelEpilogue(scf::IfOp ifOp, Value curPhase,
                                      Value nextPhase,
                                      PatternRewriter &rewriter) {
  // A phase update on the MMA completion barrier is done inside an if op.
  // Rewrite the if op to add the phase for the next iteration in its results.
  SmallVector<Type> resTypes(ifOp.getResultTypes());
  resTypes.push_back(nextPhase.getType());

  auto newIfOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), resTypes,
                                            ifOp.getCondition(), true);

  auto updateYield = [&](scf::YieldOp yield, Value phase) {
    rewriter.setInsertionPoint(yield);
    SmallVector<Value> operands = yield.getOperands();
    operands.push_back(phase);
    rewriter.create<scf::YieldOp>(yield.getLoc(), operands);
    yield.erase();
  };

  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  // It is assumed that the phase update and other epilogue operations are
  // done in the "then" block
  updateYield(newIfOp.thenYield(), nextPhase);

  if (ifOp.elseBlock()) {
    newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
    // In the "else" block, the phase is not flipped (since MMAs for one tile
    // have not finished)
    updateYield(newIfOp.elseYield(), curPhase);
  } else {
    auto yield =
        newIfOp.getElseBodyBuilder().create<scf::YieldOp>(ifOp.getLoc());
    updateYield(yield, curPhase);
  }

  int resultIdx = 0;
  for (auto result : ifOp.getResults()) {
    result.replaceAllUsesWith(newIfOp->getResult(resultIdx++));
  }

  rewriter.eraseOp(ifOp);

  return newIfOp->getResult(resultIdx);
}

// Manually add arrive and wait on mmav5 when there is no epilogue group
// Return the updated phase for the next iteration (flips when a tile finishes)
Value arriveAndWaitMMAv5(MMAv5OpInterface op, Value tmemFull, Value phase,
                         PatternRewriter &rewriter) {
  auto forOp = dyn_cast<scf::ForOp>(op->getBlock()->getParentOp());
  assert(forOp);

  for (auto user : op.getAccumulator().getUsers()) {
    if (isa<TMEMLoadOp>(user) || isa<TMEMSubSliceOp>(user)) {
      assert(user->getParentOfType<scf::ForOp>() == forOp);
      auto loc = op->getLoc();
      rewriter.setInsertionPoint(user);
      rewriter.create<nvws::ArriveBarrierOp>(loc, tmemFull, true);
      rewriter.create<WaitBarrierOp>(loc, tmemFull, phase);

      if (auto epiIfOp = dyn_cast<scf::IfOp>(user->getParentOp())) {
        // Persistent matmul with fused loops, tmem_load is done inside an
        // epilogue if op
        Value minus = rewriter.create<arith::SubIOp>(
            loc, forOp.getInductionVar(), forOp.getLowerBound());
        Value k = rewriter.create<arith::DivSIOp>(loc, minus, forOp.getStep());
        auto nextPhase = getNextPhase(k, phase, /*numStage */ 1, loc, rewriter);
        rewriter.setInsertionPoint(epiIfOp);
        return rewritePersistentKernelEpilogue(epiIfOp, phase, nextPhase,
                                               rewriter);
      } else if (auto forOp = dyn_cast<scf::ForOp>(user->getParentOp())) {
        // FMHA kernel
        return getNextPhase(forOp.getInductionVar(), phase,
                            /*numStage */ 1, loc, rewriter);
      } else {
        llvm_unreachable("Expects fused persistent matmul or FMHA.");
      }
    }
  }

  llvm_unreachable("tmem_load not found after MMA");

  return phase;
}

Value updatePhaseCalculation(Operation *op, Value tmaIdx, Value phase,
                             PatternRewriter &rewriter, int depth) {
  rewriter.setInsertionPointAfter(op);
  auto loc = op->getLoc();
  return getNextPhase(tmaIdx, phase, depth, loc, rewriter);
}

LogicalResult lowerArefGetExitOp(ArefGetExitOp op, ArefValue arefVal,
                                 PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(op);
  auto loc = op.getLoc();
  MLIRContext *ctx = op.getContext();
  // emit the code below only if arefIdx is non-negative
  auto arefIdx = op.getOperand(1);
  Value emptyBarrier = getEmptyBarrierAt(ctx, loc, rewriter, arefVal, arefIdx);
  rewriter.replaceOpWithNewOp<nvws::ArriveBarrierOp>(op, emptyBarrier,
                                                     /*commit*/ false);
  return success();
}

LogicalResult lowerArefGetEnterOp(ArefGetEnterOp op, PatternRewriter &rewriter,
                                  Value phase, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  MLIRContext *ctx = op.getContext();

  // wait full before tma_get
  Value fullBarrier =
      getFullBarrierAt(ctx, loc, rewriter, arefVal, op.getOperand(1));
  rewriter.create<triton::nvidia_gpu::WaitBarrierOp>(loc, fullBarrier, phase);

  auto arefIdx = op.getOperand(1);
  auto stage = rewriter.create<arith::RemSIOp>(
      loc, arefIdx,
      rewriter.create<arith::ConstantIntOp>(loc, arefVal.depth, 32));
  auto views = getSubViews(arefVal, stage, loc, rewriter);
  assert(views.size() == op.getResults().size());

  for (int i = 0; i < arefVal.buffers.size(); ++i) {
    op->getResult(i).replaceAllUsesWith(views[i]);
  }

  return success();
}

void lowerTMALoad(ArefPutOp putOp, const SmallVector<Operation *> &putBodyOps,
                  const SmallVector<Value> &views, Value fullBarrier,
                  PatternRewriter &rewriter, Location loc) {
  Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);

  int sizeInBytes = 0;
  for (auto buf : views) {
    auto memDesc = cast<MemDescType>(buf.getType());
    auto bufShape = memDesc.getShape();
    auto elementType = memDesc.getElementType();
    SmallVector<int64_t> tensorShape(bufShape.begin(), bufShape.end());
    sizeInBytes +=
        product(tensorShape) * elementType.getIntOrFloatBitWidth() / 8;
  }
  rewriter.create<triton::nvidia_gpu::BarrierExpectOp>(loc, fullBarrier,
                                                       sizeInBytes, pred);

  SmallVector<DescriptorLoadOp> descriptorLoads;
  for (auto op : putBodyOps) {
    if (auto descLoad = dyn_cast<DescriptorLoadOp>(*op)) {
      descriptorLoads.push_back(descLoad);
    }
  }
  assert(descriptorLoads.size() == views.size());
  // now rewrite code in region
  //     %1 = tt.experimental_descriptor_load
  //     ttg.local_store %val, %1
  //  to rewrite with
  //     ttng.cp_async_tma_global_to_local
  for (auto [loadOp, view] : zip(descriptorLoads, views)) {
    SmallVector<Operation *> users(loadOp->user_begin(), loadOp->user_end());
    assert(users.size() == 1);
    assert(isa<LocalStoreOp>(users[0]));
    Value tmaPtr = rewriter.create<triton::nvidia_gpu::TensorDescToTMAPtrOp>(
        loc, loadOp.getDesc());
    auto indices = translateTMAIndices(
        rewriter, loadOp.getLoc(),
        loadOp.getDesc().getType().getBlockType().getEncoding(),
        loadOp.getIndices());
    rewriter.create<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(
        loc, tmaPtr, indices, fullBarrier, view, pred);

    users[0]->erase();
    loadOp->erase();
  }
}

Attribute getMemorySpace(ArefCreateOp op) {
  auto memdesc = mlir::cast<MemDescType>(op.getOperands()[0].getType());
  return memdesc.getMemorySpace();
}

Attribute getMemorySpace(ArefValue aref) {
  auto memdesc = mlir::cast<MemDescType>(aref.buffers[0].getType());
  return memdesc.getMemorySpace();
}

SmallVector<Value> producerAcquire(ArefValue arefVal, Value arefIdx,
                                   Value phase, MLIRContext *ctx, Location loc,
                                   PatternRewriter &rewriter) {
  auto stage = rewriter.create<arith::RemSIOp>(
      loc, arefIdx,
      rewriter.create<arith::ConstantIntOp>(loc, arefVal.depth, 32));
  Value emptyBarrier =
      getBarrierAt(ctx, loc, rewriter, arefVal.emptyMbars, stage);
  rewriter.create<triton::nvidia_gpu::WaitBarrierOp>(loc, emptyBarrier, phase);
  return getSubViews(arefVal, stage, loc, rewriter);
}

LogicalResult lowerArefPutEnter(ArefPutEnterOp putOp, PatternRewriter &rewriter,
                                Value phase, ArefValue arefVal) {
  auto arefIdx = putOp.getOperand(1);
  MLIRContext *ctx = putOp.getContext();
  auto loc = putOp.getLoc();
  rewriter.setInsertionPoint(putOp);

  auto views = producerAcquire(arefVal, arefIdx, phase, ctx, loc, rewriter);
  assert(views.size() == putOp.getResults().size());

  for (int i = 0; i < putOp.getResults().size(); ++i) {
    putOp.getResult(i).replaceAllUsesWith(views[i]);
  }

  return success();
}

enum class PutOpProducer { MMAv5, CpAsync, SIMT };

void producerCommit(Value fullBarrier, PutOpProducer producerOp, Location loc,
                    PatternRewriter &rewriter) {
  // TODO: Add "TrackedAsyncOp" enum, equivalent to PutOpPRoducer above, to
  // ArriveBarrierOp parameter and move this logic to LLVM codegen
  if (producerOp == PutOpProducer::CpAsync) {
    llvm_unreachable(
        "Need to support lowering ArriveBarrierOp to cp.async.mbarrier.arrive");
  } else if (producerOp == PutOpProducer::MMAv5) {
    rewriter.create<nvws::ArriveBarrierOp>(loc, fullBarrier,
                                           /*commit*/ true);
  } else if (producerOp == PutOpProducer::SIMT) {
    rewriter.create<nvws::ArriveBarrierOp>(loc, fullBarrier,
                                           /*commit*/ false);
  }
}

LogicalResult lowerArefPut(ArefPutOp putOp, PatternRewriter &rewriter,
                           Value phase, ArefValue arefVal) {
  auto arefIdx = putOp.getOperand(1);
  MLIRContext *ctx = putOp.getContext();
  auto loc = putOp.getLoc();
  rewriter.setInsertionPoint(putOp);

  auto views = producerAcquire(arefVal, arefIdx, phase, ctx, loc, rewriter);

  auto putOpBody = &putOp->getRegion(0).front();
  assert(putOpBody->getArguments().size() == views.size());
  for (auto [arg, view] : zip(putOpBody->getArguments(), views)) {
    arg.replaceAllUsesWith(view);
  }
  SmallVector<Operation *> bodyOps;
  for (auto &op : putOpBody->without_terminator()) {
    bodyOps.push_back(&op);
  }
  for (auto op : bodyOps) {
    op->moveBefore(putOp);
  }

  auto arefMemorySpace = getMemorySpace(arefVal);
  Value fullBarrier = getFullBarrierAt(ctx, loc, rewriter, arefVal, arefIdx);

  // The following assumes that the lowered op and the arrival mechanism is
  // uniquely determined by the associated memory space. This breaks down
  // for attention on blackwell.
  if (isa<triton::gpu::SharedMemorySpaceAttr>(arefMemorySpace)) {
    lowerTMALoad(putOp, bodyOps, views, fullBarrier, rewriter, loc);
  } else if (isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(arefMemorySpace)) {
    assert(views.size() == 1);
    rewriter.setInsertionPointAfter(putOp);
    producerCommit(fullBarrier, PutOpProducer::MMAv5, loc, rewriter);
  } else {
    llvm_unreachable("ArefPut on unsupported memory space.");
  }

  return success();
}

LogicalResult lowerArefPutExitOp(ArefPutExitOp op, ArefValue arefVal,
                                 PatternRewriter &rewriter) {
  rewriter.setInsertionPoint(op);
  auto loc = op.getLoc();
  MLIRContext *ctx = op.getContext();
  auto arefIdx = op.getOperand(1);
  Value fullBarrier = getFullBarrierAt(ctx, loc, rewriter, arefVal, arefIdx);
  // Note, hardcoded for MMAv5. For now it is assumed that PutExit is only used
  // after MMAv5.
  producerCommit(fullBarrier, PutOpProducer::MMAv5, loc, rewriter);
  return success();
}

scf::YieldOp createNewYieldOp(scf::ForOp newForOp, Operation *op,
                              ArefValue aref, Value arefIdx, Value phaseVar,
                              PatternRewriter &rewriter,
                              const SmallVector<Value> &additionalPhases = {}) {
  // update phase
  auto selectOp =
      updatePhaseCalculation(op, arefIdx, phaseVar, rewriter, aref.depth);
  SmallVector<Value> nextPhases{selectOp};
  nextPhases.insert(nextPhases.end(), additionalPhases.begin(),
                    additionalPhases.end());
  // create new yieldOp for loop
  auto yieldOp = mlir::cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  rewriter.setInsertionPoint(yieldOp);
  auto newYieldOp = cloneYieldOp(yieldOp, rewriter, nextPhases);
  yieldOp.erase();
  return newYieldOp;
}

LogicalResult lowerArefPut(SmallVector<ArefPutOp> &putOps, ArefValue aref,
                           PatternRewriter &rewriter, Location loc) {
  Value phase = getPhaseInitValue(putOps[0], aref.depth, true, rewriter, loc);
  // for aref_put
  // phase = init_phase = 0
  // for aref_put in aref_put_ops in program order
  //   assert aref_put is inside loop
  //   loop_var = phase
  //   mbar_wait(loop_var)
  //   select = (idx + 1) % D == 0 ? loop_var ^ 1 : loop_var
  //   phase = new_for_op_result
  //
  LogicalResult result = success();
  for (auto putOp : putOps) {
    if (auto forOp = putOp->getParentOfType<scf::ForOp>()) {
      rewriter.setInsertionPointAfter(forOp);
      auto newForOp = cloneForOp(forOp, rewriter, SmallVector<Value>{phase});
      // Add the new block argument for the new loop-carried variable
      auto newBlockArg =
          newForOp.getBody()->addArgument(phase.getType(), phase.getLoc());
      // New aref phase for the next for loop in causal attention
      phase = newForOp.getResults()[forOp.getResults().size()];

      // Is this preserving the original ops? let's check here
      bool found = false;
      for (auto &op : newForOp.getBody()->getOperations()) {
        if (auto putOp2 = dyn_cast<ArefPutOp>(&op)) {
          if (putOp2 == putOp) {
            found = true;
            break;
          }
        }
      }
      assert(found && "aref_put is not perserved");

      // use the newBlockArg as phase
      result = lowerArefPut(putOp, rewriter, newBlockArg, aref);

      createNewYieldOp(newForOp, putOp, aref, putOp->getOperand(1), newBlockArg,
                       rewriter);

      rewriter.eraseOp(forOp);
    } else {
      result = lowerArefPut(putOp, rewriter, phase, aref);
    }
  }
  return result;
}

LogicalResult lowerArefPutEnter(SmallVector<ArefPutEnterOp> &putOps,
                                ArefValue aref, PatternRewriter &rewriter,
                                Location loc) {
  Value phase = getPhaseInitValue(putOps[0], aref.depth, true, rewriter, loc);

  LogicalResult result = success();
  for (auto putOp : putOps) {
    if (auto forOp = putOp->getParentOfType<scf::ForOp>()) {
      rewriter.setInsertionPointAfter(forOp);
      auto newForOp = cloneForOp(forOp, rewriter, SmallVector<Value>{phase});
      // Add the new block argument for the new loop-carried variable
      auto newBlockArg =
          newForOp.getBody()->addArgument(phase.getType(), phase.getLoc());
      // New aref phase for the next for loop in causal attention
      phase = newForOp.getResults()[forOp.getResults().size()];
      result = lowerArefPutEnter(putOp, rewriter, newBlockArg, aref);
      createNewYieldOp(newForOp, putOp, aref, putOp->getOperand(1), newBlockArg,
                       rewriter);
      rewriter.eraseOp(forOp);
    } else {
      result = lowerArefPutEnter(putOp, rewriter, phase, aref);
    }
  }
  return result;
}

scf::ForOp getTMEMLoadParentForOp(MMAv5OpInterface mma) {
  for (auto user : mma.getAccumulator().getUsers()) {
    if (isa<TMEMLoadOp>(user) || isa<TMEMSubSliceOp>(user)) {
      return user->getParentOfType<scf::ForOp>();
    }
  }
  return nullptr;
}

LogicalResult lowerTMAGet(SmallVector<ArefGetEnterOp> &tmaGetOps,
                          ArefValue aref, PatternRewriter &rewriter,
                          Location loc) {
  LogicalResult result = success();

  auto makeCstI32 = [&](int c) {
    return rewriter.create<arith::ConstantIntOp>(loc, c, 32);
  };

  // handle tma_get ops in program order
  // if the first tma_get op is outside a loop insert before the first tma_get
  // op, otherwise insert before the the corresponding loop
  Value mathPhase =
      getPhaseInitValue(tmaGetOps[0], aref.depth, false, rewriter, loc);

  // for tma_get
  // phase = init_phase = 0
  // for tma_get in tma_get_ops in program order
  //   if tma_get is outside a loop
  //     mbar_wait(phase)
  //     phase = (idx + 1) % D == 0 ? phase ^ 1 : phase
  //   else
  //     loop_var = phase
  //     mbar_wait(loop_var)
  //     select = (idx + 1) % D == 0 ? loop_var ^ 1 : loop_var
  //     yield select
  //     phase = new_for_op_result
  //
  for (auto tmaGetOp : tmaGetOps) {
    auto arefIdx = tmaGetOp->getOperand(1);
    if (auto forOp = tmaGetOp->getParentOfType<scf::ForOp>()) {
      // tma_get is inside a loop
      rewriter.setInsertionPointAfter(forOp);

      // use the current mathPhase as the init value
      SmallVector<Value> newInitVals{mathPhase};
      SmallVector<MMAv5OpInterface> mmav5OpsToSync;

      for (auto user : tmaGetOp.getOperand(0).getUsers()) {
        if (auto mmav5 = dyn_cast<MMAv5OpInterface>(user)) {
          if (auto tmemLoadParentForOp = getTMEMLoadParentForOp(mmav5)) {
            if (tmemLoadParentForOp == mmav5->getParentOfType<scf::ForOp>()) {
              // TMEM load is not hoisted outside of the loop, so the epilogue
              // group is not created. Persistent matmul with fused loops or
              // FMHA.
              newInitVals.push_back(makeCstI32(0));
              mmav5OpsToSync.push_back(mmav5);
            }
          }
        }
      }

      auto newForOp = cloneForOp(forOp, rewriter, newInitVals);
      // Add the new block argument for the new loop-carried variable
      auto newBlockArg = newForOp.getBody()->addArgument(mathPhase.getType(),
                                                         mathPhase.getLoc());
      // New math phase for the next for loop in causal attention
      mathPhase = newForOp.getResults()[forOp.getResults().size()];

      // let's check if takeBody preserves the original ops
      bool found = false;
      for (auto &op : newForOp.getBody()->getOperations()) {
        if (auto tmaGetOp2 = dyn_cast<ArefGetEnterOp>(&op)) {
          if (tmaGetOp2 == tmaGetOp) {
            found = true;
            break;
          }
        }
      }
      assert(found && "tma_get is not perserved");

      result = lowerArefGetEnterOp(tmaGetOp, rewriter, newBlockArg, aref);

      SmallVector<Value> mmaCompletionPhases;

      // Manually add arrive and wait on mmav5 for persistent matmul with fused
      // loops and FMHA.
      for (auto &mmav5Op : mmav5OpsToSync) {
        Value tmemFull;
        {
          OpBuilder::InsertionGuard g(rewriter);
          auto barrierAllocPoint =
              getMBarAllocInsertPoint(mmav5Op->getParentOfType<ModuleOp>());
          rewriter.setInsertionPoint(barrierAllocPoint);
          tmemFull = rewriter.create<LocalAllocOp>(
              loc, getBarrierMemDesc(mmav5Op.getContext(), rewriter, {1}),
              Value());
          rewriter.create<InitBarrierOp>(loc, tmemFull, 1);
        }

        auto phase = makeCstI32(0);
        auto phaseVar =
            newForOp.getBody()->addArgument(phase.getType(), phase.getLoc());
        auto nextPhase =
            arriveAndWaitMMAv5(mmav5Op, tmemFull, phaseVar, rewriter);
        mmaCompletionPhases.push_back(nextPhase);
      }

      createNewYieldOp(newForOp, tmaGetOp, aref, arefIdx, newBlockArg, rewriter,
                       mmaCompletionPhases);

      for (auto user : tmaGetOp.getOperand(0).getUsers()) {
        if (auto mmav5 = dyn_cast<MMAv5OpInterface>(user)) {
          auto ctx = tmaGetOp.getContext();
          rewriter.setInsertionPoint(mmav5);
          auto barriers = mmav5.getBarriersMutable();
          auto barriers_pred = mmav5.getBarrierPredsMutable();

          assert(barriers.size() == 1);
          assert(barriers_pred.size() == 1);

          auto arefIdx = barriers_pred[0].get();
          auto tmaEmpty = getEmptyBarrierAt(ctx, loc, rewriter, aref, arefIdx);
          barriers[0].set(tmaEmpty);
          barriers_pred[0].set(rewriter.create<arith::ConstantIntOp>(
              loc, 1, rewriter.getI1Type()));
        }
      }

      rewriter.eraseOp(forOp);
    } else {
      // tma_get is outside a loop
      result = lowerArefGetEnterOp(tmaGetOp, rewriter, mathPhase, aref);
      mathPhase = updatePhaseCalculation(tmaGetOp.getOperation(), arefIdx,
                                         mathPhase, rewriter, aref.depth);
    }
  }
  return result;
}

LogicalResult lowerTMEMGet(SmallVector<ArefGetEnterOp> &tmemGetOps,
                           ArefValue aref, PatternRewriter &rewriter,
                           Location loc) {
  LogicalResult result = success();

  auto makeCstI32 = [&](int c) {
    return rewriter.create<arith::ConstantIntOp>(loc, c, 32);
  };

  for (auto tmemGetOp : tmemGetOps) {
    if (auto forOp = tmemGetOp->getParentOfType<scf::ForOp>()) {
      // tmem_get is inside a loop
      rewriter.setInsertionPointAfter(forOp);
      Value initPhase = getPhaseInitValue(tmemGetOp, aref.depth, false, rewriter, loc);
      // use the current phase as the init value
      auto newForOp =
          cloneForOp(forOp, rewriter, SmallVector<Value>{initPhase});
      // Add the new block argument for the new loop-carried variable
      auto newBlockArg = newForOp.getBody()->addArgument(initPhase.getType(),
                                                         initPhase.getLoc());

      result = lowerArefGetEnterOp(tmemGetOp, rewriter, newBlockArg, aref);

      createNewYieldOp(newForOp, tmemGetOp, aref, tmemGetOp->getOperand(1),
                       newBlockArg, rewriter);
      rewriter.eraseOp(forOp);
    } else {
      rewriter.setInsertionPoint(tmemGetOp);
      Value initPhase = makeCstI32(0);
      result = lowerArefGetEnterOp(tmemGetOp, rewriter, initPhase, aref);
    }
  }
  return result;
}

auto getWSGroupProperties(ArefCreateOp op) {
  auto mod = op->getParentOfType<ModuleOp>();
  auto target = mod->getAttrOfType<StringAttr>(AttrTargetName);
  bool isMMAv3 = target == "cuda:90";
  bool isMMAv5 = target == "cuda:100";
  assert(isMMAv3 || isMMAv5);
  assert(!isMMAv3 || !isMMAv5);
  // XXX: analyze ttng.warp_group region to use correct num-warp counts
  auto numWarps =
      mlir::cast<mlir::IntegerAttr>(mod->getAttr("ttg.num-warps")).getInt();
  auto memSpace = getMemorySpace(op);

  if (isa<triton::gpu::SharedMemorySpaceAttr>(memSpace)) {
    // For MMAv5, an arrive is done by tcgen05.commit, which is issued by a
    // single thread.
    auto consumerArrivalCount = isMMAv5 ? 1 : numWarps * 32;
    SmallVector<std::string> groups{ATTR_WS_TMALOAD, ATTR_WS_MMA};
    return std::make_tuple(consumerArrivalCount, groups);
  } else if (isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(memSpace)) {
    assert(isMMAv5);
    auto consumerArrivalCount = numWarps * 32;
    SmallVector<std::string> groups{ATTR_WS_MMA, ATTR_WS_EPILOGUE};
    return std::make_tuple(consumerArrivalCount, groups);
  } else {
    llvm_unreachable("unsupported memory space for aref");
    return std::make_tuple(numWarps * 32, SmallVector<std::string>{});
  }
}

class ArefCreateLowering : public OpRewritePattern<ArefCreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ArefCreateOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.setInsertionPointAfter(op);

    auto [consumerMMArrivCount, groups] = getWSGroupProperties(op);
    auto aref = createAndInitMbar(op, rewriter, consumerMMArrivCount);
    auto [putOps, putEnterOps, putExitOps, getOps, getExitOps] =
        collectArefOps(op, groups);

    llvm::SmallSetVector<Operation *, 10> opToDelete;
    opToDelete.insert(op.getOperation());
    opToDelete.insert(putOps.begin(), putOps.end());
    opToDelete.insert(putEnterOps.begin(), putEnterOps.end());
    opToDelete.insert(putExitOps.begin(), putExitOps.end());
    opToDelete.insert(getOps.begin(), getOps.end());

    rewriter.setInsertionPoint(op);
    auto loc = op.getLoc();
    auto memSpace = getMemorySpace(op);

    if (isa<triton::gpu::SharedMemorySpaceAttr>(memSpace)) {
      if (lowerArefPut(putOps, aref, rewriter, loc).failed())
        return failure();
      if (lowerTMAGet(getOps, aref, rewriter, loc).failed())
        return failure();
    } else if (isa<triton::nvidia_gpu::TensorMemorySpaceAttr>(memSpace)) {
      if (lowerArefPutEnter(putEnterOps, aref, rewriter, loc).failed())
        return failure();
      if (lowerTMEMGet(getOps, aref, rewriter, loc).failed())
        return failure();
    } else {
      llvm_unreachable("unsupported memory space for aref");
    }

    for (auto op : getExitOps) {
      if (lowerArefGetExitOp(op, aref, rewriter).failed())
        return failure();
    }

    for (auto op : putExitOps) {
      if (lowerArefPutExitOp(op, aref, rewriter).failed())
        return failure();
    }

    // reverse travesal to delete all op
    for (auto it = opToDelete.rbegin(); it != opToDelete.rend(); ++it) {
      rewriter.eraseOp(*it);
    }

    return success();
  }
};

class TMAStoreLowering : public OpRewritePattern<DescriptorStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Attribute sharedMemorySpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
    auto loc = op.getLoc();
    auto tensorType = op.getSrc().getType();
    auto order = getOrder(tensorType);
    auto ctaLayout = getCTALayout(tensorType.getEncoding());
    auto m = op->getParentOfType<ModuleOp>();
    auto numWarps =
        mlir::cast<mlir::IntegerAttr>(m->getAttr("ttg.num-warps")).getInt();
    auto encoding =
        getEncodingFromDescriptor(op, op.getSrc().getType(), op.getDesc());

    // The copy of the previous output tile can still be inflight.
    rewriter.create<triton::nvidia_gpu::TMAStoreWaitOp>(loc, 0);
    // Ensure all threads arrive at this point to avoid race conditions between
    // two TMA stores in Blackwell tests with sub-tiling enabled. Without this
    // barrier, TMAStoreWaitOp might be executed by another warp that is
    // slightly ahead of the warp issuing AsyncTMACopyLocalToGlobal. The barrier
    // ensures that all warps proceed simultaneously after the data is copied.
    insertBarrier(rewriter, loc);

    MemDescType memDescType =
        MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                         encoding, sharedMemorySpace, /*mutableMemory=*/true);
    Value alloc = rewriter.create<LocalAllocOp>(loc, memDescType, op.getSrc());
    rewriter.create<triton::nvidia_gpu::FenceAsyncSharedOp>(loc, false);
    insertBarrier(rewriter, loc);
    Value tmaPtr = rewriter.create<triton::nvidia_gpu::TensorDescToTMAPtrOp>(
        loc, op.getDesc());
    auto indices = translateTMAIndices(
        rewriter, op.getLoc(),
        op.getDesc().getType().getBlockType().getEncoding(), op.getIndices());
    rewriter.create<triton::nvidia_gpu::AsyncTMACopyLocalToGlobalOp>(
        loc, tmaPtr, indices, alloc);
    // Do not wait immediately after issuing TMA store, to overlap the copy with
    // tmem load of the next tile when the epilogue group is decoupled.
    rewriter.eraseOp(op);
    return success();
  }
};

class TransOpRewrite : public OpRewritePattern<TransOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransOp transOp,
                                PatternRewriter &rewriter) const override {
    // FIXME: Hacky way to update TransOp layout
    // need to ensure the output type also has the "mutable" attribute
    // %47 = tt.trans %46 {order = array<i32: 1, 0>} : !tt.memdesc<128x64xf16,
    // #shared2, #ttg.shared_memory, mutable> ->
    // !tt.memdesc<64x128xf16, #shared3, #ttg.shared_memory, mutable>
    if (!transOp->hasAttr("mutable")) {
      rewriter.setInsertionPoint(transOp);
      TransOp newTransOp = rewriter.create<triton::TransOp>(
          transOp.getLoc(), transOp.getOperand(), transOp.getOrderAttr());
      newTransOp->setAttr("mutable", rewriter.getUnitAttr());
      transOp.replaceAllUsesWith(newTransOp.getResult());
    }
    return success();
  }
};

class MemDescTransOpRewrite
    : public OpRewritePattern<triton::gpu::MemDescTransOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::MemDescTransOp memDescTransOp,
                                PatternRewriter &rewriter) const override {
    // FIXME: Hacky way to update TransOp layout
    // need to ensure the output type also has the "mutable" attribute
    // %47 = tt.trans %46 {order = array<i32: 1, 0>} : !tt.memdesc<128x64xf16,
    // #shared2, #ttg.shared_memory, mutable> ->
    // !tt.memdesc<64x128xf16, #shared3, #ttg.shared_memory, mutable>
    if (!memDescTransOp->hasAttr("mutable")) {
      rewriter.setInsertionPoint(memDescTransOp);
      triton::gpu::MemDescTransOp newTransOp =
          rewriter.create<triton::gpu::MemDescTransOp>(
              memDescTransOp.getLoc(), memDescTransOp.getOperand(),
              memDescTransOp.getOrderAttr());
      newTransOp->setAttr("mutable", rewriter.getUnitAttr());
      memDescTransOp.replaceAllUsesWith(newTransOp.getResult());
    }
    return success();
  }
};

int getNumTMAStages(WarpGroupOp wgOp) {
  if (isOpInGroup(wgOp, ATTR_WS_TMALOAD)) {
    auto m = wgOp->getParentOfType<ModuleOp>();
    return TritonGPUDialect::getNumStages(m);
  } else if (isOpInGroup(wgOp, ATTR_WS_EPILOGUE)) {
    return 1; // TMA store is not pipelined
  } else {
    llvm_unreachable("Descriptor update in an unexpected warp group.\n");
  }
  return 0;
}

class TritonNvidiaGPUArefLoweringPass
    : public TritonNvidiaGPUArefLoweringPassBase<
          TritonNvidiaGPUArefLoweringPass> {
public:
  triton::FuncOp getFuncOp(ModuleOp moduleOp) {
    triton::FuncOp funcOp;
    moduleOp.walk([&](triton::FuncOp op) { funcOp = op; });
    return funcOp;
  }

  void fixupIfOp(triton::FuncOp &funcOp, SmallVector<scf::ForOp> &forOps) {
    OpBuilder builder(funcOp);
    assert(forOps.size() == 4);

    auto lastMathForOp = forOps.back();
    scf::IfOp firstMathIfOp = nullptr;
    SmallVector<Value> newThenYieldOperands;
    DenseMap<int, Value> initArgMap;
    for (auto const &init : llvm::enumerate(lastMathForOp.getInitArgs())) {
      auto defOp = init.value().getDefiningOp();
      auto forOp =
          dyn_cast_or_null<scf::ForOp>(defOp->getBlock()->getParentOp());
      auto ifOp = dyn_cast_or_null<scf::IfOp>(defOp->getBlock()->getParentOp());
      if (forOp) {
        ifOp = dyn_cast_or_null<scf::IfOp>(
            forOp.getOperation()->getBlock()->getParentOp());
      }
      if (ifOp) {
        if (firstMathIfOp == nullptr) {
          firstMathIfOp = ifOp;
        } else {
          assert(ifOp == firstMathIfOp);
        }
        newThenYieldOperands.push_back(init.value());
        initArgMap[init.index()] = init.value();
      }
    }
    if (newThenYieldOperands.empty()) {
      return;
    }

    int origIfNumResults = firstMathIfOp.getNumResults();
    builder.setInsertionPoint(firstMathIfOp);
    auto zero =
        builder.create<arith::ConstantIntOp>(firstMathIfOp.getLoc(), 0, 32);
    SmallVector<Value> newElseYieldOperands(newThenYieldOperands.size(), zero);

    auto thenYield = firstMathIfOp.thenBlock()->getTerminator();
    SmallVector<Value> thenYieldOperands = thenYield->getOperands();
    for (auto &v : newThenYieldOperands) {
      thenYieldOperands.push_back(v);
    }
    auto elseYield = firstMathIfOp.elseBlock()->getTerminator();
    SmallVector<Value> elseYieldOperands = elseYield->getOperands();
    for (auto &v : newElseYieldOperands) {
      elseYieldOperands.push_back(v);
    }
    SmallVector<Type> tensorTypes;
    for (auto &v : thenYieldOperands) {
      tensorTypes.push_back(v.getType());
    }

    auto newIfOp = builder.create<scf::IfOp>(
        firstMathIfOp.getLoc(), tensorTypes, firstMathIfOp.getCondition(),
        /*withElseRegion=*/true);
    builder.setInsertionPointToEnd(newIfOp.thenBlock());
    auto newThenYield =
        builder.create<scf::YieldOp>(newIfOp.getLoc(), thenYieldOperands);
    builder.setInsertionPointToEnd(newIfOp.elseBlock());
    auto newElseYield =
        builder.create<scf::YieldOp>(newIfOp.getLoc(), elseYieldOperands);
    for (mlir::Operation &op : llvm::make_early_inc_range(llvm::make_range(
             firstMathIfOp.thenBlock()->getOperations().begin(),
             firstMathIfOp.thenBlock()->getTerminator()->getIterator()))) {
      op.moveBefore(newThenYield);
    }
    // the else block should be empty
    assert(firstMathIfOp.elseBlock()->getOperations().size() == 1);
    for (int i = 0; i < origIfNumResults; ++i) {
      firstMathIfOp.getResult(i).replaceAllUsesWith(newIfOp.getResult(i));
    }

    // now replace the init args of the forOp to the corresponding result of
    // the new ifOp
    SmallVector<Value> newLoopInit = lastMathForOp.getInitArgs();
    int i = 0;
    for (auto p : initArgMap) {
      newLoopInit[p.first] = newIfOp.getResult(origIfNumResults + i);
      ++i;
    }
    IRMapping mapping;
    builder.setInsertionPointAfter(newIfOp);
    auto newForOp = builder.create<scf::ForOp>(
        lastMathForOp.getLoc(), lastMathForOp.getLowerBound(),
        lastMathForOp.getUpperBound(), lastMathForOp.getStep(), newLoopInit);
    for (const auto &arg : llvm::enumerate(lastMathForOp.getRegionIterArgs())) {
      mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
    }
    mapping.map(lastMathForOp.getInductionVar(), newForOp.getInductionVar());
    builder.setInsertionPointToStart(newForOp.getBody());
    for (Operation &op : lastMathForOp.getBody()->getOperations()) {
      builder.clone(op, mapping);
    }
    lastMathForOp.replaceAllUsesWith(newForOp.getResults());
    lastMathForOp.erase();
    firstMathIfOp.erase();
  }

  void fixUpSync(ModuleOp m) {
    auto target = m->getAttrOfType<StringAttr>(AttrTargetName);

    m.walk([&](WarpGroupOp wgOp) {
      auto numWarps = wgOp.getNumWarps();

      if (isOpInGroup(wgOp, ATTR_WS_EPILOGUE)) {
        // When TMA store is not used, add barrier at the end of the epilogue
        // loop to make sure that warps are in sync. Without this, there could
        // be a race between the current iteration LDS of some warps and the
        // next iteration STS of other warps.
        // When TMA store is used, the barrier inserted by TMAStoreLowering
        // does the same job.
        wgOp.getRegion(0).front().walk([&](triton::StoreOp op) {
          OpBuilder builder(op);
          builder.setInsertionPointAfter(op);
          insertBarrier(builder, op->getLoc());
        });
      } else if ((isOpInGroup(wgOp, ATTR_WS_TMALOAD) ||
                  isOpInGroup(wgOp, ATTR_WS_MMA)) &&
                 numWarps > 1 && target == "cuda:100") {
        // On Blackwell, we sychronize warps at the beginning of the K loop for
        // load and MMA groups. A hang was observed without this barrier.
        // TODO: Understand why this barrier is necessary in some cases,
        // especially for small K problems.
        wgOp.getRegion(0).front().walk([&](scf::ForOp forOp) {
          if (isInnerMostLoop(forOp)) {
            OpBuilder builder(forOp);
            builder.setInsertionPointToStart(forOp.getBody());
            insertBarrier(builder, forOp->getLoc());
          }
        });
      }
    });
  }

  void runOnOperation() override {
    // std::cout << "===== TritonNvidiaGPUArefLoweringPass =====\n";
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    LLVM_DEBUG({
      DBGS() << "Module before aref lowering:\n";
      m.dump();
    });

    // aref lowering
    mlir::RewritePatternSet patterns(context);
    patterns.add<ArefCreateLowering, TMAStoreLowering>(context);
    GreedyRewriteConfig config;

    if (applyPatternsGreedily(m, std::move(patterns), config).failed()) {
      signalPassFailure();
    }

    SmallVector<scf::ForOp> forOps = collectForOpsInWg(m);

    // Multibuffer TMA descriptor update
    for (auto forOp : forOps) {
      for (auto &op : forOp.getBody()->getOperations()) {
        if (isa<MakeTensorDescOp>(op)) {
          auto wgOp = op.getParentOfType<WarpGroupOp>();
          int numStages = getNumTMAStages(wgOp);
          lowerTMADescriptors(forOp, numStages + 1);
          break;
        }
      }
    }

    // for causal FA, we added ifOp to wrap around the first math forOp. Here
    // we need to fixup yield operands and corresponding dataflow
    if (forOps.size() == 4) {
      triton::FuncOp funcOp = getFuncOp(m);
      fixupIfOp(funcOp, forOps);
    }

    // Add other synchronizations
    fixUpSync(m);

    // update trans op
    mlir::RewritePatternSet transPattern(context);
    transPattern.add<TransOpRewrite>(context);
    transPattern.add<MemDescTransOpRewrite>(context);
    // not sure why .failed() will always return true, so just remove it here
    auto result = applyPatternsGreedily(m, std::move(transPattern));

    LLVM_DEBUG({
      DBGS() << "Module after aref lowering:\n";
      m.dump();
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::createTritonNvidiaGPUArefLoweringPass() {
  return std::make_unique<TritonNvidiaGPUArefLoweringPass>();
}
