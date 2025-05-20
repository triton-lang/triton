#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WSUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/ErrorHandling.h"

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

triton::FuncOp getFuncOp(ModuleOp moduleOp) {
  triton::FuncOp funcOp;
  moduleOp.walk([&](triton::FuncOp op) { funcOp = op; });
  return funcOp;
}

auto collectArefOps(ArefCreateOp arefOp) {
  SmallVector<ArefPutEnterOp> putEnterOps;
  SmallVector<ArefPutExitOp> putExitOps;
  SmallVector<ArefGetEnterOp> getEnterOps;
  SmallVector<ArefGetExitOp> getExitOps;

  auto useAref = [&](Operation *user) {
    while (!isa<ArefCreateOp>(user) && user->getOperands().size() > 0) {
      user = user->getOperand(0).getDefiningOp();
    }
    return user == arefOp.getOperation();
  };

  auto moduleOp = arefOp->getParentOfType<ModuleOp>();
  for (auto func : moduleOp.getOps<triton::FuncOp>()) {
    func.walk([&](Operation *op) {
      if (auto getEnterOp = dyn_cast<ArefGetEnterOp>(op)) {
        Operation *defOp =
            getEnterOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(defOp)) {
          getEnterOps.push_back(getEnterOp);
        }
      } else if (auto getExitOp = dyn_cast<ArefGetExitOp>(op)) {
        Operation *defOp =
            getExitOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(defOp)) {
          getExitOps.push_back(getExitOp);
        }
      } else if (auto putEnterOp = dyn_cast<ArefPutEnterOp>(op)) {
        Operation *defOp =
            putEnterOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(defOp)) {
          putEnterOps.push_back(putEnterOp);
        }
      } else if (auto putExitOp = dyn_cast<ArefPutExitOp>(op)) {
        Operation *defOp =
            putExitOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(defOp)) {
          putExitOps.push_back(putExitOp);
        }
      }

      return WalkResult::advance();
    });
  }
  return std::make_tuple(putEnterOps, putExitOps, getEnterOps, getExitOps);
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
  stage->setAttr("empty_mbar", rewriter.getUnitAttr());
  return getBarrierAt(ctx, loc, rewriter, aref.emptyMbars, stage);
}

Value getFullBarrierAt(MLIRContext *ctx, Location loc,
                       PatternRewriter &rewriter, ArefValue aref,
                       Value arefIdx) {
  auto stage = rewriter.create<arith::RemSIOp>(
      loc, arefIdx, rewriter.create<arith::ConstantIntOp>(loc, aref.depth, 32));
  stage->setAttr("full_mbar", rewriter.getUnitAttr());
  return getBarrierAt(ctx, loc, rewriter, aref.fullMbars, stage);
}

std::pair<int, int> getArrivalCount(ArefCreateOp op) {
  auto mod = op->getParentOfType<ModuleOp>();
  auto target = mod->getAttrOfType<StringAttr>(AttrTargetName);
  bool isMMAv3 = target == "cuda:90";
  bool isMMAv5 = target == "cuda:100";
  assert(isMMAv3 || isMMAv5);
  assert(!isMMAv3 || !isMMAv5);

  std::optional<int> producerArrivalCount, consumerArrivalCount;
  auto setProducerArrivalCount = [&](int count) {
    if (producerArrivalCount && *producerArrivalCount != count)
      llvm_unreachable("inconsistent producer arrival count");
    producerArrivalCount = count;
  };
  auto setConsumerArrivalCount = [&](int count) {
    if (consumerArrivalCount && *consumerArrivalCount != count)
      llvm_unreachable("inconsistent consumer arrival count");
    consumerArrivalCount = count;
  };
  for (auto user : op->getUsers()) {
    auto wgOp = user->getParentOfType<WarpGroupOp>();
    auto numWarps = wgOp.getNumWarps();
    if (auto putExitOp = dyn_cast<ArefPutExitOp>(user)) {
      for (auto prod : putExitOp.getProducers()) {
        auto kind = dyn_cast<ArefProducerAttr>(prod).getValue();
        switch (kind) {
        case ArefProducer::UMMA:
        case ArefProducer::TMALDG:
          setProducerArrivalCount(1);
          break;
        case ArefProducer::LDGSTS:
        case ArefProducer::STS:
        case ArefProducer::STTM:
          setProducerArrivalCount(numWarps * 32);
          break;
        default:
          llvm_unreachable("unknown producer kind");
        }
      }
    } else if (auto getExitOp = dyn_cast<ArefGetExitOp>(user)) {
      for (auto consumer : getExitOp.getConsumers()) {
        auto kind = cast<ArefConsumerAttr>(consumer).getValue();
        switch (kind) {
        case ArefConsumer::UMMA:
          setConsumerArrivalCount(1);
          break;
        case ArefConsumer::LDS:
        case ArefConsumer::WGMMA:
        case ArefConsumer::LDTM:
          setConsumerArrivalCount(numWarps * 32);
          break;
        default:
          llvm_unreachable("unknown consumer kind");
        }
      }
    }
  }

  assert(producerArrivalCount);
  assert(consumerArrivalCount);

  return {*producerArrivalCount, *consumerArrivalCount};
};

ArefValue createAndInitMbar(ArefCreateOp op, PatternRewriter &rewriter) {
  auto [producerArrivalCount, consumerArrivalCount] = getArrivalCount(op);

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
    auto mbars = i == 0 ? emptyMbars : fullMbars;
    auto singleBarrier =
        getBarrierAt(ctx, loc, rewriter, mbars, dLoop.getInductionVar());
    int arrivalCount = i == 0 ? consumerArrivalCount : producerArrivalCount;
    rewriter.create<InitBarrierOp>(loc, singleBarrier, arrivalCount);
  }

  return ArefValue{emptyMbars, fullMbars, static_cast<int>(depth),
                   op.getOperands()};
}

SmallVector<Value> getSubViews(ArefValue arefVal, Value stage, Location loc,
                               OpBuilder &rewriter) {
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

struct LastPhaseValue {
  Value put = {};
  Value get = {};
};
using ArefLastPhaseMap = llvm::MapVector<Value /*aref*/, LastPhaseValue>;

struct ArefFirstUse {
  ArefPutEnterOp putOp = {};
  ArefGetEnterOp getOp = {};
};
using ArefFirstUseMap = llvm::MapVector<Value /*aref*/, ArefFirstUse>;

ArefFirstUseMap AnalyzeArefUseInBlock(Block *block, ArefFirstUseMap arefUse) {
  for (auto &op : *block) {
    if (auto putEnterOp = dyn_cast<ArefPutEnterOp>(op)) {
      if (!putEnterOp.getPhase()) {
        auto aref = putEnterOp.getAref();
        if (!arefUse[aref].putOp)
          arefUse[aref].putOp = putEnterOp;
      }
    } else if (auto getEnterOp = dyn_cast<ArefGetEnterOp>(op)) {
      auto aref = getEnterOp.getAref();
      if (!getEnterOp.getPhase()) {
        if (!arefUse[aref].getOp)
          arefUse[aref].getOp = getEnterOp;
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // recursive visit for-op body to gather nested uses of arefs in put/get
      auto &region = forOp.getRegion();
      auto block = &region.getBlocks().front();

      auto arefUseBlock = AnalyzeArefUseInBlock(block, ArefFirstUseMap{});
      for (auto [aref, useMap] : arefUseBlock) {
        if (!arefUse[aref].putOp)
          arefUse[aref].putOp = useMap.putOp;
        if (!arefUse[aref].getOp)
          arefUse[aref].getOp = useMap.getOp;
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // recursive visit if-op then/else body to gather nested uses of arefs
      auto &thenRegion = ifOp.getThenRegion();
      auto thenBlock = &thenRegion.getBlocks().front();
      auto thenArefUse = AnalyzeArefUseInBlock(thenBlock, ArefFirstUseMap{});
      for (auto [aref, useMap] : thenArefUse) {
        if (!arefUse[aref].putOp)
          arefUse[aref].putOp = useMap.putOp;
        if (!arefUse[aref].getOp)
          arefUse[aref].getOp = useMap.getOp;
      }

      if (ifOp.elseBlock()) {
        auto &elseRegion = ifOp.getElseRegion();
        auto elseBlock = &elseRegion.getBlocks().front();
        auto elseArefUse = AnalyzeArefUseInBlock(elseBlock, ArefFirstUseMap{});
        for (auto [aref, useMap] : elseArefUse) {
          if (!arefUse[aref].putOp)
            arefUse[aref].putOp = useMap.putOp;
          if (!arefUse[aref].getOp)
            arefUse[aref].getOp = useMap.getOp;
        }
      }
    }
  }
  return arefUse;
}

ArefLastPhaseMap arefPhaseAssignmentInBlock(Block *inpBlock,
                                            ArefLastPhaseMap arefLastPhase,
                                            OpBuilder &builder) {
  auto getNextPhase = [&](Operation *op, Value phase) {
    auto loc = op->getLoc();
    // Phase calculation assumes arefIdx increments by one with each use of
    // ArefPutEnterOp or ArefGetEnterOp, currently handled in
    // SplitWarpGroupLoops. Future ArefDepth pass will compute both arefIdx
    // and phase  simultaneously, as they are related.
    auto nextPhase = builder.create<arith::AddIOp>(
        loc, phase, builder.create<arith::ConstantIntOp>(loc, 1, 32));
    nextPhase->setAttr("next_phase", builder.getUnitAttr());
    return nextPhase;
  };

  SmallVector<Operation *> staleOps;
  for (auto &op : *inpBlock) {
    if (auto putEnterOp = dyn_cast<ArefPutEnterOp>(op)) {
      if (!putEnterOp.getPhase()) {
        assert(arefLastPhase.contains(putEnterOp.getAref()));
        auto &phase = arefLastPhase[putEnterOp.getAref()].put;
        putEnterOp.getPhaseMutable().assign(phase);
        builder.setInsertionPointAfter(putEnterOp);
        phase = getNextPhase(putEnterOp, phase);
      }
    } else if (auto getEnterOp = dyn_cast<ArefGetEnterOp>(op)) {
      if (!getEnterOp.getPhase()) {
        assert(arefLastPhase.contains(getEnterOp.getAref()));
        auto &phase = arefLastPhase[getEnterOp.getAref()].get;
        getEnterOp.getPhaseMutable().assign(phase);
        builder.setInsertionPointAfter(getEnterOp);
        phase = getNextPhase(getEnterOp, phase);
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto arefUseInBlock = AnalyzeArefUseInBlock(
          &forOp.getRegion().getBlocks().front(), ArefFirstUseMap{});
      if (arefUseInBlock.empty())
        continue;

      // there are arefs used in the loop-body, recrusively assigned phase
      // there

      SmallVector<Value> initArgs(forOp.getInitArgs().begin(),
                                  forOp.getInitArgs().end());
      // add initial phases to the loop
      SmallVector<Value *> arefLastPhaseArgs;
      for (auto [aref, useMap] : arefUseInBlock) {
        if (useMap.putOp) {
          initArgs.push_back(arefLastPhase[aref].put);
          arefLastPhaseArgs.push_back(&arefLastPhase[aref].put);
        }
        if (useMap.getOp) {
          initArgs.push_back(arefLastPhase[aref].get);
          arefLastPhaseArgs.push_back(&arefLastPhase[aref].get);
        }
      }

      // create new forOp
      builder.setInsertionPoint(forOp);
      scf::ForOp newForOp = builder.create<mlir::scf::ForOp>(
          forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
          forOp.getStep(), initArgs);

      // update uses
      for (auto [oldArg, newArg] :
           llvm::zip(forOp.getResults(), newForOp.getResults())) {
        oldArg.replaceAllUsesWith(newArg);
      }
      // move loop body
      newForOp.getRegion().takeBody(forOp.getRegion());

      // update the last phase value to use inside the loop body
      int nOld = forOp.getResults().size();
      int nNew = newForOp.getResults().size();
      int n = nNew - nOld;
      assert(n == arefLastPhaseArgs.size());
      for (int i = 0; i < n; ++i) {
        auto val = newForOp.getBody()->addArgument(
            newForOp.getResult(nOld + i).getType(),
            newForOp.getResult(nOld + i).getLoc());
        *arefLastPhaseArgs[i] = val;
      }

      // assign phases in the loop body
      auto arefLastPhaseInBlock = arefPhaseAssignmentInBlock(
          &newForOp.getRegion().getBlocks().front(), arefLastPhase, builder);

      // update yieldOp to return new phases
      auto yieldOp =
          mlir::cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
      SmallVector<Value> newYieldVals(yieldOp.getOperands().begin(),
                                      yieldOp.getOperands().end());
      for (auto [aref, useMap] : arefUseInBlock) {
        if (useMap.putOp)
          newYieldVals.push_back(arefLastPhaseInBlock[aref].put);
        if (useMap.getOp)
          newYieldVals.push_back(arefLastPhaseInBlock[aref].get);
      }
      builder.setInsertionPoint(yieldOp);
      builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldVals);
      yieldOp.erase();

      // finaly, update phases with results from newForOp
      for (int i = 0; i < n; ++i) {
        *arefLastPhaseArgs[i] = newForOp.getResults()[nOld + i];
      }

      staleOps.push_back(forOp);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // do the same with if-then/else blocks
      auto arefUseInThenBlock =
          AnalyzeArefUseInBlock(ifOp.thenBlock(), ArefFirstUseMap{});
      if (arefUseInThenBlock.empty())
        continue;

      bool hasElseBlock = ifOp.elseBlock();

      auto arefUseInElseBlock =
          ifOp.elseBlock()
              ? AnalyzeArefUseInBlock(ifOp.elseBlock(), ArefFirstUseMap{})
              : ArefFirstUseMap{};
      ArefFirstUseMap arefUseInIfOp = arefUseInThenBlock;

      for (auto [aref, useMap] : arefUseInElseBlock) {
        if (!arefUseInIfOp[aref].putOp)
          arefUseInIfOp[aref].putOp = useMap.putOp;
        if (!arefUseInIfOp[aref].getOp)
          arefUseInIfOp[aref].getOp = useMap.getOp;
      }

      SmallVector<Type, 4> newIfResultTypes(ifOp.getResultTypes().begin(),
                                            ifOp.getResultTypes().end());
      SmallVector<Value *> arefLastPhaseArgs;
      for (auto [aref, useMap] : arefUseInIfOp) {
        if (useMap.putOp) {
          arefLastPhaseArgs.push_back(&arefLastPhase[aref].put);
          newIfResultTypes.push_back(arefLastPhase[aref].put.getType());
        }
        if (useMap.getOp) {
          arefLastPhaseArgs.push_back(&arefLastPhase[aref].get);
          newIfResultTypes.push_back(arefLastPhase[aref].get.getType());
        }
      }

      builder.setInsertionPoint(ifOp);
      auto newIfOp = builder.create<scf::IfOp>(ifOp.getLoc(), newIfResultTypes,
                                               ifOp.getCondition(),
                                               /*withElseRegion=*/true);

      int nOld = ifOp.getResults().size();
      int nNew = newIfOp.getResults().size();
      int n = nNew - nOld;
      assert(n == arefLastPhaseArgs.size());

      for (auto [oldArg, newArg] :
           llvm::zip(ifOp.getResults(), newIfOp.getResults())) {
        oldArg.replaceAllUsesWith(newArg);
      }
      newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
      if (ifOp.elseBlock())
        newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());

      auto arefLastPhaseInThenBlock = arefPhaseAssignmentInBlock(
          newIfOp.thenBlock(), arefLastPhase, builder);
      auto arefLastPhaseInElseBlock =
          ifOp.elseBlock() ? arefPhaseAssignmentInBlock(newIfOp.elseBlock(),
                                                        arefLastPhase, builder)
                           : arefLastPhase;

      // update yieldOp to return new phases
      {
        auto thenYieldOp =
            mlir::cast<scf::YieldOp>(newIfOp.thenBlock()->getTerminator());
        SmallVector<Value> newThenYieldVals(thenYieldOp.getOperands().begin(),
                                            thenYieldOp.getOperands().end());
        for (auto [aref, useMap] : arefUseInIfOp) {
          if (useMap.putOp)
            newThenYieldVals.push_back(arefLastPhaseInThenBlock[aref].put);
          if (useMap.getOp)
            newThenYieldVals.push_back(arefLastPhaseInThenBlock[aref].get);
        }
        builder.setInsertionPoint(thenYieldOp);
        builder.create<scf::YieldOp>(thenYieldOp.getLoc(), newThenYieldVals);
        thenYieldOp.erase();
      }
      if (hasElseBlock) {
        auto elseYieldOp =
            mlir::cast<scf::YieldOp>(newIfOp.elseBlock()->getTerminator());
        SmallVector<Value> newElseYieldVals(elseYieldOp.getOperands().begin(),
                                            elseYieldOp.getOperands().end());
        for (auto [aref, useMap] : arefUseInIfOp) {
          if (useMap.putOp) {
            newElseYieldVals.push_back(arefLastPhaseInElseBlock[aref].put);
          }
          if (useMap.getOp)
            newElseYieldVals.push_back(arefLastPhaseInElseBlock[aref].get);
        }
        builder.setInsertionPoint(elseYieldOp);
        builder.create<scf::YieldOp>(elseYieldOp.getLoc(), newElseYieldVals);
        elseYieldOp.erase();
      } else {
        // if there is no elseBlock in ifOp, we still need to create one
        // because we return updated phase values
        SmallVector<Value> newElseYieldVals;
        for (auto [aref, useMap] : arefUseInIfOp) {
          if (useMap.putOp) {
            newElseYieldVals.push_back(arefLastPhaseInElseBlock[aref].put);
          }
          if (useMap.getOp)
            newElseYieldVals.push_back(arefLastPhaseInElseBlock[aref].get);
        }
        // sanity check, if we have nothing to return, we should rewrite ifOp
        assert(!newElseYieldVals.empty());
        OpBuilder builder = OpBuilder::atBlockEnd(newIfOp.elseBlock());
        builder.create<scf::YieldOp>(newIfOp.getLoc(), newElseYieldVals);
      }

      for (int i = 0; i < n; ++i) {
        // update arefCounter vlaues with results from newForOp
        *arefLastPhaseArgs[i] = newIfOp.getResults()[nOld + i];
      }

      staleOps.push_back(ifOp);
    }
  }
  for (auto op : staleOps)
    op->erase();
  return arefLastPhase;
}

LogicalResult arefPhaseAssignment(ModuleOp mod) {
  // TODO: Verify that if a put/get already has a phase assigned, then all
  // put/get operations associated with the same aref must have a phase
  // assigned. Mixing and matching is not permitted.
  auto funcOp = getFuncOp(mod);
  OpBuilder builder(funcOp);
  SmallVector<WarpGroupOp> wgOps;
  funcOp.walk([&](WarpGroupOp wgOp) { wgOps.push_back(wgOp); });

  // Verify that if a put/get already has a phase assigned, then all
  // put/get operations associated with the same aref must have a phase
  // assigned. Mixing and matching is not permitted.
  for (auto arefOp : funcOp.getOps<ArefCreateOp>()) {
    bool putHasPhase = false, getHasPhase = false;
    for (auto uses : arefOp->getUsers()) {
      if (auto putEnterOp = dyn_cast<ArefPutEnterOp>(uses))
        putHasPhase |= (bool)putEnterOp.getPhase();
      else if (auto getEnterOp = dyn_cast<ArefGetEnterOp>(uses))
        getHasPhase |= (bool)getEnterOp.getPhase();
    }

    for (auto uses : arefOp->getUsers()) {
      if (auto putEnterOp = dyn_cast<ArefPutEnterOp>(uses)) {
        assert(putHasPhase == (bool)putEnterOp.getPhase());
      } else if (auto getEnterOp = dyn_cast<ArefGetEnterOp>(uses)) {
        assert(getHasPhase == (bool)getEnterOp.getPhase());
      }
    }

    // If phase needs to be assigned, verify that all puts/get are in the same
    // warp-group. We do not currently support if some put/get in one
    // warp-group and others are in some other.
    WarpGroupOp wgPut, wgGet;
    for (auto uses : arefOp->getUsers()) {
      if (auto putEnterOp = dyn_cast<ArefPutEnterOp>(uses)) {
        if (!putEnterOp.getPhase()) {
          if (wgPut)
            assert(wgPut == putEnterOp->getParentOfType<WarpGroupOp>());
          else
            wgPut = putEnterOp->getParentOfType<WarpGroupOp>();
        }
      } else if (auto getEnterOp = dyn_cast<ArefGetEnterOp>(uses)) {
        if (!getEnterOp.getPhase()) {
          if (wgGet)
            assert(wgGet == getEnterOp->getParentOfType<WarpGroupOp>());
          else
            wgGet = getEnterOp->getParentOfType<WarpGroupOp>();
        }
      }
    }
  }

  // gather all arefUses
  ArefFirstUseMap arefUse;
  for (auto wgOp : wgOps) {
    auto block = &wgOp.getRegions().front()->getBlocks().front();
    auto arefUseBlock = AnalyzeArefUseInBlock(block, ArefFirstUseMap{});
    for (auto [aref, useMap] : arefUseBlock) {
      if (!arefUse[aref].putOp)
        arefUse[aref].putOp = useMap.putOp;
      if (!arefUse[aref].getOp)
        arefUse[aref].getOp = useMap.getOp;
    }
  }

  // initialize put/get phases
  ArefLastPhaseMap arefLastPhase;
  for (auto [aref, useMap] : arefUse) {
    builder.setInsertionPointAfter(aref.getDefiningOp());
    arefLastPhase[aref].put =
        builder.create<arith::ConstantIntOp>(aref.getLoc(), 0, 32);
    arefLastPhase[aref].get =
        builder.create<arith::ConstantIntOp>(aref.getLoc(), 0, 32);
  }

  for (auto wgOp : wgOps) {
    auto block = &wgOp.getRegions().front()->getBlocks().front();
    arefPhaseAssignmentInBlock(block, arefLastPhase, builder);
  }
  return success();
}

void lowerTMAloads(ArefPutEnterOp op, PatternRewriter &rewriter,
                   ArefValue arefVal) {
  auto loc = op.getLoc();
  // for now handle TMA loads in PutEnterOp
  SmallVector<Operation *> descLoads;
  SmallVector<LocalStoreOp> storeOps;
  for (auto result : op.getResults())
    for (auto user : result.getUsers()) {
      // idenfity users of buffer a LoadDescriptorOp + LocalStoreOp
      if (auto localStore = dyn_cast<triton::gpu::LocalStoreOp>(user)) {
        auto maybeLoad = localStore.getSrc().getDefiningOp();
        if (isa<DescriptorLoadOp, DescriptorGatherOp>(maybeLoad)) {
          descLoads.push_back(maybeLoad);
          storeOps.push_back(localStore);
        }
      }
    }
  assert(descLoads.size() <= op.getResults().size());
  if (descLoads.empty())
    return;

  // matching ArefPutExitOp is assumed to textually follow ArefPutEnterOp
  //   %bufs:n = aref_put.enter %aref[%enter_idx]
  //   tma_load %bufs[0]
  //   ..
  //   tma_load %bufs[n-1]
  //   aref_put.exit %aref[%exit_idx]

  // locate the following aref_put.exit, to get full barrier
  auto nextOp = descLoads.back()->getNextNode();
  while (nextOp) {
    if (isa<ArefPutExitOp>(nextOp))
      break;
    nextOp = nextOp->getNextNode();
  }
  assert(nextOp && "Expecting ArefPutExitOp");
  auto arefPutExitOp = cast<ArefPutExitOp>(nextOp);
  assert(arefPutExitOp.getAref() == op.getAref() &&
         "Expecting matching Aref on the ArefPutExitOp");

  Value fullBarrier = getFullBarrierAt(op.getContext(), loc, rewriter, arefVal,
                                       arefPutExitOp.getIndex());
  nvidia_gpu::createBarrierExpectOp(loc, rewriter, descLoads, fullBarrier);

  for (auto [op, storeOp] : llvm::zip(descLoads, storeOps)) {
    Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    auto alloc = cast<TypedValue<MemDescType>>(storeOp.getDst());
    if (auto descLoad = dyn_cast<DescriptorLoadOp>(op)) {
      nvidia_gpu::createTMALoad(descLoad, rewriter, fullBarrier, alloc, pred);
    } else {
      nvidia_gpu::createTMAGather(cast<DescriptorGatherOp>(op), rewriter,
                                  fullBarrier, alloc, pred);
    }
    replaceUsesWithLocalLoad(rewriter, op->getResult(0), alloc);
    op->erase();
  }
}

LogicalResult rewritePutEnterOp(ArefCreateOp arefOp, ArefPutEnterOp op,
                                PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  // get empty barrier at a given stage
  Value emptyBarrier =
      getEmptyBarrierAt(op.getContext(), loc, rewriter, arefVal, op.getIndex());

  // get barrier phase, and issue barrier wait
  // put_phase = ((phase / depth) & 1) ^ 1
  Operation *phase = rewriter.create<arith::DivSIOp>(
      loc, op.getPhase(),
      rewriter.create<arith::ConstantIntOp>(loc, arefVal.depth, 32));
  phase->setAttr("put_phase", rewriter.getUnitAttr());
  phase = rewriter.create<arith::AndIOp>(
      loc, phase->getResult(0),
      rewriter.create<arith::ConstantIntOp>(loc, 1, 32));
  phase->setAttr("put_phase", rewriter.getUnitAttr());
  phase = rewriter.create<arith::XOrIOp>(
      loc, phase->getResult(0),
      rewriter.create<arith::ConstantIntOp>(loc, 1, 32));
  phase->setAttr("put_phase", rewriter.getUnitAttr());
  rewriter.create<triton::nvidia_gpu::WaitBarrierOp>(loc, emptyBarrier,
                                                     phase->getResult(0));

  // generate views
  auto stage = rewriter.create<arith::RemSIOp>(
      loc, op.getIndex(),
      rewriter.create<arith::ConstantIntOp>(loc, arefVal.depth, 32));
  stage->setAttr("put_view", rewriter.getUnitAttr());
  auto views = getSubViews(arefVal, stage, loc, rewriter);
  assert(views.size() == op.getResults().size());

  // TMA load need special handling as it requires fullMbarrier that we need
  // to get from matching ArefPutExitOp
  lowerTMAloads(op, rewriter, arefVal);

  // replaces uses with views
  for (int i = 0; i < arefVal.buffers.size(); ++i)
    op.getResult(i).replaceAllUsesWith(views[i]);

  return success();
}

LogicalResult rewriteGetEnterOp(ArefCreateOp arefOp, ArefGetEnterOp op,
                                PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  // get full barrier at a given stage
  Value fullBarrier =
      getFullBarrierAt(op.getContext(), loc, rewriter, arefVal, op.getIndex());

  // get barrier phase, and issue barrier wait
  // get_phase = (phase / depth) & 1
  Operation *phase = rewriter.create<arith::DivSIOp>(
      loc, op.getPhase(),
      rewriter.create<arith::ConstantIntOp>(loc, arefVal.depth, 32));
  phase->setAttr("get_phase", rewriter.getUnitAttr());
  phase = rewriter.create<arith::AndIOp>(
      loc, phase->getResult(0),
      rewriter.create<arith::ConstantIntOp>(loc, 1, 32));
  phase->setAttr("get_phase", rewriter.getUnitAttr());
  rewriter.create<triton::nvidia_gpu::WaitBarrierOp>(loc, fullBarrier,
                                                     phase->getResult(0));

  // update uses of views
  auto stage = rewriter.create<arith::RemSIOp>(
      loc, op.getIndex(),
      rewriter.create<arith::ConstantIntOp>(loc, arefVal.depth, 32));
  stage->setAttr("get_view", rewriter.getUnitAttr());
  auto views = getSubViews(arefVal, stage, loc, rewriter);
  assert(views.size() == op.getResults().size());

  for (int i = 0; i < arefVal.buffers.size(); ++i)
    op.getResult(i).replaceAllUsesWith(views[i]);

  return success();
}

nvws::TrackedAsyncOp translateArefProducerKind(ArefProducer producer) {
  nvws::TrackedAsyncOp trackedOp;
  if (producer == ArefProducer::UMMA)
    return nvws::TrackedAsyncOp::UMMA;
  else if (producer == ArefProducer::TMALDG)
    return nvws::TrackedAsyncOp::TMALDG;
  else if (producer == ArefProducer::LDGSTS)
    return nvws::TrackedAsyncOp::LDGSTS;
  else if (producer == ArefProducer::STS || producer == ArefProducer::STTM)
    return nvws::TrackedAsyncOp::NONE;
  else
    llvm_unreachable("unexpected producer kind");
}

SmallVector<nvws::TrackedAsyncOp>
translateArefProducerKind(const SmallVector<Attribute> &producers) {
  SmallVector<nvws::TrackedAsyncOp> trackedOps;
  for (auto producerAttr : producers) {
    auto kind = dyn_cast<ArefProducerAttr>(producerAttr).getValue();
    trackedOps.push_back(translateArefProducerKind(kind));
  }
  return trackedOps;
};

LogicalResult rewritePutExitOp(ArefCreateOp arefOp, ArefPutExitOp op,
                               PatternRewriter &rewriter, ArefValue arefVal) {
  SmallVector<Attribute> producers(op.getProducers().begin(),
                                   op.getProducers().end());
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  assert(producers.size() == arefOp.getOperands().size() &&
         "must have a producer per buffer");

  Value fullBarrier =
      getFullBarrierAt(op.getContext(), loc, rewriter, arefVal, op.getIndex());

  auto trackedOps = translateArefProducerKind(producers);
  assert(llvm::all_of(trackedOps,
                      [&](nvws::TrackedAsyncOp trackedOp) {
                        return trackedOp == trackedOps[0];
                      }) &&
         "cannot mix & match different producer kinds");

  if (trackedOps[0] != nvws::TrackedAsyncOp::TMALDG) {
    // For TMA, the arrive is done by HW
    rewriter.create<nvws::ArriveBarrierOp>(
        loc, fullBarrier,
        nvws::TrackedAsyncOpAttr::get(op.getContext(), trackedOps[0]));
  }

  return success();
}

nvws::TrackedAsyncOp translateArefConsumerKind(ArefConsumer consumer) {
  nvws::TrackedAsyncOp trackedOp;
  if (consumer == ArefConsumer::UMMA) {
    return nvws::TrackedAsyncOp::UMMA;
  } else if (consumer == ArefConsumer::LDS || consumer == ArefConsumer::LDTM ||
             consumer == ArefConsumer::WGMMA) {
    return nvws::TrackedAsyncOp::NONE;
  } else {
    llvm_unreachable("unexpected consumer kind");
  }
}

SmallVector<nvws::TrackedAsyncOp>
translateArefConsumerKind(const SmallVector<Attribute> &consumers) {
  SmallVector<nvws::TrackedAsyncOp> trackedOps;
  for (auto consumerAttr : consumers) {
    auto kind = dyn_cast<ArefConsumerAttr>(consumerAttr).getValue();
    trackedOps.push_back(translateArefConsumerKind(kind));
  }
  return trackedOps;
};

LogicalResult rewriteGetExitOp(ArefCreateOp arefOp, ArefGetExitOp op,
                               PatternRewriter &rewriter, ArefValue arefVal) {
  SmallVector<Attribute> consumers(op.getConsumers().begin(),
                                   op.getConsumers().end());
  rewriter.setInsertionPointAfter(op);
  auto loc = op.getLoc();
  assert(consumers.size() == arefOp.getOperands().size() &&
         "must have a consumer per buffer");
  Value emptyBarrier =
      getEmptyBarrierAt(op.getContext(), loc, rewriter, arefVal, op.getIndex());

  auto trackedOps = translateArefConsumerKind(consumers);
  assert(llvm::all_of(trackedOps,
                      [&](nvws::TrackedAsyncOp trackedOp) {
                        return trackedOp == trackedOps[0];
                      }) &&
         "cannot mix & match different consumer kinds");

  rewriter.create<nvws::ArriveBarrierOp>(loc, emptyBarrier, trackedOps[0]);
  return success();
}

class ArefCreateLowering : public OpRewritePattern<ArefCreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ArefCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto aref = createAndInitMbar(op, rewriter);
    llvm::SmallSetVector<Operation *, 10> opToDelete;
    opToDelete.insert(op.getOperation());
    for (auto userOp : op->getUsers()) {
      if (auto user = dyn_cast<ArefPutEnterOp>(userOp)) {
        opToDelete.insert(user);
        if (rewritePutEnterOp(op, user, rewriter, aref).failed())
          return failure();
      } else if (auto user = dyn_cast<ArefGetEnterOp>(userOp)) {
        opToDelete.insert(user);
        if (rewriteGetEnterOp(op, user, rewriter, aref).failed())
          return failure();
      } else if (auto user = dyn_cast<ArefPutExitOp>(userOp)) {
        opToDelete.insert(user);
        if (rewritePutExitOp(op, user, rewriter, aref).failed())
          return failure();
      } else if (auto user = dyn_cast<ArefGetExitOp>(userOp)) {
        opToDelete.insert(user);
        if (rewriteGetExitOp(op, user, rewriter, aref).failed())
          return failure();
      } else {
        llvm_unreachable("users of aref can only be ArefPut or ArefGet");
      }
    }

    for (auto it = opToDelete.rbegin(); it != opToDelete.rend(); ++it)
      rewriter.eraseOp(*it);

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
    // %47 = tt.trans %46 {order = array<i32: 1, 0>} :
    // !tt.memdesc<128x64xf16, #shared2, #ttg.shared_memory, mutable> ->
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
    // %47 = tt.trans %46 {order = array<i32: 1, 0>} :
    // !tt.memdesc<128x64xf16, #shared2, #ttg.shared_memory, mutable> ->
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
  MemDescType getArefbufMemDescType(MemDescType memDescType,
                                    int32_t AREF_SIZE) {
    auto shape = memDescType.getShape();
    SmallVector<int64_t> bufferShape(shape.begin(), shape.end());
    bufferShape.insert(bufferShape.begin(), AREF_SIZE);
    return MemDescType::get(bufferShape, memDescType.getElementType(),
                            memDescType.getEncoding(),
                            memDescType.getMemorySpace(), true);
  };

  void fixUpSync(ModuleOp m) {
    auto target = m->getAttrOfType<StringAttr>(AttrTargetName);

    m.walk([&](WarpGroupOp wgOp) {
      auto numWarps = wgOp.getNumWarps();

      if (isOpInGroup(wgOp, ATTR_WS_EPILOGUE)) {
        // When TMA store is not used, add barrier at the end of the
        // epilogue loop to make sure that warps are in sync. Without this,
        // there could be a race between the current iteration LDS of some
        // warps and the next iteration STS of other warps. When TMA store
        // is used, the barrier inserted by TMAStoreLowering does the same
        // job.
        wgOp.getRegion(0).front().walk([&](triton::StoreOp op) {
          OpBuilder builder(op);
          builder.setInsertionPointAfter(op);
          insertBarrier(builder, op->getLoc());
        });
      } else if ((isOpInGroup(wgOp, ATTR_WS_TMALOAD) ||
                  isOpInGroup(wgOp, ATTR_WS_MMA)) &&
                 numWarps > 1 && target == "cuda:100") {
        // On Blackwell, we sychronize warps at the beginning of the K loop
        // for load and MMA groups. A hang was observed without this
        // barrier.
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

  template <typename T>
  llvm::SmallSet<scf::ForOp, 10> getForOpsContaining(ModuleOp m) {
    llvm::SmallSet<scf::ForOp, 10> forOps;
    m.walk([&](T op) {
      if (auto forOp = op->template getParentOfType<scf::ForOp>()) {
	forOps.insert(forOp);
      }
    });
    return forOps;
  }

  void runOnOperation() override {
    // std::cout << "===== TritonNvidiaGPUArefLoweringPass =====\n";
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    LLVM_DEBUG({
      DBGS() << "Module before aref lowering:\n";
      m.dump();
    });

    if (arefPhaseAssignment(m).failed())
      signalPassFailure();

    // aref lowering
    mlir::RewritePatternSet arefPatterns(context);
    arefPatterns.add<ArefCreateLowering>(context);
    if (applyPatternsGreedily(m, std::move(arefPatterns)).failed()) {
      signalPassFailure();
    }

    // TMA store / gather lowering
    for (auto forOp : getForOpsContaining<DescriptorStoreLikeOpInterface>(m)) {
      pipelineTMAStores(forOp);
    }

    // Multibuffer TMA descriptor update
    for (auto forOp : getForOpsContaining<MakeTensorDescOp>(m)) {
      auto wgOp = forOp->getParentOfType<WarpGroupOp>();
      int numStages = getNumTMAStages(wgOp);
      lowerTMADescriptors(forOp, numStages + 1);
    }

    fixUpSync(m);

    // update trans op
    mlir::RewritePatternSet transPattern(context);
    transPattern.add<TransOpRewrite>(context);
    transPattern.add<MemDescTransOpRewrite>(context);
    // not sure why .failed() will always return true, so just remove it
    // here
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
