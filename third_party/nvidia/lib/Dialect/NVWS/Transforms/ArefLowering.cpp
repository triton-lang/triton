/*
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>

namespace mlir {

#define GEN_PASS_DEF_NVWSAREFLOWERINGPASS
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-aref-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
using namespace triton::nvws;

tt::FuncOp getFuncOp(ModuleOp moduleOp) {
  tt::FuncOp funcOp;
  moduleOp.walk([&](tt::FuncOp op) { funcOp = op; });
  return funcOp;
}

auto collectArefOps(ttng::ArefCreateOp arefOp) {
  SmallVector<ttng::ArefPutEnterOp> putEnterOps;
  SmallVector<ttng::ArefPutExitOp> putExitOps;
  SmallVector<ttng::ArefGetEnterOp> getEnterOps;
  SmallVector<ttng::ArefGetExitOp> getExitOps;

  auto useAref = [&](Operation *user) {
    while (!isa<ttng::ArefCreateOp>(user) && user->getOperands().size() > 0) {
      user = user->getOperand(0).getDefiningOp();
    }
    return user == arefOp.getOperation();
  };

  auto moduleOp = arefOp->getParentOfType<ModuleOp>();
  for (auto func : moduleOp.getOps<tt::FuncOp>()) {
    func.walk([&](Operation *op) {
      if (auto getEnterOp = dyn_cast<ttng::ArefGetEnterOp>(op)) {
        Operation *defOp =
            getEnterOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(defOp)) {
          getEnterOps.push_back(getEnterOp);
        }
      } else if (auto getExitOp = dyn_cast<ttng::ArefGetExitOp>(op)) {
        Operation *defOp =
            getExitOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(defOp)) {
          getExitOps.push_back(getExitOp);
        }
      } else if (auto putEnterOp = dyn_cast<ttng::ArefPutEnterOp>(op)) {
        Operation *defOp =
            putEnterOp.getOperation()->getOperand(0).getDefiningOp();
        if (useAref(defOp)) {
          putEnterOps.push_back(putEnterOp);
        }
      } else if (auto putExitOp = dyn_cast<ttng::ArefPutExitOp>(op)) {
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
  SmallVector<ttng::WarpGroupOp> wgOps = findWarpGroupOps(m);

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

ttg::MemDescType getBarrierMemDesc(MLIRContext *ctx, PatternRewriter &rewriter,
                                   llvm::ArrayRef<int64_t> shape) {
  Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
  auto barrierCTALayout = ttg::CTALayoutAttr::get(
      /*context=*/ctx, /*CTAsPerCGA=*/{1},
      /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding =
      ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0}, barrierCTALayout);
  return ttg::MemDescType::get(shape, rewriter.getI64Type(), barrierEncoding,
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
  return rewriter.create<ttg::MemDescSubviewOp>(loc, memDesc, mbars,
                                                offsetsVal);
}

Value getEmptyBarrierAt(MLIRContext *ctx, Location loc,
                        PatternRewriter &rewriter, ArefValue aref,
                        Value arefIdx) {
  // stage= arefIdx % depth
  Value stage;
  auto remsi = rewriter.create<arith::RemSIOp>(
      loc, arefIdx, rewriter.create<arith::ConstantIntOp>(loc, aref.depth, 32));
  remsi->setAttr("empty_mbar", rewriter.getUnitAttr());
  stage = remsi;

  return getBarrierAt(ctx, loc, rewriter, aref.emptyMbars, stage);
}

Value getFullBarrierAt(MLIRContext *ctx, Location loc,
                       PatternRewriter &rewriter, ArefValue aref,
                       Value arefIdx) {
  Value stage;
  auto remsi = rewriter.create<arith::RemSIOp>(
      loc, arefIdx, rewriter.create<arith::ConstantIntOp>(loc, aref.depth, 32));
  remsi->setAttr("full_mbar", rewriter.getUnitAttr());
  stage = remsi;
  return getBarrierAt(ctx, loc, rewriter, aref.fullMbars, stage);
}

std::pair<int, int> getArrivalCount(ttng::ArefCreateOp op) {
  std::optional<int> producerArrivalCount, consumerArrivalCount;

  for (auto user : op->getUsers()) {
    auto wgOp = user->getParentOfType<ttng::WarpGroupOp>();
    auto numWarps = wgOp.getNumWarps();

    if (auto putExitOp = dyn_cast<ttng::ArefPutExitOp>(user)) {
      int count = 0;
      for (auto prod : putExitOp.getProducers()) {
        auto kind = dyn_cast<ttng::ArefProducerAttr>(prod).getValue();
        switch (kind) {
        case ttng::ArefProducer::UMMA:
        case ttng::ArefProducer::TMALDG:
          count += 1;
          break;
        case ttng::ArefProducer::LDGSTS:
        case ttng::ArefProducer::STS:
        case ttng::ArefProducer::STTM:
          count += numWarps * 32;
          break;
        default:
          llvm_unreachable("unknown producer kind");
        }
      }

      if (producerArrivalCount) {
        assert(*producerArrivalCount == count &&
               "inconsistent producer arrival count");
      } else {
        producerArrivalCount = count;
      }
    } else if (auto getExitOp = dyn_cast<ttng::ArefGetExitOp>(user)) {
      int count = 0;
      for (auto consumer : getExitOp.getConsumers()) {
        auto kind = dyn_cast<ttng::ArefConsumerAttr>(consumer).getValue();
        switch (kind) {
        case ttng::ArefConsumer::UMMA:
          count += 1;
          break;
        case ttng::ArefConsumer::LDS:
        case ttng::ArefConsumer::WGMMA:
        case ttng::ArefConsumer::LDTM:
          count += numWarps * 32;
          break;
        default:
          llvm_unreachable("unknown consumer kind");
        }
      }

      if (consumerArrivalCount) {
        assert(*consumerArrivalCount == count &&
               "inconsistent consumer arrival count");
      } else {
        consumerArrivalCount = count;
      }
    }
  }

  assert(producerArrivalCount);
  assert(consumerArrivalCount);

  return {*producerArrivalCount, *consumerArrivalCount};
};

ArefValue createAndInitMbar(ttng::ArefCreateOp op, PatternRewriter &rewriter) {
  auto [producerArrivalCount, consumerArrivalCount] = getArrivalCount(op);

  MLIRContext *ctx = op.getContext();
  auto loc = op.getLoc();
  auto arefTy = op.getType();
  auto baseType = arefTy.getBaseType();
  auto arefBufTypes =
      llvm::to_vector(llvm::map_range(arefTy.getBaseType(), [](Type type) {
        return cast<ttg::MemDescType>(type);
      }));
  auto shape = arefBufTypes[0].getShape();
  auto depth = shape[0];

  ttg::MemDescType barrierMemDescType =
      getBarrierMemDesc(ctx, rewriter, {depth});
  // Create two mbarriers
  auto emptyMbars =
      rewriter.create<ttg::LocalAllocOp>(loc, barrierMemDescType, Value());
  emptyMbars->setAttr("aref_empty_mbarriers", rewriter.getUnitAttr());
  auto fullMbars =
      rewriter.create<ttg::LocalAllocOp>(loc, barrierMemDescType, Value());
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
    rewriter.create<ttng::InitBarrierOp>(loc, singleBarrier, arrivalCount);
  }

  return ArefValue{emptyMbars, fullMbars, static_cast<int>(depth),
                   op.getOperands()};
}

SmallVector<Value> getSubViews(ArefValue arefVal, Value stage, Location loc,
                               OpBuilder &rewriter) {
  SmallVector<Value> views;
  for (auto buffer : arefVal.buffers) {
    SmallVector<Value> offsetsVal{stage};
    auto memDescType = cast<ttg::MemDescType>(buffer.getType());
    auto shape = memDescType.getShape();
    auto rank = shape.size() - 1;

    for (int i = 0; i < rank; ++i) {
      offsetsVal.push_back(rewriter.create<arith::ConstantIntOp>(
          loc, 0, rewriter.getIntegerType(32)));
    }
    SmallVector<int64_t> tensorShape(shape.begin() + 1, shape.end());
    auto memDescTypeNew = ttg::MemDescType::get(
        tensorShape, memDescType.getElementType(), memDescType.getEncoding(),
        memDescType.getMemorySpace(), true);
    Value singleBuffer = rewriter.create<ttg::MemDescSubviewOp>(
        loc, memDescTypeNew, buffer, offsetsVal);
    views.push_back(singleBuffer);
  }

  return views;
}

void lowerAsyncLoads(ttng::ArefPutEnterOp op, PatternRewriter &rewriter,
                     ArefValue arefVal) {
  auto loc = op.getLoc();
  // for now handle TMA loads in PutEnterOp
  SmallVector<Operation *> loadOps;
  SmallVector<ttg::LocalStoreOp> storeOps;
  for (auto result : op.getResults())
    for (auto user : result.getUsers()) {
      // idenfity users of buffer a LoadDescriptorOp + LocalStoreOp
      if (auto localStore = dyn_cast<ttg::LocalStoreOp>(user)) {
        auto maybeLoad = localStore.getSrc().getDefiningOp();
        if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp, tt::LoadOp>(
                maybeLoad)) {
          loadOps.push_back(maybeLoad);
          storeOps.push_back(localStore);
        }
      }
    }
  assert(loadOps.size() <= op.getResults().size());
  if (loadOps.empty())
    return;

  // matching ArefPutExitOp is with ArefPutEnterOp
  // we use aref_tag to match the two
  //   %bufs:n = aref_put.enter %aref[%enter_idx] {aref_tag = tag}
  //   tma_load %bufs[0]
  //   ..
  //   tma_load %bufs[n-1]
  //   aref_put.exit %aref[%exit_idx] {aref_tag = tag}

  // locate the matching aref_put.exit with the same tag, to get full barrier
  ttng::ArefPutExitOp arefPutExitOp;
  auto arefTag = op->getAttrOfType<StringAttr>("aref_tag").str();
  for (auto user : op.getAref().getUsers()) {
    if (auto exitOp = dyn_cast<ttng::ArefPutExitOp>(user)) {
      if (exitOp->getAttrOfType<StringAttr>("aref_tag").str() == arefTag) {
        arefPutExitOp = exitOp;
        break;
      }
    }
  }
  assert(arefPutExitOp);
  assert(arefPutExitOp.getAref() == op.getAref() &&
         "Expecting matching Aref on the ArefPutExitOp");

  Value fullBarrier = getFullBarrierAt(op.getContext(), loc, rewriter, arefVal,
                                       arefPutExitOp.getIndex());
  ttng::createBarrierExpectOp(loc, rewriter, loadOps, fullBarrier);

  // Note: it is essential to set the insertion point right before putExitOp
  // otherwise one of the MOE kernel fails to compile, and if it is set
  // somewhere else it would result in  perf regression from:
  //
  // $ python python/tutorials/10-block-scaled-matmul-persistent.py --bench
  //   --ws --K_range 512 8192
  //
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(arefPutExitOp);
  for (auto [loadOp, storeOp] : llvm::zip(loadOps, storeOps)) {
    Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
    auto alloc = cast<TypedValue<ttg::MemDescType>>(storeOp.getDst());
    if (auto descLoad = dyn_cast<tt::DescriptorLoadOp>(loadOp)) {
      ttng::createTMALoad(descLoad, rewriter, fullBarrier, alloc, pred);
    } else if (auto descGather = dyn_cast<tt::DescriptorGatherOp>(loadOp)) {
      ttng::createTMAGather(descGather, rewriter, fullBarrier, alloc, pred);
    } else if (auto load = dyn_cast<tt::LoadOp>(loadOp)) {
      rewriter.create<ttg::AsyncCopyGlobalToLocalOp>(
          loc, load.getPtr(), alloc, load.getMask(), load.getOther(),
          load.getCache(), load.getEvict(), load.getIsVolatile());
    } else {
      llvm_unreachable("Unknown load op");
    }
    tt::replaceUsesWithLocalLoad(rewriter, loadOp->getResult(0), alloc);
    loadOp->erase();
  }
}

LogicalResult rewritePutEnterOp(ttng::ArefCreateOp arefOp,
                                ttng::ArefPutEnterOp op,
                                PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  // get empty barrier at a given stage
  Value emptyBarrier =
      getEmptyBarrierAt(op.getContext(), loc, rewriter, arefVal, op.getIndex());

  // get barrier phase, and issue barrier wait
  // put_phase = ((phase / depth) & 1) ^ 1
  Operation *phase = rewriter.create<arith::DivSIOp>(
      loc, op.getIndex(),
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
  rewriter.create<ttng::WaitBarrierOp>(loc, emptyBarrier, phase->getResult(0));

  // generate views
  Value stage;
  auto arefIdx = op.getIndex();
  auto remsi = rewriter.create<arith::RemSIOp>(
      loc, arefIdx,
      rewriter.create<arith::ConstantIntOp>(loc, arefVal.depth, 32));
  remsi->setAttr("put_view", rewriter.getUnitAttr());
  stage = remsi;
  auto views = getSubViews(arefVal, stage, loc, rewriter);
  assert(views.size() == op.getResults().size());

  // TMA and cpasync load need special handling as it requires fullMbarrier that
  // we need to get from matching ArefPutExitOp
  lowerAsyncLoads(op, rewriter, arefVal);

  // replaces uses with views
  for (int i = 0; i < arefVal.buffers.size(); ++i)
    op.getResult(i).replaceAllUsesWith(views[i]);

  return success();
}

LogicalResult rewriteGetEnterOp(ttng::ArefCreateOp arefOp,
                                ttng::ArefGetEnterOp op,
                                PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  // get full barrier at a given stage
  Value fullBarrier =
      getFullBarrierAt(op.getContext(), loc, rewriter, arefVal, op.getIndex());

  // get barrier phase, and issue barrier wait
  // get_phase = (phase / depth) & 1
  Operation *phase = rewriter.create<arith::DivSIOp>(
      loc, op.getIndex(),
      rewriter.create<arith::ConstantIntOp>(loc, arefVal.depth, 32));
  phase->setAttr("get_phase", rewriter.getUnitAttr());
  phase = rewriter.create<arith::AndIOp>(
      loc, phase->getResult(0),
      rewriter.create<arith::ConstantIntOp>(loc, 1, 32));
  phase->setAttr("get_phase", rewriter.getUnitAttr());
  rewriter.create<ttng::WaitBarrierOp>(loc, fullBarrier, phase->getResult(0));

  // update uses of views
  Value stage;
  auto arefIdx = op.getIndex();
  auto remsi = rewriter.create<arith::RemSIOp>(
      loc, arefIdx,
      rewriter.create<arith::ConstantIntOp>(loc, arefVal.depth, 32));
  remsi->setAttr("get_view", rewriter.getUnitAttr());
  stage = remsi;
  auto views = getSubViews(arefVal, stage, loc, rewriter);
  assert(views.size() == op.getResults().size());

  for (int i = 0; i < arefVal.buffers.size(); ++i)
    op.getResult(i).replaceAllUsesWith(views[i]);

  return success();
}

TrackedAsyncOp translateArefProducerKind(ttng::ArefProducer producer) {
  TrackedAsyncOp trackedOp;
  if (producer == ttng::ArefProducer::UMMA)
    return TrackedAsyncOp::UMMA;
  else if (producer == ttng::ArefProducer::TMALDG)
    return TrackedAsyncOp::TMALDG;
  else if (producer == ttng::ArefProducer::LDGSTS)
    return TrackedAsyncOp::LDGSTS;
  else if (producer == ttng::ArefProducer::STS ||
           producer == ttng::ArefProducer::STTM ||
           producer == ttng::ArefProducer::NONE)
    return TrackedAsyncOp::NONE;
  else
    llvm_unreachable("unexpected producer kind");
}

SmallVector<TrackedAsyncOp> translateArefProducerKind(ArrayAttr producers) {
  SmallVector<TrackedAsyncOp> trackedOps;
  for (auto producerAttr : producers) {
    auto kind = dyn_cast<ttng::ArefProducerAttr>(producerAttr).getValue();
    trackedOps.push_back(translateArefProducerKind(kind));
  }
  return trackedOps;
};

LogicalResult rewritePutExitOp(ttng::ArefCreateOp arefOp,
                               ttng::ArefPutExitOp op,
                               PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  Value fullBarrier =
      getFullBarrierAt(op.getContext(), loc, rewriter, arefVal, op.getIndex());

  auto trackedOps = translateArefProducerKind(op.getProducers());
  for (auto trackedOp : trackedOps) {
    if (trackedOp != TrackedAsyncOp::TMALDG) {
      // For TMA, the arrive is done by HW
      rewriter.create<ArriveBarrierOp>(
          loc, fullBarrier,
          TrackedAsyncOpAttr::get(op.getContext(), trackedOp));
    }
  }

  return success();
}

TrackedAsyncOp translateArefConsumerKind(ttng::ArefConsumer consumer) {
  TrackedAsyncOp trackedOp;
  if (consumer == ttng::ArefConsumer::UMMA) {
    return TrackedAsyncOp::UMMA;
  } else if (consumer == ttng::ArefConsumer::LDS ||
             consumer == ttng::ArefConsumer::LDTM ||
             consumer == ttng::ArefConsumer::WGMMA ||
             consumer == ttng::ArefConsumer::NONE) {
    return TrackedAsyncOp::NONE;
  } else {
    llvm_unreachable("unexpected consumer kind");
  }
}

SmallVector<TrackedAsyncOp> translateArefConsumerKind(ArrayAttr consumers) {
  SmallVector<TrackedAsyncOp> trackedOps;
  for (auto consumerAttr : consumers) {
    auto kind = dyn_cast<ttng::ArefConsumerAttr>(consumerAttr).getValue();
    trackedOps.push_back(translateArefConsumerKind(kind));
  }
  return trackedOps;
};

LogicalResult rewriteGetExitOp(ttng::ArefCreateOp arefOp,
                               ttng::ArefGetExitOp op,
                               PatternRewriter &rewriter, ArefValue arefVal) {
  rewriter.setInsertionPointAfter(op);
  auto loc = op.getLoc();
  Value emptyBarrier =
      getEmptyBarrierAt(op.getContext(), loc, rewriter, arefVal, op.getIndex());

  auto trackedOps = translateArefConsumerKind(op.getConsumers());
  for (auto trackedOp : trackedOps) {
    rewriter.create<ArriveBarrierOp>(
        loc, emptyBarrier, TrackedAsyncOpAttr::get(op.getContext(), trackedOp));
  }

  return success();
}

class ArefCreateLowering : public OpRewritePattern<ttng::ArefCreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::ArefCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto aref = createAndInitMbar(op, rewriter);
    llvm::SmallSetVector<Operation *, 10> opToDelete;
    opToDelete.insert(op.getOperation());
    for (auto userOp : op->getUsers()) {
      if (auto user = dyn_cast<ttng::ArefPutEnterOp>(userOp)) {
        opToDelete.insert(user);
        if (rewritePutEnterOp(op, user, rewriter, aref).failed())
          return failure();
      } else if (auto user = dyn_cast<ttng::ArefGetEnterOp>(userOp)) {
        opToDelete.insert(user);
        if (rewriteGetEnterOp(op, user, rewriter, aref).failed())
          return failure();
      } else if (auto user = dyn_cast<ttng::ArefPutExitOp>(userOp)) {
        opToDelete.insert(user);
        if (rewritePutExitOp(op, user, rewriter, aref).failed())
          return failure();
      } else if (auto user = dyn_cast<ttng::ArefGetExitOp>(userOp)) {
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

class TransOpRewrite : public OpRewritePattern<tt::TransOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::TransOp transOp,
                                PatternRewriter &rewriter) const override {
    // FIXME: Hacky way to update TransOp layout
    // need to ensure the output type also has the "mutable" attribute
    // %47 = tt.trans %46 {order = array<i32: 1, 0>} :
    // !tt.memdesc<128x64xf16, #shared2, #ttg.shared_memory, mutable> ->
    // !tt.memdesc<64x128xf16, #shared3, #ttg.shared_memory, mutable>
    if (!transOp->hasAttr("mutable")) {
      rewriter.setInsertionPoint(transOp);
      tt::TransOp newTransOp = rewriter.create<tt::TransOp>(
          transOp.getLoc(), transOp.getOperand(), transOp.getOrderAttr());
      newTransOp->setAttr("mutable", rewriter.getUnitAttr());
      transOp.replaceAllUsesWith(newTransOp.getResult());
    }
    return success();
  }
};

class MemDescTransOpRewrite : public OpRewritePattern<ttg::MemDescTransOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttg::MemDescTransOp memDescTransOp,
                                PatternRewriter &rewriter) const override {
    // FIXME: Hacky way to update TransOp layout
    // need to ensure the output type also has the "mutable" attribute
    // %47 = tt.trans %46 {order = array<i32: 1, 0>} :
    // !tt.memdesc<128x64xf16, #shared2, #ttg.shared_memory, mutable> ->
    // !tt.memdesc<64x128xf16, #shared3, #ttg.shared_memory, mutable>
    if (!memDescTransOp->hasAttr("mutable")) {
      rewriter.setInsertionPoint(memDescTransOp);
      ttg::MemDescTransOp newTransOp = rewriter.create<ttg::MemDescTransOp>(
          memDescTransOp.getLoc(), memDescTransOp.getOperand(),
          memDescTransOp.getOrderAttr());
      newTransOp->setAttr("mutable", rewriter.getUnitAttr());
      memDescTransOp.replaceAllUsesWith(newTransOp.getResult());
    }
    return success();
  }
};

int getNumTMAStages(ttng::WarpGroupOp wgOp) {
  if (isOpInGroup(wgOp, ATTR_WS_TMALOAD)) {
    auto m = wgOp->getParentOfType<ModuleOp>();
    return ttg::TritonGPUDialect::getNumStages(m);
  } else if (isOpInGroup(wgOp, ATTR_WS_EPILOGUE)) {
    return 1; // TMA store is not pipelined
  } else {
    llvm_unreachable("Descriptor update in an unexpected warp group.\n");
  }
  return 0;
}

} // namespace

class NVWSArefLoweringPass
    : public impl::NVWSArefLoweringPassBase<NVWSArefLoweringPass> {

  using impl::NVWSArefLoweringPassBase<
      NVWSArefLoweringPass>::NVWSArefLoweringPassBase;

public:
  ttg::MemDescType getArefbufMemDescType(ttg::MemDescType memDescType,
                                         int32_t AREF_SIZE) {
    auto shape = memDescType.getShape();
    SmallVector<int64_t> bufferShape(shape.begin(), shape.end());
    bufferShape.insert(bufferShape.begin(), AREF_SIZE);
    return ttg::MemDescType::get(bufferShape, memDescType.getElementType(),
                                 memDescType.getEncoding(),
                                 memDescType.getMemorySpace(), true);
  };

  void fixUpSync(ModuleOp m) {
    auto target = m->getAttrOfType<StringAttr>(ttg::AttrTargetName);

    m.walk([&](ttng::WarpGroupOp wgOp) {
      auto numWarps = wgOp.getNumWarps();

      if (isOpInGroup(wgOp, ATTR_WS_EPILOGUE)) {
        // When TMA store is not used, add barrier at the end of the
        // epilogue loop to make sure that warps are in sync. Without this,
        // there could be a race between the current iteration LDS of some
        // warps and the next iteration STS of other warps. When TMA store
        // is used, the barrier inserted by TMAStoreLowering does the same
        // job.
        wgOp.getRegion(0).front().walk([&](tt::StoreOp op) {
          OpBuilder builder(op);
          builder.setInsertionPointAfter(op);
          insertBarrier(builder, op->getLoc());
        });
      } else if ((isManuallyGrouped(m)                  //
                  || isOpInGroup(wgOp, ATTR_WS_TMALOAD) //
                  || isOpInGroup(wgOp, ATTR_WS_MMA)     //
                  ) &&
                 (numWarps > 1 && target == "cuda:100")) {
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

  void removeTokensFromArefs(tt::FuncOp funcOp) {
    funcOp.walk([&](ttng::ArefEnterOpInterface enterOp) {
      if (enterOp.getTokens().empty())
        return;
      SmallVector<Type> buffers, tokens;
      for (auto buffer : enterOp.getBuffers())
        buffers.push_back(buffer.getType());
      OpBuilder b(enterOp);
      auto newEnterOp =
          isa<ttng::ArefPutEnterOp>(enterOp)
              ? b.create<ttng::ArefPutEnterOp>(enterOp->getLoc(), buffers,
                                               tokens, enterOp.getAref(),
                                               enterOp.getIndex())
              : b.create<ttng::ArefGetEnterOp>(enterOp->getLoc(), buffers,
                                               tokens, enterOp.getAref(),
                                               enterOp.getIndex());
      // copy attributes from enterOp to newEnterOp
      if (enterOp->hasAttr("groups"))
        newEnterOp->setAttr("groups", enterOp->getAttr("groups"));
      newEnterOp->setAttr("aref_tag", enterOp->getAttr("aref_tag"));

      SmallVector<Value> replacements = newEnterOp->getResults();
      Value dummy = b.create<ub::PoisonOp>(funcOp.getLoc(),
                                           b.getType<ttg::AsyncTokenType>());
      for (auto _ : newEnterOp->getResults())
        replacements.push_back(dummy);
      enterOp->replaceAllUsesWith(replacements);
      enterOp->erase();
    });
  }

  void removeStaleAllocs(triton::FuncOp funcOp) {
    SmallVector<ttng::TMEMAllocOp> allocOps;
    funcOp.walk(
        [&](ttng::TMEMAllocOp allocOp) { allocOps.push_back(allocOp); });
    for (auto allocOp : allocOps)
      if (allocOp.use_empty())
        allocOp->erase();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    LLVM_DEBUG({
      DBGS() << "Module before aref lowering:\n";
      m.dump();
    });

    m.walk([&](tt::FuncOp funcOp) {
      removeTokensFromArefs(funcOp);
      removeStaleAllocs(funcOp);
    });
    LLVM_DEBUG({
      DBGS() << "after::removeAsyncTokensFromArefs:\n" << m << "\n";
    });

    // aref lowering
    mlir::RewritePatternSet arefPatterns(context);
    arefPatterns.add<ArefCreateLowering>(context);
    if (applyPatternsGreedily(m, std::move(arefPatterns)).failed()) {
      signalPassFailure();
    }

    // TMA store / gather lowering
    for (auto forOp :
         getForOpsContaining<tt::DescriptorStoreLikeOpInterface>(m)) {
      tt::pipelineTMAStores(forOp);
    }

    // Multibuffer TMA descriptor update
    for (auto forOp : getForOpsContaining<tt::MakeTensorDescOp>(m)) {
      auto wgOp = forOp->getParentOfType<ttng::WarpGroupOp>();
      int numStages = getNumTMAStages(wgOp);
      ttg::lowerTMADescriptors(forOp, numStages + 1);
    }

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

} // namespace mlir
