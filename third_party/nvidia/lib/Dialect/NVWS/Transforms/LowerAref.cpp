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

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

#define DEBUG_TYPE "nvws-lower-aref"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSLOWERAREF
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

// ----------------------------------------------------------------------------

void assignStageCluster(Operation *op, StageCluster stageCluster,
                        OpBuilder &builder) {
  if (stageCluster) {
    op->setAttr(triton::kLoopStageAttrName,
                builder.getI32IntegerAttr(stageCluster->first));
    op->setAttr(triton::kLoopClusterAttrName,
                builder.getI32IntegerAttr(stageCluster->second));
  }
}

struct ArefValue {
  Value emptyMbars;
  Value fullMbars;
  int depth;
  SmallVector<Value> buffers;
};

Value getEmptyBarrier(PatternRewriter &rewriter, Location loc, ArefValue aref,
                      Value stage) {
  return createSingleBufferView(rewriter, aref.emptyMbars, stage);
}

Value getFullBarrier(PatternRewriter &rewriter, Location loc, ArefValue aref,
                     Value stage) {
  return createSingleBufferView(rewriter, aref.fullMbars, stage);
}

std::pair<WarpGroupOp, int> getWarpGroupIdx(Operation *op) {
  if (auto wgOp = dyn_cast<WarpGroupOp>(op->getParentOp())) {
    auto region = op->getParentRegion();
    return {wgOp, region->getRegionNumber()};
  }
  if (isa<triton::FuncOp>(op))
    return {nullptr, -1};
  return getWarpGroupIdx(op->getParentOp());
}

struct BarrierCount {
  int producerPendingCount;
  int consumerPendingCount;
};

BarrierCount getArrivalCount(ArefCreateOp op) {
  std::optional<int> producerPendingCount, consumerPendingCount;
  SetVector<int> consumerGroups;

  for (auto user : op->getUsers()) {
    auto [wgOp, idx] = getWarpGroupIdx(user);
    auto numWarps = wgOp.getNumWarps()[idx];

    if (auto putExitOp = dyn_cast<ArefPutExitOp>(user)) {
      int pendingCount = 0;
      for (auto prod : putExitOp.getAsyncOps()) {
        auto kind = dyn_cast<AsyncOpAttr>(prod).getValue();
        switch (kind) {
        case AsyncOp::TC5MMA:
        case AsyncOp::TMALoad:
        case AsyncOp::NONE:
          pendingCount += 1;
          break;
        default:
          llvm_unreachable("unsupported producer kind");
        }
      }

      if (consumerPendingCount) {
        assert(*consumerPendingCount == pendingCount &&
               "inconsistent consumer pending count");
      } else {
        consumerPendingCount = pendingCount;
      }
    } else if (auto getExitOp = dyn_cast<ArefGetExitOp>(user)) {
      int pendingCount = 0;
      for (auto consumer : getExitOp.getAsyncOps()) {
        auto kind = dyn_cast<AsyncOpAttr>(consumer).getValue();
        switch (kind) {
        case AsyncOp::TC5MMA:
        case AsyncOp::WGMMA:
        case AsyncOp::NONE:
          pendingCount += 1;
          break;
        default:
          llvm_unreachable("unsupported consumer kind");
        }
      }

      if (producerPendingCount) {
        assert(*producerPendingCount == pendingCount &&
               "inconsistent producer pending count");
      }
      producerPendingCount = pendingCount;
      consumerGroups.insert(idx);
    }
  }

  assert(producerPendingCount);
  assert(consumerPendingCount);
  int numGroupConsumers = consumerGroups.size();
  *producerPendingCount *= numGroupConsumers;

  return {*producerPendingCount, *consumerPendingCount};
}

ArefValue createAndInitMbar(ArefCreateOp op, PatternRewriter &rewriter) {
  BarrierCount count = getArrivalCount(op);

  MLIRContext *ctx = op.getContext();
  auto loc = op.getLoc();
  auto arefTy = op.getType();
  auto baseType = arefTy.getBaseType();
  auto arefBufTypes = llvm::to_vector(llvm::map_range(
      arefTy.getBaseType(), [](Type type) { return cast<MemDescType>(type); }));
  auto shape = arefBufTypes[0].getShape();
  auto depth = shape[0];

  ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
  auto emptyMbars = createScalarAlloc(builder, rewriter.getI64Type(), depth);
  auto fullMbars = createScalarAlloc(builder, rewriter.getI64Type(), depth);
  auto lb = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
  auto ub = rewriter.create<arith::ConstantIntOp>(loc, depth, 32);
  auto step = rewriter.create<arith::ConstantIntOp>(loc, 1, 32);
  auto dLoop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
  rewriter.setInsertionPointToStart(dLoop.getBody());

  for (int i = 0; i < 2; ++i) {
    bool isProducer = i == 0;
    auto mbars = isProducer ? emptyMbars : fullMbars;
    auto singleBarrier =
        createSingleBufferView(rewriter, mbars, dLoop.getInductionVar());
    int pendingCount =
        isProducer ? count.producerPendingCount : count.consumerPendingCount;
    rewriter.create<InitBarrierOp>(loc, singleBarrier, pendingCount);
  }

  // insert inval_barrier right after WarpGroupOp
  WarpGroupOp wgOp = getWarpGroupIdx(*op->getUsers().begin()).first;
  rewriter.setInsertionPointAfter(wgOp);
  auto invalLoop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
  rewriter.setInsertionPointToStart(invalLoop.getBody());

  for (int i = 0; i < 2; ++i) {
    auto mbar = createSingleBufferView(
        rewriter, i == 0 ? emptyMbars : fullMbars, invalLoop.getInductionVar());
    rewriter.create<InvalBarrierOp>(loc, mbar);
  }

  return ArefValue{emptyMbars, fullMbars, static_cast<int>(depth),
                   op.getOperands()};
}

SmallVector<Value> getSubViews(ArefValue arefVal, Value stage, Location loc,
                               OpBuilder &rewriter) {
  SmallVector<Value> views;
  for (auto buffer : arefVal.buffers) {
    auto memDescType = cast<MemDescType>(buffer.getType());
    auto shape = memDescType.getShape();
    auto rank = shape.size() - 1;

    SmallVector<int64_t> tensorShape(shape.begin() + 1, shape.end());
    auto memDescTypeNew = MemDescType::get(
        tensorShape, memDescType.getElementType(), memDescType.getEncoding(),
        memDescType.getMemorySpace(), true);
    Value singleBuffer =
        rewriter.create<MemDescIndexOp>(loc, memDescTypeNew, buffer, stage);
    views.push_back(singleBuffer);
  }

  return views;
}

void lowerAsyncLoads(ArefPutEnterOp op, PatternRewriter &rewriter,
                     ArefValue arefVal) {
  auto loc = op.getLoc();
  // for now handle TMA loads in PutEnterOp
  SmallVector<Operation *> loadOps;
  for (auto result : op.getResults()) {
    for (auto user : result.getUsers()) {
      // Temporary workaround for lit testing: handle TMA loads here until a
      // dedicated tma_load op is added to the NVWS dialect
      if (user->getName().getStringRef() == "tma_load")
        loadOps.push_back(user);
    }
  }
  assert(loadOps.size() <= op.getResults().size());
  if (loadOps.empty())
    return;

  // Use the token to find the matching enter / exit pair
  //   %bufs:n, %token = aref_put.enter %aref[%enter_idx]
  //   tma_load %bufs[0]
  //   ..
  //   tma_load %bufs[n-1]
  //   aref_put.exit %aref[%exit_idx], %token
  ArefPutExitOp arefPutExitOp;
  for (auto user : op.getToken().getUsers()) {
    if (auto exitOp = dyn_cast<ArefPutExitOp>(user)) {
      arefPutExitOp = exitOp;
      break;
    }
  }
  assert(arefPutExitOp);
  assert(arefPutExitOp.getAref() == op.getAref() &&
         "Expecting matching Aref on the ArefPutExitOp");

  Value fullBarrier =
      getFullBarrier(rewriter, loc, arefVal, arefPutExitOp.getStage());
  Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  rewriter.create<triton::nvidia_gpu::BarrierExpectOp>(loc, fullBarrier, 0,
                                                       pred);
  return;
}

void rewritePutEnterOp(ArefCreateOp arefOp, ArefPutEnterOp op,
                       PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  // get empty barrier at a given stage
  Value emptyBarrier = getEmptyBarrier(rewriter, loc, arefVal, op.getStage());

  auto waitOp =
      rewriter.create<WaitBarrierOp>(loc, emptyBarrier, op.getPhase());
  assignStageCluster(waitOp, getStageCluster(op), rewriter);
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter);
  assert(views.size() == op.getBuffers().size());

  // TMA load need special handling as it requires fullMbarrier that
  // we need to get from matching ArefPutExitOp
  lowerAsyncLoads(op, rewriter, arefVal);

  // replaces uses with views
  for (int i = 0; i < arefVal.buffers.size(); ++i)
    op.getBuffers()[i].replaceAllUsesWith(views[i]);
}

void rewriteGetEnterOp(ArefCreateOp arefOp, ArefGetEnterOp op,
                       PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  Value fullBarrier = getFullBarrier(rewriter, loc, arefVal, op.getStage());
  auto waitOp = rewriter.create<WaitBarrierOp>(loc, fullBarrier, op.getPhase());
  assignStageCluster(waitOp, getStageCluster(op), rewriter);
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter);
  assert(views.size() == op.getBuffers().size());

  for (int i = 0; i < arefVal.buffers.size(); ++i)
    op.getBuffers()[i].replaceAllUsesWith(views[i]);
}

void insertArriveBarrier(Location loc, ArrayAttr asyncOps,
                         PatternRewriter &rewriter, Value mbar,
                         StageCluster stageCluster) {
  for (auto asyncOp : asyncOps) {
    auto asyncOpEnum = cast<AsyncOpAttr>(asyncOp).getValue();
    Operation *arriveOp = {};
    switch (asyncOpEnum) {
    case AsyncOp::NONE:
    case AsyncOp::WGMMA:
      arriveOp = rewriter.create<nvidia_gpu::ArriveBarrierOp>(loc, mbar, 1);
      break;
    case AsyncOp::TC5MMA:
    case AsyncOp::TMEMCopy:
      arriveOp = rewriter.create<nvidia_gpu::TCGen5CommitOp>(loc, mbar);
      break;

    case AsyncOp::TMALoad:
      // nothing to do, TMA load is handled by lowering putEnterOp
      break;
    case AsyncOp::CpAsync:
    default:
      llvm_unreachable("unknown async op");
    }
    if (arriveOp)
      assignStageCluster(arriveOp, stageCluster, rewriter);
  }
}

void rewritePutExitOp(ArefPutExitOp op, PatternRewriter &rewriter,
                      ArefValue arefVal) {
  auto loc = op->getLoc();
  rewriter.setInsertionPointAfter(op);
  Value fullBarrier = getFullBarrier(rewriter, loc, arefVal, op.getStage());
  insertArriveBarrier(loc, op.getAsyncOps(), rewriter, fullBarrier,
                      getStageCluster(op));
}

void rewriteGetExitOp(ArefGetExitOp op, PatternRewriter &rewriter,
                      ArefValue arefVal) {
  auto loc = op->getLoc();
  rewriter.setInsertionPointAfter(op);
  Value emptyBarrier = getEmptyBarrier(rewriter, loc, arefVal, op.getStage());
  insertArriveBarrier(loc, op.getAsyncOps(), rewriter, emptyBarrier,
                      getStageCluster(op));
}

class LowerArefCreate : public OpRewritePattern<ArefCreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ArefCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto aref = createAndInitMbar(op, rewriter);
    SetVector<Operation *> opToDelete;
    opToDelete.insert(op.getOperation());
    for (auto userOp : op->getUsers()) {
      opToDelete.insert(userOp);
      if (auto user = dyn_cast<ArefPutEnterOp>(userOp)) {
        rewritePutEnterOp(op, user, rewriter, aref);
      } else if (auto user = dyn_cast<ArefGetEnterOp>(userOp)) {
        rewriteGetEnterOp(op, user, rewriter, aref);
      } else if (auto user = dyn_cast<ArefPutExitOp>(userOp)) {
        rewritePutExitOp(user, rewriter, aref);
      } else if (auto user = dyn_cast<ArefGetExitOp>(userOp)) {
        rewriteGetExitOp(user, rewriter, aref);
      } else {
        llvm_unreachable("users of aref can only be ArefPut or ArefGet");
      }
    }

    auto sorted = topologicalSort(opToDelete);
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it) {
      rewriter.eraseOp(*it);
    }

    return success();
  }
};

// ----------------------------------------------------------------------------

template <class... Ts> struct ArefIndex;
template <class T> struct ArefIndex<T> {
  struct Index {
    // Having stage and phase as separate values, rather than encoding them
    // into a single index, results in better performance. Same approach is used
    // in CUTLASS and CUTEDSL, and this may allow PTXAS to better optimize code.
    Value stage;
    Value phase;
  };
  using ArefIndexMap = llvm::MapVector<Value /*aref*/, Index>;
  using ArefUseSet = llvm::SetVector<Value /*aref*/>;

  static ArefUseSet analyzeArefUseInBlock(Block *block, ArefUseSet arefUseSet) {
    for (auto &op : *block) {
      if (auto opT = dyn_cast<T>(op)) {
        arefUseSet.insert(opT.getAref());
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        arefUseSet = analyzeArefUseInBlock(forOp.getBody(), arefUseSet);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        arefUseSet = analyzeArefUseInBlock(ifOp.thenBlock(), arefUseSet);
        if (ifOp.elseBlock())
          arefUseSet = analyzeArefUseInBlock(ifOp.elseBlock(), arefUseSet);
      }
    }
    return arefUseSet;
  }

  static void assignArefIndexInForOp(scf::ForOp forOp,
                                     ArefIndexMap &arefIndexMap) {

    // find uses of arefs in forOp body
    auto arefUseInBlock = analyzeArefUseInBlock(forOp.getBody(), {});
    if (arefUseInBlock.empty())
      return;

    // add extra iterArgs to the forOp
    SmallVector<Value> extraIterArgs;
    SmallVector<Value *> arefIndexRefs;
    for (auto aref : arefUseInBlock) {
      auto index = arefIndexMap.lookup(aref);
      extraIterArgs.push_back(index.stage);
      arefIndexRefs.push_back(&arefIndexMap[aref].stage);
      if (index.phase) {
        extraIterArgs.push_back(index.phase);
        arefIndexRefs.push_back(&arefIndexMap[aref].phase);
      }
    }

    // create new forOp with extra iterArgs
    OpBuilder builder(forOp);
    size_t nArgs = forOp.getRegionIterArgs().size();
    forOp = addIterArgsToLoop(builder, forOp, extraIterArgs);

    // update arefIndex with iterArgs in the forOp body
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *arefIndexRefs[idx - nArgs] = forOp.getRegionIterArgs()[idx];

    // assign arefIndex in the forOp body
    auto arefIndexMapInBlock =
        assignArefIndexInBlock(forOp.getBody(), arefIndexMap);

    // update yieldOp to return new indexes
    SmallVector<Value> extraYieldArgs;
    for (auto aref : arefUseInBlock) {
      auto &index = arefIndexMapInBlock[aref];
      extraYieldArgs.push_back(index.stage);
      if (index.phase)
        extraYieldArgs.push_back(index.phase);
    }
    appendToForOpYield(forOp, extraYieldArgs);

    // update arefIndex with results from newForOp
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *arefIndexRefs[idx - nArgs] = forOp.getResult(idx);
  }

  static void assignArefIndexInIfOp(scf::IfOp ifOp,
                                    ArefIndexMap &arefIndexMap) {

    // find uses of aref in then-block
    auto arefUseInIfOp = analyzeArefUseInBlock(ifOp.thenBlock(), {});
    if (arefUseInIfOp.empty())
      return;

    // find uses of aref in else-block
    arefUseInIfOp = ifOp.elseBlock()
                        ? analyzeArefUseInBlock(ifOp.elseBlock(), arefUseInIfOp)
                        : arefUseInIfOp;

    // add extra results to the ifOp
    SmallVector<Type> extraIfResults;
    SmallVector<Value *> arefIndexRefs;
    for (auto aref : arefUseInIfOp) {
      auto index = arefIndexMap.lookup(aref);
      extraIfResults.push_back(index.stage.getType());
      arefIndexRefs.push_back(&arefIndexMap[aref].stage);
      if (index.phase) {
        extraIfResults.push_back(index.phase.getType());
        arefIndexRefs.push_back(&arefIndexMap[aref].phase);
      }
    }

    // create new ifOp with extra results
    OpBuilder builder(ifOp);
    size_t nArgs = ifOp.getResults().size();
    auto newIfOp = replaceIfOpWithNewSignature(builder, ifOp, extraIfResults);

    // assign arefIndex in then-body
    auto arefIndexInThenBlock =
        assignArefIndexInBlock(newIfOp.thenBlock(), arefIndexMap);

    // assign arefIndex in else-body
    auto arefIndexInElseBlock =
        ifOp.elseBlock()
            ? assignArefIndexInBlock(newIfOp.elseBlock(), arefIndexMap)
            : arefIndexMap;

    // update yieldOp to return new indexes
    auto thenYieldOp = newIfOp.thenYield();
    auto elseYieldOp = newIfOp.elseYield();
    // insert new indexes to the yieldOp
    for (auto aref : arefUseInIfOp) {
      auto &thenIndex = arefIndexInThenBlock[aref];
      auto &elseIndex = arefIndexInElseBlock[aref];
      thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                  thenIndex.stage);
      elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                  elseIndex.stage);
      if (thenIndex.phase) {
        thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                    thenIndex.phase);
        elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                    elseIndex.phase);
      }
    }
    ifOp.erase();

    // update arefIndex with results from newIfOp
    for (size_t idx = nArgs; idx < newIfOp.getResults().size(); ++idx)
      *arefIndexRefs[idx - nArgs] = newIfOp.getResult(idx);
  }

  static ArefIndexMap assignArefIndexInBlock(Block *block,
                                             ArefIndexMap arefIndexMap) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto opT = dyn_cast<T>(op)) {
        auto index = arefIndexMap.lookup(opT.getAref());

        OpBuilder builder(opT);
        builder.setInsertionPointAfter(opT);

        // compute next stage
        opT.getStageMutable().assign(index.stage);
        auto nextStage = builder.create<arith::AddIOp>(
            opT.getLoc(), index.stage,
            builder.create<arith::ConstantIntOp>(opT.getLoc(), 1, 32));
        auto arefBuf = opT.getAref()
                           .template getDefiningOp<nvws::ArefCreateOp>()
                           .getOperand(0);
        auto depth = cast<MemDescType>(arefBuf.getType()).getShape().front();

        auto cnd = builder.create<arith::CmpIOp>(
            opT.getLoc(), arith::CmpIPredicate::eq, nextStage,
            builder.create<arith::ConstantIntOp>(opT.getLoc(), depth, 32));
        auto zero = builder.create<arith::ConstantIntOp>(opT.getLoc(), 0, 32);
        arefIndexMap[opT.getAref()].stage =
            builder.create<arith::SelectOp>(opT.getLoc(), cnd, zero, nextStage);

        if (index.phase) {
          // if this is an enterOp, compute next phase
          opT->setOperand(2, index.phase);
          auto nextPhase = builder.create<arith::XOrIOp>(
              opT.getLoc(), index.phase,
              builder.create<arith::ConstantIntOp>(opT.getLoc(), 1, 32));
          arefIndexMap[opT.getAref()].phase = builder.create<arith::SelectOp>(
              opT.getLoc(), cnd, nextPhase, index.phase);
        }

      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assignArefIndexInForOp(forOp, arefIndexMap);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        assignArefIndexInIfOp(ifOp, arefIndexMap);
      }
    }

    return arefIndexMap;
  }

  static LogicalResult run(WarpGroupOp wgOp, std::string opName) {
    ArefUseSet arefUse;
    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      arefUse = analyzeArefUseInBlock(block, arefUse);
    }

    // initialize indexes
    ArefIndexMap arefIndexMap;
    for (auto aref : arefUse) {
      OpBuilder builder(aref.getDefiningOp());
      builder.setInsertionPointAfter(aref.getDefiningOp());
      arefIndexMap[aref].stage =
          builder.create<arith::ConstantIntOp>(aref.getLoc(), 0, 32);
      if (std::is_same_v<T, ArefPutEnterOp>) {
        arefIndexMap[aref].phase =
            builder.create<arith::ConstantIntOp>(aref.getLoc(), 1, 32);
      } else if (std::is_same_v<T, ArefGetEnterOp>) {
        arefIndexMap[aref].phase =
            builder.create<arith::ConstantIntOp>(aref.getLoc(), 0, 32);
      } else {
        arefIndexMap[aref].phase = {};
      }
    }

    for (auto region : wgOp.getRegions()) {
      auto block = &region->getBlocks().front();
      assignArefIndexInBlock(block, arefIndexMap);
    }
    return success();
  }
};

template <> struct ArefIndex<> {
  static LogicalResult run(WarpGroupOp wgOp) {
    if (failed(ArefIndex<ArefPutEnterOp>::run(wgOp, "ArefPutEnterOp")))
      return failure();
    if (failed(ArefIndex<ArefPutExitOp>::run(wgOp, "ArefPutExitOp")))
      return failure();
    if (failed(ArefIndex<ArefGetEnterOp>::run(wgOp, "ArefGetEnterOp")))
      return failure();
    if (failed(ArefIndex<ArefGetExitOp>::run(wgOp, "ArefGetExitOp")))
      return failure();
    return success();
  }
};

// ----------------------------------------------------------------------------

} // anonymous namespace

class NVWSLowerAref : public impl::NVWSLowerArefBase<NVWSLowerAref> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    SmallVector<WarpGroupOp> wgOps;
    m.walk([&](WarpGroupOp wgOp) { wgOps.push_back(wgOp); });
    for (auto wgOp : wgOps) {
      if (failed(ArefIndex<>::run(wgOp)))
        signalPassFailure();
    }
    LLVM_DEBUG(llvm::dbgs() << "After arefIndexAssignment\n" << m << "\n");

    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerArefCreate>(context);
    GreedyRewriteConfig config;
    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();
  }
}; // namespace triton

} // namespace triton
} // namespace mlir
