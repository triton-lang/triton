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

#include "Utilities.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
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
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
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

void assignStageCluster(Operation *op, std::optional<PartitionId> partitionId,
                        StageCluster stageCluster, OpBuilder &builder) {
  if (partitionId) {
    op->setAttr(kPartitionAttrName,
                builder.getI32IntegerAttr(partitionId->index()));
    if (stageCluster) {
      op->setAttr(triton::kLoopStageAttrName,
                  builder.getI32IntegerAttr(stageCluster->first));
      op->setAttr(triton::kLoopClusterAttrName,
                  builder.getI32IntegerAttr(stageCluster->second));
    }
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

struct BarrierCount {
  int producerPendingCount;
  int consumerPendingCount;
};

BarrierCount getArrivalCount(ArefCreateOp op) {
  std::optional<int> producerPendingCount, consumerPendingCount;
  std::set<PartitionId> consumerGroups;

  for (auto user : op->getUsers()) {
    auto partitionId = getPartitionId(user);
    if (!partitionId)
      continue;

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
      consumerGroups.insert(*partitionId);
    }
  }

  assert(producerPendingCount);
  assert(consumerPendingCount);
  int numGroupConsumers = consumerGroups.size();
  *producerPendingCount *= numGroupConsumers;

  return {*producerPendingCount, *consumerPendingCount};
}

Value createBarriers(ImplicitLocOpBuilder &b1, ImplicitLocOpBuilder &b2,
                     int numBarriers, int arrivalCount) {
  Value barrierAlloc = createScalarAlloc(b1, b1.getI64Type(), numBarriers);
  for (unsigned i = 0; i < numBarriers; i++) {
    Value barrierView = createSingleBufferView(b1, barrierAlloc, i);
    b1.create<InitBarrierOp>(barrierView, arrivalCount);
  }
  // Invalidate and deallocate the barriers.
  for (unsigned i = 0; i < numBarriers; i++) {
    Value barrierView = createSingleBufferView(b2, barrierAlloc, i);
    b2.create<InvalBarrierOp>(barrierView);
  }
  b2.create<LocalDeallocOp>(barrierAlloc);
  return barrierAlloc;
}

ArefValue createAndInitMbar(ArefCreateOp op, PatternRewriter &rewriter) {
  BarrierCount count = getArrivalCount(op);

  auto arefTy = op.getType();
  auto arefBufTypes = llvm::to_vector(llvm::map_range(
      arefTy.getBaseType(), [](Type type) { return cast<MemDescType>(type); }));
  auto shape = arefBufTypes[0].getShape();
  auto depth = shape[0];

  SetVector<Operation *> arefUsers;
  for (auto user : op->getUsers())
    arefUsers.insert(user);
  auto sorted = topologicalSort(arefUsers);

  ImplicitLocOpBuilder b1(op->getLoc(), op), b2(op->getLoc(), op);
  auto op1 = op->getBlock()->findAncestorOpInBlock(*sorted.back());
  b2.setInsertionPointAfter(op1);

  auto emptyMbars = createBarriers(b1, b2, depth, count.producerPendingCount);
  auto fullMbars = createBarriers(b1, b2, depth, count.consumerPendingCount);

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

void insertWaitOp(PatternRewriter &rewriter, Operation *op, Value barrier,
                  Value phase, Value stage) {
  auto waitOp = rewriter.create<WaitBarrierOp>(op->getLoc(), barrier, phase);
  assignStageCluster(waitOp, getPartitionId(op), getStageCluster(op), rewriter);
}

void rewritePutEnterOp(ArefCreateOp arefOp, ArefPutEnterOp op,
                       PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  // get empty barrier at a given stage
  Value emptyBarrier = getEmptyBarrier(rewriter, loc, arefVal, op.getStage());

  insertWaitOp(rewriter, op, emptyBarrier, op.getPhase(), op.getStage());
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
  insertWaitOp(rewriter, op, fullBarrier, op.getPhase(), op.getStage());
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter);
  assert(views.size() == op.getBuffers().size());

  for (int i = 0; i < arefVal.buffers.size(); ++i)
    op.getBuffers()[i].replaceAllUsesWith(views[i]);
}

void insertArriveBarrier(Location loc, ArrayAttr asyncOps,
                         PatternRewriter &rewriter, Value mbar,
                         std::optional<PartitionId> partitionId,
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
      assignStageCluster(arriveOp, partitionId, stageCluster, rewriter);
  }
}

void rewritePutExitOp(ArefPutExitOp op, PatternRewriter &rewriter,
                      ArefValue arefVal) {
  auto loc = op->getLoc();
  rewriter.setInsertionPointAfter(op);
  Value fullBarrier = getFullBarrier(rewriter, loc, arefVal, op.getStage());
  insertArriveBarrier(loc, op.getAsyncOps(), rewriter, fullBarrier,
                      getPartitionId(op), getStageCluster(op));
}

void rewriteGetExitOp(ArefGetExitOp op, PatternRewriter &rewriter,
                      ArefValue arefVal) {
  auto loc = op->getLoc();
  rewriter.setInsertionPointAfter(op);
  Value emptyBarrier = getEmptyBarrier(rewriter, loc, arefVal, op.getStage());
  insertArriveBarrier(loc, op.getAsyncOps(), rewriter, emptyBarrier,
                      getPartitionId(op), getStageCluster(op));
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
    OpBuilder b(op);
    auto replToken =
        b.create<ub::PoisonOp>(op.getLoc(), b.getType<AsyncTokenType>());
    for (auto op : sorted) {
      if (auto enterOp = dyn_cast<ArefPutEnterOp>(op))
        enterOp.getToken().replaceAllUsesWith(replToken);
      else if (auto enterOp = dyn_cast<ArefGetEnterOp>(op))
        enterOp.getToken().replaceAllUsesWith(replToken);
    }
    for (auto it = sorted.rbegin(); it != sorted.rend(); ++it)
      rewriter.eraseOp(*it);

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

    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerArefCreate>(context);
    GreedyRewriteConfig config;
    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();
  }
}; // namespace triton

} // namespace triton
} // namespace mlir
