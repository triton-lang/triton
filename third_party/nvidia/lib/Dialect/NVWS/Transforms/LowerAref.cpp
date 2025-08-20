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
#include "mlir/IR/Dominance.h"
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
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
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
  int producerPendingCount{0};
  int consumerPendingCount{0};
};

SmallVector<AsyncOp> castAsyncOpAttrs(ArrayAttr opAttrs) {
  SmallVector<AsyncOp> kinds;
  for (auto asyncKind : opAttrs) {
    kinds.push_back(cast<AsyncOpAttr>(asyncKind).getValue());
  }
  return kinds;
}

BarrierCount getArrivalCount(ArefCreateOp op) {
  SetVector<int> producerGroups, consumerGroups;
  BarrierCount count;

  for (auto user : op->getUsers()) {
    auto idx = getWarpGroupIdx(user).second;
    if (auto putExitOp = dyn_cast<ArefPutExitOp>(user)) {
      if (!producerGroups.insert(idx)) {
        continue;
      }
      for (auto kind : castAsyncOpAttrs(putExitOp.getAsyncOps())) {
        switch (kind) {
        case AsyncOp::TC5MMA:
        case AsyncOp::TMALoad:
        case AsyncOp::NONE:
          count.consumerPendingCount += 1;
          break;
        default:
          llvm_unreachable("unsupported producer kind");
        }
      }
    } else if (auto getExitOp = dyn_cast<ArefGetExitOp>(user)) {
      if (!consumerGroups.insert(idx)) {
        continue;
      }
      for (auto kind : castAsyncOpAttrs(getExitOp.getAsyncOps())) {
        switch (kind) {
        case AsyncOp::TC5MMA:
        case AsyncOp::WGMMA:
        case AsyncOp::NONE:
          count.producerPendingCount += 1;
          break;
        default:
          llvm_unreachable("unsupported consumer kind");
        }
      }
    }
  }

  return count;
}

ArefValue createAndInitMbar(ArefCreateOp op, PatternRewriter &rewriter) {
  BarrierCount count = getArrivalCount(op);

  auto arefTy = op.getType();
  auto arefBufTypes = llvm::to_vector(llvm::map_range(
      arefTy.getBaseType(), [](Type type) { return cast<MemDescType>(type); }));
  auto shape = arefBufTypes[0].getShape();
  auto depth = shape[0];

  auto wgOp = getWarpGroupIdx(*op->getUsers().begin()).first;
  auto emptyMbars =
      triton::createBarrierAlloc(wgOp, depth, count.producerPendingCount);
  auto fullMbars =
      triton::createBarrierAlloc(wgOp, depth, count.consumerPendingCount);

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

void createTMALoad(triton::nvws::DescriptorLoadOp op, PatternRewriter &rewriter,
                   Value barrierAlloc, Value pred) {
  auto indices = translateTMAIndices(
      rewriter, op.getLoc(),
      op.getDesc().getType().getBlockType().getEncoding(), op.getIndices());
  auto newLoadOp =
      rewriter.create<triton::nvidia_gpu::AsyncTMACopyGlobalToLocalOp>(
          op.getLoc(), op.getDesc(), indices, barrierAlloc, op.getResult(),
          pred);
  assignStageCluster(newLoadOp, getStageCluster(op), rewriter);
};

void createTMAGather(triton::nvws::DescriptorGatherOp op,
                     PatternRewriter &rewriter, Value barrierAlloc,
                     Value pred) {
  auto newGatherOp = rewriter.create<triton::nvidia_gpu::AsyncTMAGatherOp>(
      op.getLoc(), op.getDesc(), op.getXOffsets(), op.getYOffset(),
      barrierAlloc, op.getResult(), pred);
  assignStageCluster(newGatherOp, getStageCluster(op), rewriter);
}

void lowerTMALoad(ArefPutEnterOp op, Value fullBarrier,
                  PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  int txCount = 0;
  // for now handle TMA loads in PutEnterOp
  SmallVector<Operation *> loadOps;
  for (auto buffer : op.getBuffers()) {
    for (auto user : buffer.getUsers()) {
      if (auto loadOp =
              dyn_cast<triton::nvws::DescriptorLoadOpInterface>(user)) {
        loadOps.push_back(loadOp);
        txCount += loadOp.getTxCount();
      }
    }
  }
  assert(loadOps.size() <= op.getBuffers().size());
  if (loadOps.empty())
    return;

  Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  auto expectOp = rewriter.create<triton::nvidia_gpu::BarrierExpectOp>(
      loc, fullBarrier, txCount, pred);
  assignStageCluster(expectOp, getStageCluster(op), rewriter);

  for (auto loadOp : loadOps) {
    rewriter.setInsertionPoint(loadOp);
    if (auto descLoad = dyn_cast<triton::nvws::DescriptorLoadOp>(loadOp)) {
      createTMALoad(descLoad, rewriter, fullBarrier, pred);
    } else if (auto descGather =
                   dyn_cast<triton::nvws::DescriptorGatherOp>(loadOp)) {
      createTMAGather(descGather, rewriter, fullBarrier, pred);
    } else {
      llvm_unreachable("Unknown load op");
    }
    loadOp->erase();
  }
}

void rewritePutEnterOp(ArefPutEnterOp op, PatternRewriter &rewriter,
                       ArefValue arefVal,
                       const DenseSet<MMAv5OpInterface> &mmav5Ops) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  // get empty barrier at a given stage
  Value emptyBarrier = getEmptyBarrier(rewriter, loc, arefVal, op.getStage());

  auto waitOp =
      rewriter.create<WaitBarrierOp>(loc, emptyBarrier, op.getPhase());
  assignStageCluster(waitOp, getStageCluster(op), rewriter);
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter);
  assert(views.size() == op.getBuffers().size());

  // Use the token to find the matching enter / exit pair
  //   %bufs:n, %token = aref_put.enter %aref[%enter_idx]
  //   tma_load %bufs[0]
  //   ..
  //   tma_load %bufs[n-1]
  //   aref_put.exit %aref[%exit_idx], %token
  ArefPutExitOp exitOp;
  for (auto user : op.getToken().getUsers()) {
    if (auto op = dyn_cast<ArefPutExitOp>(user)) {
      exitOp = op;
      break;
    }
  }
  assert(exitOp);
  assert(exitOp.getAref() == op.getAref() &&
         "Expecting matching Aref on the ArefPutExitOp");

  auto asyncKinds = castAsyncOpAttrs(exitOp.getAsyncOps());
  auto hasAsyncLoad = [](AsyncOp kind) {
    return kind == AsyncOp::TMALoad || kind == AsyncOp::CpAsync;
  };
  auto hasTMA = [](AsyncOp kind) { return kind == AsyncOp::TMALoad; };

  if (llvm::any_of(asyncKinds, hasTMA)) {
    Value fullBarrier = getFullBarrier(rewriter, loc, arefVal, op.getStage());
    lowerTMALoad(op, fullBarrier, rewriter, arefVal);
  }

  if (llvm::any_of(asyncKinds, hasAsyncLoad)) {
    for (auto mmav5 : mmav5Ops) {
      mmav5.setIsAsync(true);
    }
  }

  for (auto [oldBuffer, view] : llvm::zip(op.getBuffers(), views)) {
    oldBuffer.replaceAllUsesWith(view);
  }
}

static MemDescType getAsMutable(MemDescType type) {
  return MemDescType::get(type.getShape(), type.getElementType(),
                          type.getEncoding(), type.getMemorySpace(),
                          /*mutableMemory=*/true);
}

static void propagateMutability(Value value) {
  for (Operation *user : value.getUsers()) {
    if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      user->getResult(0).setType(
          getAsMutable(cast<MemDescType>(user->getResult(0).getType())));
      propagateMutability(user->getResult(0));
    }
  }
}

void rewriteGetEnterOp(ArefGetEnterOp op, PatternRewriter &rewriter,
                       ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  Value fullBarrier = getFullBarrier(rewriter, loc, arefVal, op.getStage());
  auto waitOp = rewriter.create<WaitBarrierOp>(loc, fullBarrier, op.getPhase());
  assignStageCluster(waitOp, getStageCluster(op), rewriter);
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter);
  assert(views.size() == op.getBuffers().size());

  for (auto [oldBuffer, view] : llvm::zip(op.getBuffers(), views)) {
    oldBuffer.replaceAllUsesWith(view);
    // Before aref lowering, memdesc_trans consumes an immutable buffer from
    // a get enter op. After lowering, all buffers are mutable.
    propagateMutability(view);
  }
}

void insertArriveBarrier(Location loc, ArrayRef<AsyncOp> asyncOps,
                         PatternRewriter &rewriter, Value mbar,
                         StageCluster stageCluster) {
  for (auto asyncOpEnum : asyncOps) {
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
      // nothing to do, the arrive is done by HW
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
  insertArriveBarrier(loc, castAsyncOpAttrs(op.getAsyncOps()), rewriter,
                      fullBarrier, getStageCluster(op));
}

void rewriteGetExitOp(ArefGetExitOp op, PatternRewriter &rewriter,
                      ArefValue arefVal) {
  auto loc = op->getLoc();
  auto stageCluster = getStageCluster(op);
  auto asyncKinds = castAsyncOpAttrs(op.getAsyncOps());
  rewriter.setInsertionPointAfter(op);

  bool needFence = [&]() {
    bool isGenericProxy = llvm::any_of(
        asyncKinds, [](AsyncOp kind) { return kind == AsyncOp::NONE; });
    if (!isGenericProxy) {
      return false;
    }
    for (auto arefUser : op.getAref().getUsers()) {
      if (auto putExit = dyn_cast<ArefPutExitOp>(arefUser)) {
        bool isProducerTMA =
            llvm::any_of(castAsyncOpAttrs(putExit.getAsyncOps()),
                         [](AsyncOp kind) { return kind == AsyncOp::TMALoad; });
        if (isProducerTMA) {
          return true;
        }
      }
    }
    return false;
  }();

  if (needFence) {
    auto fence = rewriter.create<FenceAsyncSharedOp>(loc, /*bCluster=*/false);
    assignStageCluster(fence, stageCluster, rewriter);
  }

  Value emptyBarrier = getEmptyBarrier(rewriter, loc, arefVal, op.getStage());
  return insertArriveBarrier(loc, asyncKinds, rewriter, emptyBarrier,
                             stageCluster);
}

DenseSet<MMAv5OpInterface> getAsyncMMAv5Consumers(Value aref) {
  DenseSet<MMAv5OpInterface> mmav5Ops;
  for (auto arefUser : aref.getUsers()) {
    if (auto getEnter = dyn_cast<ArefGetEnterOp>(arefUser)) {
      if (getWarpGroupIdx(getEnter).second == 0) {
        // Ignore mmav5 ops in the default partition. They are not warp
        // specialized.
        continue;
      }

      for (auto buffer : getEnter.getBuffers()) {
        for (auto consumer : buffer.getUsers()) {
          if (auto mmav5 = dyn_cast<MMAv5OpInterface>(consumer)) {
            mmav5Ops.insert(mmav5);
          } else if (auto forOp = consumer->getParentOfType<scf::ForOp>()) {
            auto users =
                getTopLevelUsersInLoop(consumer, forOp, [](Operation *user) {
                  return isa<MMAv5OpInterface>(user);
                });
            for (auto user : users) {
              mmav5Ops.insert(cast<MMAv5OpInterface>(user));
            }
          }
        }
      }
    }
  }
  return mmav5Ops;
}

class LowerArefCreate : public OpRewritePattern<ArefCreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ArefCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto aref = createAndInitMbar(op, rewriter);
    SetVector<Operation *> opToDelete;
    opToDelete.insert(op.getOperation());

    // setIsAsync(true) will be invoked on these mmav5 ops during
    // rewritePutEnterOp when the producer is async loads. Since collecting
    // consumer mmav5 ops requires the corresponding get enter op to be still
    // used in the IR, collect them here.
    auto mmav5Ops = getAsyncMMAv5Consumers(op.getResult());

    for (auto userOp : op->getUsers()) {
      opToDelete.insert(userOp);
      if (auto user = dyn_cast<ArefPutEnterOp>(userOp)) {
        rewritePutEnterOp(user, rewriter, aref, mmav5Ops);
      } else if (auto user = dyn_cast<ArefGetEnterOp>(userOp)) {
        rewriteGetEnterOp(user, rewriter, aref);
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

bool isProducerLoad(ArefCreateOp arefOp) {
  for (auto user : arefOp.getResult().getUsers()) {
    if (auto putOp = dyn_cast<ArefPutEnterOp>(user)) {
      for (auto buffer : putOp.getBuffers()) {
        for (auto user : buffer.getUsers()) {
          if (isa<triton::nvws::DescriptorLoadOpInterface>(user)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void multiBufferAref(const SmallVector<ArefCreateOp> &arefOps, int numStages) {
  SmallVector<Operation *> allocsToErase;
  for (auto arefOp : arefOps) {
    SmallVector<Value> allocOps;
    SmallVector<Type> arefTypes;

    bool eligible = true;
    for (auto opnd : arefOp.getOperands()) {
      if (!opnd.getDefiningOp()) {
        eligible = false;
      }
    }

    if (!eligible) {
      continue;
    }

    OpBuilder builder(arefOp);
    for (auto opnd : arefOp.getOperands()) {
      auto oldAlloc = opnd.getDefiningOp();
      auto arefBufType = cast<MemDescType>(opnd.getType());
      arefBufType =
          getMultiBufferedType(getBufferViewType(arefBufType, true), numStages);
      Operation *newAlloc = triton::nvws::createAlloc(
          builder, oldAlloc->getLoc(), arefBufType, Value());
      allocOps.push_back(newAlloc->getResult(0));
      arefTypes.push_back(arefBufType);
      oldAlloc->replaceAllUsesWith(newAlloc);
      allocsToErase.push_back(oldAlloc);
    }

    auto newAref =
        createArefCreateOp(builder, arefTypes, allocOps, arefOp.getLoc());

    arefOp.getResult().replaceAllUsesWith(newAref.getResult());
    arefOp.erase();
  }

  for (auto alloc : allocsToErase) {
    alloc->erase();
  }
}

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

template <typename EnterOp, typename ExitOp>
void createCombinedArefOps(SmallVector<EnterOp> &enterOps,
                           SmallVector<ExitOp> &exitOps, ArefCreateOp aref,
                           OpBuilder &builder) {
  auto firstEnter = *llvm::min_element(enterOps, [](EnterOp a, EnterOp b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  auto lastExit = *llvm::max_element(exitOps, [](ExitOp a, ExitOp b) {
    assert(a->getBlock() == b->getBlock());
    return a->isBeforeInBlock(b);
  });

  SmallVector<Type> arefEnterBuffers;
  for (auto enterOp : enterOps) {
    arefEnterBuffers.push_back(enterOp.getResult(0).getType());
  }

  llvm::SmallSetVector<Attribute, 5> opAttrsSet;
  for (ExitOp exitOp : exitOps) {
    opAttrsSet.insert(exitOp.getAsyncOps().begin(), exitOp.getAsyncOps().end());
  }

  builder.setInsertionPointAfter(aref);
  auto zero = builder.create<arith::ConstantIntOp>(aref.getLoc(), 0, 32);

  builder.setInsertionPoint(firstEnter);
  auto enter = builder.create<EnterOp>(firstEnter.getLoc(), arefEnterBuffers,
                                       builder.getType<AsyncTokenType>(), aref,
                                       zero, zero);
  assignStageCluster(enter, getStageCluster(firstEnter), builder);

  builder.setInsertionPoint(lastExit);
  llvm::SmallVector<Attribute> AsyncOpAttrs(opAttrsSet.begin(),
                                            opAttrsSet.end());
  auto exit =
      builder.create<ExitOp>(firstEnter.getLoc(), aref, enter.getToken(), zero,
                             builder.getArrayAttr(AsyncOpAttrs));
  assignStageCluster(exit, getStageCluster(lastExit), builder);

  for (auto [idx, enterOp] : llvm::enumerate(enterOps)) {
    enterOp.getBuffers()[0].replaceAllUsesWith(enter.getBuffers()[idx]);
  }
}

SmallVector<Operation *> findSharedMemorySinkOps(Value value) {
  SmallVector<Operation *> sinkOps;
  for (Operation *user : value.getUsers()) {
    if (isa<MMAv5OpInterface, LocalLoadOp>(user)) {
      sinkOps.push_back(user);
    } else if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
      auto rec = findSharedMemorySinkOps(user->getResult(0));
      sinkOps.insert(sinkOps.end(), rec.begin(), rec.end());
    }
  }
  return sinkOps;
}

Operation *getDominantConsumer(ArefGetEnterOp getEnterOp, Block &container,
                               DominanceInfo &domInfo) {
  assert(getEnterOp->getNumResults() && "Expect a single-result ArefGenterOp");
  auto buf = getEnterOp->getResult(0);
  SmallVector<Operation *> sinkOps = findSharedMemorySinkOps(buf);
  if (sinkOps.empty()) {
    return nullptr;
  }
  Operation *liveBeforeOp = findNearestCommonDominator(sinkOps, domInfo);
  return container.findAncestorOpInBlock(*liveBeforeOp);
}

// This is an optimization to combine arefs for TMA load into one, so that
// barrier arrive and wait are coalesced.
void combineArefs(scf::ForOp loop) {
  SmallVector<ArefGetEnterOp> getEnterOps;
  loop.walk([&](ArefGetEnterOp op) { getEnterOps.push_back(op); });

  // Arefs whose get-enter ops share the same dominant consumer can be combined
  DominanceInfo domInfo(loop);
  llvm::DenseMap<Operation *, SmallVector<ArefGetEnterOp>> liveBeforeGroups;
  for (auto getEnterOp : getEnterOps) {
    if (auto liveBeforeOp =
            getDominantConsumer(getEnterOp, *loop.getBody(), domInfo)) {
      liveBeforeGroups[liveBeforeOp].push_back(getEnterOp);
    }
  }

  for (auto getEnterOps : llvm::make_second_range(liveBeforeGroups)) {
    if (getEnterOps.size() == 1) {
      continue;
    }

    SmallVector<ArefCreateOp> arefs;
    for (auto getEnterOp : getEnterOps) {
      arefs.push_back(cast<ArefCreateOp>(getEnterOp.getAref().getDefiningOp()));
    }

    SmallVector<ArefPutEnterOp> putEnterOps;
    SmallVector<ArefPutExitOp> putExitOps;
    SmallVector<ArefGetExitOp> getExitOps;
    SmallVector<int> producerGroupIds;
    for (auto aref : arefs) {
      for (auto user : aref->getUsers()) {
        if (auto putEnterOp = dyn_cast<ArefPutEnterOp>(user)) {
          putEnterOps.push_back(putEnterOp);
          producerGroupIds.push_back(getWarpGroupIdx(putEnterOp).second);
        } else if (auto putExitOp = dyn_cast<ArefPutExitOp>(user)) {
          putExitOps.push_back(putExitOp);
        } else if (auto getExitOp = dyn_cast<ArefGetExitOp>(user)) {
          getExitOps.push_back(getExitOp);
        }
      }
    }

    // Producer arefs must be in the same partition.
    if (llvm::any_of(producerGroupIds,
                     [&](int id) { return id != producerGroupIds[0]; })) {
      continue;
    }

    SmallVector<Type> arefBufTypes;
    SmallVector<Value> arefBufs;
    for (auto aref : arefs) {
      arefBufTypes.push_back(aref.getOperands()[0].getType());
      arefBufs.push_back(aref.getOperands()[0]);
    }

    // set insertion point at the last aref_create
    auto lastAref = *llvm::max_element(arefs, [](auto a, auto b) {
      assert(a->getBlock() == b->getBlock());
      return a->isBeforeInBlock(b);
    });

    OpBuilder builder(lastAref);
    auto aref =
        createArefCreateOp(builder, arefBufTypes, arefBufs, lastAref->getLoc());

    createCombinedArefOps(putEnterOps, putExitOps, aref, builder);
    createCombinedArefOps(getEnterOps, getExitOps, aref, builder);

    for (auto putExitOp : putExitOps)
      putExitOp->erase();
    for (auto putEnterOp : putEnterOps)
      putEnterOp->erase();
    for (auto getExitOp : getExitOps)
      getExitOp->erase();
    for (auto getEnterOp : getEnterOps)
      getEnterOp->erase();
    for (auto aref : arefs)
      aref->erase();
  }
}

} // anonymous namespace

class NVWSLowerAref : public impl::NVWSLowerArefBase<NVWSLowerAref> {
  using impl::NVWSLowerArefBase<NVWSLowerAref>::NVWSLowerArefBase;

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    SmallVector<scf::ForOp> loops;
    m.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(triton::kWarpSpecializeAttrName))
        loops.push_back(loop);
    });
    for (scf::ForOp loop : loops) {
      combineArefs(loop);
    }

    SmallVector<ArefCreateOp> arefOps;
    m.walk([&](ArefCreateOp arefOp) {
      // Only handles arefs whose producer (a partition with PutEnter / Exit)
      // does load from global to shared memory.
      if (isProducerLoad(arefOp)) {
        arefOps.push_back(arefOp);
      }
    });
    multiBufferAref(arefOps, numStages);

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
