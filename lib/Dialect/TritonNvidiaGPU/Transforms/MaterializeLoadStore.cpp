/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <numeric>

//===----------------------------------------------------------------------===//
//
// This pass works after pipeline pass, converts the remaining tt.LoadOp taking
// ptr<tensor> as input into ttg.InsertSliceAsyncOp and emit proper barriers
//
//===----------------------------------------------------------------------===//

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::getCTALayout;
using ::mlir::triton::gpu::NvidiaMmaEncodingAttr;
using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

struct MaterializeLoadStorePass
    : public MaterializeLoadStoreBase<MaterializeLoadStorePass> {

public:
  MaterializeLoadStorePass() = default;
  MaterializeLoadStorePass(int numWarps, int computeCapability) {
    this->numWarps = numWarps;
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    Block *entryBlock;
    getOperation()->walk([&](mlir::triton::FuncOp funcOp) {
      entryBlock = &funcOp.getBody().front();
    });

    SmallVector<mlir::triton::LoadOp> loadOpWorkList;
    getOperation()->walk([&](mlir::triton::LoadOp load) {
      if (isLoadFromTensorPtr(load)) {
        loadOpWorkList.push_back(load);
      }
    });

    immutableMbarrierAlloc(&entryBlock->front(), loadOpWorkList.size());

    unsigned count = 0;
    for (auto load : loadOpWorkList) {
      materializeLoadTilePtr(load, barrierLists[count++]);
    }

    SmallVector<mlir::triton::StoreOp> storeOpWorkList;
    getOperation()->walk([&](mlir::triton::StoreOp store) {
      if (isStoreToTensorPtr(store)) {
        storeOpWorkList.push_back(store);
      }
    });
    for (auto store : storeOpWorkList) {
      materializeStoreTilePtr(store);
    }
  }

private:
  SmallVector<Value> barrierLists;
  void materializeLoadTilePtr(mlir::triton::LoadOp load, Value mBarrier);
  void materializeStoreTilePtr(mlir::triton::StoreOp store);
  void immutableMbarrierAlloc(Operation *entryOp, unsigned loadNum);
  Value getPhase(scf::ForOp forOp, bool initCall);
};

void MaterializeLoadStorePass::materializeLoadTilePtr(mlir::triton::LoadOp load,
                                                      Value mBarrier) {
  if (computeCapability < 90)
    return;
  if (!::triton::tools::getBoolEnv("ENABLE_TMA"))
    return;
  auto loc = load.getLoc();
  OpBuilder b(load);
  auto loadTy = load.getType().dyn_cast<RankedTensorType>();
  auto loadShape = loadTy.getShape();
  auto CTASplitNum = ttg::getCTASplitNum(loadTy.getEncoding());
  auto shapePerSlice = ttg::getShapePerCTA(CTASplitNum, loadShape);
  auto elemTy = loadTy.getElementType();
  assert(loadTy);
  SmallVector<int64_t> bufferShape(loadShape.begin(), loadShape.end());
  bufferShape.insert(bufferShape.begin(), 1);

  auto sharedEncoding = getSharedEncoding(loadTy);
  auto bufferTy = RankedTensorType::get(bufferShape, elemTy, sharedEncoding);
  Value buffer = b.create<ttg::AllocTensorOp>(loc, bufferTy);
  unsigned elems = std::accumulate(shapePerSlice.begin(), shapePerSlice.end(),
                                   1, std::multiplies{});
  elems *= (elemTy.getIntOrFloatBitWidth() / 8);
  Value _0 = b.create<arith::ConstantIntOp>(loc, 0, 32);
  Value threadId = b.create<ttng::GetThreadIdOp>(loc);
  Value pred =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, threadId, _0);
  b.create<ttng::MBarrierArriveOp>(loc, mBarrier, pred, /*remoteCtaId*/ nullptr,
                                   /*trackAsyncOp*/ false, elems);
  Value inserted = b.create<ttng::InsertSliceTMAOp>(
      loc, bufferTy, load.getPtr(), buffer,
      /*index*/ _0, mBarrier, load.getMask(), load.getOther(), load.getCache(),
      load.getEvict(), load.getIsVolatile(),
      /*axis*/ 0);
  auto extractedTy = RankedTensorType::get(loadShape, elemTy, sharedEncoding);
  Value extracted = b.create<mlir::triton::gpu::ExtractSliceOp>(
      loc, extractedTy, inserted,
      SmallVector<OpFoldResult>{b.getI64IntegerAttr(0), b.getI64IntegerAttr(0),
                                b.getI64IntegerAttr(0)},
      SmallVector<OpFoldResult>{b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(loadShape[0]),
                                b.getI64IntegerAttr(loadShape[1])},
      SmallVector<OpFoldResult>{b.getI64IntegerAttr(1), b.getI64IntegerAttr(1),
                                b.getI64IntegerAttr(1)});
  Value phase;
  if (auto forOp = load->getParentOfType<mlir::scf::ForOp>()) {
    phase = getPhase(forOp, true);
  } else {
    phase = b.create<arith::ConstantIntOp>(loc, 0, 1);
  }
  b.create<ttng::MBarrierWaitOp>(loc, mBarrier, phase);
  Value newValue =
      b.create<ttg::ConvertLayoutOp>(loc, load.getType(), extracted);
  load.getResult().replaceAllUsesWith(newValue);
  load->erase();
}

void MaterializeLoadStorePass::materializeStoreTilePtr(
    mlir::triton::StoreOp store) {
  if (computeCapability < 90 || !::triton::tools::getBoolEnv("ENABLE_TMA"))
    return;
  auto loc = store.getLoc();
  OpBuilder builder(store);
  auto value = store.getValue();
  auto dst = store.getPtr();

  auto cvtOp = llvm::dyn_cast_or_null<mlir::triton::gpu::ConvertLayoutOp>(
      value.getDefiningOp());
  if (cvtOp) {
    auto srcTy = cvtOp.getOperand().getType().cast<RankedTensorType>();
    auto dstTy = cvtOp.getResult().getType().cast<RankedTensorType>();
    auto elemTy = srcTy.getElementType();
    auto srcMmaLayout = srcTy.getEncoding().dyn_cast<NvidiaMmaEncodingAttr>();
    auto dstBlockedLayout = dstTy.getEncoding().dyn_cast<BlockedEncodingAttr>();
    auto truncFOP = llvm::dyn_cast_or_null<arith::TruncFOp>(
        cvtOp.getOperand().getDefiningOp());
    unsigned numElems = ttg::getTotalElemsPerThread(srcTy);
    auto inOrd = ttg::getOrder(srcTy.getEncoding());
    auto outOrd = ttg::getOrder(dstTy.getEncoding());
    if (srcMmaLayout && srcMmaLayout.isHopper() && dstBlockedLayout &&
        truncFOP && elemTy.getIntOrFloatBitWidth() == 16 && numElems >= 16 &&
        inOrd == outOrd) {
      builder.create<ttng::StoreAsyncTMAOp>(loc, dst, cvtOp.getOperand());
      builder.create<ttg::AsyncBulkCommitGroupOp>(loc);
      builder.create<ttg::AsyncBulkWaitOp>(loc, 0);
      store->erase();
      return;
    }
  }

  auto *ctx = store.getContext();
  auto storeTy = value.getType().dyn_cast<RankedTensorType>();
  assert(storeTy);
  auto storeElemTy = storeTy.getElementType();
  auto ctaLayout = getCTALayout(storeTy.getEncoding());
  auto storeShape = storeTy.getShape();
  SmallVector<int64_t> bufferShape(storeShape.begin(), storeShape.end());
  auto rank = storeShape.size();
  // The order of smem should be consistent with gmem.
  auto makeTensorPtrOp = getMakeTensorPtrOp(dst);
  SmallVector<unsigned> sharedOrder;
  for (auto o : makeTensorPtrOp.getOrder()) {
    sharedOrder.emplace_back(o);
  }
  auto sharedEncoding = SharedEncodingAttr::get(ctx, storeShape, sharedOrder,
                                                ctaLayout, storeElemTy);
  auto bufferTy =
      RankedTensorType::get(bufferShape, storeElemTy, sharedEncoding);
  Value cvt = builder.create<ttg::ConvertLayoutOp>(loc, bufferTy, value);
  builder.create<ttng::StoreAsyncTMAOp>(loc, dst, cvt);
  builder.create<mlir::triton::gpu::AsyncBulkCommitGroupOp>(loc);
  builder.create<mlir::triton::gpu::AsyncBulkWaitOp>(loc, 0);
  store->erase();
}

void MaterializeLoadStorePass::immutableMbarrierAlloc(Operation *entryOp,
                                                      unsigned loadNum) {
  if (computeCapability < 90 || !::triton::tools::getBoolEnv("ENABLE_TMA"))
    return;
  OpBuilder builder(entryOp);
  Location loc = entryOp->getLoc();
  for (unsigned i = 0; i < loadNum; ++i) {
    auto mBarrierTy =
        mlir::triton::PointerType::get(builder.getIntegerType(64), 3);
    Value mBarrier = builder.create<ttng::AllocMBarrierOp>(loc, mBarrierTy, 1);
    barrierLists.push_back(mBarrier);
  }
}

/// Func Brief
///   getPhase function inserts phase to every nested forOp recursively and
///   returns the phase loop-carried by current forOp.
/// Design Specification
///   Each loadOp phase need to be loop-carried by all external forOps.
///   Just like this structure:
///   scf.for ... iter_args(%arg2, %arg1)
///     loadOp
///     mbarrier_wait(%arg1)
///     scf.for ... iter_args(%arg2)
///       loadOp
///       mbarrier_wait(%arg2)
///       %arg3 = %arg2 ^ 1
///       yield(%arg3)
///     %arg4 = %arg1 ^ 1
///     yiled(%arg3, %arg4)
///   In this structure, internal forOp operand depends on external forOp
///   blockArgument, so the forOp recreate order is from external to internal.
///   BTW, the yieldOp of external forOp depends on the result of internal
///   forOp, so the yielOp insertOperand will delay one iteration.
Value MaterializeLoadStorePass::getPhase(scf::ForOp forOp, bool initCall) {
  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();

  // Step 1: get initVal
  Value initVal;
  if (auto parentForOp = forOp->getParentOfType<scf::ForOp>())
    initVal = getPhase(parentForOp, false);
  else
    initVal = builder.create<arith::ConstantIntOp>(loc, 0, 1);

  // Step 2: create newForOp
  llvm::SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                       forOp.getInitArgs().end());
  newInitArgs.push_back(initVal);
  builder.setInsertionPoint(forOp);
  auto newForOp = builder.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                             forOp.getUpperBound(),
                                             forOp.getStep(), newInitArgs);
  newForOp.getRegion().takeBody(forOp.getRegion());
  for (auto i : llvm::seq<int>(0, forOp.getNumResults()))
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  forOp.erase();

  // Step 3: add block argument
  Block *block = newForOp.getBody();
  Value curPhase = block->addArgument(builder.getI1Type(), loc);

  // Step 4: flip phase and insert current yieldOp operand
  if (initCall) {
    auto yieldOp = llvm::cast<scf::YieldOp>(block->getTerminator());
    builder.setInsertionPoint(yieldOp);
    Value one = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    Value yieldPhase = builder.create<arith::XOrIOp>(loc, curPhase, one);
    yieldOp->insertOperands(yieldOp->getNumOperands(), yieldPhase);
  }

  // Step 5: insert parent yieldOp operand
  if (auto parentForOp = newForOp->getParentOfType<scf::ForOp>()) {
    Block *parentBlock = parentForOp.getBody();
    auto parentYieldOp = llvm::cast<scf::YieldOp>(parentBlock->getTerminator());
    auto curResult = newForOp->getResults().back();
    parentYieldOp->insertOperands(parentYieldOp.getNumOperands(), curResult);
  }

  return curPhase;
}

} // anonymous namespace

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUMaterializeLoadStorePass(int numWarps,
                                                    int computeCapability) {
  return std::make_unique<MaterializeLoadStorePass>(numWarps,
                                                    computeCapability);
}
