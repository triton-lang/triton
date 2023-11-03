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
using ::mlir::triton::gpu::MmaEncodingAttr;
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

    SmallVector<scf::ForOp> forOpLists;
    getOperation()->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
      if (auto forOp = llvm::dyn_cast<scf::ForOp>(subOp)) {
        forOpLists.push_back(forOp);
      }
    });

    for (auto &op : forOpLists) {
      scf::ForOp parentForOp = op->getParentOfType<scf::ForOp>();
      bool hasLoadOp = false;
      for (Operation &subOp : *op.getBody()) {
        if (isa<mlir::triton::LoadOp>(&subOp)) {
          hasLoadOp = true;
          break;
        }
      }
      scf::ForOp childForOp;
      bool hasChildForOp = false;
      for (Operation &subOp : *op.getBody()) {
        if (childForOp = llvm::dyn_cast<scf::ForOp>(&subOp)) {
          hasChildForOp = true;
          break;
        }
      }
      if (hasLoadOp) {
        if (hasChildForOp) {
          createNewParentLoop(op, childForOp);
        } else {
          createNewChildLoop(op, parentForOp);
        }
      }
    }

    DenseMap<mlir::triton::LoadOp, bool> loadOpWorkList;
    getOperation()->walk<WalkOrder::PreOrder>(
        [&](mlir::triton::LoadOp load) -> void {
          if (isLoadFromTensorPtr(load)) {
            if (llvm::dyn_cast<mlir::scf::ForOp>(load->getParentOp())) {
              loadOpWorkList[load] = true;
            } else {
              loadOpWorkList[load] = false;
            }
          }
        });

    SmallVector<Value> barrierLists;
    barrierLists =
        immutableMbarrierAlloc(&entryBlock->front(), loadOpWorkList.size());

    unsigned count = 0;
    for (auto opMap : loadOpWorkList) {
      unsigned idx = count++;
      materializeLoadTilePtr(opMap.first, barrierLists[idx], opMap.second);
    }

    SmallVector<mlir::triton::StoreOp> storeOpWorklists;
    getOperation()->walk([&](mlir::triton::StoreOp store) -> void {
      if (isStoreToTensorPtr(store)) {
        storeOpWorklists.push_back(store);
      }
    });
    for (auto store : storeOpWorklists) {
      materializeStoreTilePtr(store);
    }
  }

private:
  void materializeLoadTilePtr(mlir::triton::LoadOp load, Value mBarrier,
                              bool hasParentForOp);
  void materializeStoreTilePtr(mlir::triton::StoreOp store);
  void createNewChildLoop(scf::ForOp forOp, scf::ForOp parentForOp);
  void createNewParentLoop(scf::ForOp forOp, scf::ForOp childForOp);
  SmallVector<Value> immutableMbarrierAlloc(Operation *entryOp,
                                            unsigned loadNum);
};

void MaterializeLoadStorePass::materializeLoadTilePtr(mlir::triton::LoadOp load,
                                                      Value mBarrier,
                                                      bool hasParentForOp) {
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
  Value inserted = b.create<ttng::InsertSliceAsyncV2Op>(
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
  if (hasParentForOp) {
    auto parentForOp = llvm::dyn_cast<mlir::scf::ForOp>(load->getParentOp());
    Value phase_ = parentForOp.getBody()->getArguments().back();
    Value one = b.create<arith::ConstantIntOp>(loc, 1, 32);
    phase = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, phase_, one);
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
    auto srcMmaLayout = srcTy.getEncoding().dyn_cast<MmaEncodingAttr>();
    auto dstBlockedLayout = dstTy.getEncoding().dyn_cast<BlockedEncodingAttr>();
    auto truncFOP = llvm::dyn_cast_or_null<arith::TruncFOp>(
        cvtOp.getOperand().getDefiningOp());
    unsigned numElems = ttg::getTotalElemsPerThread(srcTy);
    auto inOrd = ttg::getOrder(srcTy.getEncoding());
    auto outOrd = ttg::getOrder(dstTy.getEncoding());
    if (srcMmaLayout && srcMmaLayout.isHopper() && dstBlockedLayout &&
        truncFOP && elemTy.getIntOrFloatBitWidth() == 16 && numElems >= 16 &&
        inOrd == outOrd) {
      builder.create<ttng::StoreAsyncOp>(loc, dst, cvtOp.getOperand());
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
  builder.create<ttng::StoreAsyncOp>(loc, dst, cvt);
  builder.create<mlir::triton::gpu::AsyncBulkCommitGroupOp>(loc);
  builder.create<mlir::triton::gpu::AsyncBulkWaitOp>(loc, 0);
  store->erase();
}

void MaterializeLoadStorePass::createNewChildLoop(scf::ForOp forOp,
                                                  scf::ForOp parentForOp) {
  auto loc = forOp.getLoc();
  Block *body = forOp.getBody();
  OpBuilder builder(forOp.getContext());
  builder.setInsertionPoint(forOp);

  auto curPhase = body->insertArgument(body->getNumArguments(), builder.getI32Type(), loc);

  SmallVector<Value> newLoopArgs;
  for (auto operand : forOp.getInitArgs())
    newLoopArgs.push_back(operand);
  builder.setInsertionPoint(forOp);
  Value curInitPhase;
  if (parentForOp) {
    unsigned idx = parentForOp.getBody()->getNumArguments();
    curInitPhase = parentForOp.getBody()->getArgument(idx - 2);
  } else {
    curInitPhase = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  }
  newLoopArgs.push_back(curInitPhase);

  auto newForOp = builder.create<scf::ForOp>(
    loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
    newLoopArgs);
  newForOp.getRegion().takeBody(forOp.getRegion());

  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  forOp.erase();

  // child phase yield
  auto yieldOp = llvm::cast<scf::YieldOp>(body->getTerminator());
  builder.setInsertionPoint(yieldOp);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value curYieldPhase = builder.create<arith::XOrIOp>(loc, curPhase, one);
  yieldOp->insertOperands(yieldOp.getNumOperands(), {curYieldPhase});

  // parent phase yeild
  if (parentForOp) {
    auto pYieldOp = llvm::cast<scf::YieldOp>(parentForOp.getBody()->getTerminator());
    builder.setInsertionPoint(pYieldOp);
    Value one_ = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    Value pYieldPhase = builder.create<arith::XOrIOp>(loc, parentForOp.getBody()->getArguments().back(), one_);
    Value childYieldPhase = newForOp.getResults().back();
    pYieldOp->insertOperands(pYieldOp.getNumOperands(), {childYieldPhase, pYieldPhase});
  }
}

void MaterializeLoadStorePass::createNewParentLoop(scf::ForOp forOp,
                                                   scf::ForOp childForOp) {
  auto loc = forOp.getLoc();
  Block *body = forOp.getBody();
  OpBuilder builder(forOp.getContext());
  builder.setInsertionPoint(forOp);

  auto childPhase = body->insertArgument(body->getNumArguments(), builder.getI32Type(), loc);
  auto curPhase = body->insertArgument(body->getNumArguments(), builder.getI32Type(), loc);

  SmallVector<Value> newLoopArgs;
  for (auto operand : forOp.getInitArgs())
    newLoopArgs.push_back(operand);
  builder.setInsertionPoint(forOp);
  Value childInitPhase = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value curInitPhase = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  newLoopArgs.append({childInitPhase, curInitPhase});

  auto newForOp = builder.create<scf::ForOp>(
    loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
    newLoopArgs);
  newForOp.getRegion().takeBody(forOp.getRegion());

  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  forOp.erase();
}

SmallVector<Value>
MaterializeLoadStorePass::immutableMbarrierAlloc(Operation *entryOp,
                                                 unsigned loadNum) {
  SmallVector<Value> barrierLists;
  OpBuilder builder(entryOp);
  Location loc = entryOp->getLoc();
  builder.setInsertionPoint(entryOp);
  for (unsigned i = 0; i < loadNum; ++i) {
    auto mBarrierTy =
        mlir::triton::PointerType::get(builder.getIntegerType(64), 3);
    Value mBarrier = builder.create<ttng::AllocMBarrierOp>(loc, mBarrierTy, 1);
    barrierLists.push_back(mBarrier);
  }
  return barrierLists;
}

} // anonymous namespace

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUMaterializeLoadStorePass(int numWarps,
                                                    int computeCapability) {
  return std::make_unique<MaterializeLoadStorePass>(numWarps,
                                                    computeCapability);
}
