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

#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/IR/OperationSupport.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

enum class LoadType {
  Uninitialized,
  InsertSliceAsyncOp,
  InsertSliceAsyncV2Op,
  MultiKinds,
};

//===----------------------------------------------------------------------===//
// Materialize GetAgentIdOp
//===----------------------------------------------------------------------===//

void materializeGetAgentIdOp(Operation *parentOp) {
  parentOp->walk([](ttng::GetAgentIdOp op) {
    // In Hopper, each agent is a warpgroup consisting with 4 warps.
    auto loc = op.getLoc();
    OpBuilder builder(op);
    auto cast = builder.create<UnrealizedConversionCastOp>(
        loc, TypeRange{builder.getIntegerType(32)},
        ValueRange{builder.create<::mlir::gpu::ThreadIdOp>(
            loc, builder.getIndexType(), ::mlir::gpu::Dimension::x)});
    Value threadId = cast.getResult(0);
    Value _128 = builder.create<arith::ConstantIntOp>(loc, 128, 32);
    Value ret = builder.create<arith::DivUIOp>(loc, threadId, _128);
    op.getResult().replaceAllUsesWith(ret);
    op->erase();
  });
}

//===----------------------------------------------------------------------===//
// Materialize token operations
//===----------------------------------------------------------------------===//
Value getMBarrierPhaseBit(OpBuilder &builder, Operation *op,
                          bool skipFirstWait) {
  // TODO: currently we only support one loop, no nested loop, while or
  // condition.
  auto loc = op->getLoc();
  auto forOp = op->getParentOfType<scf::ForOp>();
  if (!forOp) {
    return builder.create<arith::ConstantIntOp>(loc, skipFirstWait, 1);
  }

  auto defOp = op->getOperand(0).getDefiningOp();
  assert(isa<ttng::CreateTokenOp>(defOp) &&
         "mbarrier's definingOp is not createTokenOp");
  ttng::CreateTokenOp createTokenOp = dyn_cast<ttng::CreateTokenOp>(defOp);
  Value numStage =
      builder.create<arith::ConstantIntOp>(loc, createTokenOp.getNum(), 32);
  Value curStep = forOp.getBody()->getArguments().back();
  if (curStep.getType() == builder.getIndexType()) {
    curStep =
        builder.create<arith::IndexCastOp>(loc, numStage.getType(), curStep);
  }
  Value curPhase = builder.create<arith::DivUIOp>(loc, curStep, numStage);
  if (skipFirstWait) {
    // If skipFirstWait, it waits for phaseBit 1
    Value _1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
    curPhase = builder.create<arith::AddIOp>(loc, curPhase, _1);
  }
  Value _2 = builder.create<arith::ConstantIntOp>(loc, 2, 32);
  // TODO: May use alternative methods of phaseBit calculation to avoid high
  // overhead of RemOp
  Value phaseBit = builder.create<arith::RemUIOp>(loc, curPhase, _2);
  Value _0 = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, phaseBit,
                                       _0);
}

template <typename T> int getTxBytes(Operation &op) {
  auto load = dyn_cast<T>(op);
  // Both support ptr of tensor and tensor of ptr.
  RankedTensorType srcTensorType;
  if (auto srcType = dyn_cast<RankedTensorType>(load.getSrc().getType())) {
    srcTensorType = srcType;
  } else if (auto srcType =
                 dyn_cast<triton::PointerType>(load.getSrc().getType())) {
    srcTensorType = dyn_cast<RankedTensorType>(srcType.getPointeeType());
  } else {
    llvm_unreachable("Unexpected src type");
  }
  auto elemTy =
      dyn_cast<RankedTensorType>(load.getDst().getType()).getElementType();
  int bytesPerElem = elemTy.getIntOrFloatBitWidth() / 8;
  return srcTensorType.getNumElements() * bytesPerElem;
}

void applyCommit(OpBuilder &builder, ttng::ProducerCommitOp &op, Value mbarrier,
                 int &txCnt, int &isTrackAsyncTrue) {
  // TODO: currently it only handles loads in ProducerCommitOp's nearest parent
  // block. Neither support multiple ProducerCommitOp, e.g. fused attention,
  // epilogue fusion.
  SmallVector<Operation *> deprecatedOps;
  auto agentIds = getAgentIds(op);
  // Materialize InsertSliceOp
  for (auto &ItrOp : op->getBlock()->getOperations()) {
    // Check operations before ProducerCommitOp
    if (OperationEquivalence::isEquivalentTo(&ItrOp, op.getOperation(),
                                             OperationEquivalence::None)) {
      break;
    }
    if (auto insertOp = dyn_cast<ttg::InsertSliceOp>(ItrOp)) {
      deprecatedOps.push_back(&ItrOp);
      builder.setInsertionPoint(insertOp);
      if (!::mlir::triton::isTensorPointerType(insertOp.getSrc().getType())) {
        // Transform to InsertSliceAsyncOp
        Value v = builder.create<triton::gpu::InsertSliceAsyncOp>(
            /*loc=*/insertOp.getLoc(), /*result=*/insertOp.getDst().getType(),
            /*src=*/insertOp.getSrc(), /*dst=*/insertOp.getDst(),
            /*index=*/insertOp.getIndex(),
            /*mask=*/insertOp.getMask(), /*other=*/insertOp.getOther(),
            /*cache=*/insertOp.getCache(), /*evict=*/insertOp.getEvict(),
            /*isVolatile=*/insertOp.getIsVolatile(),
            /*axis=*/insertOp.getAxis());
        insertOp.getResult().replaceAllUsesWith(v);
        setAgentIds(v.getDefiningOp(), agentIds);
      } else {
        // Transform to InsertSliceAsyncV2Op
        auto extractBarrierOp = dyn_cast<ttng::ExtractMBarrierOp>(
            builder.clone(*(mbarrier.getDefiningOp())));
        Value v = builder.create<ttng::InsertSliceAsyncV2Op>(
            /*loc=*/insertOp.getLoc(), /*result=*/insertOp.getDst().getType(),
            /*src=*/insertOp.getSrc(), /*dst=*/insertOp.getDst(),
            /*index=*/insertOp.getIndex(),
            /*mbar*/ extractBarrierOp.getResult(), /*mask=*/insertOp.getMask(),
            /*other=*/insertOp.getOther(),
            /*cache=*/insertOp.getCache(), /*evict=*/insertOp.getEvict(),
            /*isVolatile=*/insertOp.getIsVolatile(),
            /*axis=*/insertOp.getAxis());
        insertOp.getResult().replaceAllUsesWith(v);
        setAgentIds(v.getDefiningOp(), agentIds);
      }
    }
  }
  builder.setInsertionPoint(op);
  for (auto d : deprecatedOps) {
    d->erase();
  }

  // Collect commit stats
  txCnt = 0;
  LoadType loadType = LoadType::Uninitialized;
  for (auto &ItrOp : op->getBlock()->getOperations()) {
    if (OperationEquivalence::isEquivalentTo(&ItrOp, op.getOperation(),
                                             OperationEquivalence::None)) {
      break;
    }
    if (isa<ttg::InsertSliceAsyncOp>(ItrOp)) {
      loadType = (loadType == LoadType::Uninitialized ||
                  loadType == LoadType::InsertSliceAsyncOp)
                     ? LoadType::InsertSliceAsyncOp
                     : LoadType::MultiKinds;
    } else if (isa<ttng::InsertSliceAsyncV2Op>(ItrOp)) {
      txCnt += getTxBytes<ttng::InsertSliceAsyncV2Op>(ItrOp);
      loadType = (loadType == LoadType::Uninitialized ||
                  loadType == LoadType::InsertSliceAsyncV2Op)
                     ? LoadType::InsertSliceAsyncV2Op
                     : LoadType::MultiKinds;
    }
  }

  assert(loadType != LoadType::Uninitialized && "Load type is not recognized");
  assert(loadType != LoadType::MultiKinds &&
         "Multiple kinds of load types are not expected");
  isTrackAsyncTrue = loadType == LoadType::InsertSliceAsyncOp ? 1 : 0;
  // TODO: when we use one thread to arrive, then this average is no longer
  // needed.
  txCnt = loadType == LoadType::InsertSliceAsyncV2Op ? txCnt / 128 : txCnt;
}

void processProducerAcquireOp(OpBuilder &builder, ttng::ProducerAcquireOp op,
                              Value bufferEmpty) {
  auto loc = op.getLoc();
  // The first producer_aquire should be met immediately, so initailly producer
  // skips the fisrt wait
  Value phase = getMBarrierPhaseBit(builder, op, 1);
  auto waitOp = builder.create<ttng::MBarrierWaitOp>(loc, bufferEmpty, phase);
  assert(op.getOperation()->hasAttr("async_agent"));
  setAgentIds(waitOp, getAgentIds(op.getOperation()));
}

void processProducerCommitOp(OpBuilder &builder, ttng::ProducerCommitOp op,
                             Value bufferFull) {
  auto loc = op.getLoc();
  Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  int txCount = 0, isTrackAsyncTrue = 0;
  applyCommit(builder, op, bufferFull, txCount, isTrackAsyncTrue);
  auto arriveOp = builder.create<ttng::MBarrierArriveOp>(
      loc, bufferFull, pred,
      /*remoteCTAId*/ nullptr, isTrackAsyncTrue == 0 ? false : true, txCount);
  assert(op.getOperation()->hasAttr("async_agent"));
  setAgentIds(arriveOp, getAgentIds(op.getOperation()));
}

void processConsumerWaitOp(OpBuilder &builder, ttng::ConsumerWaitOp op,
                           Value bufferFull) {
  auto loc = op.getLoc();
  Value phase = getMBarrierPhaseBit(builder, op, 0);
  auto waitOp = builder.create<ttng::MBarrierWaitOp>(loc, bufferFull, phase);
  assert(op.getOperation()->hasAttr("async_agent"));
  setAgentIds(waitOp, getAgentIds(op.getOperation()));
}

void processConsumerReleaseOp(OpBuilder &builder, ttng::ConsumerReleaseOp op,
                              Value bufferEmpty) {
  auto loc = op.getLoc();
  Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  auto arriveOp =
      builder.create<ttng::MBarrierArriveOp>(loc, bufferEmpty, pred,
                                             /*remoteCTAId*/ nullptr, false, 0);
  assert(op.getOperation()->hasAttr("async_agent"));
  setAgentIds(arriveOp, getAgentIds(op.getOperation()));
}

void materializeTokenOperations(Operation *parentOp) {
  SmallVector<Operation *> deprecatedOps;
  parentOp->walk([&](ttng::CreateTokenOp createTokenOp) {
    // mBarrierTy
    MLIRContext *context = createTokenOp.getContext();
    auto i64Ty = IntegerType::get(context, 64);
    auto mBarrierTy = triton::PointerType::get(i64Ty, 3);

    // mBarriersTy
    auto CTALayout = ttg::CTALayoutAttr::get(context, {1}, {1}, {0});
    auto sharedLayout =
        ttg::SharedEncodingAttr::get(context, 1, 1, 1, {0}, CTALayout, false);
    auto mBarriersTy =
        RankedTensorType::get({createTokenOp.getNum()}, i64Ty, sharedLayout);

    // Process CreateTokenOp
    OpBuilder builder(createTokenOp);
    // TODO: count of AllocMBarrierOp could be less hard-coded
    // TODO: implement optimization of arrive_count = 1, as cutlass does.
    Value bufferFullArray = builder.create<ttng::AllocMBarrierOp>(
        createTokenOp.getLoc(), mBarriersTy, 128);
    Value bufferEmptyArray = builder.create<ttng::AllocMBarrierOp>(
        createTokenOp.getLoc(), mBarriersTy, 128);

    // Helper function for extracting bufferFull
    auto extractBufferFull = [&](Location loc, Value idx) -> Value {
      return builder.create<ttng::ExtractMBarrierOp>(loc, mBarrierTy,
                                                     bufferFullArray, idx);
    };

    // Helper function for extracting bufferEmpty
    auto extractBufferEmpty = [&](Location loc, Value idx) -> Value {
      return builder.create<ttng::ExtractMBarrierOp>(loc, mBarrierTy,
                                                     bufferEmptyArray, idx);
    };

    // Process token users
    for (Operation *user : createTokenOp.getResult().getUsers()) {
      auto loc = user->getLoc();
      builder.setInsertionPoint(user);
      if (auto op = dyn_cast<ttng::ProducerAcquireOp>(user)) {
        Value bufferEmpty = extractBufferEmpty(loc, op.getIdx());
        assert(user->hasAttr("async_agent"));
        setAgentIds(bufferEmpty.getDefiningOp(), getAgentIds(user));
        processProducerAcquireOp(builder, op, bufferEmpty);
      } else if (auto op = dyn_cast<ttng::ProducerCommitOp>(user)) {
        Value bufferFull = extractBufferFull(loc, op.getIdx());
        assert(user->hasAttr("async_agent"));
        setAgentIds(bufferFull.getDefiningOp(), getAgentIds(user));
        processProducerCommitOp(builder, op, bufferFull);
      } else if (auto op = dyn_cast<ttng::ConsumerWaitOp>(user)) {
        Value bufferFull = extractBufferFull(loc, op.getIdx());
        assert(user->hasAttr("async_agent"));
        setAgentIds(bufferFull.getDefiningOp(), getAgentIds(user));
        processConsumerWaitOp(builder, op, bufferFull);
      } else if (auto op = dyn_cast<ttng::ConsumerReleaseOp>(user)) {
        Value bufferEmpty = extractBufferEmpty(loc, op.getIdx());
        assert(user->hasAttr("async_agent"));
        setAgentIds(bufferEmpty.getDefiningOp(), getAgentIds(user));
        processConsumerReleaseOp(builder, op, bufferEmpty);
      } else {
        llvm_unreachable("Unexpected user of token");
      }
      // user->erase();
      deprecatedOps.push_back(user);
    }

    // createTokenOp.erase();
    deprecatedOps.push_back(createTokenOp);
  });
  for (auto op : deprecatedOps) {
    op->erase();
  }
}

//===----------------------------------------------------------------------===//
// Materialize mutex operations
//===----------------------------------------------------------------------===//

void processLockOp(OpBuilder &builder, ttng::LockOp op, Value barrier) {
  // TODO
  auto loc = op.getLoc();
  Value numThreads = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  builder.create<ttng::NamedBarrierWaitOp>(loc, barrier, numThreads);
}

void processUnlockOp(OpBuilder &builder, ttng::UnlockOp op, Value barrier) {
  // TODO
  auto loc = op.getLoc();
  Value numThreads = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  builder.create<ttng::NamedBarrierArriveOp>(loc, barrier, numThreads);
}

void materializeMutexOperations(Operation *parentOp) {
  parentOp->walk([](ttng::CreateMutexOp createMutexOp) {
    // Process CreateMutexOp
    OpBuilder builder(createMutexOp);
    Value barrier =
        builder.create<arith::ConstantIntOp>(createMutexOp.getLoc(), 0, 32);

    // Process mutex users
    for (Operation *user : createMutexOp.getResult().getUsers()) {
      auto loc = user->getLoc();
      builder.setInsertionPoint(user);
      if (auto op = dyn_cast<ttng::LockOp>(user))
        processLockOp(builder, op, barrier);
      else if (auto op = dyn_cast<ttng::UnlockOp>(user))
        processUnlockOp(builder, op, barrier);
      else
        llvm_unreachable("Unexpected user of mutex");
      user->erase();
    }

    createMutexOp.erase();
  });
}

//===----------------------------------------------------------------------===//
// WSMaterializationPass
//===----------------------------------------------------------------------===//

struct WSMaterializationPass
    : public TritonGPUWSMaterializationBase<WSMaterializationPass> {
  WSMaterializationPass() = default;
  WSMaterializationPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (!ttng::TritonNvidiaGPUDialect::getWSSupportedAttr(mod))
      return signalPassFailure();

    if (computeCapability / 10 < 9) {
      llvm_unreachable("WSMaterialization pass only supports sm_9x as of now.");
      signalPassFailure();
    }

    materializeGetAgentIdOp(mod);
    materializeTokenOperations(mod);
    materializeMutexOperations(mod);

    mod->walk([](Operation *op) {
      bool hasTensor = 0;
      auto results = op->getResults();
      auto operands = op->getOperands();
      for (auto i : results) {
        if (isa<RankedTensorType>(i.getType())) {
          hasTensor = 1;
          break;
        }
      }
      if (!hasTensor) {
        for (auto i : operands) {
          if (isa<RankedTensorType>(i.getType())) {
            hasTensor = 1;
            break;
          }
        }
      }

      if (!hasTensor && !isa<ttng::MBarrierWaitOp>(op) &&
          !isa<ttng::ExtractMBarrierOp>(op) &&
          !isa<ttng::MBarrierArriveOp>(op)) {
        op->removeAttr("async_agent");
      }
    });

    // TODO: More flexible way to set num-warps
    // One dma, one math warp group, set num-warps = 8
    auto i32_ty = IntegerType::get(mod->getContext(), 32);
    mod->setAttr("triton_gpu.num-warps",
                 IntegerAttr::get(i32_ty, llvm::APInt(32, 8)));
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// createTritonNvidiaGPUWSMaterializationPass
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUWSMaterializationPass(int computeCapability) {
  return std::make_unique<WSMaterializationPass>(computeCapability);
}
