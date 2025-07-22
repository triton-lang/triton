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

#include "Utility.hpp"
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
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

#define DEBUG_TYPE "nvws-lower-semaphore"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#if 1
#define MULTIPHASE
#endif

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSLOWERSEMAPHORE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

namespace {

std::pair<WarpGroupOp, int> getWarpGroupIdx(Operation *op) {
  if (auto wgOp = dyn_cast<WarpGroupOp>(op->getParentOp())) {
    auto region = op->getParentRegion();
    return {wgOp, region->getRegionNumber()};
  }
  if (isa<triton::FuncOp>(op))
    return {nullptr, -1};
  return getWarpGroupIdx(op->getParentOp());
}

int getPendingCount(SemaphoreCreateOp op) {
  std::optional<int> arrivalCount;

  for (auto user : op->getUsers()) {
    auto [wgOp, idx] = getWarpGroupIdx(user);
    auto numWarps = wgOp.getNumWarps()[idx];

    if (auto releaseOp = dyn_cast<SemaphoreReleaseOp>(user)) {
      int count = 0;
      for (auto prod : releaseOp.getAsyncOps()) {
        auto kind = dyn_cast<AsyncOpAttr>(prod).getValue();
        switch (kind) {
        case AsyncOp::TC5MMA:
        case AsyncOp::TMALoad:
          count += 1;
          break;
        case AsyncOp::CpAsync:
          count += numWarps * 32;
          break;
        case AsyncOp::NONE:
          // TODO: this should be 'numWarps * 32' when we transition to
          //       multi-threaded arrive
          count += 1;
          break;
        default:
          llvm_unreachable("unknown producer kind");
        }
      }

      if (arrivalCount) {
        assert(*arrivalCount == count && "inconsistent producer arrival count");
      } else {
        arrivalCount = count;
      }
    }
  }

  assert(arrivalCount);

  return *arrivalCount;
}

Value createAndInitMbar(SemaphoreCreateOp op, PatternRewriter &rewriter) {
  auto pendingCount = getPendingCount(op);

  MLIRContext *ctx = op.getContext();
  auto loc = op.getLoc();
  auto numStages = *op.getType().getNumStages();

  ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
  auto mbars = createScalarAlloc(builder, rewriter.getI64Type(), numStages);
  for (int i = 0; i < numStages; i++) {
    auto singleBarrier = createSingleBufferView(rewriter, mbars, i);
    rewriter.create<InitBarrierOp>(loc, singleBarrier, pendingCount);
  }
  return mbars;
}

void rewriteAcquireOp(SemaphoreCreateOp semaphoreOp, SemaphoreAcquireOp op,
                      PatternRewriter &rewriter, Value mbars) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());
#ifdef MULTIPHASE
  // extract phase for a given stage from the phase bit-vector
  // phase = (phase_vector >> stage) & 1
  Value phaseBit =
      rewriter.create<arith::ShRSIOp>(loc, op.getPhase(), op.getStage());
  phaseBit = rewriter.create<arith::AndIOp>(
      loc, phaseBit, rewriter.create<arith::ConstantIntOp>(loc, 1, 32));
#else
  Value phaseBit = op.getPhase();
#endif
  rewriter.create<WaitBarrierOp>(loc, mbar, phaseBit);
}

void rewriteReleaseOp(SemaphoreCreateOp semaphoreOp, SemaphoreReleaseOp op,
                      PatternRewriter &rewriter, Value mbars) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);
  auto mbar = createSingleBufferView(rewriter, mbars, op.getStage());
  for (auto asyncOp : op.getAsyncOps()) {
    auto asyncOpEnum = cast<AsyncOpAttr>(asyncOp).getValue();
    switch (asyncOpEnum) {
    case AsyncOp::NONE:
    case AsyncOp::WGMMA:
      rewriter.create<nvidia_gpu::ArriveBarrierOp>(loc, mbar, 1);
      break;
    case AsyncOp::TC5MMA:
    case AsyncOp::TMEMCopy:
      rewriter.create<nvidia_gpu::TCGen5CommitOp>(loc, mbar);
      break;

    case AsyncOp::TMALoad:
      // nothing to do, TMA load is handled by lowering putEnterOp
      break;
    case AsyncOp::CpAsync:
    default:
      llvm_unreachable("unknown async op");
    }
  }
}

class LowerSemaphoreCreate : public OpRewritePattern<SemaphoreCreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SemaphoreCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto mbars = createAndInitMbar(op, rewriter);
    llvm::SmallSetVector<Operation *, 10> opToDelete;
    opToDelete.insert(op.getOperation());
    for (auto userOp : op->getUsers()) {
      if (auto user = dyn_cast<SemaphoreAcquireOp>(userOp)) {
        opToDelete.insert(user);
        rewriteAcquireOp(op, user, rewriter, mbars);
      } else if (auto user = dyn_cast<SemaphoreReleaseOp>(userOp)) {
        opToDelete.insert(user);
        rewriteReleaseOp(op, user, rewriter, mbars);
      } else {
        llvm_unreachable("users of aref can only be ArefPut or ArefGet");
      }
    }

    for (auto it = opToDelete.rbegin(); it != opToDelete.rend(); ++it)
      rewriter.eraseOp(*it);

    return success();
  }
};

} // anonymous namespace

class NVWSLowerSemaphore
    : public impl::NVWSLowerSemaphoreBase<NVWSLowerSemaphore> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();
    SmallVector<WarpGroupOp> wgOps;
    m.walk([&](WarpGroupOp wgOp) { wgOps.push_back(wgOp); });
    for (auto wgOp : wgOps) {
      auto updatePhase = [](ImplicitLocOpBuilder &b, Value phase,
                            Operation *op) -> Value {
        auto acquireOp = cast<SemaphoreAcquireOp>(op);
        acquireOp.getPhaseMutable().assign(phase);
#ifdef MULTIPHASE
        // the phase is a bit-vector, each bit for each stage
        // next_phase = phase_vector ^ (1 << stage)
        auto phaseBit = b.create<arith::ShLIOp>(
            b.create<arith::ConstantIntOp>(1, 32), op->getOperand(1));
        return b.create<arith::XOrIOp>(phase, phaseBit);
#else
        // pattern match stage computation
        // when stage & phase are in sync
        //   nextStage = stage + 1
        //   cnd = nextStage == depth
        //   nextStage = select(cnd, 0, nextStage)

        Operation *addi = {};
        for (auto user : acquireOp.getStage().getUsers()) {
          if (isa<arith::AddIOp>(user)) {
            assert(!addi);
            addi = user;
          }
        }
        assert(addi);
        Operation *cnd = {};
        for (auto user : addi->getUsers()) {
          if (isa<arith::CmpIOp>(user)) {
            assert(!cnd);
            cnd = user;
          }
        }
        assert(cnd);
        Operation *select = {};
        for (auto user : cnd->getUsers()) {
          if (isa<arith::SelectOp>(user)) {
            assert(!select);
            select = user;
          }
        }
        assert(select);
        {
          // and inject after select of nextStage
          //   nextPhase = phase ^ 1
          //   nextPhase = select(cnd, nextPhase, phase)
          ImplicitLocOpBuilder::InsertionGuard guard(b);
          b.setInsertionPointAfter(select);
          auto nextPhase = b.create<arith::XOrIOp>(
              phase, b.create<arith::ConstantIntOp>(1, 32));
          return b.create<arith::SelectOp>(cnd->getResult(0), nextPhase, phase);
        }
#endif
      };
      auto initPhase = [](ImplicitLocOpBuilder &b, Operation *op) -> Value {
        auto semaOp = cast<SemaphoreCreateOp>(op);
        bool isReleased = semaOp.getIsReleased();
#ifdef MULTIPHASE
        return b.create<arith::ConstantIntOp>(
            isReleased ? 0xFFFFFFFF : 0x00000000, 32);
#else
        return b.create<arith::ConstantIntOp>(isReleased ? 1 : 0, 32);
#endif
      };
      ThreadValue<SemaphoreAcquireOp>::run(wgOp, initPhase, updatePhase);
    }
    LLVM_DEBUG(llvm::dbgs() << "After SemaphorePhaseAssignment\n" << m << "\n");

    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerSemaphoreCreate>(context);
    GreedyRewriteConfig config;
    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();
  }
}; // namespace triton

} // namespace triton
} // namespace mlir
