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

#include "Utility.hpp"

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

struct ArefValue {
  Value emptySemaphore;
  Value fullSemaphore;
  SmallVector<Value> buffers;
};

ArefValue createSemaphores(ArefCreateOp op, PatternRewriter &rewriter) {
  MLIRContext *ctx = op.getContext();
  auto loc = op.getLoc();
  auto arefTy = op.getType();
  auto baseType = arefTy.getBaseType();
  auto arefBufTypes = llvm::to_vector(llvm::map_range(
      arefTy.getBaseType(), [](Type type) { return cast<MemDescType>(type); }));
  auto shape = arefBufTypes[0].getShape();
  auto depth = shape[0];
  ImplicitLocOpBuilder b(op.getLoc(), rewriter);
  auto semaTy = triton::nvws::SemaphoreType::get(b.getContext(), depth);
  auto emptySempahore = b.create<SemaphoreCreateOp>(loc, semaTy, true);
  auto fullSempahore = b.create<SemaphoreCreateOp>(loc, semaTy, false);

  return ArefValue{emptySempahore, fullSempahore, op.getOperands()};
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
    Value singleBuffer = rewriter.create<MemDescSubviewOp>(loc, memDescTypeNew,
                                                           buffer, offsetsVal);
    views.push_back(singleBuffer);
  }

  return views;
}

void lowerAsyncLoads(ArefPutEnterOp op, PatternRewriter &rewriter,
                     ArefValue arefVal) {
  auto loc = op.getLoc();
  // for now handle TMA loads in PutEnterOp
  SmallVector<Operation *> loadOps;
  for (auto result : op.getResults())
    for (auto user : result.getUsers()) {
      // Temporary workaround for lit testing: handle TMA loads here until a
      // dedicated tma_load op is added to the NVWS dialect
      if (user->getName().getStringRef() == "tma_load")
        loadOps.push_back(user);
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
  ArefPutExitOp arefPutExitOp;
  auto arefTag = op->getAttrOfType<StringAttr>("aref_tag").str();
  for (auto user : op.getAref().getUsers()) {
    if (auto exitOp = dyn_cast<ArefPutExitOp>(user)) {
      if (exitOp->getAttrOfType<StringAttr>("aref_tag").str() == arefTag) {
        arefPutExitOp = exitOp;
        break;
      }
    }
  }
  assert(arefPutExitOp);
  assert(arefPutExitOp.getAref() == op.getAref() &&
         "Expecting matching Aref on the ArefPutExitOp");

#if 0
Value fullBarrier =
      getFullBarrier(rewriter, loc, arefVal, arefPutExitOp.getStage());
  Value pred = rewriter.create<arith::ConstantIntOp>(loc, 1, 1);
  rewriter.create<triton::nvidia_gpu::BarrierExpectOp>(loc, fullBarrier, 0,
                                                       pred);
#else
  llvm_unreachable("not implemented");
#endif
  return;
}

void rewritePutEnterOp(ArefCreateOp arefOp, ArefPutEnterOp op,
                       PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  rewriter.create<SemaphoreAcquireOp>(loc, arefVal.emptySemaphore,
                                      op.getStage(), Value());
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter);
  assert(views.size() == op.getResults().size());

  // TMA load need special handling as it requires fullMbarrier that
  // we need to get from matching ArefPutExitOp
  lowerAsyncLoads(op, rewriter, arefVal);

  // replaces uses with views
  for (int i = 0; i < arefVal.buffers.size(); ++i)
    op.getResult(i).replaceAllUsesWith(views[i]);
}

void rewriteGetEnterOp(ArefCreateOp arefOp, ArefGetEnterOp op,
                       PatternRewriter &rewriter, ArefValue arefVal) {
  auto loc = op.getLoc();
  rewriter.setInsertionPointAfter(op);

  rewriter.create<SemaphoreAcquireOp>(loc, arefVal.fullSemaphore, op.getStage(),
                                      Value());
  auto views = getSubViews(arefVal, op.getStage(), loc, rewriter);
  assert(views.size() == op.getResults().size());

  for (int i = 0; i < arefVal.buffers.size(); ++i)
    op.getResult(i).replaceAllUsesWith(views[i]);
}

void rewritePutExitOp(ArefPutExitOp op, PatternRewriter &rewriter,
                      ArefValue arefVal) {
  auto loc = op->getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<SemaphoreReleaseOp>(loc, arefVal.fullSemaphore, op.getStage(),
                                      op.getAsyncOps());
}

void rewriteGetExitOp(ArefGetExitOp op, PatternRewriter &rewriter,
                      ArefValue arefVal) {
  auto loc = op->getLoc();
  rewriter.setInsertionPointAfter(op);
  rewriter.create<SemaphoreReleaseOp>(loc, arefVal.emptySemaphore,
                                      op.getStage(), op.getAsyncOps());
}

class LowerArefCreate : public OpRewritePattern<ArefCreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ArefCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto aref = createSemaphores(op, rewriter);
    llvm::SmallSetVector<Operation *, 10> opToDelete;
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

    for (auto it = opToDelete.rbegin(); it != opToDelete.rend(); ++it)
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

    SmallVector<WarpGroupOp> wgOps;
    m.walk([&](WarpGroupOp wgOp) { wgOps.push_back(wgOp); });
    for (auto wgOp : wgOps) {
      auto updateStage = [](ImplicitLocOpBuilder &b, Value stage,
                            Operation *op) -> Value {
        // set current stage
        op->setOperand(1, stage);
        // compute next stage
        auto nextStage = b.create<arith::AddIOp>(
            stage, b.create<arith::ConstantIntOp>(1, 32));
        auto arefBuf = op->getOperand(0)
                           .template getDefiningOp<nvws::ArefCreateOp>()
                           .getOperand(0);
        auto depth = cast<MemDescType>(arefBuf.getType()).getShape().front();

        auto cnd =
            b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, nextStage,
                                    b.create<arith::ConstantIntOp>(depth, 32));
        auto zero = b.create<arith::ConstantIntOp>(0, 32);
        return b.create<arith::SelectOp>(cnd, zero, nextStage);
      };
      auto initStage = [](ImplicitLocOpBuilder &b, Operation *op) -> Value {
        return b.create<arith::ConstantIntOp>(0, 32);
      };
      ThreadValue<ArefPutEnterOp>::run(wgOp, initStage, updateStage);
      ThreadValue<ArefPutExitOp>::run(wgOp, initStage, updateStage);
      ThreadValue<ArefGetEnterOp>::run(wgOp, initStage, updateStage);
      ThreadValue<ArefGetExitOp>::run(wgOp, initStage, updateStage);
    }
    LLVM_DEBUG(llvm::dbgs() << "After ArefStageAssignment\n" << m << "\n");

    mlir::RewritePatternSet patterns(context);
    patterns.add<LowerArefCreate>(context);
    GreedyRewriteConfig config;
    if (applyPatternsGreedily(m, std::move(patterns), config).failed())
      signalPassFailure();
  }
}; // namespace triton

} // namespace triton
} // namespace mlir
