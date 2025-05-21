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
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/WSUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <memory>

// #define GEN_PASS_CLASSES
// #include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-aref-async-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
using namespace ttg;

void insertWgmmaWait(triton::FuncOp funcOp) {
  SmallVector<ttng::WarpGroupDotOp> dotOps;
  funcOp->walk([&](ttng::WarpGroupDotOp dotOp) { dotOps.push_back(dotOp); });
  for (auto dotOp : dotOps) {
    dotOp.setIsAsync(true);
    OpBuilder builder(dotOp);
    builder.setInsertionPointAfter(dotOp);
    int pendings = 0;
    auto waitOp = builder.create<ttng::WarpGroupDotWaitOp>(
        dotOp.getLoc(), dotOp.getResult(), pendings);
    dotOp.getResult().replaceAllUsesExcept(waitOp.getResult(0), waitOp);
  }
}

void fixUpArriveWaitMMAv5(tt::FuncOp funcOp) {
  std::function<ttng::TMEMAllocOp(Operation *)> findTMEMAllocOp =
      [&](Operation *op) -> ttng::TMEMAllocOp {
    if (isa<ttng::ArefGetEnterOp>(op)) {
      // stop at aref_get.enter, no need to propagate futher
      return {};
    } else if (auto allocOp = dyn_cast<ttng::TMEMAllocOp>(op)) {
      return allocOp;
    } else {
      for (auto user : op->getOperands())
        if (auto userOp = user.getDefiningOp())
          if (auto producer = findTMEMAllocOp(userOp))
            return producer;
      return {};
    }
  };

  std::function<bool(Operation *)> isUsedByMMAv5 = [&](Operation *op) {
    // transitively check if one of the user of the op is mmav5
    if (isa<ttng::MMAv5OpInterface>(op)) {
      return true;
    } else {
      for (auto user : op->getUsers())
        if (isUsedByMMAv5(user))
          return true;
      return false;
    }
  };

  DenseMap<Value, std::pair<ttng::ArefCreateOp, MemDescType>> alloc2arefMap;
  int mmaSyncTag = 0;
  funcOp.walk([&](ttng::TMEMLoadOp loadOp) {
    if (auto allocOp = findTMEMAllocOp(loadOp)) {
      if (allocOp->hasAttr("aref_buffer"))
        return WalkResult::advance();

      if (!isUsedByMMAv5(allocOp))
        return WalkResult::advance();

      // We have quite a few use cases where we don't do much with MMAv5,
      // e.g.
      //   %a = alloc
      //   ..
      //   mma %a  {isAsync}
      //   ..
      //   %b = tmem_load %a

      // we need to do some ad-hoc patching to make such mma wait for %a to
      // be ready before we tmem_load it.

      // Due to lack of data flow analysis, we conservatively assume that
      // %src in tmem_load is produced by tcgen5.mma, if it originates from
      // tmem_alloc and not from aref_get.enter, and, transitively, one of
      // the user of this tmem_alloc is also mma

      // If %src comes from %alloc, we need to patch it:
      //    %a = alloc : <128x128>
      //    ...
      //    mma %a
      //    ...
      //    tma_load %a
      //    ...
      // becomes:
      //    %a' = alloc  <1x128x128>
      //    %aref = aref_create %a' : <1x128x128>
      // .  %a'' = view %a'[0] : <128x128>
      //    ...
      //    mma %a''
      //    ...
      //    aref_put.enter %aref                   (try_wait consumed)
      //    aref_put.exit %aref, producer=[umma]   (tcgen5.commit produced)
      //    aref_get.enter %aref                   (try_wait produced)
      //    tma_load %a''
      //    aref_get.exit %aref, consumer=[ldtm]   (arrive consumed)
      //    ...

      // arefLower could later optimize away first put.enter and last
      // get.exit

      // Ideally, we want to patch if there is a data-flow edge between
      // tcgen5.mma and tcgen5.ldtm. Once we have tmem data flow analysis,
      // we can update this logic, but for now we stay conservative.

      OpBuilder builder(allocOp);
      builder.setInsertionPointAfter(allocOp);

      if (!alloc2arefMap.count(allocOp)) {
        // Multiple tmem_loads may refer to the same alloc. After updating
        // the alloc with aref_create, map the new alloc to its aref. This
        // ensures that subsequent tmem_loads referencing the same alloc
        // won't update alloc again.
        auto ctx = allocOp.getContext();
        auto allocTy = allocOp.getType();
        auto arefBufTy = getArefbufMemDescType(allocOp.getType(), 1);
        auto arefTy =
            ttng::ArefType::get(ctx, TypeArrayAttr::get(ctx, arefBufTy));
        auto arefAlloc = builder.create<ttng::TMEMAllocOp>(
            allocOp.getLoc(),
            SmallVector<Type>{arefBufTy, builder.getType<AsyncTokenType>()},
            Value{});
        arefAlloc->setAttr("mma_sync_buffer", builder.getUnitAttr());
        auto arefOp = builder.create<ttng::ArefCreateOp>(
            allocOp.getLoc(), arefTy, arefAlloc.getResult());
        arefOp->setAttr("aref_mma_sync", builder.getUnitAttr());
        alloc2arefMap[arefAlloc] = {arefOp, allocTy};

        SmallVector<Value> offsetsVal{
            mkConstant(builder, loadOp.getLoc(), 0, 32, {})};
        auto memDescType = cast<MemDescType>(arefAlloc.getType());
        auto shape = memDescType.getShape();
        auto rank = shape.size() - 1;
        for (int i = 0; i < rank; ++i)
          offsetsVal.push_back(
              mkConstant(builder, allocOp.getLoc(), 0, 32, {}));
        SmallVector<int64_t> tensorShape(shape.begin() + 1, shape.end());
        auto memDescTypeNew = MemDescType::get(
            tensorShape, memDescType.getElementType(),
            memDescType.getEncoding(), memDescType.getMemorySpace(), true);
        Value singleBuffer = builder.create<triton::gpu::MemDescSubviewOp>(
            allocOp.getLoc(), memDescTypeNew, arefAlloc.getResult(),
            offsetsVal);
        allocOp.getResult().replaceAllUsesWith(singleBuffer);
        if (allocOp.getToken())
          allocOp.getToken().replaceAllUsesWith(arefAlloc.getToken());
        allocOp.erase();
        allocOp = arefAlloc;
      }

      // get aref associated with allocOp
      auto [arefOp, allocTy] = alloc2arefMap[allocOp];

      builder.setInsertionPoint(loadOp);

      auto tag = std::string("mma_sync_") + std::to_string(mmaSyncTag);

      auto putEnterOp = builder.create<triton::nvidia_gpu::ArefPutEnterOp>(
          loadOp.getLoc(), allocTy, SmallVector<Type>{}, arefOp,
          mkConstant(builder, loadOp.getLoc(), 0, 32, {}));
      putEnterOp->setAttr("aref_tag", builder.getStringAttr(tag));

      auto putExitOp = builder.create<triton::nvidia_gpu::ArefPutExitOp>(
          loadOp.getLoc(), arefOp,
          mkConstant(builder, loadOp.getLoc(), 0, 32, {}),
          ArrayAttr::get(builder.getContext(),
                         ttng::ArefProducerAttr::get(
                             builder.getContext(), ttng::ArefProducer::UMMA)));
      putExitOp->setAttr("aref_tag", builder.getStringAttr(tag));

      auto getEnterOp = builder.create<ttng::ArefGetEnterOp>(
          loadOp.getLoc(), allocTy, SmallVector<Type>{}, arefOp,
          mkConstant(builder, loadOp.getLoc(), 0, 32, {}));
      getEnterOp->setAttr("aref_tag", builder.getStringAttr(tag));

      builder.setInsertionPointAfter(loadOp);

      auto getExitOp = builder.create<ttng::ArefGetExitOp>(
          loadOp.getLoc(), arefOp,
          mkConstant(builder, loadOp.getLoc(), 0, 32, {}),
          builder.getArrayAttr(ttng::ArefConsumerAttr::get(
              builder.getContext(), ttng::ArefConsumer::LDTM)));
      getExitOp->setAttr("aref_tag", builder.getStringAttr(tag));
      ++mmaSyncTag;
    }
    return WalkResult::advance();
  });
}

void makeMMAv5Async(triton::FuncOp funcOp) {
  funcOp.walk([&](ttng::MMAv5OpInterface op) { op.setIsAsync(true); });
}

class NVWSArefAsyncOps : public NVWSArefAsyncOpsBase<NVWSArefAsyncOps> {

public:
  void runOnFunc(triton::FuncOp funcOp) {
    insertWgmmaWait(funcOp);
    LLVM_DEBUG({ DBGS() << "after::insertWgmmaWait:\n" << funcOp << "\n"; });

    makeMMAv5Async(funcOp);

    fixUpArriveWaitMMAv5(funcOp);
    LLVM_DEBUG({
      DBGS() << "after::fixUpArriveWaitMMAv5:\n" << funcOp << "\n";
    });
  }

  void runOnOperation() override {
    auto mod = getOperation();
    mod.walk([&](triton::FuncOp funcOp) { runOnFunc(funcOp); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createNVWSArefAsyncOpsPass() {
  return std::make_unique<NVWSArefAsyncOps>();
}
