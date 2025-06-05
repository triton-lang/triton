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
#include "nvidia/include/Dialect/NVWS/Transforms/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
using namespace triton::nvws;
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
  int mmaSyncTag = 0;
  funcOp.walk([&](ttng::TMEMLoadOp loadOp) {
    if (!loadOp.getToken())
      return;

    // if load has token, follow token to find token producer
    auto [op, buffer] = getTokenProducerOp(loadOp.getDep());
    if (!op || !isa<ttng::MMAv5OpInterface>(op))
      return;

    // if producer is mma, we need to sync via arefs before load in same group

    // We have quite a few use cases where we don't do much with MMAv5,
    // e.g.
    //   %a = alloc
    //   ..
    //   mma %a  {isAsync}
    //   ..
    //   %b = tmem_load %a

    // we need to do some ad-hoc patching to make such mma wait for %a to
    // be ready before we tmem_load it.

    // If %src comes from %alloc, we need to patch it:
    //    %a = alloc : <128x128>
    //    ...
    //    %tok = mma %a
    //    ...
    //    tma_load %a[%tok]
    //    ...
    // becomes:
    //    %a = alloc : <128x128>
    //    %sync_buf = alloc  <1x1>
    //    %aref = aref_create %a' : <1x1>
    //    ...
    //    %tok = mma %a
    //    ...
    //    aref_put.enter %aref                   (try_wait consumed)
    //    aref_put.exit %aref, producer=[umma]   (tcgen5.commit produced)
    //    aref_get.enter %aref                   (try_wait produced)
    //    tma_load %a[%tok]
    //    aref_get.exit %aref, consumer=[ldtm]   (arrive consumed)
    //    ...

    // arefLower could later optimize away first put.enter and last
    // get.exit

    OpBuilder builder(buffer.getDefiningOp());
    builder.setInsertionPointToStart(&buffer.getDefiningOp()
                                          ->getParentOfType<ttng::WarpGroupOp>()
                                          ->getRegion(0)
                                          .front());

    auto ctx = buffer.getContext();
    auto bufferType = cast<MemDescType>(buffer.getType());
    // Allocate a 1xi32 element for aref buffer in smem tht is used as a
    // synchronization mechanism between MMA ops and loads in the same group.
    // The buffer itself is not used, we just need it to create aref.
    auto CTALayout = ttg::CTALayoutAttr::get(
        /*context=*/ctx, /*CTAsPerCGA=*/{1, 1},
        /*CTASplitNum=*/{1, 1}, /*CTAOrder=*/{1, 0});
    auto encoding =
        ttg::SwizzledSharedEncodingAttr::get(ctx, 1, 1, 1, {0, 1}, CTALayout);
    auto arefBufTy = ttg::MemDescType::get(
        {1, 1}, builder.getI32Type(), encoding,
        SharedMemorySpaceAttr::get(builder.getContext()), true);
    auto dataBufType = getDataMemDescType(arefBufTy, true);
    auto arefTy =
        ttng::ArefType::get(ctx, ttg::TypeArrayAttr::get(ctx, arefBufTy));
    auto arefAlloc =
        builder.create<ttg::LocalAllocOp>(buffer.getLoc(), arefBufTy, Value{});
    arefAlloc->setAttr("mma_sync_buffer", builder.getUnitAttr());
    auto arefOp = builder.create<ttng::ArefCreateOp>(buffer.getLoc(), arefTy,
                                                     arefAlloc.getResult());
    arefOp->setAttr("aref_mma_sync", builder.getUnitAttr());
    builder.setInsertionPoint(loadOp);

    auto tag = std::string("mma_sync_") + std::to_string(mmaSyncTag);

    auto putEnterOp = builder.create<triton::nvidia_gpu::ArefPutEnterOp>(
        loadOp.getLoc(), dataBufType, SmallVector<Type>{}, arefOp,
        mkConstant(builder, loadOp.getLoc(), 0, 32, {}));
    putEnterOp->setAttr("aref_tag", builder.getStringAttr(tag));

    auto putExitOp = builder.create<triton::nvidia_gpu::ArefPutExitOp>(
        loadOp.getLoc(), arefOp,
        mkConstant(builder, loadOp.getLoc(), 0, 32, {}),
        ArrayAttr::get(builder.getContext(),
                       ttng::ArefProducerAttr::get(builder.getContext(),
                                                   ttng::ArefProducer::UMMA)));
    putExitOp->setAttr("aref_tag", builder.getStringAttr(tag));

    auto getEnterOp = builder.create<ttng::ArefGetEnterOp>(
        loadOp.getLoc(), dataBufType, SmallVector<Type>{}, arefOp,
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
