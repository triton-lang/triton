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
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/WSUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include <memory>

// #define GEN_PASS_CLASSES
// #include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define GEN_PASS_CLASSES
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"

#define DEBUG_TYPE "nvws-copy-lowering"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
namespace tt = triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

LogicalResult lowerArefCopyOp(ttng::ArefCopyOp op, OpBuilder &rewriter) {
  auto src = op.getSrc();
  auto dst = op.getDst();

  if (auto srcTok = op.getSrcDep()) {
    // If copy elimination optimization does not succeed to eliminate copy ops,
    // in order to functionally support workloads we need to lower aref copies
    // to ops that Triton can compile.
    //
    // Specifically, lower
    //    %tok = arefCopy %src[%srcTok], %dst[%dstTok]
    // to
    //    %val = tmem_load %src[%srcTok]
    //    %tok2 = tmem_store %val, %dst[%dstTok]
    // replace all uses of  %tok with %tok2, if present

    auto memDescType = cast<MemDescType>(src.getType());
    auto shape = memDescType.getShape();
    int mDim = getShapePerCTA(src.getType())[0];
    int nDim = getShapePerCTA(src.getType())[1];
    int numWarps = ttg::lookupNumWarps(op);

    auto ctaLayout = CTALayoutAttr::get(
        /*context=*/op.getContext(), /*CTAsPerCGA=*/{1, 1},
        /*CTASplitNum=*/{1, 1}, /*CTAOrder=*/{0, 1});
    int threadsPerWarp = ttg::lookupThreadsPerWarp(rewriter);
    auto encoding = getDefaultBlockedEncoding(op.getContext(), shape, numWarps,
                                              threadsPerWarp, 1);
    RankedTensorType tensorType = RankedTensorType::get(
        memDescType.getShape(), memDescType.getElementType(), encoding);

    Attribute newDistributedEncoding =
        ttng::getTmemCompatibleLayout(mDim, nDim, tensorType, numWarps);
    auto loadType = RankedTensorType::get(memDescType.getShape(),
                                          memDescType.getElementType(),
                                          newDistributedEncoding);

    auto load =
        rewriter.create<ttng::TMEMLoadOp>(op.getLoc(), loadType, src, srcTok);
    Type tokTy = op.getToken() ? AsyncTokenType::get(op.getContext()) : Type();
    auto vTrue = rewriter.create<arith::ConstantIntOp>(load.getLoc(), 1, 1);
    auto store = rewriter.create<ttng::TMEMStoreOp>(
        op.getLoc(), tokTy, dst, op.getDstDep(), load.getResult(), vTrue);

    if (auto tok = store.getToken())
      op.getToken().replaceAllUsesWith(tok);

    auto enterOp =
        tokTy ? op.getSrc().getDefiningOp<ttng::ArefEnterOpInterface>()
              : op.getDst().getDefiningOp<ttng::ArefEnterOpInterface>();
    auto isEnterPut = isa<ttng::ArefPutEnterOp>(enterOp);
    auto arefTag = enterOp->getAttrOfType<StringAttr>("aref_tag").str();
    ttng::ArefExitOpInterface exitOp;
    for (auto user : enterOp.getAref().getUsers()) {
      if (auto op = dyn_cast<ttng::ArefExitOpInterface>(user)) {
        auto isExitPut = isa<ttng::ArefPutExitOp>(op);
        if (isEnterPut == isExitPut &&
            op->getAttrOfType<StringAttr>("aref_tag").str() == arefTag) {
          exitOp = op;
          break;
        }
      }
    }

    Attribute attribute;
    if (tokTy) {
      attribute = ttng::ArefConsumerAttr::get(rewriter.getContext(),
                                              ttng::ArefConsumer::LDTM);
    } else {
      attribute = ttng::ArefProducerAttr::get(rewriter.getContext(),
                                              ttng::ArefProducer::STTM);
    }
    exitOp.setAssociatedOpAttrs(rewriter.getArrayAttr(attribute));

    return success();
  }
  llvm_unreachable("unsupported w/o token");
}

// Insert fences to ensure that STS are visible to LDS in other groups.
void insertFences(tt::FuncOp funcOp) {
  funcOp.walk([&](ttng::ArefPutExitOp putExitOp) {
    OpBuilder builder(putExitOp);
    int isProducerSTS = false;
    auto producers = putExitOp.getProducersAttr();
    for (auto prod : putExitOp.getProducers()) {
      auto kind = cast<ttng::ArefProducerAttr>(prod).getValue();
      if (kind == ttng::ArefProducer::STS) {
        isProducerSTS = true;
        break;
      }
    }
    if (isProducerSTS) {
      // insert async fence so that STS will be visible to LDS in other groups
      auto fence = builder.create<triton::nvidia_gpu::FenceAsyncSharedOp>(
          putExitOp.getLoc(), false);
      setGroups(fence, getGroups(putExitOp));
    }
    return WalkResult::advance();
  });
}

LogicalResult lowerArefCopyOp(tt::FuncOp funcOp) {
  SmallVector<ttng::ArefCopyOp> copyOps;
  funcOp.walk([&](ttng::ArefCopyOp op) { copyOps.push_back(op); });
  for (auto op : copyOps) {
    OpBuilder builder(op);
    if (lowerArefCopyOp(op, builder).failed())
      return failure();
    op->erase();
  }
  return success();
}

class NVWSArefCopyLowering
    : public NVWSArefCopyLoweringBase<NVWSArefCopyLowering> {
public:
  void runOnFunc(tt::FuncOp funcOp) {
    if (lowerArefCopyOp(funcOp).failed())
      signalPassFailure();

    insertFences(funcOp);
  }
  void runOnOperation() override {
    auto mod = getOperation();
    mod.walk([&](tt::FuncOp funcOp) { runOnFunc(funcOp); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createNVWSArefCopyLoweringPass() {
  return std::make_unique<NVWSArefCopyLowering>();
}
