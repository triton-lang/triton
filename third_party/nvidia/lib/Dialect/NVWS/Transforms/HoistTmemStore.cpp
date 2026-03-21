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

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/Utils/InferIntRangeCommon.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <optional>
#include <utility>

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSHOISTTMEMSTORE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"
namespace {

bool underWSLoop(Operation *op) {
  scf::ForOp topLevelFor = op->getParentOfType<scf::ForOp>();
  if (!topLevelFor) {
    return false;
  }

  if (topLevelFor->hasAttr(kWarpSpecializeAttrName)) {
    return true;
  } else {
    while (auto outer = topLevelFor->getParentOfType<scf::ForOp>()) {
      topLevelFor = outer;
      if (outer->hasAttr(kWarpSpecializeAttrName)) {
        return true;
      }
    }
  }

  return false;
}

class FoldTmemStoreIntoAlloc : public OpRewritePattern<ttng::TMEMAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMAllocOp alloc,
                                PatternRewriter &rewriter) const override {
    if (alloc.getSrc() || !underWSLoop(alloc)) {
      return failure();
    }

    for (auto user : alloc->getUsers()) {
      if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
        auto storeSrc = store.getSrc();
        if (auto storeSrcDef = storeSrc.getDefiningOp()) {
          DominanceInfo dom(storeSrcDef);
          if (dom.dominates(storeSrcDef, alloc)) {
            auto newAlloc = ttng::TMEMAllocOp::create(
                rewriter, alloc.getLoc(), alloc.getResultTypes()[0],
                rewriter.getType<AsyncTokenType>(), storeSrc);

            if (auto allocTok = alloc.getToken()) {
              allocTok.replaceAllUsesWith(newAlloc.getToken());
            }
            if (auto storeTok = store.getToken()) {
              storeTok.replaceAllUsesWith(newAlloc.getToken());
            }
            if (hasPartition(store)) {
              // The alloc op can have multiple partitions at this point. But
              // aref-tmem-insert requires a single owner, which should be the
              // partiton that tmem_store belongs to.
              setPartition(newAlloc, getPartitionIds(store));
            }
            rewriter.eraseOp(store);
            rewriter.replaceOp(alloc, newAlloc);
            return success();
          }
        }
      }
    }

    return failure();
  }
};

std::optional<std::pair<scf::ForOp, ttng::MMAv5OpInterface>>
getUniqueUserLoopAndMMA(ttng::TMEMAllocOp tmemAlloc) {
  auto tok = tmemAlloc.getToken();
  if (!tok || !tok.hasOneUse())
    return std::nullopt;
  auto loop = dyn_cast<scf::ForOp>(*tok.getUsers().begin());
  if (!loop)
    return std::nullopt;
  auto loopTok = loop.getBody()->getArgument(
      tok.getUses().begin()->getOperandNumber() - 2);
  if (!loopTok.hasOneUse())
    return std::nullopt;
  auto mma = dyn_cast<ttng::MMAv5OpInterface>(*loopTok.getUsers().begin());
  if (mma)
    return std::make_pair(loop, mma);
  return std::nullopt;
}

// Check if this alloc is used by an MMA op with useD initialized to false
bool canRemoveTmemStore(ttng::TMEMAllocOp tmemAlloc) {
  auto opt = getUniqueUserLoopAndMMA(tmemAlloc);
  if (!opt)
    return false;
  auto [loop, mma] = *opt;
  auto useD = dyn_cast<BlockArgument>(mma.useAccumulator());
  if (!useD)
    return false;
  auto parent = useD.getParentBlock()->getParentOp();
  if (parent != loop)
    return false;
  auto loopInit = loop.getInitArgs()[useD.getArgNumber() - 1];
  auto val = getBoolFromConstant(loopInit);
  return val && val.value() == false;
}

bool canProveExecuteOnce(scf::ForOp forOp) {
  auto getAssumedBound = [&](Value v) -> std::optional<ConstantIntRanges> {
    mlir::ForwardSliceOptions opt;
    SetVector<Operation *> slice;
    (void)getForwardSlice(v, &slice, opt);

    // For simplicity, we only handle an assume op directly operating on v. It's
    // possible to support more general cases, but they require a range
    // analysis.
    for (auto op : slice) {
      if (auto assumeOp = dyn_cast<LLVM::AssumeOp>(op)) {
        auto cond = assumeOp.getCond();
        if (auto cmpOp = cond.getDefiningOp<arith::CmpIOp>();
            cmpOp && (cmpOp.getLhs() == v || cmpOp.getRhs() == v)) {
          if (auto bound = getBoundFromCmpOp(cmpOp, v)) {
            return *bound;
          }
        }
      }
    }
    return std::nullopt;
  };

  auto getConstIntBound = [&](Value v) {
    unsigned bitWidth = ConstantIntRanges::getStorageBitwidth(v.getType());
    if (auto cst = getConstantIntValue(getAsOpFoldResult(v))) {
      APInt apVal = {bitWidth, static_cast<uint64_t>(*cst), /*signed*/ true};
      return mlir::ConstantIntRanges::constant(apVal);
    } else if (auto assumedBound = getAssumedBound(v)) {
      return *assumedBound;
    } else {
      APInt min = APInt::getSignedMinValue(bitWidth);
      APInt max = APInt::getSignedMaxValue(bitWidth);
      return mlir::ConstantIntRanges::range(min, max, true);
    }
  };

  auto lbBound = getConstIntBound(forOp.getLowerBound());
  auto ubBound = getConstIntBound(forOp.getUpperBound());
  return mlir::intrange::evaluatePred(mlir::intrange::CmpPredicate::slt,
                                      lbBound, ubBound)
      .value_or(false);
}

bool hoistTmemAlloc(ttng::TMEMAllocOp allocToHoist) {
  // extra loop nest
  SmallVector<scf::ForOp> loopNest;
  auto currentForOp = allocToHoist->getParentOfType<scf::ForOp>();
  while (currentForOp && !currentForOp->hasAttr(kWarpSpecializeAttrName)) {
    loopNest.push_back(currentForOp);
    currentForOp = currentForOp->getParentOfType<scf::ForOp>();
  }

  if (!currentForOp) {
    return false;
  }

  loopNest.push_back(currentForOp);

  {
    // Check if hoisting across all loop nests is valid. Hoisting is invalid
    // when the inner loop that does MMA executes variable number of times
    // depending on the outer loop variables, and some instances of the inner
    // loops never execute while others do. So we hoist across loop nests only
    // in the following cases:
    // 1. The loop iteration counts for all loops do not depend on their outer
    // loop variables.
    // 2. If there is a loop whose iteration count depends on outer loop
    // varaibles, there is an llvm.intr.assume op from which we can prove that
    // the number of iteration is greater than zero.
    auto opt = getUniqueUserLoopAndMMA(allocToHoist);
    if (!opt) {
      return false;
    }

    SmallVector<scf::ForOp> innerLoopNest{opt->first};
    innerLoopNest.insert(innerLoopNest.begin(), loopNest.begin(),
                         loopNest.end() - 1);

    // Does the expression x depend on y?
    auto dependOn = [](Value x, Value y) {
      mlir::BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      SetVector<Operation *> slice;
      (void)getBackwardSlice(x, &slice, opt);
      for (auto user : y.getUsers()) {
        if (x.getDefiningOp() == user || slice.count(user)) {
          return true;
        }
      }
      return false;
    };

    for (auto [i, innerFor] : llvm::enumerate(innerLoopNest)) {
      for (int j = i; j < loopNest.size(); ++j) {
        auto outerForIter = loopNest[j].getInductionVar();
        if ((dependOn(innerFor.getLowerBound(), outerForIter) ||
             dependOn(innerFor.getUpperBound(), outerForIter)) &&
            !canProveExecuteOnce(innerFor)) {
          // Cannot hoist this tmem alloc across the outer loop loopNest[j]
          return false;
        }
      }
    }
  }

  // hoist to outside tt.warp_specialized loop
  allocToHoist->moveBefore(currentForOp);
  allocToHoist->removeAttr(kPartitionAttrName);

  Value token = allocToHoist.getToken();
  assert(token.hasOneUse());
  auto &tokenUse = *token.getUses().begin();
  auto tokenPos =
      tokenUse.getOperandNumber() - currentForOp.getNumControlOperands();
  auto tokenPartition = getPartitionOutputs(tokenUse.getOwner())[tokenPos];

  // thread token to for-op init/iter args from outer-to inner
  std::reverse(loopNest.begin(), loopNest.end());
  for (auto &forOp : loopNest) {
    OpBuilder b(forOp);
    int nArgs = forOp.getRegionIterArgs().size();
    forOp = addIterArgsToLoop(b, forOp, {token});

    // update partitions for the forOp
    if (forOp->hasAttr(kPartitionOutputsAttrName)) {
      auto partitionOuputs = getPartitionOutputs(forOp);
      partitionOuputs.push_back(tokenPartition);
      setPartitionOutputs(forOp, partitionOuputs);
    } else {
      setPartitionOutputs(forOp, {tokenPartition});
    }
    auto partitions = getPartitionIds(forOp);
    partitions.insert(tokenPartition.begin(), tokenPartition.end());
    setPartition(forOp, partitions);

    token = forOp.getRegionIterArg(nArgs);
  }

  // set inner loop init_args with updated token
  tokenUse.set(token);

  // get last produced token, the one w/o use
  token = tokenUse.getOwner()->getResult(tokenPos);
  while (!token.use_empty()) {
    assert(token.hasOneUse());
    auto tokenUser = *token.getUsers().begin();
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(tokenUser)) {
      token = load.getToken();
    } else if (auto store = dyn_cast<ttng::TMEMStoreOp>(tokenUser)) {
      token = store.getToken();
    } else {
      auto mma = cast<ttng::MMAv5OpInterface>(tokenUser);
      token = mma.getToken();
    }
  }

  // append token to yield, from inner to outer loop
  std::reverse(loopNest.begin(), loopNest.end());
  for (auto forOp : loopNest) {
    appendToForOpYield(forOp, {token});
    setPartition(forOp.getBody()->getTerminator(), getPartitionIds(forOp));
    token = forOp->getResults().back();
  }

  return true;
}

} // namespace

class NVWSHoistTmemStore
    : public impl::NVWSHoistTmemStoreBase<NVWSHoistTmemStore> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    OpPassManager pm;
    mlir::RewritePatternSet patterns(context);
    patterns.add<FoldTmemStoreIntoAlloc>(context);
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsGreedily(m, std::move(patterns))))
      signalPassFailure();

    m.walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kWarpSpecializeAttrName)) {
        SmallVector<ttng::TMEMAllocOp> tmemAllocToHoist;
        loop.walk([&](ttng::TMEMAllocOp tmemAlloc) {
          if (tmemAlloc.getSrc() && canRemoveTmemStore(tmemAlloc)) {
            tmemAllocToHoist.push_back(tmemAlloc);
          }
        });

        for (auto alloc : tmemAllocToHoist) {
          if (!hoistTmemAlloc(alloc)) {
            SetVector<int> mmaPartition;
            mmaPartition.insert(1);
            // tmem store remaining in the outer loop must belong to the MMA
            // partition. This is required by aref-tmem-insert for correctly
            // double buffering this accumulator.
            setPartition(alloc, mmaPartition);
          }
        }
      }
    });
  }
}; // namespace triton

} // namespace triton
} // namespace mlir
