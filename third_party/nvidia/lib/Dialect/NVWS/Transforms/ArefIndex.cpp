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

struct LastIndexValue {
  Value putEnter = {};
  Value getEnter = {};
  Value putExit = {};
  Value getExit = {};
};
using ArefLastIndexMap = llvm::MapVector<Value /*aref*/, LastIndexValue>;

struct ArefFirstUse {
  ttng::ArefPutEnterOp putEnterOp = {};
  ttng::ArefGetEnterOp getEnterOp = {};
  ttng::ArefPutExitOp putExitOp = {};
  ttng::ArefGetExitOp getExitOp = {};
};

using ArefFirstUseMap = llvm::MapVector<Value /*arefs*/, ArefFirstUse>;
Value getIndex(ttng::ArefEnterOpInterface enterOp) {
  return isConstant(enterOp.getIndex(), 0) ? Value{} : enterOp.getIndex();
};
Value getIndex(ttng::ArefExitOpInterface exitOp) {
  return isConstant(exitOp.getIndex(), 0) ? Value{} : exitOp.getIndex();
};

void setIndex(ttng::ArefEnterOpInterface enterOp, Value index) {
  assert(isConstant(enterOp.getIndex(), 0));
  enterOp.setIndex(index);
};
void setIndex(ttng::ArefExitOpInterface exitOp, Value index) {
  assert(isConstant(exitOp.getIndex(), 0));
  exitOp.setIndex(index);
};

ArefFirstUseMap AnalyzeArefUseInBlock(Block *block, ArefFirstUseMap arefUse) {
  for (auto &op : *block) {
    if (auto enterOp = dyn_cast<ttng::ArefEnterOpInterface>(op)) {
      if (!getIndex(enterOp)) {
        auto aref = enterOp.getAref();
        if (isa<ttng::ArefPutEnterOp>(enterOp)) {
          if (!arefUse[aref].putEnterOp)
            arefUse[aref].putEnterOp = cast<ttng::ArefPutEnterOp>(enterOp);
          ;
        } else {
          if (!arefUse[aref].getEnterOp)
            arefUse[aref].getEnterOp = cast<ttng::ArefGetEnterOp>(enterOp);
        }
      }
    } else if (auto exitOp = dyn_cast<ttng::ArefExitOpInterface>(op)) {
      if (!getIndex(exitOp)) {
        auto aref = exitOp.getAref();
        if (isa<ttng::ArefPutExitOp>(exitOp)) {
          if (!arefUse[aref].putExitOp)
            arefUse[aref].putExitOp = cast<ttng::ArefPutExitOp>(exitOp);
        } else {
          if (!arefUse[aref].getExitOp)
            arefUse[aref].getExitOp = cast<ttng::ArefGetExitOp>(exitOp);
        }
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // recursive visit for-op body to gather nested uses of arefs in put/get
      auto &region = forOp.getRegion();
      auto block = &region.getBlocks().front();

      auto arefUseBlock = AnalyzeArefUseInBlock(block, ArefFirstUseMap{});
      for (auto [aref, useMap] : arefUseBlock) {
        if (!arefUse[aref].putEnterOp)
          arefUse[aref].putEnterOp = useMap.putEnterOp;
        if (!arefUse[aref].getEnterOp)
          arefUse[aref].getEnterOp = useMap.getEnterOp;
        if (!arefUse[aref].putExitOp)
          arefUse[aref].putExitOp = useMap.putExitOp;
        if (!arefUse[aref].getExitOp)
          arefUse[aref].getExitOp = useMap.getExitOp;
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      // recursive visit if-op then/else body to gather nested uses of arefs
      auto &thenRegion = ifOp.getThenRegion();
      auto thenBlock = &thenRegion.getBlocks().front();
      auto thenArefUse = AnalyzeArefUseInBlock(thenBlock, ArefFirstUseMap{});
      for (auto [aref, useMap] : thenArefUse) {
        if (!arefUse[aref].putEnterOp)
          arefUse[aref].putEnterOp = useMap.putEnterOp;
        if (!arefUse[aref].getEnterOp)
          arefUse[aref].getEnterOp = useMap.getEnterOp;
        if (!arefUse[aref].putExitOp)
          arefUse[aref].putExitOp = useMap.putExitOp;
        if (!arefUse[aref].getExitOp)
          arefUse[aref].getExitOp = useMap.getExitOp;
      }
      if (ifOp.elseBlock()) {
        auto &elseRegion = ifOp.getElseRegion();
        auto elseBlock = &elseRegion.getBlocks().front();
        auto elseArefUse = AnalyzeArefUseInBlock(elseBlock, ArefFirstUseMap{});
        for (auto [aref, useMap] : elseArefUse) {
          if (!arefUse[aref].putEnterOp)
            arefUse[aref].putEnterOp = useMap.putEnterOp;
          if (!arefUse[aref].getEnterOp)
            arefUse[aref].getEnterOp = useMap.getEnterOp;
          if (!arefUse[aref].putExitOp)
            arefUse[aref].putExitOp = useMap.putExitOp;
          if (!arefUse[aref].getExitOp)
            arefUse[aref].getExitOp = useMap.getExitOp;
        }
      }
    }
  }
  return arefUse;
}

ArefLastIndexMap arefIndexAssignmentInBlock(Block *inpBlock,
                                            ArefLastIndexMap arefLastIndex,
                                            OpBuilder &builder);

void assignInForOp(scf::ForOp forOp, ArefLastIndexMap &arefLastIndex,
                   OpBuilder &builder, SmallVector<Operation *> &staleOps) {
  auto arefUseInBlock = AnalyzeArefUseInBlock(
      &forOp.getRegion().getBlocks().front(), ArefFirstUseMap{});
  if (arefUseInBlock.empty())
    return;

  // there are arefs used in the loop-body, recrusively assigned
  // index there

  SmallVector<Value> initArgs(forOp.getInitArgs().begin(),
                              forOp.getInitArgs().end());
  // add initial indexes to the loop
  SmallVector<Value *> arefLastIndexArgs;
  for (auto [aref, useMap] : arefUseInBlock) {
    if (useMap.putEnterOp) {
      initArgs.push_back(arefLastIndex[aref].putEnter);
      arefLastIndexArgs.push_back(&arefLastIndex[aref].putEnter);
    }
    if (useMap.getEnterOp) {
      initArgs.push_back(arefLastIndex[aref].getEnter);
      arefLastIndexArgs.push_back(&arefLastIndex[aref].getEnter);
    }
    if (useMap.putExitOp) {
      initArgs.push_back(arefLastIndex[aref].putExit);
      arefLastIndexArgs.push_back(&arefLastIndex[aref].putExit);
    }
    if (useMap.getExitOp) {
      initArgs.push_back(arefLastIndex[aref].getExit);
      arefLastIndexArgs.push_back(&arefLastIndex[aref].getExit);
    }
  }

  // create new forOp
  builder.setInsertionPoint(forOp);
  scf::ForOp newForOp = builder.create<mlir::scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), initArgs);

  // update uses
  for (auto [oldArg, newArg] :
       llvm::zip(forOp.getResults(), newForOp.getResults())) {
    oldArg.replaceAllUsesWith(newArg);
  }
  // move loop body
  newForOp.getRegion().takeBody(forOp.getRegion());

  // update the last index value to use inside the loop body
  int nOld = forOp.getResults().size();
  int nNew = newForOp.getResults().size();
  int n = nNew - nOld;
  assert(n == arefLastIndexArgs.size());
  for (int i = 0; i < n; ++i) {
    auto val =
        newForOp.getBody()->addArgument(newForOp.getResult(nOld + i).getType(),
                                        newForOp.getResult(nOld + i).getLoc());
    *arefLastIndexArgs[i] = val;
  }

  // assign indexes in the loop body
  auto arefLastIndexInBlock = arefIndexAssignmentInBlock(
      &newForOp.getRegion().getBlocks().front(), arefLastIndex, builder);

  // update yieldOp to return new indexes
  auto yieldOp = mlir::cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  SmallVector<Value> newYieldVals(yieldOp.getOperands().begin(),
                                  yieldOp.getOperands().end());
  for (auto [aref, useMap] : arefUseInBlock) {
    if (useMap.putEnterOp)
      newYieldVals.push_back(arefLastIndexInBlock[aref].putEnter);
    if (useMap.getEnterOp)
      newYieldVals.push_back(arefLastIndexInBlock[aref].getEnter);
    if (useMap.putExitOp)
      newYieldVals.push_back(arefLastIndexInBlock[aref].putExit);
    if (useMap.getExitOp)
      newYieldVals.push_back(arefLastIndexInBlock[aref].getExit);
  }
  builder.setInsertionPoint(yieldOp);
  builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldVals);
  yieldOp.erase();

  // finaly, update indexes with results from newForOp
  for (int i = 0; i < n; ++i)
    *arefLastIndexArgs[i] = newForOp.getResults()[nOld + i];

  staleOps.push_back(forOp);
};

void assignInIfOp(scf::IfOp ifOp, ArefLastIndexMap &arefLastIndex,
                  OpBuilder &builder, SmallVector<Operation *> &staleOps) {
  // do the same with if-then/else blocks
  auto arefUseInThenBlock =
      AnalyzeArefUseInBlock(ifOp.thenBlock(), ArefFirstUseMap{});
  if (arefUseInThenBlock.empty())
    return;

  bool hasElseBlock = ifOp.elseBlock();

  auto arefUseInElseBlock =
      ifOp.elseBlock()
          ? AnalyzeArefUseInBlock(ifOp.elseBlock(), ArefFirstUseMap{})
          : ArefFirstUseMap{};
  ArefFirstUseMap arefUseInIfOp = arefUseInThenBlock;

  for (auto [aref, useMap] : arefUseInElseBlock) {
    if (!arefUseInIfOp[aref].putEnterOp)
      arefUseInIfOp[aref].putEnterOp = useMap.putEnterOp;
    if (!arefUseInIfOp[aref].getEnterOp)
      arefUseInIfOp[aref].getEnterOp = useMap.getEnterOp;
    if (!arefUseInIfOp[aref].putExitOp)
      arefUseInIfOp[aref].putExitOp = useMap.putExitOp;
    if (!arefUseInIfOp[aref].getExitOp)
      arefUseInIfOp[aref].getExitOp = useMap.getExitOp;
  }

  SmallVector<Type, 4> newIfResultTypes(ifOp.getResultTypes().begin(),
                                        ifOp.getResultTypes().end());
  SmallVector<Value *> arefLastIndexArgs;
  for (auto [aref, useMap] : arefUseInIfOp) {
    if (useMap.putEnterOp) {
      arefLastIndexArgs.push_back(&arefLastIndex[aref].putEnter);
      newIfResultTypes.push_back(arefLastIndex[aref].putEnter.getType());
    }
    if (useMap.getEnterOp) {
      arefLastIndexArgs.push_back(&arefLastIndex[aref].getEnter);
      newIfResultTypes.push_back(arefLastIndex[aref].getEnter.getType());
    }
    if (useMap.putExitOp) {
      arefLastIndexArgs.push_back(&arefLastIndex[aref].putExit);
      newIfResultTypes.push_back(arefLastIndex[aref].putExit.getType());
    }
    if (useMap.getExitOp) {
      arefLastIndexArgs.push_back(&arefLastIndex[aref].getExit);
      newIfResultTypes.push_back(arefLastIndex[aref].getExit.getType());
    }
  }

  builder.setInsertionPoint(ifOp);
  auto newIfOp = builder.create<scf::IfOp>(ifOp.getLoc(), newIfResultTypes,
                                           ifOp.getCondition(),
                                           /*withElseRegion=*/true);

  int nOld = ifOp.getResults().size();
  int nNew = newIfOp.getResults().size();
  int n = nNew - nOld;
  assert(n == arefLastIndexArgs.size());

  for (auto [oldArg, newArg] :
       llvm::zip(ifOp.getResults(), newIfOp.getResults())) {
    oldArg.replaceAllUsesWith(newArg);
  }
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
  if (ifOp.elseBlock())
    newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());

  auto arefLastIndexInThenBlock =
      arefIndexAssignmentInBlock(newIfOp.thenBlock(), arefLastIndex, builder);
  auto arefLastIndexInElseBlock =
      ifOp.elseBlock() ? arefIndexAssignmentInBlock(newIfOp.elseBlock(),
                                                    arefLastIndex, builder)
                       : arefLastIndex;

  // update yieldOp to return new indexes
  {
    auto thenYieldOp =
        mlir::cast<scf::YieldOp>(newIfOp.thenBlock()->getTerminator());
    SmallVector<Value> newThenYieldVals(thenYieldOp.getOperands().begin(),
                                        thenYieldOp.getOperands().end());
    for (auto [aref, useMap] : arefUseInIfOp) {
      if (useMap.putEnterOp)
        newThenYieldVals.push_back(arefLastIndexInThenBlock[aref].putEnter);
      if (useMap.getEnterOp)
        newThenYieldVals.push_back(arefLastIndexInThenBlock[aref].getEnter);
      if (useMap.putExitOp)
        newThenYieldVals.push_back(arefLastIndexInThenBlock[aref].putExit);
      if (useMap.getExitOp)
        newThenYieldVals.push_back(arefLastIndexInThenBlock[aref].getExit);
    }
    builder.setInsertionPoint(thenYieldOp);
    builder.create<scf::YieldOp>(thenYieldOp.getLoc(), newThenYieldVals);
    thenYieldOp.erase();
  }
  if (hasElseBlock) {
    auto elseYieldOp =
        mlir::cast<scf::YieldOp>(newIfOp.elseBlock()->getTerminator());
    SmallVector<Value> newElseYieldVals(elseYieldOp.getOperands().begin(),
                                        elseYieldOp.getOperands().end());
    for (auto [aref, useMap] : arefUseInIfOp) {
      if (useMap.putEnterOp) {
        newElseYieldVals.push_back(arefLastIndexInElseBlock[aref].putEnter);
      }
      if (useMap.getEnterOp)
        newElseYieldVals.push_back(arefLastIndexInElseBlock[aref].getEnter);
      if (useMap.putExitOp)
        newElseYieldVals.push_back(arefLastIndexInElseBlock[aref].putExit);
      if (useMap.getExitOp)
        newElseYieldVals.push_back(arefLastIndexInElseBlock[aref].getExit);
    }
    builder.setInsertionPoint(elseYieldOp);
    builder.create<scf::YieldOp>(elseYieldOp.getLoc(), newElseYieldVals);
    elseYieldOp.erase();
  } else {
    // if there is no elseBlock in ifOp, we still need to create one
    // because we return updated index values
    SmallVector<Value> newElseYieldVals;
    for (auto [aref, useMap] : arefUseInIfOp) {
      if (useMap.putEnterOp) {
        newElseYieldVals.push_back(arefLastIndexInElseBlock[aref].putEnter);
      }
      if (useMap.getEnterOp)
        newElseYieldVals.push_back(arefLastIndexInElseBlock[aref].getEnter);
      if (useMap.putExitOp)
        newElseYieldVals.push_back(arefLastIndexInElseBlock[aref].putExit);
      if (useMap.getExitOp)
        newElseYieldVals.push_back(arefLastIndexInElseBlock[aref].getExit);
    }
    // sanity check, if we have nothing to return, we should rewrite
    // ifOp
    assert(!newElseYieldVals.empty());
    OpBuilder builder = OpBuilder::atBlockEnd(newIfOp.elseBlock());
    builder.create<scf::YieldOp>(newIfOp.getLoc(), newElseYieldVals);
  }

  // update arefCounter vlaues with results from newForOp
  for (int i = 0; i < n; ++i)
    *arefLastIndexArgs[i] = newIfOp.getResults()[nOld + i];

  staleOps.push_back(ifOp);
};

ArefLastIndexMap arefIndexAssignmentInBlock(Block *inpBlock,
                                            ArefLastIndexMap arefLastIndex,
                                            OpBuilder &builder) {
  auto getNextIndex = [&](Operation *op, Value index) {
    auto loc = op->getLoc();
    auto nextIndex = builder.create<arith::AddIOp>(
        loc, index, builder.create<arith::ConstantIntOp>(loc, 1, 32));
    nextIndex->setAttr("next_aref_index", builder.getUnitAttr());
    return nextIndex;
  };
  SmallVector<Operation *> staleOps;

  for (auto &op : *inpBlock) {
    if (auto enterOp = dyn_cast<ttng::ArefEnterOpInterface>(op)) {
      if (!getIndex(enterOp)) {
        assert(arefLastIndex.contains(enterOp.getAref()));
        auto &index = isa<ttng::ArefPutEnterOp>(enterOp)
                          ? arefLastIndex[enterOp.getAref()].putEnter
                          : arefLastIndex[enterOp.getAref()].getEnter;

        setIndex(enterOp, index);
        builder.setInsertionPointAfter(enterOp);
        index = getNextIndex(enterOp, index);
      }
    } else if (auto exitOp = dyn_cast<ttng::ArefExitOpInterface>(op)) {
      if (!getIndex(exitOp)) {
        assert(arefLastIndex.contains(exitOp.getAref()));
        auto &index = isa<ttng::ArefPutExitOp>(exitOp)
                          ? arefLastIndex[exitOp.getAref()].putExit
                          : arefLastIndex[exitOp.getAref()].getExit;
        setIndex(exitOp, index);
        builder.setInsertionPointAfter(exitOp);
        index = getNextIndex(exitOp, index);
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      assignInForOp(forOp, arefLastIndex, builder, staleOps);
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      assignInIfOp(ifOp, arefLastIndex, builder, staleOps);
    }
  }

  for (auto op : staleOps)
    op->erase();
  return arefLastIndex;
}

LogicalResult arefIndexAssignment(triton::FuncOp funcOp) {
  OpBuilder builder(funcOp);
  SmallVector<ttng::WarpGroupOp> wgOps;
  funcOp.walk([&](ttng::WarpGroupOp wgOp) { wgOps.push_back(wgOp); });

  // Verify that if a put/get already has an index assigned, i.e. not constant
  // 0, then all put/get operations associated with the same aref must have a
  // index assigned. Mixing and matching is not permitted.
  for (auto arefOp : funcOp.getOps<ttng::ArefCreateOp>()) {
    bool putHasIndex = false, getHasIndex = false;
    for (auto uses : arefOp->getUsers()) {
      if (auto putEnterOp = dyn_cast<ttng::ArefPutEnterOp>(uses))
        putHasIndex |= (bool)getIndex(putEnterOp);
      else if (auto getEnterOp = dyn_cast<ttng::ArefGetEnterOp>(uses))
        getHasIndex |= (bool)getIndex(getEnterOp);
    }

    for (auto uses : arefOp->getUsers()) {
      if (auto putEnterOp = dyn_cast<ttng::ArefPutEnterOp>(uses)) {
        assert(putHasIndex == (bool)getIndex(putEnterOp));
      } else if (auto getEnterOp = dyn_cast<ttng::ArefGetEnterOp>(uses)) {
        assert(getHasIndex == (bool)getIndex(getEnterOp));
      }
    }

    // If index needs to be assigned, verify that all puts/get are in the
    // same warp-group. We do not currently support if some put/get in one
    // warp-group and others are in some other.
    ttng::WarpGroupOp wgPut, wgGet;
    for (auto uses : arefOp->getUsers()) {
      if (auto putEnterOp = dyn_cast<ttng::ArefPutEnterOp>(uses)) {
        if (!getIndex(putEnterOp)) {
          if (wgPut)
            assert(wgPut == putEnterOp->getParentOfType<ttng::WarpGroupOp>());
          else
            wgPut = putEnterOp->getParentOfType<ttng::WarpGroupOp>();
        }
      } else if (auto getEnterOp = dyn_cast<ttng::ArefGetEnterOp>(uses)) {
        if (!getIndex(getEnterOp)) {
          if (wgGet)
            assert(wgGet == getEnterOp->getParentOfType<ttng::WarpGroupOp>());
          else
            wgGet = getEnterOp->getParentOfType<ttng::WarpGroupOp>();
        }
      }
    }
  }

  // gather all arefUses
  ArefFirstUseMap arefUse;
  for (auto wgOp : wgOps) {
    auto block = &wgOp.getRegions().front()->getBlocks().front();
    auto arefUseBlock = AnalyzeArefUseInBlock(block, ArefFirstUseMap{});
    for (auto [aref, useMap] : arefUseBlock) {
      if (!arefUse[aref].putEnterOp)
        arefUse[aref].putEnterOp = useMap.putEnterOp;
      if (!arefUse[aref].getEnterOp)
        arefUse[aref].getEnterOp = useMap.getEnterOp;
      if (!arefUse[aref].putExitOp)
        arefUse[aref].putExitOp = useMap.putExitOp;
      if (!arefUse[aref].getExitOp)
        arefUse[aref].getExitOp = useMap.getExitOp;
    }
  }

  // initialize put/get indexes
  ArefLastIndexMap arefLastIndex;
  for (auto [aref, useMap] : arefUse) {
    builder.setInsertionPointAfter(aref.getDefiningOp());
    arefLastIndex[aref].putEnter =
        builder.create<arith::ConstantIntOp>(aref.getLoc(), 0, 32);
    arefLastIndex[aref].putExit =
        builder.create<arith::ConstantIntOp>(aref.getLoc(), 0, 32);
    arefLastIndex[aref].getEnter =
        builder.create<arith::ConstantIntOp>(aref.getLoc(), 0, 32);
    arefLastIndex[aref].getExit =
        builder.create<arith::ConstantIntOp>(aref.getLoc(), 0, 32);
  }

  for (auto wgOp : wgOps) {
    auto block = &wgOp.getRegions().front()->getBlocks().front();
    arefIndexAssignmentInBlock(block, arefLastIndex, builder);
  }
  return success();
}

class NVWSArefIndex : public NVWSArefIndexBase<NVWSArefIndex> {

public:
  void runOnFunc(triton::FuncOp funcOp) {
    if (arefIndexAssignment(funcOp).failed())
      signalPassFailure();
  }

  void runOnOperation() override {
    auto mod = getOperation();
    mod.walk([&](triton::FuncOp funcOp) { runOnFunc(funcOp); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createNVWSArefIndexPass() {
  return std::make_unique<NVWSArefIndex>();
}
