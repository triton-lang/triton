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

#define DEBUG_TYPE "nvws-aref-canonicalize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
using namespace triton::nvws;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

std::optional<bool> getBoolFromConstant(Value cst) {
  auto constantOp = cst.getDefiningOp<arith::ConstantOp>();
  if (!constantOp) {
    return std::nullopt;
  }
  assert(constantOp.getValue());
  if (auto boolAttr = dyn_cast<BoolAttr>(constantOp.getValue())) {
    return boolAttr.getValue();
  }
  return std::nullopt;
}

template <typename OpT> struct HasToken : public OpT {
  using OpT::OpT;

  static bool classof(Operation *op) {
    if (auto tmemOp = dyn_cast<OpT>(op))
      return !!tmemOp.getToken();
    return false;
  }
};
using TMEMTokenStoreOp = HasToken<ttng::TMEMStoreOp>;

class RemoveUnusedTMEMStore : public OpRewritePattern<TMEMTokenStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMTokenStoreOp store,
                                PatternRewriter &rewriter) const override {
    if (!getBoolFromConstant(store.getPred()))
      return failure(); // we've already processed this
    auto tok = store.getToken();
    if (!tok.hasOneUse())
      return failure();
    auto loop = dyn_cast<scf::ForOp>(*tok.getUsers().begin());
    if (!loop)
      return failure();
    auto loopTok = loop.getBody()->getArgument(
        tok.getUses().begin()->getOperandNumber() - 2);
    if (!loopTok.hasOneUse())
      return failure();
    auto mma = dyn_cast<ttng::MMAv5OpInterface>(*loopTok.getUsers().begin());
    if (!mma)
      return failure();
    auto useD = dyn_cast<BlockArgument>(mma.useAccumulator());
    if (!useD)
      return failure();
    auto parent = useD.getParentBlock()->getParentOp();
    if (parent != loop)
      return failure();
    auto loopInit = loop.getInitArgs()[useD.getArgNumber() - 1];
    auto val = getBoolFromConstant(loopInit);
    if (!val)
      return failure();
    if (val.value() == true)
      return failure();
    rewriter.replaceAllUsesWith(store.getToken(), store.getDep());
    rewriter.eraseOp(store);

    // auto loc = store.getLoc();
    // rewriter.setInsertionPoint(store);
    // Value diff = rewriter.create<arith::SubIOp>(loc, loop.getUpperBound(),
    //                                             loop.getLowerBound());
    // Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0,
    // diff.getType()); Value cond = rewriter.create<arith::CmpIOp>(loc,
    // arith::CmpIPredicate::sle,
    //                                             diff, zero);
    // store.getPredMutable().assign(cond);
    return success();
  }
};

class NVWSArefCanonicalize
    : public NVWSArefCanonicalizeBase<NVWSArefCanonicalize> {
  void correctGroups(triton::FuncOp funcOp) {
    // results from forOp can only contain groups where forOp is
    // present
    funcOp.walk([&](scf::ForOp forOp) {
      auto forOpGroups = getGroups(forOp);

      for (auto result : forOp.getResults()) {
        auto idx = result.getResultNumber();
        auto groups = getGroupsIdx(forOp, idx);
        std::set<std::string> resultGroups;
        for (auto group : groups)
          if (forOpGroups.count(group) > 0)
            resultGroups.insert(group);
        setGroupsIdx(forOp, idx, resultGroups);
      }
    });

    // tmem-alloc w/o source is in all groups that use it
    funcOp.walk([&](ttng::TMEMAllocOp alloc) {
      std::set<std::string> consumerGroups;
      if (!alloc.getSrc()) {
        // TMEMalloc w/o source is mutable, and can be used by multiple groups
        // for store so we put in multilpe groups. ArefCopy removal pass
        // will eliminatge TMEMAllocs that are not needed
        for (auto user : alloc.getResult().getUsers())
          for (auto group : getGroups(user))
            consumerGroups.insert(group);
        setGroups(alloc, consumerGroups);
      } else {
        // tmem-alloc w/ source should be in the same group as the source,
        // otherwise we'd pass the operand of tmem-alloc in shared memory buffer
        // between groups then immediate copy it to tmem. putting them in the
        // same group would make the operand pass directly in tmem
        assert(isa<RankedTensorType>(alloc.getSrc().getType()));
        auto srcGroups = getGroups(alloc.getSrc().getDefiningOp());
        auto allocGroups = getGroups(alloc);
        if (srcGroups != allocGroups) {
          setGroups(alloc, srcGroups);
        }
      }
    });

    // fix group for tt.reduce.return
    funcOp.walk([&](triton::ReduceReturnOp reduceReturnOp) {
      auto groups =
          getGroups(reduceReturnOp->getParentOfType<triton::ReduceOp>());
      assert(!groups.empty());
      setGroups(reduceReturnOp, groups);
    });

    // place tmem_store before tcgen5_mma in the same groups a src of tmem_store
    funcOp.walk([&](ttng::TMEMStoreOp store) {
      auto srcGroups = getGroups(store.getSrc().getDefiningOp());
      // we just need one group, pick first one if there are multiple
      auto group = *srcGroups.begin();
      setGroups(store, {group});
    });

    // place tmem_load after tcgen5_mma in the same group as consumer of
    // tmem_load
    funcOp.walk([&](ttng::TMEMLoadOp load) {
      auto loadGroups = getGroups(load);
      loadGroups.clear();
      for (auto user : load.getResult().getUsers()) {
        auto groups = getGroups(user);
        loadGroups.insert(groups.begin(), groups.end());
      }
      setGroups(load, loadGroups);
    });

    // For each loop body, verify that loop results are only annotated with
    // groups that contain their producing operations.
    funcOp.walk([&](scf::ForOp forOp) {
      auto forOpGroups = getGroups(forOp);
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (auto [idx, opnd] : llvm::enumerate(yieldOp->getOperands())) {
        auto groups = getGroups(opnd.getDefiningOp());
        for (auto group : groups)
          if (forOpGroups.count(group) == 0)
            llvm_unreachable("result is annotated with a group that is not "
                             "present in the loop");
        setGroupsIdx(forOp, idx, groups);
      }
    });
  }

  void updateTokenByProducerGroup(triton::FuncOp funcOp) {
    funcOp.walk([&](scf::YieldOp yieldOp) {
      for (auto [idx, opnd] : llvm::enumerate(yieldOp->getOperands())) {
        if (isa<AsyncTokenType>(opnd.getType())) {
          auto op = opnd.getDefiningOp();
          // if op is a load, get the op that created the dep for the load
          if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op))
            op = loadOp.getDep().getDefiningOp();

          std::set<std::string> groups;
          if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
            groups = getGroups(op);
          } else if (auto mmav5 = dyn_cast<ttng::MMAv5OpInterface>(op)) {
            groups = getGroups(op);
          } else if (isa<scf::ForOp, scf::IfOp>(op)) {
            int tokPos = -1;
            for (auto [pos, arg] : llvm::enumerate(op->getResults())) {
              if (arg == opnd) {
                tokPos = pos;
                break;
              }
            }
            assert(tokPos != -1);
            groups = getGroupsIdx(op, tokPos);
          }
          // update token to be the one produced by the store/mma operation
          if (!groups.empty()) {
            assert(groups.size() == 1);
            setGroupsIdx(yieldOp->getParentOp(), idx, groups);
          }
        }
      }
    });
  }

  void assignScalarsToAllGroups(triton::FuncOp funcOp) {
    auto groupMap = collectGroups(funcOp->getParentOfType<ModuleOp>());
    bool stable = false;
    while (!stable) {
      stable = true;

      funcOp.walk([&](Operation *op) {
        if (isa<triton::FuncOp, triton::ReduceOp>(op))
          return;

        if (op->getNumResults() != 1 || op->getNumRegions() != 0)
          return;

        if (op->getParentOp() && isa<triton::ReduceOp>(op->getParentOp()))
          return;

        if (!op->getResult(0).getType().isIntOrIndexOrFloat())
          return;

        auto opGroups = getGroups(op);
        for (auto &use : op->getUses()) {
          auto useGroups = getGroups(use.getOwner());
          if (auto forOp = dyn_cast<scf::ForOp>(use.getOwner())) {
            auto pos = use.getOperandNumber();
            if (pos >= 3) {
              useGroups = getGroupsIdx(forOp, pos);
            }
          } else if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
            auto pos = use.getOperandNumber();
            useGroups = getGroupsIdx(yieldOp->getParentOp(), pos);
          }
          for (auto g : useGroups) {
            if (opGroups.count(g) == 0) {
              stable = false;
              opGroups.insert(g);
            }
          }
        }
        setGroups(op, opGroups);
      });
    };
  }

  void blackwellFASequenceMatmuls(triton::FuncOp funcOp) {
    // sequence first matmuls
    //
    //   %buf1, %tok1 = tmem_alloc
    //   %tok1' = mma %buf1[%tok1'], useD=%false
    //   tmem_load %buf1[%tok1']
    //   %buf2, %tok2 = tmem_alloc
    //   %tok2' = mma %buf2[%tok2'], useD=%false
    //   tmem_load %buf2[%tok2']
    //
    // to
    //
    //   %buf1, %tok1 = tmem_alloc
    //   %tok1' = mma %buf1[%tok1'], useD=%false
    //   %tok2 = tmem_load %buf1[%tok1']
    //   %tok2' = mma %buf1%tok2], useD=%false
    //   tmem_load %buf1[%tok2']
    //

    SmallVector<ttng::TMEMAllocOp> allocs;
    ;
    funcOp.walk([&](ttng::TMEMAllocOp allocOp) {
      auto groups = getGroups(allocOp);
      if (groups.count("nvws.group.mma1") || groups.count("nvws.group.mma2")) {
        if (groups.count("nvws.group.sm1") || groups.count("nvws.group.sm2")) {
          if (!groups.count("nvws.group.correction"))
            allocs.push_back(allocOp);
        }
      }
    });

    if (allocs.size() != 2)
      return;

    auto allocS1 = allocs[0];
    auto allocS2 = allocs[1];
    if (getGroups(allocS1).count("nvws.group.sm2")) {
      std::swap(allocS1, allocS2);
    }

    if (!allocS1.getToken().hasOneUse())
      return;
    if (!allocS2.getToken().hasOneUse())
      return;

    auto forOp1 = cast<scf::ForOp>(*allocS1.getToken().getUsers().begin());
    auto forOp2 = cast<scf::ForOp>(*allocS2.getToken().getUsers().begin());
    if (forOp1 != forOp2)
      return;

    auto operand1 = allocS1.getToken().getUses().begin()->getOperandNumber();
    auto operand2 = allocS2.getToken().getUses().begin()->getOperandNumber();
    assert(operand1 >= 3);
    assert(operand2 >= 3);

    auto blockedArg1 = forOp1.getRegionIterArg(operand1 - 3);
    auto blockedArg2 = forOp1.getRegionIterArg(operand2 - 3);
    assert(isa<AsyncTokenType>(blockedArg1.getType()));
    assert(isa<AsyncTokenType>(blockedArg2.getType()));
    assert(blockedArg1.hasOneUse());
    assert(blockedArg2.hasOneUse());

    auto mma1 = cast<ttng::TCGen5MMAOp>(*blockedArg1.getUsers().begin());
    auto mma2 = cast<ttng::TCGen5MMAOp>(*blockedArg2.getUsers().begin());

    auto load1 = cast<ttng::TMEMLoadOp>(*mma1.getToken().getUsers().begin());
    auto load2 = cast<ttng::TMEMLoadOp>(*mma2.getToken().getUsers().begin());

    OpBuilder b(mma1);
    b.setInsertionPoint(mma1);
    auto allocOp = cast<ttng::TMEMAllocOp>(b.clone(*allocS1));

    mma1.getDMutable().assign(allocOp.getResult());
    mma1.getAccDepMutable().assign(allocOp.getToken());
    load1.getSrcMutable().assign(mma1.getD());

    if (!load1->isBeforeInBlock(mma2))
      return;

    mma2.getDMutable().assign(load1.getSrc());
    mma2.getAccDepMutable().assign(load1.getToken());
    load2.getSrcMutable().assign(mma2.getD());

    auto tok1 = allocS1.getToken();
    auto tok2 = allocS2.getToken();

    // remove tokens from the loop
    int pos1 = -1, pos2 = -1;
    SmallVector<Value> operands;
    for (auto [idx, arg] : llvm::enumerate(forOp1.getInitArgs())) {
      if (arg != tok1 && arg != tok2) {
        operands.push_back(arg);
      } else if (arg == tok1) {
        assert(pos1 == -1);
        pos1 = idx;
      } else if (arg == tok2) {
        assert(pos2 == -1);
        pos2 = idx;
      }
    }
    assert(operands.size() == forOp1.getInitArgs().size() - 2);
    b.setInsertionPoint(forOp1);
    scf::ForOp newLoop = b.create<scf::ForOp>(
        forOp1.getLoc(), forOp1.getLowerBound(), forOp1.getUpperBound(),
        forOp1.getStep(), operands);
    newLoop->setAttrs(forOp1->getAttrs());
    newLoop.getBody()->erase();
    newLoop.getRegion().getBlocks().splice(
        newLoop.getRegion().getBlocks().begin(),
        forOp1.getRegion().getBlocks());

    for (int oldIdx = 0, newIdx = 0; oldIdx < forOp1.getResults().size();
         ++oldIdx) {
      if (oldIdx == pos1 || oldIdx == pos2)
        continue;
      forOp1.getResult(oldIdx).replaceAllUsesWith(newLoop.getResult(newIdx++));
      setGroupsIdx(newLoop, newIdx, getGroupsIdx(forOp1, oldIdx));
    }

    auto body = newLoop.getBody();
    auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
    SmallVector<Value> yieldOperands;
    for (auto [pos, operand] : llvm::enumerate(yieldOp.getOperands())) {
      if (pos != pos1 && pos != pos2) {
        yieldOperands.push_back(operand);
      }
    }

    if (pos1 > pos2) {
      std::swap(pos1, pos2);
    }
    body->eraseArgument(pos2 + 1);
    body->eraseArgument(pos1 + 1);

    b.setInsertionPoint(yieldOp);
    b.create<scf::YieldOp>(yieldOp.getLoc(), yieldOperands);
    yieldOp.erase();

    forOp1.erase();
    allocS1.erase();
    allocS2.erase();
  }

  void fixUpForIfOpGroups(triton::FuncOp funcOp) {
  funcOp.walk([&](scf::IfOp ifOp) {
    auto groups = getGroups(ifOp);
    for (auto result : ifOp.getResults()) {
      auto g = getGroups(result);
      groups.insert(g.begin(), g.end());
    }
    setGroups(ifOp, groups);
  });
  funcOp.walk([&](scf::ForOp forOp) {
    auto groups = getGroups(forOp);
    for (auto result : forOp.getResults()) {
      auto g = getGroups(result);
      groups.insert(g.begin(), g.end());
    }
    setGroups(forOp, groups);
  });
}

public:
  void runOnFunc(triton::FuncOp funcOp) {
    mlir::RewritePatternSet patterns(funcOp.getContext());
    patterns.add<RemoveUnusedTMEMStore>(funcOp.getContext());
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      llvm_unreachable("Failed to remove tmem_store");

    blackwellFASequenceMatmuls(funcOp);

    correctGroups(funcOp);

    updateTokenByProducerGroup(funcOp);

    assignScalarsToAllGroups(funcOp);

    fixUpForIfOpGroups(funcOp);
  }

  void runOnOperation() override {
    auto mod = getOperation();
    mod.walk([&](triton::FuncOp funcOp) { runOnFunc(funcOp); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createNVWSArefCanonicalizePass() {
  return std::make_unique<NVWSArefCanonicalize>();
}
