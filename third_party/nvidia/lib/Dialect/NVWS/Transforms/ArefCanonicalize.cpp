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

#define DEBUG_TYPE "nvws-aref-canonicalize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace triton::gpu;
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
      auto producerGroups = getGroups(forOp);
      for (auto [idx, arg] : llvm::enumerate(forOp.getInitArgs())) {
        auto groups = getGroupsIdx(forOp, idx);
        std::set<std::string> correctedGroups;
        for (auto group : groups)
          if (producerGroups.count(group) > 0)
            correctedGroups.insert(group);
        setGroupsIdx(forOp, idx, correctedGroups);
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
      }
    });

    // fix group for tt.reduce.return
    funcOp.walk([&](triton::ReduceReturnOp reduceReturnOp) {
      auto groups = getGroups(reduceReturnOp->getParentOfType<triton::ReduceOp>());
      assert(!groups.empty());
      setGroups(reduceReturnOp, groups);
    });
  }

public:
  void runOnFunc(triton::FuncOp funcOp) {
    mlir::RewritePatternSet patterns(funcOp.getContext());
    patterns.add<RemoveUnusedTMEMStore>(funcOp.getContext());
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      llvm_unreachable("Failed to remove tmem_store");

    correctGroups(funcOp);
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
