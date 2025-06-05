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

    // place tmem_store before tcgen5_mma in the same groups a src of tmem_store
    funcOp.walk([&](ttng::TMEMStoreOp store) {
      auto srcGroups = getGroups(store.getSrc().getDefiningOp());
      // we just need one group, pick first one if there are multiple
      auto group = *srcGroups.begin();
      setGroups(store, {group});
    });
  }

  void updateTokens(triton::FuncOp funcOp) {
    // This function patches tokens as preparation for aref insertion.
    // The aref insertion pass makes local decisions by examining token uses -
    // if a token is used in a different group, it inserts an aref. However,
    // token creation alone doesn't necessarily indicate memory updates. The
    // aref pass intentionally avoids analyzing op semantics and operands, and
    // only focuses on immediate uses to keep the logic simple and composable.
    // This local decision-making could generate unnecessary copies that would
    // be difficult to optimize away, since copy elimination in some cases would
    // require inter-group analysis. Instead, we canonicalize the input IR here
    // to prevent unnecessary copies from being inserted in the first place.

    // Consider the following input:
    //   %ret = for  .. iter_args(%tok = %tok0) {
    //      %tok1 = update %buf[%tok]   @gr1
    //      %tok2 = load %buf[%tok1]     @gr2
    //      yield %tok2
    //   }  [@gr1, @gr2] {.0=[@gr1, @gr2]}
    //
    // Here result.0 from the forOp is produced by both @gr1 and @gr2. However,
    // %tok2 that is in `yield %tok2`, is produced by @gr2.  Aref-insertion will
    // need to insert aref between `load %tok1` and `yield %tok2`:
    //
    //   %ret = for  .. iter_args(%tok = %tok0) {
    //      %tok1 = update %buf[%tok]        @gr1
    //      copy %buf[%tok1], %aref1         @gr1
    //      %tok1' = copy %aref1, %buf        @gr2
    //      %tok2 = load %buf[%tok1']         @gr2
    //      copy %buf[%tok2] -> %aref2        @gr2
    //      %tok2' = copy %aref2 -> %buf     @gr1
    //      %tok2'' = phi %tok2, %tok2'
    //      yield %tok2''
    //   }  [@gr1, @gr2] {.0=[@gr1, @gr2]}
    //
    //  and after code split we have
    //
    //   @gr1 {
    //     %ret = for  .. iter_args(%tok = %tok0) {
    //        %tok1 = update %buf[%tok]         @gr1
    //        copy %buf[%tok1], %aref1          @gr1
    //        %tok2' = copy %aref2 -> %buf      @gr1
    //        yield %tok2'
    //     } [@gr1] {.0=[@gr1]}
    //   }
    //
    //   @gr2 {
    //     %ret = for  .. iter_args(%tok = %tok0) {
    //        %tok1' = copy %aref1, %buf        @gr2
    //        %tok2 = load %buf[%tok1']         @gr2
    //        copy %buf[%tok2] -> %aref2        @gr2
    //        yield %tok2
    //     } [@gr2]  {.0=[@gr2]}
    //   }
    //
    // The second aref copy is unnecessary since it copies an unmodified buffer
    // from @gr2 back to @gr1 solely to propagate token dependencies. While this
    // redundant copy could be eliminated through inter-group analysis of memory
    // access patterns later, such analysis would be non-trivial after
    // code-split.
    // The aref-insertion and code-split passes intentionally use local decision
    // making to remain simple and composable. Rather than complicate those
    // passes, we canonicalize the IR here to prevent unnecessary copies from
    // being generated in the first place. This maintains the clean separation
    // of concerns between passes while still achieving the desired
    // optimization.
    //
    // The canonicalization is performed in two steps:
    // 1. For token-returning results of the forOp, we annotate them only with
    //    the group of the last buffer update operation in the loop body
    // 2. We return the token from this last  update, rather than the last
    //    produced token which may be a read-operation.
    //
    // Given the input IR above, this pass will produce
    //   %ret = for  .. iter_args(%tok = %tok0) {
    //      %tok1 = update %buf[%tok]   @gr1
    //      %tok2 = load %buf[%tok1]    @gr2
    //      yield %tok1
    //   } [@gr1, @gr2] {.0=[@gr1]}
    //
    // It is valid, because right after this pass we will run aref-insertion
    // And arefs between update and load will ensure sequencing of operation,
    // i.e.
    //
    //   %ret = for  .. iter_args(%tok = %tok0) {
    //      %tok1 = update %buf[%tok]   @gr1
    //      copy %buf[tok1], %aref1     @gr1
    //      %tok1' = copy %aref1, %buf   @gr2
    //      %tok2  = load %buf[%tok1']   @gr2
    //      yield %tok1
    //   } [@gr1, @gr2] {.0=[@gr1]}
    //
    // and after code-split we will have
    //
    //   @gr1 {
    //     %ret = for  .. iter_args(%tok = %tok0) {
    //        %tok1 = update %buf[%tok]   @gr1
    //        copy %buf[tok1], %aref1     @gr1
    //        yield %tok1
    //     } [@gr1] {.0=[@gr1]}
    //   }
    //
    //   @gr2 {
    //     for  .. iter_args() {
    //        %tok1' = copy %aref1, %buf  @gr2
    //        %tok2  = load %buf[%tok1']  @gr2
    //     } [@gr2]
    //   }
    //
    // and copy is not present, by canonicalizing input IR.

    funcOp.walk([&](scf::ForOp forOp) {
      auto body = forOp.getBody();
      auto yieldOp = cast<scf::YieldOp>(body->getTerminator());
      IRMapping mapping;
      SmallVector<Value> yieldOperands;
      for (auto opnd : yieldOp->getOperands())
        yieldOperands.push_back(opnd);

      bool cloneYield = true;
      for (auto [idx, opnd] : llvm::enumerate(yieldOperands)) {
        if (isa<AsyncTokenType>(opnd.getType())) {
          auto op = opnd.getDefiningOp();
          // if op is a load, get the op that created the dep for the load
          if (auto loadOp = dyn_cast<ttng::TMEMLoadOp>(op))
            op = loadOp.getDep().getDefiningOp();
          Value token;
          // assme, the op here must be either a tmem_store or mmav5
          if (auto storeOp = dyn_cast<ttng::TMEMStoreOp>(op)) {
            token = storeOp.getToken();
          } else if (auto mmav5 = dyn_cast<ttng::MMAv5OpInterface>(op)) {
            token = mmav5.getToken();
          }
          // update token to be the one produced by the store/mma operation
          if (token) {
            auto groups = getGroups(op);
            assert(!groups.empty());

            // enforce that only one group updates tmem
            groups = {*groups.begin()};
            setGroups(op, groups);

            // ensure token is produced by the group that makes last buf update
            setGroupsIdx(forOp, idx, groups);

            mapping.map(opnd, token);
            cloneYield = true;
          }
        }
      }
      if (cloneYield) {
        OpBuilder builder(yieldOp);
        builder.clone(*yieldOp, mapping);
        yieldOp.erase();
      }
    });
  }

  void blackwellFlashAttentionPatches(triton::FuncOp funcOp) {
    // For Flash Attention, we want the partitioner to generate input IR in a
    // specific way. However, since the partitioner is currently under
    // development, for now we patch the input IR from the propagation pass to
    // add the group annotations we need.

    // When a for-loop yields a token that comes from an MMA operaton in
    // the @mma2 group and is consumed by the @correction group on the next
    // iteration, we want to annotate that loop output as being produced by the
    // correction group instead. This is done even though the actual computation
    // is performed by the @mma2 group. For example:
    //
    // %tok1 = for .. %tok = %tok0 {
    //    %tok2 = tmem_load %tok1      @correction
    //    %tok3 = tmem_store %tok2     @correction
    //    %tok4 = mma  %tok3           @mma2
    //    yield %tok4
    // } .0=[@correction]
    //
    // This transformation ensures that a copy operation is inserted between mma
    // and yield ops, rather than between iter_args and tmem_load. This
    // placement generates cleaner IR after aref-insertion, making it easier to
    // eliminate redundant copies later.

    funcOp.walk([&](scf::ForOp forOp) {
      auto forOpGroups = getGroups(forOp);
      auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      for (auto [idx, opnd] : llvm::enumerate(yieldOp->getOperands())) {
        if (!isa<ttng::MMAv5OpInterface>(opnd.getDefiningOp()))
          continue;
        auto groups = getGroups(opnd.getDefiningOp());
        bool hasMMA2 = false;
        for (auto group : groups) {
          if (group == "nvws.group.mma2") {
            hasMMA2 = true;
            break;
          }
        }
        if (hasMMA2) {
          groups = {"nvws.group.correction"};
        }
        setGroupsIdx(forOp, idx, groups);
      }
    });
  }

public:
  void runOnFunc(triton::FuncOp funcOp) {
    mlir::RewritePatternSet patterns(funcOp.getContext());
    patterns.add<RemoveUnusedTMEMStore>(funcOp.getContext());
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
      llvm_unreachable("Failed to remove tmem_store");

    correctGroups(funcOp);

    updateTokens(funcOp);

    blackwellFlashAttentionPatches(funcOp);
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
