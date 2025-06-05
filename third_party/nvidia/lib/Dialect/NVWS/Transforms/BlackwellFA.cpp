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

void arefReuse(triton::FuncOp funcOp) {
  std::map<std::string, SmallVector<scf::ForOp>> forOps;
  funcOp.walk([&](ttng::WarpGroupOp wgOp) {
    auto group = getGroup(wgOp);
    // llvm::errs() << "XXX1 group: " << group << "\n";
    if (group == "nvws.group.mma1") {
      wgOp.walk([&](scf::ForOp forOp) { forOps[group].push_back(forOp); });
    } else {
      return WalkResult::advance();
    }
    for (auto [group, forOps] : forOps) {
      // llvm::errs() << "XXX group: " << group
      //              << " forOps.size(): " << forOps.size() << "\n";
      if (forOps.size() != 2)
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (forOps.empty())
    return;
  if (forOps.at("nvws.group.mma1").size() != 2)
    return;

  SmallVector<Value> arefs_mma1[2];
  for (auto [i, forOp] : llvm::enumerate(forOps.at("nvws.group.mma1"))) {
    auto idx = i;
    forOp.walk([&](ttng::ArefPutEnterOp enterOp) {
      arefs_mma1[idx].push_back(enterOp.getAref());
    });
  }
  assert(arefs_mma1[0].size() == 1);
  assert(arefs_mma1[1].size() == 1);

  // reuse arefs from 1st loop in the 2nd loop
  arefs_mma1[1][0].replaceAllUsesWith(arefs_mma1[0][0]);

  arefs_mma1[1][0].getDefiningOp()->erase();
  // llvm::errs() << "ZZZ we are here\n";
}

bool removeCopyOpInSameBlock(ttng::ArefCopyOp &copyOp) {
  /*
    use case:

      %buf, %tok0 = alloc
      // no uses of %buf here
      %tok1 = for ..  %tok = %tok0 {
          %tok1 = mma .. %buf[%tok0]
          %dst, %dstTok = put.enter
          %tok2 = copy %buf[%tok1], %dst[%dstTok]
          put.exit
          yield %tok2
      }
      // no uses of %buf here

     becomes

      for .. {
          %dst, %dstTok = put.enter
          %tok1 = mma .. %dst[%tokTok]
          put.exit
          yield %tok1
      }
  */
  if (!copyOp.getToken())
    return false;

  // follow the uses of token, and ensure it only ends up in yieldOp
  // that is no other uses of src buffer after copyOp
  {
    SmallVector<OpOperand *> tokens;
    for (auto &use : copyOp.getToken().getUses())
      tokens.push_back(&use);
    while (!tokens.empty()) {
      auto token = tokens.pop_back_val();
      if (!isa<scf::YieldOp>(token->getOwner()))
        return false;

      auto yieldOp = cast<scf::YieldOp>(token->getOwner());
      auto result =
          yieldOp->getParentOp()->getResults()[token->getOperandNumber()];
      for (auto &use : result.getUses())
        tokens.push_back(&use);
    }
  }

  // now find first write to buffer
  auto buf = copyOp.getSrc();
  auto bufDef = buf.getDefiningOp();

  // ensure that buffer is result of tmem_alloc, and not some other put/get/etc
  if (!isa<ttng::TMEMAllocOp>(bufDef))
    return false;
  if (!isa<ttng::ArefPutEnterOp>(copyOp.getDst().getDefiningOp()))
    return false;

  auto allocOp = cast<ttng::TMEMAllocOp>(bufDef);
  Value allocTok = allocOp.getToken();

  Operation *firstUse = {};
  SmallVector<OpOperand *> uses;
  for (auto &use : allocTok.getUses())
    uses.push_back(&use);

  while (!uses.empty()) {
    auto use = uses.pop_back_val();
    if (auto forOp = dyn_cast<scf::ForOp>(use->getOwner())) {
      auto idx = use->getOperandNumber();
      assert(idx >= 3); // for loop has first three operands, lb/ub/step
      auto blockArgs = forOp.getBody()->getArguments();
      // blockArgs first value is loop induction variable,
      auto blockArg = blockArgs[idx - 3 + 1];
      assert(!blockArg.use_empty());
      for (auto &use : blockArg.getUses())
        uses.push_back(&use);
    } else if (auto fifOp = dyn_cast<scf::IfOp>(use->getOwner())) {
      llvm_unreachable("unsupported use in ifOp");
    } else if (isa<ttng::TMEMStoreOp, ttng::MMAv5OpInterface>(
                   use->getOwner())) {
      firstUse = use->getOwner();
      break;
    }
  }
  if (!firstUse)
    return false;
  assert(firstUse);

  // check that all uses of buffer are in the same block as copyOp
  for (auto users : allocOp.getResult().getUsers())
    if (users->getBlock() != copyOp->getBlock())
      return false;

  assert(firstUse->isBeforeInBlock(copyOp));

  // move put enter before first sue
  auto putEnterOp = cast<ttng::ArefPutEnterOp>(copyOp.getDst().getDefiningOp());
  putEnterOp->moveBefore(firstUse);
  OpBuilder builder(putEnterOp);
  putEnterOp.getIndexMutable().assign(
      mkConstant(builder, putEnterOp.getLoc(), 0, 32, getGroups(putEnterOp)));

  copyOp.getSrc().replaceUsesWithIf(
      copyOp.getDst(), [&](OpOperand &use) -> bool {
        return use.getOwner()->isBeforeInBlock(copyOp);
      });
  copyOp.getSrcDep().replaceAllUsesWith(copyOp.getDstDep());

  // remove copyOp
  copyOp->erase();

  // erase allocOp and token from the loop iter_args

  return true;
}
class NVWSBlackwellFA : public NVWSBlackwellFABase<NVWSBlackwellFA> {

public:
  void runOnFunc(triton::FuncOp funcOp) {
    SmallVector<ttng::ArefCopyOp> copyOps;
    funcOp.walk([&](ttng::ArefCopyOp copyOp) { copyOps.push_back(copyOp); });

    for (auto copyOp : copyOps)
      removeCopyOpInSameBlock(copyOp);
    arefReuse(funcOp);
  }

  void runOnOperation() override {
    auto mod = getOperation();
    mod.walk([&](triton::FuncOp funcOp) { runOnFunc(funcOp); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createNVWSBlackwellFAPass() {
  return std::make_unique<NVWSBlackwellFA>();
}
