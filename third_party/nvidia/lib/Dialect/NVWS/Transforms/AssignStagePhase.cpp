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

#include "Utilities.h"
#include "mlir/Analysis/TopologicalSortUtils.h"

#include "Utilities.h"
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
#include "triton/Dialect/TritonGPU/Transforms/PartitionBuilder.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir::triton;
using namespace mlir::triton::gpu;
using namespace mlir::triton::nvidia_gpu;
using namespace mlir::triton::nvws;

#define DEBUG_TYPE "nvws-lower-aref"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_NVWSASSIGNSTAGEPHASE
#include "nvidia/include/Dialect/NVWS/Transforms/Passes.h.inc"
namespace {
template <class T> struct AssignStagePhase {
  struct StagePhase {
    Value stage;
    Value phase;
    Value token;
  };
  Value aref;
  PartitionId partitionId;
  DenseMap<Value, int> tokToStagePosMap;

  AssignStagePhase(Value aref, PartitionId partitionId)
      : aref(aref), partitionId(partitionId) {}

  T isValidOp(Operation *op) {
    if (isa<T>(op) && op->getOperand(0) == aref) {
      auto opPartitionId = getPartitionId(op);
      if (!opPartitionId || *opPartitionId == partitionId)
        return cast<T>(op);
    }
    return {};
  }

  bool analyzeArefUseInBlock(Block *block) {
    for (auto &op : *block) {
      if (isValidOp(&op)) {
        return true;
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (analyzeArefUseInBlock(forOp.getBody()))
          return true;
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (analyzeArefUseInBlock(ifOp.thenBlock()))
          return true;
        if (ifOp.elseBlock() && analyzeArefUseInBlock(ifOp.elseBlock()))
          return true;
      }
    }
    return false;
  }

  void assignArefIndexInForOp(scf::ForOp forOp, StagePhase &index) {

    // find uses of arefs in forOp body
    if (!analyzeArefUseInBlock(forOp.getBody()))
      return;

    // add extra iterArgs to the forOp
    SmallVector<Value> extraIterArgs{index.stage, index.phase};
    SmallVector<Value *> arefIndexRefs{&index.stage, &index.phase};
    llvm::MapVector<int, Value *> arefTokenRefs;

    if (auto pos = findValuePosInRange(forOp.getInitArgs(), index.token)) {
      // keep reference of the token position to latest token value
      // we will need it update with the value returned from forOp
      arefTokenRefs[*pos] = &index.token;
      // update token value with iter argument
      index.token = forOp.getRegionIterArgs()[*pos];
    }
    // create new forOp with extra iterArgs
    OpBuilder builder(forOp);
    size_t nArgs = forOp.getRegionIterArgs().size();
    forOp = addIterArgsToLoop(builder, forOp, extraIterArgs);

    // update arefIndex with iterArgs in the forOp body
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *arefIndexRefs[idx - nArgs] = forOp.getRegionIterArgs()[idx];

    // assign arefIndex in the forOp body
    auto indexInBlock = assignArefIndexInBlock(forOp.getBody(), index);

    // update yieldOp to return new indexes
    SmallVector<Value> extraYieldArgs;
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());

    // associate token with stage positional argument in the yieldOp
    // we will need this in propagateStage function that will assign stage
    // to arefBuffer and arefExit ops
    tokToStagePosMap[indexInBlock.token] = nArgs + extraYieldArgs.size();
    extraYieldArgs.push_back(indexInBlock.stage);
    if (index.phase)
      extraYieldArgs.push_back(indexInBlock.phase);
    appendToForOpYield(forOp, extraYieldArgs);

    // update arefIndex with results from newForOp
    for (size_t idx = nArgs; idx < forOp.getRegionIterArgs().size(); ++idx)
      *arefIndexRefs[idx - nArgs] = forOp.getResult(idx);
    for (auto [idx, arefTokenRef] : arefTokenRefs)
      *arefTokenRef = forOp.getResult(idx);
  }

  void assignArefIndexInIfOp(scf::IfOp ifOp, StagePhase &index) {

    auto useInThenBlock = analyzeArefUseInBlock(ifOp.thenBlock());
    auto useInElseBlock =
        ifOp.elseBlock() ? analyzeArefUseInBlock(ifOp.elseBlock()) : false;
    if (!useInThenBlock && !useInElseBlock)
      return;

    // add extra results to the ifOp
    SmallVector<Type> extraIfResults{index.stage.getType(),
                                     index.phase.getType()};
    SmallVector<Value *> arefIndexRefs{&index.stage, &index.phase};

    // create new ifOp with extra results
    OpBuilder builder(ifOp);
    size_t nArgs = ifOp.getResults().size();
    auto newIfOp = replaceIfOpWithNewSignature(builder, ifOp, extraIfResults);

    // assign arefIndex in then-body
    auto thenIndex = assignArefIndexInBlock(newIfOp.thenBlock(), index);

    // assign arefIndex in else-body
    auto elseIndex = newIfOp.elseBlock()
                         ? assignArefIndexInBlock(newIfOp.elseBlock(), index)
                         : index;

    // update yieldOp to return new indexes
    auto thenYieldOp = newIfOp.thenYield();
    auto elseYieldOp = newIfOp.elseYield();
    // insert new indexes to the yieldOp
    llvm::MapVector<int, Value *> arefTokenRefs;

    // find token pos in yieldOp and make a reference to  arefIndexMap value
    if (auto pos =
            findValuePosInRange(thenYieldOp->getOperands(), index.token)) {
      arefTokenRefs[*pos] = &index.token;
    }
    if (auto pos =
            findValuePosInRange(elseYieldOp->getOperands(), index.token)) {
      arefTokenRefs[*pos] = &index.token;
    }

    tokToStagePosMap[thenIndex.token] = thenYieldOp.getNumOperands();
    tokToStagePosMap[elseIndex.token] = elseYieldOp.getNumOperands();
    thenYieldOp->insertOperands(thenYieldOp.getNumOperands(), thenIndex.stage);
    elseYieldOp->insertOperands(elseYieldOp.getNumOperands(), elseIndex.stage);
    if (thenIndex.phase) {
      thenYieldOp->insertOperands(thenYieldOp.getNumOperands(),
                                  thenIndex.phase);
      elseYieldOp->insertOperands(elseYieldOp.getNumOperands(),
                                  elseIndex.phase);
    }
    ifOp.erase();

    // update arefIndex with results from newIfOp
    for (size_t idx = nArgs; idx < newIfOp.getResults().size(); ++idx)
      *arefIndexRefs[idx - nArgs] = newIfOp.getResult(idx);
    for (auto [idx, arefTokenRef] : arefTokenRefs)
      *arefTokenRef = newIfOp.getResult(idx);
  }

  StagePhase assignArefIndexInBlock(Block *block, StagePhase index) {
    for (auto &op : llvm::make_early_inc_range(*block)) {
      if (auto opT = isValidOp(&op)) {
        ImplicitLocOpBuilder b(opT.getLoc(), opT);

        auto nextStage = b.create<arith::AddIOp>(
            index.stage, b.create<arith::ConstantIntOp>(1, 32));
        auto arefBuf = opT.getAref()
                           .template getDefiningOp<nvws::ArefCreateOp>()
                           .getOperand(0);
        auto depth = cast<MemDescType>(arefBuf.getType()).getShape().front();

        auto cnd =
            b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, nextStage,
                                    b.create<arith::ConstantIntOp>(depth, 32));
        auto zero = b.create<arith::ConstantIntOp>(0, 32);
        index.stage = b.create<arith::SelectOp>(cnd, zero, nextStage);

        auto nextPhase = b.create<arith::XOrIOp>(
            index.phase, b.create<arith::ConstantIntOp>(1, 32));
        index.phase = b.create<arith::SelectOp>(cnd, nextPhase, index.phase);

        index.token = opT.getToken();
        opT.getStageMutable().assign(index.stage);
        opT->setOperand(2, index.phase);

      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        assignArefIndexInForOp(forOp, index);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        assignArefIndexInIfOp(ifOp, index);
      }
    }

    return index;
  }

  void propagateStage(Value token, Value stage,
                      DenseSet<Operation *> &visited) {
    for (auto &tokUse : token.getUses()) {
      auto owner = tokUse.getOwner();
      if (visited.contains(owner))
        continue;
      visited.insert(owner);
      if (isa<ArefGetExitOp, ArefPutExitOp>(owner)) {
        owner->setOperand(2, stage);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
        auto tokPos = tokUse.getOperandNumber();
        auto stagePos = tokToStagePosMap.at(token);
        auto parentOp = yieldOp->getParentOp();
        if (auto forOp = dyn_cast<scf::ForOp>(parentOp))
          propagateStage(forOp.getRegionIterArgs()[tokPos],
                         forOp.getRegionIterArgs()[stagePos], visited);
        propagateStage(parentOp->getResult(tokPos),
                       parentOp->getResult(stagePos), visited);
      }
    }
  }

  static LogicalResult run(ArefCreateOp arefOp) {

    std::set<PartitionId> partitionIds;
    for (auto user : arefOp->getUsers()) {
      // Each partition requires its own stage/phase tracking for proper
      // multi-user handling; collect partition IDs in which this aref is used
      if (isa<T>(user)) {
        if (auto partitionId = getPartitionId(user))
          partitionIds.insert(*partitionId);
      }
    }

    // initialize indexes
    StagePhase index;
    ImplicitLocOpBuilder b(arefOp.getLoc(), arefOp);
    b.setInsertionPointAfter(arefOp);
    auto depth =
        cast<MemDescType>(arefOp.getOperand(0).getType()).getShape().front();
    index.stage = b.create<arith::ConstantIntOp>(depth - 1, 32);

    static_assert(std::is_same_v<T, ArefPutEnterOp> ||
                      std::is_same_v<T, ArefGetEnterOp>,
                  "ArefPutEnterOp or ArefGetEnterOp expected");
    auto initPhase = std::is_same_v<T, ArefPutEnterOp> ? 0 : 1;
    index.phase = b.create<arith::ConstantIntOp>(initPhase, 32);

    for (auto partitionId : partitionIds) {
      // assign stage/phase to enter/exit Ops in each partition aref is used
      AssignStagePhase arefIndex(arefOp.getResult(), partitionId);

      // assign stage/phase to enterOps
      arefIndex.assignArefIndexInBlock(arefOp->getBlock(), index);

      // propagate stage to exitOps following enterOp token
      for (auto user : arefOp->getUsers())
        if (auto enterOp = dyn_cast<T>(user)) {
          DenseSet<Operation *> visited;
          arefIndex.propagateStage(enterOp.getToken(), enterOp.getStage(),
                                   visited);
        }
    }

    return success();
  }
};

static LogicalResult assignStagePhase(triton::FuncOp funcOp) {
  SmallVector<ArefCreateOp> arefOps;
  funcOp.walk([&](ArefCreateOp arefOp) { arefOps.push_back(arefOp); });
  for (auto arefOp : arefOps) {
    if (failed(AssignStagePhase<ArefPutEnterOp>::run(arefOp)))
      return failure();
    if (failed(AssignStagePhase<ArefGetEnterOp>::run(arefOp)))
      return failure();
  }
  return success();
}

// ----------------------------------------------------------------------------

} // anonymous namespace

class NVWSAssignStagePhase
    : public impl::NVWSAssignStagePhaseBase<NVWSAssignStagePhase> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp m = getOperation();

    m.walk([&](triton::FuncOp funcOp) {
      if (failed(assignStagePhase(funcOp)))
        signalPassFailure();
    });
  }
}; // namespace triton

} // namespace triton
} // namespace mlir
