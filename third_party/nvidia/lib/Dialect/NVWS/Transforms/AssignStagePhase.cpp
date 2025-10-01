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
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
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
  int partitionId;
  DenseMap<Value, int> tokToStagePosMap;

  AssignStagePhase(Value aref, int partitionId)
      : aref(aref), partitionId(partitionId) {}

  T getTypedOp(Operation *op) {
    if (auto opT = dyn_cast<T>(op)) {
      if (opT.getAref() == aref) {
        auto opPartitionIds = getPartitionIds(op);
        if (!opPartitionIds || llvm::is_contained(*opPartitionIds, partitionId))
          return opT;
      }
    }
    return {};
  }
  bool isBufferUsed(ArefBufferOp bufOp, Value token) {
    if (!bufOp)
      return false;
    if (bufOp.getAref() != this->aref)
      return false;
    return token == bufOp.getToken();
  }

  bool analyzeArefUseInBlock(Block *block, Value token) {
    for (auto &op : *block) {
      if (getTypedOp(&op) || isBufferUsed(dyn_cast<ArefBufferOp>(op), token)) {
        return true;
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        Value newTok;
        if (auto pos = findValuePosInRange(forOp.getInitArgs(), token)) {
          newTok = forOp.getRegionIterArgs()[*pos];
        }
        if (analyzeArefUseInBlock(forOp.getBody(), newTok))
          return true;
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (analyzeArefUseInBlock(ifOp.thenBlock(), token))
          return true;
        if (ifOp.elseBlock() && analyzeArefUseInBlock(ifOp.elseBlock(), token))
          return true;
      }
    }
    return false;
  }

  void assignArefIndexInForOp(scf::ForOp forOp, StagePhase &index) {

    Value newTok;
    if (auto pos = findValuePosInRange(forOp.getInitArgs(), index.token)) {
      newTok = forOp.getRegionIterArgs()[*pos];
    }
    // find uses of arefs in forOp body
    if (!analyzeArefUseInBlock(forOp.getBody(), newTok))
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

    auto useInThenBlock = analyzeArefUseInBlock(ifOp.thenBlock(), index.token);
    auto useInElseBlock =
        ifOp.elseBlock() ? analyzeArefUseInBlock(ifOp.elseBlock(), index.token)
                         : false;
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
      if (auto opT = getTypedOp(&op)) {
        ImplicitLocOpBuilder builder(opT.getLoc(), opT);
        auto partitionIds = getPartitionIds(&op);
        auto stageCluster = getStageCluster(&op);

        auto createInto = [&](auto opTy, auto... args) {
          using ty = decltype(opTy);
          return triton::gpu::createInto<ty>(
              builder, builder.getLoc(), partitionIds, stageCluster,
              std::forward<decltype(args)>(args)...);
        };

        auto nextStage = createInto(arith::AddIOp{}, index.stage,
                                    createInto(arith::ConstantIntOp{}, 1, 32));
        auto arefBuf = opT.getAref()
                           .template getDefiningOp<nvws::ArefCreateOp>()
                           .getOperand(0);
        auto depth = getArefDepth(cast<MemDescType>(arefBuf.getType()));

        auto cnd =
            createInto(arith::CmpIOp{}, arith::CmpIPredicate::eq, nextStage,
                       createInto(arith::ConstantIntOp{}, depth, 32));
        auto zero = createInto(arith::ConstantIntOp{}, 0, 32);
        index.stage = createInto(arith::SelectOp{}, cnd, zero, nextStage);

        auto nextPhase = createInto(arith::XOrIOp{}, index.phase,
                                    createInto(arith::ConstantIntOp{}, 1, 32));
        index.phase =
            createInto(arith::SelectOp{}, cnd, nextPhase, index.phase);

        index.token = opT.getToken();
        opT.getStageMutable().assign(index.stage);
        opT.getPhaseMutable().assign(index.phase);
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
      if (auto stageOp = dyn_cast<ArefStageInterface>(owner)) {
        stageOp.setStage(stage);
      } else if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        auto tokPos = tokUse.getOperandNumber() - 3;
        auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
        auto stagePos = tokToStagePosMap.at(yieldOp.getOperand(tokPos));
        propagateStage(forOp.getRegionIterArgs()[tokPos],
                       forOp.getRegionIterArgs()[stagePos], visited);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(owner)) {
        auto tokPos = tokUse.getOperandNumber();
        auto stagePos = tokToStagePosMap.at(token);
        auto parentOp = yieldOp->getParentOp();
        propagateStage(parentOp->getResult(tokPos),
                       parentOp->getResult(stagePos), visited);
      }
    }
  }

  static LogicalResult run(ArefCreateOp arefOp) {
    std::set<int> partitionIds;
    for (auto user : arefOp->getUsers()) {
      // Each partition requires its own stage/phase tracking for proper
      // multi-user handling; collect partition IDs in which this aref is used
      if (isa<T>(user)) {
        if (auto ids = getPartitionIds(user))
          partitionIds.insert(ids->begin(), ids->end());
      }
    }
    if (partitionIds.empty()) {
      // if partitionIds is an empty set, it means aref ops used outside ttg.ws
      // so we to insert a dummy partitionId for this aref, since we still need
      // to assign correct phase
      partitionIds.insert({0, 0});
    }

    // initialize indexes
    StagePhase index;
    ImplicitLocOpBuilder b(arefOp.getLoc(), arefOp);
    b.setInsertionPointAfter(arefOp);
    auto depth =
        getArefDepth(cast<MemDescType>(arefOp.getOperand(0).getType()));
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

void visitBackwardSlice(scf::ForOp wsLoop, Value value,
                        std::function<void(Operation *)> callback,
                        DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      if (forOp->hasAttr(kWarpSpecializeAttrName))
        return;
      auto pos = findValuePosInRange(forOp.getRegionIterArgs(), value);
      assert(pos);
      visitBackwardSlice(wsLoop, forOp.getInitArgs()[*pos], callback, visited);
    }
  } else if (auto defOp = value.getDefiningOp();
             isa<scf::IfOp, scf::ForOp>(defOp)) {
    auto pos = findValuePosInRange(defOp->getResults(), value);
    assert(pos);
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      visitBackwardSlice(wsLoop, ifOp.thenYield()->getOperand(*pos), callback,
                         visited);
      if (ifOp.elseBlock())
        visitBackwardSlice(wsLoop, ifOp.elseYield()->getOperand(*pos), callback,
                           visited);
      visitBackwardSlice(wsLoop, ifOp.getCondition(), callback, visited);
    } else {
      auto forOp = cast<scf::ForOp>(defOp);
      visitBackwardSlice(wsLoop,
                         forOp.getBody()->getTerminator()->getOperand(*pos),
                         callback, visited);
    }
  } else if (wsLoop.getBody()->findAncestorOpInBlock(*defOp)) {
    callback(defOp);
    for (auto operand : defOp->getOperands()) {
      visitBackwardSlice(wsLoop, operand, callback, visited);
    }
  }
}

LogicalResult assignStagePhase(triton::FuncOp funcOp) {
  SmallVector<ArefCreateOp> arefOps;
  funcOp.walk([&](ArefCreateOp arefOp) { arefOps.push_back(arefOp); });
  for (auto arefOp : arefOps) {
    if (failed(AssignStagePhase<ArefPutEnterOp>::run(arefOp)))
      return failure();
    if (failed(AssignStagePhase<ArefGetEnterOp>::run(arefOp)))
      return failure();
  }

  auto callback = [&](Operation *op) {
    if (!isa<scf::YieldOp, scf::IfOp, scf::ForOp, triton::ReduceOp>(op)) {
      auto partitionIds = getPartitionIds(op);
      assert(partitionIds);
      partitionIds->insert(0);
      setPartition(op, *partitionIds);
    }
  };

  funcOp.walk([&](scf::ForOp forOp) {
    DenseSet<Value> visited;
    if (forOp->hasAttr(kWarpSpecializeAttrName)) {
      for (auto result : forOp.getResults()) {
        // if result is of scalar type and is used outside of for-op, visit
        // all dependencies and assign default partition to them
        if (isa<IntegerType, FloatType>(result.getType()) &&
            !result.use_empty()) {
          auto arg = forOp.getBody()->getTerminator()->getOperand(
              result.getResultNumber());
          visitBackwardSlice(forOp, arg, callback, visited);
        }
      }
    }
  });
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
