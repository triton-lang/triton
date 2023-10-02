/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
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

#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Utility.h"

#include "mlir/Analysis/SliceAnalysis.h"

#include <algorithm>
#include <unordered_set>

using namespace mlir;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {
struct Channel {
public:
  using Relation = std::pair<int, int>;

  Channel(int producer, int consumer, Operation *src, Operation *dst)
      : relation(producer, consumer), srcOp(src), dstOp(dst) {}

  bool operator==(const Channel &c) {
    return relation == c.relation && srcOp == c.srcOp && dstOp == c.dstOp;
  }

  Relation relation;
  Operation *srcOp;
  Operation *dstOp;
};

//===----------------------------------------------------------------------===//
// createToken
//===----------------------------------------------------------------------===//

DenseMap<Channel *, Value>
createToken(const DenseMap<Operation *, SmallVector<Channel *>> &map,
            triton::FuncOp funcOp, int numStages) {
  DenseMap<Channel *, Value> ret;
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  for (auto it = map.begin(); it != map.end(); ++it) {
    Value v;
    if (it->second.front()->srcOp->getParentOfType<scf::ForOp>()) {
      v = builder.create<ttng::CreateTokenOp>(funcOp.getLoc(), numStages);
    } else {
      // No need to pipeline
      v = builder.create<ttng::CreateTokenOp>(funcOp.getLoc(), 1);
    }
    for (auto &c : it->second) {
      ret[c] = v;
    }
  }
  return ret;
}

//===----------------------------------------------------------------------===//
// createBuffer
//===----------------------------------------------------------------------===//

DenseMap<Channel *, Value> createBuffer(const SmallVector<Channel *> &channels,
                                        triton::FuncOp funcOp, int numStages) {
  DenseMap<Channel *, Value> bufferMap;
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  for (const auto &c : channels) {
    auto loadOp = dyn_cast<triton::LoadOp>(c->srcOp);
    Value loadResult = loadOp.getResult();
    if (auto tensorType = loadResult.getType().dyn_cast<RankedTensorType>()) {
      // Get basic information from tensorType
      auto order = ttg::getOrder(tensorType.getEncoding());
      auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
      auto elemType = tensorType.getElementType();

      // Get shape, layout and type of a slice
      auto sliceShape = tensorType.getShape();
      auto sharedLayout = ttg::SharedEncodingAttr::get(
          context, sliceShape, order, CTALayout, elemType);
      auto sliceType =
          RankedTensorType::get(sliceShape, elemType, sharedLayout);

      // Get shape, layout and type of the complete buffer
      SmallVector<int64_t> bufferShape(sliceShape.begin(), sliceShape.end());
      if (loadOp->getParentOfType<scf::ForOp>()) {
        bufferShape.insert(bufferShape.begin(), numStages);
      } else {
        // No need to pipeline
        bufferShape.insert(bufferShape.begin(), 1);
      }
      auto bufferType =
          RankedTensorType::get(bufferShape, elemType, sharedLayout);
      Value buffer =
          builder.create<ttg::AllocTensorOp>(funcOp.getLoc(), bufferType);
      bufferMap[c] = buffer;
    } else {
      llvm_unreachable("Unexpected result type");
    }
  }
  return bufferMap;
}

//===----------------------------------------------------------------------===//
// appendPipelineIdxToLoopArgs
//===----------------------------------------------------------------------===//

scf::ForOp appendPipelineIdxToLoopArgs(scf::ForOp forOp, int numStages,
                                       scf::ForOp &parentForOp) {
  auto loc = forOp.getLoc();
  Block *body = forOp.getBody();

  // The agentId set of pipelineIdx is the union of agentId sets of all ops in
  // the for loop
  OpBuilderWithAgentIds builder(forOp.getContext());
  builder.setAgentIdsFromArray(collectAgentIds(forOp));

  builder.setInsertionPoint(forOp);
  Value numStagesVal =
      builder.createWithAgentIds<arith::ConstantIntOp>(loc, numStages, 32);
  // Append pipelineIdx to block arguments
  Value pipelineIdx =
      body->insertArgument(body->getNumArguments(), builder.getI32Type(), loc);

  // pipelineIdx = (pipelineIdx + 1) % numStages
  auto yieldOp = llvm::cast<scf::YieldOp>(body->getTerminator());
  builder.setInsertionPoint(yieldOp);
  Value one = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 1, 32);

  Value pipelineIdxPlusOne =
      builder.createWithAgentIds<arith::AddIOp>(loc, pipelineIdx, one);

  // Append pipelineIdx to yield operands
  yieldOp->insertOperands(yieldOp.getNumOperands(), {pipelineIdxPlusOne});

  // Copy iter operands of forOp
  SmallVector<Value> newLoopArgs;
  for (auto operand : forOp.getIterOperands())
    newLoopArgs.push_back(operand);

  // Append initial value of pipelineIdx to newLoopArgs
  builder.setInsertionPoint(forOp);
  Value initValue;
  if (parentForOp) {
    // Make sure prior pipelineIdx is inserted in the end of parentForOp
    initValue = parentForOp.getBody()->getArguments().back();
    Value numSteps = builder.createWithAgentIds<arith::SubIOp>(
        loc, forOp.getUpperBound(), forOp.getLowerBound());
    auto one = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 1, 32);
    numSteps = builder.createWithAgentIds<arith::AddIOp>(loc, numSteps,
                                                         forOp.getStep());
    numSteps = builder.createWithAgentIds<arith::SubIOp>(loc, numSteps, one);
    numSteps = builder.createWithAgentIds<arith::DivUIOp>(loc, numSteps,
                                                          forOp.getStep());
    initValue =
        builder.createWithAgentIds<arith::MulIOp>(loc, initValue, numSteps);
  } else {
    initValue = builder.createWithAgentIds<arith::ConstantIntOp>(loc, 0, 32);
  }
  newLoopArgs.push_back(initValue);

  // Create newForOp and take the region of forOp
  auto newForOp = builder.createWithAgentIds<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      newLoopArgs);
  newForOp.getRegion().takeBody(forOp.getRegion());

  // Replace forOp with newForOp
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  forOp.erase();

  return newForOp;
}

//===----------------------------------------------------------------------===//
// appendPipelineIdxArgs
//===----------------------------------------------------------------------===//

void appendPipelineIdxArgs(SmallVector<Operation *> &backbone, int numStages) {

  SmallVector<scf::ForOp> orderedForOps;
  for (auto &op : backbone) {
    op->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
      if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
        orderedForOps.push_back(forOp);
      }
    });
  }

  for (auto &op : orderedForOps) {
    scf::ForOp parentForOp = op->getParentOfType<scf::ForOp>();
    auto newForOp = appendPipelineIdxToLoopArgs(op, numStages, parentForOp);
    auto backboneForItr =
        std::find(backbone.begin(), backbone.end(), op.getOperation());
    if (backboneForItr != backbone.end()) {
      // Update backbone
      *backboneForItr = newForOp.getOperation();
    }
  }
}

//===----------------------------------------------------------------------===//
// checkDependencyAndCollectUsedArgs
//===----------------------------------------------------------------------===//

SmallVector<unsigned> checkDependencyAndCollectUsedArgs(
    scf::ForOp forOp, AgentId agentId,
    DenseMap<BlockArgument, Value> &blockArgToYieldOperand) {

  std::unordered_set<Operation *> visited;
  SetVector<unsigned> argSet;

  // DFS
  std::function<void(Operation *)> dfs = [&](Operation *op) {
    if (visited.find(op) != visited.end())
      return;
    visited.insert(op);
    for (Value operand : op->getOperands()) {
      if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
        if (!blockArgToYieldOperand[blockArg])
          continue;
        argSet.insert(blockArg.getArgNumber() - forOp.getNumInductionVars());
        operand = blockArgToYieldOperand[blockArg];
      }
      Operation *depOp = operand.getDefiningOp();
      assert(depOp && "Unexpected Value with no defining op");
      if (depOp->getBlock() != forOp.getBody())
        continue;
      assert(hasAgentId(depOp, agentId) && "Dependency error");
      dfs(depOp);
    }
  };

  // Start from operations that are marked with this agentId explicitly and
  // check dependency with DFS traversal
  forOp.walk([&](Operation *op) {
    if (hasAgentId(op, agentId) && !isa<scf::YieldOp>(op))
      dfs(op);
  });

  // Collect used block args
  SmallVector<unsigned> args(argSet.begin(), argSet.end());
  llvm::sort(args);
  return args;
}

//===----------------------------------------------------------------------===//
// createForOpsForEachAgentId
//===----------------------------------------------------------------------===//

DenseMap<AgentId, scf::ForOp> createForOpsForEachAgentId(scf::ForOp forOp) {
  // Collect operation list for each agentId
  DenseMap<AgentId, SmallVector<Operation *>> opList;
  for (Operation &op : forOp.getBody()->without_terminator())
    for (AgentId agentId : getAgentIds(&op))
      opList[agentId].push_back(&op);

  // Prepare blockArgToYieldOperand mapping
  DenseMap<BlockArgument, Value> blockArgToYieldOperand;
  auto yieldOp = llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  assert(yieldOp.getNumOperands() == forOp.getNumRegionIterArgs());
  for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i)
    blockArgToYieldOperand[forOp.getRegionIterArg(i)] = yieldOp.getOperand(i);

  auto loc = forOp.getLoc();
  OpBuilderWithAgentIds builder(forOp.getContext());
  DenseMap<AgentId, scf::ForOp> agentsToForOp;

  // Create newForOp for each agent
  for (AgentId agentId : collectAgentIds(forOp)) {
    auto usedArgs = checkDependencyAndCollectUsedArgs(forOp, agentId,
                                                      blockArgToYieldOperand);

    // Prepare newLoopArgs
    SmallVector<Value> newLoopArgs;
    for (unsigned argNumber : usedArgs)
      newLoopArgs.push_back(forOp.getIterOperands()[argNumber]);

    // Create newForOp
    builder.setAgentIdsFromArray({agentId});
    builder.setInsertionPoint(forOp);
    auto newForOp = builder.createWithAgentIds<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        newLoopArgs);

    // Initialize Value mapping from forOp to newForOp
    IRMapping mapping;
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (unsigned i = 0; i < usedArgs.size(); ++i) {
      auto oldArg = forOp.getRegionIterArgs()[usedArgs[i]];
      auto newArg = newForOp.getRegionIterArgs()[i];
      mapping.map(oldArg, newArg);
    }

    // Clone all operations with this agentId to newForOp
    builder.setInsertionPointToStart(newForOp.getBody());
    for (Operation *op : opList[agentId]) {
      Operation *newOp = builder.clone(*op, mapping);
      setAgentIds(newOp, {agentId});
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
    }

    // Create YieldOp for newForOp
    SmallVector<Value> newYieldOperands;
    for (unsigned i : usedArgs)
      newYieldOperands.push_back(mapping.lookup(yieldOp.getOperand(i)));
    auto newYieldOp =
        builder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    setAgentIds(newYieldOp, {agentId});

    // Replace results of forOp with results of newForOp
    for (unsigned i = 0; i < usedArgs.size(); ++i) {
      auto oldResult = forOp.getResult(usedArgs[i]);
      auto newResult = newForOp.getResult(i);
      oldResult.replaceAllUsesWith(newResult);
    }

    agentsToForOp[agentId] = newForOp;
  }

  return agentsToForOp;
}

//===----------------------------------------------------------------------===//
// createIfOpsForEachAgentId
//===----------------------------------------------------------------------===//

DenseMap<AgentId, scf::IfOp> createIfOpsForEachAgentId(scf::IfOp ifOp) {
  // TODO: to be implemented
  OpBuilderWithAgentIds builder(ifOp.getContext());
  DenseMap<AgentId, scf::IfOp> agentsToIfOp;
  return agentsToIfOp;
}

//===----------------------------------------------------------------------===//
// SpecializeAgentRegion
//===----------------------------------------------------------------------===//

DenseMap<AgentId, scf::IfOp> SpecializeAgentRegion(triton::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(context);
  auto loc = funcOp.getLoc();

  // Get block from funcOp
  Block *block = &funcOp.getBody().front();
  auto returnOp = llvm::cast<triton::ReturnOp>(block->getTerminator());

  // Collect original operations
  SmallVector<Operation *> opList;
  for (Operation &op : block->getOperations())
    opList.push_back(&op);

  // Get curAgentId
  builder.setInsertionPoint(returnOp);
  Value curAgentId = builder.create<ttng::GetAgentIdOp>(loc);

  // Resources for each agentId
  DenseMap<AgentId, std::shared_ptr<OpBuilderWithAgentIds>> agentsToBuilders;
  DenseMap<AgentId, scf::IfOp> agentsToIfOp;
  DenseMap<AgentId, IRMapping> agentsToIRMappings;

  for (AgentId agentId : collectAgentIds(funcOp)) {
    // Create IfOp for each agentId
    Value cond = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, curAgentId,
        builder.create<arith::ConstantIntOp>(loc, agentId, 32));

    auto ifOp = builder.create<scf::IfOp>(loc, cond);
    agentsToIfOp[agentId] = ifOp;
    setAgentIds(ifOp, {agentId});

    // Create OpBuilderWithAgentIds for each agent
    auto agentBuilder = std::make_shared<OpBuilderWithAgentIds>(context);
    agentsToBuilders[agentId] = agentBuilder;
    agentBuilder->setAgentIdsFromArray({agentId});

    // Set insertion point before yieldOp
    auto yieldOp = ifOp.thenYield();
    setAgentIds(yieldOp, {agentId});
    agentBuilder->setInsertionPoint(yieldOp);
  }

  // Clone all operations into corresponding if blocks
  SmallVector<Operation *> cloned;
  for (Operation *op : opList) {
    auto agentIds = getAgentIds(op);
    if (!agentIds.empty()) {
      cloned.push_back(op);
      for (AgentId agentId : getAgentIds(op)) {
        IRMapping &mapping = agentsToIRMappings[agentId];
        Operation *newOp = agentsToBuilders[agentId]->clone(*op, mapping);
        for (unsigned i = 0; i < op->getNumResults(); ++i)
          mapping.map(op->getResult(i), newOp->getResult(i));
      }
    }
  }

  // Remove original operations that have been cloned in reverse order
  for (auto it = cloned.rbegin(); it != cloned.rend(); ++it) {
    Operation *op = *it;
    op->erase();
  }

  return agentsToIfOp;
}

//===----------------------------------------------------------------------===//
// collectAsyncChannels
//===----------------------------------------------------------------------===//

void collectAsyncChannels(SmallVector<std::unique_ptr<Channel>> &channels,
                          triton::FuncOp &funcOp) {
  funcOp.walk([&](Operation *op) {
    for (auto result : op->getResults()) {
      if (result.use_empty() || !op->hasAttr("async_agent")) {
        continue;
      }
      auto producerAgent =
          op->getAttrOfType<DenseIntElementsAttr>("async_agent");
      if (producerAgent.getValues<int>().size() > 1) {
        continue;
      }
      for (Operation *userOp : result.getUsers()) {
        if (!userOp->hasAttr("async_agent") ||
            userOp->getAttrOfType<DenseIntElementsAttr>("async_agent")
                    .getValues<int>()
                    .size() > 1) {
          continue;
        }
        auto consumerAgentId =
            userOp->getAttrOfType<DenseIntElementsAttr>("async_agent")
                .getValues<int>()[0];
        auto producerAgentId = producerAgent.getValues<int>()[0];
        if (producerAgentId != consumerAgentId) {
          channels.push_back(std::make_unique<Channel>(
              producerAgentId, consumerAgentId, op, userOp));
        }
      }
    }
  });
}

//===----------------------------------------------------------------------===//
// reduceChannels
//===----------------------------------------------------------------------===//

void reduceChannels(SmallVector<Channel *> &channels,

                    DenseMap<Operation *, SmallVector<Channel *>> &map) {
  // If producers or their consumers has the same convergent comsumer,
  // and those producers, producers' consumers and the convergent comsumer are
  // in the same block, They share the same token.
  auto checkConverge = [](Operation *op1, Operation *op2) -> Operation * {
    // Only check level-0 and level-1 convergence, e.g.
    // producer:       load0          load1
    //                   |              |
    // consumer:  convertLayout0  convertLayout1
    //                    \             /
    // consumer:                 dot
    // The example above is level-1 convergence.
    // If convertLayoutOps converge in deeper depth, this function will
    // fail to detect.
    // TODO: implement general level-N convergence.
    if (op1 == op2) {
      return op1;
    }
    if (op1->getBlock() == op2->getBlock() && op1->hasOneUse() &&
        op2->hasOneUse() &&
        *(op1->getUsers().begin()) == *(op2->getUsers().begin()) &&
        (*(op1->getUsers().begin()))->getBlock() == op1->getBlock()) {
      return *(op1->getUsers().begin());
    }
    return nullptr;
  };
  assert(channels.size() > 0 && "channel size is zero");
  // Compare with existing channels in map
  for (auto c0 = channels.begin(); c0 != channels.end(); ++c0) {
    bool isConvergent = false;
    for (auto &kv : map) {
      if (kv.second.size() > 0 &&
          (*c0)->srcOp->getBlock() == kv.second.front()->srcOp->getBlock()) {
        if (auto cvg = checkConverge((*c0)->dstOp, kv.second.front()->dstOp)) {
          kv.second.push_back(*c0);
          isConvergent = true;
          break;
        }
      }
    }
    if (!isConvergent) {
      map[(*c0)->dstOp].push_back(*c0);
    }
  }

  // Reorder channels and maps based on locations of producers
  for (auto &kv : map) {
    if (kv.second.size() > 1) {
      auto &allOps = kv.second.front()->srcOp->getBlock()->getOperations();
      std::sort(
          kv.second.begin(), kv.second.end(), [&](Channel *a, Channel *b) {
            auto itrA =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == a->srcOp;
                });
            auto itrB =
                std::find_if(allOps.begin(), allOps.end(), [&](Operation &op) {
                  Operation *opPointer = &op;
                  return opPointer == b->srcOp;
                });
            assert(itrA != allOps.end() && itrB != allOps.end());
            return std::distance(itrA, itrB) < 0;
          });
    }
  }
}

//===----------------------------------------------------------------------===//
// getBackbone
//===----------------------------------------------------------------------===//

SmallVector<Operation *> getBackbone(triton::FuncOp funcOp,
                                     const SmallVector<Channel *> &channels) {
  // Backbone: outermost Ops with regions in funcOp which contain at least one
  // relation between producer and consumer. It assumes producer-consumer
  // relation going across two outermost Ops in funcOp is forbidden. For
  // example, In the example of runOnOperation(), only the outermost ForOp is
  // backbone, the inner ForOp is not.
  SmallVector<Operation *> backboneOps;
  auto isBackbone = [&](Operation *backbone) -> bool {
    for (auto c : channels) {
      Operation *producer = c->srcOp, *consumer = c->dstOp;
      while (producer && !isa<triton::FuncOp>(producer->getParentOp())) {
        producer = producer->getParentOp();
      }
      while (consumer && !isa<triton::FuncOp>(consumer->getParentOp())) {
        consumer = consumer->getParentOp();
      }
      if (producer == backbone && consumer == backbone) {
        return true;
      }
      assert((producer != backbone ||
              isa<triton::FuncOp>(producer->getParentOp())) &&
             (consumer != backbone ||
              isa<triton::FuncOp>(consumer->getParentOp())) &&
             "Error: producer and consumer belongs to different backboneOps");
    }
    return false;
  };
  Operation *op;
  for (Operation &bodyOp : funcOp.getBody().front().getOperations()) {
    op = &bodyOp;
    if (op->getNumRegions() > 0) {
      // If this op as a whole is a producer or consumer, continue
      if (getAgentIds(op).size() == 1) {
        continue;
      }
      if (isBackbone(op)) {
        backboneOps.push_back(op);
      }
    }
  }
  return backboneOps;
}

//===----------------------------------------------------------------------===//
// buildAsyncComm
//===----------------------------------------------------------------------===//

void buildAsyncComm(const DenseMap<Operation *, SmallVector<Channel *>> &map,
                    const DenseMap<Channel *, Value> &tokenMap,
                    const DenseMap<Channel *, Value> &bufferMap,
                    int numStages) {

  auto getSameLevelOp = [](Operation *p, Operation *c) -> Operation * {
    while (!isa<triton::FuncOp>(c)) {
      if (c->getParentOp() == p->getParentOp()) {
        return c;
      }
      c = c->getParentOp();
    }
    llvm_unreachable("Falied to find consumer's same level Op with producer");
  };

  auto consumerReleaseHeutistic = [&](Operation *p,
                                      Operation *c) -> Operation * {
    if (c->getBlock() == p->getBlock()) {
      auto consumerAgentId =
          c->getAttrOfType<DenseIntElementsAttr>("async_agent")
              .getValues<int>()[0];
      for (auto it = c->getBlock()->rbegin(); it != c->getBlock()->rend();
           ++it) {
        if (!it->hasAttr("async_agent")) {
          continue;
        }
        auto asyncAttr = it->getAttrOfType<DenseIntElementsAttr>("async_agent")
                             .getValues<int>();
        if (asyncAttr.size() == 1 && asyncAttr[0] == consumerAgentId) {
          return &(*it);
        }
      }
      return nullptr;
    } else {
      return getSameLevelOp(p, c);
    }
  };

  auto getAgents = [&](Operation *p, Operation *c, SmallVector<AgentId> &agentP,
                       SmallVector<AgentId> &agentC,
                       SmallVector<AgentId> &agentsPC) -> void {
    agentP = collectAgentIds(p);
    agentC = collectAgentIds(c);
    agentsPC.reserve(agentP.size() + agentC.size());
    agentsPC.insert(agentsPC.end(), agentP.begin(), agentP.end());
    agentsPC.insert(agentsPC.end(), agentC.begin(), agentC.end());
  };
  // TODO: try to optimize locations of arriving and waiting token
  // for fused-attention
  for (auto kv : map) {
    /*****************Token related*****************/
    auto headProducer = kv.second.front()->srcOp;
    auto tailProducer = kv.second.back()->srcOp;
    auto headConsumer = kv.second.front()->dstOp;
    auto tailConsumer = kv.second.back()->dstOp;
    auto token = tokenMap.find(kv.second.front())->second;
    SmallVector<AgentId> agentP, agentC, agentsPC;
    getAgents(headProducer, headConsumer, agentP, agentC, agentsPC);
    OpBuilderWithAgentIds builder(headProducer->getContext());

    if (auto funcOp = dyn_cast<triton::FuncOp>(headProducer->getParentOp())) {
      builder.setInsertionPointToStart(&(funcOp.getBody().front()));
    } else {
      builder.setInsertionPoint(headProducer->getParentOp());
    }
    builder.setAgentIdsFromArray(agentsPC);
    Value pipelineIdx;
    Value numStagesVal = builder.createWithAgentIds<arith::ConstantIntOp>(
        headProducer->getLoc(), numStages, 32);
    if (auto forOp = headProducer->getParentOfType<scf::ForOp>()) {
      pipelineIdx = forOp.getBody()->getArguments().back();
    } else {
      // existing");
      pipelineIdx = builder.createWithAgentIds<arith::ConstantIntOp>(
          headProducer->getLoc(), 0, 32);
    }

    // insert ProducerAcquireOp
    builder.setInsertionPoint(headProducer);
    if (headProducer->getParentOfType<scf::ForOp>()) {
      pipelineIdx = builder.createWithAgentIds<arith::RemSIOp>(
          headProducer->getLoc(), pipelineIdx, numStagesVal);
    }
    builder.setAgentIdsFromArray(agentP);
    builder.createWithAgentIds<ttng::ProducerAcquireOp>(headProducer->getLoc(),
                                                        token, pipelineIdx);

    // insert ProducerCommitOp
    builder.setInsertionPointAfter(tailProducer);
    builder.createWithAgentIds<ttng::ProducerCommitOp>(tailProducer->getLoc(),
                                                       token, pipelineIdx);

    builder.setAgentIdsFromArray(agentC);
    // insert ConsumerWaitOp
    auto consumerWaitPoint = getSameLevelOp(headProducer, headConsumer);
    builder.setInsertionPoint(consumerWaitPoint);
    builder.createWithAgentIds<ttng::ConsumerWaitOp>(headConsumer->getLoc(),
                                                     token, pipelineIdx);

    // insert ConsumerReleaseOp
    auto consumerReleasePoint =
        consumerReleaseHeutistic(tailProducer, tailConsumer);
    builder.setInsertionPointAfter(consumerReleasePoint);
    builder.createWithAgentIds<ttng::ConsumerReleaseOp>(
        consumerReleasePoint->getLoc(), token, pipelineIdx);

    /*****************Buffer related*****************/
    /// splitLoadsInForLoop
    for (auto &c : kv.second) {
      assert(isa<triton::LoadOp>(c->srcOp) && "prodcuerOp is not tt.load");
      auto loadOp = cast<triton::LoadOp>(c->srcOp);
      auto buffer = bufferMap.find(c)->second;
      MLIRContext *context = loadOp->getContext();
      OpBuilderWithAgentIds builder(context);
      builder.setInsertionPoint(loadOp->getParentOp());
      builder.setAgentIdsFromArray(agentsPC);

      builder.setInsertionPoint(loadOp);
      Value loadResult = loadOp.getResult();
      if (auto tensorType = loadResult.getType().dyn_cast<RankedTensorType>()) {
        // Get basic information from tensorType
        auto order = ttg::getOrder(tensorType.getEncoding());
        auto CTALayout = ttg::getCTALayout(tensorType.getEncoding());
        auto elemType = tensorType.getElementType();

        // Get shape, layout and type of a slice
        auto sliceShape = tensorType.getShape();
        auto sharedLayout = ttg::SharedEncodingAttr::get(
            context, sliceShape, order, CTALayout, elemType);
        auto sliceType =
            RankedTensorType::get(sliceShape, elemType, sharedLayout);

        // Get shape, layout and type of the complete buffer
        SmallVector<int64_t> bufferShape(sliceShape.begin(), sliceShape.end());
        if (loadOp->getParentOfType<scf::ForOp>()) {
          bufferShape.insert(bufferShape.begin(), numStages);
        } else {
          bufferShape.insert(bufferShape.begin(), 1);
        }
        auto bufferType =
            RankedTensorType::get(bufferShape, elemType, sharedLayout);

        // Create InsertSliceOp
        builder.setAgentIdsFromOp(loadOp);
        builder.setInsertionPointAfter(loadOp);
        auto insertSliceOp = builder.createWithAgentIds<ttg::InsertSliceOp>(
            /*loc=*/loadOp.getLoc(), /*result=*/bufferType,
            /*src=*/loadOp.getPtr(), /*dst=*/buffer, /*index=*/pipelineIdx,
            /*mask=*/loadOp.getMask(), /*other=*/loadOp.getOther(),
            /*cache=*/loadOp.getCache(), /*evict=*/loadOp.getEvict(),
            /*isVolatile=*/loadOp.getIsVolatile(), /*axis=*/0);

        // Create ExtractSliceOp
        auto attr = [&](int val) { return builder.getI64IntegerAttr(val); };
        SmallVector<OpFoldResult> offsets = {pipelineIdx, attr(0), attr(0)};
        SmallVector<OpFoldResult> sizes = {attr(1), attr(sliceShape[0]),
                                           attr(sliceShape[1])};
        SmallVector<OpFoldResult> strides = {attr(1), attr(1), attr(1)};
        builder.setAgentIdsFromValueUsers(loadResult);
        builder.setInsertionPoint(c->dstOp);
        auto extractSliceOp = builder.createWithAgentIds<ttg::ExtractSliceOp>(
            loadOp.getLoc(), sliceType, buffer, offsets, sizes, strides);

        // Replace all uses of loadResult
        loadResult.replaceAllUsesWith(extractSliceOp.getResult());
        loadOp.erase();
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// agentDivision
//===----------------------------------------------------------------------===//

DenseMap<AgentId, Operation *> agentDivision(Operation *backbone) {
  // A general agent division in backbone could be:
  // *  If opWithRegion has results, e.g. scf.for, this opWithRegion will be
  //    splitted into several new operations, each agent has one, which
  //    has the part of results related to this agent. One agent could own
  //    all orginal results or none of them, but one result must belong to
  //    one and only one agent.
  // *  if opWithRegions doesn't have result. Simply split for every agent.
  // *  So does operands of opWithRegions
  // However, current backbones are all ForOps and IfOps. So we customize
  // the implementation.
  DenseMap<AgentId, Operation *> agentBackbone;
  backbone->walk([&](Operation *op) {
    auto ids = getAgentIds(op);
    if (op->getNumRegions() > 0 && ids.size() > 1) {
      // ForOp: change iterArgs and yield results
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        auto forOps = createForOpsForEachAgentId(forOp);
        if (op == backbone) {
          for (auto kv : forOps) {
            auto f = kv.second;
            auto id = getAgentIds(f.getOperation());
            assert(id.size() == 1 &&
                   "generated ForOp doesn't have one and only one agentId");
            agentBackbone[id.front()] = f.getOperation();
          }
        }
        forOp.erase();
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        // TODO: to be implemented
        llvm_unreachable("If Op is unsupported");
        auto ifOps = createIfOpsForEachAgentId(ifOp);
        assert(ifOps.size() > 0);
        if (op == backbone) {
          for (auto kv : ifOps) {
            auto i = kv.second;
            auto id = getAgentIds(i.getOperation());
            assert(id.size() == 1 &&
                   "generated IfOp doesn't have one and only one agentId");
            agentBackbone[id.front()] = i.getOperation();
          }
        }
      } else {
        llvm_unreachable("Unexpected Op with regions");
      }
    }
  });
  assert(agentBackbone.size() > 0 && "Agent division failed");
  return agentBackbone;
}

//===----------------------------------------------------------------------===//
// cloneBackboneForEachAgentId
//===----------------------------------------------------------------------===//

void cloneBackboneForEachAgentId(SmallVector<Operation *> &backbone) {
  SmallVector<Operation *> newBackBone;

  for (Operation *op : backbone) {
    auto loc = op->getLoc();
    OpBuilderWithAgentIds builder(op->getContext());
    builder.setInsertionPoint(op);
    // First, agent division
    DenseMap<AgentId, Operation *> agentBackbone = agentDivision(op);

    // Second, remove irrelavant Ops
    for (auto kv : agentBackbone) {
      SmallVector<Operation *> deleteOps;
      AgentId targetId = kv.first;
      Operation *newBackbone = kv.second;
      newBackbone->walk([&](Operation *subOp) {
        auto ids = getAgentIds(subOp);
        if (std::find(ids.begin(), ids.end(), targetId) == ids.end()) {
          deleteOps.push_back(subOp);
        }
      });
      for (auto it = deleteOps.rbegin(); it != deleteOps.rend(); ++it) {
        (*it)->erase();
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// WSPipelinePass
//===----------------------------------------------------------------------===//

struct WSPipelinePass : public TritonGPUWSPipelineBase<WSPipelinePass> {
  WSPipelinePass() = default;
  WSPipelinePass(int numStages, int numWarps, int computeCapability) {
    this->numStages = numStages;
    this->numWarps = numWarps;
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    auto mod = getOperation();
    if (!ttng::TritonNvidiaGPUDialect::getWSSupportedAttr(mod))
      return signalPassFailure();

    mod.walk([&](triton::FuncOp funcOp) {
      assert(funcOp.getBody().hasOneBlock() &&
             "FuncOp with more than one blocks is not supported");
      // Maintain all structures between funcOp and producer/consumer Op, for
      // example:
      /*  +-----------------------------------+
       *  | scf.for:                          |
       *  |   A = tt.load {agentId = 0}       |
       *  |   scf.for:                        |
       *  |     B = tt.load {agentId = 0}     |
       *  |     C = tt.dot A, B {agentId = 1} |
       *  +-----------------------------------+
       *                    ||
       *                   \||/
       *                    \/
       *  +-----------------------------------------+
       *  | token0 = create_token()                 |
       *  | token1 = create_token()                 |
       *  | buffer0 = alloc_buffer()                |
       *  | buffer1 = alloc_buffer()                |
       *  | if agent0:                              |
       *  |   scf.for:                              |
       *  |     producer_aquire token0              |
       *  |     buffer0 = tt.load           (load A)|
       *  |     producer_commit token0              |
       *  |     scf.for:                            |
       *  |       producer_aquire token1            |
       *  |       buffer1 = tt.load         (load B)|
       *  |       producer_commit token1            |
       *  | if agent1:                              |
       *  |   scf.for:                              |
       *  |     consumer_wait token0                |
       *  |     scf.for:                            |
       *  |       consumer_wait token1              |
       *  |       A = extract_slice buffer0         |
       *  |       B = extract_slice buffer1         |
       *  |       C = tt.dot A, B                   |
       *  |       consumer_arrive token1            |
       *  |     consumer_arrive token0              |
       *  +-----------------------------------------+
       */

      // First step: collect channels
      SmallVector<std::unique_ptr<Channel>> channelsOrigin;
      collectAsyncChannels(channelsOrigin, funcOp);
      SmallVector<Channel *> channels;
      for (const auto &c : channelsOrigin) {
        channels.push_back(c.get());
      }

      // cvgOp-channels map
      DenseMap<Operation *, SmallVector<Channel *>> map;
      reduceChannels(channels, map);

      // Prepare phase, getBackbone, appendPipelineIdxArgs
      SmallVector<Operation *> backbone = getBackbone(funcOp, channels);
      appendPipelineIdxArgs(backbone, numStages);

      // Create token, buffer and data tranfer between async agents
      DenseMap<Channel *, Value> tokenMap = createToken(map, funcOp, numStages);
      DenseMap<Channel *, Value> bufferMap =
          createBuffer(channels, funcOp, numStages);
      buildAsyncComm(map, tokenMap, bufferMap, numStages);

      // Clone backbone, remove irrelevant blockArgument for {forOp, ifOp}
      cloneBackboneForEachAgentId(backbone);

      // Specialize agent region
      SpecializeAgentRegion(funcOp);
    });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// createTritonNvidiaGPUWSPipelinePass
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUWSPipelinePass(int numStages, int numWarps,
                                          int computeCapability) {
  return std::make_unique<WSPipelinePass>(numStages, numWarps,
                                          computeCapability);
}
