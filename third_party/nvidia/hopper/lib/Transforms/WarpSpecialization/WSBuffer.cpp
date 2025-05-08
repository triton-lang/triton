#include "CodePartitionUtility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <list>
#include <unordered_set>

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

#define DEBUG_TYPE "tritongpu-warp-spec-buffer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static unsigned getNumBuffersOrDefault(scf::ForOp forOp, unsigned numBuffers) {
  // Use the attribute attached to the loop if it exists otherwise use the
  // global control.
  if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
    return numBuffers;
  return mlir::cast<IntegerAttr>(
             forOp->getAttr(mlir::triton::kNumStagesAttrName))
      .getInt();
}

static bool enclosingAChannel(Operation *ctrlOp,
                              const DenseSet<Operation *> &opsWithChannels) {
  for (auto *op : opsWithChannels) {
    if (ctrlOp == op)
      return true;
    if (auto forOp = dyn_cast<scf::ForOp>(ctrlOp))
      if (enclosing(forOp, op))
        return true;
    if (auto ifOp = dyn_cast<scf::IfOp>(ctrlOp))
      if (enclosing(ifOp, op))
        return true;
  }
  return false;
}

unsigned getLoopDepth(Operation *op) {
  unsigned depth = 0;
  auto pOp = op->getParentOfType<scf::ForOp>();
  while (pOp) {
    ++depth;
    pOp = pOp->getParentOfType<scf::ForOp>();
  }
  return depth;
}

static unsigned getNumChannelsInOp(Operation *op,
                                   const SmallVector<Channel *> &channels,
                                   SmallVector<Channel *> &channelsInOp) {
  unsigned num = 0;
  for (auto *ch : channels) {
    // Get the immediate parent.
    auto srcParent = ch->getSrcOp()->getParentOp();
    auto dstParent = ch->getDstOp()->getParentOp();
    if (srcParent == op && dstParent == op)
      channelsInOp.push_back(ch);
  }
  return channelsInOp.size();
}

// Generate code
//   numSteps = ((upperBound - lowerBound) + forOpStep - 1) / forOpStep
Value getNumSteps(scf::ForOp forOp, OpBuilderWithAsyncTaskIds &builder) {
  auto loc = forOp.getLoc();
  // numSteps = ((upperBound - lowerBound) + forOpStep - 1) / forOpStep
  Value numSteps = builder.createWithAsyncTaskIds<arith::SubIOp>(
      loc, forOp.getUpperBound(), forOp.getLowerBound());
  numSteps = builder.createWithAsyncTaskIds<arith::AddIOp>(loc, numSteps,
                                                           forOp.getStep());
  if (forOp.getStep().getType() != builder.getI64Type())
    numSteps = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
        loc, builder.getI64Type(), numSteps);

  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
  numSteps = builder.createWithAsyncTaskIds<arith::SubIOp>(loc, numSteps, one);
  Value innerForStep = forOp.getStep();
  if (forOp.getStep().getType() != builder.getI64Type())
    innerForStep = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
        loc, builder.getI64Type(), forOp.getStep());
  numSteps = builder.createWithAsyncTaskIds<arith::DivUIOp>(loc, numSteps,
                                                            innerForStep);
  return numSteps;
}

// Update preOrderOps with a list
// of Ctrl Ops that will need accumCnt as arguments/results of CtrlOp.
void getAccumCntsPreOrder(Operation *ctrlOp,
                          const DenseSet<Operation *> &opsWithChannels,
                          SmallVector<Operation *> &preOrderOps) {
  for (auto *op : opsWithChannels) {
    LDBG("getAccumCntsPreOrder: " << ctrlOp << " opsWithChannels " << op);
  }
  ctrlOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    // This will walk ctrlOp.
    if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
      LDBG("-- getAccumCntsPreOrder: walk forOp " << subOp);
      LDBG("-- opsWithChannels: " << opsWithChannels.size() << " "
                                  << opsWithChannels.count(subOp));
      for (auto *op : opsWithChannels) {
        if (subOp == op) {
          LDBG("-- opsWithChannels push to result");
          preOrderOps.push_back(subOp);
        }
      }
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(subOp)) {
      LDBG("-- getAccumCntsPreOrder: walk IfOp " << subOp);
      LDBG("-- opsWithChannels: " << opsWithChannels.size() << " "
                                  << opsWithChannels.count(subOp));
      for (auto *op : opsWithChannels) {
        if (subOp == op) {
          preOrderOps.push_back(subOp);
          LDBG("-- opsWithChannels push to result");
        }
      }
    }
  });
  LDBG("-- getAccumCntsPreOrder: " << ctrlOp << " size " << preOrderOps.size());
}

static bool
needAccumulatedLoopCnt(scf::IfOp ifOp,
                       const DenseSet<Operation *> &opsWithChannels) {
  return enclosingAChannel(ifOp.getOperation(), opsWithChannels);
}

// op is up-to-date (i.e will be updated when a control op is re-written).
Value updateAccumLoopCount(SmallVector<Operation *> &opList,
                           unsigned numBuffers,
                           SmallVector<Operation *> &taskTopOps,
                           DenseSet<Operation *> &opsWithChannels,
                           Value prevAccum);

scf::ForOp createNewLoopWrapper(scf::ForOp origForOp, unsigned numBuffers,
                                SmallVector<Operation *> &taskTopOps,
                                DenseSet<Operation *> &opsWithChannels,
                                Value prevAccum);

// For certain cases, we need to add an additional output for
// IfOp to track the accumulatedLoopCount, we may need to add
// a corresponding elseBlock with yieldOp.
scf::IfOp rewriteIfOp(scf::IfOp ifOp, unsigned numBuffers,
                      SmallVector<Operation *> &taskTopOps,
                      DenseSet<Operation *> &opsWithChannels, Value prevAccum) {
  LLVM_DEBUG({
    LDBG("rewrite ifOp for smem sharing ");
    ifOp.dump();
  });

  OpBuilderWithAsyncTaskIds ifBuilder(ifOp.getContext());
  ifBuilder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(ifOp));
  ifBuilder.setInsertionPoint(ifOp);

  unsigned numAccumCnts = getAccumCnts(ifOp.getOperation(), opsWithChannels);

  SmallVector<Type> newResultTypes(ifOp->getResultTypes());
  for (unsigned i = 0; i < numAccumCnts; ++i)
    newResultTypes.push_back(ifBuilder.getI64Type());
  LDBG("rewrite ifOp: add " << numAccumCnts << " accumCnts");
  assert(numAccumCnts > 0);
  // Create else block if we need to generate accumulated loop count.
  auto newIfOp = ifBuilder.createWithAsyncTaskIds<scf::IfOp>(
      ifOp.getLoc(), newResultTypes, ifOp.getCondition(), true, true);

  // Move the existing blocks to the new if.
  newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());

  ifBuilder.setInsertionPointToEnd(newIfOp.thenBlock());
  SmallVector<Operation *> opList;
  for (Operation &op : newIfOp.thenBlock()->getOperations()) {
    if (auto tOp = dyn_cast<scf::ForOp>(&op))
      opList.push_back(&op);
    if (auto tOp = dyn_cast<scf::IfOp>(&op))
      opList.push_back(&op);
  }

  // Update yields
  auto loc = ifOp.getLoc();
  auto updateYield = [&](scf::YieldOp yield, SmallVector<Value> &operands) {
    ifBuilder.setInsertionPoint(yield);
    ifBuilder.createWithAsyncTaskIds<scf::YieldOp>(loc, operands);
    yield.erase();
  };

  // Update opsWithChannels now that newIfOp takes over the body.
  auto tmpIter3 = std::find(opsWithChannels.begin(), opsWithChannels.end(),
                            ifOp.getOperation());
  if (tmpIter3 != opsWithChannels.end()) {
    LDBG("rewrite ifOp: update opsWithChannels "
         << ifOp.getOperation() << " --> " << newIfOp.getOperation());
    *tmpIter3 = newIfOp.getOperation();
  }

  // Add one more operand to then Yield.
  Value endAccum = updateAccumLoopCount(opList, numBuffers, taskTopOps,
                                        opsWithChannels, prevAccum);

  SmallVector<Value> ifYieldOperands = newIfOp.thenYield().getOperands();

  // Handle elseRegion of the IfOp.
  if (ifOp.elseBlock()) {
    ifBuilder.setInsertionPointToEnd(newIfOp.elseBlock());
    newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());
    SmallVector<Operation *> opListElse;
    for (Operation &op : newIfOp.elseBlock()->getOperations()) {
      if (auto tOp = dyn_cast<scf::ForOp>(&op))
        opListElse.push_back(&op);
      if (auto tOp = dyn_cast<scf::IfOp>(&op))
        opListElse.push_back(&op);
    }
    {
      // We need to differentiate channels in then region vs. in else region.
      // For now, only handle the case where channels are in then region.
      for (auto *op : opListElse)
        assert(!enclosingAChannel(op, opsWithChannels));
    }
  } else {
    // Create an empty yield
    auto yieldOp =
        newIfOp.getElseBodyBuilder().create<scf::YieldOp>(ifOp.getLoc());
  }

  SmallVector<Value> elseYieldOperands = newIfOp.elseYield().getOperands();
  OpBuilderWithAsyncTaskIds elseBuilder(ifOp.getContext());
  elseBuilder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(ifOp));
  elseBuilder.setInsertionPoint(newIfOp.elseYield());
  ifBuilder.setInsertionPoint(newIfOp.thenYield());

  auto parentForOp = newIfOp->getParentOfType<scf::ForOp>();
  unsigned tSize, parentTCnts = 0;
  SmallVector<Operation *> preOrderOpsOfParent;
  if (parentForOp) {
    tSize = parentForOp.getBody()->getArguments().size();
    getAccumCntsPreOrder(parentForOp.getOperation(), opsWithChannels,
                         preOrderOpsOfParent);
    parentTCnts = getAccumCnts(parentForOp.getOperation(), opsWithChannels);
  }
  LDBG("rewrite ifOp: parentFor " << parentTCnts << " accumCnts");

  //  Update both ifYieldOperands and elseYieldOperands.
  //  See below for an example of how to update yieldOp of IfA and IfB.
  //  ForA (accumForA, accumIfA, accumForB, accumIfB)
  //    IfA (accumIfA, accumForB)
  //      Channel A --> uses ForA.arg[accumIfA] to calculate (bufIdx, phase)
  //      ForB (accumForB)
  //        Channel B --> uses ForB.arg[accumForB]
  //    ThenYield ForA.arg[accumIfA] + 1, ForB.res[accumForB]
  //    ElseYield ForA.arg[accumIfA], ForA.arg[accumForB]
  //    ForC (accumForC, accumIfB)
  //      IfB
  //        Channel C --> uses ForC.arg[accumIfB]
  //      ThenYield ForC.arg[accumIfB] + 1
  //      ElseYield ForC.arg[accumIfB]
  //    Channel D --> uses ForA.arg[accumForA]
  //  Check to see if ifOp has a channel directly under. Both IfA and IfB fall
  //  into this case.
  for (auto *op : opsWithChannels) {
    if (newIfOp.getOperation() == op) {
      // Find enclosing Forop, use arg + 1; If no enclosing forOp, use 0.
      // arg is parentForOp.arg[newIfOp]
      Value endAccum, endAccumElse;
      if (parentForOp) {
        // Get corresponding argument of accumCnt: forOp.accumCnt[ifOp].
        unsigned accumArgId = getAccumArgIdx(parentForOp, op, opsWithChannels);
        LDBG("rewrite ifOp: ifOp itself parentArg " << tSize << " "
                                                    << accumArgId);
        Value arg = parentForOp.getBody()->getArgument(tSize - parentTCnts +
                                                       accumArgId);
        Value one =
            ifBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
        endAccum =
            ifBuilder.createWithAsyncTaskIds<arith::AddIOp>(loc, arg, one);
        endAccumElse = arg;
      } else {
        endAccum =
            ifBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 64);
        endAccumElse = elseBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(
            loc, 0, 64);
      }
      ifYieldOperands.push_back(endAccum);
      elseYieldOperands.push_back(endAccumElse);
      LLVM_DEBUG({
        LDBG("Update yieldOperands ");
        endAccum.dump();
        endAccumElse.dump();
      });
      break;
    }
  }
  // Go through ops in thenBlock, which should be preorder.
  for (auto *op : opList) {
    if (!enclosingAChannel(op, opsWithChannels))
      continue;
    // Push op.accumCnts as ifYield, push parentForOp.accumCnts[...] as
    // elseYield.
    SmallVector<Operation *> preOrderOps;
    getAccumCntsPreOrder(op, opsWithChannels, preOrderOps);
    auto numRes = op->getNumResults();
    unsigned tCnts = preOrderOps.size();
    LDBG("rewrite ifOp: thenBlock " << tCnts << " accumCnts");
    unsigned accumArgId;
    if (parentForOp && preOrderOps.size() > 0)
      // arg is parentForOp.arg[preOrderOps[0]]
      accumArgId = getAccumArgIdx(parentForOp, preOrderOps[0], opsWithChannels);
    for (unsigned i = 0; i < tCnts; ++i) {
      Value endAccum = op->getResult(numRes - tCnts + i);
      ifYieldOperands.push_back(endAccum);
      // Find the corresponding accumArgId from parentForOp.
      Value elseVal;
      if (parentForOp) {
        elseVal = parentForOp.getBody()->getArgument(tSize - parentTCnts +
                                                     accumArgId + i);
        LDBG("rewrite ifOp: elseYield parentArg " << tSize << " " << accumArgId
                                                  << " " << i);
      } else
        elseVal = elseBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(
            loc, 0, 64);
      elseYieldOperands.push_back(elseVal);
      LLVM_DEBUG({
        LDBG("Update yieldOperands ");
        endAccum.dump();
        elseVal.dump();
      });
    }
  }
  // Add one more operand to else Yield.
  updateYield(newIfOp.thenYield(), ifYieldOperands);
  //}
  updateYield(newIfOp.elseYield(), elseYieldOperands);

  int resultIdx = 0;
  // Replace old if with the new one.
  for (auto result : ifOp.getResults()) {
    result.replaceAllUsesWith(newIfOp->getResult(resultIdx++));
  }

  ifOp.erase();
  return newIfOp;
}

scf::ForOp createNewLoop(scf::ForOp forOp, int numBuffers,
                         scf::ForOp &parentForOp,
                         SmallVector<Value> &initialAccum,
                         Value accumulatedLoopCount) {
  auto loc = forOp.getLoc();
  Block *body = forOp.getBody();

  OpBuilderWithAsyncTaskIds builder(forOp.getContext());
  builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(forOp));
  builder.setInsertionPoint(forOp);

  unsigned numAccumCnts = initialAccum.size();

  // Step 1: Append accumCnts as forOp arguments.
  for (unsigned i = 0; i < numAccumCnts; i++)
    body->insertArgument(body->getNumArguments(), builder.getI64Type(), loc);
  Value tmpAccumLoopCount;
  auto yieldOp = llvm::cast<scf::YieldOp>(body->getTerminator());
  builder.setInsertionPoint(yieldOp);
  // Step 3: Add accumCnts to yieldOp.
  unsigned tSize = body->getNumArguments();
  // Will be fixed in the caller.
  for (unsigned i = 0; i < numAccumCnts; i++)
    yieldOp->insertOperands(yieldOp.getNumOperands(),
                            {body->getArgument(tSize - numAccumCnts + i)});

  // Step 4: Create loop arguments for the new ForOp.
  SmallVector<Value> newLoopArgs;
  for (auto operand : forOp.getInitArgs())
    newLoopArgs.push_back(operand);

  builder.setInsertionPoint(forOp);
  Value initCntIdx;
  for (unsigned i = 0; i < numAccumCnts; i++) {
    initCntIdx = initialAccum[i];
    newLoopArgs.append({initCntIdx /*, initPhase, initBufferIdx*/});
  }

  // Step 5: Create newForOp and take the region of the original forOp.
  auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      newLoopArgs);
  if (forOp->getAttr("tt.loop_schedule"))
    newForOp->setAttr("tt.loop_schedule", forOp->getAttr("tt.loop_schedule"));
  newForOp.getRegion().takeBody(forOp.getRegion());

  // Step 6: Replace forOp with newForOp.
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  forOp.erase();

  return newForOp;
}

// Here we assume the source and destination ops are in the same region op.
// Go through channels, and get a set of region ops containing channels.
void collectRegionsWithChannels(const SmallVector<Channel *> &channels,
                                DenseSet<Operation *> &opsWithChannels) {
  for (auto *ch : channels) {
    auto *dst = ch->getDstOp();
    auto *pOp = dst->getParentOp();
    if (!pOp)
      continue;
    if (auto forOp = dyn_cast<scf::ForOp>(pOp))
      opsWithChannels.insert(pOp);
    if (auto ifOp = dyn_cast<scf::IfOp>(pOp))
      opsWithChannels.insert(pOp);
  }
}

// Go through a list of operations under one scope.
Value updateAccumLoopCount(SmallVector<Operation *> &opList,
                           unsigned numBuffers,
                           SmallVector<Operation *> &taskTopOps,
                           DenseSet<Operation *> &opsWithChannels,
                           Value prevAccum) {
  DenseMap<Operation *, Operation *> oldToNew;
  for (Operation *op : opList) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto newForOp = createNewLoopWrapper(forOp, numBuffers, taskTopOps,
                                           opsWithChannels, prevAccum);
      oldToNew[op] = newForOp.getOperation();
      // Update prevAccum to be after the loop.
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (needAccumulatedLoopCnt(ifOp, opsWithChannels)) {
        auto newIfOp = rewriteIfOp(ifOp, numBuffers, taskTopOps,
                                   opsWithChannels, prevAccum);
        oldToNew[op] = newIfOp.getOperation();
        // update prevAccum to be result of the new IfOp.
        assert(newIfOp.getNumResults() >= 1);
        auto numRes = newIfOp.getNumResults();
        LDBG("update prevAccum with result from IfOp");
        prevAccum = newIfOp.getResult(numRes - 1); // last result
      } else {
        // Still need to process ForOps in pre-order.
        SmallVector<scf::ForOp> innerForOps;
        ifOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
          if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
            innerForOps.push_back(forOp);
          }
        });
        for (auto innerFor : innerForOps) {
          auto newFor = createNewLoopWrapper(innerFor, numBuffers, taskTopOps,
                                             opsWithChannels, prevAccum);
          oldToNew[innerFor.getOperation()] = newFor.getOperation();
        }
      }
    }
  }
  for (unsigned i = 0; i < opList.size(); i++) {
    auto *oldOp = opList[i];
    if (oldToNew.find(oldOp) != oldToNew.end())
      opList[i] = oldToNew[oldOp];
  }
  return prevAccum;
}

scf::ForOp createNewLoopWrapper(scf::ForOp origForOp, unsigned numBuffers,
                                SmallVector<Operation *> &taskTopOps,
                                DenseSet<Operation *> &opsWithChannels,
                                Value prevAccum) {
  LLVM_DEBUG({
    LDBG("call createNewLoop on");
    origForOp.dump();
  });

  scf::ForOp parentForOp = origForOp->getParentOfType<scf::ForOp>();
  scf::ForOp newForOp;
  // for(...) -> for(..., phase, bufferIdx)
  unsigned loopNumBuffers = getNumBuffersOrDefault(origForOp, numBuffers);

  Value accumulatedLoopCount = prevAccum;
  // ForOp will have a list of accumCnts, starting with
  // argument value.
  // Get initial value of accumCnts prior to the loop.
  SmallVector<Value> initialAccum;
  unsigned tSize = 0, tNum = 0, accumArgId = 0;
  if (parentForOp) {
    tSize = parentForOp.getBody()->getArguments().size();
    tNum = getAccumCnts(parentForOp.getOperation(), opsWithChannels);
    LDBG("-- has parentForOp");
  }
  SmallVector<Operation *> preOrderOps;
  getAccumCntsPreOrder(origForOp.getOperation(), opsWithChannels, preOrderOps);
  unsigned tCnts = preOrderOps.size();
  if (preOrderOps.size() > 0 && parentForOp) {
    // Check for accumArgId in parentForOp for the first preOrderOp of the
    // ForOp.
    accumArgId = getAccumArgIdx(parentForOp, preOrderOps[0], opsWithChannels);
  }
  for (unsigned i = 0; i < tCnts; ++i) {
    Value startAccum;
    if (parentForOp)
      startAccum =
          parentForOp.getBody()->getArgument(tSize - tNum + accumArgId + i);
    else {
      OpBuilderWithAsyncTaskIds builder(origForOp->getContext());
      builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(origForOp));
      builder.setInsertionPoint(origForOp);
      auto loc = origForOp.getLoc();
      startAccum =
          builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 64);
    }
    initialAccum.push_back(startAccum);
  }

  newForOp = createNewLoop(origForOp, loopNumBuffers, parentForOp, initialAccum,
                           accumulatedLoopCount);
  LLVM_DEBUG({
    LDBG("after createNewLoop ");
    newForOp.dump();
  });
  // origForOp is erased in createNewLoop. If origForOp is a top operation
  // (i.e in taskTopOps), make sure taskTopOps is updated with the newForOp.
  auto asyncTaskLoopForItr =
      std::find(taskTopOps.begin(), taskTopOps.end(), origForOp.getOperation());
  if (asyncTaskLoopForItr != taskTopOps.end()) {
    // Update taskTopOps.
    *asyncTaskLoopForItr = newForOp.getOperation();
  }

  // opsWithChannels
  auto tmpIter3 = std::find(opsWithChannels.begin(), opsWithChannels.end(),
                            origForOp.getOperation());
  if (tmpIter3 != opsWithChannels.end()) {
    LDBG("createNewLoopWrapper: update opsWithChannels "
         << origForOp.getOperation() << " --> " << newForOp.getOperation());
    *tmpIter3 = newForOp.getOperation();
  }

  // Handle ops in loop body, only IfOps and ForOps.
  SmallVector<Operation *> opList;
  for (Operation &op : newForOp.getBody()->without_terminator()) {
    if (auto tOp = dyn_cast<scf::ForOp>(&op))
      opList.push_back(&op);
    if (auto tOp = dyn_cast<scf::IfOp>(&op))
      opList.push_back(&op);
  }
  Value endAccum = updateAccumLoopCount(opList, numBuffers, taskTopOps,
                                        opsWithChannels, prevAccum);
  LLVM_DEBUG({
    LDBG("-- before replacing yieldOp ");
    newForOp.dump();
  });

  // Update yieldOp.
  Operation *yieldOp = newForOp.getBody()->getTerminator();
  // ForA (accumForA, accumIfA, accumForB, accumIfB)
  //   IfA (accumIfA, accumForB)
  //     Channel A --> uses ForA.arg[accumIfA] to calculate (bufIdx, phase)
  //     ForB (accumForB)
  //       Channel B --> uses ForB.arg[accumForB]
  //   ThenYield ForA.arg[accumIfA] + 1, ForB.res[accumForB]
  //   ElseYield ForA.arg[accumIfA], ForA.arg[accumForB]
  //   ForC (accumForC, accumIfB)
  //     IfB
  //       Channel C --> uses ForC.arg[accumIfB]
  //     ThenYield ForC.arg[accumIfB] + 1
  //     ElseYield ForC.arg[accumIfB]
  //   Channel D --> uses ForA.arg[accumForA]
  // Yield ForA.arg[accumForA]+1, IfA.res[accumIfA], IfA.res[accumForB],
  // ForC.res[accumIfB]
  tSize = newForOp.getBody()->getArguments().size();
  auto numAccumCnts = initialAccum.size();
  if (numAccumCnts == 0)
    return newForOp;
  accumArgId = tSize - numAccumCnts; // first accumCnt: ForA.arg[accumForA]
  LDBG("-- tSize, numAccumCnts, accumArgId " << tSize << " " << numAccumCnts
                                             << " " << accumArgId);

  // If there is a channel directly in forOp, yield ForA.arg[accumForA]+1.
  for (auto *op : opsWithChannels) {
    if (newForOp.getOperation() == op) {
      Value arg = newForOp.getBody()->getArgument(accumArgId);
      OpBuilderWithAsyncTaskIds builder(newForOp->getContext());
      builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(newForOp));
      builder.setInsertionPoint(yieldOp);
      auto loc = newForOp.getLoc();
      Value one =
          builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
      Value endAccum =
          builder.createWithAsyncTaskIds<arith::AddIOp>(loc, arg, one);
      yieldOp->replaceUsesOfWith(arg, endAccum);
      ++accumArgId;
      break;
    }
  }
  LDBG("-- accumArgId after channels directly under " << accumArgId);
  // This order should align with the preorder that is used for accumCnts.
  SmallVector<Operation *> dummy;
  for (auto *op : opList) {
    if (!enclosingAChannel(op, opsWithChannels))
      continue;
    auto numRes = op->getNumResults();
    unsigned tCnts = getAccumCnts(op, opsWithChannels);
    LDBG("-- numRes, tCnts, accumArgId " << numRes << " " << tCnts << " "
                                         << accumArgId);
    for (unsigned i = 0; i < tCnts; ++i) {
      Value arg = newForOp.getBody()->getArgument(accumArgId);
      Value endAccum = op->getResult(numRes - tCnts + i);
      LLVM_DEBUG({
        LDBG("-- replace use of arg with result " << numRes - tCnts + i);
        op->dump();
      });
      yieldOp->replaceUsesOfWith(arg, endAccum);
      LLVM_DEBUG(yieldOp->dump());
      ++accumArgId;
    }
  }
  LLVM_DEBUG({
    LDBG("-- after all replacing ");
    newForOp.dump();
  });
  //}
  return newForOp;
}

// Design this as a generic module that goes through a function op, together
// with a list of channels. Separate the buffer-reuse logic.
void addAccumCountForRegion(triton::FuncOp funcOp,
                            const SmallVector<Channel *> &channels) {
  SmallVector<Operation *> taskTopOps = getTaskTopRegion(funcOp, channels);

  DenseSet<Operation *> regionsWithChannels;
  collectRegionsWithChannels(channels, regionsWithChannels);

  SmallVector<Operation *> opList;
  for (auto &op : taskTopOps) {
    if (auto origIfOp = dyn_cast<scf::IfOp>(op)) {
      opList.push_back(op);
    }
    if (auto origForOp = dyn_cast<scf::ForOp>(op))
      opList.push_back(op);
  }
  Value tmpAccumLoopCount;
  updateAccumLoopCount(opList, 0 /*numBuffers*/, taskTopOps,
                       regionsWithChannels, tmpAccumLoopCount);
}

// TODO: Support for mapToRepresenting and opsWithBufferReuse.
Value appendBufferIdxArgs(SmallVector<Operation *> &taskTopOps,
                          unsigned numBuffers,
                          const SmallVector<Channel *> &channels,
                          DenseSet<Operation *> &opsWithChannels) {

  SmallVector<Operation *> opList;
  for (auto &op : taskTopOps) {
    if (auto origIfOp = dyn_cast<scf::IfOp>(op)) {
      opList.push_back(op);
    }
    if (auto origForOp = dyn_cast<scf::ForOp>(op))
      opList.push_back(op);
  }
  Value tmpAccumLoopCount;
  updateAccumLoopCount(opList, numBuffers, taskTopOps, opsWithChannels,
                       tmpAccumLoopCount);

  return tmpAccumLoopCount;
}

} // namespace mlir
