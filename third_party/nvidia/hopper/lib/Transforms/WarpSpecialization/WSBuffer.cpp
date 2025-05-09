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

#define DEBUG_TYPE "nvgpu-ws-buffer"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static bool
enclosingAChannel(Operation *ctrlOp,
                  const DenseSet<Operation *> &regionsWithChannels) {
  for (auto *op : regionsWithChannels) {
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

// Update preOrderOps with a list of region Ops nested under ctrlOp that will
// need accumCnt. The list is in pre-order.
void getAccumCntsPreOrder(Operation *ctrlOp,
                          const DenseSet<Operation *> &regionsWithChannels,
                          SmallVector<Operation *> &preOrderOps) {
  ctrlOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    // This will walk ctrlOp itself.
    if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
      for (auto *op : regionsWithChannels) {
        if (subOp == op) {
          preOrderOps.push_back(subOp);
        }
      }
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(subOp)) {
      for (auto *op : regionsWithChannels) {
        if (subOp == op) {
          preOrderOps.push_back(subOp);
        }
      }
    }
  });
}

// Go through all the regions in opList and correctly add accumCnt. taskTopOps
// will be updated if it is replaced in the process.
Value updateAccumLoopCount(SmallVector<Operation *> &opList,
                           SmallVector<Operation *> &taskTopOps,
                           DenseSet<Operation *> &regionsWithChannels,
                           Value prevAccum);

// prevAccum is the accumCnt prior to the forOp. This function goes through
// the forOp and insert accumCnt when necessary.
scf::ForOp createNewLoopWrapper(scf::ForOp origForOp,
                                SmallVector<Operation *> &taskTopOps,
                                DenseSet<Operation *> &regionsWithChannels,
                                Value prevAccum);

scf::IfOp rewriteIfOp(scf::IfOp ifOp, SmallVector<Operation *> &taskTopOps,
                      DenseSet<Operation *> &regionsWithChannels,
                      Value prevAccum) {
  OpBuilderWithAsyncTaskIds ifBuilder(ifOp.getContext());
  ifBuilder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(ifOp));
  ifBuilder.setInsertionPoint(ifOp);

  // Calculate how many accumCnts we will need for this IfOp.
  unsigned numAccumCnts =
      getAccumCnts(ifOp.getOperation(), regionsWithChannels);
  // Add one i64 result value for each needed accumCnt.
  SmallVector<Type> newResultTypes(ifOp->getResultTypes());
  for (unsigned i = 0; i < numAccumCnts; ++i)
    newResultTypes.push_back(ifBuilder.getI64Type());

  LDBG("rewrite ifOp: add " << numAccumCnts << " accumCnts");
  assert(numAccumCnts > 0);
  // Create else block since we need to generate accumulated count for then and
  // else.
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

  // Create new Yield and erase original Yield.
  auto loc = ifOp.getLoc();
  auto updateYield = [&](scf::YieldOp yield, SmallVector<Value> &operands) {
    ifBuilder.setInsertionPoint(yield);
    ifBuilder.createWithAsyncTaskIds<scf::YieldOp>(loc, operands);
    yield.erase();
  };

  // Update regionsWithChannels withe newIfOp.
  auto tmpIter3 = std::find(regionsWithChannels.begin(),
                            regionsWithChannels.end(), ifOp.getOperation());
  if (tmpIter3 != regionsWithChannels.end()) {
    LDBG("rewrite ifOp: update regionsWithChannels "
         << ifOp.getOperation() << " --> " << newIfOp.getOperation());
    *tmpIter3 = newIfOp.getOperation();
  }

  // Go through region ops in the thenBlock. updateAccumLoopCount takes current
  // accumCnt value and returns the value at the end of the thenBlock.
  Value endAccum =
      updateAccumLoopCount(opList, taskTopOps, regionsWithChannels, prevAccum);

  SmallVector<Value> ifYieldOperands = newIfOp.thenYield().getOperands();

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
    // We need to differentiate channels in then region vs. in else region.
    // For now, only handle the case where channels are in then region.
    for (auto *op : opListElse)
      assert(!enclosingAChannel(op, regionsWithChannels));
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
  unsigned parentArgSize, parentTCnts = 0;
  SmallVector<Operation *> preOrderOpsOfParent;
  if (parentForOp) {
    parentArgSize = parentForOp.getBody()->getArguments().size();
    getAccumCntsPreOrder(parentForOp.getOperation(), regionsWithChannels,
                         preOrderOpsOfParent);
    parentTCnts = getAccumCnts(parentForOp.getOperation(), regionsWithChannels);
  }
  LDBG("rewrite ifOp: parentFor " << parentTCnts << " accumCnts");

  // For this IfOp, add accumCnts in preorder, starting with the IfOp itself
  // if it contains a channel. It then goes through the body of thenBlock, add
  // accumCnts for each region op of the thenBlock.
  // Check to see if newIfOp has channels directly in.
  bool hasDirectChannel = false;
  for (auto *op : regionsWithChannels) {
    if (newIfOp.getOperation() == op) {
      hasDirectChannel = true;
      break;
    }
  }
  if (hasDirectChannel) {
    // Set up value for thenYield and elseYield for accumCnt associated with
    // "newIfOp".
    auto *op = newIfOp.getOperation();
    Value endAccum, endAccumElse;
    if (parentForOp) {
      // Get corresponding argument of accumCnt for "op" in parentForOp.
      unsigned accumArgId =
          getAccumArgIdx(parentForOp, op, regionsWithChannels);
      LDBG("rewrite ifOp: ifOp itself parentArg " << parentArgSize << " "
                                                  << accumArgId);
      // All the accumCnts are at the end of argument list. When accumArgId
      // is parentTCnts - 1, the corresponding accumCnt will be the last
      // argument.
      Value arg = parentForOp.getBody()->getArgument(parentArgSize -
                                                     parentTCnts + accumArgId);
      // Either parent[accumCnt] + 1 or parent[accumCnt].
      Value one =
          ifBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
      endAccum = ifBuilder.createWithAsyncTaskIds<arith::AddIOp>(loc, arg, one);
      endAccumElse = arg;
    } else {
      endAccum =
          ifBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
      endAccumElse =
          elseBuilder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 64);
    }
    ifYieldOperands.push_back(endAccum);
    elseYieldOperands.push_back(endAccumElse);
    LLVM_DEBUG({
      LDBG("Update yieldOperands ");
      endAccum.dump();
      endAccumElse.dump();
    });
  }

  // Go through ops in thenBlock.
  for (auto *op : opList) {
    if (!enclosingAChannel(op, regionsWithChannels))
      continue;

    SmallVector<Operation *> preOrderOps;
    getAccumCntsPreOrder(op, regionsWithChannels, preOrderOps);
    auto numRes = op->getNumResults();
    unsigned tCnts = preOrderOps.size();
    LDBG("rewrite ifOp: thenBlock " << tCnts << " accumCnts");

    unsigned accumArgId;
    if (parentForOp && preOrderOps.size() > 0)
      // Find accumArgId for preOrderOps[0] in parentForOp.
      accumArgId =
          getAccumArgIdx(parentForOp, preOrderOps[0], regionsWithChannels);

    // Set up value for thenYield and elseYield for accumCnts nested under "op".
    // Each accumCnt nested under "op", it will have a corresponding argument in
    // this "IfOp". If "op" has tCnts, this "IfOp" will have the same number of
    // corresponding accumCnts, in the same order.
    for (unsigned i = 0; i < tCnts; ++i) {
      // Handle each accumCnt for "op".
      Value endAccum = op->getResult(numRes - tCnts + i);
      ifYieldOperands.push_back(endAccum);

      // Find the corresponding accumArgId from parentForOp.
      Value elseVal;
      if (parentForOp) {
        elseVal = parentForOp.getBody()->getArgument(
            parentArgSize - parentTCnts + accumArgId + i);
        LDBG("rewrite ifOp: elseYield parentArg " << parentArgSize << " "
                                                  << accumArgId << " " << i);
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
  // Update Yields.
  updateYield(newIfOp.thenYield(), ifYieldOperands);
  updateYield(newIfOp.elseYield(), elseYieldOperands);

  int resultIdx = 0;
  // Replace old if with the new one.
  for (auto result : ifOp.getResults()) {
    result.replaceAllUsesWith(newIfOp->getResult(resultIdx++));
  }
  ifOp.erase();
  return newIfOp;
}

// Handle the forOp given initial accumCnts.
scf::ForOp createNewLoop(scf::ForOp forOp, scf::ForOp &parentForOp,
                         SmallVector<Value> &initialAccums,
                         Value accumulatedLoopCount) {
  auto loc = forOp.getLoc();
  Block *body = forOp.getBody();

  OpBuilderWithAsyncTaskIds builder(forOp.getContext());
  builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(forOp));
  builder.setInsertionPoint(forOp);

  unsigned numAccumCnts = initialAccums.size();

  // Step 1: Append accumCnts as forOp arguments.
  for (unsigned i = 0; i < numAccumCnts; i++)
    body->insertArgument(body->getNumArguments(), builder.getI64Type(), loc);

  // Step 2: Add accumCnts to yieldOp.
  auto yieldOp = llvm::cast<scf::YieldOp>(body->getTerminator());
  builder.setInsertionPoint(yieldOp);
  unsigned tSize = body->getNumArguments();
  // Pass argument value as yield. This will be fixed in the caller.
  for (unsigned i = 0; i < numAccumCnts; i++)
    yieldOp->insertOperands(yieldOp.getNumOperands(),
                            {body->getArgument(tSize - numAccumCnts + i)});

  // Step 3: Create loop arguments for the new ForOp.
  SmallVector<Value> newLoopArgs;
  for (auto operand : forOp.getInitArgs())
    newLoopArgs.push_back(operand);
  builder.setInsertionPoint(forOp);
  for (unsigned i = 0; i < numAccumCnts; i++) {
    newLoopArgs.append({initialAccums[i]});
  }

  // Step 4: Create newForOp and take the region of the original forOp.
  auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      newLoopArgs);
  newForOp.getRegion().takeBody(forOp.getRegion());

  // Step 5: Replace forOp with newForOp.
  for (unsigned i = 0; i < forOp.getNumResults(); ++i)
    forOp.getResult(i).replaceAllUsesWith(newForOp.getResult(i));
  forOp.erase();

  return newForOp;
}

// Here we assume the source and destination ops are in the same region op.
// Go through channels, and get a set of region ops containing channels.
void collectRegionsWithChannels(const SmallVector<Channel *> &channels,
                                DenseSet<Operation *> &regionsWithChannels) {
  for (auto *ch : channels) {
    auto *dst = ch->getDstOp();
    auto *pOp = dst->getParentOp();
    if (!pOp)
      continue;
    if (auto forOp = dyn_cast<scf::ForOp>(pOp))
      regionsWithChannels.insert(pOp);
    if (auto ifOp = dyn_cast<scf::IfOp>(pOp))
      regionsWithChannels.insert(pOp);
  }
}

// Go through a list of operations in opList, recursively call into
// createNewLoopWrapper or rewriteIfOp.
Value updateAccumLoopCount(SmallVector<Operation *> &opList,
                           SmallVector<Operation *> &taskTopOps,
                           DenseSet<Operation *> &regionsWithChannels,
                           Value prevAccum) {
  DenseMap<Operation *, Operation *> oldToNew;
  for (Operation *op : opList) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      auto newForOp = createNewLoopWrapper(forOp, taskTopOps,
                                           regionsWithChannels, prevAccum);
      oldToNew[op] = newForOp.getOperation();
    } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (enclosingAChannel(ifOp.getOperation(), regionsWithChannels)) {
        auto newIfOp =
            rewriteIfOp(ifOp, taskTopOps, regionsWithChannels, prevAccum);
        oldToNew[op] = newIfOp.getOperation();

        // Update prevAccum to be result of the new IfOp.
        assert(newIfOp.getNumResults() >= 1);
        auto numRes = newIfOp.getNumResults();
        prevAccum =
            newIfOp.getResult(numRes - 1); // accumCnt is the last result.
      } else {
        // Still need to process nested ForOps in pre-order.
        SmallVector<scf::ForOp> innerForOps;
        ifOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
          if (auto forOp = dyn_cast<scf::ForOp>(subOp)) {
            innerForOps.push_back(forOp);
          }
        });
        for (auto innerFor : innerForOps) {
          auto newFor = createNewLoopWrapper(innerFor, taskTopOps,
                                             regionsWithChannels, prevAccum);
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

scf::ForOp createNewLoopWrapper(scf::ForOp origForOp,
                                SmallVector<Operation *> &taskTopOps,
                                DenseSet<Operation *> &regionsWithChannels,
                                Value prevAccum) {
  LLVM_DEBUG({
    LDBG("call createNewLoop on");
    origForOp.dump();
  });

  scf::ForOp parentForOp = origForOp->getParentOfType<scf::ForOp>();
  Value accumulatedLoopCount = prevAccum;

  unsigned pArgSize = 0, pCnts = 0, accumArgId = 0;
  if (parentForOp) {
    pArgSize = parentForOp.getBody()->getArguments().size();
    pCnts = getAccumCnts(parentForOp.getOperation(), regionsWithChannels);
  }

  SmallVector<Operation *> preOrderOps;
  getAccumCntsPreOrder(origForOp.getOperation(), regionsWithChannels,
                       preOrderOps);
  unsigned tCnts = preOrderOps.size();

  if (preOrderOps.size() > 0 && parentForOp) {
    // Find the accumArgId for preOrderOps[0] in parentForOp.
    accumArgId =
        getAccumArgIdx(parentForOp, preOrderOps[0], regionsWithChannels);
  }

  // Get initial value of accumCnts prior to the loop.
  SmallVector<Value> initialAccums;
  for (unsigned i = 0; i < tCnts; ++i) {
    // If there is an outer loop, use the corresponding argument value.
    Value startAccum;
    if (parentForOp)
      startAccum =
          parentForOp.getBody()->getArgument(pArgSize - pCnts + accumArgId + i);
    else {
      OpBuilderWithAsyncTaskIds builder(origForOp->getContext());
      builder.setAsynTaskIdsFromArray(getNestedAsyncTaskIds(origForOp));
      builder.setInsertionPoint(origForOp);
      auto loc = origForOp.getLoc();
      startAccum =
          builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 0, 64);
    }
    initialAccums.push_back(startAccum);
  }

  scf::ForOp newForOp = createNewLoop(origForOp, parentForOp, initialAccums,
                                      accumulatedLoopCount);
  LLVM_DEBUG({
    LDBG("after createNewLoop ");
    newForOp.dump();
  });

  // origForOp is erased in createNewLoop. Make sure taskTopOps is updated with
  // the newForOp.
  auto asyncTaskLoopForItr =
      std::find(taskTopOps.begin(), taskTopOps.end(), origForOp.getOperation());
  if (asyncTaskLoopForItr != taskTopOps.end()) {
    *asyncTaskLoopForItr = newForOp.getOperation();
  }
  auto tmpIter3 =
      std::find(regionsWithChannels.begin(), regionsWithChannels.end(),
                origForOp.getOperation());
  if (tmpIter3 != regionsWithChannels.end()) {
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
  Value endAccum =
      updateAccumLoopCount(opList, taskTopOps, regionsWithChannels, prevAccum);
  LLVM_DEBUG({
    LDBG("-- before replacing yieldOp ");
    newForOp.dump();
  });

  // Update yieldOp.
  Operation *yieldOp = newForOp.getBody()->getTerminator();
  unsigned tSize = newForOp.getBody()->getArguments().size();
  auto numAccumCnts = initialAccums.size();
  if (numAccumCnts == 0)
    return newForOp;

  // Start with the first accumCnt.
  accumArgId = tSize - numAccumCnts;
  LDBG("-- tSize, numAccumCnts, accumArgId " << tSize << " " << numAccumCnts
                                             << " " << accumArgId);

  // If there is a channel directly in forOp, it should be the first accumCnt.
  for (auto *op : regionsWithChannels) {
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

      // Make sure accumCnt = argValue + 1, increment by 1.
      // In createNewLoop, yieldOp yields the argument value directly, it is
      // fixed here.
      yieldOp->replaceUsesOfWith(arg, endAccum);
      ++accumArgId;
      break;
    }
  }
  LDBG("-- accumArgId after channels directly under " << accumArgId);

  // Handle the loop body. This order should align with the preorder that is
  // used for accumCnts.
  SmallVector<Operation *> dummy;
  for (auto *op : opList) {
    if (!enclosingAChannel(op, regionsWithChannels))
      continue;

    auto numRes = op->getNumResults();
    unsigned tCnts = getAccumCnts(op, regionsWithChannels);
    LDBG("-- numRes, tCnts, accumArgId " << numRes << " " << tCnts << " "
                                         << accumArgId);
    // Each accumCnt nested under "op", it will have a corresponding argument in
    // this "ForOp". If "op" has tCnts, this "ForOp" will have the same number
    // of corresponding accumCnts, in the same order.
    for (unsigned i = 0; i < tCnts; ++i) {
      Value arg = newForOp.getBody()->getArgument(accumArgId);
      Value endAccum = op->getResult(numRes - tCnts + i);
      LLVM_DEBUG({
        LDBG("-- replace use of arg with result " << numRes - tCnts + i);
        op->dump();
      });
      // In createNewLoop, yieldOp yields the argument value directly, it is
      // fixed here. Now, it will yield the accumCnt from the "op".
      yieldOp->replaceUsesOfWith(arg, endAccum);
      LLVM_DEBUG(yieldOp->dump());
      ++accumArgId;
    }
  }
  LLVM_DEBUG({
    LDBG("-- after all replacing ");
    newForOp.dump();
  });
  return newForOp;
}

void appendAccumCntsForOps(SmallVector<Operation *> &taskTopOps,
                           const SmallVector<Channel *> &channels,
                           DenseSet<Operation *> &regionsWithChannels) {

  SmallVector<Operation *> opList;
  for (auto &op : taskTopOps) {
    if (auto origIfOp = dyn_cast<scf::IfOp>(op)) {
      opList.push_back(op);
    }
    if (auto origForOp = dyn_cast<scf::ForOp>(op))
      opList.push_back(op);
  }
  Value tmpAccumLoopCount;
  // Go through all the regions in opList and correctly add accumCnt. taskTopOps
  // will be updated if it is replaced in the process.
  // tmpAccumLoopCount is the current accumCnt;
  updateAccumLoopCount(opList, taskTopOps, regionsWithChannels,
                       tmpAccumLoopCount);
}

// As an example, suppose we have 4 channels with the following control flow.
// We have 4 regions that immediately enclosing a channel:
//   ForA for channel D, IfA for channel A, ForB for channel B and
//   IfB for channel C
// All 4 regions are nested under ForA, thus we will have 4 accumCnts. And the
// counts are ordered in pre-order traversal.
// accumForA is the execution count for ForA, accumIfA is the execution count
// for IfA, etc. Barriers for channel A will use the corresponding value from
// the immedate enclosing region which is IfA.
// ForA (accumForA, accumIfA, accumForB, accumIfB)
//   IfA (accumIfA, accumForB)
//     Channel A --> uses ForA.arg[accumIfA] to calculate (bufIdx, phase)
//     ForB (accumForB)
//       Channel B --> uses ForB.arg[accumForB]
//   ThenYield ForA.arg[accumIfA] + 1, ForB.result[accumForB]
//   ElseYield ForA.arg[accumIfA], ForA.arg[accumForB]
//   ForC (accumForC, accumIfB)
//     IfB
//       Channel C --> uses ForC.arg[accumIfB]
//     ThenYield ForC.arg[accumIfB] + 1
//     ElseYield ForC.arg[accumIfB]
//   Channel D --> uses ForA.arg[accumForA]
// Design this as a generic module that goes through a function op, together
// with a list of channels. Separate the buffer-reuse logic.
void addAccumCountForRegion(triton::FuncOp funcOp,
                            const SmallVector<Channel *> &channels) {
  // Get top-level region ops that contain channels.
  SmallVector<Operation *> taskTopOps = getTaskTopRegion(funcOp, channels);

  // Get immediately enclosing region ops that contain channels.
  DenseSet<Operation *> regionsWithChannels;
  collectRegionsWithChannels(channels, regionsWithChannels);

  appendAccumCntsForOps(taskTopOps, channels, regionsWithChannels);
}

} // namespace mlir
