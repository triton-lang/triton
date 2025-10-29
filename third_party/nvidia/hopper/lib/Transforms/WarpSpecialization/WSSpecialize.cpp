#include "CodePartitionUtility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
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

#define DEBUG_TYPE "nvgpu-ws-specialize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

Operation *SpecializeOp(Operation *op, IRMapping &mapping,
                        OpBuilderWithAsyncTaskIds &builder,
                        AsyncTaskId asyncTaskId);

unsigned scanRegUsage(Block *block, AsyncTaskId asyncTaskId,
                      unsigned requestedRegisters) {
  assert(asyncTaskId != 0 && "producer group should not request registers");
  // TODO: scan ops to estimate register usage
  return requestedRegisters == 0 ? 232 : requestedRegisters;
}

// Collect argument indices that are used by the specific taskId.
static SmallVector<unsigned> collectBlockArgsForTask(scf::ForOp forOp,
                                                     int asyncTaskId) {

  // Collect argument indices that can be reached along the definition chain.
  SetVector<unsigned> argIndices;
  std::function<void(scf::ForOp, Value, unsigned)> dfs =
      [&](scf::ForOp nestedForOp, Value arg, unsigned argIdx) {
        for (auto user : arg.getUsers()) {
          // Skip ops that are not in the same async task
          if (!hasAsyncTaskId(user, asyncTaskId))
            continue;

          if (isa<scf::YieldOp>(user)) {
            if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp())) {
              // For block arguments, we need to check the initial value as
              // well.
              if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
                auto initArg =
                    nestedForOp.getInitArgs()[blockArg.getArgNumber() - 1];
                if (Operation *def = initArg.getDefiningOp()) {
                  if (hasAsyncTaskId(def, asyncTaskId)) {
                    argIndices.insert(argIdx);
                    return;
                  }
                } else {
                  llvm_unreachable("Initial value should have a defining op");
                }
              }
            }

            // Skip control flow ops that are shared by all async tasks
            continue;
          }

          // If use is the initial value of ForOp argument.
          if (auto userFor = dyn_cast<scf::ForOp>(user)) {
            // For block arguments, we need to check the initial value as well.
            if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
              auto initArg =
                  nestedForOp.getInitArgs()[blockArg.getArgNumber() - 1];
              if (Operation *def = initArg.getDefiningOp()) {
                if (hasAsyncTaskId(def, asyncTaskId)) {
                  argIndices.insert(argIdx);
                  return;
                }
              } else {
                // Recursive search the nested loop for the real users.
                // find corresponding arg of userFor
                Value userArg;
                for (auto item : llvm::enumerate(userFor.getInitArgs())) {
                  if (item.value() == arg) {
                    userArg = userFor.getRegionIterArg(item.index());
                    break;
                  }
                }
                if (userArg) {
                  dfs(userFor, userArg, argIdx);
                }
              }
            }
            // Skip control flow ops that are shared by all async tasks
            continue;
          }

          // Found a real user, the arg is needed
          if (user->getNumRegions() == 0) {
            argIndices.insert(argIdx);
            return;
          }

          // Iterate through all regions of the user operation
          for (auto &region : user->getRegions()) {
            for (auto regionArg : region.getArguments()) {
              if (arg == regionArg)
                dfs(nestedForOp, regionArg, argIdx);
            }
          }
        }
      };

  // check dependency with DFS traversal for loop args and results.
  mlir::Block &block = forOp.getRegion().front();
  for (unsigned i = forOp.getNumInductionVars(); i < block.getNumArguments();
       ++i) {
    auto arg = block.getArgument(i);
    dfs(forOp, arg, i - forOp.getNumInductionVars());
  }
  for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
    auto result = forOp->getResult(i);
    dfs(forOp, result, i);
  }

  SmallVector<unsigned> args(argIndices.begin(), argIndices.end());
  llvm::sort(args);
  return args;
}

Operation *SpecializeIfOp(scf::IfOp ifOp, IRMapping &mapping,
                          OpBuilderWithAsyncTaskIds &builder,
                          AsyncTaskId asyncTaskId) {
  LLVM_DEBUG({
    LDBG("specialize ifOp ");
    ifOp.dump();
  });

  // It is possible that we need to reduce the results. One example
  // is that the defining op for the yield operation is not for this
  // taskId and the defining op is not specialized, thus we should
  // remove the result.
  // We need to update the result types correctly here.
  unsigned resultIdx = 0;
  SmallVector<unsigned> keptResultVec;
  if (!ifOp->getResultTypes().empty()) {
    for (Value yieldV : ifOp.thenYield().getOperands()) {
      // Check the defining op for the corresponding result.
      if (Operation *def = yieldV.getDefiningOp()) {
        bool hasTaskId = hasAsyncTaskId(def, asyncTaskId);
        if (hasTaskId) {
          keptResultVec.push_back(resultIdx);
        }
      } else {
        assert(isa<BlockArgument>(yieldV) && "Unexpected yield value");
        auto bbArg = cast<BlockArgument>(yieldV);
        // Find transitive defining op for the block arg
        Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
        if (auto forOp = dyn_cast<scf::ForOp>(bbAargOwner)) {
          // track initial value
          auto initArg = forOp.getInitArgs()[bbArg.getArgNumber() - 1];
          if (Operation *def = initArg.getDefiningOp()) {
            if (hasAsyncTaskId(def, asyncTaskId))
              keptResultVec.push_back(resultIdx);
          } else {
            llvm_unreachable("Initial value should have a defining op");
          }
        } else {
          llvm_unreachable("Unexpected block argument owner");
        }
      }
      ++resultIdx;
    }
  }

  SmallVector<Type> newResultTypes;
  for (auto idx : keptResultVec) {
    newResultTypes.push_back(ifOp->getResultTypes()[idx]);
  }
  auto newIfOp = builder.createWithAsyncTaskIds<scf::IfOp>(
      ifOp.getLoc(), newResultTypes, mapping.lookup(ifOp.getCondition()), true,
      ifOp.elseBlock());

  OpBuilderWithAsyncTaskIds ifBuilder(ifOp.getContext());
  ifBuilder.setAsynTaskIdsFromArray({asyncTaskId});

  // Handle thenRegion of this IfOp.
  ifBuilder.setInsertionPointToEnd(newIfOp.thenBlock());
  for (Operation &thenOp : ifOp.thenBlock()->getOperations()) {
    SpecializeOp(&thenOp, mapping, ifBuilder, asyncTaskId);
  }

  // Update yields
  auto updateYield = [&](scf::YieldOp yield, SmallVector<Value> &operands) {
    ifBuilder.setInsertionPoint(yield);
    ifBuilder.createWithAsyncTaskIds<scf::YieldOp>(yield.getLoc(), operands);
    yield.erase();
  };
  if (keptResultVec.size() < ifOp->getResultTypes().size()) {
    SmallVector<Value> ifYieldOperands;
    for (auto idx : keptResultVec) {
      ifYieldOperands.push_back(newIfOp.thenYield().getOperand(idx));
    }
    updateYield(newIfOp.thenYield(), ifYieldOperands);
  }

  // Handle elseRegion of the IfOp.
  if (ifOp.elseBlock()) {
    ifBuilder.setInsertionPointToEnd(newIfOp.elseBlock());
    for (Operation &elseOp : ifOp.elseBlock()->getOperations()) {
      SpecializeOp(&elseOp, mapping, ifBuilder, asyncTaskId);
    }
    if (keptResultVec.size() < ifOp->getResultTypes().size()) {
      SmallVector<Value> elseYieldOperands;
      for (auto idx : keptResultVec) {
        elseYieldOperands.push_back(newIfOp.elseYield().getOperand(idx));
      }
      updateYield(newIfOp.elseYield(), elseYieldOperands);
    }
  }

  unsigned newResIdx = 0;
  for (auto idx : keptResultVec) {
    mapping.map(ifOp.getResult(idx), newIfOp.getResult(newResIdx));
    ++newResIdx;
  }
  return newIfOp;
}

Operation *SpecializeForOp(scf::ForOp forOp, IRMapping &mapping,
                           OpBuilderWithAsyncTaskIds &builder,
                           AsyncTaskId asyncTaskId) {
  // Create newForOp for each task Id.
  auto usedArgs = collectBlockArgsForTask(forOp, asyncTaskId);

  // Prepare newLoopArgs.
  SmallVector<Value> newLoopArgs;
  for (unsigned argNumber : usedArgs) {
    auto arg = forOp.getInitArgs()[argNumber];
    auto newArg = mapping.lookupOrDefault(arg);
    assert(newArg && "Unexpected missing mapping");
    newLoopArgs.push_back(newArg);
  }

  // Prepare loop bounds.
  auto newLowerBound = mapping.lookupOrDefault(forOp.getLowerBound());
  auto newUpperBound = mapping.lookupOrDefault(forOp.getUpperBound());
  auto newStep = mapping.lookupOrDefault(forOp.getStep());

  // Create newForOp.
  auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
      forOp.getLoc(), newLowerBound, newUpperBound, newStep, newLoopArgs);
  if (forOp->getAttr("tt.loop_schedule"))
    newForOp->setAttr("tt.loop_schedule", forOp->getAttr("tt.loop_schedule"));

  // Initialize Value mapping from forOp to newForOp
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (unsigned i = 0; i < usedArgs.size(); ++i) {
    auto oldArg = forOp.getRegionIterArgs()[usedArgs[i]];
    auto newArg = newForOp.getRegionIterArgs()[i];
    mapping.map(oldArg, newArg);
  }

  // Recursively clone all operations with this asyncTaskId to newForOp.
  OpBuilderWithAsyncTaskIds forBuilder(forOp.getContext());
  forBuilder.setAsynTaskIdsFromArray({asyncTaskId});
  forBuilder.setInsertionPointToStart(newForOp.getBody());
  for (Operation &op : forOp.getBody()->without_terminator()) {
    SpecializeOp(&op, mapping, forBuilder, asyncTaskId);
  }

  // Create YieldOp for newForOp.
  auto yieldOp = llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> newYieldOperands;
  for (unsigned i : usedArgs)
    newYieldOperands.push_back(mapping.lookup(yieldOp.getOperand(i)));

  bool createNewYield = true;
  if (newForOp.getBody()->mightHaveTerminator()) {
    auto initialYield =
        llvm::cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    if (newYieldOperands.size() == 0) {
      setAsyncTaskIds(initialYield, {asyncTaskId});
      createNewYield = false;
    }
  }
  if (createNewYield) {
    auto newYieldOp =
        scf::YieldOp::create(forBuilder, yieldOp.getLoc(), newYieldOperands);
    setAsyncTaskIds(newYieldOp, {asyncTaskId});
  }

  // Replace results of forOp with results of newForOp.
  for (unsigned i = 0; i < usedArgs.size(); ++i) {
    auto oldResult = forOp.getResult(usedArgs[i]);
    auto newResult = newForOp.getResult(i);
    mapping.map(oldResult, newResult);
  }

  return newForOp;
}

Operation *SpecializeOp(Operation *op, IRMapping &mapping,
                        OpBuilderWithAsyncTaskIds &builder,
                        AsyncTaskId asyncTaskId) {
  auto taskIds = getAsyncTaskIds(op);
  // yieldOp are sometimes implict, meaning they do not necessarily have a task
  // id, but they should be shared by all async tasks.
  if (!hasAsyncTaskId(op, asyncTaskId) && !isa<scf::YieldOp>(op))
    return nullptr;

  if (op->getNumRegions() == 0) {
    Operation *newOp = builder.clone(*op, mapping);
    setAsyncTaskIds(newOp, asyncTaskId);
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      mapping.map(op->getResult(i), newOp->getResult(i));
    return newOp;
  } else {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return SpecializeIfOp(ifOp, mapping, builder, asyncTaskId);
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return SpecializeForOp(forOp, mapping, builder, asyncTaskId);
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      Operation *newOp = builder.clone(*op, mapping);
      // recursively set async task ids for child ops
      newOp->walk(
          [&](Operation *childOp) { setAsyncTaskIds(childOp, asyncTaskId); });
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
      return newOp;
    } else {
      llvm_unreachable("Unexpected Op with regions");
    }
  }

  return nullptr;
}

void specializeRegion(triton::FuncOp funcOp, unsigned requestedRegisters) {

  LLVM_DEBUG({
    LDBG("\n\n");
    LDBG("Start specializing region");
  });

  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(context);
  auto loc = funcOp.getLoc();

  // Collect original operations
  SmallVector<Operation *> opList;
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &op : block.getOperations()) {
      auto taskIds = getAsyncTaskIds(&op);
      if (!taskIds.empty())
        opList.push_back(&op);
    }
  }

  LLVM_DEBUG({
    LDBG("ops to be specialized: ");
    for (Operation *op : opList) {
      op->dump();
    }
  });

  // Create GetAsyncTaskIdOp.
  Block *lastBlock = &funcOp.getBody().back();
  auto returnOp = llvm::cast<triton::ReturnOp>(lastBlock->getTerminator());
  builder.setInsertionPoint(returnOp);

  // Instead of a new IfOp for each task, we create one partitionRegion.
  auto nTaskIds = getNestedAsyncTaskIds(funcOp);
  SmallVector<int32_t> partitionNumWarps;
  for (AsyncTaskId asyncTaskId : nTaskIds) {
    if (asyncTaskId == 0)
      continue;
    partitionNumWarps.push_back(4);
  }
  ArrayRef<Type> dummyTypes;
  ImplicitLocOpBuilder impB(opList[0]->getLoc(), opList[0]);
  impB.setInsertionPoint(returnOp);
  auto wsOp = ttg::WarpSpecializeOp::create(impB, dummyTypes, partitionNumWarps,
                                            nTaskIds.size() - 1);

  // Clone all operations into the corresponding if blocks. If the operation
  // has multiple taskIds, it will be cloned for multiple if blocks.
  // If the original code has an IfOp, we should only clone its
  // body with the right asyncTaskId, instead of cloning the IfOp.
  // Handle producer WG.
  {
    AsyncTaskId asyncTaskId = nTaskIds[0];
    OpBuilderWithAsyncTaskIds taskBuilder(context);
    taskBuilder.setAsynTaskIdsFromArray({asyncTaskId});
    Block *defaultBlock = impB.createBlock(&wsOp.getDefaultRegion());
    taskBuilder.setInsertionPointToStart(defaultBlock);
    IRMapping mapping;
    for (Operation *op : opList) {
      SpecializeOp(op, mapping, taskBuilder, asyncTaskId);
    }
    SmallVector<Value> opnds;
    ttg::WarpYieldOp::create(taskBuilder, loc, opnds);
  }

  unsigned idx = 1;
  SmallVector<int32_t> estRegUsage;
  for (Region *region : wsOp.getPartitionRegions()) {
    AsyncTaskId asyncTaskId = nTaskIds[idx];
    OpBuilderWithAsyncTaskIds taskBuilder(context);
    taskBuilder.setAsynTaskIdsFromArray({asyncTaskId});
    LDBG("region idx " << idx << " " << nTaskIds.size());
    ++idx;
    Block *partitionBlock = impB.createBlock(region);
    taskBuilder.setInsertionPointToStart(partitionBlock);

    IRMapping mapping;
    for (Operation *op : opList) {
      SpecializeOp(op, mapping, taskBuilder, asyncTaskId);
    }
    ttg::WarpReturnOp::create(taskBuilder, loc);
    auto regAlloc =
        scanRegUsage(partitionBlock, asyncTaskId, requestedRegisters);
    estRegUsage.push_back(regAlloc);
  }

  // The default region doesn't request registers.
  wsOp.setRequestedRegisters(estRegUsage);

  // The capture set is the same for every partition region, so now find the
  // captures and thread them in to the regions.
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(wsOp.getPartitionOpHolder(), captures);
  for (Value capture : captures) {
    // Rematerialize constants.
    if (capture.getDefiningOp() &&
        capture.getDefiningOp()->hasTrait<OpTrait::ConstantLike>()) {
      for (Region *region : wsOp.getPartitionRegions()) {
        impB.setInsertionPointToStart(&region->front());
        Value copy = impB.clone(*capture.getDefiningOp())->getResult(0);
        replaceAllUsesInRegionWith(capture, copy, *region);
      }
      continue;
    }

    if (isa<RankedTensorType>(capture.getType())) {
      mlir::emitWarning(capture.getLoc(),
                        "FIXME: capturing tensor values into warp "
                        "partitions is not supported");
    }
    wsOp->insertOperands(wsOp.getNumOperands(), capture);
    for (Region *region : wsOp.getPartitionRegions()) {
      // Does this include default region?
      BlockArgument arg =
          region->addArgument(capture.getType(), capture.getLoc());
      replaceAllUsesInRegionWith(capture, arg, *region);
    }
  }

  LLVM_DEBUG({
    LDBG("\n\nWith task Id checks");
    funcOp.dump();
  });

  // Remove original operations that have been cloned in reverse order.
  for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
    Operation *op = *it;
    LLVM_DEBUG({
      LDBG("erasing op ");
      op->dump();
    });
    // For debugging purposes, check to see if the original op is still in use.
    bool hasUse = false;
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      for (Operation *user : op->getResult(i).getUsers()) {
        hasUse = true;
        LLVM_DEBUG({
          LDBG("op has use ");
          user->dump();
        });
      }
    }
    op->erase();
  }
}

} // namespace mlir
