#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "triton/Tools/StrUtil.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "triton-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

namespace {

/////////////////////////////
// UTILS
/////////////////////////////

int getSelfLatencyFromAttr(Operation *op) {
  auto module = op->getParentOfType<ModuleOp>();
  auto helper = TritonDialect::getLoaded(module)->getSelfLatencyAttrHelper();
  if (!helper.isAttrPresent(op))
    return 0;
  int val = helper.getAttr(op).getInt();
  helper.removeAttr(op);
  return val;
}

DenseSet<Operation *>
getTopLevelUsersInLoop(Operation *op, scf::ForOp forOp,
                       std::function<bool(Operation *)> filter = nullptr) {
  DenseSet<Operation *> topLevelUsers;
  SmallVector<OpOperand *> q;
  for (auto &use : op->getUses())
    q.push_back(&use);
  while (!q.empty()) {
    auto use = q.pop_back_val();
    auto yieldOp = dyn_cast<scf::YieldOp>(use->getOwner());
    if (yieldOp && yieldOp->getParentOp() == forOp) {
      for (auto &use :
           forOp.getRegionIterArgs()[use->getOperandNumber()].getUses())
        q.push_back(&use);
      continue;
    }
    // Don't count view operations as uses. Follow them through to their
    // users.
    if (use->getOwner()->hasTrait<OpTrait::MemDescViewTrait>()) {
      for (auto &use : use->getOwner()->getUses())
        q.push_back(&use);
      continue;
    }
    if (filter && !filter(use->getOwner()))
      continue;
    Operation *topLevelUser =
        forOp.getBody()->findAncestorOpInBlock(*use->getOwner());
    topLevelUsers.insert(topLevelUser);
  }
  return topLevelUsers;
}

Operation *getFirstUseOfPipelinedOp(SmallVector<Operation *> ops,
                                    scf::ForOp forOp,
                                    CoarseSchedule &schedule) {
  Operation *firstUser = nullptr;
  DenseSet<Operation *> topLevelUsers;
  for (Operation *op : ops) {
    auto users = getTopLevelUsersInLoop(op, forOp);
    topLevelUsers.insert(users.begin(), users.end());
  }
  for (Operation *topLevelUser : topLevelUsers) {
    assert(schedule.count(topLevelUser) && "op user not found in the schedule");
    auto [_useStage, _useCluster] = schedule[topLevelUser];
    if (!firstUser) {
      firstUser = topLevelUser;
    } else {
      auto [_firstUserStage, _firstUserCluster] = schedule[firstUser];
      if (_useStage < _firstUserStage ||
          (_useStage == _firstUserStage &&
           schedule.clusters.isBefore(_useCluster, _firstUserCluster)) ||
          (_useStage == _firstUserStage && _useCluster == _firstUserCluster &&
           topLevelUser->isBeforeInBlock(firstUser))) {
        firstUser = topLevelUser;
      }
    }
  }
  return firstUser;
}

// Check if the load can be pipelined entirely in shared memory,
// or if we need to load to registers.
bool mustLoadToRegisters(Operation *op) {
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    // AsyncCopyGlobalToLocalOp does not support the non-zero "other" value.
    // With consumer consuming directly the shared memory, there would be no way
    // to replace masked values with the "other" value.
    if (loadOp.getOther() && !isZeroConst(loadOp.getOther()))
      return true;
  }

  if (!op->hasOneUse())
    return true;
  Operation *user = *op->getUsers().begin();
  auto alloc = dyn_cast<ttg::LocalAllocOp>(user);
  if (!alloc)
    return true;

  Attribute loadEncoding;
  if (auto descLoad = dyn_cast<DescriptorLoadOp>(op)) {
    loadEncoding = nvidia_gpu::getEncodingFromDescriptor(op, descLoad.getType(),
                                                         descLoad.getDesc());
  } else if (auto descGather = dyn_cast<DescriptorGatherOp>(op)) {
    loadEncoding = nvidia_gpu::getEncodingFromDescriptor(
        op, descGather.getType(), descGather.getDesc());
  }
  return loadEncoding && (loadEncoding != alloc.getType().getEncoding());
}

int getDefUseStageDiff(Operation *op, scf::ForOp forOp,
                       CoarseSchedule &schedule) {
  assert(schedule.count(op) && "Op not found in the schedule");
  int defStage = schedule[op].first;
  std::optional<int> useStage;
  DenseSet<Operation *> topLevelUsers = getTopLevelUsersInLoop(op, forOp);
  // Special case for loads used by local_alloc:
  // we must consider the uses of the local_alloc, as it may be removed and its
  // uses will become direct uses of the async load.
  // TODO: This is overly conservative, we may need to restrict to cases where
  // local_alloc is used by a dot product and has correct encoding.
  if (isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op)) {
    DenseSet<Operation *> allocUsers;
    for (Operation *topLevelUser : topLevelUsers) {
      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(topLevelUser)) {
        DenseSet<Operation *> users = getTopLevelUsersInLoop(localAlloc, forOp);
        allocUsers.insert(users.begin(), users.end());
      }
    }
    topLevelUsers.insert(allocUsers.begin(), allocUsers.end());
  }
  DenseSet<Operation *> topLevelWaitUsers;
  for (Operation *topLevelUser : topLevelUsers) {
    if (isa<ttng::WaitBarrierOp>(topLevelUser)) {
      topLevelWaitUsers.insert(topLevelUser);
    }
  }
  for (Operation *topLevelUser : topLevelUsers) {
    int _useStage = schedule[topLevelUser].first;
    useStage = std::min(_useStage, useStage.value_or(_useStage));
  }
  // Waits tells us the buffer is still in use until the wait completes, we
  // can't simply load from the buffer and replace the uses of the buffer with
  // the load. The stage diff needs to account for the furthest wait.
  for (Operation *topLevelUser : topLevelWaitUsers) {
    int _useStage = schedule[topLevelUser].first;
    useStage = std::max(_useStage, useStage.value_or(_useStage));
  }
  if (!useStage)
    return 0;
  assert(useStage >= defStage && "Op used before defined");
  return useStage.value() - defStage;
}

void replaceAllUsesDominatedBy(Operation *domOp, Value newValue, Value oldValue,
                               DominanceInfo &domInfo) {
  if (newValue == oldValue)
    return;
  oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
    return domInfo.properlyDominates(domOp, use.getOwner());
  });
}

/////////////////////////////
// LOWER LOADS
/////////////////////////////

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, Operation *loadOp,
                         ttg::SharedEncodingTrait sharedEnc,
                         unsigned distance) {
  return triton::createAlloc(
      forOp, cast<RankedTensorType>(loadOp->getResultTypes().front()),
      loadOp->getLoc(), sharedEnc, distance);
}

void createAsyncCopy(scf::ForOp forOp, tt::LoadOp loadOp, Value alloc,
                     Value insertIdx, Value extractIdx,
                     CoarseSchedule &schedule) {
  OpBuilderForStage builder(loadOp.getLoc(), forOp, schedule);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);

  Operation *firstUse = getFirstUseOfPipelinedOp({loadOp}, forOp, schedule);
  assert(firstUse && "LoadOp has no users");
  // Replace the load with async copy, wait and loal_load.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loadOp);
  builder.setStageCluster(schedule[loadOp]);
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();
  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());

  // Create async copy
  Value view = createSingleBufferView(builder, alloc, insertIdx);
  Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      src, view, mask, other, loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile());
  Operation *commit =
      builder.create<ttg::AsyncCommitGroupOp>(copy->getResult(0));

  // Create wait and local load
  builder.setStageCluster(schedule[firstUse]);
  auto wait = builder.create<ttg::AsyncWaitOp>(commit->getResult(0), 0);
  auto viewLoad = createSingleBufferView(builder, alloc, extractIdx);

  if (!loadOp.getOther() || isZeroConst(loadOp.getOther())) {
    // If masking isn't required, load directly from shared
    replaceUsesWithLocalLoad(builder, loadOp->getResult(0), viewLoad,
                             wait.getResult());
  } else if (loadOp->use_begin() != loadOp->use_end()) {
    // Otherwise, create a select for non-zero other values as they are not
    // handled by AsyncCopyGlobalToLocalOp for now.
    auto sharedLoad = builder.create<ttg::LocalLoadOp>(
        loadOp.getType(), viewLoad, wait.getResult());
    auto select = builder.create<arith::SelectOp>(
        loadOp.getType(),
        // Use the mask operand from the original load, not the one with a
        // potentially transformed layout.
        loadOp.getMask(), sharedLoad.getResult(), other);
    loadOp->replaceAllUsesWith(select->getResults());
  }
  schedule.erase(loadOp);
  loadOp->erase();
}

void createTMAAsyncCopy(
    scf::ForOp forOp, Operation *loadOp, Value desc, Value alloc,
    Value insertIdx, Value extractIdx, Value barrier, Operation *waitOp,
    CoarseSchedule &schedule,
    function_ref<void(OpBuilderForStage &, Value, Value, Value, Value)>
        createCopy) {
  OpBuilderForStage builder(loadOp->getLoc(), forOp, schedule);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);

  Operation *firstUse = getFirstUseOfPipelinedOp({loadOp}, forOp, schedule);
  assert(firstUse && "LoadOp has no users");
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());

  builder.setInsertionPoint(loadOp);
  builder.setStageCluster(schedule[loadOp]);
  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());

  // Create async copy
  Value view = createSingleBufferView(builder, alloc, insertIdx);

  Value pred = builder.create<arith::ConstantIntOp>(1, 1);
  createCopy(builder, desc, barrier, view, pred);

  // Create local load after the wait
  builder.setInsertionPointAfter(waitOp);
  builder.setStageCluster(schedule[firstUse]);
  auto viewLoad = createSingleBufferView(builder, alloc, extractIdx);
  replaceUsesWithLocalLoad(builder, loadOp->getResult(0), viewLoad);

  schedule.erase(loadOp);
  loadOp->erase();
}

void createTMAAsyncLoad(scf::ForOp forOp, tt::DescriptorLoadOp loadOp,
                        Value alloc, Value insertIdx, Value extractIdx,
                        Value barrier, Operation *waitOp,
                        CoarseSchedule &schedule) {
  return createTMAAsyncCopy(
      forOp, loadOp, loadOp.getDesc(), alloc, insertIdx, extractIdx, barrier,
      waitOp, schedule,
      [&](OpBuilderForStage &builder, Value tmaPtr, Value barrier, Value view,
          Value pred) {
        auto indices = ttng::translateTMAIndices(
            builder, loadOp.getLoc(),
            loadOp.getDesc().getType().getBlockType().getEncoding(),
            loadOp.getIndices());
        builder.create<ttng::AsyncTMACopyGlobalToLocalOp>(
            loadOp.getLoc(), tmaPtr, indices, barrier, view, pred);
      });
}

void createTMAAsyncGather(scf::ForOp forOp, tt::DescriptorGatherOp gatherOp,
                          Value alloc, Value insertIdx, Value extractIdx,
                          Value barrier, Operation *waitOp,
                          CoarseSchedule &schedule) {
  return createTMAAsyncCopy(forOp, gatherOp, gatherOp.getDesc(), alloc,
                            insertIdx, extractIdx, barrier, waitOp, schedule,
                            [&](OpBuilderForStage &builder, Value tmaPtr,
                                Value barrier, Value view, Value pred) {
                              builder.create<ttng::AsyncTMAGatherOp>(
                                  gatherOp.getLoc(), tmaPtr,
                                  gatherOp.getXOffsets(), gatherOp.getYOffset(),
                                  barrier, view, pred);
                            });
}

struct AsyncLoad {
  int stageDiff;
  Value alloc;
  Value barrier;
  Operation *waitOp;
  SharedEncodingTrait sharedEncoding;
};
struct LoadGroupInfo {
  Value insertIdx;
  Value extractIdx;
  Value phase;
  bool hasTMALoad = false;
};

void createTMABarrierAndWait(
    scf::ForOp forOp, llvm::MapVector<Operation *, AsyncLoad> &asyncLoads,
    const llvm::MapVector<int, LoadGroupInfo> &loadGroups,
    CoarseSchedule &schedule) {
  SmallVector<SmallVector<Operation *>> commonWaitGroups;
  llvm::SmallDenseSet<Operation *> visited;
  // Find groups of loads that can share the same barrier. We look consecutive
  // loads and check that there are uses in between.
  for (auto &[loadOp, asyncLoad] : asyncLoads) {
    if (!isTMALoad(loadOp) || visited.count(loadOp))
      continue;
    llvm::SmallDenseSet<Operation *> users;
    SmallVector<Operation *> group;
    Block *loadBlock = loadOp->getBlock();
    auto addToGroup = [&](Operation *loadOp) {
      group.push_back(loadOp);
      visited.insert(loadOp);
      for (Operation *user : loadOp->getUsers()) {
        // Special case for MMAv3 loads, we can ignore the alloc and only
        // consider uses of the alloc op since it will be removed.
        if (!mustLoadToRegisters(loadOp)) {
          assert(loadOp->hasOneUse());
          auto alloc = cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin());
          if (alloc->getBlock() == loadBlock) {
            users.insert(alloc->getUsers().begin(), alloc->getUsers().end());
            continue;
          }
        }
        Operation *userInBlock = loadBlock->findAncestorOpInBlock(*user);
        if (userInBlock)
          users.insert(userInBlock);
      }
    };
    addToGroup(loadOp);
    Operation *nextOp = loadOp->getNextNode();
    int numBuffers = asyncLoad.stageDiff;
    while (nextOp) {
      if (users.count(nextOp) || visited.count(nextOp))
        break;
      if (isTMALoad(nextOp) && asyncLoads.count(nextOp)) {
        if (asyncLoads[nextOp].stageDiff != numBuffers)
          break;
        if (group.size() > 0 && schedule[group[0]] == schedule[nextOp]) {
          addToGroup(nextOp);
        }
      }
      nextOp = nextOp->getNextNode();
    }
    commonWaitGroups.push_back(group);
  }

  // For each group calculate the size and insert the barrier after the last
  // load.
  for (SmallVector<Operation *> &group : commonWaitGroups) {
    int sizeInBytes = 0;
    int numBuffers = asyncLoads[group[0]].stageDiff;
    const LoadGroupInfo loadGroup = loadGroups.find(numBuffers)->second;
    for (Operation *op : group) {
      auto tensorTy = cast<RankedTensorType>(op->getResultTypes()[0]);
      int loadSize = product(getShapePerCTA(tensorTy));
      sizeInBytes += loadSize * tensorTy.getElementTypeBitWidth() / 8;
    }

    Value barrierAlloc = triton::createBarrierAlloc(forOp, numBuffers);
    OpBuilderForStage builder(forOp.getLoc(), group[0], schedule);
    Value barrier = triton::createSingleBufferView(builder, barrierAlloc,
                                                   loadGroup.insertIdx);
    Value pred = builder.create<arith::ConstantIntOp>(1, 1);
    builder.create<ttng::BarrierExpectOp>(barrier, sizeInBytes, pred);

    builder.setInsertionPointAfter(group.back());
    Operation *firstUse = getFirstUseOfPipelinedOp(group, forOp, schedule);
    builder.setStageCluster(schedule[firstUse]);
    Value barrierViewWait = triton::createSingleBufferView(
        builder, barrierAlloc, loadGroup.extractIdx);
    auto wait =
        builder.create<ttng::WaitBarrierOp>(barrierViewWait, loadGroup.phase);

    // Update the async loads info.
    for (Operation *op : group) {
      asyncLoads[op].barrier = barrier;
      asyncLoads[op].waitOp = wait;
    }
  }
}

// Check if load requires additional buffer for a mma pipelining
bool loadRequiresAdditionalBuffer(Operation *loadOp) {
  auto skipViewOps = [](Operation *op) -> Operation * {
    while (op->hasOneUse() && op->hasTrait<OpTrait::MemDescViewTrait>()) {
      op = *op->getUsers().begin();
    }
    return op;
  };
  // Pattern match the op sequence used for loading mmav3 operands
  if (!mustLoadToRegisters(loadOp)) {
    assert(loadOp->hasOneUse());
    ttg::LocalAllocOp alloc =
        dyn_cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin());
    if (alloc) {
      return llvm::any_of(alloc->getUsers(), [&](Operation *op) {
        return isa<ttng::WarpGroupDotOp>(skipViewOps(op));
      });
    }
  }
  return false;
}

scf::ForOp lowerLoads(scf::ForOp forOp, CoarseSchedule &schedule,
                      triton::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  llvm::MapVector<Operation *, AsyncLoad> asyncLoads;
  llvm::MapVector<int, LoadGroupInfo> loadGroups;
  // Only visit the top level ops, we do not support pipelining conditional
  // loads for now
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op)) {
      int stageDiff = getDefUseStageDiff(&op, forOp, schedule);
      if (stageDiff == 0 || !isa<RankedTensorType>(op.getResultTypes()[0])) {
        // Don't care about non-pipelined loads. Don't use async loads for
        // scalar values.
        continue;
      }
      SharedEncodingTrait sharedEncoding = getSharedEncoding(&op);
      // Do not create async loads for small loads (cp.async requires at least 4
      // bytes)
      bool canUseAsyncCp =
          isa<tt::LoadOp>(op) &&
          canBeConvertedToAsyncLoad(cast<tt::LoadOp>(op), axisInfoAnalysis);
      int copyVecBytes = getCopyVecBytes(
          cast<RankedTensorType>(op.getResultTypes()[0]), sharedEncoding);
      canUseAsyncCp &= copyVecBytes >= 4;
      if (canUseAsyncCp || isTMALoad(&op)) {
        if (loadRequiresAdditionalBuffer(&op)) {
          // Allocate additional buffer required by the wgmma pipelining.
          stageDiff += 1;
        }
        auto &asyncLoad = asyncLoads[&op];
        asyncLoad.stageDiff = stageDiff;
        asyncLoad.sharedEncoding = sharedEncoding;
      } else if (stageDiff > 1) {
        // Distance-1 loads can in most cases be pipelined in registers without
        // any performance degradation, as the schedule will usually reorder the
        // user and the producer so there is no liverange overlap, and no copy
        // needed.
        op.emitRemark() << "Pipelining load that cannot use vectorized "
                           "copy. This will likely "
                           "lead to pipelining in registers and severe "
                           "performance degradation.";
      }
    }
  }

  if (asyncLoads.empty())
    return forOp;

  for (auto &[loadOp, asyncLoad] : asyncLoads) {
    Value alloc = createAlloc(forOp, loadOp, asyncLoad.sharedEncoding,
                              asyncLoad.stageDiff);
    asyncLoad.alloc = alloc;
    loadGroups.insert({asyncLoad.stageDiff, {}});
    if (isTMALoad(loadOp)) {
      loadGroups[asyncLoad.stageDiff].hasTMALoad = true;
    }
  }

  IRRewriter builder(forOp);
  builder.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  // Create a counter to index into the allocations per loop iteration.
  // NOTE: We create two duplicates values, insertIdx and extractIdx so that the
  // pipeliner will re-materialize the value in later stages of the pipeline
  // instead of carrying it as a dependency across multiple iterations.
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  SmallVector<Value> newOperands;
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  for (auto [_, loadGroup] : loadGroups) {
    newOperands.push_back(minusOne); // insertIdx
    newOperands.push_back(minusOne); // extractIdx
    if (loadGroup.hasTMALoad) {
      // A single barrier arrival sequence is a "phase" and two phases can
      // overlap, provided the phases are differentiated with an alternating
      // boolean value.
      newOperands.push_back(zero); // phase
    }
  }

  // Patch the loop to add the new loop carried dependencies.
  forOp = addIterArgsToLoop(builder, forOp, newOperands);

  // Update yield op with temporary yield values
  auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  for (unsigned i = 0; i < newOperands.size(); ++i) {
    forYield.getResultsMutable().append(newOperands[i]);
  }

  builder.setInsertionPoint(forOp);
  loc = forOp.getLoc();
  int argIdx = newOperandIndex;
  for (auto &[numBuffers, loadGroup] : loadGroups) {
    Value insertIdx = forOp.getBody()->getArgument(argIdx);
    argIdx++;
    Value extractIdx = forOp.getBody()->getArgument(argIdx);
    argIdx++;
    Value phase = nullptr;
    if (loadGroup.hasTMALoad) {
      phase = forOp.getBody()->getArgument(argIdx);
      argIdx++;
    }

    // Create two counters for the insert and extract indices to avoid creating
    // long liverange.
    builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());

    Value numBuffersVal =
        builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
    loadGroup.insertIdx = createIncrementModulo(builder, loc, insertIdx,
                                                numBuffersVal, zero, one);
    Value cndExt = nullptr;
    loadGroup.extractIdx = createIncrementModulo(
        builder, loc, extractIdx, numBuffersVal, zero, one, &cndExt);
    if (phase) {
      Value nextPhase = builder.create<arith::XOrIOp>(loc, phase, one);
      phase = builder.create<arith::SelectOp>(loc, cndExt, nextPhase, phase);
      loadGroup.phase = phase;
    }
  }

  createTMABarrierAndWait(forOp, asyncLoads, loadGroups, schedule);

  bool hasAsyncLoads = false;
  for (auto [op, asyncLoad] : asyncLoads) {
    auto [insertIdx, extractIdx, phase, _] = loadGroups[asyncLoad.stageDiff];
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      createAsyncCopy(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                      schedule);
      hasAsyncLoads = true;
    } else if (auto loadOp = dyn_cast<tt::DescriptorLoadOp>(op)) {
      createTMAAsyncLoad(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                         asyncLoad.barrier, asyncLoad.waitOp, schedule);
    } else if (auto loadOp = dyn_cast<tt::DescriptorGatherOp>(op)) {
      createTMAAsyncGather(forOp, loadOp, asyncLoad.alloc, insertIdx,
                           extractIdx, asyncLoad.barrier, asyncLoad.waitOp,
                           schedule);
    }
  }
  // Patch the yield with the updated counters. Subtract to account for the loop
  // counter.
  argIdx = newOperandIndex - 1;
  for (auto &[numBuffers, loadGroup] : loadGroups) {
    forYield.setOperand(argIdx++, loadGroup.insertIdx);
    forYield.setOperand(argIdx++, loadGroup.extractIdx);
    if (loadGroup.phase)
      forYield.setOperand(argIdx++, loadGroup.phase);
  }

  // Automatically discover dependencies and schedule new insert/extract ops to
  // correct stages.
  scheduleDependencies(forOp, schedule);

  if (hasAsyncLoads) {
    // Insert sync point for any possibly outstanding loads after the loop. This
    // can happen as we speculatively execute loads in the loop.
    builder.setInsertionPointAfter(forOp);
    builder.create<ttg::AsyncWaitOp>(loc, ValueRange({}), 0);
  }

  // Make sure all ops have attributes.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!schedule.count(&op)) {
      op.emitError() << "op not found in the schedule";
    }
    assert(schedule.count(&op) && "op not found in the schedule");
  }
  return forOp;
}

/////////////////////////////
// LOWER MMA
/////////////////////////////

std::pair<Operation *, Operation *>
getTmemUseStageBoundOps(ttng::TMEMAllocOp alloc, scf::ForOp forOp,
                        CoarseSchedule &schedule) {
  std::pair<Operation *, Operation *> bounds = {nullptr, nullptr};
  for (auto user : alloc->getUsers()) {
    if (!forOp->isAncestor(user->getParentOp())) {
      continue;
    }
    auto topLevelUser = forOp.getBody()->findAncestorOpInBlock(*user);
    if (!bounds.first) {
      bounds.first = topLevelUser;
    }
    if (!bounds.second) {
      bounds.second = topLevelUser;
    }
    if (schedule.isOpBefore(topLevelUser, bounds.first)) {
      bounds.first = topLevelUser;
    }
    if (schedule.isOpBefore(bounds.second, topLevelUser)) {
      bounds.second = topLevelUser;
    }
  }
  return bounds;
}

Operation *hoistBufferOutOfLoop(scf::ForOp forOp, Operation *op,
                                CoarseSchedule &schedule) {
  Operation *newStore = nullptr;
  if (!isa<ttng::TMEMAllocOp, ttg::LocalAllocOp>(op))
    return nullptr;
  // If the alloc is already out of the loop, there is nothing to do.
  if (!forOp->isAncestor(op))
    return nullptr;
  OpBuilderForStage builder(op->getLoc(), forOp, schedule);
  auto allocType = dyn_cast<MemDescType>(op->getResult(0).getType());
  auto newType = triton::gpu::MemDescType::get(
      allocType.getShape(), allocType.getElementType(), allocType.getEncoding(),
      allocType.getMemorySpace(),
      /*mutableMemory=*/true);
  auto newAlloc = builder.clone(*op);
  newAlloc->getResult(0).setType(newType);
  builder.setStageCluster(schedule[op]);
  if (auto tmemAlloc = dyn_cast<ttng::TMEMAllocOp>(newAlloc)) {
    tmemAlloc.getSrcMutable().clear();
    builder.setInsertionPointAfter(op);
    Value trueVal = builder.create<arith::ConstantIntOp>(1, 1);
    newStore = builder.create<ttng::TMEMStoreOp>(tmemAlloc.getResult(),
                                                 op->getOperand(0), trueVal);
  } else {
    auto localAlloc = cast<ttg::LocalAllocOp>(newAlloc);
    localAlloc.getSrcMutable().clear();
    builder.setInsertionPointAfter(op);
    newStore = builder.create<ttg::LocalStoreOp>(op->getOperand(0),
                                                 localAlloc.getResult());
  }
  replaceUsesAndPropagateType(builder, op, newAlloc->getResult(0));
  op->erase();
  return newStore;
}

void createBarrierAndWaitOps(scf::ForOp forOp, CoarseSchedule &schedule,
                             ttng::MMAv5OpInterface mma, int mmaSelfLatency,
                             ttng::TMEMAllocOp alloc, int phaseArgIdx,
                             int barrierIdxArgIdx) {
  auto isLoadToBePipelined = [&](Operation *op) {
    return schedule[mma].first > schedule[op].first;
  };

  std::optional<Operation *> latestSyncPoint;
  for (auto user : alloc->getUsers()) {
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (load->getBlock() != mma->getBlock()) {
        continue;
      }
      if (!latestSyncPoint || schedule.isOpBefore(load, *latestSyncPoint)) {
        latestSyncPoint = load;
      }
    }
  }

  ttng::MMAv5PipelineableOperandsHelper mmaPipeHelper(mma, forOp,
                                                      isLoadToBePipelined);

  SmallVector<Operation *> updatedDefs;
  for (auto def : mmaPipeHelper.unpipelineableOperandDefs) {
    auto newStore = hoistBufferOutOfLoop(forOp, def, schedule);
    if (newStore) {
      updatedDefs.push_back(newStore);
    } else {
      updatedDefs.push_back(def);
    }
  }

  if (!mmaPipeHelper.isPipelineable &&
      mmaPipeHelper.isOperandsStateDetermined) {
    // If the operands are not pipelineable, we need to insert a sync point
    // before the earliest operand load
    for (auto def : updatedDefs) {
      if (!latestSyncPoint || schedule.isOpBefore(def, *latestSyncPoint)) {
        latestSyncPoint = def;
      }
    }
  }

  int mainWaitStage = schedule[mma].first + mmaSelfLatency;
  CoarseSchedule::Cluster mainWaitCluster = schedule[mma].second;
  if (latestSyncPoint && mmaPipeHelper.isOperandsStateDetermined) {
    if (schedule.isOpBefore(*latestSyncPoint, mma)) {
      mainWaitStage = schedule[mma].first + 1;
      mainWaitCluster = schedule.clusters.newBefore(
          schedule.splitClusterBefore(*latestSyncPoint, forOp));
    } else {
      mainWaitStage = schedule[*latestSyncPoint].first;
      mainWaitCluster = schedule.clusters.newBefore(
          schedule.splitClusterBefore(*latestSyncPoint, forOp));
    }
  }

  int numStages = mainWaitStage - schedule[mma].first + 1;

  OpBuilderForStage builder(mma.getLoc(), mma, schedule);
  Value barrierAlloc = createBarrierAlloc(forOp, numStages);
  Value vTrue = builder.create<arith::ConstantIntOp>(1, 1);
  Value phase = forOp.getRegionIterArg(phaseArgIdx);
  Value barrierIdx = forOp.getRegionIterArg(barrierIdxArgIdx);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
  Value numStagesVal =
      builder.create<arith::ConstantIntOp>(forOp.getLoc(), numStages, 32);

  Value barrierSlice = barrierAlloc;
  if (numStages > 1) {
    barrierSlice =
        triton::createSingleBufferView(builder, barrierAlloc, barrierIdx);
  }
  mma.addCompletionBarrier(barrierSlice, vTrue);

  // List of buffers that may be used until wait completes
  SmallVector<Value> waitBuffers;
  auto mmaAsDotOp = cast<DotOpInterface>(mma.getOperation());
  waitBuffers.push_back(mmaAsDotOp.getA());
  waitBuffers.push_back(mmaAsDotOp.getB());
  if (auto mmaAsScaledDotOp =
          dyn_cast<ttng::TCGen5MMAScaledOp>(mma.getOperation())) {
    waitBuffers.push_back(mmaAsScaledDotOp.getAScale());
    waitBuffers.push_back(mmaAsScaledDotOp.getBScale());
  }

  builder.setInsertionPointAfter(mma);
  builder.setStageCluster({mainWaitStage, mainWaitCluster});
  builder.create<ttng::WaitBarrierOp>(barrierSlice, phase, waitBuffers);

  // Add waits before loads in conditional blocks
  for (auto user : alloc->getUsers()) {
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (load->getBlock() == mma->getBlock()) {
        continue;
      }
      auto topLevelUser = forOp.getBody()->findAncestorOpInBlock(*load);
      if (!topLevelUser) {
        continue;
      }
      auto [loadStage, loadCluster] = schedule[topLevelUser];
      if (loadStage < mainWaitStage) {
        builder.setStageCluster({loadStage, loadCluster});
        builder.setInsertionPoint(load);
        builder.create<ttng::WaitBarrierOp>(barrierSlice, phase, waitBuffers);
      }
    }
  }

  builder.setStageCluster(schedule[mma]);
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  builder.setInsertionPoint(yieldOp);
  Value newPhase = builder.create<arith::XOrIOp>(phase, one);
  Value newBarrierIdx = barrierIdx;
  if (numStages > 1) {
    Value barWrap;
    newBarrierIdx = createIncrementModulo(builder, builder.getLoc(), barrierIdx,
                                          numStagesVal, zero, one, &barWrap);
    newPhase = builder.create<arith::SelectOp>(phase.getType(), barWrap,
                                               newPhase, phase);
  }
  yieldOp->replaceUsesOfWith(phase, newPhase);
  yieldOp->replaceUsesOfWith(barrierIdx, newBarrierIdx);
}

void multibufferTensorMemory(scf::ForOp forOp, CoarseSchedule &schedule,
                             ttng::TMEMAllocOp alloc, int bufIdxArgIdx,
                             int tmemUseNumStages) {
  DominanceInfo domInfo(forOp);
  Value bufIdx = forOp.getRegionIterArg(bufIdxArgIdx);
  SmallVector<std::pair<Operation *, Value>> bufIdxDefs;
  auto getCurrBufIdx = [&](Operation *op) {
    for (auto [_op, _val] : llvm::reverse(bufIdxDefs)) {
      if (domInfo.properlyDominates(_op, op)) {
        return _val;
      }
    }
    return Value();
  };
  bufIdxDefs.push_back({&forOp.getBody()->front(), bufIdx});

  OpBuilderForStage builder(alloc.getLoc(), alloc, schedule);
  auto newAlloc = createTMemAlloc(builder, alloc, true, tmemUseNumStages);
  Value numStagesVal =
      builder.create<arith::ConstantIntOp>(tmemUseNumStages, 32);
  Value zero = builder.create<arith::ConstantIntOp>(0, 32);
  Value one = builder.create<arith::ConstantIntOp>(1, 32);

  bool multibufferingIsValid = false;

  SmallVector<Operation *> allocUsers =
      llvm::to_vector(alloc.getResult().getUsers());
  Value replTok = OpBuilder(forOp).create<ub::PoisonOp>(
      forOp.getLoc(), builder.getType<AsyncTokenType>());
  for (auto user : allocUsers) {
    if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
      if (forOp->isAncestor(store)) {
        store.getDepMutable().clear();
        store.getToken().replaceAllUsesWith(replTok);
        // We can multibuffer, since the store is a point where we can
        // change the buffer index
        multibufferingIsValid = true;
        builder.setStageCluster(schedule[store]);
        builder.setInsertionPoint(store);
        // Change the buffer index to the new buffer index on store.
        Value curBufIdx = getCurrBufIdx(store);
        Value newBufIdx = createIncrementModulo(
            builder, forOp.getLoc(), curBufIdx, numStagesVal, zero, one);
        if (Value pred = store.getPred()) {
          newBufIdx = builder.create<arith::SelectOp>(newBufIdx.getType(), pred,
                                                      newBufIdx, curBufIdx);
        }
        replaceAllUsesDominatedBy(store, newBufIdx, curBufIdx, domInfo);
        bufIdxDefs.push_back({store, newBufIdx});
        auto tmemSlice =
            triton::createSingleBufferView(builder, newAlloc, newBufIdx);
        store.getDstMutable().assign(tmemSlice);
      } else {
        // Store before the loop
        assert(store->isBeforeInBlock(forOp) && "Store is not before the loop");
        builder.setInsertionPoint(store);
        auto tmemSlice =
            triton::createSingleBufferView(builder, newAlloc, zero);
        store.getDstMutable().assign(tmemSlice);
      }
    } else if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (forOp->isAncestor(load)) {
        load.getDepMutable().clear();
        load.getToken().replaceAllUsesWith(replTok);
        builder.setStageCluster(schedule[load]);
        builder.setInsertionPoint(load);
        Value curBufIdx = getCurrBufIdx(load);
        auto tmemSlice =
            triton::createSingleBufferView(builder, newAlloc, curBufIdx);
        load.getSrcMutable().assign(tmemSlice);
      } else {
        // Load after the loop
        assert(forOp->isBeforeInBlock(load) && "Load is not after the loop");
        builder.setInsertionPoint(load);
        auto tmemSlice = triton::createSingleBufferView(
            builder, newAlloc, forOp->getResult(bufIdxArgIdx));
        load.getSrcMutable().assign(tmemSlice);
      }
    } else if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(user)) {
      mma.getAccDepMutable().clear();
      mma.getToken().replaceAllUsesWith(replTok);
      builder.setStageCluster(schedule[mma]);
      builder.setInsertionPoint(mma);
      // We can legally switch to next buffer index if the mma does not use the
      // accumulator
      auto isConstTrue = [](Value v) {
        if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
          if (auto attr = dyn_cast<BoolAttr>(constOp.getValueAttr())) {
            return attr.getValue();
          }
        }
        return false;
      };
      multibufferingIsValid = !isConstTrue(mma.useAccumulator());
      Value curBufIdx = getCurrBufIdx(mma.getOperation());
      Value newBufIdx = createIncrementModulo(
          builder, forOp.getLoc(), curBufIdx, numStagesVal, zero, one);
      newBufIdx = builder.create<arith::SelectOp>(
          newBufIdx.getType(), mma.useAccumulator(), curBufIdx, newBufIdx);
      replaceAllUsesDominatedBy(mma.getOperation(), newBufIdx, curBufIdx,
                                domInfo);
      bufIdxDefs.push_back({mma.getOperation(), newBufIdx});
      auto tmemSlice =
          triton::createSingleBufferView(builder, newAlloc, newBufIdx);
      mma.setAccumulator(tmemSlice);
    } else {
      llvm::errs() << "Unsupported user of the accumulator: " << *user << "\n";
      llvm::report_fatal_error("Unsupported user of the accumulator");
    }
  }
  if (!multibufferingIsValid) {
    llvm::report_fatal_error(
        "Trying to multibuffer TMEM while there is no store to the "
        "accumulator, and the mma uses the accumulator all the time.");
  }
  alloc.getToken().replaceAllUsesWith(newAlloc.getToken());
  alloc->erase();

  Value newBufIdx = bufIdxDefs.back().second;
  replaceAllUsesDominatedBy(newBufIdx.getDefiningOp(), newBufIdx, bufIdx,
                            domInfo);
}

scf::ForOp lowerMMA(ttng::MMAv5OpInterface mma, scf::ForOp forOp,
                    CoarseSchedule &schedule) {
  auto isLoadToBePipelined = [&](Operation *op) {
    return schedule[mma].first > schedule[op].first;
  };
  auto alloc = mma.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!alloc) {
    return forOp;
  }

  int mmaSelfLatency = getSelfLatencyFromAttr(mma.getOperation());
  if (mmaSelfLatency == 0) {
    return forOp;
  }

  // Create barrier and wait ops
  std::pair<Operation *, Operation *> tmemUseStageBoundOps =
      getTmemUseStageBoundOps(alloc, forOp, schedule);
  int tmemUseNumStages = schedule[tmemUseStageBoundOps.second].first -
                         schedule[tmemUseStageBoundOps.first].first;
  // If def is in the earlier cluster than the use, we will have a liverange
  // overlap and need to add an extra buffer.
  if (schedule.isOpInEarlierCluster(tmemUseStageBoundOps.first,
                                    tmemUseStageBoundOps.second) ||
      (schedule.isOpInSameCluster(tmemUseStageBoundOps.first,
                                  tmemUseStageBoundOps.second) &&
       tmemUseStageBoundOps.first->isBeforeInBlock(
           tmemUseStageBoundOps.second))) {
    tmemUseNumStages += 1;
  }

  OpBuilder builder(forOp);
  Value minusOne = builder.create<arith::ConstantIntOp>(forOp.getLoc(), -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);

  // Add arguments to the forOp
  unsigned newOperandIndex = forOp.getInitArgs().size();
  SmallVector<Value> newOperands = {
      zero, // phase
      zero, // barrierIdx
  };
  if (tmemUseNumStages > 1) {
    newOperands.push_back(minusOne); // bufIdx
  }
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;

  int phaseArgIdx = newOperandIndex + 0;
  int barrierIdxArgIdx = newOperandIndex + 1;
  int bufIdxArgIdx = newOperandIndex + 2;
  Value phase = forOp.getRegionIterArg(phaseArgIdx);
  Value barrierIdx = forOp.getRegionIterArg(barrierIdxArgIdx);

  SmallVector<Value> newYieldOperands = {phase, barrierIdx};
  if (tmemUseNumStages > 1) {
    Value bufIdx = forOp.getRegionIterArg(bufIdxArgIdx);
    newYieldOperands.push_back(bufIdx);
  }
  cast<scf::YieldOp>(forOp.getBody()->getTerminator())
      .getResultsMutable()
      .append(newYieldOperands);

  createBarrierAndWaitOps(forOp, schedule, mma, mmaSelfLatency, alloc,
                          phaseArgIdx, barrierIdxArgIdx);

  if (tmemUseNumStages > 1) {
    multibufferTensorMemory(forOp, schedule, alloc, bufIdxArgIdx,
                            tmemUseNumStages);
  }

  return forOp;
}

scf::ForOp lowerMMAs(scf::ForOp forOp, CoarseSchedule &schedule) {
  SmallVector<ttng::MMAv5OpInterface> mmas;
  forOp.walk([&](ttng::MMAv5OpInterface mma) { mmas.push_back(mma); });
  for (auto mma : mmas) {
    forOp = lowerMMA(mma, forOp, schedule);
  }
  return forOp;
}

/////////////////////////////
// LOWER LOOP
/////////////////////////////

void lowerLoop(scf::ForOp forOp,
               triton::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  CoarseSchedule schedule;
  if (failed(schedule.deSerialize(forOp))) {
    return;
  }
  scf::ForOp newForOp = lowerMMAs(forOp, schedule);
  newForOp = lowerLoads(newForOp, schedule, axisInfoAnalysis);
  newForOp = lowerTMADescriptors(newForOp, schedule);
  schedule.serialize(newForOp);
}

} // namespace

void lowerLoops(ModuleOp moduleOp) {
  triton::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    lowerLoop(forOp, axisInfoAnalysis);
  }
}

} // namespace gpu
} // namespace triton
} // namespace mlir
