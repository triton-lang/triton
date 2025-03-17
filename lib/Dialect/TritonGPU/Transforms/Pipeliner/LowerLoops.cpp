#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/Support/Debug.h"

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

class OpBuilderForStage : public OpBuilder {
  std::optional<int> _stage;
  std::optional<CoarseSchedule::Cluster> _cluster;
  CoarseSchedule &_schedule;

public:
  explicit OpBuilderForStage(Operation *op, CoarseSchedule &schedule, int stage,
                             CoarseSchedule::Cluster cluster)
      : OpBuilder(op, nullptr), _schedule(schedule), _stage(stage),
        _cluster(cluster) {}
  explicit OpBuilderForStage(Operation *op, CoarseSchedule &schedule)
      : OpBuilder(op, nullptr), _schedule(schedule) {
    if (_schedule.count(op)) {
      auto sc = _schedule[op];
      _stage = sc.first;
      _cluster = sc.second;
    }
  }
  void setStageCluster(std::pair<int, CoarseSchedule::Cluster> stageCluster) {
    _stage = stageCluster.first;
    _cluster = stageCluster.second;
  }

  template <typename OpTy, typename... Args> OpTy create(Args &&...args) {
    OpTy op = OpBuilder::create<OpTy>(std::forward<Args>(args)...);
    if (_stage && _cluster) {
      _schedule.insert(op, *_stage, *_cluster);
    }
    return op;
  }
};

bool isTMALoad(Operation *op) {
  return isa<tt::ExperimentalDescriptorLoadOp,
             tt::ExperimentalDescriptorGatherOp>(op);
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
           schedule.clusters.isBefore(_useCluster, _firstUserCluster))) {
        firstUser = topLevelUser;
      }
    }
  }
  return firstUser;
}

// Check if the load can be pipelined entirely in shared memory, with user
// consuming directly the shared memory, without going through registers.
bool canBeShmemPipelined(Operation *op) {
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    // AsyncCopyGlobalToLocalOp does not support the non-zero "other" value.
    // With consumer consuming directly the shared memory, there would be no way
    // to replace masked values with the "other" value.
    if (loadOp.getOther() && !isZeroConst(loadOp.getOther()))
      return false;
  }

  if (!op->hasOneUse())
    return false;
  Operation *user = *op->getUsers().begin();
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
    return isa<ttg::NVMMASharedEncodingAttr>(alloc.getType().getEncoding());
  }
  return false;
}

int getDefUseStageDiff(Operation *op, scf::ForOp forOp,
                       CoarseSchedule &schedule) {
  assert(schedule.count(op) && "Op not found in the schedule");
  auto [defStage, _] = schedule[op];
  // Special case for loads used by local_alloc:
  // we must consider the uses of the local_alloc, as it may be removed and its
  // uses will become direct uses of the async load.
  // TODO: This is overly conservative, we may need to restrict to cases where
  // local_alloc is used by a dot product and has correct encoding.
  if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp,
          tt::ExperimentalDescriptorGatherOp>(op) &&
      op->hasOneUse()) {
    if (auto localAlloc =
            dyn_cast<ttg::LocalAllocOp>(*op->getUsers().begin())) {
      return schedule[localAlloc].first - defStage +
             getDefUseStageDiff(localAlloc, forOp, schedule);
    }
  }
  std::optional<int> useStage;
  DenseSet<Operation *> topLevelUsers = getTopLevelUsersInLoop(op, forOp);
  DenseSet<Operation *> topLevelUsersOnlyWait = getTopLevelUsersInLoop(
      op, forOp, [](Operation *op) { return isa<ttng::WaitBarrierOp>(op); });
  for (Operation *topLevelUser : topLevelUsers) {
    auto [_useStage, _] = schedule[topLevelUser];
    useStage = std::min(_useStage, useStage.value_or(_useStage));
  }
  // Waits tells us the buffer is still in use until the wait completes, we
  // can't simply load from the buffer and replace the uses of the buffer with
  // the load. The stage diff needs to account for the furthest wait.
  for (Operation *topLevelUser : topLevelUsersOnlyWait) {
    auto [_useStage, _] = schedule[topLevelUser];
    useStage = std::max(_useStage, useStage.value_or(_useStage));
  }
  if (!useStage)
    return 0;
  assert(useStage >= defStage && "Op used before defined");
  return useStage.value() - defStage;
}

template <typename BuilderT>
Value createIncrementModulo(BuilderT &builder, Location loc, Value counter,
                            Value modulus, Value zero, Value one,
                            Value *outCond = nullptr) {
  Value addOne = builder.template create<arith::AddIOp>(loc, counter, one);
  Value inRangeCond = builder.template create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, addOne, modulus);
  if (outCond)
    *outCond = inRangeCond;
  return builder.template create<arith::SelectOp>(loc, inRangeCond, addOne,
                                                  zero);
}

/////////////////////////////
// LOWER LOADS
/////////////////////////////

ttg::SharedEncodingTrait getSharedEncoding(Operation *op) {
  // Try to use local alloc encoding if possible.
  ttg::SharedEncodingTrait localAllocEnc;
  if (llvm::any_of(op->getUsers(), [&](Operation *user) {
        return isa<ttg::LocalAllocOp>(user);
      })) {
    for (auto user : op->getUsers()) {
      auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
      if (!localAlloc)
        continue;
      auto enc = mlir::cast<ttg::SharedEncodingTrait>(
          localAlloc.getType().getEncoding());
      if (!localAllocEnc) {
        localAllocEnc = enc;
      }
      if (enc != localAllocEnc) {
        // Some users have different encoding than others.
        // Use one of the encodings, and warn about the performance issue.
        op->emitRemark()
            << "Pipelining load with different use encodings. This will lead "
               "to layout conversions and performance degradation.";
        continue;
      }
    }
  }

  auto ty = cast<RankedTensorType>(op->getResultTypes()[0]);
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  auto order = ttg::getOrder(ty);
  if (isTMALoad(op)) {
    // For TMA, the encoding compatible with it takes precedence over local
    // alloc created for the MMA operand.
    if (localAllocEnc) {
      if (auto sharedMMALayout =
              dyn_cast<ttg::NVMMASharedEncodingAttr>(localAllocEnc)) {
        assert(!sharedMMALayout.getFp4Padded() &&
               "TMA load for mixed precision MMAv5 is not supported yet.");
      }
    }
    return ttg::NVMMASharedEncodingAttr::get(
        ty.getContext(), ty.getShape(), order, ctaLayout, ty.getElementType(),
        /*fp4Padded*/ false);
  }

  if (localAllocEnc)
    return localAllocEnc;

  // Try to use dot encoding if possible.
  bool incompatible = false;
  localAllocEnc =
      getSharedEncIfAllUsersAreDotEnc(op->getResult(0), incompatible)
          .value_or(nullptr);

  if (localAllocEnc)
    return localAllocEnc;

  // Use generic layout. This won't be optimal for 2D tensors.
  return ttg::SwizzledSharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order,
                                              ctaLayout);
}

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, Operation *loadOp,
                         ttg::SharedEncodingTrait sharedEnc,
                         unsigned distance) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type memdescType = ttg::MemDescType::get(bufferShape, ty.getElementType(),
                                           sharedEnc, sharedMemorySpace,
                                           /*mutableMemory=*/true);
  Value alloc =
      builder.create<ttg::LocalAllocOp>(loadOp->getLoc(), memdescType);

  builder.setInsertionPointAfter(forOp);
  builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);
  return alloc;
}

template <typename BuilderT, typename... Args>
Operation *createWithStage(BuilderT &builder, Location loc, int stage,
                           CoarseSchedule::Cluster cluster, Args &&...args) {
  Operation *op = builder.template create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, std::forward<Args>(args)...);

  return op;
}

void createAsyncCopy(scf::ForOp forOp, tt::LoadOp loadOp, Value alloc,
                     Value insertIdx, Value extractIdx,
                     CoarseSchedule &schedule) {
  OpBuilderForStage builder(forOp, schedule);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);

  Operation *firstUse = getFirstUseOfPipelinedOp({loadOp}, forOp, schedule);
  assert(firstUse && "LoadOp has no users");
  // Replace the load with async copy, wait and loal_load.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loadOp);
  builder.setStageCluster(schedule[loadOp]);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();
  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());

  // Create async copy
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  ttg::MemDescType subviewTy = ttg::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true,
      /*allocShape=*/allocTy.getAllocShape());
  auto view =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);
  Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, src, view, mask, other, loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile());
  Operation *commit =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copy->getResult(0));

  // Create wait and local load
  builder.setStageCluster(schedule[firstUse]);
  Operation *wait =
      builder.create<ttg::AsyncWaitOp>(loc, commit->getResult(0), 0);
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);

  if (!loadOp.getOther() || isZeroConst(loadOp.getOther())) {
    // Remove redundant local_load -> local_alloc, but only if
    // we are not using the other value. AsyncCopyGlobalToLocalOp does not
    // support the masking.
    SmallVector<ttg::LocalAllocOp> allocsToErase;
    for (Operation *user : loadOp->getUsers()) {
      if (auto userAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        if (allocTy.getEncoding() == userAlloc.getType().getEncoding()) {
          tt::replaceUsesAndPropagateType(builder, userAlloc,
                                          viewLoad.getResult());
          allocsToErase.push_back(userAlloc);
        }
      }
    }
    for (auto alloc : allocsToErase) {
      alloc.erase();
    }
  }

  // If there are some uses that were not local_allocs, we need to create a
  // local_load for them.
  if (loadOp->use_begin() != loadOp->use_end()) {
    auto sharedLoad = builder.create<ttg::LocalLoadOp>(
        loc, loadOp.getType(), viewLoad, wait->getResult(0));
    auto result = sharedLoad->getResults();

    // Create a select for non-zero other values as they are not handled by
    // AsyncCopyGlobalToLocalOp for now.
    if (other && !isZeroConst(other)) {
      auto select = builder.create<arith::SelectOp>(
          loc, loadOp.getType(),
          // Use the mask operand from the original load, not the one with a
          // potentially transformed layout.
          loadOp.getMask(), sharedLoad.getResult(), other);
      result = select->getResults();
    }
    loadOp->replaceAllUsesWith(result);
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
  OpBuilderForStage builder(forOp, schedule);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);

  Operation *firstUse = getFirstUseOfPipelinedOp({loadOp}, forOp, schedule);
  assert(firstUse && "LoadOp has no users");
  Attribute sharedMemorySpace =
      ttg::SharedMemorySpaceAttr::get(forOp.getContext());
  Location loc = loadOp->getLoc();

  builder.setInsertionPoint(loadOp);
  builder.setStageCluster(schedule[loadOp]);
  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());

  // Create async copy
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  ttg::MemDescType subviewTy = ttg::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true,
      /*allocShape=*/allocTy.getAllocShape());
  auto view =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);

  Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  Value tmaPtr =
      builder.create<triton::nvidia_gpu::TensorDescToTMAPtrOp>(loc, desc);
  createCopy(builder, tmaPtr, barrier, view, pred);

  // Create local load after the wait
  builder.setInsertionPointAfter(waitOp);
  builder.setStageCluster(schedule[firstUse]);
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  // Remove redundant local_load -> local_alloc
  SmallVector<ttg::LocalAllocOp> allocsToErase;
  for (Operation *user : loadOp->getUsers()) {
    if (auto userAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      if (allocTy.getEncoding() == userAlloc.getType().getEncoding()) {
        tt::replaceUsesAndPropagateType(builder, userAlloc,
                                        viewLoad.getResult());
        allocsToErase.push_back(userAlloc);
      }
    }
  }
  for (auto alloc : allocsToErase) {
    alloc.erase();
  }

  // If there are some uses that were not local_allocs, we need to create a
  // local_load for them.
  if (loadOp->use_begin() != loadOp->use_end()) {
    auto sharedLoad = builder.create<ttg::LocalLoadOp>(
        loc, loadOp->getResultTypes().front(), viewLoad);
    auto result = sharedLoad->getResults();
    loadOp->replaceAllUsesWith(result);
  }
  schedule.erase(loadOp);
  loadOp->erase();
}

void createTMAAsyncLoad(scf::ForOp forOp,
                        tt::ExperimentalDescriptorLoadOp loadOp, Value alloc,
                        Value insertIdx, Value extractIdx, Value barrier,
                        Operation *waitOp, CoarseSchedule &schedule) {
  return createTMAAsyncCopy(forOp, loadOp, loadOp.getDesc(), alloc, insertIdx,
                            extractIdx, barrier, waitOp, schedule,
                            [&](OpBuilderForStage &builder, Value tmaPtr,
                                Value barrier, Value view, Value pred) {
                              builder.create<ttng::AsyncTMACopyGlobalToLocalOp>(
                                  loadOp.getLoc(), tmaPtr, loadOp.getIndices(),
                                  barrier, view, pred);
                            });
}

void createTMAAsyncGather(scf::ForOp forOp,
                          tt::ExperimentalDescriptorGatherOp gatherOp,
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
        if (canBeShmemPipelined(loadOp)) {
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
      int loadSize = product(tensorTy.getShape());
      sizeInBytes +=
          loadSize * tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
    }

    Value barrierAlloc = triton::createBarrierAlloc(forOp, numBuffers);
    Location loc = forOp.getLoc();
    OpBuilderForStage builder(group[0], schedule);
    Value barrier = triton::createSingleBufferView(builder, barrierAlloc,
                                                   loadGroup.insertIdx);
    Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    builder.create<ttng::BarrierExpectOp>(loc, barrier, sizeInBytes, pred);

    builder.setInsertionPointAfter(group.back());
    Operation *firstUse = getFirstUseOfPipelinedOp(group, forOp, schedule);
    builder.setStageCluster(schedule[firstUse]);
    Value barrierViewWait = triton::createSingleBufferView(
        builder, barrierAlloc, loadGroup.extractIdx);
    auto wait = builder.create<ttng::WaitBarrierOp>(loc, barrierViewWait,
                                                    loadGroup.phase);

    // Update the async loads info.
    for (Operation *op : group) {
      asyncLoads[op].barrier = barrier;
      asyncLoads[op].waitOp = wait;
    }

    triton::createBarrierDealloc(forOp, barrierAlloc, numBuffers);
  }
}

// Check if load requires additional buffer for a mma pipelining
bool loadRequiresAdditionalBuffer(Operation *loadOp) {
  auto skipViewOps = [](Operation *op) -> Operation * {
    while (op->hasOneUse() && isa<ttg::MemDescTransOp>(op)) {
      op = *op->getUsers().begin();
    }
    return op;
  };
  // Pattern match the op sequence used loading mmav3 operands
  if (canBeShmemPipelined(loadOp) && loadOp->hasOneUse()) {
    ttg::LocalAllocOp alloc =
        dyn_cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin());
    if (alloc && alloc->hasOneUse()) {
      if (isa<ttng::WarpGroupDotOp>(skipViewOps(*alloc->getUsers().begin()))) {
        return true;
      }
    }
  }
  return false;
}

scf::ForOp lowerLoads(scf::ForOp forOp, CoarseSchedule &schedule) {
  llvm::MapVector<Operation *, AsyncLoad> asyncLoads;
  llvm::MapVector<int, LoadGroupInfo> loadGroups;
  // Only visit the top level ops, we do not support pipelining conditional
  // loads for now
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp,
            tt::ExperimentalDescriptorGatherOp>(op)) {
      int stageDiff = getDefUseStageDiff(&op, forOp, schedule);
      if (stageDiff == 0 || !isa<RankedTensorType>(op.getResultTypes()[0])) {
        // Don't care about non-pipelined loads. Don't use async loads for
        // scalar values.
        continue;
      }
      SharedEncodingTrait sharedEncoding = getSharedEncoding(&op);
      // Do not create async loads for small loads (cp.async requires at least 4
      // bytes)
      int copyVecBytes = getCopyVecBytes(
          cast<RankedTensorType>(op.getResultTypes()[0]), sharedEncoding);
      if (copyVecBytes >= 4 || isTMALoad(&op)) {
        if (loadRequiresAdditionalBuffer(&op)) {
          // Allocate additional buffer required by the wgmma pipelining.
          stageDiff += 1;
        }
        asyncLoads[&op] = {.stageDiff = stageDiff,
                           .sharedEncoding = sharedEncoding};
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
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;

  // Update yield op with temporary yield values
  auto forYield = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  for (unsigned i = 0; i < newOperands.size(); ++i) {
    forYield.getResultsMutable().append(newOperands[i]);
  }

  builder.setInsertionPoint(forOp);
  loc = forOp.getLoc();
  int argIdx = newOperandIndex;
  for (auto &[numBuffers, loadGroup] : loadGroups) {
    Value insertIdx = newForOp.getBody()->getArgument(argIdx);
    argIdx++;
    Value extractIdx = newForOp.getBody()->getArgument(argIdx);
    argIdx++;
    Value phase = nullptr;
    if (loadGroup.hasTMALoad) {
      phase = newForOp.getBody()->getArgument(argIdx);
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
      phase = builder.create<arith::SelectOp>(loc, cndExt, phase, nextPhase);
      loadGroup.phase = phase;
    }
  }

  createTMABarrierAndWait(forOp, asyncLoads, loadGroups, schedule);

  for (auto [op, asyncLoad] : asyncLoads) {
    auto [insertIdx, extractIdx, phase, _] = loadGroups[asyncLoad.stageDiff];
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      createAsyncCopy(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                      schedule);
    } else if (auto loadOp = dyn_cast<tt::ExperimentalDescriptorLoadOp>(op)) {
      createTMAAsyncLoad(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                         asyncLoad.barrier, asyncLoad.waitOp, schedule);
    } else if (auto loadOp = dyn_cast<tt::ExperimentalDescriptorGatherOp>(op)) {
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

  // Insert sync point for any possibly outstanding loads after the loop. This
  // can happen as we speculatively execute loads in the loop.
  builder.setInsertionPointAfter(forOp);
  builder.create<ttg::AsyncWaitOp>(loc, ValueRange({}), 0);

  // Make sure all ops have attributes.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    assert(schedule.count(&op) && "op not found in the schedule");
  }
  return forOp;
}

/////////////////////////////
// LOWER TMA DESCRIPTORS
/////////////////////////////

LogicalResult
allocTMABuffers(scf::ForOp forOp,
                llvm::MapVector<Operation *, Value> &tmaBufferMapping,
                int numStages) {
  IRRewriter rewriter(forOp);

  // Create a multi-buffered allocation for each MakeTensorDescOp call in the
  // loop
  forOp.walk([&](tt::MakeTensorDescOp op) {
    // TODO peter: walk to loop yield to find the init value if this is a
    // loop-carried value. That would save us from allocating another buffer
    // just for the init value
    auto loc = op.getLoc();
    Value alloc = rewriter.create<triton::gpu::GlobalScratchAllocOp>(
        loc, triton::getPointerType(rewriter.getI8Type()),
        numStages * ttng::TMA_SIZE_BYTES, ttng::TMA_ALIGN);
    tmaBufferMapping[op.getOperation()] = alloc;
  });
  return success();
}

template <typename BuilderT>
Value subviewTMADescriptor(BuilderT &builder, Location loc, Value alloc,
                           Value counter) {
  Value tmaSizeVal = builder.template create<arith::ConstantIntOp>(
      loc, ttng::TMA_SIZE_BYTES, 32);
  Value offset =
      builder.template create<arith::MulIOp>(loc, tmaSizeVal, counter);
  return builder.template create<triton::AddPtrOp>(loc, alloc.getType(), alloc,
                                                   offset);
}

LogicalResult rewriteTMABufferUpdates(
    scf::ForOp forOp,
    const llvm::MapVector<Operation *, Value> &tmaBufferMapping,
    ArrayRef<BlockArgument> tmaCounters, int numStages, Value one, Value zero,
    CoarseSchedule &schedule) {
  assert(tmaBufferMapping.size() == tmaCounters.size());

  Value numStagesVal = mlir::OpBuilder(forOp).create<arith::ConstantIntOp>(
      forOp.getLoc(), numStages, 32);

  for (auto [iOp, pair] : llvm::enumerate(tmaBufferMapping)) {
    auto &[op, alloc] = pair;

    // Rewriter MakeTensorDescOp as writing a TMA descriptor
    auto makeDescOp = cast<tt::MakeTensorDescOp>(op);

    OpBuilderForStage stageBuilder(makeDescOp, schedule);
    auto loc = makeDescOp.getLoc();

    BlockArgument counter = tmaCounters[iOp];
    Value nextBuf = subviewTMADescriptor(stageBuilder, loc, alloc, counter);
    if (failed(ttng::createTMADesc(nextBuf, makeDescOp, stageBuilder))) {
      return failure();
    }
    stageBuilder.create<triton::ExperimentalTensormapFenceproxyAcquireOp>(
        loc, nextBuf);
    Value nextDesc = stageBuilder.create<triton::ReinterpretTensorDescOp>(
        loc, makeDescOp.getType(), nextBuf);

    makeDescOp.getResult().replaceAllUsesWith(nextDesc);

    // Increment the buffer index counter
    Value nextCounter = createIncrementModulo(stageBuilder, loc, counter,
                                              numStagesVal, zero, one);

    // If we are in a (potentially nested) if region, propagate the counter
    // up to the main for op body scope
    Operation *curOp = op;
    Operation *parent = op->getParentOp();
    while (parent != forOp.getOperation()) {
      auto ifOp = dyn_cast<scf::IfOp>(parent);
      if (!ifOp) {
        std::string msg;
        llvm::raw_string_ostream ss(msg);
        ss << "Cannot pipeline MakeTensorDescOp inside:\n";
        parent->print(ss);
        ss << "\nOnly scf.if regions are supported";
        return makeDescOp->emitOpError(std::move(msg));
      }

      IRRewriter rewriter(parent);
      auto newIfOp =
          replaceIfOpWithNewSignature(rewriter, ifOp, {nextCounter.getType()});

      auto yieldNewBlock = newIfOp.thenBlock();
      auto yieldOldBlock = newIfOp.elseBlock();

      if (yieldNewBlock != curOp->getBlock()) {
        std::swap(yieldNewBlock, yieldOldBlock);
      }
      cast<scf::YieldOp>(yieldNewBlock->getTerminator())
          .getResultsMutable()
          .append(nextCounter);
      cast<scf::YieldOp>(yieldOldBlock->getTerminator())
          .getResultsMutable()
          .append(counter);

      ifOp.erase();
      nextCounter = newIfOp.getResults().back();
      curOp = newIfOp;
      parent = newIfOp->getParentOp();
    }

    // Finally, rewrite the loop level yield
    auto forYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    forYield.setOperand(counter.getArgNumber() - 1, nextCounter);
  }
  return success();
}

scf::ForOp lowerTMADescriptors(scf::ForOp forOp, CoarseSchedule &schedule) {
  llvm::MapVector<Operation *, Value> tmaBufferMapping;
  if (failed(
          allocTMABuffers(forOp, tmaBufferMapping, schedule.getNumStages()))) {
    llvm_unreachable("TMA pipelining failed");
  }

  IRRewriter builder(forOp);
  Location loc = forOp.getLoc();
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  SmallVector<Value> newOperands;
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Create one counter per TMA buffer. This allows the descriptors to be
  // updated independently without needing to write duplicate of existing tma
  // descriptors.
  unsigned tmaCounterArgsStartIdx = newOperandIndex + newOperands.size();
  for (int i = 0; i < tmaBufferMapping.size(); ++i) {
    newOperands.push_back(zero);
  }

  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;

  auto tmaCounters = ArrayRef<BlockArgument>(newForOp.getBody()->getArguments())
                         .slice(tmaCounterArgsStartIdx);

  // Update yield op with temporary yield values
  auto forYield = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
  for (unsigned i = 0; i < newOperands.size(); ++i) {
    forYield.getResultsMutable().append(newOperands[i]);
  }

  if (failed(rewriteTMABufferUpdates(newForOp, tmaBufferMapping, tmaCounters,
                                     schedule.getNumStages(), one, zero,
                                     schedule))) {
    llvm_unreachable("Failed to rewrite TMA ops");
  }
  return newForOp;
}

/////////////////////////////
// LOWER MMA
/////////////////////////////

std::pair<int, int> getTmemUseStageBounds(ttng::TMEMAllocOp alloc,
                                          scf::ForOp forOp,
                                          CoarseSchedule &schedule) {
  std::pair<int, int> bounds = {std::numeric_limits<int>::max(),
                                std::numeric_limits<int>::min()};
  for (auto user : alloc->getUsers()) {
    if (!forOp->isAncestor(user->getParentOp())) {
      continue;
    }
    auto topLevelUser = forOp.getBody()->findAncestorOpInBlock(*user);
    if (schedule[topLevelUser].first < bounds.first) {
      bounds.first = schedule[topLevelUser].first;
    }
    if (schedule[topLevelUser].first > bounds.second) {
      bounds.second = schedule[topLevelUser].first;
    }
  }
  assert(bounds.first <= bounds.second && "Invalid stage bounds");
  return bounds;
}

// TODO: clean up the references to the phase and barrierIdx, this is a mess
void createBarrierAndWaitOps(scf::ForOp forOp, CoarseSchedule &schedule,
                             ttng::MMAv5OpInterface mma,
                             ttng::TMEMAllocOp alloc, Value &phase,
                             Value &barrierIdx, int numStages,
                             Value numStagesVal, Value zero, Value one) {
  OpBuilderForStage builder(mma, schedule);
  builder.setInsertionPoint(forOp);
  Value barrierAlloc = createBarrierAlloc(forOp, numStages);

  builder.setInsertionPoint(mma);
  Location loc = mma->getLoc();
  Value barWrap;
  barrierIdx = createIncrementModulo(builder, loc, barrierIdx, numStagesVal,
                                     zero, one, &barWrap);
  phase = builder.create<arith::SelectOp>(
      loc, phase.getType(), barWrap,
      builder.create<arith::XOrIOp>(loc, phase, one), phase);
  Value barrierSlice =
      triton::createSingleBufferView(builder, barrierAlloc, barrierIdx);
  mma.setBarrier(barrierSlice);

  builder.setInsertionPointAfter(mma);
  auto [mmaStage, mmaCluster] = schedule[mma];
  // Put wait in the next stage after the MMA even if other users of the mma
  // are in the same stage, or there are no users.
  int waitStage = mmaStage + numStages - 1;
  builder.setStageCluster({waitStage, mmaCluster});
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
  builder.create<ttng::WaitBarrierOp>(loc, barrierSlice, phase, waitBuffers);

  createBarrierDealloc(forOp, barrierAlloc, numStages);

  // Look for loads from the accumulator in stages earlier than the wait
  // and insert a barrier before them
  for (auto user : alloc->getUsers()) {
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      auto topLevelUser = forOp.getBody()->findAncestorOpInBlock(*load);
      if (topLevelUser && schedule[topLevelUser].first < waitStage) {
        // Put the wait in the same stage as the load
        builder.setInsertionPoint(load);
        builder.setStageCluster(schedule[load]);
        builder.create<ttng::WaitBarrierOp>(loc, barrierSlice, phase,
                                            waitBuffers);
      }
    }
  }
}

ttng::TMEMAllocOp createTMemAlloc(OpBuilder &builder,
                                  ttng::TMEMAllocOp oldTMemAllocOp,
                                  bool multiBufferred, int numStages) {
  Location loc = oldTMemAllocOp.getLoc();
  auto oldRetType = oldTMemAllocOp.getType();
  SmallVector<int64_t> shape = {oldRetType.getShape().begin(),
                                oldRetType.getShape().end()};
  if (multiBufferred) {
    shape.insert(shape.begin(), numStages);
  }
  Type accMemDescType = triton::gpu::MemDescType::get(
      shape, oldRetType.getElementType(), oldRetType.getEncoding(),
      oldRetType.getMemorySpace(), /*mutableMemory=*/true);
  return builder.create<ttng::TMEMAllocOp>(oldTMemAllocOp.getLoc(),
                                           accMemDescType, nullptr);
}

void multibufferTensorMemory(scf::ForOp forOp, CoarseSchedule &schedule,
                             ttng::MMAv5OpInterface mma,
                             ttng::TMEMAllocOp alloc, Value &bufIdx,
                             int bufIdxArgIdx, int tmemUseNumStages) {
  SmallVector<Operation *> allocUsers = llvm::to_vector(alloc->getUsers());
  // If we are multibuffering, we require that a store is present in the loop,
  // as this is the only point we can change the accumulator buffer.
  assert(
      llvm::any_of(allocUsers,
                   [](Operation *op) { return isa<ttng::TMEMStoreOp>(op); }) &&
      "No tmem_store found in the loop");

  OpBuilder builder(alloc);
  auto newAlloc = createTMemAlloc(builder, alloc, true, tmemUseNumStages);
  Value numStagesVal = builder.create<arith::ConstantIntOp>(
      forOp.getLoc(), tmemUseNumStages, 32);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
  // Put the bufIdx increment at the beginning of the loop
  builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  bufIdx = createIncrementModulo(builder, forOp.getLoc(), bufIdx, numStagesVal,
                                 zero, one);
  for (auto user : allocUsers) {
    if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
      if (forOp->isAncestor(store)) {
        OpBuilderForStage stageBuilder(store, schedule);
        auto tmemSlice =
            triton::createSingleBufferView(stageBuilder, newAlloc, bufIdx);
        store.getDstMutable().assign(tmemSlice);
        stageBuilder.setInsertionPointAfter(store);
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
        OpBuilderForStage stageBuilder(load, schedule);
        auto tmemSlice =
            triton::createSingleBufferView(stageBuilder, newAlloc, bufIdx);
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
      OpBuilderForStage stageBuilder(mma, schedule);
      auto tmemSlice =
          triton::createSingleBufferView(stageBuilder, newAlloc, bufIdx);
      mma.setAccumulator(tmemSlice);
    } else {
      llvm::errs() << "Unsupported user of the accumulator: " << *user << "\n";
      llvm::report_fatal_error("Unsupported user of the accumulator");
    }
  }
  alloc->erase();
}

scf::ForOp lowerMMA(ttng::MMAv5OpInterface mma, scf::ForOp forOp,
                    CoarseSchedule &schedule) {
  auto isLoadPipelineable = [&](Operation *op) {
    return schedule[mma].first > schedule[op].first;
  };
  if (!mmaHasPipelineableOperands(mma, forOp, isLoadPipelineable)) {
    return forOp;
  }
  auto alloc = mma.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
  if (!alloc) {
    return forOp;
  }

  // Create barrier and wait ops
  std::pair<int, int> tmemUseStageBounds =
      getTmemUseStageBounds(alloc, forOp, schedule);
  int tmemUseNumStages =
      tmemUseStageBounds.second - tmemUseStageBounds.first + 1;
  int waitNumStages = tmemUseStageBounds.second - schedule[mma].first + 1;
  if (waitNumStages == 1 && !hasAccReadModifyWrite(mma, forOp)) {
    // Overlap the mma with itself, even if there is no use of the accumulator
    // after the mma
    waitNumStages = 2;
  }

  OpBuilder builder(forOp);
  Value minusOne = builder.create<arith::ConstantIntOp>(forOp.getLoc(), -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 1, 32);
  Value waitNumStagesVal =
      builder.create<arith::ConstantIntOp>(forOp.getLoc(), waitNumStages, 32);

  // Add arguments to the forOp
  unsigned newOperandIndex = forOp.getInitArgs().size();
  SmallVector<Value> newOperands = {
      one,      // phase
      minusOne, // barrierIdx
      minusOne, // bufIdx
  };
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;

  Value phase = forOp.getRegionIterArg(newOperandIndex + 0);
  Value barrierIdx = forOp.getRegionIterArg(newOperandIndex + 1);
  Value bufIdx = forOp.getRegionIterArg(newOperandIndex + 2);

  if (waitNumStages > 1) {
    createBarrierAndWaitOps(forOp, schedule, mma, alloc, phase, barrierIdx,
                            waitNumStages, waitNumStagesVal, zero, one);
  }

  if (tmemUseNumStages > 1) {
    multibufferTensorMemory(forOp, schedule, mma, alloc, bufIdx,
                            newOperandIndex + 2, tmemUseNumStages);
  }

  SmallVector<Value> newYieldOperands;
  newYieldOperands.push_back(phase);
  newYieldOperands.push_back(barrierIdx);
  newYieldOperands.push_back(bufIdx);
  appendToForOpYield(forOp, newYieldOperands);

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

void lowerLoop(scf::ForOp forOp) {
  CoarseSchedule schedule;
  if (failed(schedule.deSerialize(forOp))) {
    return;
  }
  scf::ForOp newForOp = lowerMMAs(forOp, schedule);
  newForOp = lowerLoads(newForOp, schedule);
  newForOp = lowerTMADescriptors(newForOp, schedule);
  schedule.serialize(newForOp);
}

} // namespace

void lowerLoops(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    lowerLoop(forOp);
  }
}

} // namespace gpu
} // namespace triton
} // namespace mlir
