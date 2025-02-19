#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace gpu {

namespace {
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

DenseSet<Operation *> getTopLevelUsersInLoop(Operation *op, scf::ForOp forOp) {
  struct Use {
    Use(OpOperand &use)
        : op(use.getOwner()), operandNumber(use.getOperandNumber()) {}
    Operation *op;
    unsigned int operandNumber;
  };
  DenseSet<Operation *> topLevelUsers;
  SmallVector<Use> q(op->use_begin(), op->use_end());
  while (!q.empty()) {
    auto use = q.pop_back_val();
    auto yieldOp = dyn_cast<scf::YieldOp>(use.op);
    if (yieldOp && yieldOp->getParentOp() == forOp) {
      q.append(forOp.getRegionIterArgs()[use.operandNumber].use_begin(),
               forOp.getRegionIterArgs()[use.operandNumber].use_end());
      continue;
    }
    Operation *topLevelUser = forOp.getBody()->findAncestorOpInBlock(*use.op);
    topLevelUsers.insert(topLevelUser);
  }
  return topLevelUsers;
}

Operation *getFirstUseOfPipelinedOp(Operation *op, scf::ForOp forOp,
                                    CoarseSchedule &schedule) {
  Operation *firstUser = nullptr;
  DenseSet<Operation *> topLevelUsers = getTopLevelUsersInLoop(op, forOp);
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

int getDefUseStageDiff(Operation *op, scf::ForOp forOp,
                       CoarseSchedule &schedule) {
  assert(schedule.count(op) && "LoadOp not found in the schedule");
  auto [defStage, _] = schedule[op];
  std::optional<int> useStage;
  DenseSet<Operation *> topLevelUsers = getTopLevelUsersInLoop(op, forOp);
  for (Operation *topLevelUser : topLevelUsers) {
    auto [_useStage, _] = schedule[topLevelUser];
    useStage = std::min(_useStage, useStage.value_or(_useStage));
  }
  if (!useStage)
    return 0;
  assert(useStage >= defStage && "LoadOp used before defined");
  return useStage.value() - defStage;
}

ttg::SharedEncodingTrait getSharedEncoding(tt::LoadOp loadOp) {
  // Try to use local alloc encoding if possible.
  ttg::SharedEncodingTrait localAllocEnc;
  if (llvm::any_of(loadOp->getUsers(), [&](Operation *user) {
        return isa<ttg::LocalAllocOp>(user);
      })) {
    for (auto user : loadOp->getUsers()) {
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
        loadOp->emitRemark()
            << "Pipelining load with different use encodings. This will lead "
               "to layout conversions and performance degradation.";
        return localAllocEnc;
      }
    }
    if (localAllocEnc)
      return localAllocEnc;
  }

  // TODO pawel: Add the case for TMA loads.

  // Try to use dot encoding if possible.
  bool incompatible = false;
  localAllocEnc =
      getSharedEncIfAllUsersAreDotEnc(loadOp.getResult(), incompatible)
          .value_or(nullptr);

  if (localAllocEnc)
    return localAllocEnc;

  // Use generic layout. This won't be optimal for 2D tensors.
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  auto ctaLayout = ttg::getCTALayout(ty.getEncoding());
  auto blockedOrder = ttg::getOrder(ty.getEncoding());
  SmallVector<unsigned> order;
  if (blockedOrder.size() == 3) {
    for (unsigned i = 0; i < blockedOrder.size(); ++i) {
      if (blockedOrder[i] == 0)
        continue;
      order.push_back(blockedOrder[i]);
    }
    order.push_back(0);
  } else {
    order = blockedOrder;
  }
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
  return alloc;
}

template <typename BuilderT, typename... Args>
Operation *createWithStage(BuilderT &builder, Location loc, int stage,
                           CoarseSchedule::Cluster cluster, Args &&...args) {
  Operation *op = builder.template create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, std::forward<Args>(args)...);

  return op;
}

// Check if the load can be pipelined entirely in shared memory, with user
// consuming directly the shared memory, without going through registers.
bool canBeShmemPipelined(tt::LoadOp loadOp) {
  // AsyncCopyGlobalToLocalOp does not support the non-zero "other" value.
  // With consumer consuming directly the shared memory, there would be no way
  // to replace masked values with the "other" value.
  if (loadOp.getOther() && !isZeroConst(loadOp.getOther()))
    return false;

  if (!loadOp->hasOneUse())
    return false;
  Operation *user = *loadOp->getUsers().begin();
  if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
    return isa<ttg::NVMMASharedEncodingAttr>(alloc.getType().getEncoding());
  }
  return false;
}

void createAsyncCopy(scf::ForOp forOp, tt::LoadOp loadOp, Value alloc,
                     Value insertIdx, Value extractIdx,
                     CoarseSchedule &schedule) {
  OpBuilderForStage builder(forOp, schedule);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);

  Operation *firstUse = getFirstUseOfPipelinedOp(loadOp, forOp, schedule);
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

  // TODO pawel: think about optimizing the blocked layout of indirect loads
  // here

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

  if (canBeShmemPipelined(loadOp)) {
    auto user = *loadOp->getUsers().begin();
    assert(isa<triton::gpu::LocalAllocOp>(user));
    auto alloc = cast<ttg::LocalAllocOp>(user);
    tt::replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
    alloc.erase();
  } else {
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

    auto sharedLoad = builder.create<ttg::LocalLoadOp>(
        loc, loadOp.getType(), viewLoad, wait->getResult(0));
    auto result = sharedLoad->getResults();

    // Create a select for non-zero other values as they are not handled by
    // AsyncCopyGlobalToLocalOp for now.
    Value other = loadOp.getOther();
    if (other && !isZeroConst(other)) {
      auto select = builder.create<arith::SelectOp>(
          loc, loadOp.getType(),
          // Use the mask operand from the original load, not the one with a
          // potentially transformed layout.
          loadOp.getMask(), sharedLoad.getResult(), other);
      result = select->getResults();
    }

    loadOp->replaceAllUsesWith(result);

    // TODO: Think about prefetching the load for MMAv2
  }
  schedule.erase(loadOp);
  loadOp->erase();
}

scf::ForOp lowerLoads(scf::ForOp forOp, CoarseSchedule &schedule) {
  struct AsyncLoad {
    int stageDiff;
    Value alloc;
    Value barrier;
    SharedEncodingTrait sharedEncoding;
  };
  struct LoadGroupInfo {
    Value insertIdx;
    Value extractIdx;
    Value phase;
    bool hasTMALoad = false;
  };
  llvm::MapVector<tt::LoadOp, AsyncLoad> asyncLoads;
  llvm::MapVector<int, LoadGroupInfo> loadGroups;
  forOp.getBody()->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      int stageDiff = getDefUseStageDiff(loadOp, forOp, schedule);
      SharedEncodingTrait sharedEncoding = getSharedEncoding(loadOp);
      // Do not create async loads for small loads (cp.async requires at least 4
      // bytes)
      int copyVecBytes = getCopyVecBytes(
          cast<RankedTensorType>(loadOp.getType()), sharedEncoding);
      if (stageDiff > 0 && copyVecBytes >= 4) {
        if (canBeShmemPipelined(loadOp)) {
          // Allocate additional buffer required by the wgmma pipelining.
          stageDiff += 1;
        }
        asyncLoads[loadOp] = {.stageDiff = stageDiff,
                              .sharedEncoding = sharedEncoding};
      } else if (stageDiff > 1) {
        // Distance-1 loads can in most cases be pipelined in registers without
        // any performance degradation, as the schedule will usually reorder the
        // user and the producer so there is no liverange overlap, and no copy
        // needed.
        loadOp->emitRemark() << "Pipelining load that cannot use vectorized "
                                "copy. This will likely "
                                "lead to pipelining in registers and severe "
                                "performance degradation.";
      }
    } else if (isa<scf::ForOp>(op)) {
      // Skip nested loops.
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });

  if (asyncLoads.empty())
    return forOp;

  for (auto &[loadOp, asyncLoad] : asyncLoads) {
    Value alloc = createAlloc(forOp, loadOp, asyncLoad.sharedEncoding,
                              asyncLoad.stageDiff);
    asyncLoad.alloc = alloc;
    loadGroups.insert({asyncLoad.stageDiff, {}});
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
    // TODO: Add tma support / phase
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
    // TODO: Add tma support / phase

    // Create two counters for the insert and extract indices to avoid creating
    // long liverange.
    builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());

    Value numBuffersVal =
        builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
    insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
    Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 insertIdx, numBuffersVal);
    insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);
    loadGroup.insertIdx = insertIdx;

    extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
    // Duplicate the constant to keep it from being carried across loops.
    numBuffersVal = builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
    Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 extractIdx, numBuffersVal);
    extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);
    loadGroup.extractIdx = extractIdx;
  }

  for (auto [loadOp, asyncLoad] : asyncLoads) {
    auto [insertIdx, extractIdx, phase, _] = loadGroups[asyncLoad.stageDiff];
    createAsyncCopy(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                    schedule);
  }
  // Patch the yield with the updated counters. Subtract to account for the loop
  // counter.
  argIdx = newOperandIndex - 1;
  for (auto &[numBuffers, loadGroup] : loadGroups) {
    forYield.setOperand(argIdx++, loadGroup.insertIdx);
    forYield.setOperand(argIdx++, loadGroup.extractIdx);
  }

  // Automatically discover dependencies and schedule new insert/extract ops to
  // correct stages.
  scheduleDependencies(forOp, schedule);

  // Make sure all ops have attributes.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    assert(schedule.count(&op) && "op not found in the schedule");
  }
  return forOp;
}

void lowerLoop(scf::ForOp forOp) {
  CoarseSchedule schedule;
  if (failed(schedule.deSerialize(forOp))) {
    return;
  }
  scf::ForOp newForOp = lowerLoads(forOp, schedule);
  schedule.serialize(newForOp);
  LLVM_DEBUG({ DBGS() << "Loop after lowering loads:\n" << newForOp << "\n"; });
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
