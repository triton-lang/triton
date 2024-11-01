#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

#include <list>

#define DEBUG_TYPE "triton-matmul-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// TODO: We can extra some helpers into common utilities once we add more
// schedules.

namespace {

struct LoadInfo {
  // Layout of the data in the shared memory.
  ttg::SharedEncodingAttr sharedEncoding = nullptr;
  // Blocked encoding is used for loads not used by the dot.
  ttg::BlockedEncodingAttr blockedEncoding = nullptr;
  bool loadIsMMAV3 = false;
  int distToUse = 0;
  bool usedByDot = false;
};

} // namespace

class OpBuilderWithStage : public OpBuilder {
public:
  explicit OpBuilderWithStage(Operation *op,
                              OpBuilder::Listener *listener = nullptr)
      : OpBuilder(op, listener) {}
  explicit OpBuilderWithStage(Region &region, Listener *listener = nullptr)
      : OpBuilder(region, listener) {}

  template <typename OpTy, typename... Args>
  OpTy createWithStage(Location location, int stage, int cluster,
                       Args &&...args) {
    OpTy op = OpBuilder::create<OpTy>(location, std::forward<Args>(args)...);
    auto ctx = getContext();
    op->setAttr(mlir::triton::kLoopStageAttrName,
                IntegerAttr::get(IntegerType::get(ctx, 32), stage));
    op->setAttr(mlir::triton::kLoopClusterAttrName,
                IntegerAttr::get(IntegerType::get(ctx, 32), cluster));
    return op;
  }
  using OpBuilder::create;
};

static std::pair<int, int> getStageCluster(Operation *op) {
  auto stage = cast<IntegerAttr>(op->getAttr(mlir::triton::kLoopStageAttrName))
                   .getValue()
                   .getSExtValue();
  auto clusterId =
      cast<IntegerAttr>(op->getAttr(mlir::triton::kLoopClusterAttrName))
          .getValue()
          .getSExtValue();
  return std::make_pair(stage, clusterId);
}

static bool sameStageCluster(Operation *op1, Operation *op2) {
  auto [s1, c1] = getStageCluster(op1);
  auto [s2, c2] = getStageCluster(op2);
  return s1 == s2 && c1 == c2;
}

static void setStageCluster(scf::ForOp &forOp, Operation *op, int stage,
                            int cluster) {
  auto ctx = forOp.getContext();
  op->setAttr(mlir::triton::kLoopStageAttrName,
              IntegerAttr::get(IntegerType::get(ctx, 32), stage));
  op->setAttr(mlir::triton::kLoopClusterAttrName,
              IntegerAttr::get(IntegerType::get(ctx, 32), cluster));
}

// Return the minClusterId and maxClusterId for the given ForOp.
static std::pair<int, int> getMinMaxCluster(scf::ForOp &forOp) {
  int minClusterId = -1, maxClusterId = -1;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (!op.hasAttr(mlir::triton::kLoopStageAttrName) ||
        !op.hasAttr(mlir::triton::kLoopClusterAttrName))
      continue;
    auto [_, cluster] = getStageCluster(&op);
    if (maxClusterId < 0) {
      minClusterId = cluster;
      maxClusterId = cluster;
      continue;
    }
    maxClusterId = cluster > maxClusterId ? cluster : maxClusterId;
    minClusterId = cluster < minClusterId ? cluster : minClusterId;
  }
  return std::make_pair(minClusterId, maxClusterId);
}

// Return user of a loadOp with the lowest stage, if two users have the
// same stage, return the user with lower cluster.
static Operation *getFirstUseOfPipelinedLoad(Operation *loadOp) {
  Operation *firstUser = nullptr;
  for (Operation *user : loadOp->getUsers()) {
    if (user->getBlock() == loadOp->getBlock()) {
      auto [stage, clusterId] = getStageCluster(user);
      // Update FirstUse if this use has lower stage or lower cluster.
      if (!firstUser)
        firstUser = user;
      else {
        auto [stageForFirstUse, clusterForFirstUse] =
            getStageCluster(firstUser);
        if (stage < stageForFirstUse ||
            (stage == stageForFirstUse && clusterId < clusterForFirstUse))
          firstUser = user;
      }
    }
  }
  return firstUser;
}

static int createAsyncCopy(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                           Value insertIdx, Value extractIdx,
                           llvm::MapVector<Operation *, LoadInfo> &loadToInfo,
                           int numStages, int maxClusterId) {
  int retCode = -1;
  OpBuilderWithStage builder(forOp);
  auto [stage, clusterId] = getStageCluster(loadOp);
  auto *firstUse = getFirstUseOfPipelinedLoad(loadOp);
  auto [stageForFirstUse, clusterForFirstUse] = getStageCluster(firstUse);

  Value zero = builder.createWithStage<arith::ConstantIntOp>(
      forOp.getLoc(), stage, clusterId, 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();
  if (!isExpensiveLoadOrStore(loadOp) && loadToInfo[loadOp].blockedEncoding) {
    // For inexpensive loads that do not directly feed into dot ops
    // we want to use optimal layout for the data.
    ttg::BlockedEncodingAttr encoding = loadToInfo[loadOp].blockedEncoding;
    auto convertBlockLayout = [&](Value src) {
      auto ty = cast<RankedTensorType>(src.getType());
      auto newTy =
          RankedTensorType::get(ty.getShape(), ty.getElementType(), encoding);
      auto cvt = builder.createWithStage<ttg::ConvertLayoutOp>(
          loadOp->getLoc(), stage, clusterId, newTy, src);
      return cvt.getResult();
    };
    src = convertBlockLayout(src);
    if (mask)
      mask = convertBlockLayout(mask);
    if (other)
      other = convertBlockLayout(other);
  }

  tt::MemDescType allocTy = cast<tt::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  tt::MemDescType subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto view = builder.createWithStage<ttg::MemDescSubviewOp>(
      loc, stage, clusterId, subviewTy, alloc, copyOffsets);
  Operation *copy = builder.createWithStage<ttg::AsyncCopyGlobalToLocalOp>(
      loc, stage, clusterId, src, view, mask, other, loadOp.getCache(),
      loadOp.getEvict(), loadOp.getIsVolatile());
  Operation *commmit = builder.createWithStage<ttg::AsyncCommitGroupOp>(
      loc, stage, clusterId, copy->getResult(0));
  Operation *wait = builder.createWithStage<ttg::AsyncWaitOp>(
      loc, stageForFirstUse, clusterForFirstUse, commmit->getResult(0), 0);

  bool isMMV3Load = loadToInfo[loadOp].loadIsMMAV3;

  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad = builder.createWithStage<ttg::MemDescSubviewOp>(
      loc, stageForFirstUse, clusterForFirstUse, subviewTy, alloc, loadOffsets);
  if (isMMV3Load) {
    auto alloc = cast<ttg::LocalAllocOp>((*loadOp->getUsers().begin()));
    replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
    alloc.erase();
  } else {
    SmallVector<ttg::LocalAllocOp> allocsToErase;
    for (Operation *user : loadOp->getUsers()) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
        allocsToErase.push_back(alloc);
      }
    }
    for (auto alloc : allocsToErase) {
      alloc.erase();
    }

    auto sharedLoad = builder.createWithStage<ttg::LocalLoadOp>(
        loc, stageForFirstUse, clusterForFirstUse, loadOp.getType(), viewLoad,
        wait->getResult(0));
    auto result = sharedLoad->getResults();

    // Create a select for non-zero other values as they are not handled by
    // AsyncCopyGlobalToLocalOp for now.
    Value other = loadOp.getOther();
    if (other && !isZeroConst(other)) {
      auto select = builder.createWithStage<arith::SelectOp>(
          loc, stageForFirstUse, clusterForFirstUse, loadOp.getType(), mask,
          sharedLoad.getResult(), other);
      result = select->getResults();
    }

    loadOp->replaceAllUsesWith(result);

    // Prefetch load if is not MMAV3 and is used by the dot.
    if (loadToInfo[loadOp].usedByDot) {
      assert(stageForFirstUse >= 1);
      setStageCluster(forOp, wait, stageForFirstUse - 1, maxClusterId + 1);
      setStageCluster(forOp, viewLoad, stageForFirstUse - 1, maxClusterId + 1);
      retCode = stageForFirstUse - 1;
    }
  }
  loadOp.erase();
  return retCode;
}

static void
createTMAAsyncCopy(scf::ForOp &forOp, tt::ExperimentalDescriptorLoadOp loadOp,
                   Value alloc, Value insertIdx, Value extractIdx,
                   Value barrier, Operation *waitOp, Value phase,
                   llvm::MapVector<Operation *, LoadInfo> &loadToInfo,
                   int numStages) {
  assert(phase && "Phase value is required for TMA async copy.");
  OpBuilderWithStage builder(forOp);
  auto [stage, clusterId] = getStageCluster(loadOp);
  auto *firstUse = getFirstUseOfPipelinedLoad(loadOp);
  auto [stageForFirstUse, clusterForFirstUse] = getStageCluster(firstUse);

  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  Value zero = builder.createWithStage<arith::ConstantIntOp>(
      forOp.getLoc(), stage, clusterId, 0, 32);
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  tt::MemDescType allocTy = cast<tt::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  tt::MemDescType subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto view = builder.createWithStage<ttg::MemDescSubviewOp>(
      loc, stage, clusterId, subviewTy, alloc, copyOffsets);

  Value pred = builder.createWithStage<arith::ConstantIntOp>(loc, stage,
                                                             clusterId, 1, 1);
  Operation *copy = builder.createWithStage<ttng::AsyncTMACopyGlobalToLocalOp>(
      loc, stage, clusterId, loadOp.getDescPtr(), loadOp.getIndices(), barrier,
      view, pred);

  bool isMMV3Load = loadToInfo[loadOp].loadIsMMAV3;

  builder.setInsertionPointAfter(waitOp);
  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad = builder.createWithStage<ttg::MemDescSubviewOp>(
      loc, stageForFirstUse, clusterForFirstUse, subviewTy, alloc, loadOffsets);
  if (isMMV3Load) {
    auto alloc = cast<ttg::LocalAllocOp>((*loadOp->getUsers().begin()));
    replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
    alloc.erase();
  } else {
    SmallVector<ttg::LocalAllocOp> allocsToErase;
    for (Operation *user : loadOp->getUsers()) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
        allocsToErase.push_back(alloc);
      }
    }
    for (auto alloc : allocsToErase) {
      alloc.erase();
    }

    auto sharedLoad = builder.createWithStage<ttg::LocalLoadOp>(
        loc, stage, clusterId, loadOp.getType(),
        viewLoad /*,wait->getResult(0)*/);
    auto result = sharedLoad->getResults();
    loadOp->replaceAllUsesWith(result);
  }
  loadOp.erase();
}

static ttg::BlockedEncodingAttr
getBlockedEncoding(tt::LoadOp loadOp, tt::ModuleAxisInfoAnalysis &axisInfo) {
  Value src = loadOp.getPtr();
  auto ty = cast<RankedTensorType>(src.getType());
  auto mod = loadOp->getParentOfType<ModuleOp>();
  int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  tt::AxisInfo::DimVectorT contiguity =
      axisInfo.getAxisInfo(src)->getContiguity();
  SmallVector<unsigned> order = argSort(contiguity);
  unsigned currPerThread = getNumElementsPerThread(loadOp, order, axisInfo);
  SmallVector<unsigned> sizePerThread(order.size(), 1);
  sizePerThread[order[0]] = currPerThread;
  ttg::CTALayoutAttr ctaLayout = ttg::getCTALayout(ty.getEncoding());
  return ttg::BlockedEncodingAttr::get(loadOp->getContext(), ty.getShape(),
                                       sizePerThread, order, numWarps,
                                       threadsPerWarp, ctaLayout);
}

static std::optional<ttg::SharedEncodingAttr>
getSharedEncoding(Operation *loadOp, bool isMMAV3) {
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
  if (isMMAV3) {
    return ttg::SharedEncodingAttr::get(ty.getContext(), ty.getShape(), order,
                                        ctaLayout, ty.getElementType());
  }

  // If the load is used by a LocalAllocOp, use the same encoding as the allocs.
  // If the allocs don't all have the same encoding, bail.
  if (llvm::any_of(loadOp->getUsers(), [&](Operation *user) {
        return isa<ttg::LocalAllocOp>(user);
      })) {
    ttg::SharedEncodingAttr localAllocEnc;
    for (auto user : loadOp->getUsers()) {
      auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
      if (!localAlloc)
        continue;
      auto enc = mlir::cast<ttg::SharedEncodingAttr>(
          localAlloc.getType().getEncoding());
      if (!localAllocEnc) {
        localAllocEnc = enc;
      }
      if (enc != localAllocEnc)
        return std::nullopt;
    }
    return localAllocEnc;
  }

  // Use non-swizzled layout for loads that do not feed into dot ops.
  // TODO: This won't be optimal for 2D tensors.
  return ttg::SharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order,
                                      ctaLayout);
}

static bool hasSharedEncodingHelper(Operation *loadOp) {
  // If the load is used by a LocalAllocOp, use the same encoding as the allocs.
  // If the allocs don't all have the same encoding, bail.
  if (llvm::any_of(loadOp->getUsers(), [&](Operation *user) {
        return isa<ttg::LocalAllocOp>(user);
      })) {
    ttg::SharedEncodingAttr localAllocEnc;
    for (auto user : loadOp->getUsers()) {
      auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
      if (!localAlloc)
        continue;
      auto enc = mlir::cast<ttg::SharedEncodingAttr>(
          localAlloc.getType().getEncoding());
      if (!localAllocEnc) {
        localAllocEnc = enc;
      }
      if (enc != localAllocEnc)
        return false;
    }
    return true;
  }
  return true;
}

// When loop doesn't have num_stages attributes, we will look for any load or
// dot (only the first one in the chain). With the attribute we should look for
// any op, but also only the first one.
static llvm::SmallVector<Operation *>
getTransitiveUserInBlock(Operation *baseOp, scf::ForOp &forOp) {
  llvm::SmallVector<Operation *> users;
  DenseSet<Operation *> seen;
  bool loopHasAttribute = forOp->hasAttr(tt::kNumStagesAttrName);
  std::function<void(Operation *, Operation *, bool)> dfs =
      [&](Operation *op, Operation *baseOp, bool anyOp) {
        if (!seen.insert(op).second)
          return;
        if (op != baseOp) {
          if (anyOp) {
            // Only track the first op in the dependence chain.
            users.push_back(op);
            return;
          }
          if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op) ||
              op->hasTrait<OpTrait::DotLike>()) {
            // Stop recursion when hitting a LoadOp or a DotOp.
            users.push_back(op);
            return;
          }
        }
        for (Operation *user : op->getUsers())
          if (user->getBlock() == op->getBlock())
            dfs(user, baseOp, anyOp);
      };
  dfs(baseOp, baseOp, false /*anyOp*/);
  // For now, this needs to be aligned with loadOpsToIndirectionLevelAndUse
  // since only those uses are in the schedule. Once we move all scheduling to
  // happen before lowering, any direct use will have a stage assigned.
  if (loopHasAttribute) {
    seen.clear();
    dfs(baseOp, baseOp, true /*anyOp*/);
  }
  return users;
}

static llvm::MapVector<Operation *, LoadInfo>
assignMemoryLayouts(scf::ForOp &forOp,
                    tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  llvm::MapVector<Operation *, LoadInfo> loadToInfo;

  // Go through all loads in the loop, check to see if they are pipelined.
  llvm::DenseSet<Operation *> loadsToPipeline;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (!isa<tt::LoadOp>(op) && !isa<tt::ExperimentalDescriptorLoadOp>(op))
      continue;
    if (loadToInfo.count(&op))
      // TODO pawel: err, we'd need to verify that the distance is the same
      continue;
    if (!op.hasAttr(mlir::triton::kLoopStageAttrName))
      continue;

    // Check stage for uses. If any use is in a different stage, treat it
    // as a pipelined load.
    auto users = getTransitiveUserInBlock(&op, forOp);
    LLVM_DEBUG({
      LDBG("TransitiveUser for load " << op);
      for (const auto user : users) {
        LDBG("  - use: " << *user);
      }
    });

    bool isPipelined = false;
    auto [sLoad, _cLoad] = getStageCluster(&op);
    for (auto user : users) {
      auto [stage, _cluster] = getStageCluster(user);
      if (stage != sLoad) {
        isPipelined = true;
        break;
      }
    }
    if (!isPipelined)
      continue;

    loadsToPipeline.insert(&op);
    LoadInfo loadInfo;
    for (auto use : users) {
      if (use->hasTrait<OpTrait::DotLike>()) {
        loadInfo.usedByDot = true;
        if (loadIsMMAv3(&op)) {
          loadInfo.loadIsMMAV3 = true;
          loadInfo.sharedEncoding =
              getSharedEncoding(&op, /*loadIsMMAv3=*/true).value_or(nullptr);
        } else if (isa<tt::ExperimentalDescriptorLoadOp>(op)) {
          loadInfo.sharedEncoding =
              getSharedEncoding(&op, /*loadIsMMAv3=*/true).value_or(nullptr);
        } else if (auto dot = dyn_cast<tt::DotOp>(use)) {
          bool incompatible = false;
          loadInfo.sharedEncoding =
              getSharedEncIfAllUsersAreDotEnc(op.getResult(0), incompatible)
                  .value_or(nullptr);
        }
      }

      // If we still don't have a shared encoding, try a "generic" shared
      // encoding.
      if (!loadInfo.sharedEncoding && !isa<ttng::WarpGroupDotOp>(use)) {
        loadInfo.sharedEncoding =
            getSharedEncoding(&op, /*isMMAV3=*/loadInfo.loadIsMMAV3)
                .value_or(nullptr);
        if (auto loadOp = dyn_cast<tt::LoadOp>(op))
          loadInfo.blockedEncoding =
              getBlockedEncoding(loadOp, axisInfoAnalysis);
      }

      // If that still didn't work, bail on pipelining this load.
      if (!loadInfo.sharedEncoding)
        continue;
    }
    loadToInfo[&op] = loadInfo;
  }
  // Make sure all loads in loadsToPipeline are in loadToInfo.
  for (auto *load : loadsToPipeline)
    assert(loadToInfo.count(load) &&
           "pipelined loads should have sharedEncoding");

  return loadToInfo;
}

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, Operation *loadOp,
                         ttg::SharedEncodingAttr sharedEnc, unsigned distance) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type memdescType = mlir::triton::MemDescType::get(
      bufferShape, ty.getElementType(), sharedEnc, sharedMemorySpace,
      /*mutableMemory*/ true);
  Value alloc = builder.create<mlir::triton::gpu::LocalAllocOp>(
      loadOp->getLoc(), memdescType, Value());
  return alloc;
}

// Create an allocation to hold the mbarriers.
static Value createBarrierAlloc(scf::ForOp &forOp, unsigned distance) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  Location loc = forOp.getLoc();
  auto context = forOp.getContext();
  auto barrierCTALayout =
      ttg::CTALayoutAttr::get(context, /*CTAsPerCGA=*/{1},
                              /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
  auto barrierEncoding =
      ttg::SharedEncodingAttr::get(context, 1, 1, 1, {0}, barrierCTALayout);
  Type barrierMemDescType = tt::MemDescType::get(
      {distance}, builder.getI64Type(), barrierEncoding, sharedMemorySpace,
      /*mutableMemory=*/true);
  Type singleBarrierMemDescType =
      tt::MemDescType::get({1}, builder.getI64Type(), barrierEncoding,
                           sharedMemorySpace, /*mutableMemory=*/true);
  Value barrierAlloc = builder.create<mlir::triton::gpu::LocalAllocOp>(
      loc, barrierMemDescType, Value());
  for (unsigned i = 0; i < distance; i++) {
    Value idx = builder.create<arith::ConstantIntOp>(loc, i, 32);
    Value barrierView = builder.create<ttg::MemDescSubviewOp>(
        loc, singleBarrierMemDescType, barrierAlloc, idx);
    builder.create<ttng::InitBarrierOp>(forOp->getLoc(), barrierView, 1);
  }
  return barrierAlloc;
}

struct AsyncLoad {
  AsyncLoad(Operation *loadOp, Value alloc) : loadOp(loadOp), alloc(alloc) {}
  Operation *loadOp;
  Value alloc;
  Value barrier;
  Operation *waitOp = nullptr;
  int firstUseStage, firstUseCluster;
  bool isTMALoad = false;
};

// Create barriers and wait ops for the async loads. Barriers may be shared by
// multiple loads is the schedule allows it.
static void createTMABarrierAndWait(
    scf::ForOp &forOp, SmallVector<AsyncLoad> &asyncLoads, Value insertIdx,
    Value extractIdx, Value phase, int numBuffers, SmallVector<Value> &barriers,
    const llvm::MapVector<Operation *, LoadInfo> &loadToInfo) {
  llvm::SmallDenseMap<Operation *, AsyncLoad *> loadToAsyncLoad;
  for (AsyncLoad &asyncLoad : asyncLoads) {
    loadToAsyncLoad[asyncLoad.loadOp] = &asyncLoad;
  }
  SmallVector<SmallVector<AsyncLoad *>> loadGroups;
  llvm::SmallDenseSet<Operation *> visited;
  // Find groups of loads that can share the same barrier. We look consecutive
  // loads and check that there are uses in between.
  for (AsyncLoad &asyncLoad : asyncLoads) {
    if (!asyncLoad.isTMALoad || visited.count(asyncLoad.loadOp))
      continue;
    llvm::SmallDenseSet<Operation *> users;
    SmallVector<AsyncLoad *> group;
    Block *loadBlock = asyncLoad.loadOp->getBlock();
    auto addToGroup = [&](AsyncLoad *loadInfo) {
      group.push_back(loadInfo);
      visited.insert(loadInfo->loadOp);
      for (Operation *user : loadInfo->loadOp->getUsers()) {
        auto it = loadToInfo.find(loadInfo->loadOp);
        if (it != loadToInfo.end()) {
          // Special case for MMAv3 loads, we can ignore the alloc and only
          // consider uses of the alloc op since it will be removed.
          if (it->second.loadIsMMAV3) {
            auto alloc = cast<ttg::LocalAllocOp>(
                (*loadInfo->loadOp->getUsers().begin()));
            if (alloc->getBlock() == loadBlock) {
              users.insert(alloc->getUsers().begin(), alloc->getUsers().end());
              continue;
            }
          }
        }
        Operation *userInBlock = loadBlock->findAncestorOpInBlock(*user);
        if (userInBlock)
          users.insert(userInBlock);
      }
    };
    addToGroup(&asyncLoad);
    Operation *nextOp = asyncLoad.loadOp->getNextNode();
    while (nextOp) {
      if (users.count(nextOp) || visited.count(nextOp))
        break;
      if (isa<tt::ExperimentalDescriptorLoadOp>(nextOp)) {
        auto it = loadToAsyncLoad.find(nextOp);
        if (it != loadToAsyncLoad.end() && it->second->isTMALoad) {
          if (group.size() > 0 &&
              sameStageCluster(group[0]->loadOp, it->second->loadOp))
            addToGroup(it->second);
        }
      }
      nextOp = nextOp->getNextNode();
    }
    loadGroups.push_back(group);
  }

  // For each group calculate the size and insert the barrier after the last
  // load.
  for (SmallVector<AsyncLoad *> &group : loadGroups) {
    int sizeInBytes = 0;
    for (AsyncLoad *asyncLoad : group) {
      auto tensorTy =
          cast<RankedTensorType>(asyncLoad->loadOp->getResult(0).getType());
      int loadSize = product(tensorTy.getShape());
      sizeInBytes +=
          loadSize * tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
    }

    auto [stage, cluster] = getStageCluster(group[0]->loadOp);
    Value barrierAlloc = createBarrierAlloc(forOp, numBuffers);
    barriers.push_back(barrierAlloc);
    Location loc = forOp.getLoc();
    OpBuilderWithStage builder(forOp);
    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(builder.getContext());
    tt::MemDescType barrierTy = tt::MemDescType::get(
        {1}, builder.getI64Type(),
        cast<tt::MemDescType>(barrierAlloc.getType()).getEncoding(),
        sharedMemorySpace,
        /*mutableMemory=*/true);
    builder.setInsertionPoint(group[0]->loadOp);
    Value barrier = builder.createWithStage<ttg::MemDescSubviewOp>(
        loc, stage, cluster, barrierTy, barrierAlloc,
        ArrayRef<Value>({insertIdx}));
    Value pred = builder.createWithStage<arith::ConstantIntOp>(loc, stage,
                                                               cluster, 1, 1);
    Operation *expect = builder.createWithStage<ttng::BarrierExpectOp>(
        forOp.getLoc(), stage, cluster, barrier, sizeInBytes, pred);

    builder.setInsertionPointAfter(group.back()->loadOp);
    Value barrierViewWait = builder.createWithStage<ttg::MemDescSubviewOp>(
        loc, group[0]->firstUseStage, group[0]->firstUseCluster, barrierTy,
        barrierAlloc, ArrayRef<Value>({extractIdx}));
    Operation *wait = builder.createWithStage<ttng::WaitBarrierOp>(
        loc, group[0]->firstUseStage, group[0]->firstUseCluster,
        barrierViewWait, phase);
    // Update the async loads info.
    for (AsyncLoad *asyncLoad : group) {
      asyncLoad->barrier = barrier;
      asyncLoad->waitOp = wait;
    }
  }
}

// Similar to CoarseSchedule::insertDepsOfOp, we set <stage, cluster>
// for ops that are on the def chain for the given op.
static void insertDepsOfOpOnAttributes(scf::ForOp forOp, Operation *op,
                                       int stage, int cluster,
                                       bool includeArg) {
  for (Value operand : op->getOperands()) {
    Value v = operand;
    llvm::SmallDenseSet<Value> seen;
    while (auto arg = dyn_cast<BlockArgument>(v)) {
      if (!includeArg)
        break;
      if (!seen.insert(v).second)
        break;
      if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
        auto yieldOp = op->getBlock()->getTerminator();
        v = yieldOp->getOperand(arg.getArgNumber() - 1);
        continue;
      }
      break;
    }
    Operation *defOp = v.getDefiningOp();
    if (defOp && defOp->getBlock() == op->getBlock()) {
      // check to see if defOp has <stage, cluster>, if no, set stage, cluster
      // and call
      if (!defOp->hasAttr(mlir::triton::kLoopStageAttrName) ||
          !defOp->hasAttr(mlir::triton::kLoopClusterAttrName)) {
        setStageCluster(forOp, defOp, stage, cluster);
        insertDepsOfOpOnAttributes(forOp, defOp, stage, cluster, includeArg);
      }
    }
  }
}

// For ops that don't have <stage, cluster>, try to add them stage by stage.
// This is similar to scheduleDependencies in loopScheduling.
static void scheduleDependenciesOnAttributes(scf::ForOp forOp, int numStages) {
  auto [minClusterId, maxClusterId] = getMinMaxCluster(forOp);

  SmallVector<SmallVector<std::tuple<Operation *, int, int>>, 8> orderClusters(
      maxClusterId + 1);
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!op.hasAttr(mlir::triton::kLoopStageAttrName) ||
        !op.hasAttr(mlir::triton::kLoopClusterAttrName))
      continue;

    auto [stage, clusterId] = getStageCluster(&op);
    orderClusters[clusterId].push_back(std::make_tuple(&op, stage, clusterId));
  }

  SmallVector<std::tuple<Operation *, int, int>> opsInOrder;
  for (int i = 0; i < orderClusters.size(); i++) {
    for (auto [op, stage, cluster] : orderClusters[i]) {
      opsInOrder.push_back({op, stage, cluster});
    }
  }

  // Schedule dependencies stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : opsInOrder) {
      if (stage_ != stage)
        continue;
      insertDepsOfOpOnAttributes(forOp, op, stage, cluster, false);
    }
  }
}

// This is similar to CoarseSchedule.createFinalSchedule.
static std::vector<std::pair<Operation *, unsigned>>
getFinalSchedule(scf::ForOp &forOp, int numStages) {
  auto [minClusterId, maxClusterId] = getMinMaxCluster(forOp);
  SmallVector<SmallVector<Operation *>, 8> orderClusters(maxClusterId -
                                                         minClusterId + 1);
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (!op.hasAttr(mlir::triton::kLoopStageAttrName) ||
        !op.hasAttr(mlir::triton::kLoopClusterAttrName))
      continue;

    auto [stage, clusterId] = getStageCluster(&op);
    assert(stage < numStages && "Op with invalid stage!");
    orderClusters[clusterId - minClusterId].push_back(&op);
  }
  std::vector<std::pair<Operation *, unsigned>> fSchedule;
  for (int i = 0; i < orderClusters.size(); i++) {
    for (auto op : orderClusters[i]) {
      auto [stage, _] = getStageCluster(op);
      fSchedule.push_back({op, stage});
    }
  }
  return fSchedule;
}

// Convert load ops into their asyn version and apply multi-buffering based on
// the required number of buffers.
static SmallVector<Value>
createAsyncOps(scf::ForOp &forOp,
               llvm::MapVector<Operation *, LoadInfo> &loadToInfo,
               SmallVector<Value> &barriers, int numStages) {
  // Calculate the number of buffers needed for each load.
  // TODO pawel: we could do more fine-grained allocation here and
  // allocate only the number of buffers that specific loads need.
  // Instead, we allocate the maximum number of buffers needed by any load.
  int numBuffers =
      llvm::max_element(llvm::make_second_range(loadToInfo), [](auto &lhs,
                                                                auto &rhs) {
        return lhs.distToUse < rhs.distToUse;
      })->distToUse;
  bool hasMMAV3 =
      llvm::any_of(loadToInfo, [](auto &kv) { return kv.second.loadIsMMAV3; });
  if (hasMMAV3) {
    // For MMAv3, we need an extra buffer as this is assumed in the wgmma
    // pipelining post-processing.
    numBuffers++;
  };

  SmallVector<AsyncLoad> asyncLoads;
  SmallVector<Value> allocs;
  bool hasTMALoad = false;
  for (auto &[loadOp, info] : loadToInfo) {
    assert(info.sharedEncoding && "LoadOp shared encoding not defined.");
    Value alloc = createAlloc(forOp, loadOp, info.sharedEncoding, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    allocs.push_back(alloc);
    asyncLoads.emplace_back(loadOp, alloc);
    if (isa<tt::ExperimentalDescriptorLoadOp>(loadOp)) {
      hasTMALoad = true;
      asyncLoads.back().isTMALoad = true;
    }
    auto *firstUse = getFirstUseOfPipelinedLoad(loadOp);
    auto [firstUseStage, firstUseCluster] = getStageCluster(firstUse);
    asyncLoads.back().firstUseStage = firstUseStage;
    asyncLoads.back().firstUseCluster = firstUseCluster;
  }

  IRRewriter builder(forOp.getContext());
  builder.setInsertionPoint(forOp);

  Location loc = forOp.getLoc();
  // Create two new counters to index into the allocs.
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value insertIdx = minusOne;
  Value extractIdx = minusOne;
  Value phase = Value();
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
  SmallVector<Value> newOperands;
  newOperands.push_back(insertIdx);
  newOperands.push_back(extractIdx);
  if (hasTMALoad) {
    phase = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    newOperands.push_back(phase);
  }
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;
  insertIdx = newForOp.getBody()->getArgument(newOperandIndex);
  extractIdx = newForOp.getBody()->getArgument(newOperandIndex + 1);
  if (phase) {
    phase = newForOp.getBody()->getArgument(newOperandIndex + 2);
  }

  // FIXME: loads can be in different (stage, cluster)
  // Create two counters for the insert and extract indices to avoid creating
  // long liverange.
  builder.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
  insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
  Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               insertIdx, numBuffersVal);
  insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);

  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);
  if (phase) {
    Value nextPhase = builder.create<arith::XOrIOp>(loc, phase, one);
    phase = builder.create<arith::SelectOp>(loc, cndExt, phase, nextPhase);
  }
  createTMABarrierAndWait(forOp, asyncLoads, insertIdx, extractIdx, phase,
                          numBuffers, barriers, loadToInfo);

  auto [_, maxClusterId] = getMinMaxCluster(forOp);
  for (AsyncLoad &asyncLoad : asyncLoads) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(asyncLoad.loadOp)) {
      createAsyncCopy(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                      loadToInfo, numStages, maxClusterId);
    } else {
      auto descLoad = cast<tt::ExperimentalDescriptorLoadOp>(asyncLoad.loadOp);
      createTMAAsyncCopy(forOp, descLoad, asyncLoad.alloc, insertIdx,
                         extractIdx, asyncLoad.barrier, asyncLoad.waitOp, phase,
                         loadToInfo, numStages);
    }
  }
  SmallVector<Value> newYieldOperands = {insertIdx, extractIdx};
  if (phase)
    newYieldOperands.push_back(phase);
  // Patch the yield with the updated counters.
  appendToForOpYield(forOp, newYieldOperands);

  scheduleDependenciesOnAttributes(forOp, numStages);

  // Make sure all ops have attributes.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    assert(op.hasAttr(mlir::triton::kLoopStageAttrName) &&
           op.hasAttr(mlir::triton::kLoopClusterAttrName));
  }
  return allocs;
}

static void invalidateBarriers(OpBuilder &builder,
                               SmallVector<Value> &barriers) {
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(builder.getContext());
  for (Value barrier : barriers) {
    int numBarriers = cast<tt::MemDescType>(barrier.getType()).getShape()[0];
    for (int i = 0; i < numBarriers; i++) {
      Value idx = builder.create<arith::ConstantIntOp>(barrier.getLoc(), i, 32);
      tt::MemDescType barrierTy = tt::MemDescType::get(
          {1}, builder.getI64Type(),
          cast<tt::MemDescType>(barrier.getType()).getEncoding(),
          sharedMemorySpace,
          /*mutableMemory=*/true);
      Value barrierView = builder.create<ttg::MemDescSubviewOp>(
          barrier.getLoc(), barrierTy, barrier, idx);
      builder.create<ttng::InvalBarrierOp>(barrier.getLoc(), barrierView);
    }
  }
}

bool mlir::triton::preProcessLoopAndGetSchedule(
    scf::ForOp &forOp, int numStages, mlir::triton::PipeliningOption &options) {

  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  // Check which loads are good for pipelining, and assign them
  // memory layouts.
  llvm::MapVector<Operation *, LoadInfo> loadToInfo =
      assignMemoryLayouts(forOp, axisInfoAnalysis);
  if (loadToInfo.empty())
    return false;

  // Distance from the load to the use.
  for (auto &[loadOp, info] : loadToInfo) {
    auto *use = getFirstUseOfPipelinedLoad(loadOp);
    auto [stage, _] = getStageCluster(loadOp);
    auto [stageUse, t_] = getStageCluster(use);
    loadToInfo[loadOp].distToUse = stageUse - stage;
  }

  SmallVector<Value> barriers;
  // Convert the loads into async loads and create the allocs.
  SmallVector<Value> allocs =
      createAsyncOps(forOp, loadToInfo, barriers, numStages);

  // Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      getFinalSchedule(forOp, numStages);

  // Fill out the pipeline options.
  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = tt::predicateOp;
  options.supportDynamicLoops = true;
  options.annotateFn = [](Operation *op,
                          mlir::triton::PipeliningOption::PipelinerPart part,
                          unsigned iteration) {};
  // Insert a wait 0 after the loop
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), ValueRange({}), 0);
  // Invalidate any mbarrier create
  invalidateBarriers(builder, barriers);
  // Explicitly deallocate allocated tensors after the wait op
  for (auto alloc : allocs)
    builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);
  return true;
}

/// Find the minimum number of async_commit_group ops between the wait
/// and the associated async_commit_group. This can be safely used as the wait
/// number.
static int minNumInterleavedCommitOps(Operation *waitOp) {
  auto countCommitsBetween = [](Operation *op1, Operation *op2) {
    int count = 0;
    for (auto op = op1; op != op2; op = op->getNextNode()) {
      if (isa<ttg::AsyncCommitGroupOp>(op))
        count++;
      // Intentionally skip block ops' children. This will give us
      // convervatively low number of insert ops.
    }
    return count;
  };

  int minCommitNumber = INT_MAX;

  // DFS the def chain of the extract op to find the insert op. On each path
  // we calculate the number of async_commit. Then we select the minimum number
  // of async_commit ops among all the paths.
  std::function<int(Value, Operation *, int)> minOverHistories =
      [&](Value val, Operation *sinkOp, int thisHistorySum) -> int {
    if (Operation *defOp = val.getDefiningOp()) {
      thisHistorySum += countCommitsBetween(defOp->getNextNode(), sinkOp);
      minCommitNumber = std::min(minCommitNumber, thisHistorySum);
      return minCommitNumber;
    }
    if (auto arg = mlir::dyn_cast<BlockArgument>(val)) {
      Block *block = arg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

      // Failed to track, return 0 conservatively.
      if (!forOp)
        return 0;

      Operation *firstForInst = &*forOp.getBody()->begin();
      int insertsBetween = countCommitsBetween(firstForInst, sinkOp);
      thisHistorySum += insertsBetween;
      if (thisHistorySum >= minCommitNumber)
        return minCommitNumber;

      // get the value value assigned to the argument coming from outside the
      // loop
      Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
      int min1 = minOverHistories(incomingVal, forOp, thisHistorySum);

      // get the value value assigned to the argument coming from the previous
      // iteration
      Operation *yieldOp = block->getTerminator();
      Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
      int min2 = minOverHistories(prevVal, yieldOp, thisHistorySum);
      return std::min(std::min(min1, min2), minCommitNumber);
    }
    // Failed to track, return 0 conservatively.
    return 0;
  };

  if (waitOp->getNumOperands() != 1)
    return 0;
  int minCommits = minOverHistories(waitOp->getOperand(0), waitOp, 0);
  return minCommits;
}

// Look for consecutive wait ops and combine them into a single wait op.
static void
combineRedundantWaitOps(llvm::SmallSetVector<ttg::AsyncWaitOp, 8> &waitOps) {
  llvm::MapVector<ttg::AsyncWaitOp, ttg::AsyncWaitOp> toDelete;
  for (auto waitOp : waitOps) {
    if (toDelete.count(waitOp))
      continue;
    SmallVector<ttg::AsyncWaitOp> waitGroup = {waitOp};
    SmallVector<Value> depTokens;
    unsigned minWaitNumber = waitOp.getNum();
    Operation *next = waitOp->getNextNode();
    while (next && isa<ttg::MemDescSubviewOp, ttg::AsyncWaitOp>(next)) {
      if (auto nextWait = dyn_cast<ttg::AsyncWaitOp>(next)) {
        waitGroup.push_back(nextWait);
        minWaitNumber = std::min(minWaitNumber, nextWait.getNum());
        depTokens.append(nextWait.getOperands().begin(),
                         nextWait.getOperands().end());
      }
      next = next->getNextNode();
    }
    if (waitGroup.size() == 1)
      continue;
    OpBuilder builder(waitGroup.back());
    auto newWaitOp = builder.create<ttg::AsyncWaitOp>(waitOp.getLoc(),
                                                      depTokens, minWaitNumber);
    for (auto waitOp : waitGroup) {
      toDelete[waitOp] = newWaitOp;
    }
  }
  for (auto waitOp : toDelete) {
    waitOp.first->replaceAllUsesWith(waitOp.second);
    waitOp.first->erase();
  }
}

/// Update wait op number by analyzing the number of async_commit_group ops
/// along all paths.
void mlir::triton::updateWaits(ModuleOp module) {
  llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
  module.walk([&](ttg::AsyncWaitOp waitOp) {
    int minNumCommits = minNumInterleavedCommitOps(waitOp);
    waitOp.setNum(minNumCommits);
    waitOps.insert(waitOp);
  });
  combineRedundantWaitOps(waitOps);
}

// Add the given values as operands of the given wait, and replace all uses of
// the values with the wait.  Also adds related MemDesc's to the wait.
//
// Threading %a through the wait transforms
//
//   %a = <...>
//   (%x', %y') = ttng.async_wait %x, %y
//   %b = fn(%a)
//
// into
//
//   %a = <...>
//   (%x', %y', %a') = ttng.async_wait %x, %y, %a
//   %b = fn(%a')
//
// The wait must dominate all uses of the elements of `values`.
//
// In addition to adding each value from `values` to the wait, this function
// also adds some MemDesc's to the wait.  The idea is that if you have
//
//   %alloc = ttg.local_alloc ...
//   %a = ttng.warp_group_dot %alloc
//   %a1 = ttng.warp_group_dot_wait %a
//
// then we want the wait to depend on %alloc as well as %a.  This extends the
// live range of %alloc, so that it won't be destroyed until after the dot is
// waited on.
//
// Specifically, this function finds all warp_group_dot ops that elements of
// `values` depend on.  Then it adds the MemDesc operands of those dots to the
// wait.
static void threadValuesThroughWait(ttng::WarpGroupDotWaitOp wait,
                                    MutableArrayRef<Value> values) {
  IRRewriter builder(wait.getContext());
  builder.setInsertionPoint(wait);

  // Operands are only added to the wait through this function, so we can have
  // the invariant that the wait has no duplicates.  This makes things a bit
  // easier below.
  size_t origNumOperands = wait.getNumOperands();
  SetVector<Value> newOperands(wait.getOperands().begin(),
                               wait.getOperands().end());
  assert(newOperands.size() == origNumOperands &&
         "Wait op has duplicate operands.");

  newOperands.insert(values.begin(), values.end());

  // Find memdefs depended on by `values` through async dot ops.
  SmallVector<ttng::WarpGroupDotOp> asyncDots;
  for (Value v : values) {
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *op) {
      if (auto dot = dyn_cast<ttng::WarpGroupDotOp>(op)) {
        asyncDots.push_back(dot);
        return false;
      }
      return op->getBlock() == wait->getBlock();
    };
    SetVector<Operation *> slice;
    getBackwardSlice(v, &slice, options);
  }

  for (ttng::WarpGroupDotOp dot : asyncDots) {
    for (Value operand : dot.getOperands()) {
      if (isa<tt::MemDescType>(operand.getType())) {
        newOperands.insert(operand);
      }
    }
  }

  // We can't use replaceWithNewOp because we're changing the number of return
  // values in the operation.
  auto newWait = builder.create<ttng::WarpGroupDotWaitOp>(
      wait.getLoc(), llvm::to_vector(newOperands), wait.getPendings());

  auto dominatedByNewWait = [&](OpOperand &operand) {
    auto opInThisBlock =
        newWait->getBlock()->findAncestorOpInBlock(*operand.getOwner());
    return opInThisBlock && newWait->isBeforeInBlock(opInThisBlock);
  };
  for (int i = 0; i < origNumOperands; i++) {
    Value operand = wait.getResult(i);
    if (!isa<tt::MemDescType>(operand.getType()))
      operand.replaceAllUsesWith(newWait.getResult(i));
  }
  for (int i = origNumOperands; i < newOperands.size(); i++) {
    Value operand = newWait.getOperand(i);
    if (!isa<tt::MemDescType>(operand.getType()))
      operand.replaceUsesWithIf(newWait.getResult(i), dominatedByNewWait);
  }
  wait->erase();
}

// Determines whether a given MMAv3 dot op, represented as ttng.warp_group_dot,
// needs a wait immediately after it.
//
// In PTX, MMAv3 exists only as an asynchronous op.  In Triton, we can represent
// MMAv3 ops as either ttng.warp_group_dot {isAsync=True} or ttng.warp_group_dot
// {isAsync=False}.  But even if we use ttng.warp_group_dot {isAsync=True}, the
// conservative thing is to make a dot "effectively synchronous" by inserting a
// `ttng.warp_group_dot_wait {pendings=0}` right after it.
//
// We can omit the wait and create a "properly async" dot if all of the
// following are true.
//
//  1. All operands that touch shared memory are multi-buffered, i.e. can't read
//     an incomplete value while it's being written asynchronously by a load.
//
//  2. If the dot is used by any op in the loop, it must be used under an `if`,
//     and will be synced with a `wait 0` at the beginning of the `if` block.
//
//  3. During iteration i, between the start of the loop up until the first
//     `ttng.warp_group_dot_wait {pendings=0}` op, the result of the dot from
//     iteration i-1 is consumed only by other MMAv3 dots as the `c` operand.
//
//     This is safe because the following pseudo-PTX is valid:
//
//        %accum = warp_group_dot %a1, %b1, %c1
//        %accum = warp_group_dot %a2, %b2, %accum
//
//     That is, the second async dot can use the result of the first one without
//     an intervening wait.  However, the only operation that can legally read
//     %accum before the wait is another warp_group_dot, and this only works for
//     the `c` operand, not `a` or `b`.  See
//     https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
//     (ttng::WarpGroupDotOp corresponds to wgmma.fence followed by one or more
//     wgmma.async ops, so our understanding is that the two
//     ttng::WarpGroupDotOps don't have to correspond to wgmma.async ops with
//     the same shapes as specified in the docs, because there's an intervening
//     fence.)
//
// If the op can be properly async, this function returns the index of the dot
// in the loop's iter_args.  (Rule (2) above ensures this is well-defined.)
//
static std::optional<int> dotCanBeProperlyAsync(ttng::WarpGroupDotOp dotOp,
                                                scf::ForOp forOp) {
  LDBG("Considering whether to make MMAv3 dot properly async: " << dotOp);

  // Rule 1: All shmem operands are multi-buffered.
  auto checkOperand = [&](Value operand) {
    if (!isa<ttg::SharedEncodingAttr>(
            cast<TensorOrMemDesc>(operand.getType()).getEncoding())) {
      return true;
    }

    // If it's a shmem operand, it must either be defined outside the loop, or
    // come from an MemDescSubview op.  Only ConvertLayout and Trans ops are
    // allowed in between.
    Value transitiveOperand = operand;
    while (isa_and_nonnull<ttg::ConvertLayoutOp, tt::TransOp>(
               transitiveOperand.getDefiningOp()) ||
           isa<BlockArgument>(transitiveOperand)) {
      auto blockArg = dyn_cast<BlockArgument>(transitiveOperand);
      if (blockArg && blockArg.getOwner() == forOp.getBody()) {
        transitiveOperand =
            cast<scf::YieldOp>(blockArg.getOwner()->getTerminator())
                .getOperand(blockArg.getArgNumber() - 1);
      } else if (Operation *def = transitiveOperand.getDefiningOp()) {
        transitiveOperand = def->getOperand(0);
      }
    }
    return forOp.isDefinedOutsideOfLoop(transitiveOperand) ||
           transitiveOperand.getDefiningOp<ttg::MemDescSubviewOp>();
  };

  // We don't have to call checkOperand on getC() because it's always in
  // registers, never in shmem.
  assert(isa<ttg::NvidiaMmaEncodingAttr>(dotOp.getC().getType().getEncoding()));
  if (!checkOperand(dotOp.getA()) || !checkOperand(dotOp.getB())) {
    LDBG("Can't make dot async because shmem operands aren't multi-buffered");
    return std::nullopt;
  }

  // Rule 2: The dot cannot be unconditionally used by any op in the loop.
  // Uses under `if` are allowed, as can be explicitly synced with a `wait 0`.
  int iterArgIdx = -1;
  Value iterArg = nullptr;
  SmallVector<std::pair<Operation *, int>> queue;
  for (auto &use : dotOp->getUses()) {
    queue.push_back({use.getOwner(), use.getOperandNumber()});
  }
  while (!queue.empty()) {
    auto [user, argIdx] = queue.pop_back_val();
    if (user->getParentOp() == forOp) {
      if (isa<scf::YieldOp>(user)) {
        if (iterArg) {
          // The dot is used by the loop's yield, but we can't have any other
          // uses.
          LDBG("Can't make dot async because dot is used by multiple ops in "
               "the loop.");
          return std::nullopt;
        }
        iterArgIdx = argIdx;
        iterArg = forOp.getRegionIterArg(argIdx);
        continue;
      }
      LDBG("Can't make dot async because dot is unconditionally used in the "
           "loop.");
      return std::nullopt;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp())) {
      if (isa<scf::YieldOp>(user)) {
        // The result is returned by the if, follow it further.
        auto uses = ifOp.getResult(argIdx).getUses();
        for (auto &use : uses) {
          queue.push_back({use.getOwner(), use.getOperandNumber()});
        }
      }
    } else {
      return std::nullopt;
    }
  }

  // Rule 3a: Are the only users of the dot's result from iteration i-1 other
  // MMAv3 dots?  If so, we're done, this dot can be properly async.
  if (llvm::all_of(iterArg.getUses(), [&](OpOperand &use) {
        return isa<ttng::WarpGroupDotOp>(use.getOwner()) &&
               use.getOperandNumber() == 2;
      })) {
    return iterArgIdx;
  }

  // Rule 3b: Are all users of the dot's result from iteration i-1 after the
  // first `warp_group_dot_wait {pendings=0}` op?  If so, the dot can be
  // properly async, but we have to thread its result from iteration i-1 through
  // the wait.
  auto waitOps = forOp.getBody()->getOps<ttng::WarpGroupDotWaitOp>();
  auto firstWaitOpIter = llvm::find_if(
      waitOps, [&](auto waitOp) { return waitOp.getPendings() == 0; });
  if (firstWaitOpIter != waitOps.end() &&
      llvm::all_of(iterArg.getUsers(), [&](Operation *user) {
        assert(forOp->isAncestor(user));
        while (user->getParentOp() != forOp) {
          user = user->getParentOp();
        }
        return (*firstWaitOpIter)->isBeforeInBlock(user);
      })) {
    LDBG("MMAv3 dot can be properly async because it follows a "
         "warp_group_dot_wait "
         "{pendings=0}.\n"
         << "  wait: " << *firstWaitOpIter << "\n"
         << "  dot: " << dotOp);
    threadValuesThroughWait(*firstWaitOpIter, {iterArg});
    return iterArgIdx;
  }

  LDBG("Can't make dot async because its result from i-1 is used by "
       "something other than another MMAv3 dot as the `c` operand.");
  return std::nullopt;
}

// If necessary, insert a dot-wait inside the loop, waiting for the results of
// the properly-async dots from iteration i-1 to complete.  (We pipeline to
// depth 2, so there are at most 2 copies of each warp_group_dot in flight at a
// time.)
//
// We can skip inserting the wait if we have a `warp_group_dot_wait
// {pendings=0}` somewhere in the loop.  To see why, consider:
//
//   warp_group_dot
//   warp_group_dot; wait 0  // synchronous dot
//   warp_group_dot
//   warp_group_dot
//
// In this example, there are three properly-async dots, so we'd normally put
// `wait 3` at the end of the loop, meaning "wait until there are 3 or fewer
// pending async dots".  But note that when this iteration of the loop
// completes, there are only *two* pending async dots from this iteration, so
// this wait would do nothing.  This is true in general, no matter where the
// `wait 0` appears.
static void insertAsyncWarpGroupDotWaitInLoop(
    scf::ForOp forOp,
    const llvm::MapVector<Operation *, int /*iterArgIdx*/> &properlyAsyncDots) {
  if (properlyAsyncDots.empty())
    return;

  if (llvm::any_of(forOp.getBody()->getOps<ttng::WarpGroupDotWaitOp>(),
                   [](auto wait) { return wait.getPendings() == 0; })) {
    return;
  }

  // Insert waits before the users of the properly async dots other than loop
  // yield.
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    SmallVector<OpOperand *> uses;
    for (auto &use : asyncDot->getUses()) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner())) {
        continue;
      }
      uses.push_back(&use);
    }

    DenseMap<Block *, SmallVector<Value>> blockToUsers;
    for (auto use : uses) {
      auto block = use->getOwner()->getBlock();
      blockToUsers[block].push_back(use->get());
    }

    for (auto [block, users] : blockToUsers) {
      OpBuilder builder(block, block->begin());
      auto newWait = builder.create<ttng::WarpGroupDotWaitOp>(
          asyncDot->getLoc(), ArrayRef<Value>{}, 0);

      threadValuesThroughWait(newWait, users);
    }
  }

  // Add the wait right after the last properly-async dot.  This only needs to
  // wait for all properly-async dots from the i-1'th iteration to complete, IOW
  // we wait until there are most `asyncDots.size()` dots in flight.
  //
  // (You might want to put the wait at the end of the loop instead of right
  // after the last dot, but there could be a load into shmem between the last
  // async dot and the end of the loop, and that could clobber memory being used
  // by a dot.)
  IRRewriter builder(forOp.getContext());
  auto lastAsyncDot = properlyAsyncDots.back().first;
  builder.setInsertionPointAfter(lastAsyncDot);
  auto wait = builder.create<ttng::WarpGroupDotWaitOp>(
      lastAsyncDot->getLoc(),
      /*inputs=*/ArrayRef<Value>{}, properlyAsyncDots.size());

  // Thread the results of the async dots through the wait.
  SmallVector<Value> addlWaitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    addlWaitOperands.push_back(asyncDot->getResult(0));
  }
  threadValuesThroughWait(wait, addlWaitOperands);
}

// Convert MMAv3 ttng::WarpGroupDotOps {isAsync = False} (i.e. Hopper wgmma)
// into ttng::WarpGroupDotOps {isAsync = True} and insert
// ttng::WarpGroupDotWaitOps as necessary.
//
// We assume we have space for each dot to be pipelined to depth 2, i.e. each
// dot op in the loop can have at most 2 warp_group_dot ops in flight at once.
// (Each warp_group_dot op usually corresponds to a series of wgmma.async ops.)
void triton::asyncLaunchDots(scf::ForOp forOp) {
  LDBG("Original loop:\n" << *forOp);

  // First, change every MMAv3 ttng.warp_group_dot {isAsync=false}
  // into ttng.warp_group_dot {isAsync=true}.
  // The rest of this function is concerned with inserting
  // ttng.warp_group_dot_wait ops in the appropriate places.
  //
  // We call those dots that don't need to be followed immediately by a `wait 0`
  // "properly async", or sometimes just "async".
  //
  // For each dot, determine whether it can be properly async, or if it needs a
  // sync immediately after.  If it can be properly async, we know its only use
  // is in the loop's `yield` statement; asyncDots maps the op to its index in
  // the yield op.
  IRRewriter builder(forOp.getContext());
  llvm::MapVector<Operation *, int /*iterArgIdx*/> properlyAsyncDots;
  for (auto WarpGroupDotOp : forOp.getBody()->getOps<ttng::WarpGroupDotOp>()) {
    WarpGroupDotOp.setIsAsync(true);
    if (auto iterArgIdx = dotCanBeProperlyAsync(WarpGroupDotOp, forOp)) {
      properlyAsyncDots[WarpGroupDotOp] = *iterArgIdx;
    } else {
      builder.setInsertionPointAfter(WarpGroupDotOp);
      auto wait = builder.create<ttng::WarpGroupDotWaitOp>(
          WarpGroupDotOp.getLoc(), ArrayRef<Value>{},
          /*pendings=*/0);
      SmallVector<Value> waitOperands = {WarpGroupDotOp.getResult()};
      threadValuesThroughWait(wait, waitOperands);
    }
  }

  if (properlyAsyncDots.empty()) {
    LDBG("No properly async dots.");
    return;
  }

  // Next, insert a wait inside the loop.  We pipeline to depth 2, so the third
  // iteration's set of asynchronous dots (and their corresponding async copies
  // from global to shmem) can't start until the first iteration's set has
  // completed.
  insertAsyncWarpGroupDotWaitInLoop(forOp, properlyAsyncDots);

  // Finally, insert a wait after the loop, waiting for dots from the final
  // iteration of the loop.
  SmallVector<Value> waitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    waitOperands.push_back(forOp.getResult(iterArgIdx));
  }
  // Wait until there are 0 outstanding async dot ops.
  builder.setInsertionPointAfter(forOp);
  auto WarpGroupDotWaitAfterLoop = builder.create<ttng::WarpGroupDotWaitOp>(
      forOp.getLoc(), ArrayRef<Value>{}, 0);
  threadValuesThroughWait(WarpGroupDotWaitAfterLoop, waitOperands);
}
