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

// Replace the ForOp's yield with a new one with the given operands appended.
static void appendToYield(scf::ForOp forOp, ArrayRef<Value> newOperands) {
  // Fix up the yield op.
  Operation *yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value> operands(yieldOp->getOperands());
  operands.append(newOperands.begin(), newOperands.end());

  OpBuilder builder(yieldOp);
  builder.create<scf::YieldOp>(yieldOp->getLoc(), operands);
  yieldOp->erase();
}

static void createAsyncCopy(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                            Value insertIdx, Value extractIdx,
                            tt::CoarseSchedule &schedule,
                            tt::CoarseSchedule::Cluster prefetchCluster,
                            llvm::MapVector<Operation *, LoadInfo> &loadToInfo,
                            int numStages) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
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
      auto cvt =
          builder.create<ttg::ConvertLayoutOp>(loadOp->getLoc(), newTy, src);
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
  auto view =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);
  Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, src, view, mask, other, loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile());
  Operation *commmit =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copy->getResult(0));
  Operation *wait =
      builder.create<ttg::AsyncWaitOp>(loc, commmit->getResult(0), 0);

  bool isMMV3Load = loadToInfo[loadOp].loadIsMMAV3;
  auto [stage, cluster] = schedule[loadOp];
  schedule.erase(loadOp);
  schedule.insert(copy, stage, cluster);
  schedule.insert(commmit, stage, cluster);

  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  if (isMMV3Load) {
    auto alloc = cast<ttg::LocalAllocOp>((*loadOp->getUsers().begin()));
    alloc.replaceAllUsesWith(viewLoad.getResult());
    alloc.erase();
  } else {
    SmallVector<ttg::LocalAllocOp> allocsToErase;
    for (Operation *user : loadOp->getUsers()) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        alloc.replaceAllUsesWith(viewLoad.getResult());
        allocsToErase.push_back(alloc);
      }
    }
    for (auto alloc : allocsToErase) {
      alloc.erase();
    }

    auto sharedLoad = builder.create<ttg::LocalLoadOp>(
        loc, loadOp.getType(), viewLoad, wait->getResult(0));
    auto result = sharedLoad->getResults();

    // Create a select for non-zero other values as they are not handled by
    // AsyncCopyGlobalToLocalOp for now.
    Value other = loadOp.getOther();
    if (other && !isZeroConst(other)) {
      auto select = builder.create<arith::SelectOp>(
          loc, loadOp.getType(), mask, sharedLoad.getResult(), other);
      result = select->getResults();
    }

    loadOp->replaceAllUsesWith(result);

    // Prefetch load if is not MMAV3 and is used by the dot.
    if (loadToInfo[loadOp].usedByDot) {
      schedule.insert(wait, numStages - 2, prefetchCluster);
      schedule.insert(viewLoad, numStages - 2, prefetchCluster);
    }
  }
  loadOp.erase();
}

static void createTMAAsyncCopy(
    scf::ForOp &forOp, tt::ExperimentalDescriptorLoadOp loadOp, Value alloc,
    Value insertIdx, Value extractIdx, Value barrier, Operation *waitOp,
    Value phase, tt::CoarseSchedule &schedule,
    llvm::MapVector<Operation *, LoadInfo> &loadToInfo, int numStages) {
  assert(phase && "Phase value is required for TMA async copy.");
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  tt::MemDescType allocTy = cast<tt::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  tt::MemDescType subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto view =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);

  Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
  Operation *copy = builder.create<ttng::AsyncTMACopyGlobalToLocalOp>(
      loc, loadOp.getDescPtr(), loadOp.getIndices(), barrier, view, pred);

  bool isMMV3Load = loadToInfo[loadOp].loadIsMMAV3;
  auto [stage, cluster] = schedule[loadOp];
  schedule.erase(loadOp);
  schedule.insert(copy, stage, cluster);

  builder.setInsertionPointAfter(waitOp);
  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  if (isMMV3Load) {
    auto alloc = cast<ttg::LocalAllocOp>((*loadOp->getUsers().begin()));
    alloc.replaceAllUsesWith(viewLoad.getResult());
    alloc.erase();
  } else {
    SmallVector<ttg::LocalAllocOp> allocsToErase;
    for (Operation *user : loadOp->getUsers()) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        alloc.replaceAllUsesWith(viewLoad.getResult());
        allocsToErase.push_back(alloc);
      }
    }
    for (auto alloc : allocsToErase) {
      alloc.erase();
    }

    auto sharedLoad = builder.create<ttg::LocalLoadOp>(
        loc, loadOp.getType(), viewLoad /*,wait->getResult(0)*/);
    auto result = sharedLoad->getResults();
    loadOp->replaceAllUsesWith(result);
  }
  loadOp.erase();
}

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return the shared encoding that needs to be
// used to be compatible with users' layouts. If there are imcompatible shared
// encodings set `incompatible` to true.
static std::optional<ttg::SharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value val, bool &incompatible) {
  ttg::SharedEncodingAttr attr;
  incompatible = false;
  for (Operation *user : val.getUsers()) {
    ttg::SharedEncodingAttr tempAttr;
    if (user->getNumResults() != 1)
      return std::nullopt;
    if (auto memDesc =
            dyn_cast<triton::MemDescType>(user->getResult(0).getType())) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SharedEncodingAttr>(memDesc.getEncoding());
      if (!getSharedEncIfAllUsersAreDotEnc(user->getResult(0), incompatible)
               .has_value())
        return std::nullopt;
    } else {
      if (!isa<ttg::LocalLoadOp, ttg::ConvertLayoutOp>(user))
        return std::nullopt;
      auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
          cast<TensorOrMemDesc>(user->getResult(0).getType()).getEncoding());
      if (!dotOpEnc)
        return std::nullopt;
      auto srcTy = cast<TensorOrMemDesc>(val.getType());
      auto CTALayout = ttg::getCTALayout(srcTy.getEncoding());
      auto order = ttg::getOrder(srcTy.getEncoding());
      unsigned bitWidth = srcTy.getElementType().getIntOrFloatBitWidth();
      tempAttr = ttg::SharedEncodingAttr::get(
          val.getContext(), dotOpEnc, srcTy.getShape(),
          ttg::getOrder(srcTy.getEncoding()),
          ttg::getCTALayout(srcTy.getEncoding()),
          srcTy.getElementType().getIntOrFloatBitWidth(), /*needTrans=*/false);
    }
    // Check that the shared encodings needed by the users are compatible.
    if (attr != nullptr && attr != tempAttr) {
      incompatible = true;
      return std::nullopt;
    }
    attr = tempAttr;
  }
  return attr;
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

// Create a map from load ops to their indirection level and the
// final use of the load op (another load op, or a dot op).
// Indirection level is "0" for the load op directly used by the dot op,
// "1" for the load op used by the load op used by the dot op, and so on.
static llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
loadOpsToIndirectionLevelAndUse(scf::ForOp forOp) {
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
      loadOpToIndLevelAndUse;
  DenseSet<Operation *> seen;

  std::function<void(Operation * op, int, Operation *)> dfs =
      [&](Operation *op, int distance, Operation *use) {
        if (!seen.insert(op).second)
          return;
        if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op)) {
          // TODO: What if there are multiple uses at different distances?
          loadOpToIndLevelAndUse.push_back(std::make_tuple(op, distance, use));
          use = op;
          distance++;
        }
        for (Value operand : op->getOperands()) {
          Value v = operand;
          Operation *defOp = v.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock()) {
            dfs(defOp, distance, use);
          }
        }
      };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!op.hasTrait<OpTrait::DotLike>())
      continue;
    seen.clear();
    dfs(&op, 0, &op);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (forOp->hasAttr(tt::kNumStagesAttrName)) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op))
        dfs(&op, 0, &op);
    }
  }

  return loadOpToIndLevelAndUse;
}

static bool loadIsMMAv3(Operation *loadOp) {
  if (!loadOp->hasOneUse())
    return false;
  auto alloc = dyn_cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin());
  if (!alloc)
    return false;
  auto sharedEnc = cast<ttg::SharedEncodingAttr>(alloc.getType().getEncoding());
  if (!sharedEnc.getHasLeadingOffset())
    return false;

  // MMA V3 case.
  auto newOrder = sharedEnc.getOrder();
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  auto oldOrder = ttg::getOrder(ty.getEncoding());

  // The operand of MMAv3 is in SharedEncoding and its order should not
  // be changed after FuseTranspositions Pass. So we only pipeline the
  // load if the order of the loaded BlockedEncoding is the same as the
  // order of the SharedEncoding it is converted to.
  return oldOrder == newOrder;
}

static llvm::MapVector<Operation *, LoadInfo>
assignMemoryLayouts(llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
                        &loadOpToIndLevelAndUse,
                    tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  llvm::MapVector<Operation *, LoadInfo> loadToInfo;

  for (auto &[op, dist, use] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(op))
      // TODO pawel: err, we'd need to verify that the distance is the same
      continue;
    LoadInfo loadInfo;

    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      assert(!isLoadFromTensorPtr(loadOp) &&
             "Block ptr should have been lowered before this pass.");
      auto ptr = loadOp.getPtr();
      unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
      if (auto mask = loadOp.getMask())
        vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

      auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
      if (!tensorTy)
        continue;
      auto ty =
          cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
      unsigned width = vec * ty.getIntOrFloatBitWidth();

      // We do not pipeline all loads for the following reasons:
      // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8, or 16.
      // 2. It's likely that pipling small loads won't offer much performance
      //    improvement and may even hurt performance by increasing register
      //    pressure.
      LDBG("Load " << *loadOp << " has width " << width);
      if (width < 32)
        continue;
    }

    if (use->hasTrait<OpTrait::DotLike>()) {
      loadInfo.usedByDot = true;
      if (loadIsMMAv3(op)) {
        loadInfo.loadIsMMAV3 = true;
        loadInfo.sharedEncoding =
            getSharedEncoding(op, /*loadIsMMAv3=*/true).value_or(nullptr);
      } else if (isa<tt::ExperimentalDescriptorLoadOp>(op)) {
        loadInfo.sharedEncoding =
            getSharedEncoding(op, /*loadIsMMAv3=*/true).value_or(nullptr);
      } else if (auto dot = dyn_cast<tt::DotOp>(use)) {
        bool incompatible = false;
        loadInfo.sharedEncoding =
            getSharedEncIfAllUsersAreDotEnc(op->getResult(0), incompatible)
                .value_or(nullptr);
        // If we can't agree on a shared encoding skip pipelinig the load.
        if (incompatible)
          continue;
        // HACK: Triton LLVM codegen has a bug where local_loads from #shared to
        // #mma layout can lead to invalid code if the loaded shape is smaller
        // than the mma tile (e.g. loading a 128x1 tensor for an MMAv2 dot with
        // tile {16,8} is bad because 1 < 8).  To work around this, don't
        // pipeline such loads.
        //
        // The codegen bug is caught by an assertion, so if you think you've
        // fixed it, feel free to delete this code and see if the assert still
        // fails.  :)
        if (!loadInfo.sharedEncoding) {
          if (auto dotEnc = dyn_cast<ttg::NvidiaMmaEncodingAttr>(
                  dot.getResult().getType().getEncoding())) {
            auto loadTy = cast<RankedTensorType>(op->getResultTypes()[0]);
            auto mmaInstrShape = dotEnc.getInstrShape();
            if (loadTy.getRank() < mmaInstrShape.size())
              continue;
            bool ok = true;
            for (int i = 0; i < mmaInstrShape.size(); i++) {
              if (loadTy.getShape()[loadTy.getRank() - mmaInstrShape.size() +
                                    i] < mmaInstrShape[i]) {
                ok = false;
                break;
              }
            }
            // If this load might trigger the bug, don't do the fallback logic
            // below, which might allow the load to be pipelined.
            if (!ok)
              continue;
          }
        }
      }
    } else if (auto loadOp = dyn_cast<tt::LoadOp>(use)) {
      // The use of this loadOp is another loadOp. If the use is not in the
      // loadsToPipeline already, it means that the use is not valid for
      // pipelining for some reason. We should skip this loadOp, too. Note that
      // we have an assumption that distAndUse.second (i.e. the use of this
      // loadOp) has already be processed in a previous loop iteration. This
      // assumption is held by how loadOpsToIndirectionLevelAndUse recursively
      // collects loadOpToIndLevelAndUse using DFS.
      if (loadToInfo.count(loadOp) == 0) {
        continue;
      }
    }

    // If we still don't have a shared encoding, try a "generic" shared
    // encoding.
    if (!loadInfo.sharedEncoding && !isa<ttng::WarpGroupDotOp>(use)) {
      loadInfo.sharedEncoding =
          getSharedEncoding(op, /*isMMAV3=*/loadInfo.loadIsMMAV3)
              .value_or(nullptr);
      if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
        loadInfo.blockedEncoding = getBlockedEncoding(loadOp, axisInfoAnalysis);
      }
    }

    // If that still didn't work, bail on pipelining this load.
    if (!loadInfo.sharedEncoding) {
      continue;
    }
    loadToInfo[op] = loadInfo;
  }

  return loadToInfo;
}

static llvm::MapVector<Operation *, LoadInfo>
scheduleLoads(scf::ForOp forOp, tt::CoarseSchedule &schedule,
              DenseSet<Operation *> &rootUsers, int numStages) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // Get all loads that are (transitively) used by dot ops and their distance
  // to the dot op.
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
      loadOpToIndLevelAndUse = loadOpsToIndirectionLevelAndUse(forOp);
  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
    for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
      LDBG("  - load: " << *l);
      LDBG("    at indirection level: " << i);
      LDBG("    used by op: " << *u);
    }
  });
  if (loadOpToIndLevelAndUse.empty())
    return {};

  // Check which loads are good for pipelining, and assign them
  // memory layouts.
  llvm::MapVector<Operation *, LoadInfo> loadToInfo =
      assignMemoryLayouts(loadOpToIndLevelAndUse, axisInfoAnalysis);

  if (loadToInfo.empty())
    return {};

  // Calculate the stage distance between applicable loads.
  int maxIndirectionLevel = -1;
  for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(loadOp) == 0)
      continue;
    maxIndirectionLevel = std::max(maxIndirectionLevel, dist);
  }
  unsigned stagesBetweenLoads =
      ceil<unsigned>(numStages - 2, maxIndirectionLevel + 1);

  tt::CoarseSchedule::Cluster rootUsersCluster = schedule.clusters.newAtFront();
  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(loadOp) == 0)
      continue;
    // Non-LoadOp(s) are the root uses of all LoadOp(s) and should be
    // always present in the opInfo
    if (!isa<tt::LoadOp>(use)) {
      schedule.insert(use, numStages - 1, rootUsersCluster);
      rootUsers.insert(use);
    }
  }

  SmallVector<tt::CoarseSchedule::Cluster> loadsClusters;
  for (int i = 0; i < maxIndirectionLevel + 1; i++) {
    loadsClusters.push_back(schedule.clusters.newAtBack());
  }
  // Assign stages to the loads.
  for (auto [loadOp, indLevel, _] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(loadOp) == 0)
      continue;
    int stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads;
    schedule.insert(loadOp, stage, loadsClusters[indLevel]);
  }

  // Distance from the load to the use.
  for (auto [loadOp, _, use] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(loadOp) == 0)
      continue;
    loadToInfo[loadOp].distToUse = schedule[use].first - schedule[loadOp].first;
  }

  return loadToInfo;
}

// Schedule the prologue and epilogue `if` ops in the loop, pushing them as
// close to the loop boundaries as possible. Return the cluster after the
// prologue (or the beginning of the loop if there is no prologue).
static tt::CoarseSchedule::Cluster
schedulePrologueAndEpilogue(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                            DenseSet<Operation *> &rootUsers, int numStages) {
  tt::CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  // Look for the IfOp that is in the backward slice any of the currently
  // scheduled ops and put it at the beginning of the loop.
  DenseMap<scf::IfOp, int> ifsToStage;
  // Go stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : schedule.getOpsInOrder(forOp)) {
      if (stage_ != stage)
        continue;
      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      getBackwardSlice((Operation *)op, &backwardSlice, opt);

      for (auto op : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          ifsToStage.insert({ifOp, stage});
        }
      }
    }
  }
  tt::CoarseSchedule::Cluster prologueCluster = schedule.clusters.newAtFront();
  for (auto [ifOp, stage] : ifsToStage) {
    schedule.insert(ifOp, stage, prologueCluster);
  }

  // Look for the IfOp that is in the forward slice of the root users and put it
  // at the end of the loop.
  tt::CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (auto rootUser : rootUsers) {
    SetVector<Operation *> forwardSlice;
    getForwardSlice(rootUser, &forwardSlice);

    int stage = schedule[rootUser].first;
    for (auto op : forwardSlice) {
      scf::IfOp ifOp = dyn_cast<scf::IfOp>(op);
      if (ifOp == nullptr) {
        // check if the op is in the body of an if op that's part of the loop
        auto parentOp = op->getParentOp();
        if (parentOp != nullptr &&
            parentOp->getParentOp() == forOp.getOperation()) {
          ifOp = dyn_cast<scf::IfOp>(parentOp);
        }
      }
      if (ifOp) {
        schedule.insertIfAbsent(ifOp, stage,
                                epilogueCluster); // after prefetch extracts
      }
    }
  }
  return afterPrologue;
}

// Add dependencies of anchor ops to the coarse schedule. Schedule them to
// the same stage and ordering cluster as the anchor op.
static void scheduleDependencies(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                                 int numStages) {
  SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
      opsInOrder = schedule.getOpsInOrder(forOp);
  // Schedule dependencies stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : opsInOrder) {
      if (stage_ != stage)
        continue;
      schedule.insertDepsOfOp(op, stage, cluster, false);
    }
  }
}

// Find dependencies with distance of 1. They will go to the next stage,
// but in the cluster before the current op.
static void scheduleDistanceOneDependencies(scf::ForOp forOp,
                                            tt::CoarseSchedule &schedule,
                                            int numStages) {
  auto getNestedOperands = [](Operation *op) -> SmallVector<Value> {
    SmallVector<Value> operands;
    op->walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (operand.getParentBlock()->getParentOp()->isAncestor(nestedOp))
          operands.push_back(operand);
      }
    });
    return operands;
  };

  // Mapping from the cluster to the cluster before it.
  DenseMap<tt::CoarseSchedule::Cluster *, tt::CoarseSchedule::Cluster>
      dist1Cluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      continue;
    auto [stage, cluster] = schedule[&op];
    // Can't schedule past the last stage.
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op.getBlock()) {
          auto yieldOp = op.getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp && schedule.count(defOp) == 0) {
            if (isa<tt::LoadOp>(defOp)) {
              // Exception: Schedule loads with a distance of 1 together
              // with the current op.
              schedule.insertIfAbsent(defOp, stage, cluster);
              schedule.insertDepsOfOp(defOp, stage, cluster, true);
            } else {
              if (dist1Cluster.count(&cluster) == 0) {
                dist1Cluster[&cluster] = schedule.clusters.newBefore(cluster);
              }
              schedule.insertIfAbsent(defOp, stage + 1, dist1Cluster[&cluster]);
              schedule.insertDepsOfOp(defOp, stage + 1, dist1Cluster[&cluster],
                                      true);
            }
          }
        }
      }
    }
  }
}

static void
scheduleRemainingToLastStage(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                             tt::CoarseSchedule::Cluster afterPrologue,
                             int numStages) {
  // Assign the rest of the ops to the last stage.
  // Take care of the ordering of the ops - uses cannot be scheduled to the
  // cluster before the definition.
  DenseMap<Operation *, tt::CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0) {
      opToCluster[&op] = afterPrologue;
    }
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    // We really only care about the producers from the last stage.
    // Others will be scheduled before these ops anyway.
    if (stage == numStages - 1) {
      queue.push_back(op);
    }
  }
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (auto user : op->getUsers()) {
      if (opToCluster.count(user)) {
        tt::CoarseSchedule::Cluster userCluster = opToCluster[user];
        tt::CoarseSchedule::Cluster opCluster = schedule[op].second;
        if (*userCluster < *opCluster) {
          opToCluster[user] = opCluster;
          queue.push_back(user);
        }
      }
    }
  }
  for (auto [op, cluster] : opToCluster) {
    schedule.insert(op, numStages - 1, cluster);
  }
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
  bool isTMALoad = false;
};

// Create barriers and wait ops for the async loads. Barriers may be shared by
// multiple loads is the schedule allows it.
static void createTMABarrierAndWait(
    scf::ForOp &forOp, SmallVector<AsyncLoad> &asyncLoads, Value insertIdx,
    Value extractIdx, Value phase, int numBuffers, tt::CoarseSchedule &schedule,
    SmallVector<Value> &barriers,
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

    Value barrierAlloc = createBarrierAlloc(forOp, numBuffers);
    barriers.push_back(barrierAlloc);
    Location loc = forOp.getLoc();
    OpBuilder builder(forOp);
    Attribute sharedMemorySpace =
        triton::gpu::SharedMemorySpaceAttr::get(builder.getContext());
    tt::MemDescType barrierTy = tt::MemDescType::get(
        {1}, builder.getI64Type(),
        cast<tt::MemDescType>(barrierAlloc.getType()).getEncoding(),
        sharedMemorySpace,
        /*mutableMemory=*/true);
    builder.setInsertionPoint(group[0]->loadOp);
    Value barrier = builder.create<ttg::MemDescSubviewOp>(
        loc, barrierTy, barrierAlloc, ArrayRef<Value>({insertIdx}));
    Value pred = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    Operation *expect = builder.create<ttng::BarrierExpectOp>(
        forOp.getLoc(), barrier, sizeInBytes, pred);
    auto [stage, cluster] = schedule[asyncLoads[0].loadOp];
    schedule.insert(expect, stage, cluster);

    builder.setInsertionPointAfter(group.back()->loadOp);
    Value barrierViewWait = builder.create<ttg::MemDescSubviewOp>(
        loc, barrierTy, barrierAlloc, ArrayRef<Value>({extractIdx}));
    Operation *wait =
        builder.create<ttng::WaitBarrierOp>(loc, barrierViewWait, phase);
    // Update the async loads info.
    for (AsyncLoad *asyncLoad : group) {
      asyncLoad->barrier = barrier;
      asyncLoad->waitOp = wait;
    }
  }
}

// Convert load ops into their asyn version and apply multi-buffering based on
// the required number of buffers.
static SmallVector<Value>
createAsyncOps(scf::ForOp &forOp, tt::CoarseSchedule &schedule,
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
                          numBuffers, schedule, barriers, loadToInfo);

  // Create a cluster for the prefetches. It may end up being empty, but this
  // is OK.
  tt::CoarseSchedule::Cluster prefetchCluster = schedule.clusters.newAtBack();

  for (AsyncLoad &asyncLoad : asyncLoads) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(asyncLoad.loadOp)) {
      createAsyncCopy(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                      schedule, prefetchCluster, loadToInfo, numStages);
    } else {
      auto descLoad = cast<tt::ExperimentalDescriptorLoadOp>(asyncLoad.loadOp);
      createTMAAsyncCopy(forOp, descLoad, asyncLoad.alloc, insertIdx,
                         extractIdx, asyncLoad.barrier, asyncLoad.waitOp, phase,
                         schedule, loadToInfo, numStages);
    }
  }
  SmallVector<Value> newYieldOperands = {insertIdx, extractIdx};
  if (phase)
    newYieldOperands.push_back(phase);
  // Patch the yield with the updated counters.
  appendToYield(forOp, newYieldOperands);

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
  // Schedule the loads and root ops (dot ops) in the loop. This will give us
  // a scaffold for the final schedule.
  DenseSet<Operation *> rootUsers;
  tt::CoarseSchedule coarseSchedule(numStages);
  llvm::MapVector<Operation *, LoadInfo> loadToInfo =
      scheduleLoads(forOp, coarseSchedule, rootUsers, numStages);
  if (loadToInfo.empty())
    return false;

  LLVM_DEBUG({
    LDBG("Coarse schedule loads only:");
    coarseSchedule.dump();
  });

  SmallVector<Value> barriers;
  // Convert the loads into async loads and create the allocs.
  SmallVector<Value> allocs =
      createAsyncOps(forOp, coarseSchedule, loadToInfo, barriers, numStages);

  LLVM_DEBUG({
    LDBG("Coarse schedule with async loads:");
    coarseSchedule.dump();
  });

  tt::CoarseSchedule::Cluster afterPrologue =
      schedulePrologueAndEpilogue(forOp, coarseSchedule, rootUsers, numStages);
  LLVM_DEBUG({
    LDBG("Coarse schedule with prologue and epilogue:");
    coarseSchedule.dump();
  });

  scheduleDependencies(forOp, coarseSchedule, numStages);
  LLVM_DEBUG({
    LDBG("Coarse schedule with dependencies:");
    coarseSchedule.dump();
  });

  scheduleDistanceOneDependencies(forOp, coarseSchedule, numStages);
  LLVM_DEBUG({
    LDBG("Coarse schedule with dist 1:");
    coarseSchedule.dump();
  });

  scheduleRemainingToLastStage(forOp, coarseSchedule, afterPrologue, numStages);
  LLVM_DEBUG({
    LDBG("Final coarse schedule:");
    coarseSchedule.dump();
  });

  // Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      coarseSchedule.createFinalSchedule(forOp);

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
        transitiveOperand.getDefiningOp())) {
      transitiveOperand = transitiveOperand.getDefiningOp()->getOperand(0);
    }
    return forOp.isDefinedOutsideOfLoop(transitiveOperand) ||
           isa<ttg::MemDescSubviewOp>(transitiveOperand.getDefiningOp());
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
