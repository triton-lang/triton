#include "PipelineExpander.h"
#include "PipeliningUtility.h"
#include "Schedule.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

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

/*struct PipelinedOpInfo {
  int stage = -1;
  Operation *use = nullptr;

  // Layout of the data in the shared memory.
  ttg::SharedEncodingAttr sharedEncoding = nullptr;
  // Blocked encoding is used for loads feeding into other loads.
  ttg::BlockedEncodingAttr blockedEncoding = nullptr;
  bool loadIsMMAV3 = false;
};*/

struct LoadInfo {
  // Layout of the data in the shared memory.
  ttg::SharedEncodingAttr sharedEncoding = nullptr;
  // Blocked encoding is used for loads not used by the dot.
  ttg::BlockedEncodingAttr blockedEncoding = nullptr;
  bool loadIsMMAV3 = false;
  int distToUse = -1;
  bool usedByDot = false;
};

} // namespace

static bool isMMAv3Dot(Operation *op) {
  auto dot = dyn_cast<tt::DotOp>(op);
  if (!dot)
    return false;
  auto enc = dot.getType().getEncoding().dyn_cast<ttg::NvidiaMmaEncodingAttr>();
  return enc && enc.isHopper();
}

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

static void createAsyncCopy(
    scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc, Value insertIdx,
    Value extractIdx,
    llvm::DenseMap<Operation *, std::pair<int, int>> &opToStageAndOrder,
    llvm::MapVector<tt::LoadOp, LoadInfo> &loadToInfo, int numStages) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
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
  }

  tt::MemDescType allocTy = cast<tt::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  tt::MemDescType subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), /*mutableMemory=*/true);
  auto view =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);
  Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, src, view, mask, loadOp.getOther(), loadOp.getCache(),
      loadOp.getEvict(), loadOp.getIsVolatile());
  Operation *commmit =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copy->getResult(0));
  Operation *wait =
      builder.create<ttg::AsyncWaitOp>(loc, commmit->getResult(0), 0);

  auto [stage, order] = opToStageAndOrder[loadOp];
  bool isMMV3Load = loadToInfo[loadOp].loadIsMMAV3;
  opToStageAndOrder[copy] = {stage, order};
  opToStageAndOrder[commmit] = {stage, order};
  opToStageAndOrder.erase(loadOp);

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
    auto sharedLoad =
        builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad);
    loadOp->replaceAllUsesWith(sharedLoad->getResults());

    // Prefetch load if is not MMAV3 and is used by the dot.
    if (loadToInfo[loadOp].usedByDot) {
      opToStageAndOrder[wait] = {numStages - 2, numStages * 2};
      opToStageAndOrder[viewLoad] = {numStages - 2, numStages * 2};
    }
  }
  loadOp.erase();
}

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and get the shared encoding that
// needs to be used to be compatible with users' layouts.
static std::optional<ttg::SharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value val) {
  ttg::SharedEncodingAttr attr;
  for (Operation *user : val.getUsers()) {
    ttg::SharedEncodingAttr tempAttr;
    if (user->getNumResults() != 1)
      return std::nullopt;
    if (auto memDesc =
            dyn_cast<triton::MemDescType>(user->getResult(0).getType())) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = memDesc.getEncoding().cast<ttg::SharedEncodingAttr>();
      if (!getSharedEncIfAllUsersAreDotEnc(user->getResult(0)).has_value())
        return std::nullopt;
    } else {
      if (!isa<ttg::LocalLoadOp, ttg::ConvertLayoutOp>(user))
        return std::nullopt;
      auto dotOpEnc = user->getResult(0)
                          .getType()
                          .cast<TensorOrMemDesc>()
                          .getEncoding()
                          .dyn_cast<ttg::DotOperandEncodingAttr>();
      if (!dotOpEnc)
        return std::nullopt;
      auto srcTy = val.getType().cast<TensorOrMemDesc>();
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
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return std::nullopt;
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
getSharedEncoding(tt::LoadOp loadOp, bool isMMAV3) {
  auto ty = cast<RankedTensorType>(loadOp.getType());
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
      auto enc =
          localAlloc.getType().getEncoding().cast<ttg::SharedEncodingAttr>();
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
static llvm::SmallVector<std::tuple<tt::LoadOp, int, Operation *>>
loadOpsToIndirectionLevelAndUse(scf::ForOp forOp) {
  llvm::SmallVector<std::tuple<tt::LoadOp, int, Operation *>>
      loadOpToIndLevelAndUse;
  DenseSet<Operation *> seen;

  std::function<void(Operation * op, int, Operation *)> dfs =
      [&](Operation *op, int distance, Operation *use) {
        if (!seen.insert(op).second)
          return;
        if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
          // TODO: What if there are multiple uses at different distances?
          loadOpToIndLevelAndUse.push_back(
              std::make_tuple(loadOp, distance, use));
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
    if (!isa<tt::DotOp>(op))
      continue;
    seen.clear();
    dfs(&op, 0, &op);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (forOp->hasAttr(tt::kNumStagesAttrName)) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<tt::LoadOp>(op))
        dfs(&op, 0, &op);
    }
  }

  return loadOpToIndLevelAndUse;
}

static bool loadIsMMAv3(tt::LoadOp loadOp) {
  if (!loadOp->hasOneUse())
    return false;
  auto alloc = dyn_cast<ttg::LocalAllocOp>(*loadOp->getUsers().begin());
  if (!alloc)
    return false;
  auto sharedEnc =
      alloc.getType().getEncoding().cast<ttg::SharedEncodingAttr>();
  if (!sharedEnc.getHasLeadingOffset())
    return false;

  // MMA V3 case.
  auto newOrder = sharedEnc.getOrder();
  auto ty = cast<RankedTensorType>(loadOp.getType());
  auto oldOrder = ttg::getOrder(ty.getEncoding());

  // The operand of MMAv3 is in SharedEncoding and its order should not
  // be changed after FuseTranspositions Pass. So we only pipeline the
  // load if the order of the loaded BlockedEncoding is the same as the
  // order of the SharedEncoding it is converted to.
  return oldOrder == newOrder;
}

static llvm::DenseSet<tt::LoadOp>
assignMemoryLayouts(llvm::MapVector<tt::LoadOp, LoadInfo> &loadToInfo,
                    llvm::SmallVector<std::tuple<tt::LoadOp, int, Operation *>>
                        &loadOpToIndLevelAndUse,
                    tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {

  llvm::DenseSet<tt::LoadOp> loadsToPipeline;

  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    assert(!isLoadFromTensorPtr(loadOp) &&
           "Block ptr should have been lowered before this pass.");

    if (loadsToPipeline.count(loadOp))
      // TODO pawel: err, we'd need to verify that the distance is the same
      continue;

    LoadInfo loadInfo;
    auto ptr = loadOp.getPtr();
    unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
    if (auto mask = loadOp.getMask())
      vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

    auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
    if (!tensorTy)
      continue;
    auto ty =
        tensorTy.getElementType().cast<tt::PointerType>().getPointeeType();
    unsigned width = vec * ty.getIntOrFloatBitWidth();

    // We do not pipeline all loads for the following reasons:
    // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8, or 16.
    // 2. It's likely that pipling small loads won't offer much performance
    //    improvement and may even hurt performance by increasing register
    //    pressure.
    LDBG("Load " << *loadOp << " has width " << width);
    if (width < 32)
      continue;

    if (auto dot = dyn_cast<tt::DotOp>(use)) {
      loadInfo.usedByDot = true;
      if (loadIsMMAv3(loadOp)) {
        loadInfo.loadIsMMAV3 = true;
        loadInfo.sharedEncoding =
            getSharedEncoding(loadOp, /*loadIsMMAv3=*/true).value_or(nullptr);
      } else {
        loadInfo.sharedEncoding =
            getSharedEncIfAllUsersAreDotEnc(loadOp.getResult())
                .value_or(nullptr);

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
          if (auto dotEnc = dot.getResult()
                                .getType()
                                .getEncoding()
                                .dyn_cast<ttg::NvidiaMmaEncodingAttr>()) {
            auto loadTy = loadOp.getType().cast<RankedTensorType>();
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
      if (loadsToPipeline.count(loadOp) == 0) {
        continue;
      }
    }

    // If we still don't have a shared encoding, try a "generic" shared
    // encoding.
    if (!loadInfo.sharedEncoding && !isMMAv3Dot(use)) {
      loadInfo.sharedEncoding =
          getSharedEncoding(loadOp, /*isMMAV3=*/loadInfo.loadIsMMAV3)
              .value_or(nullptr);
      loadInfo.blockedEncoding = getBlockedEncoding(loadOp, axisInfoAnalysis);
    }

    // If that still didn't work, bail on pipelining this load.
    if (!loadInfo.sharedEncoding) {
      continue;
    }

    loadsToPipeline.insert(loadOp);
    loadToInfo[loadOp] = loadInfo;
  }

  return loadsToPipeline;
}

/// Collect ops to pipeline. Returns true if any ops are found to pipeline.
static bool createCoarseSchedule(
    scf::ForOp forOp,
    llvm::DenseMap<Operation *, std::pair<int, int>> &opToStageAndOrder,
    llvm::MapVector<tt::LoadOp, LoadInfo> &loadToInfo, int numStages) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // Get all loads that are (transitively) used by dot ops and their distance
  // to the dot op.
  llvm::SmallVector<std::tuple<tt::LoadOp, int, Operation *>>
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
    return false;

  // Check which loads are good for pipelining, and assign them
  // memory layouts.
  DenseSet<tt::LoadOp> loadsToPipeline =
      assignMemoryLayouts(loadToInfo, loadOpToIndLevelAndUse, axisInfoAnalysis);

  if (loadsToPipeline.empty())
    return false;

  DenseSet<Operation *> rootUsers;

  // Calculate the stage distance between applicable loads.
  int maxIndirectionLevel = -1;
  for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse) {
    if (loadsToPipeline.count(loadOp) == 0)
      continue;
    maxIndirectionLevel = std::max(maxIndirectionLevel, dist);
  }
  unsigned stagesBetweenLoads =
      ceil<unsigned>(numStages - 2, maxIndirectionLevel + 1);

  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    // Non-LoadOp(s) are the root uses of all LoadOp(s) and should be
    // always present in the opInfo
    if (!isa<tt::LoadOp>(use)) {
      opToStageAndOrder[use] = std::make_pair(numStages - 1, 0);
      rootUsers.insert(use);
    }
  }

  // Assign stages to the loads.
  for (auto [loadOp, indLevel, _] : loadOpToIndLevelAndUse) {
    if (loadsToPipeline.count(loadOp) == 0)
      continue;
    int stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads;
    opToStageAndOrder[loadOp] =
        std::make_pair(stage, (numStages - 1 - stage) * 2);
  }

  // Distance from the load to the use.
  for (auto [loadOp, _, use] : loadOpToIndLevelAndUse) {
    if (loadsToPipeline.count(loadOp) == 0)
      continue;
    loadToInfo[loadOp].distToUse =
        opToStageAndOrder[use].first - opToStageAndOrder[loadOp].first;
  }

  // Look for the IfOp that is in the backward slice of the loads and put it in
  // the beginning of the loop.
  for (auto loadOp : loadsToPipeline) {
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    getBackwardSlice((Operation *)loadOp, &backwardSlice, opt);

    int loadStage = opToStageAndOrder[loadOp].first;
    for (auto op : backwardSlice) {
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        opToStageAndOrder[ifOp] = {
            loadStage, -2}; // -2 will be even before regular dist1uses
      }
    }
  }

  // Look for the IfOp that is in the forward slice of the root users and put it
  // at the end of the loop.
  for (auto rootUser : rootUsers) {
    SetVector<Operation *> forwardSlice;
    getForwardSlice(rootUser, &forwardSlice);

    int stage = opToStageAndOrder[rootUser].first;
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
      if (ifOp && !opToStageAndOrder.count(ifOp)) {
        opToStageAndOrder[ifOp] = {
            stage, numStages * 2 + 2}; // 2 will be even after extracts
      }
    }
  }

  return true;
}

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, tt::LoadOp loadOp,
                         ttg::SharedEncodingAttr sharedEnc, unsigned distance) {
  OpBuilder builder(forOp);
  auto ty = cast<RankedTensorType>(loadOp.getType());
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type memdescType = mlir::triton::MemDescType::get(
      bufferShape, ty.getElementType(), sharedEnc, /*mutableMemory*/ true);
  Value alloc = builder.create<mlir::triton::gpu::LocalAllocOp>(
      loadOp.getLoc(), memdescType, Value());
  return alloc;
}

// Convert load ops into their asyn version and apply multi-buffering based on
// the required number of buffers.
static SmallVector<Value> createAsyncOps(
    scf::ForOp &forOp,
    llvm::DenseMap<Operation *, std::pair<int, int>> &opToStageAndOrder,
    llvm::MapVector<tt::LoadOp, LoadInfo> &loadToInfo, int numBuffers,
    int numStages) {
  struct AsyncLoad {
    AsyncLoad(tt::LoadOp loadOp, Value alloc) : loadOp(loadOp), alloc(alloc) {}
    tt::LoadOp loadOp;
    Value alloc;
  };
  SmallVector<AsyncLoad> asyncLoads;
  SmallVector<Value> allocs;
  for (auto &[loadOp, info] : loadToInfo) {
    assert(info.sharedEncoding && "LoadOp shared encoding not defined.");
    Value alloc = createAlloc(forOp, loadOp, info.sharedEncoding, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    allocs.push_back(alloc);
    asyncLoads.emplace_back(loadOp, alloc);
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
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
  SmallVector<Value> newOperands;
  newOperands.push_back(insertIdx);
  newOperands.push_back(extractIdx);
  Value phase;
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;
  insertIdx = newForOp.getBody()->getArgument(newOperandIndex);
  extractIdx = newForOp.getBody()->getArgument(newOperandIndex + 1);

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

  for (AsyncLoad &asyncLoad : asyncLoads) {
    createAsyncCopy(forOp, asyncLoad.loadOp, asyncLoad.alloc, insertIdx,
                    extractIdx, opToStageAndOrder, loadToInfo, numStages);
  }
  SmallVector<Value> newYieldOperands = {insertIdx, extractIdx};
  // Patch the yield with the updated counters.
  appendToYield(forOp, newYieldOperands);

  return allocs;
}

static void
printSchedule(std::vector<std::pair<Operation *, unsigned>> &schedule,
              int numStages) {
  LDBG("Schedule:");
  for (int i = 0; i < numStages; i++) {
    LDBG("- Stage " << i);
    for (auto &pair : schedule) {
      if (pair.second == i) {
        pair.first->dump();
      }
    }
  }
}

// Create the schedule for a matmul loop. This is ad hoc based on how we know
// matmul loops should be pipelined and is not a generic scheduler.
static std::vector<std::pair<Operation *, unsigned>> createSchedule(
    scf::ForOp forOp, int numStages,
    llvm::DenseMap<Operation *, std::pair<int, int>> &opToStageAndOrder,
    llvm::MapVector<tt::LoadOp, LoadInfo> &loadToInfo, bool prefetchExtract) {
  LLVM_DEBUG({
    LDBG("For loop:");
    forOp.dump();

    LDBG("Initial schedule:");
    for (int i = 0; i < numStages; i++) {
      LDBG("- Ops in stage " << i);
      for (auto &[op, stage] : opToStageAndOrder) {
        if (i == stage.first) {
          llvm::outs() << " ord: " << stage.second << " ";
          op->dump();
        }
      }
    }
  });

  int minOrder = *llvm::min_element(
      llvm::make_second_range(llvm::make_second_range(opToStageAndOrder)));
  int maxOrder = *llvm::max_element(
      llvm::make_second_range(llvm::make_second_range(opToStageAndOrder)));

  llvm::DenseMap<Operation *, std::pair<int, int>> newOpToStageAndOrder;

  DenseSet<Operation *> anchorOps;
  for (auto &[op, stage] : opToStageAndOrder) {
    if (stage.second == minOrder) {
      anchorOps.insert(op);
    }
  }

  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opToStageAndOrder.count(&op) == 0)
      continue;
    auto [stage, order] = opToStageAndOrder[&op];
    DenseSet<Operation *> anchorOpsAndDeps;
    tt::addDep(&op, anchorOpsAndDeps, false,
               &anchorOps); // don't go through anchorOps - they are what they
                            // have to be
    for (auto dep : anchorOpsAndDeps) {
      int depStage = stage;
      int depOrder = order;
      if (newOpToStageAndOrder.count(dep) > 0) {
        depStage = std::min(newOpToStageAndOrder[dep].first, depStage);
        depOrder = std::min(newOpToStageAndOrder[dep].second, depOrder);
      }

      if (opToStageAndOrder.count(dep) == 0) {
        newOpToStageAndOrder[dep] = {depStage, depOrder};
      }
    }
  }

  for (auto &[op, stageAndOrder] : newOpToStageAndOrder) {
    if (opToStageAndOrder.count(op) == 0) {
      opToStageAndOrder[op] = stageAndOrder;
    }
  }

  LLVM_DEBUG({
    LDBG("Scheduled dependencies:");
    for (int i = 0; i < numStages; i++) {
      LDBG("- Ops in stage " << i);
      for (auto &[op, stage] : opToStageAndOrder) {
        if (i == stage.first) {
          llvm::outs() << " ord: " << stage.second << " ";
          op->dump();
        }
      }
    }
  });

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

  // Find dependencies with distance of 1.
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opToStageAndOrder.count(&op) == 0)
      continue;
    auto [stage, order] = opToStageAndOrder[&op];
    // Can't schedule past the last stage.
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      if (auto arg = operand.dyn_cast<BlockArgument>()) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op.getBlock()) {
          auto yieldOp = op.getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp && opToStageAndOrder.count(defOp) == 0) {
            // Schedule distance 1 users in the next stage, but before them in
            // order.
            DenseSet<Operation *> distanceOneUsers;
            tt::addDep(defOp, distanceOneUsers, true);
            if (isa<tt::LoadOp>(defOp)) {
              // Schedule loads with a distance of 1 together with the anchor
              // ops.
              for (auto user : distanceOneUsers) {
                if (opToStageAndOrder.count(user) == 0) {
                  opToStageAndOrder[user] = {stage, order};
                }
              }
            } else {
              for (auto user : distanceOneUsers) {
                if (opToStageAndOrder.count(user) == 0) {
                  opToStageAndOrder[user] = {stage + 1, order - 1};
                }
              }
            }
          }
        }
      }
    }
  }

  minOrder = *llvm::min_element(
      llvm::make_second_range(llvm::make_second_range(opToStageAndOrder)));
  maxOrder = *llvm::max_element(
      llvm::make_second_range(llvm::make_second_range(opToStageAndOrder)));

  LLVM_DEBUG({
    LDBG("Scheduled dependencies with dist 1:");
    for (int i = 0; i < numStages; i++) {
      LDBG("- Ops in stage " << i);
      for (auto &[op, stage] : opToStageAndOrder) {
        if (i == stage.first) {
          llvm::outs() << " ord: " << stage.second << " ";
          op->dump();
        }
      }
    }
  });

  // Assign the rest of the ops to the last stage.
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (opToStageAndOrder.count(&op) == 0) {
      opToStageAndOrder[&op] = {numStages - 1, 0};
    }
  }

  LLVM_DEBUG({
    LDBG("Remaining ops to the last stage:");
    for (int i = 0; i < numStages; i++) {
      LDBG("- Ops in stage " << i);
      for (auto &[op, stage] : opToStageAndOrder) {
        if (i == stage.first) {
          llvm::outs() << " ord: " << stage.second << " ";
          op->dump();
        }
      }
    }
  });

  // Create the actual schedule
  std::vector<std::pair<Operation *, unsigned>> schedule;

  // Insert the ops in the reverse order of the stages. This helps with saving
  // the number of required buffers.
  // TODO pawel: this causes liveranges to be longer than necessary:
  // order 0:
  //  stage n-1, n-2, ..., 1, 0
  // order 1:
  //  stage n-1, n-2, ..., 1, 0
  // Stuff from order 0 stage 1 for example used in order 1 stage 0 needs to
  // be live through all of stage 1. Previously we were scheduling stage n-1,
  // and then dist1uses followed by inserts/deps.
  for (int ord = minOrder; ord <= maxOrder; ord++) {
    for (auto &op : forOp.getBody()->without_terminator()) {
      assert(opToStageAndOrder.count(&op) && "Op not in schedule!");
      assert(opToStageAndOrder[&op].first < numStages &&
             "Op with invalid stage!");
      if (opToStageAndOrder[&op].second == ord)
        schedule.push_back({&op, opToStageAndOrder[&op].first});
    }
  }

  LLVM_DEBUG(printSchedule(schedule, numStages));

  return schedule;
}

constexpr static char kNeedWaitAttrName[] = "triton.pipeline.needs_wait";

bool mlir::triton::preProcessLoopAndGetSchedule(
    scf::ForOp &forOp, int numStages, mlir::triton::PipeliningOption &options) {
  // 1. First collect "interesting" operations with a stage where to schedule
  // them. This gives a coarse scheduling for the loop.
  llvm::DenseMap<Operation *, std::pair<int, int>> opToStageAndOrder;
  llvm::MapVector<tt::LoadOp, LoadInfo> loadToInfo;
  if (!createCoarseSchedule(forOp, opToStageAndOrder, loadToInfo, numStages))
    return false;

  bool hasMMAV3 =
      llvm::any_of(loadToInfo, [](auto &kv) { return kv.second.loadIsMMAV3; });

  // Calculate the number of buffers needed for each load.
  // TODO pawel: we could do more fine-grained allocation here and
  // allocate only the number of buffers that specific loads need.
  // Instead, we allocate the maximum number of buffers needed by any load.
  int maxNumBuffers =
      llvm::max_element(llvm::make_second_range(loadToInfo), [](auto &lhs,
                                                                auto &rhs) {
        return lhs.distToUse < rhs.distToUse;
      })->distToUse;
  if (hasMMAV3) {
    // For MMAv3, we need an extra buffer as this is assumed in the wgmma
    // pipelining post-processing.
    maxNumBuffers++;
  };

  // 2. Convert the loads into async loads and create the allocs.
  SmallVector<Value> allocs = createAsyncOps(
      forOp, opToStageAndOrder, loadToInfo, maxNumBuffers, numStages);

  // 3. Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      createSchedule(forOp, numStages, opToStageAndOrder, loadToInfo,
                     /*prefetchExtract=*/!hasMMAV3);

  // 4. Fill out the pipeline options.
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
    if (auto arg = val.dyn_cast<BlockArgument>()) {
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
  llvm::SmallSetVector<ttg::AsyncWaitOp, 8> toDelete;
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
    toDelete.insert(waitGroup.begin(), waitGroup.end());
  }
  for (auto waitOp : toDelete) {
    waitOp->erase();
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
//   %a = ttng.dot_async %alloc
//   %a1 = ttng.dot_wait %a
//
// then we want the wait to depend on %alloc as well as %a.  This extends the
// live range of %alloc, so that it won't be destroyed until after the dot is
// waited on.
//
// Specifically, this function finds all dot_async ops that elements of `values`
// depend on.  Then it adds the MemDesc operands of those dots to the wait.
static void threadValuesThroughWait(ttng::DotWaitOp wait,
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
  SmallVector<ttng::DotAsyncOp> asyncDots;
  for (Value v : values) {
    BackwardSliceOptions options;
    options.omitBlockArguments = true;
    options.filter = [&](Operation *op) {
      if (auto dot = dyn_cast<ttng::DotAsyncOp>(op)) {
        asyncDots.push_back(dot);
        return false;
      }
      return op->getBlock() == wait->getBlock();
    };
    SetVector<Operation *> slice;
    getBackwardSlice(v, &slice, options);
  }

  for (ttng::DotAsyncOp dot : asyncDots) {
    for (Value operand : dot.getOperands()) {
      if (isa<tt::MemDescType>(operand.getType())) {
        newOperands.insert(operand);
      }
    }
  }

  // We can't use replaceWithNewOp because we're changing the number of return
  // values in the operation.
  auto newWait = builder.create<ttng::DotWaitOp>(
      wait.getLoc(), llvm::to_vector(newOperands), wait.getPendings());
  for (int i = 0; i < origNumOperands; i++) {
    Value operand = wait.getResult(i);
    if (!isa<tt::MemDescType>(operand.getType()))
      operand.replaceAllUsesWith(newWait.getResult(i));
  }
  for (int i = origNumOperands; i < newOperands.size(); i++) {
    Value operand = newWait.getOperand(i);
    if (!isa<tt::MemDescType>(operand.getType()))
      operand.replaceAllUsesExcept(newWait.getResult(i), newWait);
  }
  wait->erase();
}

// Determines whether a given MMAv3 dot op, represented as ttng.dot_async, needs
// a wait immediately after it.
//
// In PTX, MMAv3 exists only as an asynchronous op.  In Triton, we can represent
// MMAv3 ops as either tt.dot (synchronous) or ttng.dot_async.  But even if we
// use ttng.dot_async, the conservative thing is to make a dot "effectively
// synchronous" by inserting a `ttng.dot_wait {pendings=0}` right after it.
//
// We can omit the wait and create a "properly async" dot if all of the
// following are true.
//
//  1. All operands that touch shared memory are multi-buffered, i.e. can't read
//     an incomplete value while it's being written asynchronously by a load.
//
//  2. During iteration i, nothing other than the loop's `yield` reads the
//     result of the dot.
//
//  3. During iteration i, between the start of the loop up until the first
//     `ttng.dot_wait {pendings=0}` op, the result of the dot from iteration i-1
//     is consumed only by other MMAv3 dots as the `c` operand.
//
//     This is safe because the following pseudo-PTX is valid:
//
//        %accum = dot_async %a1, %b1, %c1
//        %accum = dot_async %a2, %b2, %accum
//
//     That is, the second async dot can use the result of the first one without
//     an intervening wait.  However, the only operation that can legally read
//     %accum before the wait is another dot_async, and this only works for the
//     `c` operand, not `a` or `b`.  See
//     https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-fence
//     (ttng::DotAsyncOp corresponds to wgmma.fence followed by one or more
//     wgmma.async ops, so our understanding is that the two ttng::DotAsyncOps
//     don't have to correspond to wgmma.async ops with the same shapes as
//     specified in the docs, because there's an intervening fence.)
//
// If the op can be properly async, this function returns the index of the dot
// in the loop's iter_args.  (Rule (2) above ensures this is well-defined.)
//
static std::optional<int> dotCanBeProperlyAsync(ttng::DotAsyncOp dotOp,
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

  // Rule 2: The dot should only be used by the for loop's `yield`.
  if (!dotOp->hasOneUse() ||
      *dotOp->getUsers().begin() != forOp.getBody()->getTerminator()) {
    LDBG("Can't make dot async because it is not used only by the loop's "
         "`yield`.");
    return std::nullopt;
  }

  // The result of the dot becomes this loop carry value.
  auto iterArgIdx = dotOp->getUses().begin()->getOperandNumber();
  auto iterArg = forOp.getRegionIterArg(iterArgIdx);

  // Rule 3a: Are the only users of the dot's result from iteration i-1 other
  // MMAv3 dots?  If so, we're done, this dot can be properly async.
  if (llvm::all_of(iterArg.getUses(), [&](OpOperand &use) {
        return isa<ttng::DotAsyncOp>(use.getOwner()) &&
               use.getOperandNumber() == 2;
      })) {
    return iterArgIdx;
  }

  // Rule 3b: Are all users of the dot's result from iteration i-1 after the
  // first `dot_wait {pendings=0}` op?  If so, the dot can be properly async,
  // but we have to thread its result from iteration i-1 through the wait.
  auto waitOps = forOp.getBody()->getOps<ttng::DotWaitOp>();
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
    LDBG("MMAv3 dot can be properly async because it follows a dot_wait "
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
// depth 2, so there are at most 2 copies of each dot_async in flight at a
// time.)
//
// We can skip inserting the wait if we have a `dot_wait {pendings=0}` somewhere
// in the loop.  To see why, consider:
//
//   dot_async
//   dot_async; wait 0  // synchronous dot
//   dot_async
//   dot_async
//
// In this example, there are three properly-async dots, so we'd normally put
// `wait 3` at the end of the loop, meaning "wait until there are 3 or fewer
// pending async dots".  But note that when this iteration of the loop
// completes, there are only *two* pending async dots from this iteration, so
// this wait would do nothing.  This is true in general, no matter where the
// `wait 0` appears.
static void insertAsyncDotWaitInLoop(
    scf::ForOp forOp,
    const llvm::MapVector<Operation *, int /*iterArgIdx*/> &properlyAsyncDots) {
  if (properlyAsyncDots.empty())
    return;

  if (llvm::any_of(forOp.getBody()->getOps<ttng::DotWaitOp>(),
                   [](auto wait) { return wait.getPendings() == 0; })) {
    return;
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
  auto wait = builder.create<ttng::DotWaitOp>(lastAsyncDot->getLoc(),
                                              /*inputs=*/ArrayRef<Value>{},
                                              properlyAsyncDots.size());

  // Thread the results of the async dots through the wait.
  SmallVector<Value> addlWaitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    addlWaitOperands.push_back(asyncDot->getResult(0));
  }
  threadValuesThroughWait(wait, addlWaitOperands);
}

// Convert MMAv3 tt::DotOps (i.e. Hopper wgmma) into ttng::DotAsyncOps and
// insert ttng::DotWaitOps as necessary.
//
// We assume we have space for each dot to be pipelined to depth 2, i.e. each
// dot op in the loop can have at most 2 dot_async ops in flight at once.  (Each
// dot_async op usually corresponds to a series of wgmma.async ops.)
void triton::asyncLaunchDots(scf::ForOp forOp) {
  LDBG("Original loop:\n" << *forOp);

  // First, change every MMAv3 tt.dot into ttng.dot_async.  The rest of this
  // function is concerned with inserting ttng.dot_wait ops in the appropriate
  // places.
  //
  // It's not strictly necessary to convert every dot into dot_async:
  // Synchronous MMAv3 dots can be represented equally well as `tt.dot` or
  // `ttng.dot_async; wait 0`.  But this makes things easier elsewhere.
  //
  // We call those dots that don't need to be followed immediately by a `wait 0`
  // "properly async", or sometimes just "async".
  IRRewriter builder(forOp.getContext());
  for (auto dotOp : llvm::to_vector(forOp.getBody()->getOps<tt::DotOp>())) {
    if (isMMAv3Dot(dotOp)) {
      builder.setInsertionPoint(dotOp);
      builder.replaceOpWithNewOp<ttng::DotAsyncOp>(
          dotOp, dotOp.getA(), dotOp.getB(), dotOp.getC(),
          dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
    }
  }

  // For each dot, determine whether it can be properly async, or if it needs a
  // sync immediately after.  If it can be properly async, we know its only use
  // is in the loop's `yield` statement; asyncDots maps the op to its index in
  // the yield op.
  llvm::MapVector<Operation *, int /*iterArgIdx*/> properlyAsyncDots;
  for (auto dotOp : forOp.getBody()->getOps<ttng::DotAsyncOp>()) {
    if (auto iterArgIdx = dotCanBeProperlyAsync(dotOp, forOp)) {
      properlyAsyncDots[dotOp] = *iterArgIdx;
    } else {
      builder.setInsertionPointAfter(dotOp);
      auto wait =
          builder.create<ttng::DotWaitOp>(dotOp.getLoc(), ArrayRef<Value>{},
                                          /*pendings=*/0);
      SmallVector<Value> waitOperands = {dotOp.getResult()};
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
  insertAsyncDotWaitInLoop(forOp, properlyAsyncDots);

  // Finally, insert a wait after the loop, waiting for dots from the final
  // iteration of the loop.
  SmallVector<Value> waitOperands;
  for (auto [asyncDot, iterArgIdx] : properlyAsyncDots) {
    waitOperands.push_back(forOp.getResult(iterArgIdx));
  }
  // Wait until there are 0 outstanding async dot ops.
  builder.setInsertionPointAfter(forOp);
  auto dotWaitAfterLoop =
      builder.create<ttng::DotWaitOp>(forOp.getLoc(), ArrayRef<Value>{}, 0);
  threadValuesThroughWait(dotWaitAfterLoop, waitOperands);
}
