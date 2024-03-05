#include "PipelineExpander.h"
#include "PipeliningUtility.h"
#include "Schedule.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
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

struct PipelineOpInfo {
  int stage = -1;
  Operation *use = nullptr;

  // Layout of the data in the shared memory.
  ttg::SharedEncodingAttr sharedEncoding = nullptr;
  // Blocked encoding is used for loads feeding into other loads.
  ttg::BlockedEncodingAttr blockedEncoding = nullptr;
  bool loadIsMMAV3 = false;
};

}; // namespace

/// Replace the yield with a new one with the given operands appended.
static void appendToYield(scf::ForOp forOp, ArrayRef<Value> newOperands) {
  // Fix up the yield op.
  Operation *yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value> operands(yieldOp->getOperands().begin(),
                              yieldOp->getOperands().end());
  operands.append(newOperands.begin(), newOperands.end());
  OpBuilder builder(yieldOp);
  builder.create<scf::YieldOp>(yieldOp->getLoc(), operands);
  yieldOp->erase();
}

static void
createAsyncCopy(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                Value insertIdx, Value extractIdx,
                llvm::MapVector<Operation *, PipelineOpInfo> &opToInfo) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  if (!isExpensiveLoadOrStore(loadOp) && opToInfo[loadOp].blockedEncoding) {
    // For inexpensive loads that do not directly feed into dot ops
    // we want to use optimal layout for the data.
    ttg::BlockedEncodingAttr encoding = opToInfo[loadOp].blockedEncoding;
    auto convertBlockLayout = [&](Value src) {
      auto ty = src.getType().cast<RankedTensorType>();
      auto newTy =
          RankedTensorType::get(ty.getShape(), ty.getElementType(), encoding);
      auto cvt =
          builder.create<ttg::ConvertLayoutOp>(loadOp->getLoc(), newTy, src);
      return cvt.getResult();
    };
    src = convertBlockLayout(src);
    mask = convertBlockLayout(src);
  }

  SmallVector<Value> copyOffsets = {insertIdx, zero, zero};
  tt::MemDescType allocTy = alloc.getType().cast<tt::MemDescType>();
  tt::MemDescType subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), /*mutableMemory=*/true);
  auto view =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);
  Operation *copy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, src, view, loadOp.getMask(), loadOp.getOther(), loadOp.getCache(),
      loadOp.getEvict(), loadOp.getIsVolatile());
  Operation *commmit =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copy->getResult(0));
  builder.create<ttg::AsyncWaitOp>(loc, commmit->getResult(0), 0);

  int stage = opToInfo[loadOp].stage;
  bool isMMV3Load = opToInfo[loadOp].loadIsMMAV3;
  opToInfo.insert({copy, {.stage = stage}});
  opToInfo.insert({commmit, {.stage = stage}});
  opToInfo.erase(loadOp);

  // Extract part.
  SmallVector<Value> loadOffsets = {extractIdx, zero, zero};
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  if (isMMV3Load) {
    auto alloc = cast<ttg::LocalAllocOp>((*loadOp->getUsers().begin()));
    alloc.replaceAllUsesWith(viewLoad.getResult());
    alloc.erase();
  } else {
    for (Operation *user : loadOp->getUsers()) {
      if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        alloc.replaceAllUsesWith(viewLoad.getResult());
        alloc.erase();
      }
    }
    auto sharedLoad =
        builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad);
    loadOp->replaceAllUsesWith(sharedLoad->getResults());
  }
  loadOp.erase();
}

/// Create an async load equivalent to the given load.
static void
createAsyncLoad(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                Value insertIdx, Value extractIdx, Value phase,
                llvm::MapVector<Operation *, PipelineOpInfo> &opToInfo) {
  createAsyncCopy(forOp, loadOp, alloc, insertIdx, extractIdx, opToInfo);
}

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and set the shared encoding that
// needs to be used to be compatible with users' layouts.
static bool
allTransitiveUsesHaveDotEncoding(Value val,
                                 ttg::SharedEncodingAttr &sharedEnc) {
  ttg::SharedEncodingAttr attr;
  for (Operation *user : val.getUsers()) {
    ttg::SharedEncodingAttr tempAttr;
    if (user->getNumResults() != 1)
      return false;
    if (auto memDesc =
            user->getResult(0).getType().dyn_cast<triton::MemDescType>()) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      if (!tempAttr)
        tempAttr = memDesc.getEncoding().cast<ttg::SharedEncodingAttr>();
      ttg::SharedEncodingAttr nextEncoding;
      bool hasDotEncodingUse =
          allTransitiveUsesHaveDotEncoding(user->getResult(0), nextEncoding);
      if (!hasDotEncodingUse)
        return false;
    } else {
      if (!isa<ttg::LocalLoadOp, ttg::ConvertLayoutOp>(user))
        return false;
      auto dotOpEnc = user->getResult(0)
                          .getType()
                          .cast<TensorOrMemDesc>()
                          .getEncoding()
                          .dyn_cast<ttg::DotOperandEncodingAttr>();
      if (!dotOpEnc)
        return false;
      auto srcTensorType = val.getType().cast<TensorOrMemDesc>();
      auto CTALayout = ttg::getCTALayout(srcTensorType.getEncoding());
      auto order = ttg::getOrder(srcTensorType.getEncoding());
      unsigned bitWidth =
          srcTensorType.getElementType().getIntOrFloatBitWidth();
      tempAttr = ttg::SharedEncodingAttr::get(
          val.getContext(), dotOpEnc, srcTensorType.getShape(), order,
          CTALayout, bitWidth, /*needTrans=*/false);
    }
    // We need to check that the shared encoding needed by the users are
    // compatible.
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return false;
    attr = tempAttr;
  }
  sharedEnc = attr;
  return true;
}

bool loadDotOperand(tt::LoadOp loadOp, bool &hasMMAV3,
                    ttg::SharedEncodingAttr &enc) {
  if (loadOp.getResult().hasOneUse()) {
    Operation *use = *loadOp.getResult().getUsers().begin();
    if (auto alloc = llvm::dyn_cast<ttg::LocalAllocOp>(use)) {
      auto sharedEnc =
          alloc.getType().getEncoding().cast<ttg::SharedEncodingAttr>();
      if (sharedEnc.getHasLeadingOffset()) {
        // MMA V3 case.
        auto newOrder = sharedEnc.getOrder();
        auto ty = loadOp.getType().cast<RankedTensorType>();
        auto oldOrder = ttg::getOrder(ty.getEncoding());
        if (newOrder[0] == oldOrder[0] || newOrder[1] == oldOrder[1]) {
          // The operand of MMAv3 is in SharedEncoding and it's order should
          // not be changed after FuseTranspositions Pass. So we only pipeline
          // the load if the order of the loaded BlockedEncoding is the same
          // as the order of the SharedEncoding it is converted to.
          // TODO: remove this constraint once the LoadOp supports transpose
          // fusion
          hasMMAV3 = true;
          return true;
        }
      }
    }
  }
  if (allTransitiveUsesHaveDotEncoding(loadOp.getResult(), enc))
    return true;
  return false;
};

static ttg::BlockedEncodingAttr
getBlockedEncoding(tt::LoadOp loadOp, tt::ModuleAxisInfoAnalysis &axisInfo) {
  Value src = loadOp.getPtr();
  auto ty = src.getType().cast<RankedTensorType>();
  auto mod = loadOp->getParentOfType<ModuleOp>();
  int numWarps = ttg::TritonGPUDialect::getNumWarps(mod);
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);
  tt::AxisInfo::DimVectorT contiguity =
      axisInfo.getAxisInfo(src)->getContiguity();
  SmallVector<unsigned> order = argSort(contiguity);
  unsigned currPerThread = getNumElementsPerThread(loadOp, order, axisInfo);
  SmallVector<unsigned> sizePerThread(order.size(), 1);
  sizePerThread[order[0]] = currPerThread;
  ttg::CTALayoutAttr CTALayout = ttg::getCTALayout(ty.getEncoding());
  return ttg::BlockedEncodingAttr::get(loadOp->getContext(), ty.getShape(),
                                       sizePerThread, order, numWarps,
                                       threadsPerWarp, CTALayout);
}

static ttg::SharedEncodingAttr getSharedEncoding(tt::LoadOp loadOp,
                                                 Operation *use, bool isMMAV3) {

  auto ty = loadOp.getType().cast<RankedTensorType>();
  auto CTALayout = ttg::getCTALayout(ty.getEncoding());
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
  if (isa<tt::DotOp>(use)) {
    assert(isMMAV3 && "Load used by dot op should be either MMAv3 or have a "
                      "shared encoding already picked based on users layouts.");
    return ttg::SharedEncodingAttr::get(ty.getContext(), ty.getShape(), order,
                                        CTALayout, ty.getElementType());
  } else {
    assert(!isMMAV3 && "Load used by non-dot op should not be MMAv3.");
    // Use non-swizzled layout for loads that do not feed into dot ops.
    // TODO: This won't be optimal for 2D tensors.
    return ttg::SharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order,
                                        CTALayout);
  }
}

// Create a map from load ops to their distance to the nearest dot op and the
// final use of the load op (another load op, or a dot op).
static llvm::MapVector<tt::LoadOp, std::pair<int, Operation *>>
loadOpsToDistanceAndUse(scf::ForOp forOp) {
  llvm::MapVector<tt::LoadOp, std::pair<int, Operation *>> loadOpToDistAndUse;
  DenseSet<Operation *> seen;

  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  auto isCandidate = [&](tt::LoadOp loadOp) -> bool {
    assert(!isLoadFromTensorPtr(loadOp) &&
           "Block ptr should have been lowered before this pass.");
    auto ptr = loadOp.getPtr();
    unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
    if (auto mask = loadOp.getMask())
      vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

    auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
    if (!tensorTy)
      return false;
    auto ty =
        tensorTy.getElementType().cast<tt::PointerType>().getPointeeType();
    unsigned width = vec * ty.getIntOrFloatBitWidth();
    // We do not pipeline all loads for the following reasons:
    // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8 and 16.
    // 2. It's likely that pipling small loads won't offer much performance
    //    improvement and may even hurt performance by increasing register
    //    pressure.
    return (width >= 32);
  };

  std::function<void(Operation * op, int, Operation *)> dfs =
      [&](Operation *op, int distance, Operation *use) {
        if (!seen.insert(op).second)
          return;
        if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
          if (!isCandidate(loadOp))
            return;
          loadOpToDistAndUse[loadOp] = std::make_pair(distance, use);
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

  return loadOpToDistAndUse;
}

/// Collect loads to pipeline. Returns true if loads are found to pipeline.
static bool
collectOpsToPipeline(scf::ForOp forOp,
                     llvm::MapVector<Operation *, PipelineOpInfo> &opInfo,
                     int numStages, bool &hasMMAV3) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // Loads ordered by their dependency distance to the nearest dot op.
  llvm::MapVector<tt::LoadOp, std::pair<int, Operation *>> loadOpToDistAndUse =
      loadOpsToDistanceAndUse(forOp);
  if (loadOpToDistAndUse.empty())
    return false;

  int maxDistance = -1;
  for (auto &[op, distAndUse] : loadOpToDistAndUse) {
    if (distAndUse.first > maxDistance) {
      maxDistance = distAndUse.first;
    }
  }
  assert(maxDistance >= 0);

  // Start by initializing PipelineOpInfo for users of the loads.
  for (auto &[loadOp, distAndUse] : loadOpToDistAndUse)
    opInfo[distAndUse.second] = PipelineOpInfo();

  unsigned stagesBetweenLoads = ceil<unsigned>(numStages - 2, maxDistance + 1);

  // Then consider the load ops that feed into the dot ops or are used by other
  // loads.
  for (auto &[loadOp, distAndUse] : loadOpToDistAndUse) {
    PipelineOpInfo loadInfo;
    bool loadIsMMAV3 = false;
    if (isa<tt::DotOp>(distAndUse.second)) {
      ttg::SharedEncodingAttr sharedEnc;
      bool isLoadDotOperand = loadDotOperand(loadOp, loadIsMMAV3, sharedEnc);
      hasMMAV3 |= loadIsMMAV3;
      if (!isLoadDotOperand)
        continue;
      loadInfo.sharedEncoding = sharedEnc;
    } else {
      loadInfo.blockedEncoding = getBlockedEncoding(loadOp, axisInfoAnalysis);
    }
    // If we haven't already assigned a layout do it now.
    if (!loadInfo.sharedEncoding)
      loadInfo.sharedEncoding =
          getSharedEncoding(loadOp, distAndUse.second, loadIsMMAV3);
    loadInfo.loadIsMMAV3 = loadIsMMAV3;
    int stage = (maxDistance - distAndUse.first) * stagesBetweenLoads;
    loadInfo.stage = stage;
    loadInfo.use = distAndUse.second;
    opInfo[loadOp] = loadInfo;
  };

  // Last, find load users and assigning a stage for them.
  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  for (Operation &op : forOp) {
    auto iter = opInfo.find(&op);
    if (iter != opInfo.end() && iter->second.stage == -1) {
      assert(!isa<tt::LoadOp>(iter->first));
      PipelineOpInfo useOpInfo{
          .stage = numStages - 1,
      };
      iter->second = useOpInfo;
    }
  }

  return true;
}

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, tt::LoadOp loadOp,
                         ttg::SharedEncodingAttr sharedEnc, unsigned distance) {
  OpBuilder builder(forOp);
  auto ty = loadOp.getType().cast<RankedTensorType>();
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
static SmallVector<Value>
createAsynOps(scf::ForOp &forOp,
              llvm::MapVector<Operation *, PipelineOpInfo> &opToInfo,
              int numBuffers, bool hasMMAV3) {
  struct AsyncLoad {
    AsyncLoad(tt::LoadOp loadOp, Value alloc) : loadOp(loadOp), alloc(alloc) {}
    tt::LoadOp loadOp;
    Value alloc;
  };
  SmallVector<AsyncLoad> asyncLoads;
  SmallVector<Value> allocs;
  for (auto &[op, info] : opToInfo) {
    if (tt::LoadOp loadOp = dyn_cast<tt::LoadOp>(op)) {
      assert(info.sharedEncoding && "LoadOp shared encoding not defined.");
      Value alloc = createAlloc(forOp, loadOp, info.sharedEncoding, numBuffers);
      assert(alloc && "Failed to create alloc for the async load.");
      allocs.push_back(alloc);
      asyncLoads.emplace_back(loadOp, alloc);
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
    createAsyncLoad(forOp, asyncLoad.loadOp, asyncLoad.alloc, insertIdx,
                    extractIdx, phase, opToInfo);
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
    LDBG("\nStage " << i);
    for (auto &pair : schedule) {
      if (pair.second == i) {
        pair.first->dump();
      }
    }
  }
}

static bool
isScheduleValid(scf::ForOp forOp,
                std::vector<std::pair<Operation *, unsigned>> &schedule) {
  DenseSet<Operation *> seen;
  for (auto &pair : schedule) {
    if (!seen.insert(pair.first).second)
      return false;
  }
  auto loopBody = forOp.getBody()->without_terminator();
  auto numOps = std::distance(loopBody.begin(), loopBody.end());
  if (seen.size() != numOps)
    return false;
  return true;
}

// create the schedule for a matmul loop. This is ad hoc based on how we know
// matmul loops should be pipelined and is not a generic scheduler.
static std::vector<std::pair<Operation *, unsigned>>
createSchedule(scf::ForOp forOp, int numStages,
               llvm::MapVector<Operation *, PipelineOpInfo> &opToInfo,
               bool prefetchExtract) {
  LLVM_DEBUG({
    LDBG("For loop:");
    forOp.dump();

    LDBG("Initial schedule:");
    for (int i = 0; i < numStages; i++) {
      LDBG("\nOps in stage " << i);
      for (auto &[op, info] : opToInfo) {
        if (i == info.stage) {
          op->dump();
        }
      }
    }
  });

  SmallVector<Operation *> extractOps;
  // Find the insert/extract ops that will go respectively in stage 0 and stage
  // `numStages - 2`. All the other operations will go in stage `numStages - 1`.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (prefetchExtract) {
      if (isa<ttg::MemDescSubviewOp, ttg::AsyncWaitOp>(op))
        extractOps.push_back(&op);
    }
  }

  auto printDenseSet = [](DenseSet<Operation *> &set) {
    for (auto op : set) {
      op->dump();
    }
  };

  SmallVector<DenseSet<Operation *>> insertOps(numStages);
  for (auto &[op, info] : opToInfo) {
    if (isa<ttg::AsyncCopyGlobalToLocalOp, ttg::AsyncCommitGroupOp>(op)) {
      insertOps[info.stage].insert(op);
    }
  }

  // Inserts and dependencies grouped by stage.
  SmallVector<DenseSet<Operation *>> insertAndDeps(numStages);
  DenseSet<Operation *> seen;
  for (int stage = 0; stage < numStages; stage++) {
    for (Operation *op : insertOps[stage]) {
      tt::addDep(op, insertAndDeps[stage], false, &seen);
      seen.insert(insertAndDeps[stage].begin(), insertAndDeps[stage].end());
    }
  }

  LLVM_DEBUG({
    for (int stage = 0; stage < numStages; stage++) {
      LDBG("\ninsertAndDeps " << stage);
      printDenseSet(insertAndDeps[stage]);
    }
  });

  // Find dependencies with distance of 1.
  SmallVector<DenseSet<Operation *>> distanceOneUsers(numStages);
  for (int stage = 0; stage < numStages - 1; stage++) {
    auto &group = insertAndDeps[stage];
    for (Operation *op : group) {
      for (Value operand : op->getOperands()) {
        if (auto arg = operand.dyn_cast<BlockArgument>()) {
          if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
            auto yieldOp = op->getBlock()->getTerminator();
            Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
            Operation *defOp = v.getDefiningOp();
            if (defOp && group.count(defOp) == 0) {
              // Schedule distance 1 users in the next stage.
              distanceOneUsers[stage].insert(defOp);
            }
          }
        }
      }
    }
    LLVM_DEBUG({
      LDBG("\ndistanceOneUsers " << stage);
      printDenseSet(distanceOneUsers[stage]);
    });
  }

  // Schedule loads with a distance of 1 together with the insert ops.
  for (unsigned i = 0; i < distanceOneUsers.size(); i++) {
    for (auto op : distanceOneUsers[i]) {
      if (isa<tt::LoadOp>(op))
        tt::addDep(op, insertAndDeps[i], true);
    }
  }

  DenseSet<Operation *> allInsertAndDeps;
  for (auto &set : insertAndDeps) {
    allInsertAndDeps.insert(set.begin(), set.end());
  }

  // Rest of the distance 1 dependencies will be scheduled one
  // stage after the insert ops.
  SmallVector<DenseSet<Operation *>> stage1deps(numStages);
  for (unsigned i = 0; i < distanceOneUsers.size() - 1; i++) {
    auto &group = distanceOneUsers[i];
    for (auto op : group) {
      if (!isa<tt::LoadOp>(op))
        tt::addDep(op, stage1deps[i + 1], true, &allInsertAndDeps);
    }
    LLVM_DEBUG({
      LDBG("\nstage1deps " << i);
      printDenseSet(stage1deps[i]);
    });
  }

  DenseSet<Operation *> allStage1Deps;
  for (auto &set : stage1deps) {
    allStage1Deps.insert(set.begin(), set.end());
  }

  DenseSet<Operation *> extractAndDeps;
  for (Operation *op : extractOps) {
    tt::addDep(op, extractAndDeps, true, &allInsertAndDeps);
  }
  LLVM_DEBUG({
    LDBG("\nextractAndDeps:");
    printDenseSet(extractAndDeps);
  });

  std::vector<std::pair<Operation *, unsigned>> schedule;
  // Schedule stage `numStage - 1` first.
  tt::addOps(forOp, numStages - 1, schedule, [&](Operation *op) {
    return allInsertAndDeps.count(op) == 0 && allStage1Deps.count(op) == 0 &&
           extractAndDeps.count(op) == 0;
  });

  // Schedule some dependencies with distance of 1 into stage 1 to reduce
  // pressure.
  // Insert the ops in the reverse order of the stages. This helps with saving
  // the number of required buffers.
  for (int i = numStages - 1; i >= 0; i--) {
    auto &group = stage1deps[i];
    tt::addOps(forOp, i, schedule,
               [&](Operation *op) { return group.count(op); });
  }

  for (int i = numStages - 1; i >= 0; i--) {
    auto &group = insertAndDeps[i];
    tt::addOps(forOp, i, schedule,
               [&](Operation *op) { return group.count(op); });
  }

  // Finally schedule the extract ops in stage `numStage - 2` so that they get
  // pre-fetched and play well with pretech pass.
  tt::addOps(forOp, numStages - 2, schedule,
             [&](Operation *op) { return extractAndDeps.count(op); });

  LLVM_DEBUG(printSchedule(schedule, numStages));
  assert(isScheduleValid(forOp, schedule) && "Invalid schedule.");

  return schedule;
}

constexpr static char kNeedWaitAttrName[] = "triton.pipeline.needs_wait";

bool mlir::triton::preProcessLoopAndGetSchedule(
    scf::ForOp &forOp, int numStages, mlir::triton::PipeliningOption &options) {
  // 1. First collect "interesting" operations with a stage where to schedule
  // them. This gives a coarse scheduling for the loop.
  llvm::MapVector<Operation *, PipelineOpInfo> opToInfo;
  bool hasMMAV3 = false;
  if (!collectOpsToPipeline(forOp, opToInfo, numStages, hasMMAV3))
    return false;

  // Calculate the number of buffers needed for each load.
  // TODO pawel: we could do more fine-grained allocation here and
  // allocate only the number of buffers that specific loads need.
  // Instead, we allocate the maximum number of buffers needed by any load.
  int maxNumBuffers = -1;
  for (auto &[op, info] : opToInfo) {
    if (!isa<tt::LoadOp>(op))
      continue;
    assert(info.stage != -1 && "LoadOp stage not defined");
    assert(info.use && "LoadOp use not defined");
    assert(opToInfo.count(info.use) && "Use not in opToInfo");

    int defStage = info.stage;
    int useStage = opToInfo[info.use].stage;
    int numBuffers = useStage - defStage;

    if (hasMMAV3 && isa<tt::DotOp>(info.use)) {
      // For MMAv3, we need an extra buffer as this is assumed in the wgmma
      // pipelining post-processing.
      numBuffers++;
    }
    if (numBuffers > maxNumBuffers)
      maxNumBuffers = numBuffers;
  }

  // 2. Convert the loads into async loads and create the allocs.
  SmallVector<Value> allocs =
      createAsynOps(forOp, opToInfo, maxNumBuffers, hasMMAV3);

  // 3. Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      createSchedule(forOp, numStages, opToInfo, /*prefetchExtract=*/!hasMMAV3);

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

/// MMA V3 post-processing.
static bool selfDepend(tt::DotOp dotOp, scf::ForOp forOp,
                       Operation **firstUse) {
  std::function<bool(Value, int, scf::ForOp)> dependOn =
      [&dependOn](Value v, int argId, scf::ForOp forOp) {
        auto op = v.getDefiningOp();
        if (isa<BlockArgument>(v)) {
          auto iterArgs = forOp.getRegionIterArgs();
          auto iter = std::find(iterArgs.begin(), iterArgs.end(), v);
          if (iter != iterArgs.end())
            return std::distance(iterArgs.begin(), iter) == argId;
        } else {
          if (!op)
            return false;
          for (auto operand : op->getOperands()) {
            if (dependOn(operand, argId, forOp))
              return true;
          }
        }
        return false;
      };
  auto result = dotOp.getResult();
  auto yieldOp = forOp.getBody()->getTerminator();
  int argIdx = -1;
  auto iter = std::find(yieldOp->getOperands().begin(),
                        yieldOp->getOperands().end(), result);
  if (iter != yieldOp->getOperands().end())
    argIdx = std::distance(yieldOp->getOperands().begin(), iter);
  if (argIdx == -1)
    return false;
  for (auto operand : dotOp.getOperands()) {
    if (dependOn(operand, argIdx, forOp)) {
      auto iterArgs = forOp.getRegionIterArgs();
      *firstUse = iterArgs[argIdx].use_begin().getUser();
      return true;
    }
  }
  return false;
}

static void removeExtraWait(tt::nvidia_gpu::DotWaitOp dotWaitOp,
                            bool hasDotWait0) {
  if (hasDotWait0) {
    dotWaitOp->erase();
  }
}

void mlir::triton::asyncLaunchDots(scf::ForOp forOp) {
  Block *loop = forOp.getBody();
  auto getBlockNumInFor = [](Operation *op, scf::ForOp forOp) {
    if (!op)
      return -1l;
    auto lastOp = op;
    while (op->getBlock()->getParentOp() != forOp) {
      lastOp = op;
      op = op->getBlock()->getParentOp();
    }
    return std::distance(lastOp->getBlock()->getParent()->begin(),
                         lastOp->getBlock()->getIterator());
  };
  /// XXX(Keren): Clean up the following duplicate code with checkDotOp
  /// dots to be pipelined
  bool hasSyncDot = false;
  bool hasDotWait0 = false;
  SmallVector<tt::DotOp> allDots;
  SmallVector<tt::DotOp> dots;
  SmallVector<unsigned> resultNeedSync;
  for (Operation &op : *loop) {
    if (auto dotWaitOp = dyn_cast<tt::nvidia_gpu::DotWaitOp>(&op)) {
      auto attr = dotWaitOp->getAttrOfType<IntegerAttr>("pendings");
      auto pendingCount = attr.getInt();
      if (pendingCount == 0)
        hasDotWait0 = true;
    }
    if (auto dotOp = dyn_cast<tt::DotOp>(&op)) {
      allDots.push_back(dotOp);
    }
  }
  for (Operation &op : *loop) {
    if (auto dotOp = dyn_cast<tt::DotOp>(&op)) {
      RankedTensorType resTy = dotOp.getType();
      if (auto resEnc =
              resTy.getEncoding().dyn_cast<ttg::NvidiaMmaEncodingAttr>()) {
        if (resEnc && resEnc.isHopper()) {
          auto dot = dotOp.getResult();
          bool valid = true;

          // all users of dot should be scf.yield
          if (!dot.hasOneUse())
            valid = false;
          if (!isa<scf::YieldOp>(*dot.getUsers().begin()))
            valid = false;

          Operation *firstUse = nullptr;
          auto depend = selfDepend(dotOp, forOp, &firstUse);
          bool selfDirectDepend = (dotOp == firstUse);
          for (auto tempInAll : allDots) {
            auto iter = std::find(dots.begin(), dots.end(), tempInAll);
            if (iter != dots.end())
              continue;
            auto db = getBlockNumInFor(tempInAll, forOp);
            auto fb = getBlockNumInFor(firstUse, forOp);
            if (db < fb ||
                (db == fb && db >= 0 && tempInAll->isBeforeInBlock(firstUse)))
              hasSyncDot = true;
          }
          auto CArg = dotOp.getOperand(2);
          if (!(selfDirectDepend ||
                (depend && !selfDirectDepend && hasSyncDot)) ||
              !CArg.hasOneUse())
            valid = false;

          // Both operands needs to have been multi-buffered if they come from
          // shared memory otherwise we will have a race condition as the shared
          // memory will be overriden before the wgmma wait.
          // Note that we relay on the assumption that `createAsynOps` will have
          // allocated enough buffers to pipeline dot operation with 2 stages.
          for (Value operand : {dotOp.getOperand(0), dotOp.getOperand(1)}) {
            auto operandEncoding =
                operand.getType().cast<TensorOrMemDesc>().getEncoding();
            if (!operandEncoding.isa<ttg::SharedEncodingAttr>())
              continue;
            Value transitiveOperand = operand;
            while (isa_and_nonnull<ttg::ConvertLayoutOp, tt::TransOp>(
                transitiveOperand.getDefiningOp()))
              transitiveOperand =
                  transitiveOperand.getDefiningOp()->getOperand(0);
            if (forOp.isDefinedOutsideOfLoop(transitiveOperand))
              continue;
            if (transitiveOperand.getDefiningOp<ttg::MemDescSubviewOp>() ==
                nullptr)
              valid = false;
          }

          if (valid) {
            dots.push_back(dotOp);
            resultNeedSync.push_back(
                dotOp->getUses().begin()->getOperandNumber());
          }
        }
      }
    }
  }

  // Early stop: no need to continue if there is no valid dot in the loop.
  if (dots.empty())
    return;

  OpBuilder builder(forOp);
  // 0. insert dot_wait after the last dot in the loop as we implicitly pipeline
  // wgmma ops by one stage.
  // This is needed to prevent shared memory inputs to be overridden before the
  // operation is completed.
  // TODO: merge this with the rest of the pipelining transformation and look at
  // a better representation for async dots.
  tt::DotOp lastDot = dots.back();
  auto loc = lastDot.getLoc();
  builder.setInsertionPointAfter(lastDot);
  auto dotWait = builder.create<tt::nvidia_gpu::DotWaitOp>(
      lastDot.getLoc(), lastDot.getResult(), dots.size());

  // 1. replace Dot with DotAsync
  for (size_t idx = 0; idx < dots.size(); ++idx) {
    tt::DotOp dotOp = dots[idx];
    builder.setInsertionPoint(dotOp);
    auto dotAsync = builder.create<tt::nvidia_gpu::DotAsyncOp>(
        dotOp.getLoc(), dotOp.getA(), dotOp.getB(), dotOp.getC(),
        dotOp.getAllowTF32(), dotOp.getMaxNumImpreciseAcc());
    dotOp.replaceAllUsesWith(dotAsync.getResult());
    dotOp->erase();
  }

  hasDotWait0 = hasDotWait0 || hasSyncDot;

  // 2. If there's any outstanding DotAsyncOps, we need to wait for them.
  builder.setInsertionPointAfter(forOp);
  SmallVector<Value> waitOperands;
  for (int i = 0; i < resultNeedSync.size(); ++i) {
    Value result = forOp->getResult(resultNeedSync[i]);
    if (result.use_empty())
      continue;
    waitOperands.push_back(result);
  }
  if (!waitOperands.empty()) {
    auto dotWait = builder.create<tt::nvidia_gpu::DotWaitOp>(forOp.getLoc(),
                                                             waitOperands, 0);
    for (int i = 0; i < resultNeedSync.size(); ++i) {
      Value result = forOp->getResult(resultNeedSync[i]);
      result.replaceAllUsesExcept(dotWait.getResult(i), dotWait);
    }
  }

  // 3. potentially remove redundant dot_wait after dot_async if having multiple
  // DotOp in the loop
  removeExtraWait(dotWait, hasDotWait0);
}
