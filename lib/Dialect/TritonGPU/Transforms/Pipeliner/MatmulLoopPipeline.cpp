#include "PipelineExpander.h"
#include "PipeliningUtility.h"
#include "Schedule.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// TODO: We can extra some helpers into common utilities once we add more
// schedules.

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

static void createAsyncCopy(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                            Value insertIdx, Value extractIdx) {
  OpBuilder builder(forOp);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  auto insertOp = builder.create<ttg::InsertSliceAsyncOp>(
      loc, alloc.getType(), loadOp.getPtr(), alloc, insertIdx, loadOp.getMask(),
      loadOp.getOther(), loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile(), /*axis*/ 0);
  auto commmit = builder.create<ttg::AsyncCommitGroupOp>(loc);

  // Extract part.
  auto allocType = alloc.getType().cast<RankedTensorType>();
  RankedTensorType sliceType = RankedTensorType::get(
      {allocType.getShape()[1], allocType.getShape()[2]},
      allocType.getElementType(), allocType.getEncoding());
  auto extract = builder.create<ttg::ExtractSliceOp>(
      loc, sliceType, insertOp.getResult(),
      SmallVector<OpFoldResult>{extractIdx, int_attr(0), int_attr(0)},
      SmallVector<OpFoldResult>{int_attr(1), int_attr(sliceType.getShape()[0]),
                                int_attr(sliceType.getShape()[1])},
      SmallVector<OpFoldResult>{int_attr(1), int_attr(1), int_attr(1)});
  auto newCvt = builder.create<ttg::ConvertLayoutOp>(
      loadOp->getLoc(), loadOp.getType(), extract.getResult());
  loadOp->replaceAllUsesWith(newCvt->getResults());
  loadOp.erase();

  // Fix up the yield op.
  appendToYield(forOp, {insertOp});
}

static void createTMALoad(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                          Value insertIdx, Value extractIdx, Value phase) {
  OpBuilder builder(forOp);
  Location loc = loadOp.getLoc();
  auto CTALayout = ttg::CTALayoutAttr::get(loadOp.getContext(),
                                           /*CTAsPerCGA*/ {1},
                                           /*CTASplitNum*/ {1},
                                           /*CTAOrder*/ {0});
  auto sharedEncoding = ttg::SharedEncodingAttr::get(loadOp.getContext(), 1, 1,
                                                     1, {0}, CTALayout, false);
  int64_t numBuffers = alloc.getType().cast<RankedTensorType>().getShape()[0];
  auto mBarriersTy = RankedTensorType::get(
      {numBuffers}, builder.getIntegerType(64), sharedEncoding);
  // Allocate an array of mbarrier objects outside the loop.
  Value barrierArray =
      builder.create<ttng::AllocMBarrierOp>(loc, mBarriersTy, 1);
  // extract the barrier and emit arriver/copy/wait/extract code sequence.
  builder.setInsertionPoint(loadOp);
  auto mBarTy = tt::PointerType::get(builder.getIntegerType(64), 3);
  Value barrier = builder.create<ttng::ExtractMBarrierOp>(
      loc, mBarTy, barrierArray, insertIdx);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value threadId = builder.create<ttng::GetThreadIdOp>(loc);
  Value pred = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             threadId, zero);

  auto loadTy = loadOp.getType().dyn_cast<RankedTensorType>();
  auto loadShape = loadTy.getShape();
  auto CTASplitNum = ttg::getCTASplitNum(loadTy.getEncoding());
  auto shapePerSlice = ttg::getShapePerCTA(CTASplitNum, loadShape);
  auto elemTy = loadTy.getElementType();
  unsigned elems = std::accumulate(shapePerSlice.begin(), shapePerSlice.end(),
                                   1, std::multiplies{});
  elems *= (elemTy.getIntOrFloatBitWidth() / 8);
  builder.create<ttng::MBarrierArriveOp>(loc, barrier, pred,
                                         /*remoteCtaId*/ nullptr,
                                         /*trackAsyncOp*/ false, elems);
  auto allocType = alloc.getType().cast<RankedTensorType>();
  auto insertOp = builder.create<ttng::InsertSliceTMAOp>(
      loc, allocType, loadOp.getPtr(), alloc,
      /*index*/ insertIdx, barrier, loadOp.getMask(), loadOp.getOther(),
      loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile(),
      /*axis*/ 0);

  RankedTensorType sliceType = RankedTensorType::get(
      {allocType.getShape()[1], allocType.getShape()[2]},
      allocType.getElementType(), allocType.getEncoding());
  auto extract = builder.create<mlir::triton::gpu::ExtractSliceOp>(
      loc, sliceType, insertOp.getResult(),
      SmallVector<OpFoldResult>{extractIdx, int_attr(0), int_attr(0)},
      SmallVector<OpFoldResult>{int_attr(1), int_attr(sliceType.getShape()[0]),
                                int_attr(sliceType.getShape()[1])},
      SmallVector<OpFoldResult>{int_attr(1), int_attr(1), int_attr(1)});

  Value barrierWait = builder.create<ttng::ExtractMBarrierOp>(
      loc, mBarTy, barrierArray, extractIdx);
  builder.create<ttng::MBarrierWaitOp>(loc, barrierWait, phase);

  auto newCvt = builder.create<ttg::ConvertLayoutOp>(
      loadOp->getLoc(), loadOp.getType(), extract.getResult());
  loadOp->replaceAllUsesWith(newCvt->getResults());
  loadOp.erase();

  // Fix up the yield op.
  appendToYield(forOp, {insertOp});
}

/// Create an async load equivalent to the given load.
static void createAsyncLoad(scf::ForOp &forOp, tt::LoadOp loadOp, Value alloc,
                            Value insertIdx, Value extractIdx, Value phase) {
  if (isLoadFromTensorPtr(loadOp)) {
    createTMALoad(forOp, loadOp, alloc, insertIdx, extractIdx, phase);
  } else {
    createAsyncCopy(forOp, loadOp, alloc, insertIdx, extractIdx);
  }
}

namespace {
struct LoadDotOperand {
  LoadDotOperand(tt::LoadOp load,
                 ttg::DotOperandEncodingAttr dotOperandEncoding,
                 bool needTrans = false)
      : load(load), dotOperandEncoding(dotOperandEncoding),
        needTrans(needTrans) {}
  tt::LoadOp load;
  ttg::DotOperandEncodingAttr dotOperandEncoding;
  bool needTrans;
};
} // namespace

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return the encoding. Otherwise return nullptr.
// Negate `needTrans` when a TransOp is seen on the transitive use chain.
static ttg::DotOperandEncodingAttr
allTransitiveUsesHaveDotEncoding(Value val, bool &needTrans) {
  ttg::DotOperandEncodingAttr attr;
  for (Operation *user : val.getUsers()) {
    if (user->getNumResults() != 1)
      return nullptr;
    auto tensorType = user->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!tensorType)
      return nullptr;
    if (isa<triton::TransOp>(user))
      needTrans = !needTrans;
    ttg::DotOperandEncodingAttr tempAttr;
    if (tensorType.getEncoding().isa<ttg::SharedEncodingAttr>()) {
      tempAttr =
          allTransitiveUsesHaveDotEncoding(user->getResult(0), needTrans);
    } else {
      auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(user);
      if (!convertLayout)
        return nullptr;
      auto tensorType =
          convertLayout.getResult().getType().dyn_cast<RankedTensorType>();
      if (!tensorType)
        return nullptr;
      tempAttr =
          tensorType.getEncoding().dyn_cast<ttg::DotOperandEncodingAttr>();
    }
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return nullptr;
    attr = tempAttr;
  }
  return attr;
}

// Return the transitive use of the load which is a dot operand.
static std::optional<LoadDotOperand> loadDotOperand(tt::LoadOp loadOp,
                                                    bool &hasMMAV3) {
  bool isCandidate = false;
  if (loadOp.getResult().hasOneUse()) {
    Operation *use = *loadOp.getResult().getUsers().begin();
    if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(use)) {
      auto tensorType =
          convertLayout.getResult().getType().cast<RankedTensorType>();
      if (auto sharedEnc =
              tensorType.getEncoding().dyn_cast<ttg::SharedEncodingAttr>()) {
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
            return LoadDotOperand(loadOp, nullptr);
          }
        }
      }
    }
  }
  bool needTrans = false;
  ttg::DotOperandEncodingAttr attr =
      allTransitiveUsesHaveDotEncoding(loadOp.getResult(), needTrans);
  if (!attr)
    return std::nullopt;
  return LoadDotOperand(loadOp, attr, needTrans);
}

/// Collect loads to pipeline. Return success if we can pipeline this loop
static void collectOpsToPipeline(scf::ForOp forOp,
                                 SmallVectorImpl<LoadDotOperand> &ops,
                                 bool &hasMMAV3) {
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // We cannot use forOp.walk(...) here because we only want to visit the
  // operations in the loop body block. Nested blocks are handled separately.
  for (Operation &op : forOp) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(&op)) {
      bool candidate = false;
      if (isLoadFromTensorPtr(loadOp)) {
        // Map to TMA load.
        candidate = true;
      } else {
        auto ptr = loadOp.getPtr();
        unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
        if (auto mask = loadOp.getMask())
          vec =
              std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

        auto tensorTy = ptr.getType().dyn_cast<RankedTensorType>();
        if (!tensorTy || tensorTy.getRank() < 2)
          continue;
        auto ty =
            tensorTy.getElementType().cast<tt::PointerType>().getPointeeType();
        unsigned width = vec * ty.getIntOrFloatBitWidth();
        // We do not pipeline all loads for the following reasons:
        // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8 and 16.
        // 2. It's likely that pipling small loads won't offer much performance
        //    improvement and may even hurt performance by increasing register
        //    pressure.
        if (width >= 32)
          candidate = true;
      }
      if (!candidate)
        continue;
      std::optional<LoadDotOperand> loadWithDotOperand =
          loadDotOperand(loadOp, hasMMAV3);
      if (!loadWithDotOperand.has_value())
        continue;
      ops.push_back(loadWithDotOperand.value());
    }
  }
}

// Create an allocation that can old distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, tt::LoadOp loadOp,
                         ttg::DotOperandEncodingAttr dotOpEnc,
                         unsigned distance, bool needTrans) {
  OpBuilder builder(forOp);
  auto ty = loadOp.getType().cast<RankedTensorType>();
  Attribute sharedEnc;
  auto CTALayout = ttg::getCTALayout(ty.getEncoding());
  if (dotOpEnc) {
    unsigned bitWidth = ty.getElementType().getIntOrFloatBitWidth();
    // set needTrans to avoid unnecessary conversion between shared encodings.
    sharedEnc = ttg::SharedEncodingAttr::get(
        ty.getContext(), dotOpEnc, ty.getShape(),
        ttg::getOrder(ty.getEncoding()), CTALayout, bitWidth, needTrans);
  } else {
    // MMAv3
    sharedEnc = ttg::SharedEncodingAttr::get(ty.getContext(), ty.getShape(),
                                             ttg::getOrder(ty.getEncoding()),
                                             CTALayout, ty.getElementType());
  }
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  Type allocType =
      RankedTensorType::get(bufferShape, ty.getElementType(), sharedEnc);
  Value alloc = builder.create<mlir::triton::gpu::AllocTensorOp>(
      loadOp.getLoc(), allocType);
  return alloc;
}

// Convert load ops into their asyn version and apply multi-buffering based on
// the number of stages.
static SmallVector<Value> createAsynOps(scf::ForOp &forOp,
                                        ArrayRef<LoadDotOperand> loads,
                                        int numStages, bool hasMMAV3) {
  struct AsyncLoad {
    AsyncLoad(tt::LoadOp loadOp, Value alloc) : loadOp(loadOp), alloc(alloc) {}
    tt::LoadOp loadOp;
    Value alloc;
  };
  int numBuffers = numStages - 1;
  // For MMAv3 we need an extra buffer as this is assumed in the wgmma
  // pipelining post-processing.
  // TODO: Improve modeling of wgmma pipelining.
  if (hasMMAV3)
    numBuffers++;
  SmallVector<AsyncLoad> asyncLoads;
  SmallVector<Value> allocs;
  SmallVector<Value> newOperands;
  bool needsMbarrierPhase = false;
  bool needsAsyncWait = false;
  for (const LoadDotOperand &loadOperand : loads) {
    tt::LoadOp loadOp = loadOperand.load;
    Value alloc = createAlloc(forOp, loadOp, loadOperand.dotOperandEncoding,
                              numBuffers, loadOperand.needTrans);
    assert(alloc && "Failed to create alloc for the async load.");
    newOperands.push_back(alloc);
    allocs.push_back(alloc);
    asyncLoads.emplace_back(loadOp, alloc);
    if (isLoadFromTensorPtr(loadOp))
      needsMbarrierPhase = true;
    else
      needsAsyncWait = true;
  }

  OpBuilder builder(forOp);
  Location loc = forOp.getLoc();
  // Create two new counters to index into the allocs.
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value insertIdx = minusOne;
  Value extractIdx = minusOne;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
  newOperands.push_back(insertIdx);
  newOperands.push_back(extractIdx);
  Value phase;
  if (needsMbarrierPhase) {
    phase = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    newOperands.push_back(phase);
  }
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;
  for (int i = 0; i < asyncLoads.size(); i++) {
    asyncLoads[i].alloc = newForOp.getBody()->getArgument(newOperandIndex + i);
  }
  insertIdx =
      newForOp.getBody()->getArgument(newOperandIndex + asyncLoads.size());
  extractIdx =
      newForOp.getBody()->getArgument(newOperandIndex + asyncLoads.size() + 1);

  // Create two counters for the insert and extract indices to avoid creating
  // long liverange.
  builder.setInsertionPoint(asyncLoads.front().loadOp);
  insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
  Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               insertIdx, numBuffersVal);
  insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);

  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  if (needsMbarrierPhase) {
    phase = newForOp.getBody()->getArgument(newOperandIndex +
                                            asyncLoads.size() + 2);
    Value oneI1 = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    Value nextPhase = builder.create<arith::XOrIOp>(loc, phase, oneI1);
    phase = builder.create<arith::SelectOp>(loc, cndExt, phase, nextPhase);
  }

  bool firstLoad = true;
  for (AsyncLoad &asyncLoad : asyncLoads) {
    createAsyncLoad(forOp, asyncLoad.loadOp, asyncLoad.alloc, insertIdx,
                    extractIdx, phase);
    firstLoad = false;
  }
  SmallVector<Value> newYieldOperands = {insertIdx, extractIdx};
  if (needsMbarrierPhase)
    newYieldOperands.push_back(phase);
  // Patch the yield with the updated counters.
  appendToYield(forOp, newYieldOperands);

  return allocs;
}

// create the schedule for a matmul loop. This is ad hoc based on how we know
// matmul loops should be pipelined and is not a generic scheduler.
static std::vector<std::pair<Operation *, unsigned>>
createSchedule(scf::ForOp forOp, int numStages, bool prefetchExtract) {
  SmallVector<Operation *> insertOps;
  SmallVector<Operation *> extractOps;
  // Find the insert/extract ops that will go respectively in stage 0 and stage
  // `numStages - 2`. All the other operations will go in stage `numStages - 1`.
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::InsertSliceAsyncOp, ttg::AsyncCommitGroupOp,
            ttng::MBarrierArriveOp, ttng::InsertSliceTMAOp>(op))
      insertOps.emplace_back(&op);
    if (prefetchExtract) {
      if (isa<ttg::ExtractSliceOp, ttg::AsyncWaitOp>(op))
        extractOps.emplace_back(&op);
    }
  }
  DenseSet<Operation *> insertAndDeps;
  for (Operation *op : insertOps) {
    tt::addDep(op, insertAndDeps, false);
  }

  // Find dependencies with distance of 1.
  SmallVector<Operation *> distanceOneUsers;
  for (Operation *op : insertAndDeps) {
    for (Value operand : op->getOperands()) {
      if (auto arg = operand.dyn_cast<BlockArgument>()) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
          auto yieldOp = op->getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp && insertAndDeps.count(defOp) == 0) {
            distanceOneUsers.push_back(defOp);
          }
        }
      }
    }
  }
  // Schedule loads with a distance of 1 in stage 0
  for (Operation *op : distanceOneUsers) {
    if (isa<tt::LoadOp>(op)) {
      tt::addDep(op, insertAndDeps, true);
    }
  }
  // For the rest of the ops we can move then into stage 1 so that they can be
  // closer to their uses.
  DenseSet<Operation *> stage1deps;
  for (Operation *op : distanceOneUsers) {
    if (!isa<tt::LoadOp>(op)) {
      tt::addDep(op, stage1deps, true, &insertAndDeps);
    }
  }

  DenseSet<Operation *> extractAndDeps;
  for (Operation *op : extractOps) {
    tt::addDep(op, extractAndDeps, true, &insertAndDeps);
  }
  std::vector<std::pair<Operation *, unsigned>> schedule;
  // Schedule stage `numStage - 1` first.
  tt::addOps(forOp, numStages - 1, schedule, [&](Operation *op) {
    return insertAndDeps.count(op) == 0 && stage1deps.count(op) == 0 &&
           extractAndDeps.count(op) == 0;
  });

  // Schedule some dependencies with distance of 1 into stage 1 to reduce
  // pressure.
  tt::addOps(forOp, 1, schedule,
             [&](Operation *op) { return stage1deps.count(op); });

  // Then Schedule stage 0.
  tt::addOps(forOp, 0, schedule,
             [&](Operation *op) { return insertAndDeps.count(op); });

  // Finally schedule the extract ops in stage `numStage - 2` so that they get
  // pre-fetched and play well with pretech pass.
  tt::addOps(forOp, numStages - 2, schedule,
             [&](Operation *op) { return extractAndDeps.count(op); });
  return schedule;
}

constexpr static char kNeedWaitAttrName[] = "triton.pipeline.needs_wait";

bool mlir::triton::preProcessLoopAndGetSchedule(
    scf::ForOp &forOp, int numStages, mlir::triton::PipeliningOption &options) {
  // 1. First collect "interesting" operations with a stage where to schedule
  // them. This gives a coarse scheduling for the loop.
  SmallVector<LoadDotOperand> loads;
  bool hasMMAV3 = false;
  collectOpsToPipeline(forOp, loads, hasMMAV3);
  if (loads.empty())
    return false;
  bool hasAsynCp = llvm::any_of(loads, [](LoadDotOperand &load) {
    return !isLoadFromTensorPtr(load.load);
  });
  // 2. Convert the loads into async loads and create the allocs.
  SmallVector<Value> allocs = createAsynOps(forOp, loads, numStages, hasMMAV3);

  // 3. Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      createSchedule(forOp, numStages, /*prefetchExtract=*/!hasMMAV3);

  // 4. Fill out the pipeline options.
  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = tt::predicateOp;
  options.supportDynamicLoops = true;
  unsigned numLoadsInStage = (numStages - 2) * loads.size();
  options.annotateFn = [](Operation *op,
                          mlir::triton::PipeliningOption::PipelinerPart part,
                          unsigned iteration) {
    if (isa<ttg::ExtractSliceOp>(op)) {
      op->setAttr(kNeedWaitAttrName, UnitAttr::get(op->getContext()));
    }
  };

  if (hasAsynCp) {
    // Insert a wait 0 after the loop
    OpBuilder builder(forOp);
    builder.setInsertionPointAfter(forOp);
    builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), 0);
    // Explicitly deallocate allocated tensors after the wait op
    for (auto alloc : allocs)
      builder.create<ttg::DeallocTensorOp>(forOp.getLoc(), alloc);
  }
  return true;
}

/// Find the minimum number of async_commit_group ops between the extract
/// and the insert. Wait number is the number of commits-1.
static std::optional<int>
minWaitNumberForExtract(ttg::ExtractSliceOp extractOp) {
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
  // If the wait is not needed (when insert is a TMA insert), return
  // std::nullopt.
  std::function<std::optional<int>(Value, Operation *, int)> minOverHistories =
      [&](Value val, Operation *sinkOp,
          int thisHistorySum) -> std::optional<int> {
    if (Operation *defOp = val.getDefiningOp()) {
      if (isa<ttg::InsertSliceAsyncOp>(defOp)) {
        thisHistorySum += countCommitsBetween(defOp->getNextNode(), sinkOp);
        minCommitNumber = std::min(minCommitNumber, thisHistorySum);
        return minCommitNumber;
      }
      if (isa<ttng::InsertSliceTMAOp>(defOp)) {
        // We don't need to wait for TMA inserts.
        return std::nullopt;
      }
      // Failed to track, return 1 conservatively.
      return 1;
    }
    if (auto arg = val.dyn_cast<BlockArgument>()) {
      Block *block = arg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

      // Failed to track, return 1 conservatively.
      if (!forOp)
        return 1;

      Operation *firstForInst = &*forOp.getBody()->begin();
      int insertsBetween = countCommitsBetween(firstForInst, sinkOp);
      thisHistorySum += insertsBetween;
      if (thisHistorySum >= minCommitNumber)
        return minCommitNumber;

      // get the value value assigned to the argument coming from outside the
      // loop
      Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
      std::optional<int> min1 =
          minOverHistories(incomingVal, forOp, thisHistorySum);
      if (!min1.has_value())
        return std::nullopt;

      // get the value value assigned to the argument coming from the previous
      // iteration
      Operation *yieldOp = block->getTerminator();
      Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
      std::optional<int> min2 =
          minOverHistories(prevVal, yieldOp, thisHistorySum);
      if (!min1.has_value())
        return std::nullopt;
      return std::min(std::min(min1, min2).value(), minCommitNumber);
    }
    // Failed to track, return 1 conservatively.
    return 1;
  };

  std::optional<int> minCommits =
      minOverHistories(extractOp.getOperand(0), extractOp, 0);
  if (minCommits == std::nullopt)
    return std::nullopt;
  if (minCommits == 0)
    llvm::report_fatal_error("No commits between insert and extract!");
  return minCommits.value() - 1;
}

/// Insert wait ops after the extract_slice ops.
void mlir::triton::insertWaits(ModuleOp module) {
  module.walk([&](ttg::ExtractSliceOp firstExtractOp) {
    if (!firstExtractOp->hasAttr(kNeedWaitAttrName))
      return;

    Operation *extractOp = firstExtractOp;
    ttg::ExtractSliceOp lastExtractOp = firstExtractOp;

    // If there is no meaningful work between the extracts, don't insert
    // multiple waits. Insert just one wait per group of extracts.
    std::optional<int> minWaitNumber = std::nullopt;
    while (extractOp) {
      lastExtractOp = cast<ttg::ExtractSliceOp>(extractOp);
      std::optional<int> currMin = minWaitNumberForExtract(lastExtractOp);
      if (currMin.has_value())
        minWaitNumber =
            std::min(minWaitNumber.value_or(INT_MAX), currMin.value());

      extractOp->removeAttr(kNeedWaitAttrName);
      extractOp = dyn_cast<ttg::ExtractSliceOp>(extractOp->getNextNode());
    }

    if (!minWaitNumber.has_value())
      return; // Wait is not needed.
    OpBuilder builder(lastExtractOp);
    builder.setInsertionPointAfter(lastExtractOp);
    builder.create<ttg::AsyncWaitOp>(lastExtractOp.getLoc(),
                                     minWaitNumber.value());
  });
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
      auto resTy = dotOp.getResult().getType().dyn_cast<RankedTensorType>();
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
