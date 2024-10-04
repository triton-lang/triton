#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create stream operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

#define DEBUG_TYPE "tritonamdgpu-stream-pipeline-v2"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static Operation *streamPredication(RewriterBase &rewriter, Operation *op,
                                    Value pred) {
  // The epilogue peeling generates a select for the stage output. This causes
  // too much register pressure with the loop result and the epilogue-dot in
  // regs for the select. Conditionally executing the dot will allow the backend
  // to optimize the select away as redundant.
  if (auto dotOp = dyn_cast<tt::DotOp>(op)) {
    auto loc = dotOp->getLoc();
    auto ifOp = rewriter.create<scf::IfOp>(loc, dotOp.getResult().getType(),
                                           pred, /*withElseRegion=*/true);
    auto thenB = ifOp.getThenBodyBuilder();
    auto yield = thenB.create<scf::YieldOp>(loc, dotOp.getResult());
    dotOp->moveBefore(yield);
    ifOp.getElseBodyBuilder().create<scf::YieldOp>(loc, dotOp.getC());
    return ifOp;
  } else if (isa<gpu::BarrierOp>(op)) {
    return op;
  }
  return tt::predicateOp(rewriter, op, pred);
}

namespace {

// Encapsulate stream pipelining
// For each `scf.for` create a StreamPipeliner manager.
class StreamPipeliner {
public:
  StreamPipeliner(scf::ForOp _forOp, int _numStages)
      : forOp(_forOp), schedule(_numStages), numStages(_numStages),
        axisInfoAnalysis(forOp->getParentOfType<ModuleOp>()) {
    options.supportDynamicLoops = true;
    options.peelEpilogue = true;
    options.predicateFn = streamPredication;
  }

  void computeLoadOpsToIndirectionLevelAndUse();
  void assignMemoryLayouts();
  void scheduleLoads(DenseSet<Operation *> &rootUsers);
  void scheduleDependencies();
  void scheduleDistanceOneDependencies();
  void scheduleRemainingToLastStage(tt::CoarseSchedule::Cluster afterPrologue);

  bool preprocessLoopAndBuildSchedule();
  bool pipelineLoop();

  Value createAlloc(Operation *loadOp, ttg::SharedEncodingAttr sharedEnc,
                    unsigned numBuffers);
  void createStreamCopy(tt::LoadOp loadOp, Value alloc, Value extractIdx,
                        tt::CoarseSchedule::Cluster prefetchCluster);
  void createStreamOps();

private:
  scf::ForOp forOp;
  tt::CoarseSchedule schedule;
  int numStages;

  // Mapping and indirection level for each `tt.load` to its use.
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
      loadOpToIndLevelAndUse;

  struct LoadInfo {
    // Shared layout is used for loads feeding into dot ops.
    ttg::SharedEncodingAttr sharedEncoding = nullptr;
    // The distance of this load's stage to its use' stage.
    int distToUse = 0;
    bool usedByDot = false;
  };

  // Mapping for each pipelined load to scheduling details.
  llvm::MapVector<Operation *, LoadInfo> loadToInfo;

  // Lookup alignment/contiguity mappings for the current module.
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis;

  // Capture list of new shared memory buffers.
  SmallVector<Value> sharedMemAllocs;

  // Pipelining options for the PipelineExpander
  tt::PipeliningOption options;
};

} // namespace

void StreamPipeliner::createStreamCopy(
    tt::LoadOp loadOp, Value alloc, Value extractIdx,
    tt::CoarseSchedule::Cluster prefetchCluster) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();

  tt::MemDescType allocTy = cast<tt::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  Operation *copy = builder.clone(*loadOp);

  auto [stage, cluster] = schedule[loadOp];
  schedule.erase(loadOp);
  schedule.insert(copy, stage, cluster);

  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  auto subviewTy = tt::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  auto storeOp =
      builder.create<ttg::LocalStoreOp>(loc, copy->getResult(0), viewLoad);
  // Clean up old local caches.
  SmallVector<ttg::LocalAllocOp> allocsToErase;
  for (Operation *user : loadOp->getUsers()) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      alloc.replaceAllUsesWith(viewLoad.getResult());
      allocsToErase.push_back(alloc);
    }
  }
  for (auto alloc : allocsToErase)
    alloc.erase();

  auto sharedLoad =
      builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad);
  auto result = sharedLoad->getResults();

  // Create a select for non-zero other values.
  Value other = loadOp.getOther();
  if (other && !isZeroConst(other)) {
    auto select = builder.create<arith::SelectOp>(
        loc, loadOp.getType(), mask, sharedLoad.getResult(), other);
    result = select->getResults();
  }

  loadOp->replaceAllUsesWith(result);

  // Prefetch load ahead of the dot stage if is used by the dot.
  if (loadToInfo[loadOp].usedByDot) {
    assert(numStages >= 2 && "requires num_stages=2 at least");
    schedule.insert(storeOp, numStages - 2, prefetchCluster);
    schedule.insert(viewLoad, numStages - 2, prefetchCluster);
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
      tempAttr = cast<ttg::SharedEncodingAttr>(memDesc.getEncoding());
      if (!getSharedEncIfAllUsersAreDotEnc(user->getResult(0)).has_value())
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
      SmallVector<unsigned> sharedOrder;
      int rank = order.size();
      // TODO rework this when shared -> dotOp conversions support arbitrary
      // shared memory ordering
      if (rank == 3) {
        // Move the batch dimension (dim #0) to be the last so that it will be
        // the slowest varying dimension.
        for (unsigned i = 0; i < rank; ++i)
          if (order[i] != 0)
            sharedOrder.emplace_back(order[i]);
        sharedOrder.emplace_back(0);
      } else {
        sharedOrder = order;
      }
      tempAttr = ttg::SharedEncodingAttr::get(
          val.getContext(), dotOpEnc, srcTy.getShape(), sharedOrder, CTALayout,
          bitWidth, /*needTrans=*/false);
    }
    // Check that the shared encodings needed by the users are compatible.
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return std::nullopt;
    attr = tempAttr;
  }
  return attr;
}

// Create a map from load ops to their indirection levels and the final uses
// of the load op (another load op, or a dot op).
//
// Indirection level is "0" for the load op directly used by the dot op,
// "1" for the load op used by the load op used by the dot op, and so on.
void StreamPipeliner::computeLoadOpsToIndirectionLevelAndUse() {
  DenseSet<Operation *> seen;

  // Recursively visit the given op and its operands to discover all load ops
  // and collect their indirection levels and uses.
  std::function<void(Operation *, int, Operation *)> dfs =
      [&](Operation *op, int distance, Operation *use) {
        // Skip previously visited load ops.
        if (!seen.insert(op).second)
          return;

        if (isa<tt::LoadOp>(op)) {
          // TODO: What if there are multiple uses at different distances?
          loadOpToIndLevelAndUse.emplace_back(op, distance, use);
          use = op;
          ++distance;
        }
        for (Value operand : op->getOperands()) {
          Operation *defOp = operand.getDefiningOp();
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
      if (!isa<tt::LoadOp>(op))
        dfs(&op, 0, &op);
    }
  }
}

// Goes through all load ops to identify those that can be pipelined and assign
// layout to them.
void StreamPipeliner::assignMemoryLayouts() {
  for (auto &[op, dist, use] : loadOpToIndLevelAndUse) {
    if (loadToInfo.count(op))
      // TODO: We'd need to verify that the distance is the same.
      continue;

    LoadInfo loadInfo;
    auto loadOp = cast<tt::LoadOp>(op);
    assert(!isLoadFromTensorPtr(loadOp) &&
           "Block ptr should have been lowered before this pass.");
    auto ptr = loadOp.getPtr();
    unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
    if (auto mask = loadOp.getMask())
      vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

    auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
    if (!tensorTy) {
      LDBG("Skip non-tensor load " << *loadOp);
      continue;
    }

    auto pointeeTy =
        cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
    unsigned width = vec * pointeeTy.getIntOrFloatBitWidth();


    if (use->hasTrait<OpTrait::DotLike>()) {
      // Only use shared memory when feeding into a dot op.
      loadInfo.usedByDot = true;
      // Limit shared memory sharing to width >= 32 elements.
      LDBG("Load " << *loadOp << " has width " << width);
      if (width >= 32) {
        loadInfo.sharedEncoding =
            getSharedEncIfAllUsersAreDotEnc(op->getResult(0)).value_or(nullptr);
      }
    } else if (auto useOp = dyn_cast<tt::LoadOp>(use)) {
      // The use of this loadOp is another loadOp. If the use is not in the
      // loadToInfo already, it means that the use is not valid for pipelining
      // for some reason. We should skip this loadOp, too.
      //
      // Note that we have an assumption that the use of this loadOp has already
      // be processed in a previous loop iteration. This assumption is held by
      // how loadOpsToIndirectionLevelAndUse recursively collects
      // loadOpToIndLevelAndUse using DFS.
      if (loadToInfo.count(useOp) == 0) {
        continue;
      }
    }

    loadToInfo[op] = loadInfo;
  }
}

void StreamPipeliner::scheduleLoads(DenseSet<Operation *> &rootUsers) {
  // Get all loads that are (transitively) used by dot ops and their distance
  // to the dot op.
  computeLoadOpsToIndirectionLevelAndUse();
  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
    for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
      LDBG("  - load: " << *l);
      LDBG("    at indirection level: " << i);
      LDBG("    used by op: " << *u);
    }
  });
  if (loadOpToIndLevelAndUse.empty())
    return;

  // Check which loads are good for pipelining, and assign them memory layouts.
  assignMemoryLayouts();
  if (loadToInfo.empty())
    return;

  // Filter out load ops that cannot be pipelined.
  int resize = 0;
  for (int i = 0, e = loadOpToIndLevelAndUse.size(); i < e; ++i) {
    auto [loadOp, distance, use] = loadOpToIndLevelAndUse[i];
    if (loadToInfo.count(loadOp) != 0)
      loadOpToIndLevelAndUse[resize++] = loadOpToIndLevelAndUse[i];
  }
  loadOpToIndLevelAndUse.resize(resize);

  // Calculate the stage distance between applicable loads.
  int maxIndirectionLevel = -1;
  for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse)
    maxIndirectionLevel = std::max(maxIndirectionLevel, dist);

  // The stage gap between chained loads--this allows us to "spread" loads
  // with a non-one step in case the number of stages given by the user is
  // large.
  assert(numStages >= 2 && "requires num_stages=2 at least");
  unsigned stagesBetweenLoads =
      llvm::divideCeil(numStages - 2, maxIndirectionLevel + 1);
  LDBG("stagesBetweenLoads = " << stagesBetweenLoads);

  // Put the root uses of the loads in the last stage.
  tt::CoarseSchedule::Cluster rootUsersCluster = schedule.clusters.newAtFront();
  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    // Non-LoadOp(s) are the (final) root uses of all LoadOp(s).
    if (!isa<tt::LoadOp>(use)) {
      schedule.insert(use, numStages - 1, rootUsersCluster);
      rootUsers.insert(use);
    }
  }

  // Create a cluster for load ops at each indirection level.
  SmallVector<tt::CoarseSchedule::Cluster> loadsClusters;
  for (int i = 0; i <= maxIndirectionLevel; i++) {
    loadsClusters.push_back(schedule.clusters.newAtBack());
  }
  // Assign stages to the loads.
  for (auto [loadOp, indLevel, _] : loadOpToIndLevelAndUse) {
    int stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads;
    schedule.insert(loadOp, stage, loadsClusters[indLevel]);
  }

  // Calculate distance from the load to the use.
  for (auto [loadOp, _, use] : loadOpToIndLevelAndUse) {
    loadToInfo[loadOp].distToUse = schedule[use].first - schedule[loadOp].first;
  }

  LLVM_DEBUG({
    LDBG("Chosen loads to pipeline:");
    for (const auto &[load, info] : loadToInfo) {
      LDBG("  - load: " << *load);
      LDBG("    distToUse: " << info.distToUse);
      LDBG("    usedByDot: " << info.usedByDot);
    }
  });
}

// Add dependencies of anchor ops to the coarse schedule. Schedule them to
// the same stage and ordering cluster as the anchor op.
void StreamPipeliner::scheduleDependencies() {
  SmallVector<std::tuple<Operation *, int, tt::CoarseSchedule::Cluster>>
      opsInOrder = schedule.getOpsInOrder(forOp);
  // Schedule dependencies stage by stage.
  for (int stage = 0; stage < numStages; ++stage) {
    for (auto [op, stage_, cluster] : opsInOrder) {
      if (stage_ != stage)
        continue;
      schedule.insertDepsOfOp(op, stage, cluster, false);
    }
  }
}

// Find dependencies with distance of 1. They will go to the next stage,
// but in the cluster before the current op.
void StreamPipeliner::scheduleDistanceOneDependencies() {
  auto getNestedOperands = [](Operation *op) {
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
      auto arg = dyn_cast<BlockArgument>(operand);
      if (!arg || arg.getArgNumber() == 0 || arg.getOwner() != op.getBlock())
        continue;
      auto yieldOp = op.getBlock()->getTerminator();
      Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
      Operation *defOp = v.getDefiningOp();
      if (!defOp || schedule.count(defOp) != 0)
        continue;
      if (isa<tt::LoadOp>(defOp)) {
        // Exception: schedule loads with a distance of 1 together with the
        // current op.
        schedule.insertIfAbsent(defOp, stage, cluster);
        schedule.insertDepsOfOp(defOp, stage, cluster, true);
      } else {
        if (dist1Cluster.count(&cluster) == 0) {
          dist1Cluster[&cluster] = schedule.clusters.newBefore(cluster);
        }
        schedule.insertIfAbsent(defOp, stage + 1, dist1Cluster[&cluster]);
        schedule.insertDepsOfOp(defOp, stage + 1, dist1Cluster[&cluster], true);
      }
    }
  }
}

void StreamPipeliner::scheduleRemainingToLastStage(
    tt::CoarseSchedule::Cluster afterPrologue) {
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
Value StreamPipeliner::createAlloc(Operation *loadOp,
                                   ttg::SharedEncodingAttr sharedEnc,
                                   unsigned numBuffers) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), numBuffers);
  Type memdescType = tt::MemDescType::get(bufferShape, ty.getElementType(),
                                          sharedEnc, sharedMemorySpace,
                                          /*mutableMemory=*/true);
  return builder.create<ttg::LocalAllocOp>(loadOp->getLoc(), memdescType,
                                           Value());
}

// Convert load ops into shared memory allocation loads and apply
// multi-buffering based on the required number of buffers.
void StreamPipeliner::createStreamOps() {
  // Calculate the number of buffers needed for each load.
  // TODO: Use the precise number of buffers needed by the particular load.
  int numBuffers = -1;
  for (auto &[_, info] : loadToInfo)
    numBuffers = std::max(numBuffers, info.distToUse);
  LDBG("deduced shared memory buffer number = " << numBuffers);

  SmallVector<std::pair<Operation *, Value>> loadToAllocs;
  for (auto &[loadOp, info] : loadToInfo) {
    if (!info.sharedEncoding)
      continue;

    Value alloc = createAlloc(loadOp, info.sharedEncoding, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    sharedMemAllocs.push_back(alloc);
    loadToAllocs.emplace_back(loadOp, alloc);
  }

  IRRewriter builder(forOp.getContext());
  builder.setInsertionPoint(forOp);

  Location loc = forOp.getLoc();
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value extractIdx = minusOne;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);

  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, {extractIdx});
  forOp.erase();
  forOp = newForOp;

  // Create one counter for the extract indices to avoid creating long
  // live range.
  extractIdx = newForOp.getBody()->getArgument(newOperandIndex);

  builder.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  // Create a cluster for prefetching global reads for the dot.
  tt::CoarseSchedule::Cluster prefetchCluster = schedule.clusters.newAtBack();

  for (auto &[op, alloc] : loadToAllocs) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op))
      createStreamCopy(loadOp, alloc, extractIdx, prefetchCluster);
  }
  // Patch the yield with the updated counters.
  appendToForOpYield(forOp, {extractIdx});
}

bool StreamPipeliner::preprocessLoopAndBuildSchedule() {
  // Schedule the loads and root ops (dot ops) in the loop. This will give us
  // a scaffold for the final schedule.
  DenseSet<Operation *> rootUsers;
  scheduleLoads(rootUsers);
  if (loadToInfo.empty())
    return false;

  LLVM_DEBUG({
    LDBG("Coarse schedule loads only:");
    schedule.dump();
  });

  // Convert the loads into shared memory allocations and loads from them.
  createStreamOps();

  LLVM_DEBUG({
    LDBG("Coarse schedule with stream loads:");
    schedule.dump();
  });

  tt::CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  scheduleDependencies();
  LLVM_DEBUG({
    LDBG("Coarse schedule with dependencies:");
    schedule.dump();
  });

  scheduleDistanceOneDependencies();
  LLVM_DEBUG({
    LDBG("Coarse schedule with dist 1:");
    schedule.dump();
  });

  scheduleRemainingToLastStage(afterPrologue);
  LLVM_DEBUG({
    LDBG("Final coarse schedule:");
    schedule.dump();
  });

  // Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> coarseSchedule =
      schedule.createFinalSchedule(forOp);

  // Fill out the pipeline options.
  options.getScheduleFn =
      [coarseSchedule](scf::ForOp,
                       std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(coarseSchedule);
      };

  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  // Explicitly deallocate created allocations.
  for (auto alloc : sharedMemAllocs)
    builder.create<ttg::LocalDeallocOp>(forOp.getLoc(), alloc);
  return true;
}

// Return true if the preconditions for pipelining the loop are met.
static bool checkPrecondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) { return !operand.getDefiningOp(); }))
    return false;

  // Don't pipeline outer loops.
  auto hasNestedLoopInside = [forOp](Operation *op) {
    if (op != forOp && isa<scf::ForOp, scf::WhileOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  };
  return !forOp->walk(hasNestedLoopInside).wasInterrupted();
}

bool StreamPipeliner::pipelineLoop() {
  if (!checkPrecondition(forOp))
    return false;

  if (!preprocessLoopAndBuildSchedule())
    return false;
  LDBG("Loop before sending to expander:\n" << *forOp);

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  return succeeded(tt::pipelineForLoop(rewriter, forOp, options));
}

namespace {
struct PipelinePass : public TritonAMDGPUStreamPipelineV2Base<PipelinePass> {
  PipelinePass() = default;
  PipelinePass(int32_t numStages) { this->numStages = numStages; }

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (getNumStagesOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      StreamPipeliner sp(forOp, getNumStagesOrDefault(forOp));
      sp.pipelineLoop();
    }
  }

private:
  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists, otherwise use the
    // global control.
    if (auto attr = forOp->getAttrOfType<IntegerAttr>(tt::kNumStagesAttrName))
      return attr.getInt();
    return numStages;
  }
};
} // anonymous namespace

std::unique_ptr<Pass>
mlir::createTritonAMDGPUStreamPipelineV2Pass(int numStages) {
  return std::make_unique<PipelinePass>(numStages);
}
