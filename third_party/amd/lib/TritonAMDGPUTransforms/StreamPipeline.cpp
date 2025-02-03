#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/SchedInstructions.h"
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
// expander to generate the prologue and new loop and epilogue.
//===----------------------------------------------------------------------===//

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h.inc"

#define DEBUG_TYPE "tritonamdgpu-stream-pipeline"
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
  }
  return tt::predicateOp(rewriter, op, pred);
}

namespace {

//===----------------------------------------------------------------------===//
// Software pipelining generally works by anchoring on global load ops in the
// main loop and rotating the loop to schedule global load ops for future loop
// iterations together with compute for the current iteration. In this way, we
// can 1) issue memory operations earlier to hide the latency and 2) break the
// strong dependency inside on loop iteration to give backends flexiblity to
// better interleave instructions for better instruction-level parallelism.
//
// This StreamPipeliner class creates the pipelining schedule and calls the
// PipelineExpander to rewrite the `scf.for` loop accordingly. A schedule
// consists of multiple stages, where ops from different stages can overlap
// executions because the dependencies are loop carried.
//
// The general flow of this process is:
//
// 1. The user provides a `num_stages` that specifies how many stages the
//    pipeline will have. The number of stages must be larger than the distance
//    from the first independent load to the compute in order to pipeline.
// 2. A schedule is created based on the distance between the global loads
//    in the first stages and the compute that uses the loaded values in the
//    last stage (num_stages - 1). Each operation will be clustered in the
//    order to best overlap with other operations (see details below in the
//    initSchedule method).
// 3. When the compute is a tt.dot, the scheduler will insert a shared
//    memory allocation between the global load and tt.dot. The ttg.local_store
//    will save the global load value to shared memory and the ttg.local_load
//    will load the relevant tiles for the tt.dot. These operations will be
//    scheduled according to various scheduling schemes outlined below in the
//    initSchedule method (see details there).
// 4. Finally the schedule will be passed to the PipelineExpander to rewrite
//    accordingly. The new implementation will consist of:
//    a. Prologue: containing the ramp-up of num_stages-1 stages for
//       iteratorions i=[0, num_stages-1).
//    b. New loop: ordered by cluster and iterated on each operation by
//       `i + (num_stages-op_stage)`.
//    c. Epilogue: ramp-down of the last `num_stages-1` iterations for the
//       ops in stages 1 to last_stage. This must consider that the loop
//       bounds may be shorter than num_stages. In this case, the epilogue
//       iterations must align with the prologue.
//
class StreamPipeliner {
public:
  StreamPipeliner(scf::ForOp _forOp, int _numStages, bool _prefetch)
      : forOp(_forOp), prefetch(_prefetch), numStages(_numStages + prefetch),
        schedule(numStages),
        axisInfoAnalysis(forOp->getParentOfType<ModuleOp>()) {
    options.supportDynamicLoops = true;
    options.peelEpilogue = true;
    options.predicateFn = streamPredication;
  }

  LogicalResult pipelineLoop();

private:
  void initSchedule(int maxIndirectionLevel);

  void computeLoadOpsToIndirectionLevelAndUse();
  void assignMemoryLayouts();
  LogicalResult scheduleLoads(DenseSet<Operation *> &rootUsers);
  void scheduleDependencies();
  void scheduleDistanceOneDependencies();
  void scheduleRemainingToLastStage();

  LogicalResult preprocessLoopAndBuildSchedule();

  Value createAlloc(Operation *loadOp,
                    ttg::SwizzledSharedEncodingAttr sharedEnc,
                    unsigned numBuffers);
  void createStreamCopy(tt::LoadOp loadOp, Value alloc, Value extractIdx);
  void createStreamOps();

  // Define categories of scheduling details per Operation types.
  // The StreamPipeliner schedules 5 types of operations:
  // 1. GLOBAL_LOAD: tt.load
  // 2. LOCAL_STORE: ttg.local_store (created by the StreamPipeliner)
  // 3. LOCAL_LOAD:  ttg.local_load (created by the StreamPipeliner)
  // 4. COMPUTE:     ops that use the loaded data
  // 5. TAIL:        everything else in the loop
  enum SchedType {
    SCHED_GLOBAL_LOAD,
    SCHED_LOCAL_STORE,
    SCHED_LOCAL_LOAD,
    SCHED_COMPUTE,
    SCHED_TAIL
  };

  void scheduleOp(Operation *op, SchedType type, int stage = -1) {
    if (stage < 0)
      stage = config[type].stage;
    schedule.insert(op, stage, config[type].cluster);
  }

private:
  // Data members
  scf::ForOp forOp;

  // User settings
  bool prefetch;
  int numStages;

  // Scheduling clusters
  tt::CoarseSchedule schedule;

  // ScheduleConfig lookup by SchedType to get the stage and cluster.
  struct ScheduleConfig {
    int stage;
    tt::CoarseSchedule::Cluster cluster;
  };
  SmallVector<ScheduleConfig, 5> config;

  // Mapping and indirection level for each `tt.load` to its use.
  SmallVector<std::tuple<Operation *, int, Operation *>> loadOpToIndLevelAndUse;

  struct LoadInfo {
    // Shared layout is used for loads feeding into dot ops.
    ttg::SwizzledSharedEncodingAttr sharedEncoding = nullptr;
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

// Init Schedule Config based on settings and loop characteristics.
// Create clusters in order of ops in loop. This can interleave ops
// from different stages in the same cluster to achieve better backend
// scheduling.
//   WARNING: Changing the order of schedule.clusters.newAtBack() calls
//            can cause invalid schedules to be produced.
void StreamPipeliner::initSchedule(int maxIndirectionLevel) {
  int lastStage = numStages - 1;
  config.resize(SCHED_TAIL + 1);

  bool isMultibuf = numStages > (2 + maxIndirectionLevel);
  if (prefetch) {
    // Prefetch Schema cluster order and staging.
    // for i in (...):
    //   local_stores: stage=i+1
    //   global_loads: stage=i+2
    //   compute:      stage=i
    //   local_load:   stage=i+1
    //   tail:         stage=i
    config[SCHED_LOCAL_STORE] = {lastStage - 1, schedule.clusters.newAtBack()};
    auto cluster1 = schedule.clusters.newAtBack();
    config[SCHED_GLOBAL_LOAD] = {0, cluster1};
    config[SCHED_COMPUTE] = {lastStage, cluster1};
    config[SCHED_LOCAL_LOAD] = {lastStage - 1, schedule.clusters.newAtBack()};
    config[SCHED_TAIL] = {lastStage, schedule.clusters.newAtBack()};
  } else if (isMultibuf) {
    // Streaming Schema cluster order and staging for multi-buffer.
    // for i in (...):
    //   local_stores: stage=i+1
    //   global_loads: stage=i+2
    //   local_load:   stage=i
    //   compute:      stage=i
    //   tail:         stage=i
    config[SCHED_LOCAL_STORE] = {lastStage - 1, schedule.clusters.newAtBack()};
    auto cluster1 = schedule.clusters.newAtBack();
    config[SCHED_GLOBAL_LOAD] = {0, cluster1};
    config[SCHED_LOCAL_LOAD] = {lastStage, cluster1};
    config[SCHED_COMPUTE] = {lastStage, cluster1};
    config[SCHED_TAIL] = {lastStage, schedule.clusters.newAtBack()};
  } else {
    // Streaming Schema cluster order and staging for single-buffer.
    // for i in (...):
    //   global_loads: stage=i+1
    //   local_load:   stage=i
    //   compute:      stage=i
    //   local_stores: stage=i+1
    //   tail:         stage=i
    auto cluster0 = schedule.clusters.newAtBack();
    config[SCHED_GLOBAL_LOAD] = {0, cluster0};
    config[SCHED_LOCAL_LOAD] = {lastStage, schedule.clusters.newAtBack()};
    config[SCHED_COMPUTE] = {lastStage, cluster0};
    config[SCHED_LOCAL_STORE] = {lastStage - 1, schedule.clusters.newAtBack()};
    config[SCHED_TAIL] = {lastStage, schedule.clusters.newAtBack()};
  }
}

void StreamPipeliner::createStreamCopy(tt::LoadOp loadOp, Value alloc,
                                       Value extractIdx) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();
  Value src = loadOp.getPtr();
  Value mask = loadOp.getMask();
  Value other = loadOp.getOther();

  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());
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
  auto subviewTy = ttg::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), sharedMemorySpace, /*mutableMemory=*/true);
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);
  // Clean up old local caches.
  SmallVector<ttg::LocalAllocOp> allocsToErase;
  for (Operation *user : loadOp->getUsers()) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      triton::replaceUsesAndPropagateType(builder, alloc, viewLoad.getResult());
      allocsToErase.push_back(alloc);
    }
  }
  for (auto alloc : allocsToErase)
    alloc.erase();

  // Prefetch load ahead of the dot stage if is used by the dot.
  auto storeOp =
      builder.create<ttg::LocalStoreOp>(loc, copy->getResult(0), viewLoad);
  scheduleOp(viewLoad, SCHED_LOCAL_STORE);
  scheduleOp(storeOp, SCHED_LOCAL_STORE);

  // Create local load
  auto sharedLoad =
      builder.create<ttg::LocalLoadOp>(loc, loadOp.getType(), viewLoad);
  Value result = sharedLoad.getResult();
  if (prefetch)
    scheduleOp(sharedLoad, SCHED_LOCAL_LOAD);

  // If the currently processed `LoadOp` is labeled with an index regarding
  // to which `DotOp` operand the corresponding data belongs to, then label the
  // expanded `LocalStoreOp` with the same index. This is required for
  // instruction scheduling hints to correctly count the emitted `ds_write`
  // instructions for each GEMM tile.
  if (auto attr = loadOp->getAttr(triton::amdgpu::OpIdxAttr::getMnemonic())) {
    storeOp->setAttr(triton::amdgpu::OpIdxAttr::getMnemonic(), attr);
  }

  loadOp->replaceAllUsesWith(ValueRange{result});

  if (prefetch && result.hasOneUse()) {
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(*result.getUsers().begin()))
      scheduleOp(cvt, SCHED_LOCAL_LOAD);
  }

  loadOp.erase();
}

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and get the shared encoding that
// needs to be used to be compatible with users' layouts.
static std::optional<ttg::SwizzledSharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value val) {
  ttg::SwizzledSharedEncodingAttr attr;
  for (Operation *user : val.getUsers()) {
    ttg::SwizzledSharedEncodingAttr tempAttr;
    if (user->getNumResults() != 1)
      return std::nullopt;
    if (auto memDesc =
            dyn_cast<triton::gpu::MemDescType>(user->getResult(0).getType())) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SwizzledSharedEncodingAttr>(memDesc.getEncoding());
      if (!getSharedEncIfAllUsersAreDotEnc(user->getResult(0)).has_value())
        return std::nullopt;
    } else {
      if (!isa<ttg::LocalLoadOp, ttg::ConvertLayoutOp>(user))
        return std::nullopt;
      auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
          cast<triton::gpu::TensorOrMemDesc>(user->getResult(0).getType())
              .getEncoding());
      if (!dotOpEnc)
        return std::nullopt;
      auto srcTy = cast<triton::gpu::TensorOrMemDesc>(val.getType());
      auto CTALayout = ttg::getCTALayout(srcTy.getEncoding());
      auto order = ttg::getOrder(srcTy.getEncoding());
      unsigned bitWidth = srcTy.getElementType().getIntOrFloatBitWidth();
      SmallVector<unsigned> sharedOrder;
      int rank = order.size();
      // TODO rework this when shared -> dotOperand conversions support
      // arbitrary shared memory ordering
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
      tempAttr = ttg::SwizzledSharedEncodingAttr::get(
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
    if (!isa<mlir::triton::DotOpInterface>(op))
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

    // Limit shared memory sharing to width >= 32 elements.
    LDBG("Load " << *loadOp << " has width " << width);
    if (width < 32) {
      LDBG("Skip width<32 load " << *loadOp);
      continue;
    }

    if (isa<mlir::triton::DotOpInterface>(use)) {
      // Only use shared memory when feeding into a dot op.
      loadInfo.usedByDot = true;
      loadInfo.sharedEncoding =
          getSharedEncIfAllUsersAreDotEnc(op->getResult(0)).value_or(nullptr);
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

LogicalResult StreamPipeliner::scheduleLoads(DenseSet<Operation *> &rootUsers) {
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
    return failure();

  // Check which loads are good for pipelining, and assign them memory layouts.
  assignMemoryLayouts();
  if (loadToInfo.empty())
    return failure();

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

  LDBG("maxIndirectionLevel = " << maxIndirectionLevel);
  if (maxIndirectionLevel >= numStages)
    return failure();

  initSchedule(maxIndirectionLevel);

  // The stage gap between chained loads--this allows us to "spread" loads
  // with a non-one step in case the number of stages given by the user is
  // large.
  assert(numStages >= 2 && "requires num_stages=2 at least");
  unsigned stagesBetweenLoads =
      llvm::divideCeil(numStages - 2, maxIndirectionLevel + 1);
  LDBG("stagesBetweenLoads = " << stagesBetweenLoads);

  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    // Non-LoadOp(s) are the (final) root uses of all LoadOp(s).
    if (!isa<tt::LoadOp>(use)) {
      scheduleOp(use, SCHED_COMPUTE);
      rootUsers.insert(use);
    }
  }

  // Assign stages to the loads.
  for (auto [loadOp, indLevel, _] : loadOpToIndLevelAndUse) {
    int stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads;
    scheduleOp(loadOp, SCHED_GLOBAL_LOAD, stage);
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

  return success();
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

void StreamPipeliner::scheduleRemainingToLastStage() {
  int lastStage = numStages - 1;
  // Assign the rest of the ops to the last stage.
  // Take care of the ordering of the ops - uses cannot be scheduled to the
  // cluster before the definition.
  auto cluster = config[SCHED_TAIL].cluster;
  DenseMap<Operation *, tt::CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      opToCluster[&op] = cluster;
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    // We really only care about the producers from the last stage.
    // Others will be scheduled before these ops anyway.
    if (stage == lastStage) {
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
    schedule.insert(op, lastStage, cluster);
  }
}

// Create an allocation that can hold distance number of loadOp shapes.
Value StreamPipeliner::createAlloc(Operation *loadOp,
                                   ttg::SwizzledSharedEncodingAttr sharedEnc,
                                   unsigned numBuffers) {
  OpBuilder builder(forOp);
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(forOp.getContext());
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), numBuffers);
  Type memdescType = ttg::MemDescType::get(bufferShape, ty.getElementType(),
                                           sharedEnc, sharedMemorySpace,
                                           /*mutableMemory=*/true);
  auto alloc = builder.create<ttg::LocalAllocOp>(loadOp->getLoc(), memdescType);
  sharedMemAllocs.push_back(alloc);
  return alloc;
}

// Convert load ops into shared memory allocation loads and apply
// multi-buffering based on the required number of buffers.
void StreamPipeliner::createStreamOps() {
  // Calculate the number of buffers needed for each load.
  // TODO: Use the precise number of buffers needed by the particular load.
  int maxNumBuffers = -1;
  for (auto &[_, info] : loadToInfo) {
    int sharedBuffers = info.distToUse - (info.usedByDot ? prefetch : 0);
    maxNumBuffers = std::max(maxNumBuffers, sharedBuffers);
  }
  LDBG("deduced max shared memory buffer number = " << maxNumBuffers);

  SmallVector<std::pair<Operation *, Value>> loadToAllocs;
  for (auto &[loadOp, info] : loadToInfo) {
    if (!info.sharedEncoding)
      continue;

    Value alloc = createAlloc(loadOp, info.sharedEncoding, maxNumBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
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
      builder.create<arith::ConstantIntOp>(loc, maxNumBuffers, 32);

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

  // Create stream copies.
  for (auto &[op, alloc] : loadToAllocs) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op))
      createStreamCopy(loadOp, alloc, extractIdx);
  }
  // Patch the yield with the updated counters.
  appendToForOpYield(forOp, {extractIdx});
}

LogicalResult StreamPipeliner::preprocessLoopAndBuildSchedule() {
  // Schedule the loads and root ops (dot ops) in the loop. This will give us
  // a scaffold for the final schedule.
  DenseSet<Operation *> rootUsers;
  if (failed(scheduleLoads(rootUsers)))
    return failure();
  if (loadToInfo.empty())
    return failure();

  LLVM_DEBUG({
    LDBG("Coarse schedule loads only:");
    schedule.dump();
  });

  // Convert the loads into shared memory allocations and loads from them.
  createStreamOps();

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

  scheduleRemainingToLastStage();
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

  return success();
}

LogicalResult StreamPipeliner::pipelineLoop() {
  if (failed(preprocessLoopAndBuildSchedule()))
    return failure();
  LDBG("Loop before sending to expander:\n" << *forOp);

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  return tt::pipelineForLoop(rewriter, forOp, options);
}

// Return true if the preconditions for pipelining the loop are met.
static bool checkPrecondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) { return !operand.getDefiningOp(); }))
    return false;

  auto hasInvalidOp = [forOp](Operation *op) {
    // Don't pipeline outer loops.
    if (op != forOp && isa<scf::ForOp, scf::WhileOp>(op))
      return WalkResult::interrupt();
    // Don't pipeline loops with barriers.
    if (isa<gpu::BarrierOp>(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  };
  return !forOp->walk(hasInvalidOp).wasInterrupted();
}

namespace {
// Go through a single use chain to get the result of the target op after all
// unary ops - e.g., `convert_layout`, `fp_to_fp`, etc.
template <typename TargetOpType> Operation *passPrevUnaryOps(Value value) {
  auto getNextUnaryOps = [](Value value) -> Operation * {
    if (auto defOp = value.getDefiningOp()) {
      if ((defOp->getNumOperands() == 1) || llvm::dyn_cast<TargetOpType>(defOp))
        return defOp;
    }
    return nullptr;
  };

  auto unaryOp = getNextUnaryOps(value);
  while (unaryOp) {
    if (llvm::dyn_cast<TargetOpType>(unaryOp))
      return unaryOp;
    unaryOp = getNextUnaryOps(unaryOp->getOperand(0));
  }
  return nullptr;
}

// Annotate each `tt.LoadOp` instruction with its corresponding gemm operand
// index. Note, this is a part of the instruction scheduling routine. Currently,
// we support `forOp`s which contain only a single `tt.DotOp` in the bodies.
void labelLoadOpsForTritonDot(scf::ForOp forOp) {
  mlir::MLIRContext *ctx = forOp->getContext();
  if (auto dotOp = triton::getSingleDotOpIfExists(forOp)) {
    for (auto [opIdx, dotOperand] : llvm::enumerate(dotOp->getOperands())) {
      if (auto loadOp = passPrevUnaryOps<triton::LoadOp>(dotOperand)) {
        auto opIdxAttr = triton::amdgpu::OpIdxAttr::get(ctx, opIdx);
        loadOp->setAttr(triton::amdgpu::OpIdxAttr::getMnemonic(), opIdxAttr);
      }
    }
  }
}

struct PipelinePass : public TritonAMDGPUStreamPipelineBase<PipelinePass> {
  PipelinePass() = default;
  PipelinePass(int32_t numStages, int32_t prefetch) {
    this->numStages = numStages;
    this->prefetch = prefetch;
  }

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      labelLoadOpsForTritonDot(forOp);
      // Bail out for loops with num_stage <= 1.
      if (getNumStagesOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      if (!checkPrecondition(forOp))
        continue;
      StreamPipeliner sp(forOp, getNumStagesOrDefault(forOp), prefetch);
      if (failed(sp.pipelineLoop()))
        continue;
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
} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUStreamPipelinePass(int numStages,
                                                                 int prefetch) {
  return std::make_unique<PipelinePass>(numStages, prefetch);
}
