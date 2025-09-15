
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/TritonAMDGPUTransforms/Pipeline.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"
#include <variant>

#define DEBUG_TYPE "tritonamdgpu-schedule-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUSCHEDULELOOPS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Software pipelining generally works by anchoring on global load ops in the
// main loop and rotating the loop to schedule global load ops for future loop
// iterations together with compute for the current iteration. In this way, we
// can 1) issue memory operations earlier to hide the latency and 2) break the
// strong dependency inside on loop iteration to give backends flexibility to
// better interleave instructions for better instruction-level parallelism.
//
// The code here creates the pipelining schedule and calls the
// PipelineExpander to rewrite the `scf.for` loop accordingly. A schedule
// consists of multiple stages, where ops from different stages can overlap
// executions because the dependencies are loop carried.
//
// The general flow of this process is:
//
// 1. The user provides a `num_stages` that specifies how many stages the
//    pipeline will have. The number of stages must be larger than the distance
//    from the first independent load to the compute in order to pipeline.
//    1.a. User may also specify `global_prefetch=<s>` to set the number of
//         stages between tt.load and ttg.local_store ops.
//    1.b. User may also specify `local_prefetch=<s>` to set the number of
//         stages between ttg.local_load and compute.
// 2. A schedule is created based on the distance between the global loads
//    in the first stages and the compute that uses the loaded values in the
//    last stage (num_stages - 1). Each operation will be clustered in the
//    order to best overlap with other operations (see details below in the
//    initSchedule methods).
// 3. When the compute is a tt.dot, the scheduler will insert a shared
//    memory allocation between the global load and tt.dot. The global load
//    value will be saved to shared memory, via ttg.local_store or via
//    ttg.async_copy_global_to_local writing directly to shared memory, and the
//    ttg.local_load will load the relevant tiles for the tt.dot. These
//    operations will be scheduled according to various scheduling schemes
//    outlined below in the initSchedule methods (see details there).
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

// Returns the given |inputValue|'s dot user result encoding and updates |opIdx|
// with which dot operand |inputValue| is fed into if possible.
ttg::AMDMfmaEncodingAttr getDotEncoding(Value inputValue, unsigned *opIdx) {
  if (!inputValue.hasOneUse())
    return nullptr;

  Operation *user = *inputValue.getUsers().begin();
  if (user->getNumResults() != 1 ||
      user->getBlock() != inputValue.getParentBlock())
    return nullptr;

  if (auto dotOp = dyn_cast<tt::DotOpInterface>(user)) {
    OpOperand &use = *inputValue.getUses().begin();
    *opIdx = use.getOperandNumber();
    auto dotType = cast<RankedTensorType>(dotOp->getResult(0).getType());
    return dyn_cast<ttg::AMDMfmaEncodingAttr>(dotType.getEncoding());
  }
  return getDotEncoding(user->getResult(0), opIdx);
}

// Adapted from
// lib/Dialect/TritonGPU/Transforms/Utility.cpp::getSharedEncIfAllUsersAreDotEnc
// to support AMDMfmaEncodingAttr.
// TODO(max): figure out how to refactor to use upstream
//
// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and get the shared encoding that
// needs to be used to be compatible with users' layouts.
std::optional<ttg::SwizzledSharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value loadedValue) {
  ttg::SwizzledSharedEncodingAttr attr;
  for (Operation *user : loadedValue.getUsers()) {
    LDBG(" getSharedEncIfAllUsersAreDotEnc current user: " << *user);
    if (user->getNumResults() != 1)
      return std::nullopt;

    ttg::SwizzledSharedEncodingAttr tempAttr;
    Value userResult = user->getResult(0);
    Type userResType = userResult.getType();
    if (auto memDesc = dyn_cast<ttg::MemDescType>(userResType)) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SwizzledSharedEncodingAttr>(memDesc.getEncoding());
      if (!getSharedEncIfAllUsersAreDotEnc(userResult).has_value())
        return std::nullopt;
    } else {
      if (!(isa<ttg::ConvertLayoutOp>(user) ||
            user->hasTrait<OpTrait::LocalLoadTrait>()))
        return std::nullopt;

      auto srcTy = cast<ttg::TensorOrMemDesc>(loadedValue.getType());
      auto ctaLayout = ttg::getCTALayout(srcTy.getEncoding());
      auto order = getOrderForMemory(srcTy);
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

      auto userResEnc = cast<ttg::TensorOrMemDesc>(userResType).getEncoding();
      if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(userResEnc)) {
        tempAttr = ttg::SwizzledSharedEncodingAttr::get(
            loadedValue.getContext(), dotOpEnc, srcTy.getShape(), sharedOrder,
            ctaLayout, bitWidth, /*needTrans=*/false);
      } else if (auto llEnc = dyn_cast<ttg::LinearEncodingAttr>(userResEnc)) {
        // We use linear layout directly for scaled dot fp8 operands. For such
        // cases, we need to look further down the def-use chain to find the dot
        // op for the mfma layout to deduce operand index and other information.
        unsigned opIdx;
        if (auto dotEnc = getDotEncoding(userResult, &opIdx)) {
          unsigned vecSize = llEnc.getLinearLayout().getNumConsecutiveInOut();
          LDBG("deduced opIdx: " << opIdx << "; deduced vecSize: " << vecSize);
          tempAttr = dotEnc.composeSharedLayoutForOperand(
              ctaLayout, opIdx, srcTy.getShape(), order, vecSize, bitWidth,
              /*needTrans=*/false);
        }
      }
    }
    // Check that the shared encodings needed by the users are compatible.
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return std::nullopt;
    attr = tempAttr;
  }
  return attr;
}

LoadToInfoMap
preprocessLoop(triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
               scf::ForOp &forOp, int numStages) {
  auto arch = getAMDArch(forOp->getParentOfType<ModuleOp>());
  triton::AMD::ISAFamily isaFamily = triton::AMD::ISAFamily::Unknown;
  if (arch)
    isaFamily = triton::AMD::deduceISAFamily(*arch);

  bool pipelineWithoutDot = forOp->hasAttr(mlir::triton::kNumStagesAttrName);
  bool filterSmallVectors = isaFamily != triton::AMD::ISAFamily::CDNA4;
  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel =
      triton::gpu::loadOpsToIndirectionLevel(forOp, pipelineWithoutDot,
                                             axisInfoAnalysis, numStages,
                                             filterSmallVectors);

  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevel.size() << " loads to pipeline:");
    for (const auto &[l, i] : loadOpToIndLevel) {
      LDBG("  - load: " << *l);
      LDBG("    at distance: " << i.first);
      LDBG("    used by op: " << *i.second);
    }
  });

  LoadToInfoMap loadToInfo;
  for (const auto &[load, info] : loadOpToIndLevel) {
    auto [distance, use] = info;
    auto sharedEncoding =
        getSharedEncIfAllUsersAreDotEnc(load->getResult(0)).value_or(nullptr);
    loadToInfo[load] = {sharedEncoding, distance, use};
  }

  return loadToInfo;
}
} // namespace

namespace SingleDotSchedule {
LogicalResult createStages(Stages &stages, int &numBuffers, int maxDist,
                           int numStages, int globalPrefetch, int localPrefetch,
                           bool waitAtTail) {
  int lastStage = numStages - 1;
  stages[SCHED_GLOBAL_LOAD] = 0;
  stages[SCHED_LOCAL_STORE] = globalPrefetch;
  stages[SCHED_LOCAL_LOAD] = lastStage - localPrefetch;
  stages[SCHED_COMPUTE] = lastStage;
  stages[SCHED_ASYNC_WAIT] = stages[SCHED_LOCAL_LOAD];

  stages[SCHED_LOCAL_STORE] += maxDist;
  if (waitAtTail) {
    stages[SCHED_ASYNC_WAIT] = std::max(0, stages[SCHED_LOCAL_LOAD] - 1);
  }

  LDBG(
      "Stage schedule:" << "  GLOBAL_LOAD stage = " << stages[SCHED_GLOBAL_LOAD]
                        << ", LOCAL_STORE stage = " << stages[SCHED_LOCAL_STORE]
                        << ", LOCAL_LOAD stage = " << stages[SCHED_LOCAL_LOAD]
                        << ", COMPUTE stage = " << stages[SCHED_COMPUTE]
                        << ", ASYNC_WAIT stage = " << stages[SCHED_ASYNC_WAIT]
                        << "; total = " << numStages);

  if (stages[SCHED_LOCAL_STORE] >= numStages ||
      stages[SCHED_LOCAL_STORE] > stages[SCHED_LOCAL_LOAD]) {
    LDBG("Invalid stage schedule");
    return failure();
  }

  // Calculate the number of buffers needed for each load.
  // TODO: Use the precise number of buffers needed by the particular load.
  numBuffers =
      std::max(1, stages[SCHED_LOCAL_LOAD] - stages[SCHED_LOCAL_STORE]);
  // If we use AsyncCopy we need one more buffer since we are not using a
  // register buffer
  if (useAsyncCopy) {
    numBuffers += 1;
  }

  LDBG("deduced max shared memory buffer number = " << numBuffers);

  return success();
}

LogicalResult createClusters(Clusters &clusters, tt::CoarseSchedule &schedule,
                             const Stages &stages, int maxDist, int numBuffers,
                             bool waitAtTail) {
  bool pairedGlobalLoadLocalStore = stages[SCHED_LOCAL_STORE] == maxDist;
  // We place async wait as the first cluster because we want to have it being
  // the first in the main loop after pipelining.
  // In case we use async_copy with pingpong, we need to place async_wait at
  // the end of the previous iteration, so it can guarantee the correct
  // dependency when warp0 and warp1 are pipelined.
  int asyncWaitCluster = waitAtTail ? 4 : 0;
  // If tt.load and ttg.local_store are in the same stage
  //   spread them apart to allow overlap with compute
  // else
  //   Initiate ttg.local_store before tt.load
  int globalLoadCluster = 1;
  int localStoreCluster = 3;
  if (!pairedGlobalLoadLocalStore) {
    globalLoadCluster = 3;
    localStoreCluster = 2;
  }

  // If ttg.local_load and ttg.local_store are in the same stage
  //   spread them apart to allow overlap with compute
  // else if they share the buffer
  //   ttg.local_load must come first
  // else
  //   schedule ttg.local_load in the middle
  int localLoadCluster = globalLoadCluster;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_LOCAL_STORE]) {
    localLoadCluster = std::max(3, localStoreCluster + 1);
  } else if (numBuffers == 1 && localLoadCluster >= localStoreCluster) {
    // For 1 buffer, ttg.local_load must occur before ttg.local_store
    localLoadCluster = localStoreCluster - 1;
  }

  // Schedule compute with ttg.local_load if paired
  // otherwise, schedule in the middle
  int computeCluster = 2;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_COMPUTE]) {
    computeCluster = localLoadCluster;
  }

  // Make assignments
  Clusters clusterVec;
  std::generate(clusterVec.begin(), clusterVec.end(),
                [&]() { return schedule.clusters.newAtBack(); });

  llvm::outs() << "dump schedule.clusters:\n"
               << SCHED_GLOBAL_LOAD << ": " << *(clusterVec[SCHED_GLOBAL_LOAD])
               << "\n"
               << SCHED_LOCAL_STORE << ": " << *(clusterVec[SCHED_LOCAL_STORE])
               << "\n"
               << SCHED_LOCAL_LOAD << ": " << *(clusterVec[SCHED_LOCAL_LOAD])
               << "\n"
               << SCHED_COMPUTE << ": " << *(clusterVec[SCHED_COMPUTE]) << "\n"
               << SCHED_ASYNC_WAIT << ": " << *(clusterVec[SCHED_ASYNC_WAIT])
               << "\n";

  clusters[SCHED_GLOBAL_LOAD] = clusterVec[globalLoadCluster];
  clusters[SCHED_LOCAL_STORE] = clusterVec[localStoreCluster];
  clusters[SCHED_LOCAL_LOAD] = clusterVec[localLoadCluster];
  clusters[SCHED_COMPUTE] = clusterVec[computeCluster];
  clusters[SCHED_ASYNC_WAIT] = clusterVec[asyncWaitCluster];

  llvm::outs() << "dump clusters:\n"
               << SCHED_GLOBAL_LOAD << ": " << *(clusters[SCHED_GLOBAL_LOAD])
               << "\n"
               << SCHED_LOCAL_STORE << ": " << *(clusters[SCHED_LOCAL_STORE])
               << "\n"
               << SCHED_LOCAL_LOAD << ": " << *(clusters[SCHED_LOCAL_LOAD])
               << "\n"
               << SCHED_COMPUTE << ": " << *(clusters[SCHED_COMPUTE]) << "\n"
               << SCHED_ASYNC_WAIT << ": " << *(clusters[SCHED_ASYNC_WAIT])
               << "\n";

  LDBG("Cluster schedule:" << "  GLOBAL_LOAD cluster = " << globalLoadCluster
                           << ", LOCAL_STORE cluster = " << localStoreCluster
                           << ", LOCAL_LOAD cluster = " << localLoadCluster
                           << ", COMPUTE cluster = " << computeCluster
                           << ", ASYNC_WAIT cluster = " << asyncWaitCluster
                           << "; total = " << SCHED_SIZE);

  return success();
}

// Init Schedule Config based on settings and loop characteristics.
// Create clusters in order of ops in loop. This can interleave ops
// from different stages in the same cluster to achieve better backend
// scheduling.
//   WARNING: Changing the order of schedule.clusters.newAtBack() calls
//            can cause invalid schedules to be produced.
LogicalResult initSchedule(int maxDist, Stages &stages, int numStages,
                           int globalPrefetch, int localPrefetch,
                           bool useAsyncCopy, bool waitAtTail,
                           Clusters &clusters, tt::CoarseSchedule &schedule) {
  int numBuffers = 1;
  if (failed(createStages(stages, numBuffers, maxDist, numStages,
                          globalPrefetch, localPrefetch, waitAtTail)))
    return failure();

  return createClusters(clusters, schedule, stages, maxDist, numBuffers,
                        waitAtTail);
}

LogicalResult scheduleLoads(const LoadToInfoMap &loadToInfo, int maxDist,
                            int numStages, const Stages &stages,
                            const Clusters &clusters,
                            tt::CoarseSchedule &schedule) {
  // The stage gap between chained loads--this allows us to "spread" loads
  // with a non-one step in case the number of stages given by the user is
  // large.
  assert(numStages >= 2 && "requires num_stages=2 at least");
  unsigned stagesBetweenLoads = llvm::divideCeil(numStages - 2, maxDist + 1);
  LDBG("stagesBetweenLoads = " << stagesBetweenLoads);

  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, info] : loadToInfo) {
    // Non-LoadOp(s) are the (final) root uses of all LoadOp(s).
    if (!isa<tt::LoadOp>(info.use))
      schedule.insert(info.use, stages[SCHED_COMPUTE], clusters[SCHED_COMPUTE]);
  }

  // Assign stages to the loads.
  for (auto [loadOp, info] : loadToInfo) {
    int stage = (maxDist - info.distToUse) * stagesBetweenLoads;
    schedule.insert(loadOp, stages[stage], clusters[SCHED_GLOBAL_LOAD]);
  }

  return success();
}

tt::CoarseSchedule
buildSchedule(scf::ForOp &forOp, int numStages, const LoadToInfoMap &loadToInfo,
              int globalPrefetch, int localPrefetch, bool useAsyncCopy,
              bool waitAtTail,
              triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  tt::CoarseSchedule schedule(numStages);
  Stages stages;
  Clusters clusters;

  auto dumpSchedule = [&](llvm::StringRef msg) {
    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      LDBG(msg);
      schedule.dump();
    });
  };

  int maxDist = 0;
  for (auto &[l, info] : loadToInfo) {
    maxDist = std::max(maxDist, info.distToUse);
  }

  if (failed(initSchedule(maxDist, stages, numStages, globalPrefetch,
                          localPrefetch, useAsyncCopy, waitAtTail, clusters,
                          schedule)))
    return {};

  if (failed(scheduleLoads(loadToInfo, maxDist, numStages, stages, clusters,
                           schedule)))
    return {};
  dumpSchedule("Coarse schedule loads only:");

  scheduleDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dependencies:");

  triton::gpu::scheduleDistanceOneDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dist 1:");

  tt::CoarseSchedule::Cluster computeCluster = clusters[SCHED_COMPUTE];
  triton::gpu::scheduleRemainingToLastStage(forOp, schedule, computeCluster);
  dumpSchedule("Final coarse schedule:");

  std::vector<std::pair<Operation *, unsigned>> coarseSchedule =
      schedule.createFinalSchedule(forOp);

  return schedule;
}
} // namespace SingleDotSchedule

namespace ChainedDotSchedule {
LogicalResult checkPreconditions(scf::ForOp forOp, int numStages,
                                 LoadToInfoMap loadToInfo) {
  if (numStages != 4)
    return failure();

  auto dotOps = llvm::to_vector(forOp.getBody()->getOps<tt::DotOp>());

  if (dotOps.size() != 2)
    return failure();

  // Check that the first dot feeds into the second
  SetVector<Operation *> slice;
  getForwardSlice(dotOps[0]->getResult(0), &slice);
  if (!slice.contains(dotOps[1])) {
    return failure();
  }

  // Reject loops with indirect loads
  // TODO support indirect loads
  if (llvm::any_of(loadToInfo,
                   [](auto it) { return it.second.distToUse != 0; })) {
    return failure();
  }

  return success();
}

// We schedule loads one stage in front of their dots
LogicalResult
scheduleLoads(std::array<tt::DotOp, 2> dotOps,
              const llvm::MapVector<Operation *, LoadInfo> &loadToInfo,
              const ChainedDotClusters &clusters,
              tt::CoarseSchedule &schedule) {
  for (auto [loadOp, info] : loadToInfo) {
    if (info.use == dotOps[0]) {
      schedule.insert(loadOp, STAGE_GLOBAL_LOAD_1,
                      clusters[CLUSTER_GLOBAL_LOAD_1]);
    } else if (info.use == dotOps[1]) {
      schedule.insert(loadOp, STAGE_GLOBAL_LOAD_2,
                      clusters[CLUSTER_GLOBAL_LOAD_2]);
    } else {
      LDBG(*loadOp << " will not be pipelined because it's not used by a dot");
    }
  }
  return success();
}

LogicalResult scheduleOpsBetweenDots(scf::ForOp forOp,
                                     std::array<tt::DotOp, 2> dotOps,
                                     tt::CoarseSchedule &schedule,
                                     const ChainedDotClusters &clusters) {
  SetVector<Operation *> dot0Slice;
  getForwardSlice(Value(dotOps[0]), &dot0Slice);

  // For each operand of the second dot coming from the first dot we want to
  // split the ops in between into 2 parts.
  // One part will be on the same stage as dot1 but interleaved with dot2 and
  // the second part will be on the next stage and interleaved with dot1.
  // We split when we reach an op having more than one user. Splitting further
  // up would require us to duplicate the op/data to ensure the other user is
  // scheduled correctly.
  for (auto operand : dotOps[1]->getOperands()) {
    auto operandDefOp = operand.getDefiningOp();

    // Skip if the op is not part of the forward slice
    if (!operandDefOp || !dot0Slice.contains(operand.getDefiningOp()))
      continue;

    // DFS-like traversal of the def-chain to find op with more than 1 user
    llvm::SmallVector<Value> queue;
    queue.push_back(operand);

    while (!queue.empty()) {
      auto v = queue.pop_back_val();
      auto defOp = v.getDefiningOp();
      // Abort path if we hit a blockarg, left the forward slice of dot0 or the
      // op has already a schedule
      if (!defOp || !dot0Slice.contains(defOp) || schedule.count(defOp) != 0) {
        continue;
      }

      auto numUsers = llvm::range_size(defOp->getUsers());
      if (numUsers > 1) {
        // Schedule this op to interleave with dot2. All its unscheduled
        // dependencies will be scheduled the same by scheduleDependencies
        schedule.insert(defOp, STAGE_DOT_1, clusters[CLUSTER_AFTER_DOT_2]);
        // Schedule the dot2 operand to interleave with dot1. Its unscheduled
        // dependencies will be scheduled the same by scheduleDependencies
        schedule.insertIfAbsent(operandDefOp, STAGE_DOT_2,
                                clusters[CLUSTER_AFTER_DOT_1]);
        continue;
      }
      // Follow def chain
      for (Value prevOperand : defOp->getOperands()) {
        queue.push_back(prevOperand);
      }
    }
  }

  // Schedule users of dot1 but not feeding into dot2 to overlap with dot1
  auto yield = forOp.getBody()->getTerminator();
  for (auto yieldOperand : yield->getOperands()) {
    auto defOp = yieldOperand.getDefiningOp();
    if (!defOp || !dot0Slice.contains(defOp))
      continue;

    schedule.insertIfAbsent(defOp, STAGE_DOT_2, clusters[CLUSTER_AFTER_DOT_1]);
  }

  return success();
}

tt::CoarseSchedule
buildSchedule(scf::ForOp &forOp, int numStages, const LoadToInfoMap &loadToInfo,
              bool useAsyncCopy,
              triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  tt::CoarseSchedule schedule(numStages);
  ChainedDotClusters clusters;
  std::generate(clusters.begin(), clusters.end(),
                [&]() { return schedule.clusters.newAtBack(); });
  auto dumpSchedule = [&](llvm::StringRef msg) {
    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      LDBG(msg);
      schedule.dump();
    });
  };

  // Schedule dots
  auto dotOpsVec = llvm::to_vector(forOp.getBody()->getOps<tt::DotOp>());
  assert(dotOpsVec.size() == 2); // Ensure precondition
  std::array<tt::DotOp, 2> dotOps = {dotOpsVec[0], dotOpsVec[1]};

  schedule.insert(dotOps[0], STAGE_DOT_1, clusters[CLUSTER_DOT_1]);
  schedule.insert(dotOps[1], STAGE_DOT_2, clusters[CLUSTER_DOT_2]);

  if (failed(scheduleLoads(dotOps, loadToInfo, clusters, schedule)))
    return {};
  dumpSchedule("Coarse schedule load and dots only:");

  if (failed(scheduleOpsBetweenDots(forOp, dotOps, schedule, clusters))) {
    return {};
  }
  dumpSchedule("Coarse schedule after schedule ops between dots:");

  scheduleDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dependencies:");

  triton::gpu::scheduleDistanceOneDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dist 1:");

  tt::CoarseSchedule::Cluster lastCluster = clusters.back();
  triton::gpu::scheduleRemainingToLastStage(forOp, schedule, lastCluster);
  dumpSchedule("Final coarse schedule:");

  return schedule;
}
} // namespace ChainedDotSchedule

namespace {
void scheduleLoop(scf::ForOp forOp, int numStages, int globalPrefetch,
                  int localPrefetch, bool useAsyncCopy, bool waitAtTail) {
  triton::AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(
      forOp->getParentOfType<ModuleOp>());

  LoadToInfoMap loadToInfo = preprocessLoop(axisInfoAnalysis, forOp, numStages);

  if (loadToInfo.empty()) {
    LDBG("couldn't find any pipeline-able loads:\n" << *forOp);
    return;
  }

  tt::CoarseSchedule schedule;

  if (succeeded(ChainedDotSchedule::checkPreconditions(forOp, numStages,
                                                       loadToInfo))) {
    schedule = ChainedDotSchedule::buildSchedule(
        forOp, numStages, loadToInfo, useAsyncCopy, axisInfoAnalysis);
  } else {
    schedule = SingleDotSchedule::buildSchedule(
        forOp, numStages, loadToInfo, globalPrefetch, localPrefetch,
        useAsyncCopy, waitAtTail, axisInfoAnalysis);
  }

  if (schedule.empty()) {
    return;
  }

  schedule.serialize(forOp);
}
} // namespace

struct ScheduleLoops
    : public impl::TritonAMDGPUScheduleLoopsBase<ScheduleLoops> {
  using impl::TritonAMDGPUScheduleLoopsBase<
      ScheduleLoops>::TritonAMDGPUScheduleLoopsBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // check numStages
    if (globalPrefetch < 0 || globalPrefetch >= numStages) {
      moduleOp.emitError("global prefetch control must be in [0, ")
          << numStages << "); " << globalPrefetch << " is out of range";
      return signalPassFailure();
    }

    if (localPrefetch < 0 || localPrefetch >= numStages) {
      moduleOp.emitError("local prefetch control must be in [0, ")
          << numStages << "); " << localPrefetch << " is out of range";
      return signalPassFailure();
    }

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (tt::getNumStagesOrDefault(forOp, numStages) > 1)
        loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      if (!triton::gpu::isSafeToPipeline(forOp)) {
        LDBG("Loop not safe to pipeline:\n" << *forOp);
        continue;
      }
      // i.e., we can still disable `waitAtTail` by explicitly disabling
      // pingpong, which is the only use case of this scheduling variant.
      int numStagesThis = tt::getNumStagesOrDefault(forOp, numStages);
      bool waitAtTail = usePingpong && (numStagesThis == 3) && useAsyncCopy;
      scheduleLoop(forOp, numStagesThis, globalPrefetch, localPrefetch,
                   useAsyncCopy, waitAtTail);
    }
  }
};
} // namespace mlir
