#include "TritonAMDGPUTransforms/Passes.h"
// #include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "amd/lib/TritonAMDGPUTransforms/PipelineUtility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
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

#define DEBUG_TYPE "tritonamdgpu-schedule-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUSCHEDULELOOPS
#include "TritonAMDGPUTransforms/Passes.h.inc"

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

/// Returns true if for a given global load with loadType, loading instead with
/// targetLLAttr maintains at least the same level of coalescing/vectorization
/// with same amount of load ops.
static bool isCoalesced(RankedTensorType loadType,
                        ttg::LinearEncodingAttr targetLLAttr) {
  // Expect a BlockedEncoding on the load.
  auto *ctx = loadType.getContext();
  auto loadEnc = loadType.getEncoding();
  auto blockedEnc = dyn_cast_or_null<ttg::BlockedEncodingAttr>(loadEnc);
  auto shape = loadType.getShape();
  if (!blockedEnc)
    return false;

  // Contiguous (fastest) dimension as defined by the blocked encoding.
  const unsigned contigDim = blockedEnc.getOrder()[0];
  const unsigned originalContigPerThread =
      blockedEnc.getSizePerThread()[contigDim];

  // This is the correct way to compute vectorization instead of using
  // getContigPerThread. However, currently global load vectorizer doesn't
  // support vectorization that require in thread permutation (NOTE: local_load
  // op lowering does support this!) such as: #ttg.linear<{register = [[0, 2],
  // [0, 1]], ...}>, so we don't use largest vectorization here as well. This
  // should be updated once vectorization in load op lowering is fixed..
  //
  // auto ctaLayout = ttg::getCTALayout(loadType.getEncoding());
  // // Dummy shared layout that emulates global memory so we can use
  // // largestVectorisation utility.
  // auto sharedEncoding = ttg::SwizzledSharedEncodingAttr::get(
  //     ctx, 1, 1, 1, blockedEnc.getOrder(), ctaLayout);
  // auto sharedLL = triton::gpu::toLinearLayout(shape, sharedEncoding);
  // auto invertedLL = ll.invertAndCompose(sharedLL).flattenOuts();

  // auto [contigPerThreadLL, permutation] =
  //     largestVectorisation(ctx, invertedLL, bitwidth, std::nullopt);

  const unsigned targetContigPerThread =
      targetLLAttr.getContigPerThread()[contigDim];
  const unsigned targetContigPerWarp =
      targetLLAttr.getContigPerWarp()[contigDim];
  const unsigned originalContigPerWarp =
      targetLLAttr.getContigPerWarp()[contigDim];
  auto targetLL = targetLLAttr.toLinearLayout(shape);

  // 1) Require that the linear layout provides at least as much per-thread and
  // per-warp contiguity as the original load encoding.
  if (targetContigPerThread < originalContigPerThread ||
      targetContigPerWarp < originalContigPerWarp)
    return false;

  // 2) Check that there is no broadcasting along the warp dimension.
  // Broadcasting would force multiple warps to share the same elements,
  // resulting in additional global_load instructions compared to a blocked
  // layout.
  auto kWarp = StringAttr::get(ctx, "warp");

  auto basesIt = targetLL.getBases().find(kWarp);
  if (basesIt == targetLL.getBases().end())
    return false;

  const auto &bases = basesIt->second;
  for (const auto &basis : bases) {
    const bool allZero = std::all_of(basis.begin(), basis.end(),
                                     [](int64_t v) { return v == 0; });
    if (allZero)
      return false;
  }

  return true;
}

/// Determine if it is safe to bypass LDS for dot operands.
/// Normally, dot operation operands are consumed in the dot MFMA layout,
/// which is not coalesced. To better utilize global memory bandwidth,
/// operands are usually loaded in a coalesced "blocked" layout and then
/// rearranged through LDS.
///
/// However, certain optimizations allow dot operands to be preshuffled in
/// global memory. In that case, the operands can be loaded efficiently
/// (in a coalesced way) and consumed directly by the dot operation.
/// When preshuffling is used, a sequence of transpose and reshape ops
/// must be applied to the operand.
///
/// To verify that preshuffling was done correctly and the final layout
/// remains coalesced, we start from the dot MFMA layout and apply the
/// inverse of each transpose/reshape op (while ignoring convert_layout
/// ops) until we reach the load. We then inspect the resulting layout
/// to decide if it is coalesced enough to load directly, without needing
/// any further rearrangement.
static Operation *bypassLDS(Operation *load, Operation *use) {
  if (!load || !use)
    return nullptr;

  // Only applies to dot-like ops (scaled/regular) that conform to this
  // interface.
  if (!isa<tt::DotOpInterface>(use))
    return nullptr;

  // Find operands of 'use' that are in the forward slice of 'load'.
  SetVector<Operation *> fwdSlice;
  mlir::getForwardSlice(load, &fwdSlice);

  SmallVector<Operation *> defs;
  defs.reserve(use->getNumOperands());
  for (Value opnd : use->getOperands()) {
    if (Operation *def = opnd.getDefiningOp()) {
      if (fwdSlice.contains(def))
        defs.push_back(def);
    }
  }

  // Expect that 'load' op matches with a single operand for dot op.
  if (defs.size() != 1)
    return nullptr;

  Operation *def = defs.front();
  if (!def)
    return nullptr;

  // Thread encodings from 'def' back to 'load', skipping explicit converts.
  Attribute resultEnc =
      cast<RankedTensorType>(def->getResult(0).getType()).getEncoding();
  if (!resultEnc)
    return nullptr;

  Attribute srcEnc = nullptr;
  Operation *cur = def;

  while (cur && cur != load) {
    if (!isa<triton::ReshapeOp, triton::TransposeOpInterface,
             ttg::ConvertLayoutOp>(cur)) {
      return nullptr;
    }
    // Skip explicit layout converts.
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(cur)) {
      cur = cvt.getSrc().getDefiningOp();
      continue;
    }

    // Infer the source encoding that would produce 'resultEnc' from 'cur' op.
    srcEnc = inferSrcEncoding(cur, resultEnc);
    if (!srcEnc)
      return nullptr;

    resultEnc = srcEnc;
    assert(cur->getNumOperands() == 1);

    Value in = cur->getOperand(0);
    cur = in.getDefiningOp();
  }

  // Must land exactly on the original load.
  if (cur != load || !srcEnc)
    return nullptr;

  // Check coalescing under the inferred linear encoding.
  auto loadType = cast<RankedTensorType>(load->getResult(0).getType());
  auto distTrait = dyn_cast<ttg::DistributedEncodingTrait>(srcEnc);
  if (!distTrait)
    return nullptr;

  auto srcLL = ttg::toLinearEncoding(distTrait, loadType.getShape());
  if (!isCoalesced(loadType, srcLL))
    return nullptr;

  // Finally, rewrite the load to use the inferred (better) encoding.
  auto newOp = convertDistributedOpEncoding(srcEnc, load);
  return newOp;
};

// Returns the given |inputValue|'s dot user result encoding and updates |opIdx|
// and |vecSize| with which dot operand |inputValue| is fed into if possible.
ttg::AMDMfmaEncodingAttr getDotEncoding(Value inputValue, unsigned *opIdx,
                                        unsigned *vecSize) {
  if (!inputValue.hasOneUse())
    return nullptr;

  Operation *user = *inputValue.getUsers().begin();
  if (user->getNumResults() != 1 ||
      user->getBlock() != inputValue.getParentBlock())
    return nullptr;

  LDBG("getDotEncoding user: " << *user);
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(user)) {
    OpOperand &use = *inputValue.getUses().begin();
    *opIdx = use.getOperandNumber();
    auto operandType = cast<RankedTensorType>(inputValue.getType());
    *vecSize = ttg::toLinearLayout(operandType).getNumConsecutiveInOut();
    auto dotType = cast<RankedTensorType>(dotOp->getResult(0).getType());
    return dyn_cast<ttg::AMDMfmaEncodingAttr>(dotType.getEncoding());
  }

  return getDotEncoding(user->getResult(0), opIdx, vecSize);
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
  llvm::SmallVector<ttg::SwizzledSharedEncodingAttr> sharedEncs;
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
      LDBG("Deduced shared encoding candidate from memDesc: " << tempAttr);
      if (!getSharedEncIfAllUsersAreDotEnc(userResult).has_value()) {
        return std::nullopt;
      }
      sharedEncs.push_back(tempAttr);
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
        LDBG("Deduced shared encoding candidate from dot layout: " << tempAttr);
        sharedEncs.push_back(tempAttr);
      } else if (auto llEnc = dyn_cast<ttg::LinearEncodingAttr>(userResEnc)) {
        // We use linear layout directly for scaled dot fp8 operands. For such
        // cases, we need to look further down the def-use chain to find the dot
        // op for the mfma layout to deduce operand index and other information.
        unsigned opIdx;
        unsigned vecSize;
        if (auto mfmaEnc = getDotEncoding(userResult, &opIdx, &vecSize)) {
          LDBG("deduced opIdx: " << opIdx << "; deduced vecSize: " << vecSize);
          tempAttr = mfmaEnc.composeSharedLayoutForOperand(
              ctaLayout, opIdx, srcTy.getShape(), order, vecSize, bitWidth,
              /*needTrans=*/false);
          LDBG("Deduced shared encoding candidate from mfma layout: "
               << tempAttr);
          sharedEncs.push_back(tempAttr);
        }
      }
    }
  }

  auto equalSharedEncIgnoreVec = [](ttg::SwizzledSharedEncodingAttr a,
                                    ttg::SwizzledSharedEncodingAttr b) {
    if (!a || !b)
      return false;
    return (a.getPerPhase() == b.getPerPhase() &&
            a.getMaxPhase() == b.getMaxPhase() &&
            a.getOrder() == b.getOrder() &&
            a.getCTALayout() == b.getCTALayout());
  };
  if (sharedEncs.empty() || !sharedEncs.front())
    return std::nullopt;
  auto maxVecSharedEnc = sharedEncs.front();

  for (auto sharedEnc : sharedEncs) {
    if (!equalSharedEncIgnoreVec(sharedEnc, maxVecSharedEnc)) {
      LDBG("Incompatible shared encodings");
      return std::nullopt;
    }
    if (sharedEnc.getVec() > maxVecSharedEnc.getVec()) {
      maxVecSharedEnc = sharedEnc;
    }
  }

  LDBG("Deduced shared encoding: " << maxVecSharedEnc);

  return maxVecSharedEnc;
}

LoadToInfoMap
preprocessLoop(triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
               scf::ForOp &forOp, int numStages) {
  auto arch = getAMDArch(forOp->getParentOfType<ModuleOp>());
  triton::AMD::ISAFamily isaFamily = triton::AMD::ISAFamily::Unknown;
  if (arch)
    isaFamily = triton::AMD::deduceISAFamily(*arch);

  bool pipelineWithoutDot = forOp->hasAttr(mlir::triton::kNumStagesAttrName);
  bool filterSmallVectors =
      isaFamily != triton::AMD::ISAFamily::CDNA4 && !isRDNA(isaFamily);
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
    auto newLoad = bypassLDS(load, use);
    if (newLoad) {
      loadToInfo[newLoad] = {nullptr, distance, use};
    } else {
      LDBG("Deduce shared encoding for: " << *load);
      auto sharedEncoding =
          getSharedEncIfAllUsersAreDotEnc(load->getResult(0)).value_or(nullptr);
      loadToInfo[load] = {sharedEncoding, distance, use};
      LDBG("Populate loadInfo with shared encoding: " << sharedEncoding);
    }
  }

  return loadToInfo;
}

namespace SingleDotSchedule {
// Init Schedule Config based on settings and loop characteristics.
// Create clusters in order of ops in loop. This can interleave ops
// from different stages in the same cluster to achieve better backend
// scheduling.
//   WARNING: Changing the order of schedule.clusters.newAtBack() calls
//            can cause invalid schedules to be produced.
LogicalResult initSchedule(int maxDist, Stages &stages, int numStages,
                           int &numBuffers, int globalPrefetch,
                           int localPrefetch, bool useAsyncCopy,
                           bool waitAtTail, Clusters &clusters,
                           tt::CoarseSchedule &schedule) {
  LDBG("Init SingleDotSchedule");
  int lastStage = numStages - 1;
  stages[SCHED_GLOBAL_LOAD] = 0;
  stages[SCHED_LOCAL_STORE] = globalPrefetch;
  stages[SCHED_LOCAL_LOAD] = lastStage - localPrefetch;
  stages[SCHED_COMPUTE] = lastStage;
  stages[SCHED_ASYNC_WAIT] = stages[SCHED_LOCAL_LOAD];

  bool pairedGlobalLoadLocalStore = globalPrefetch == 0;
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
  if (globalPrefetch != 0) {
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
  if ((lastStage - localPrefetch) == (globalPrefetch + maxDist)) {
    localLoadCluster = std::max(3, localStoreCluster + 1);
  } else if (numBuffers == 1 && localLoadCluster >= localStoreCluster) {
    // For 1 buffer, ttg.local_load must occur before ttg.local_store
    localLoadCluster = localStoreCluster - 1;
  }

  // Schedule compute with ttg.local_load if paired
  // otherwise, schedule in the middle
  int computeCluster = 2;
  if ((lastStage - localPrefetch) == stages[SCHED_COMPUTE]) {
    computeCluster = localLoadCluster;
  }

  // Make assignments
  Clusters clusterVec;
  if (schedule.clusters.size() == 0) {
    std::generate(clusterVec.begin(), clusterVec.end(),
                  [&]() { return schedule.clusters.newAtBack(); });
  } else {
    int cnt = clusterVec.size() - schedule.clusters.size();
    for (int i = 0; i < cnt; i++) {
      schedule.clusters.newAtBack();
    }
    auto it = schedule.clusters.begin();
    for (int i = 0; i < clusterVec.size(); i++, it++) {
      clusterVec[i] = it;
    }
  }

  clusters[SCHED_GLOBAL_LOAD] = clusterVec[globalLoadCluster];
  clusters[SCHED_LOCAL_STORE] = clusterVec[localStoreCluster];
  clusters[SCHED_LOCAL_LOAD] = clusterVec[localLoadCluster];
  clusters[SCHED_COMPUTE] = clusterVec[computeCluster];
  clusters[SCHED_ASYNC_WAIT] = clusterVec[asyncWaitCluster];

  LDBG("Cluster schedule:" << "  GLOBAL_LOAD cluster = " << globalLoadCluster
                           << ", LOCAL_STORE cluster = " << localStoreCluster
                           << ", LOCAL_LOAD cluster = " << localLoadCluster
                           << ", COMPUTE cluster = " << computeCluster
                           << ", ASYNC_WAIT cluster = " << asyncWaitCluster
                           << "; total = " << SCHED_SIZE);

  return success();
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
  LDBG("Build SingleDotSchedule");
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

  int numBuffers = 1;
  if (failed(initSchedule(maxDist, stages, numStages, numBuffers,
                          globalPrefetch, localPrefetch, useAsyncCopy,
                          waitAtTail, clusters, schedule)))
    return {};

  if (failed(scheduleLoads(loadToInfo, maxDist, numStages, stages, clusters,
                           schedule)))
    return {};
  dumpSchedule("Coarse schedule loads only:");

  // Convert the loads into shared memory allocations and loads from them.
  // auto loadToStreamOp = createStreamOps(loadToInfo, forOp, numBuffers,
  //                                       useAsyncCopy, axisInfoAnalysis);
  // scheduleStreamOps(loadToStreamOp, schedule, stages, clusters);
  // dumpSchedule("Coarse schedule stream ops:");

  scheduleDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dependencies:");

  triton::gpu::scheduleDistanceOneDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dist 1:");

  tt::CoarseSchedule::Cluster computeCluster = clusters[SCHED_COMPUTE];
  triton::gpu::scheduleRemainingToLastStage(forOp, schedule, computeCluster);
  dumpSchedule("Final coarse schedule:");

  // std::vector<std::pair<Operation *, unsigned>> coarseSchedule =
  //     schedule.createFinalSchedule(forOp);

  return schedule;
}
} // namespace SingleDotSchedule

// Builds a schedule for loops containing chained dots. This schedule aims to
// better interleave mfams with alu ops which can be co-executed on GFX9. It
// works for loops which have 2 dots where the result of the first is
// transformed and used by the second dot. The dot ops will be scheduled with a
// distance of one and the ops in between will be spit into 2 parts. The first
// part will be scheduled to the same stage as the fist dot so it can interleave
// with the second dot. Whereas the second part will be scheduled to the stage
// of the second dot so it can be interleaved with the first dot. Loads will be
// double buffered and placed in between the dot/compute clusters. This
// pipeliner is meant to be used in combination with pingpong
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
  LDBG("Build ChainedDotSchedule");
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

  // // Convert the loads into shared memory allocations and loads from them.
  // // TODO support different numBuffers
  // int numBuffers = useAsyncCopy ? 2 : 1;
  // auto loadToStreamOps =
  //     createStreamOps(loadToInfo, forOp, /*numBuffers=*/numBuffers,
  //                     useAsyncCopy, axisInfoAnalysis);
  // scheduleStreamOps(loadToStreamOps, schedule, clusters);
  // dumpSchedule("Coarse schedule stream ops:");

  // for (auto [l, _] : loadToInfo) {
  //   schedule.erase(l);
  //   l->erase();
  // }

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

void pipelineLoop(scf::ForOp forOp, int numStages, int globalPrefetch,
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

struct ScheduleLoops : impl::TritonAMDGPUScheduleLoopsBase<ScheduleLoops> {
  using Base::Base;

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
      pipelineLoop(forOp, numStagesThis, globalPrefetch, localPrefetch,
                   useAsyncCopy, waitAtTail);
    }
  }
};

} // namespace mlir
