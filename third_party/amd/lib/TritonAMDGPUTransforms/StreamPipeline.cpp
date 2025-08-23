#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <variant>

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create stream operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop and epilogue.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "tritonamdgpu-stream-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUSTREAMPIPELINE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

Operation *streamPredication(RewriterBase &rewriter, Operation *op,
                             Value pred) {
  // The epilogue peeling generates a select for the stage output. This causes
  // too much register pressure with the loop result and the epilogue-dot in
  // regs for the select. Conditionally executing the dot will allow the backend
  // to optimize the select away as redundant.
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(op)) {
    auto loc = dotOp->getLoc();
    auto ifOp = rewriter.create<scf::IfOp>(loc, dotOp->getResult(0).getType(),
                                           pred, /*withElseRegion=*/true);
    auto thenB = ifOp.getThenBodyBuilder();
    auto yield = thenB.create<scf::YieldOp>(loc, dotOp->getResult(0));
    dotOp->moveBefore(yield);
    ifOp.getElseBodyBuilder().create<scf::YieldOp>(loc, dotOp->getOperand(2));
    return ifOp;
  }
  return tt::wrapInMaskOp(rewriter, op, pred);
}

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

struct LoadInfo {
  // Shared layout is used for loads feeding into dot ops.
  ttg::SwizzledSharedEncodingAttr sharedEncoding = nullptr;
  // The distance of this load's stage to its use' stage.
  int distToUse = 0;
  Operation *use = nullptr;
};
using LoadToInfoMap = llvm::MapVector<Operation *, LoadInfo>;

struct StreamCopyChainOps {
  tt::LoadOp loadOp;
  ttg::MemDescIndexOp subviewOp;
  ttg::LocalStoreOp localStoreOp;
  ttg::LocalLoadOp maybeLocalLoadOp;
};

struct AsyncCopyChainOps {
  ttg::AsyncCopyGlobalToLocalOp copyOp;
  ttg::AsyncCommitGroupOp commitOp;
  ttg::AsyncWaitOp waitOp;
  ttg::LocalLoadOp maybeLocalLoadOp;
};

using StreamOpVariant = std::variant<StreamCopyChainOps, AsyncCopyChainOps>;
using LoadToStreamOpMap = llvm::MapVector<Operation *, StreamOpVariant>;

AsyncCopyChainOps createAsyncCopy(tt::LoadOp loadOp, Value alloc,
                                  Value extractIdx) {
  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  // Extract local subview from shared allocation
  auto viewLoad = triton::createSingleBufferView(builder, alloc, extractIdx)
                      .getDefiningOp<ttg::MemDescIndexOp>();

  auto copyOp = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
      loc, loadOp.getPtr(), viewLoad, loadOp.getMask(), loadOp.getOther(),
      loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
  auto commitOp =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copyOp->getResult(0));
  ttg::AsyncWaitOp waitOp =
      builder.create<ttg::AsyncWaitOp>(loc, commitOp->getResult(0), 0);

  auto maybeSharedLoad = tt::replaceUsesWithLocalLoad(
      builder, loadOp->getResult(0), viewLoad, waitOp);

  return {copyOp, commitOp, waitOp, maybeSharedLoad};
}

void scheduleLocalLoad(ttg::LocalLoadOp localLoadOp,
                       tt::CoarseSchedule &schedule, int stage,
                       const tt::CoarseSchedule::Cluster &cluster) {
  schedule.insert(localLoadOp, stage, cluster);
  // If its only user is a ConvertLayout, we place it into the same stage so
  // it can be folded by a later pass
  if (localLoadOp->hasOneUse()) {
    auto cvt = *localLoadOp->getUsers().begin();
    if (isa<ttg::ConvertLayoutOp>(cvt)) {
      schedule.insert(cvt, stage, cluster);
    }
  }
}

StreamCopyChainOps createStreamCopy(tt::LoadOp loadOp, Value alloc,
                                    Value extractIdx) {
  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  // Extract local subview from shared allocation
  auto viewLoad = triton::createSingleBufferView(builder, alloc, extractIdx)
                      .getDefiningOp<ttg::MemDescIndexOp>();

  tt::LoadOp newLoadOp = cast<tt::LoadOp>(builder.clone(*loadOp));
  auto storeOp = builder.create<ttg::LocalStoreOp>(loc, newLoadOp, viewLoad);
  auto maybeLocalLoad =
      tt::replaceUsesWithLocalLoad(builder, loadOp->getResult(0), viewLoad);

  return {newLoadOp, viewLoad, storeOp, maybeLocalLoad};
}

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

bool canBeConvertedToAsyncLoad(unsigned numBuffers, tt::LoadOp loadOp,
                               Value alloc,
                               tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                               const tt::AMD::TargetInfo &targetInfo) {
  // If we have a single buffer we would require another barrier after the
  // local_reads so instead we fall back to pipeline with registers
  // Removing this check will create incorrect IR, see
  // MembarUtility.h:membarFilter
  if (numBuffers <= 1)
    return false;

  // Compute the final vecSize we can use for the combination of sourceEncoding
  // and sharedEncoding. We can only use AsyncCopy if the target supports the
  // requested or a smaller vecSize because we cannot stride when loading
  // directly to lds
  auto srcTy = cast<RankedTensorType>(loadOp.getPtr().getType());
  auto dstTy = cast<ttg::MemDescType>(alloc.getType());
  auto regLayout = triton::gpu::toLinearLayout(srcTy);
  // It's the allocation so we trim the multibuffer dimension
  auto srcShape = dstTy.getShape().take_back(srcTy.getRank());
  auto sharedLayout =
      triton::gpu::toLinearLayout(srcShape, dstTy.getEncoding());
  auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);

  unsigned vecSize = regToSharedLayout.getNumConsecutiveInOut();
  unsigned elemBitWidth = dstTy.getElementTypeBitWidth();

  if (fitToValidDirectToLdsVecSize(vecSize, elemBitWidth, targetInfo) == 0)
    return false;

  // Checks whether the global pointer's contiguity and mask alignment allows
  // for at least 32 bit wide loads
  return triton::canBeConvertedToAsyncLoad(loadOp, axisInfoAnalysis);
}

// Convert load ops into shared memory allocation loads and apply
// multi-buffering based on the required number of buffers.
LoadToStreamOpMap
createStreamOps(const LoadToInfoMap &loadToInfo, scf::ForOp &forOp,
                const int &numBuffers, bool useAsyncCopy,
                tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  IRRewriter builder(forOp);
  Location loc = forOp.getLoc();
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value extractIdx = minusOne;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);

  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependency.
  forOp = addIterArgsToLoop(builder, forOp, {extractIdx});

  // Create one counter for the extract indices to avoid creating long
  // live range.
  extractIdx = forOp.getBody()->getArgument(newOperandIndex);

  builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  // Patch the yield with the updated counter.
  appendToForOpYield(forOp, {extractIdx});

  LoadToStreamOpMap loadToStreamOp;
  for (auto &[l, info] : loadToInfo) {
    if (!info.sharedEncoding)
      continue;

    auto loadOp = dyn_cast<tt::LoadOp>(l);
    if (!loadOp)
      continue;

    // Create an allocation that can hold distance number of loadOp shapes.
    auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
    Value alloc = triton::createAlloc(forOp, ty, loadOp->getLoc(),
                                      info.sharedEncoding, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    auto arch = getAMDArch(loadOp->getParentOfType<ModuleOp>());
    triton::AMD::TargetInfo targetInfo(arch ? arch->str() : "");

    // Replace the old load with multi-buffered loads
    if (useAsyncCopy &&
        canBeConvertedToAsyncLoad(numBuffers, loadOp, alloc, axisInfoAnalysis,
                                  targetInfo)) {
      loadToStreamOp[loadOp] = createAsyncCopy(loadOp, alloc, extractIdx);
    } else {
      loadToStreamOp[loadOp] = createStreamCopy(loadOp, alloc, extractIdx);
    }
  }

  return loadToStreamOp;
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

namespace SingleDotSchedule {
// Define categories of scheduling details per Operation types.
// The SingleDotSchedule schedules 5 types of operations:
// 1. GLOBAL_LOAD: tt.load / ttg.async_copy_global_to_local
// 2. LOCAL_STORE: ttg.local_store
// 3. LOCAL_LOAD:  ttg.local_load
// 4. COMPUTE:     ops that use the loaded data
// 5. ASYNC_WAIT:  ttg.async_wait
// Note that ttg ops mentioned in the above list are created during scheduling.
enum SchedType {
  SCHED_GLOBAL_LOAD,
  SCHED_LOCAL_STORE,
  SCHED_LOCAL_LOAD,
  SCHED_COMPUTE,
  SCHED_ASYNC_WAIT,
  SCHED_SIZE
};

using Clusters = std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE>;
using Stages = std::array<int, SCHED_SIZE>;

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
  int lastStage = numStages - 1;
  stages[SCHED_GLOBAL_LOAD] = 0;
  stages[SCHED_LOCAL_STORE] = globalPrefetch;
  stages[SCHED_LOCAL_LOAD] = lastStage - localPrefetch;
  stages[SCHED_COMPUTE] = lastStage;
  stages[SCHED_ASYNC_WAIT] = stages[SCHED_LOCAL_LOAD];

  bool pairedGlobalLoadLocalStore = stages[SCHED_LOCAL_STORE] == 0;
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

void scheduleAsyncCopy(const AsyncCopyChainOps &asyncOps, tt::LoadOp loadOp,
                       tt::CoarseSchedule &schedule, const Stages &stages,
                       const Clusters &clusters) {
  auto [copyOp, commitOp, waitOp, maybeLocalLoadOp] = asyncOps;
  auto [loadStage, loadCluster] = schedule[loadOp];
  schedule.insert(copyOp, loadStage, loadCluster);
  // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
  // later UpdateAsyncWaitCount pass can deduce better waitcnts
  schedule.insert(commitOp, loadStage, loadCluster);
  // If the LocalLoads are scheduled to a later stage than AsyncCopy we need to
  // place the AsyncCopy prefetches after the AsyncWaits which create a barrier
  // to ensure all warps are finished reading the shared buffer we will write
  // into. This is done by scheduling AsyncWait as the first cluster.
  // If AsyncCopy and LocalLoads are in the same stage we do not assign a
  // schdule so they are placed before the LocalLoads
  if (loadStage != stages[SCHED_LOCAL_LOAD])
    schedule.insert(waitOp, stages[SCHED_ASYNC_WAIT],
                    clusters[SCHED_ASYNC_WAIT]);

  if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE]) {
    scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

void scheduleStreamCopy(const StreamCopyChainOps &streamOps,
                        tt::LoadOp oldLoadOp, tt::CoarseSchedule &schedule,
                        const Stages &stages, const Clusters &clusters) {
  auto [newLoadOp, subviewOp, localStoreOp, maybeLocalLoadOp] = streamOps;
  auto [loadStage, loadCluster] = schedule[oldLoadOp];

  schedule.insert(newLoadOp, loadStage, loadCluster);
  schedule.insert(subviewOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  schedule.insert(localStoreOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE]) {
    scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
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

void scheduleStreamOps(const LoadToStreamOpMap &loadToStreamOp,
                       tt::CoarseSchedule &schedule, const Stages &stages,
                       const Clusters &clusters) {
  for (auto [l, streamOps] : loadToStreamOp) {
    auto loadOp = dyn_cast<tt::LoadOp>(l);
    if (!loadOp)
      continue;

    if (auto asyncOps = std::get_if<AsyncCopyChainOps>(&streamOps)) {
      scheduleAsyncCopy(*asyncOps, loadOp, schedule, stages, clusters);
    } else if (auto sOps = std::get_if<StreamCopyChainOps>(&streamOps)) {
      scheduleStreamCopy(*sOps, loadOp, schedule, stages, clusters);
    }
  }
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
  auto loadToStreamOp = createStreamOps(loadToInfo, forOp, numBuffers,
                                        useAsyncCopy, axisInfoAnalysis);
  scheduleStreamOps(loadToStreamOp, schedule, stages, clusters);
  dumpSchedule("Coarse schedule stream ops:");

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

// Defines the order of scheduling clusters. The suffix numbers for memory
// operations define which dot the operations belongs to. So *_LOAD_1 loads a
// tensor consumed by the first dot. If a memory operation is used by both dots
// it has to be be assigned to the *_1 clusters to ensure a valid schedule.
enum Clusters {
  // ComputeCluster1
  CLUSTER_DOT_1,
  CLUSTER_AFTER_DOT_1,
  // MemoryCluster1
  CLUSTER_ASYNC_WAIT_2,
  CLUSTER_LOCAL_WRITE_1,
  CLUSTER_LOCAL_LOAD_2,
  CLUSTER_GLOBAL_LOAD_1,
  // ComputeCluster2
  CLUSTER_DOT_2,
  CLUSTER_AFTER_DOT_2,
  // MemoryCluster2
  CLUSTER_ASYNC_WAIT_1,
  CLUSTER_LOCAL_WRITE_2,
  CLUSTER_LOCAL_LOAD_1,
  CLUSTER_GLOBAL_LOAD_2,

  CLUSTER_COUNT
};

using ChainedDotClusters =
    std::array<tt::CoarseSchedule::Cluster, CLUSTER_COUNT>;

enum Stages {
  STAGE_DOT_1 = 2,
  STAGE_DOT_2 = 3,

  STAGE_GLOBAL_LOAD_1 = 0,
  STAGE_LOCAL_WRITE_1 = 1,
  STAGE_LOCAL_LOAD_1 = 1,

  STAGE_GLOBAL_LOAD_2 = 1,
  STAGE_LOCAL_WRITE_2 = 2,
  STAGE_LOCAL_LOAD_2 = 3,
};

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

void scheduleAsyncCopy(const AsyncCopyChainOps &asyncOps, tt::LoadOp loadOp,
                       tt::CoarseSchedule &schedule,
                       const ChainedDotClusters &clusters) {
  auto [loadStage, loadCluster] = schedule[loadOp];
  auto [copyOp, commitOp, waitOp, maybeLocalLoadOp] = asyncOps;

  schedule.insert(copyOp, loadStage, loadCluster);
  // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
  // later UpdateAsyncWaitCount pass can deduce better waitcnts
  schedule.insert(commitOp, loadStage, loadCluster);

  if (loadStage == STAGE_GLOBAL_LOAD_1) {
    schedule.insert(waitOp, STAGE_LOCAL_LOAD_1, clusters[CLUSTER_ASYNC_WAIT_1]);
    if (maybeLocalLoadOp)
      scheduleLocalLoad(maybeLocalLoadOp, schedule, STAGE_LOCAL_LOAD_1,
                        clusters[CLUSTER_LOCAL_LOAD_1]);
  } else {
    schedule.insert(waitOp, STAGE_LOCAL_LOAD_2, clusters[CLUSTER_ASYNC_WAIT_2]);
    if (maybeLocalLoadOp)
      scheduleLocalLoad(maybeLocalLoadOp, schedule, STAGE_LOCAL_LOAD_2,
                        clusters[CLUSTER_LOCAL_LOAD_2]);
  }
}

void scheduleStreamCopy(const StreamCopyChainOps &streamOps, tt::LoadOp loadOp,
                        tt::CoarseSchedule &schedule,
                        const ChainedDotClusters &clusters) {
  auto [loadStage, loadCluster] = schedule[loadOp];
  auto [copyOp, subviewOp, localStoreOp, maybeLocalLoadOp] = streamOps;
  schedule.insert(copyOp, loadStage, loadCluster);

  if (loadStage == STAGE_GLOBAL_LOAD_1) {
    schedule.insert(subviewOp, STAGE_LOCAL_WRITE_1,
                    clusters[CLUSTER_LOCAL_WRITE_1]);
    schedule.insert(localStoreOp, STAGE_LOCAL_WRITE_1,
                    clusters[CLUSTER_LOCAL_WRITE_1]);

    if (maybeLocalLoadOp)
      schedule.insert(maybeLocalLoadOp, STAGE_LOCAL_LOAD_1,
                      clusters[CLUSTER_LOCAL_LOAD_1]);
  } else {
    schedule.insert(subviewOp, STAGE_LOCAL_WRITE_2,
                    clusters[CLUSTER_LOCAL_WRITE_2]);
    schedule.insert(localStoreOp, STAGE_LOCAL_WRITE_2,
                    clusters[CLUSTER_LOCAL_WRITE_2]);
    if (maybeLocalLoadOp)
      schedule.insert(maybeLocalLoadOp, STAGE_LOCAL_LOAD_2,
                      clusters[CLUSTER_LOCAL_LOAD_2]);
  }

  if (maybeLocalLoadOp) {
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(
            *maybeLocalLoadOp->getUsers().begin())) {
      auto [localLoadStage, localLoadCluster] = schedule[maybeLocalLoadOp];
      schedule.insert(cvt, localLoadStage, localLoadCluster);
    }
  }
}

void scheduleStreamOps(const LoadToStreamOpMap &loadToStreamOp,
                       tt::CoarseSchedule &schedule,
                       const ChainedDotClusters &clusters) {
  for (auto [l, streamOps] : loadToStreamOp) {
    auto loadOp = dyn_cast<tt::LoadOp>(l);
    if (!loadOp)
      continue;

    if (auto asyncOps = std::get_if<AsyncCopyChainOps>(&streamOps)) {
      scheduleAsyncCopy(*asyncOps, loadOp, schedule, clusters);
    } else if (auto sOps = std::get_if<StreamCopyChainOps>(&streamOps)) {
      scheduleStreamCopy(*sOps, loadOp, schedule, clusters);
    }
  }
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

  // Convert the loads into shared memory allocations and loads from them.
  // TODO support different numBuffers
  int numBuffers = useAsyncCopy ? 2 : 1;
  auto loadToStreamOps =
      createStreamOps(loadToInfo, forOp, /*numBuffers=*/numBuffers,
                      useAsyncCopy, axisInfoAnalysis);
  scheduleStreamOps(loadToStreamOps, schedule, clusters);
  dumpSchedule("Coarse schedule stream ops:");

  for (auto [l, _] : loadToInfo) {
    schedule.erase(l);
    l->erase();
  }

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

FailureOr<scf::ForOp> pipelineLoop(scf::ForOp forOp, int numStages,
                                   int globalPrefetch, int localPrefetch,
                                   bool useAsyncCopy, bool waitAtTail) {

  triton::AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(
      forOp->getParentOfType<ModuleOp>());

  LoadToInfoMap loadToInfo = preprocessLoop(axisInfoAnalysis, forOp, numStages);

  if (loadToInfo.empty()) {
    LDBG("couldn't find any pipeline-able loads:\n" << *forOp);
    return failure();
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
    return failure();
  }

  // Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  auto coarseSchedule = schedule.createFinalSchedule(forOp);

  tt::PipeliningOption options;
  options.supportDynamicLoops = true;
  options.peelEpilogue = true;
  options.predicateFn = streamPredication;
  // Annotate loadOp in prologue for further moving up
  options.annotateFn = [](Operation *op,
                          tt::PipeliningOption::PipelinerPart part,
                          unsigned stage) {
    if (part != tt::PipeliningOption::PipelinerPart::Prologue)
      return;

    auto annotateLoad = [](Operation *loadOp) {
      loadOp->setAttr("amd.pipeliner_part",
                      StringAttr::get(loadOp->getContext(), "prologue"));
    };

    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      annotateLoad(loadOp);
      return;
    }
    // loadOp may be wrapped by a MaskOp as predicateFn execution
    // precedes annotation
    if (auto maskOp = dyn_cast<ttg::MaskOp>(op)) {
      for (auto &innerOp : maskOp.getBody()->without_terminator()) {
        if (auto loadOp = dyn_cast<tt::LoadOp>(&innerOp)) {
          annotateLoad(loadOp);
          return;
        }
      }
    }
  };
  // Set the final schedule as our scheduling function
  options.getScheduleFn =
      [coarseSchedule](scf::ForOp,
                       std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(coarseSchedule);
      };

  LDBG("Loop before sending to expander:\n" << *forOp);

  IRRewriter rewriter(forOp);
  return tt::pipelineForLoop(rewriter, forOp, options);
}
} // namespace

struct PipelinePass : impl::TritonAMDGPUStreamPipelineBase<PipelinePass> {
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
      (void)pipelineLoop(forOp, numStagesThis, globalPrefetch, localPrefetch,
                         useAsyncCopy, waitAtTail);
    }

    // NOTE: Leave empty for now, until we utilize customEpiloguePeeling
    DenseSet<ttg::MaskOp> peeledMaskOps;
    tt::resolveMaskOp(moduleOp, peeledMaskOps);

    if (useAsyncCopy) {
      llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
      moduleOp.walk([&](ttg::AsyncWaitOp waitOp) { waitOps.insert(waitOp); });
      tt::combineRedundantWaitOps(waitOps);
    }
  }
};

} // namespace mlir
