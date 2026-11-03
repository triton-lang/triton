#include "Dialect/TritonAMDGPU/IR/TargetFeatures.h"
#include "LDSEncoding.h"
#include "StreamOps.h"
#include "SubtileFusion.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/AsyncUtility.h"
#include "amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "amd/lib/TritonAMDGPUTransforms/Pipeliner/PipelineUtility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-pipeline-lower-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
using mlir::triton::amdgpu::TargetFeatures;

//===----------------------------------------------------------------------===//
// This file will conditionally allocate lds memory, create local/async load
// operations, and create schedule for these operations. After lowerLoops,
// schedule will be passed to expandLoops and eventually to PipelineExpander.
//
// The LDS encodings are chosen in LDSEncoding.h, the ops replacing each load
// are built by StreamOps.h and SubtileFusion.h; this file decides which ops to
// build and where they go in the schedule.
//===----------------------------------------------------------------------===//

using mlir::triton::AMD::AttrBypassLDS;

namespace mlir {
namespace {

void scheduleLocalLoad(ttg::LocalLoadOp localLoadOp,
                       tt::CoarseSchedule &schedule, int stage,
                       const tt::CoarseSchedule::Cluster &cluster) {
  schedule.insert(localLoadOp, stage, cluster);

  // Place the ConvertLayout users into the same stage so they can be folded
  // into the load by a later pass. A load left in a different stage than its
  // conversion keeps a distributed layout across the stage boundary, and the
  // conversion then lowers through LDS, which puts a round trip back on the
  // critical path of the dot the prefetch was meant to feed.
  SmallVector<ttg::ConvertLayoutOp> cvts;
  for (Operation *user : localLoadOp->getUsers())
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(user))
      if (cvt->getBlock() == localLoadOp->getBlock())
        cvts.push_back(cvt);

  // An operand shared by several dots (M/N-sliced GEMMs) carries one identical
  // conversion per consuming dot. Keep a single conversion per result layout so
  // folding yields one LDS read rather than one per consumer. The survivor is
  // moved next to the load so it dominates the uses it takes over.
  llvm::MapVector<Type, ttg::ConvertLayoutOp> canonical;
  SmallVector<ttg::ConvertLayoutOp> redundant;
  for (ttg::ConvertLayoutOp cvt : cvts)
    if (!canonical.try_emplace(cvt.getType(), cvt).second)
      redundant.push_back(cvt);

  for (auto &[type, cvt] : canonical) {
    cvt->moveAfter(localLoadOp);
    schedule.insert(cvt, stage, cluster);
  }
  for (ttg::ConvertLayoutOp cvt : redundant) {
    cvt.getResult().replaceAllUsesWith(canonical[cvt.getType()].getResult());
    schedule.erase(cvt);
    cvt.erase();
  }
}

// Convert load ops into shared memory allocation loads and apply
// multi-buffering based on the required number of buffers. The loads of each
// of `subtileGroups` share one whole-tile buffer.
LoadToStreamOpMap
createStreamOps(const LoadToInfoMap &loadToInfo, scf::ForOp &forOp,
                const int &numBuffers, bool useAsyncCopy,
                tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                ArrayRef<SubtileLoadGroup> subtileGroups = {}) {
  IRRewriter builder(forOp);
  Location loc = forOp.getLoc();
  Value minusOne = arith::ConstantIntOp::create(builder, loc, -1, 32);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  Value extractIdx = minusOne;
  Value numBuffersVal =
      arith::ConstantIntOp::create(builder, loc, numBuffers, 32);

  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependency.
  forOp = addIterArgsToLoop(builder, forOp, {extractIdx});

  // Create one counter for the extract indices to avoid creating long
  // live range.
  extractIdx = forOp.getBody()->getArgument(newOperandIndex);

  builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  extractIdx = arith::AddIOp::create(builder, loc, extractIdx, one);
  Value cndExt = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                       extractIdx, numBuffersVal);
  extractIdx = arith::SelectOp::create(builder, loc, cndExt, extractIdx, zero);

  // Patch the yield with the updated counter.
  appendToForOpYield(forOp, {extractIdx});

  DenseMap<Operation *, const SubtileLoadGroup *> groupOf;
  for (const SubtileLoadGroup &group : subtileGroups)
    for (tt::DescriptorLoadOp child : group.children)
      groupOf[child] = &group;

  LoadToStreamOpMap loadToStreamOp;
  for (auto &[op, info] : loadToInfo) {
    if (!info.sharedEncoding)
      continue;

    auto loadOp = dyn_cast<tt::LoadOp>(op);
    auto descLoadOp = dyn_cast<tt::DescriptorLoadOp>(op);
    auto gatherOp = dyn_cast<tt::DescriptorGatherOp>(op);
    if (!loadOp && !descLoadOp && !gatherOp)
      continue;

    // The first child of a group reached materializes the whole group.
    if (const SubtileLoadGroup *group = groupOf.lookup(op)) {
      if (!loadToStreamOp.count(op))
        createFusedTDMAsyncCopy(*group, info.sharedEncoding, forOp, extractIdx,
                                numBuffers, loadToStreamOp);
      continue;
    }

    // Create an allocation that can hold distance number of loadOp shapes.
    auto ty = cast<RankedTensorType>(op->getResultTypes()[0]);
    Value alloc = triton::createAlloc(forOp, ty, op->getLoc(),
                                      info.sharedEncoding, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    triton::AMD::TargetInfo targetInfo(
        getAMDArch(op->getParentOfType<ModuleOp>()));

    // Replace the old load with multi-buffered loads
    if (descLoadOp) {
      loadToStreamOp[op] = createTDMAsyncCopy(descLoadOp, alloc, extractIdx);
      continue;
    }
    if (gatherOp) {
      loadToStreamOp[op] = createTDMAsyncGather(gatherOp, alloc, extractIdx);
      continue;
    }

    if (useAsyncCopy &&
        canBeConvertedToAsyncLoad(numBuffers, loadOp, info.sharedEncoding,
                                  axisInfoAnalysis, targetInfo)) {
      unsigned vec = axisInfoAnalysis.getContiguity(loadOp.getPtr());
      if (auto mask = loadOp.getMask())
        vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));
      loadToStreamOp[loadOp] = createAsyncCopy(loadOp, alloc, extractIdx, vec);
      continue;
    }

    loadToStreamOp[loadOp] = createStreamCopy(loadOp, alloc, extractIdx);
  }

  return loadToStreamOp;
}

} // namespace

namespace SingleDotSchedule {
namespace {
using namespace mlir::SingleDotSchedule;
using ClusterMap = DenseMap<tt::CoarseSchedule::ClusterHash, int>;

ClusterMap createClusterMap(tt::CoarseSchedule &schedule) {
  DenseMap<tt::CoarseSchedule::ClusterHash, int> clusterMap;
  for (auto &[op, stageAndCluster] : schedule.opToStageAndCluster) {
    auto [stage, cluster] = stageAndCluster;
    tt::CoarseSchedule::ClusterHash clusterHash =
        tt::CoarseSchedule::hashCluster(cluster);
    clusterMap[clusterHash] = *cluster;
  }

  return clusterMap;
}

// Only consider slicing cases without chained dots (where the result of one dot
// is used as the A or B operand of another dot). Chained dots usually come from
// attention kernels, whose schedule is handled in
// ChainedDotSchedule::updateSchedule.
SmallVector<Operation *>
getInterleavableDots(scf::ForOp forOp, int numLoadChains, bool ldsPrefetch) {
  if (!ldsPrefetch)
    return {};

  SmallVector<Operation *> dots;
  for (Operation &op : forOp.getBody()->without_terminator())
    if (isa<tt::DotOpInterface>(&op))
      dots.push_back(&op);
  if (dots.size() < 2 || dots.size() != static_cast<size_t>(numLoadChains))
    return {};

  DenseSet<Operation *> dotSet(dots.begin(), dots.end());
  for (Operation *dot : dots) {
    auto dotOp = cast<tt::DotOpInterface>(dot);
    for (Value operand : {dotOp.getA(), dotOp.getB()}) {
      BackwardSliceOptions options;
      options.omitBlockArguments = true;
      SetVector<Operation *> slice;
      (void)getBackwardSlice(operand, &slice, options);
      if (llvm::any_of(slice,
                       [&](Operation *op) { return dotSet.contains(op); }))
        return {};
    }
  }
  return dots;
}

// Remap global and compute clusters to the right place. Each dot goes to its
// own group so the groups interleave; everything else in the compute cluster
// runs after the dots and therefore belongs to the last group.
void remapClusters(tt::CoarseSchedule &schedule, ClusterMap clusterMap,
                   ArrayRef<Clusters> groupClusters,
                   const DenseMap<Operation *, int> &dotToGroup) {
  int lastGroup = groupClusters.size() - 1;
  for (auto &[op, stageAndCluster] : schedule.opToStageAndCluster) {
    tt::CoarseSchedule::ClusterHash clusterHash =
        tt::CoarseSchedule::hashCluster(stageAndCluster.second);
    int oldClusterId = clusterMap[clusterHash];
    if (oldClusterId == 0) {
      // The original loads are replaced by the stream ops, which pick their own
      // group; the cluster here only has to be a valid one.
      stageAndCluster.second = groupClusters[0][SCHED_GLOBAL_LOAD];
    } else {
      assert(oldClusterId == 1);
      auto it = dotToGroup.find(op);
      int group = it != dotToGroup.end() ? it->second : lastGroup;
      stageAndCluster.second = groupClusters[group][SCHED_COMPUTE];
    }
  }
}

// Init Schedule Config based on settings and loop characteristics.
// Create clusters in order of ops in loop. This can interleave ops
// from different stages in the same cluster to achieve better backend
// scheduling.
//   WARNING: Changing the order of schedule.clusters.newAtBack() calls
//            can cause invalid schedules to be produced.
LogicalResult initSchedule(scf::ForOp forOp, int maxDist, Stages &stages,
                           int numStages, int &numBuffers, int numLoadChains,
                           bool useAsyncCopy, bool hasTDMLoad,
                           bool hasScaledDot, bool waitAtTail,
                           SmallVector<Operation *> &dots,
                           SmallVector<Clusters> &groupClusters,
                           tt::CoarseSchedule &schedule) {
  LDBG("Init SingleDotSchedule");
  int lastStage = numStages - 1;
  stages[SCHED_GLOBAL_LOAD] = 0;
  stages[SCHED_LOCAL_STORE] = 0;
  stages[SCHED_LOCAL_LOAD] = lastStage;
  stages[SCHED_COMPUTE] = lastStage;

  // LDS prefetch of dot operands: read a tile out of LDS ahead of the tt.dot
  // that consumes it, so the read overlaps earlier matrix ops. num_stages must
  // be at least 3 for there to be room between the two.
  //
  // How far ahead is a property of the loop body. Several independent dots
  // (the M/N/K slicing case) leave room inside one iteration, so only the
  // leading dot's operands cross the backedge and every other chain is read
  // one dot ahead of its consumer, where its tile retires a dot later rather
  // than an iteration later. A single dot is that rule's special case - no
  // earlier dot to be read after, so all of its operands cross the backedge in
  // registers.
  //
  // We take a conservative route and only enable LDS prefetch when the dot is
  // not scaled until more testing is done.
  // waitAtTail (pingpong) requires very specific scheduling, which is
  // incompatible with the prefetch cluster ordering below, so disable prefetch
  // in that case.
  bool ldsPrefetch =
      hasTDMLoad && !hasScaledDot && numStages >= 3 && !waitAtTail;
  if (ldsPrefetch) {
    stages[SCHED_LOCAL_LOAD] = lastStage - 1;
  }

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
  // Multi-dot prefetch reads its non-prefetched chains a stage later,
  // which would normally cost a slot, but it also refills each of those buffers
  // after the read that frees it (see scheduleStreamOps), so the extra slot is
  // not needed and the ring stays the same depth as under full prefetch.
  numBuffers =
      std::max(1, stages[SCHED_LOCAL_LOAD] - stages[SCHED_LOCAL_STORE]);
  // If we use AsyncCopy or TDM loads, we need one more buffer since we are not
  // using a register buffer
  if (useAsyncCopy || hasTDMLoad) {
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

  // ldsPrefetch ordering: lead the loop body with the tt.dot (it consumes
  // operands pre-loaded into registers the previous iteration, so it has no
  // intra-iteration dependency), then the async_wait, the next-tile TDM copy,
  // and finally the ttg.local_load that feeds the next iteration.
  if (ldsPrefetch) {
    computeCluster = 0;
    globalLoadCluster = 1;
    asyncWaitCluster = 2;
    localLoadCluster = 3;
    localStoreCluster = 4;
  }

  // TDM copy reordering to improve latency hiding.
  bool tdmReorder = hasTDMLoad && hasScaledDot && !waitAtTail;
  if (tdmReorder) {
    globalLoadCluster = 0; // all TDM copies first
    asyncWaitCluster = 1;  // single wait after the copies
    localLoadCluster = 2;  // local_loads after the wait
    computeCluster = 3;    // dot_scaled last
    localStoreCluster = 4;
  }

  // Place synchronous LDS writes after the tail wait so Membar can rendezvous
  // before issuing them. Preserve same-stage store-to-load ordering.
  if (waitAtTail && stages[SCHED_LOCAL_STORE] < stages[SCHED_LOCAL_LOAD]) {
    asyncWaitCluster = 3;
    localStoreCluster = 4;
  }

  // Repeat the whole cluster sequence once per dot. A group is one such run of
  // SCHED_SIZE clusters: group g owns clusters [g * SCHED_SIZE, (g+1) *
  // SCHED_SIZE) and leads with dot g, so an op placed in group g is emitted
  // between dot g and dot g+1. That is the only thing a group is -- a range of
  // the cluster order, not anything the schedule records. One dot leaves a
  // single group, which is the plain one-cluster-per-category schedule.
  dots = getInterleavableDots(forOp, numLoadChains, ldsPrefetch);
  int numGroups = std::max<int>(1, dots.size());
  DenseMap<Operation *, int> dotToGroup;
  for (auto [i, dot] : llvm::enumerate(dots))
    dotToGroup[dot] = i;

  // Create a hash map to associate cluster hash in old schedule with its
  // clusterID
  ClusterMap clusterMap = createClusterMap(schedule);

  // Make assignments
  SmallVector<tt::CoarseSchedule::Cluster> clusterVec;
  schedule.clusters.clear();
  for (int i = 0; i < SCHED_SIZE * numGroups; ++i)
    clusterVec.push_back(schedule.clusters.newAtBack());

  groupClusters.assign(numGroups, Clusters{});
  for (int group = 0; group < numGroups; ++group) {
    int base = group * SCHED_SIZE;
    Clusters &clusters = groupClusters[group];
    clusters[SCHED_GLOBAL_LOAD] = clusterVec[base + globalLoadCluster];
    clusters[SCHED_LOCAL_STORE] = clusterVec[base + localStoreCluster];
    clusters[SCHED_LOCAL_LOAD] = clusterVec[base + localLoadCluster];
    clusters[SCHED_COMPUTE] = clusterVec[base + computeCluster];
    clusters[SCHED_ASYNC_WAIT] = clusterVec[base + asyncWaitCluster];
  }

  remapClusters(schedule, clusterMap, groupClusters, dotToGroup);

  LDBG("Interleaving the loop body across " << numGroups << " dot groups");

  LDBG("Cluster schedule:" << "  GLOBAL_LOAD cluster = " << globalLoadCluster
                           << ", LOCAL_STORE cluster = " << localStoreCluster
                           << ", LOCAL_LOAD cluster = " << localLoadCluster
                           << ", COMPUTE cluster = " << computeCluster
                           << ", ASYNC_WAIT cluster = " << asyncWaitCluster
                           << "; total = " << SCHED_SIZE);

  return success();
}

//===----------------------------------------------------------------------===//
// Stream op placement
//
// A chain is one load's replacement: the ops that fill an LDS buffer and the
// local_load that reads the tile back. Chains of one fused whole-tile copy
// share their fill ops, so placement works on two levels. Each read is placed
// on its own, beside the dot it feeds; each buffer is filled once, placed to
// serve every read of it.
//
// A fill schedule is written in the form of a single chain's: the stages and
// clusters of the group whose read the fill serves, with the copy moved to the
// group that refills the slot.
//===----------------------------------------------------------------------===//

void scheduleTDMFill(Operation *tdmOp, Operation *waitOp, int copyStage,
                     tt::CoarseSchedule &schedule, const Stages &stages,
                     const Clusters &clusters) {
  schedule.insert(tdmOp, copyStage, clusters[SCHED_GLOBAL_LOAD]);
  // If the LocalLoads are scheduled to a later stage than AsyncCopy we need to
  // place the AsyncCopy prefetches after the AsyncWaits which create a barrier
  // to ensure all warps are finished reading the shared buffer we will write
  // into. This is done by scheduling AsyncWait as the first cluster.
  // If AsyncCopy and LocalLoads are in the same stage we do not assign a
  // schedule so they are placed before the LocalLoads
  if (copyStage != stages[SCHED_LOCAL_LOAD])
    schedule.insert(waitOp, stages[SCHED_ASYNC_WAIT],
                    clusters[SCHED_ASYNC_WAIT]);
}

void scheduleAsyncCopyFill(const AsyncCopyChainOps &asyncOps, int copyStage,
                           tt::CoarseSchedule &schedule, const Stages &stages,
                           const Clusters &clusters) {
  auto [copyOp, commitOp, waitOp, maybeLocalLoadOp] = asyncOps;
  schedule.insert(copyOp, copyStage, clusters[SCHED_GLOBAL_LOAD]);
  // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
  // later UpdateAsyncWaitCount pass can deduce better waitcnts
  schedule.insert(commitOp, copyStage, clusters[SCHED_GLOBAL_LOAD]);
  // See scheduleTDMFill.
  if (copyStage != stages[SCHED_LOCAL_LOAD])
    schedule.insert(waitOp, stages[SCHED_ASYNC_WAIT],
                    clusters[SCHED_ASYNC_WAIT]);
}

void scheduleStreamCopyFill(const StreamCopyChainOps &streamOps,
                            tt::LoadOp oldLoadOp, tt::CoarseSchedule &schedule,
                            const Stages &stages, const Clusters &clusters) {
  auto [newLoadOp, subviewOp, localStoreOp, maybeLocalLoadOp] = streamOps;
  auto [loadStage, loadCluster] = schedule[oldLoadOp];
  schedule.insert(newLoadOp, loadStage, loadCluster);
  schedule.insert(subviewOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  schedule.insert(localStoreOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
}

std::optional<int> firstConsumerDot(ttg::LocalLoadOp localLoadOp,
                                    ArrayRef<Operation *> dots) {
  SetVector<Operation *> slice;
  getForwardSlice(localLoadOp.getOperation(), &slice);
  for (auto [i, dot] : llvm::enumerate(dots))
    if (slice.contains(dot))
      return i;
  return std::nullopt;
}

// Where a chain's tile is read out of LDS. `prefetched` says the read happens
// an iteration ahead, so the tile crosses the backedge in registers.
struct ReadPlacement {
  int group;
  bool prefetched;
};

// Put each read in the group before its first consumer: group g reads
// `dot g, copy, wait, local_load`, so the tile lands just before dot g+1.
// With three dots, the chains feeding d0, d1, d2 take groups 2, 0, 1. The d0
// chain wraps to the tail, i.e. the previous iteration, and that wrap is the
// prefetch across the backedge. Collisions take the nearest free group below;
// a chain in the last group has no later group to be read in and is
// prefetched as well. Any other group from the consumer on would read the tile
// after the dot that needs it, so when none of the allowed groups is free the
// read shares the group before its consumer.
SmallVector<ReadPlacement>
assignReadPlacements(const LoadToStreamOpMap &loadToStreamOp,
                     ArrayRef<Operation *> dots, int numGroups) {
  SmallVector<ReadPlacement> reads;
  SmallVector<bool> taken(numGroups, false);
  for (auto &[l, streamOps] : loadToStreamOp) {
    // A chain no dot consumes has no preference; treat it like the leading
    // dot's, which keeps it prefetched and out of the way.
    std::optional<int> consumer;
    if (ttg::LocalLoadOp localLoad = getChainLocalLoad(streamOps))
      consumer = firstConsumerDot(localLoad, dots);

    // Probing down from the preferred group visits the groups below the
    // consumer and then wraps to the last group. The leading dot's chains are
    // prefetched wherever they land, so they may take any group.
    int first = consumer.value_or(0);
    int preferred = (first - 1 + numGroups) % numGroups;
    int numAllowed = first == 0 ? numGroups : first + 1;
    int group = preferred;
    for (int tries = 1; tries < numAllowed && taken[group]; ++tries)
      group = (group - 1 + numGroups) % numGroups;
    if (taken[group])
      group = preferred;
    taken[group] = true;

    bool prefetched = first == 0 || group + 1 == numGroups;
    assert((prefetched || group < first) &&
           "a read must precede the dot that consumes it");
    reads.push_back({group, prefetched});
  }
  return reads;
}

// The group whose copy refills the slot `read` frees. The copy goes half the
// groups further on, so the ring keeps its depth. A read that does not cross
// the backedge shares that slot with its own refill, so the refill clamps
// instead of wrapping and the read stays first.
//
// Advancing exactly half of the groups is a heuristic based on measurements on
// an fp16 GEMM kernel and is not necessary for correctness. read.group + 1
// would be enough to guarantee correctness, but in practice it gives worse
// results.
int getRefillGroup(ReadPlacement read, int numGroups) {
  int group = read.group + numGroups / 2;
  group = read.prefetched ? group % numGroups : std::min(group, numGroups - 1);
  assert((read.prefetched || group > read.group) &&
         "the copy must follow the read whose buffer it refills");
  return group;
}

// The stages a read of `read` runs in. The leading dot's chains are read a
// stage early, so their tiles live in registers from the previous iteration
// until the dot consumes them. Every other chain is read at the compute stage
// instead: with dots d0 and d1, the d1 tile is loaded right after d0 and dies
// at d1 rather than crossing the backedge. Two dots then hold one live tile
// instead of two.
Stages getReadStages(ReadPlacement read, const Stages &stages) {
  Stages readStages = stages;
  if (!read.prefetched) {
    readStages[SCHED_LOCAL_LOAD] = stages[SCHED_COMPUTE];
    readStages[SCHED_ASYNC_WAIT] = stages[SCHED_COMPUTE];
  }
  return readStages;
}

void scheduleRead(ttg::LocalLoadOp localLoad, ReadPlacement read,
                  tt::CoarseSchedule &schedule, const Stages &stages,
                  ArrayRef<Clusters> groupClusters) {
  // Without LDS prefetch the read shares the compute stage and is left to be
  // placed with its consumer.
  if (!localLoad ||
      (read.prefetched && stages[SCHED_LOCAL_LOAD] == stages[SCHED_COMPUTE]))
    return;
  scheduleLocalLoad(localLoad, schedule,
                    getReadStages(read, stages)[SCHED_LOCAL_LOAD],
                    groupClusters[read.group][SCHED_LOCAL_LOAD]);
}

// One LDS buffer: the chain whose ops fill it and every chain reading it.
struct StreamBuffer {
  Operation *origLoad;
  StreamOpVariant fill;
  SmallVector<ReadPlacement> reads;
};

// Coarse-schedule slot for one buffer's fill ops. `stages` and `clusters` are
// arrays indexed by SchedType (copy, wait, local store, ...), same shape as a
// single chain. Wait uses the earliest read (prefetched first); the copy uses
// the latest refill group's load cluster.
struct FillPlacement {
  Stages stages;
  Clusters clusters;
};

// A buffer's wait has to precede all of its reads and its copy has to follow
// all of them: the wait serves the earliest read, prefetched ones first since
// they run a stage early, and the copy goes to the latest refill group.
FillPlacement placeFill(const StreamBuffer &buffer, const Stages &stages,
                        ArrayRef<Clusters> groupClusters) {
  ReadPlacement first =
      *llvm::min_element(buffer.reads, [](ReadPlacement a, ReadPlacement b) {
        return std::make_pair(!a.prefetched, a.group) <
               std::make_pair(!b.prefetched, b.group);
      });
  int numGroups = groupClusters.size();
  int refillGroup = 0;
  for (ReadPlacement read : buffer.reads)
    refillGroup = std::max(refillGroup, getRefillGroup(read, numGroups));

  FillPlacement fill{getReadStages(first, stages), groupClusters[first.group]};
  fill.clusters[SCHED_GLOBAL_LOAD] =
      groupClusters[refillGroup][SCHED_GLOBAL_LOAD];
  return fill;
}

// placeFill parks each fused buffer's wait next to that buffer's earliest
// (prefetched) read. Fusion still emits one wait per parent tile, so two
// fused operands (A and B) get two waits, and collisions can put those
// waits in different groups:
//
//   %a1 = ttg.local_load %sliceA1            // current K
//   %b1 = ttg.local_load %sliceB1
//   d1
//   %wA = amdg.async_tdm_wait %tokA          // placeFill(A)
//   %a0 = ttg.local_load %sliceA0            // next K, feeds the backedge
//   d2
//   %wB = amdg.async_tdm_wait %tokB          // placeFill(B), one group later
//   %b0 = ttg.local_load %sliceB0
//   copy A, copy B
//
// If every current-K slice of every fused buffer is read before any next-K
// slice, that split is a single cut. Move every fused wait to the end of
// the group before the cut and every fused copy to the load cluster of the
// group after it (right after that group's dot). The waits then sit
// together and fold to one barrier, which both publishes the next-K tiles
// and frees the slots the copies refill:
//
//   %a1 = ttg.local_load %sliceA1
//   %b1 = ttg.local_load %sliceB1
//   %w  = amdg.async_tdm_wait %tokA, %tokB   // end of group 0
//   d1
//   copy A, copy B                          // after d1
//   %a0 = ttg.local_load %sliceA0
//   %b0 = ttg.local_load %sliceB0
//
// Only buffers with a next-K slice take part. A buffer read only at the
// current K waits in the compute stage ahead of its first read, which can
// come before the cut, so it keeps the fill placeFill gave it.
void gatherSharedFills(ArrayRef<StreamBuffer> buffers,
                       MutableArrayRef<FillPlacement> fills,
                       tt::CoarseSchedule &schedule,
                       ArrayRef<Clusters> groupClusters) {
  auto isGathered = [](const StreamBuffer &buffer) {
    return buffer.reads.size() >= 2 &&
           llvm::any_of(buffer.reads,
                        [](ReadPlacement read) { return read.prefetched; });
  };
  int numGroups = groupClusters.size();
  int lastCurrentRead = -1;
  int firstNextRead = numGroups;
  for (const StreamBuffer &buffer : buffers) {
    if (!isGathered(buffer))
      continue;
    for (ReadPlacement read : buffer.reads) {
      if (read.prefetched)
        firstNextRead = std::min(firstNextRead, read.group);
      else
        lastCurrentRead = std::max(lastCurrentRead, read.group);
    }
  }
  if (firstNextRead == 0 || firstNextRead == numGroups ||
      lastCurrentRead >= firstNextRead)
    return;

  const Clusters &waitGroup = groupClusters[firstNextRead - 1];
  tt::CoarseSchedule::Cluster waitCluster = waitGroup[0];
  for (tt::CoarseSchedule::Cluster cluster : waitGroup)
    if (schedule.clusters.isBefore(waitCluster, cluster))
      waitCluster = cluster;
  for (auto [buffer, fill] : llvm::zip(buffers, fills)) {
    if (!isGathered(buffer))
      continue;
    assert(fill.stages[SCHED_ASYNC_WAIT] != fill.stages[SCHED_COMPUTE] &&
           "a gathered wait must run a stage ahead of the current-K reads");
    fill.clusters[SCHED_ASYNC_WAIT] = waitCluster;
    fill.clusters[SCHED_GLOBAL_LOAD] =
        groupClusters[firstNextRead][SCHED_GLOBAL_LOAD];
  }
}

void scheduleStreamOps(const LoadToStreamOpMap &loadToStreamOp,
                       tt::CoarseSchedule &schedule, const Stages &stages,
                       ArrayRef<Clusters> groupClusters,
                       ArrayRef<Operation *> dots) {
  // Interleaving is only set up for loops that LDS prefetch, which always
  // leaves a stage between the local load and compute for a chain to be
  // dropped into.
  bool interleaved = !dots.empty();
  assert((!interleaved || stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE]) &&
         "interleaved dots require room below the compute stage");

  // A loop without interleavable dots has a single group; every chain is read
  // in it and keeps the full prefetch.
  SmallVector<ReadPlacement> reads =
      interleaved
          ? assignReadPlacements(loadToStreamOp, dots, groupClusters.size())
          : SmallVector<ReadPlacement>(loadToStreamOp.size(),
                                       ReadPlacement{0, true});

  // Group chains by the op that fills their buffer, so the subtile loads of
  // one fused whole-tile copy become one buffer with all of their reads.
  SmallVector<StreamBuffer> buffers;
  DenseMap<Operation *, unsigned> bufferIndex;
  for (auto [chain, read] : llvm::zip(loadToStreamOp, reads)) {
    auto &[l, streamOps] = chain;
    auto [it, inserted] =
        bufferIndex.try_emplace(getChainFillOp(streamOps), buffers.size());
    if (inserted)
      buffers.push_back({l, streamOps, {}});
    buffers[it->second].reads.push_back(read);
  }

  SmallVector<FillPlacement> fills;
  for (const StreamBuffer &buffer : buffers)
    fills.push_back(placeFill(buffer, stages, groupClusters));
  if (interleaved)
    gatherSharedFills(buffers, fills, schedule, groupClusters);

  for (auto [buffer, fill] : llvm::zip(buffers, fills)) {
    int copyStage = schedule[buffer.origLoad].first;
    if (auto ops = std::get_if<TDMChainOps>(&buffer.fill))
      scheduleTDMFill(ops->tdmOp, ops->waitOp, copyStage, schedule, fill.stages,
                      fill.clusters);
    else if (auto ops = std::get_if<AsyncCopyChainOps>(&buffer.fill))
      scheduleAsyncCopyFill(*ops, copyStage, schedule, fill.stages,
                            fill.clusters);
    else if (auto ops = std::get_if<StreamCopyChainOps>(&buffer.fill))
      scheduleStreamCopyFill(*ops, cast<tt::LoadOp>(buffer.origLoad), schedule,
                             fill.stages, fill.clusters);
  }

  for (auto [chain, read] : llvm::zip(loadToStreamOp, reads))
    scheduleRead(getChainLocalLoad(chain.second), read, schedule, stages,
                 groupClusters);
}

void updateSchedule(scf::ForOp &forOp, const LoadToInfoMap &loadToInfo,
                    tt::CoarseSchedule &schedule,
                    triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                    bool useAsyncCopy, bool hasTDMLoad, bool waitAtTail) {
  LDBG("SingleDotSchedule::updateSchedule");
  Stages stages;
  SmallVector<Clusters> groupClusters;

  int maxDist = 0;
  for (auto &[l, info] : loadToInfo) {
    maxDist = std::max(maxDist, info.distToUse);
  }

  bool hasScaledDot = false;
  forOp.walk([&](tt::DotScaledOp) { hasScaledDot = true; });

  int numBuffers = 1;
  SmallVector<Operation *> dots;
  if (failed(initSchedule(forOp, maxDist, stages, schedule.getNumStages(),
                          numBuffers, loadToInfo.size(), useAsyncCopy,
                          hasTDMLoad, hasScaledDot, waitAtTail, dots,
                          groupClusters, schedule)))
    return;

  // A sliced kernel issues one descriptor load per slice of an operand tile.
  // Fuse each group of slices that form one larger tile into a single TDM
  // copy of that tile: one copy and one wait, with each slice read back
  // through memdesc_subslice next to the dot that uses it.
  SmallVector<SubtileLoadGroup> subtileGroups =
      findSubtileLoadGroups(loadToInfo);

  // Convert the loads into shared memory allocations and loads from them.
  auto loadToStreamOps =
      createStreamOps(loadToInfo, forOp, numBuffers, useAsyncCopy,
                      axisInfoAnalysis, subtileGroups);

  scheduleStreamOps(loadToStreamOps, schedule, stages, groupClusters, dots);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Coarse schedule stream ops:");

  for (auto [l, _] : loadToInfo) {
    if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(l)) {
      schedule.erase(l);
      l->erase();
    }
  }

  scheduleDependencies(forOp, schedule);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Coarse schedule with dependencies:");
  ttg::scheduleDistanceOneDependencies(forOp, schedule);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Coarse schedule with dist 1:");
  tt::CoarseSchedule::Cluster computeCluster =
      groupClusters.back()[SCHED_COMPUTE];
  ttg::scheduleRemainingToLastStage(forOp, schedule, computeCluster);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Final coarse schedule:");
}
} // namespace
} // namespace SingleDotSchedule

namespace ChainedDotSchedule {
namespace {
using namespace mlir::ChainedDotSchedule;

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

void updateSchedule(scf::ForOp &forOp, const LoadToInfoMap &loadToInfo,
                    tt::CoarseSchedule &schedule,
                    triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                    bool useAsyncCopy) {
  LDBG("ChainedDotSchedule::updateSchedule");
  ChainedDotClusters clusters;
  int cnt = clusters.size() - schedule.clusters.size();
  for (int i = 0; i < cnt; i++) {
    schedule.clusters.newAtBack();
  }
  auto it = schedule.clusters.begin();
  for (int i = 0; i < clusters.size(); i++, it++) {
    clusters[i] = it;
  }

  // Convert the loads into shared memory allocations and loads from them.
  // TODO support different numBuffers
  int numBuffers = useAsyncCopy ? 2 : 1;
  auto loadToStreamOps = createStreamOps(loadToInfo, forOp, numBuffers,
                                         useAsyncCopy, axisInfoAnalysis);
  scheduleStreamOps(loadToStreamOps, schedule, clusters);

  // Now that the original load is replaced by a sequence of new Ops. It become
  // dead, and hence should be removed from the schedule and can be erased.
  for (auto [l, _] : loadToStreamOps) {
    schedule.erase(l);
    l->erase();
  }

  scheduleDependencies(forOp, schedule);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Coarse schedule with dependencies:");

  triton::gpu::scheduleDistanceOneDependencies(forOp, schedule);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Coarse schedule with dist 1:");

  tt::CoarseSchedule::Cluster lastCluster = clusters.back();
  triton::gpu::scheduleRemainingToLastStage(forOp, schedule, lastCluster);
  dumpScheduleDebug(schedule, DEBUG_TYPE, "Final coarse schedule:");
}
} // namespace
} // namespace ChainedDotSchedule

static void lowerLoop(scf::ForOp forOp,
                      triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                      bool useAsyncCopy, bool usePingpong) {
  tt::CoarseSchedule schedule;
  if (failed(schedule.deSerialize(forOp, /*normalizeClusterId=*/false))) {
    return;
  }

  dumpScheduleDebug(schedule, DEBUG_TYPE, "[lowerLoops]deserialized schedule:");

  int numStages = schedule.getNumStages();

  // i.e., we can still disable `waitAtTail` by explicitly disabling
  // pingpong, which is the only use case of this scheduling variant.
  bool waitAtTail = usePingpong && (numStages == 3) && useAsyncCopy;

  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel =
      getIndirectLevel(axisInfoAnalysis, forOp, numStages);

  TargetFeatures targetFeatures(getAMDArch(forOp->getParentOfType<ModuleOp>()));
  triton::AMD::TargetInfo targetInfo(
      getAMDArch(forOp->getParentOfType<ModuleOp>()));

  bool hasTDMLoad = false;
  LoadToInfoMap loadToInfo;
  for (const auto &[load, info] : loadOpToIndLevel) {
    auto [distance, use] = info;
    if (load->hasAttrOfType<BoolAttr>(AttrBypassLDS)) {
      load->removeAttr(AttrBypassLDS);
      loadToInfo[load] = {nullptr, distance, use};
    } else {
      if (auto descLoadLike =
              dyn_cast<tt::DescriptorLoadLikeOpInterface>(load)) {
        hasTDMLoad = true;
        auto resultType = cast<RankedTensorType>(load->getResult(0).getType());
        auto paddedEncoding =
            getEncodingFromDescriptor(load, resultType, descLoadLike.getDesc());
        LoadInfo ldInfo;
        ldInfo.sharedEncoding = paddedEncoding;
        ldInfo.distToUse = distance;
        ldInfo.use = use;
        loadToInfo[load] = ldInfo;
      } else {
        LDBG("Deduce shared encoding for: " << *load);
        auto sharedEncoding = getSharedEncIfAllUsersAreDotEnc(
                                  load, axisInfoAnalysis, targetFeatures,
                                  targetInfo, useAsyncCopy)
                                  .value_or(nullptr);
        loadToInfo[load] = {sharedEncoding, distance, use};
        LDBG("Populate loadInfo with shared encoding: " << sharedEncoding);
      }
    }
  }

  if (succeeded(mlir::ChainedDotSchedule::checkPreconditions(forOp, numStages,
                                                             loadToInfo))) {
    ChainedDotSchedule::updateSchedule(forOp, loadToInfo, schedule,
                                       axisInfoAnalysis, useAsyncCopy);
  } else {
    SingleDotSchedule::updateSchedule(forOp, loadToInfo, schedule,
                                      axisInfoAnalysis, useAsyncCopy,
                                      hasTDMLoad, waitAtTail);
  }

  dumpScheduleDebug(schedule, DEBUG_TYPE, "[lowerLoops]updated schedule:");

  schedule.serialize(forOp);
}

void lowerLoops(ModuleOp moduleOp, bool useAsyncCopy, bool usePingpong) {
  triton::AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    lowerLoop(forOp, axisInfoAnalysis, useAsyncCopy, usePingpong);
  }
}

} // namespace mlir
