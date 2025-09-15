#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PIPELINE_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PIPELINE_H_

#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include <array>

namespace mlir {
struct LoadInfo {
  // Shared layout is used for loads feeding into dot ops.
  ttg::SwizzledSharedEncodingAttr sharedEncoding = nullptr;
  // The distance of this load's stage to its use' stage.
  int distToUse = 0;
  Operation *use = nullptr;
};
using LoadToInfoMap = llvm::MapVector<Operation *, LoadInfo>;

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

LogicalResult createStages(Stages &stages, int &numBuffers, int maxDist,
                           int numStages, int globalPrefetch, int localPrefetch,
                           bool waitAtTail);
LogicalResult createClusters(Clusters &clusters, tt::CoarseSchedule &schedule,
                             const Stages &stages, int maxDist, int numBuffers,
                             bool waitAtTail);
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
                                 LoadToInfoMap loadToInfo);
} // namespace ChainedDotSchedule
} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PIPELINE_H_
