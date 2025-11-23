#ifndef TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PIPELINEUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PIPELINEUTILITY_H_

#include "mlir/IR/Operation.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"

namespace mlir {

namespace triton::AMD {
constexpr char AttrBypassLDS[] = "amdg.bypass_lds_load";
}

// This function will
// - deserialize schedule and numStages from IR.
// - calculate stages and clusters taking all factors into account, and remap
//   symbolic clusters of global load and compute ops to their real clusters.
// - create lds alloc/dealloc/load/store or async load/commit/wait ops if
//   possible.
// - schedule these new ops.
// - serialize schedule to IR for the next expandLoops function.
void lowerLoops(ModuleOp moduleOp, bool useAsyncCopy, bool usePingpong);

struct LoadInfo {
  // Shared layout is used for loads feeding into dot ops.
  triton::gpu::SharedEncodingTrait sharedEncoding = nullptr;
  // The distance of this load's stage to its use' stage.
  int distToUse = 0;
  Operation *use = nullptr;
};
using LoadToInfoMap = llvm::MapVector<Operation *, LoadInfo>;

// A slim wrapper of ttg::loadOpsToIndirectionLevel, to get the indirection
// levels and final users of load ops. For details you can check the comment of
// ttg::loadOpsToIndirectionLevel.
llvm::MapVector<Operation *, std::pair<int, Operation *>>
getIndirectLevel(triton::AMD::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                 scf::ForOp &forOp, int numStages);

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

using Clusters = std::array<triton::CoarseSchedule::Cluster, SCHED_SIZE>;
using Stages = std::array<int, SCHED_SIZE>;
} // namespace SingleDotSchedule

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
    std::array<triton::CoarseSchedule::Cluster, CLUSTER_COUNT>;

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

#endif // TRITON_THIRD_PARTY_AMD_LIB_TRITONAMDGPUTRANSFORMS_PIPELINEUTILITY_H_
