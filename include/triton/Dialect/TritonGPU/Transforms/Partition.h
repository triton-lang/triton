#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_PARTITION_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_PARTITION_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
class Region;
class Value;
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

static constexpr char kPartitionAttrName[] = "ttg.partition";
static constexpr char kLatenciesAttrName[] = "ttg.partition.latencies";

//===----------------------------------------------------------------------===//
// WarpSchedule
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
// A warp schedule divides a loop into multiple partitions. Ops in a loop are
// assigned at most one partition. A warp schedule represents asynchronous
// execution of the loop body, where partitions may execute simultaneously.
class WarpSchedule {
  static constexpr int kSentinelLatency = -1;

public:
  // A partition has a latency and contains some operation. The latency of a
  // partition determines how many cycles the partition's outputs are buffered.
  class Partition {
  public:
    Partition(int latency) : latency(latency) {}

  private:
    // The latency of the partition.
    int latency;
    // The ops in the partition.
    SmallVector<Operation *> ops;

    friend class WarpSchedule;
  };

  // Deserialize a warp schedule from an `scf.for` op using the attributes
  // tagged on operations in its body.
  static FailureOr<WarpSchedule> deserialize(scf::ForOp loop);
  // Serialize a warp schedule by writing the partition latencies and mappings
  // as attributes on operations in the loop.
  void serialize(scf::ForOp loop) const;
  // Verify that the warp schedule is valid by checking the SSA dependencies
  // between the schedules.
  LogicalResult verify(scf::ForOp loop) const;

  // Iterate the inputs of the partition. Input values are those that originate
  // from a different partition or a previous iteration of the current
  // partition. E.g. partition B(i) may have inputs from A(i) or B(i-1). Note
  // that the same value may be visited more than once.
  void iterateInputs(scf::ForOp loop, Partition *partition,
                     function_ref<void(Value)> callback) const;
  // Iterate the outputs of the partition. Output values are those that are
  // consumed by a different partition or a future iteration of the current
  // partition. E.g. partition A(i) may have outputs to B(i) or A(i+1). Note
  // that the same value may be visited more than once.
  void iterateOutputs(scf::ForOp loop, Partition *partition,
                      function_ref<void(Value)> callback) const;

private:
  // Partitions are numbered [0, N).
  SmallVector<std::unique_ptr<Partition>> partitions;
  // A mapping from operation to its partition.
  DenseMap<Operation *, Partition *> opToPartition;
  // The null partition contains operations that are not assigned to a
  // partition. Operations not assigned to partitions are assumed to be "free"
  // and can be cloned as necessary.
  Partition nullPartition = Partition(kSentinelLatency);

  friend class Partition;
};

} // namespace mlir::triton::gpu

#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_PARTITION_H_
