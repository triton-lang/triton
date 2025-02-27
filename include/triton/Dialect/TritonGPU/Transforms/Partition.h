#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_PARTITION_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_PARTITION_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
class OpOperand;
class OpResult;
class Region;
class Value;
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

static constexpr char kPartitionAttrName[] = "ttg.partition";
static constexpr char kPartitionStagesAttrName[] = "ttg.partition.stages";

//===----------------------------------------------------------------------===//
// WarpSchedule
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
struct PartitionGraph;

// A warp schedule divides a loop into multiple partitions. Ops in a loop are
// assigned at most one partition. A warp schedule represents asynchronous
// execution of the loop body, where partitions may execute simultaneously.
class WarpSchedule {
  static constexpr int kSentinel = -1;

public:
  // A partition has a stage and contains some operation. The stage of a
  // partition determines how many cycles the partition's outputs are buffered
  // relative to its consumers.
  class Partition {
  public:
    Partition(int idx, int stage) : idx(idx), stage(stage) {}

    int getIndex() const { return idx; }
    int getStage() const { return stage; }
    ArrayRef<Operation *> getOps() const { return ops; }

    void insert(Operation *op) { ops.push_back(op); }

  private:
    // The partition number.
    int idx;
    // The stage of the partition.
    int stage;
    // The ops in the partition.
    SmallVector<Operation *> ops;
  };

  // Get the partition the op belongs to.
  Partition *getPartition(Operation *op);
  // Get the partition the op belongs to.
  const Partition *getPartition(Operation *op) const;
  // Return an iterator range over the partitions.
  auto getPartitions() { return llvm::make_pointee_range(partitions); }
  // Return an iterator range over the partitions.
  auto getPartitions() const { return llvm::make_pointee_range(partitions); }
  // Get the root partition.
  Partition *getRootPartition() { return &rootPartition; }
  // Get the root partition.
  const Partition *getRootPartition() const { return &rootPartition; }

  // Deserialize a warp schedule from an `scf.for` op using the attributes
  // tagged on operations in its body.
  static FailureOr<WarpSchedule> deserialize(scf::ForOp loop);
  // Serialize a warp schedule by writing the partition stage and mappings
  // as attributes on operations in the loop.
  void serialize(scf::ForOp loop) const;
  // Verify that the warp schedule is valid by checking the SSA dependencies
  // between the schedules.
  FailureOr<PartitionGraph> verify(scf::ForOp loop) const;

  // Iterate the inputs of the partition. Input values are those that originate
  // from a different partition or a previous iteration of the current
  // partition. E.g. partition B(i) may have inputs from A(i) or B(i-1). Note
  // that the same value may be visited more than once.
  void iterateInputs(scf::ForOp loop, const Partition *partition,
                     function_ref<void(OpOperand &)> callback) const;
  // Iterate the outputs of the partition. Output values are those that are
  // consumed by a different partition or a future iteration of the current
  // partition. E.g. partition A(i) may have outputs to B(i) or A(i+1). Note
  // that the same value may be visited more than once.
  void
  iterateOutputs(scf::ForOp loop, const Partition *partition,
                 function_ref<void(Operation *, OpOperand &)> callback) const;
  // Iterate the defining ops of the inputs to the partition in the current and
  // previous iterations, including the distance in the past.
  void iterateDefs(scf::ForOp loop, const Partition *partition,
                   function_ref<void(OpResult, unsigned)> callback) const;
  // Iterate the uses of all outputs of the partition in the current iteration
  // and in future iterations, including the distance in the future.
  void iterateUses(
      scf::ForOp loop, const Partition *partition,
      function_ref<void(OpResult, OpOperand &, unsigned)> callback) const;

private:
  // Partitions are numbered [0, N).
  SmallVector<std::unique_ptr<Partition>> partitions;
  // A mapping from operation to its partition.
  DenseMap<Operation *, Partition *> opToPartition;
  // The root partition contains operations that are not assigned to a
  // partition. Operations not assigned to partitions are assumed to be "free"
  // and can be cloned as necessary.
  Partition rootPartition = Partition(kSentinel, kSentinel);
};

//===----------------------------------------------------------------------===//
// PartitionGraph
//===----------------------------------------------------------------------===//

// A temporary node structure that can be used to build a graph of partitions.
// The consumers have to be precomputed in order for the SCC iterator to have an
// acceptable runtime complexity. This assumes the underlying loop is immutable.
struct PartitionNode {
  PartitionNode(const WarpSchedule::Partition *partition)
      : partition(partition) {}

  // The partition this node represents.
  const WarpSchedule::Partition *partition;
  // Partitions that consume the outputs of this partition.
  SmallVector<std::pair<PartitionNode *, OpOperand *>> consumers;
};

// A graph of partitions that can be used to check for cycles and other schedule
// invariants.
struct PartitionGraph {
  PartitionGraph(scf::ForOp loop, const WarpSchedule &schedule);

  PartitionNode root;
  llvm::MapVector<const WarpSchedule::Partition *, PartitionNode> nodes;
};

} // namespace mlir::triton::gpu

namespace llvm {
template <> struct GraphTraits<mlir::triton::gpu::PartitionGraph *> {
  using NodeRef =
      std::pair<mlir::triton::gpu::PartitionNode *, mlir::OpOperand *>;
  static NodeRef getEntryNode(mlir::triton::gpu::PartitionGraph *graph) {
    return {&graph->root, nullptr};
  }

  using ChildIteratorType = SmallVector<NodeRef>::iterator;
  static ChildIteratorType child_begin(NodeRef node) {
    return node.first->consumers.begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node.first->consumers.end();
  }
};
} // namespace llvm

#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_PARTITION_H_
