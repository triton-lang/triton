#ifndef TRITON_TRITONGPU_TRANSFORM_PIPELINE_PARTITION_H_
#define TRITON_TRITONGPU_TRANSFORM_PIPELINE_PARTITION_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
class Operation;
class OpOperand;
class OpResult;
class Region;
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

static constexpr char kPartitionAttrName[] = "ttg.partition";
static constexpr char kPartitionStagesAttrName[] = "ttg.partition.stages";
static constexpr char kWarpSpecializeTagAttrName[] = "ttg.warp_specialize.tag";

//===----------------------------------------------------------------------===//
// PartitionSet
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
// A partition has a stage and contains some operation. The stage of a
// partition determines how many cycles the partition's outputs are buffered
// relative to its consumers.
class Partition {
public:
  Partition(int idx, int stage) : idx(idx), stage(stage) {}

  int getIndex() const {
    assert(idx >= 0);
    return idx;
  }
  int getStage() const { return stage; }
  ArrayRef<Operation *> getOps() const { return ops; }
  void addOp(Operation *op) { ops.push_back(op); }

private:
  void setIndex(int idx) { this->idx = idx; }

  // The partition number.
  int idx;
  // The stage of the partition.
  int stage;
  // The ops in the partition.
  SmallVector<Operation *> ops;
};

// A partition set divides a loop into multiple partitions. Ops in a loop are
// assigned at most one partition. A partition set represents asynchronous
// execution of the loop body, where partitions may execute simultaneously.
class PartitionSet {
public:
  // Get WarpSpecialization tag
  int getTag() const { return tag; }

  // Create a new partition with a stage.
  Partition *addPartition(unsigned stage);

  // Get the partition at the index.
  Partition *getPartition(unsigned idx);
  // Get the partition at the index.
  const Partition *getPartition(unsigned idx) const;
  // Return an iterator range over the partitions.
  auto getPartitions() { return llvm::make_pointee_range(partitions); }
  // Return an iterator range over the partitions.
  auto getPartitions() const { return llvm::make_pointee_range(partitions); }
  // Get the number of partitions.
  unsigned getNumPartitions() const { return partitions.size(); }

  // Deserialize a partition set from an `scf.for` op using the attributes
  // tagged on operations in its body.
  static FailureOr<PartitionSet> deserialize(scf::ForOp loop);

  // Debug dump the partition set.
  LLVM_DUMP_METHOD void dump() const;

private:
  // WarpSpecialization tag
  int tag;
  // Partitions are numbered [0, N).
  SmallVector<std::unique_ptr<Partition>> partitions;
};

void setPartition(Operation *op, Partition *partition);
void setPartition(Operation *op, const SetVector<int> &partitionIds);
std::optional<SetVector<int>> getPartitionIds(Operation *op);
bool hasPartition(Operation *op);
// Utility to be used when the op is known to belong to one partition
Partition *getPartition(Operation *op, PartitionSet &partitions);

// Iterate the inputs of the partition. Input values are those that originate
// from a different partition or a previous iteration of the current
// partition. E.g. partition B(i) may have inputs from A(i) or B(i-1). Note
// that the same value may be visited more than once.
void iterateInputs(scf::ForOp loop, const Partition *partition,
                   function_ref<void(OpOperand &)> callback);
// Iterate the outputs of the partition. Output values are those that are
// consumed by a different partition or a future iteration of the current
// partition. E.g. partition A(i) may have outputs to B(i) or A(i+1). Note
// that the same value may be visited more than once.
void iterateOutputs(scf::ForOp loop, const Partition *partition,
                    function_ref<void(Operation *, OpOperand &)> callback);
// Iterate the defining ops of the inputs to the partition in the current and
// previous iterations, including the distance in the past.
void iterateDefs(scf::ForOp loop, const Partition *partition,
                 function_ref<void(OpResult, unsigned)> callback);
// Iterate the uses of all outputs of the partition in the current iteration
// and in future iterations, including the distance in the future.
void iterateUses(scf::ForOp loop, const Partition *partition,
                 function_ref<void(OpResult, OpOperand &, unsigned)> callback);

} // namespace mlir::triton::gpu

#endif // TRITON_TRITONGPU_TRANSFORM_PIPELINE_PARTITION_H_
