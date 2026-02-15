#ifndef TRITON_TRITONGPU_TRANSFORMS_PARTITIONBUILDER_H
#define TRITON_TRITONGPU_TRANSFORMS_PARTITIONBUILDER_H

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::gpu {

class Partition;

using StageCluster = std::optional<std::pair<int, int>>;

// Get the stage and cluster for an operation, if it has one assigned.
void setStageCluster(OpBuilder &b, Operation *op, StageCluster stageCluster);
StageCluster getStageCluster(Operation *op);

struct PartitionBuilder : public ImplicitLocOpBuilder {
  using ImplicitLocOpBuilder::ImplicitLocOpBuilder;

  Value intCst(int value, unsigned width = 32);
  Value boolCst(bool value);

  void assignPartition(Operation *op, Partition &partition);

  template <typename OpT, typename... Args>
  auto createInto(Partition &partition, StageCluster stageCluster,
                  Args &&...args) {
    auto op = create<OpT>(std::forward<Args>(args)...);
    assignPartition(op, partition);
    setStageCluster(*this, op, stageCluster);
    return op;
  }
};

template <typename OpT, typename... Args>
OpT createInto(OpBuilder &b, Location loc,
               std::optional<SetVector<int>> partitionSet,
               StageCluster stageCluster, Args &&...args) {
  auto op = OpT::create(b, loc, std::forward<Args>(args)...);
  if (partitionSet) {
    setPartition(op, *partitionSet);
    setStageCluster(b, op, stageCluster);
  }
  return op;
}

} // namespace mlir::triton::gpu

#endif // TRITON_TRITONGPU_TRANSFORMS_PARTITIONBUILDER_H
