#ifndef TRITON_TRITONGPU_TRANSFORMS_PARTITIONBUILDER_H
#define TRITON_TRITONGPU_TRANSFORMS_PARTITIONBUILDER_H

#include "mlir/IR/ImplicitLocOpBuilder.h"

namespace mlir::triton::gpu {

class Partition;

using StageCluster = std::optional<std::pair<int, int>>;

struct PartitionBuilder : public ImplicitLocOpBuilder {
  using ImplicitLocOpBuilder::ImplicitLocOpBuilder;

  Value intCst(int value, unsigned width = 32);
  Value boolCst(bool value);

  void assignStage(Operation *op, StageCluster stageCluster);
  void assignPartition(Operation *op, Partition &partition);

  template <typename OpT, typename... Args>
  auto createInto(Partition &partition, StageCluster stageCluster,
                  Args &&...args) {
    auto op = create<OpT>(std::forward<Args>(args)...);
    assignPartition(op, partition);
    assignStage(op, stageCluster);
    return op;
  }
};

// Get the stage and cluster for an operation, if it has one assigned.
StageCluster getStageCluster(Operation *op);

} // namespace mlir::triton::gpu

#endif // TRITON_TRITONGPU_TRANSFORMS_PARTITIONBUILDER_H
