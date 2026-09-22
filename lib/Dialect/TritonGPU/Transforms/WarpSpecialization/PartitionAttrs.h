#ifndef TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_PARTITIONATTRS_H_
#define TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_PARTITIONATTRS_H_

#include "triton/Dialect/TritonGPU/IR/Partitioning.h"
#include <optional>

namespace mlir {
class Operation;
class OpOperand;
class LoopLikeOpInterface;
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

namespace mlir::triton::gpu {

inline constexpr char kPartitionStagesAttrName[] = "ttg.partition.stages";
inline constexpr char kWarpSpecializeTagAttrName[] = "ttg.warp_specialize.tag";

SetVector<int> getPartitionIds(OpOperand *use);
bool hasWarpSpecializeTag(Operation *op);
std::optional<int> getWarpSpecializeTag(Operation *op);

LogicalResult verifyPartitionedLoop(LoopLikeOpInterface loop);

} // namespace mlir::triton::gpu

#endif // TRITON_LIB_DIALECT_TRITONGPU_TRANSFORMS_WARPSPECIALIZATION_PARTITIONATTRS_H_
