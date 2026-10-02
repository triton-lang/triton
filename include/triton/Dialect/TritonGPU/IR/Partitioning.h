#ifndef TRITON_DIALECT_TRITONGPU_IR_PARTITIONING_H_
#define TRITON_DIALECT_TRITONGPU_IR_PARTITIONING_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::triton::gpu {

inline constexpr char kPartitionAttrName[] = "ttg.partition";
inline constexpr char kPartitionOutputsAttrName[] = "ttg.partition.outputs";

SetVector<int> getPartitionIds(Operation *op);
SmallVector<SetVector<int>, 4> getPartitionOutputs(Operation *op);
bool hasPartition(Operation *op);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_IR_PARTITIONING_H_
