#include "triton/Dialect/TritonGPU/IR/Partitioning.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>

namespace mlir::triton::gpu {

SetVector<int> getPartitionIds(Operation *op) {
  auto attrs = op->getAttr(kPartitionAttrName);
  SmallVector<int> partitionIds;
  for (auto id : cast<DenseI32ArrayAttr>(attrs).asArrayRef())
    partitionIds.push_back(id);
  llvm::sort(partitionIds);
  return SetVector<int>(partitionIds.begin(), partitionIds.end());
}

SmallVector<SetVector<int>, 4> getPartitionOutputs(Operation *op) {
  SmallVector<SetVector<int>, 4> partitionOutputsIds;
  if (op->getNumResults() == 0)
    return partitionOutputsIds;

  assert(op->hasAttr(kPartitionOutputsAttrName));
  auto arrayAttr = cast<ArrayAttr>(op->getAttr(kPartitionOutputsAttrName));
  for (Attribute attr : arrayAttr) {
    auto ids = cast<DenseI32ArrayAttr>(attr).asArrayRef();
    partitionOutputsIds.push_back(SetVector<int>(ids.begin(), ids.end()));
  }
  return partitionOutputsIds;
}

bool hasPartition(Operation *op) {
  return op && op->hasAttr(kPartitionAttrName);
}

} // namespace mlir::triton::gpu
