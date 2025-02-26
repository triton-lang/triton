#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

using namespace mlir;
using namespace triton;
using namespace gpu;

//===----------------------------------------------------------------------===//
// WarpSchedule
//===----------------------------------------------------------------------===//

FailureOr<WarpSchedule> WarpSchedule::deserialize(scf::ForOp loop) {
  auto latencies = loop->getAttrOfType<ArrayAttr>(kLatenciesAttrName);
  if (!latencies) {
    return mlir::emitError(loop.getLoc(), "missing '")
           << kLatenciesAttrName << "' attribute";
  }

  WarpSchedule result;
  for (Attribute attr : latencies) {
    auto latency = dyn_cast<IntegerAttr>(attr);
    if (!latency || latency.getInt() <= 0) {
      return mlir::emitError(loop.getLoc(), "latencies '")
             << kLatenciesAttrName << "' has invalid element " << attr;
    }

    result.partitions.push_back(std::make_unique<Partition>(latency.getInt()));
  }

  for (Operation &op : loop.getBody()->without_terminator()) {
    Partition *partition = &result.nullPartition;
    if (auto attr = op.getAttrOfType<IntegerAttr>(kPartitionAttrName)) {
      int64_t idx = attr.getInt();
      if (idx < 0 || idx >= result.partitions.size()) {
        return mlir::emitError(op.getLoc(), "invalid partition index ") << idx;
      }
      partition = result.partitions[idx].get();
    }

    partition->ops.push_back(&op);
    result.opToPartition[&op] = partition;
  }

  return result;
}

void WarpSchedule::serialize(scf::ForOp loop) const {
  SmallVector<Attribute> latencies;
  Builder b(loop.getContext());
  for (auto [i, partition] :
       llvm::enumerate(llvm::make_pointee_range(partitions))) {
    latencies.push_back(b.getI32IntegerAttr(partition.latency));
    for (Operation *op : partition.ops) {
      op->setAttr(kPartitionAttrName, b.getI32IntegerAttr(i));
    }
  }
  loop->setAttr(kLatenciesAttrName, b.getArrayAttr(latencies));
}

LogicalResult WarpSchedule::verify(scf::ForOp loop) const {
  // The null partition is only allowed to depend on itself.

  for (auto [i, partition] :
       llvm::enumerate(llvm::make_pointee_range(partitions))) {
    for (Operation *op : partition.ops) {
    }
  }
  return success();
}

void WarpSchedule::iterateInputs(scf::ForOp loop, Partition *partition,
                                 function_ref<void(Value)> callback) const {
  for (Operation *op : partition->ops) {
    visitNestedOperands(op, [&](Value operand) {
      // Ignore implicit captures.
      if (operand.getParentBlock() != loop.getBody())
        return;
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        assert(arg.getOwner() == loop.getBody());
        // Ignore the induction variable.
        if (arg == loop.getInductionVar())
          return;
        // This value originates from a previous iteration.
        assert(llvm::is_contained(loop.getRegionIterArgs(), arg));
        callback(arg);
      } else if (opToPartition.at(operand.getDefiningOp()) != partition) {
        // This value originates from a different partition in the same
        // iteration.
        assert(operand.getDefiningOp()->getParentOp() == loop);
        callback(operand);
      }
    });
  }
}
