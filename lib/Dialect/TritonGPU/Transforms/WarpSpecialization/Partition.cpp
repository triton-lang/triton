#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Use.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

//===----------------------------------------------------------------------===//
// PartitionSet
//===----------------------------------------------------------------------===//

Partition *PartitionSet::addPartition(unsigned stage) {
  partitions.push_back(std::make_unique<Partition>(partitions.size(), stage));
  return partitions.back().get();
}

Partition *PartitionSet::getPartition(unsigned idx) {
  return partitions[idx].get();
}
const Partition *PartitionSet::getPartition(unsigned idx) const {
  return partitions[idx].get();
}

FailureOr<PartitionSet> PartitionSet::deserialize(scf::ForOp loop) {
  auto stages = loop->getAttrOfType<ArrayAttr>(kPartitionStagesAttrName);
  if (!stages)
    return failure();

  auto tag = loop->getAttrOfType<IntegerAttr>(kWarpSpecializeTagAttrName);
  if (!tag)
    return failure();

  PartitionSet result;
  result.tag = tag.getInt();
  for (auto [idx, attr] : llvm::enumerate(stages)) {
    auto stage = dyn_cast<IntegerAttr>(attr);
    if (!stage || stage.getInt() < 0) {
      return mlir::emitError(loop.getLoc(), "partition stages attribute '")
             << kPartitionStagesAttrName << "' has invalid element " << attr;
    }

    result.partitions.push_back(
        std::make_unique<Partition>(idx, stage.getInt()));
  }

  for (Operation &op : loop.getBody()->without_terminator()) {
    if (auto attrs = getPartitionIds(&op)) {
      for (auto idx : *attrs) {
        if (idx < 0 || idx >= result.partitions.size())
          return mlir::emitError(op.getLoc(), "invalid partition index ")
                 << idx;
        result.partitions[idx]->addOp(&op);
      }
    }
  }

  return result;
}

void PartitionSet::dump() const {
  for (auto [i, partition] :
       llvm::enumerate(llvm::make_pointee_range(partitions))) {
    llvm::errs() << "=== PARTITION #" << i << " ===\n";
    for (Operation *op : partition.getOps()) {
      op->print(llvm::errs(), OpPrintingFlags().skipRegions());
      llvm::errs() << "\n";
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "\n";
}

namespace mlir::triton::gpu {

void setPartition(Operation *op, ArrayRef<int> partitionIds) {
  Builder b(op->getContext());
  op->setAttr(kPartitionAttrName, b.getDenseI32ArrayAttr(partitionIds));
}

void setPartition(Operation *op, const SetVector<int> &partitionIds) {
  SmallVector<int> partitions(partitionIds.begin(), partitionIds.end());
  setPartition(op, partitions);
}

void setPartition(Operation *op, Partition *partition) {
  if (op->getAttr(kPartitionAttrName)) {
    // Allow overwriting in this case
    // TODO: is this the right thing to do
  }

  SmallVector<int> partitions{partition->getIndex()};
  setPartition(op, partitions);
  partition->addOp(op);
}

std::optional<SetVector<int>> getPartitionIds(Operation *op) {
  if (!op) {
    return std::nullopt;
  }
  auto attrs = op->getAttr(kPartitionAttrName);
  if (!attrs) {
    return std::nullopt;
  }
  if (!isa<DenseI32ArrayAttr>(attrs)) {
    op->dump();
  }
  SetVector<int> partitionIds;
  for (auto id : cast<DenseI32ArrayAttr>(attrs).asArrayRef()) {
    partitionIds.insert(id);
  }
  return partitionIds;
}

bool hasPartition(Operation *op) { return getPartitionIds(op) != std::nullopt; }

void iterateInputs(scf::ForOp loop, const Partition *partition,
                   function_ref<void(OpOperand &)> callback) {
  for (Operation *op : partition->getOps()) {
    visitNestedOperands(op, [&](OpOperand &operand) {
      // Ignore implicit captures.
      Value value = operand.get();
      auto partitionIds = getPartitionIds(value.getDefiningOp());
      if (value.getParentBlock() != loop.getBody())
        return;
      if (auto arg = dyn_cast<BlockArgument>(value)) {
        assert(arg.getOwner() == loop.getBody());
        // Ignore the induction variable.
        if (arg == loop.getInductionVar())
          return;
        // This value originates from a previous iteration.
        assert(llvm::is_contained(loop.getRegionIterArgs(), arg));
        callback(operand);
      } else if (!partitionIds ||
                 !llvm::is_contained(*partitionIds, partition->getIndex())) {
        // This value originates from a different partition in the same
        // iteration.
        assert(value.getDefiningOp()->getParentOp() == loop);
        callback(operand);
      }
    });
  }
}

void iterateOutputs(scf::ForOp loop, const Partition *partition,
                    function_ref<void(Operation *, OpOperand &)> callback) {
  for (Operation *op : partition->getOps()) {
    for (OpOperand &use : op->getUses()) {
      Operation *owner = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      auto partitionIds = getPartitionIds(owner);
      if (isa<scf::YieldOp>(owner)) {
        // This value is used in a subsequent iteration.
        callback(owner, use);
      } else if (!partitionIds ||
                 !llvm::is_contained(*partitionIds, partition->getIndex())) {
        // This value is used in a different partition in the same iteration.
        callback(owner, use);
      }
    }
  }
}

void iterateDefs(scf::ForOp loop, const Partition *partition,
                 function_ref<void(OpResult, unsigned)> callback) {
  iterateInputs(loop, partition, [&](OpOperand &input) {
    auto [def, distance] = getDefinitionAndDistance(loop, input.get());
    if (def && def.getParentBlock() == loop.getBody())
      callback(def, distance);
  });
}

void iterateUses(scf::ForOp loop, const Partition *partition,
                 function_ref<void(OpResult, OpOperand &, unsigned)> callback) {
  SmallVector<std::tuple<OpResult, OpOperand *, unsigned>> uses;
  iterateOutputs(loop, partition, [&](Operation *owner, OpOperand &use) {
    uses.emplace_back(cast<OpResult>(use.get()), &use, 0);
  });
  while (!uses.empty()) {
    auto [output, use, distance] = uses.pop_back_val();
    Operation *owner = loop.getBody()->findAncestorOpInBlock(*use->getOwner());
    if (!isa<scf::YieldOp>(owner)) {
      callback(output, *use, distance);
      continue;
    }
    BlockArgument arg = loop.getRegionIterArg(use->getOperandNumber());
    for (OpOperand &use : arg.getUses())
      uses.emplace_back(output, &use, distance + 1);
  }
}

Partition *getPartition(Operation *op, PartitionSet &partitions) {
  auto id = getPartitionIds(op);
  assert(id && id->size() == 1);
  return partitions.getPartition((*id)[0]);
}

} // namespace mlir::triton::gpu
