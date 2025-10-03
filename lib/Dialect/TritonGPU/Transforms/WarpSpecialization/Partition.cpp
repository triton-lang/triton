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
// Partition
//===----------------------------------------------------------------------===//

bool Partition::hasOp(Operation *op) const {
  auto partitionIds = getPartitionIds(op);
  if (!partitionIds) {
    return false;
  }
  return partitionIds->contains(getIndex());
}

void Partition::iterateInputs(scf::ForOp loop,
                              function_ref<void(OpOperand &)> callback) const {
  for (Operation *op : getOps()) {
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
                 !llvm::is_contained(*partitionIds, getIndex())) {
        // This value originates from a different partition in the same
        // iteration.
        assert(value.getDefiningOp()->getParentOp() == loop);
        callback(operand);
      }
    });
  }
}

void Partition::iterateOutputs(
    scf::ForOp loop,
    function_ref<void(Operation *, OpOperand &)> callback) const {
  for (Operation *op : getOps()) {
    for (OpOperand &use : op->getUses()) {
      Operation *owner = loop.getBody()->findAncestorOpInBlock(*use.getOwner());
      auto partitionIds = getPartitionIds(owner);
      if (isa<scf::YieldOp>(owner)) {
        // This value is used in a subsequent iteration.
        callback(owner, use);
      } else if (!partitionIds ||
                 !llvm::is_contained(*partitionIds, getIndex())) {
        // This value is used in a different partition in the same iteration.
        callback(owner, use);
      }
    }
  }
}

void Partition::iterateDefs(
    scf::ForOp loop, function_ref<void(OpResult, unsigned)> callback) const {
  iterateInputs(loop, [&](OpOperand &input) {
    auto [def, distance] = getDefinitionAndDistance(loop, input.get());
    if (def && def.getParentBlock() == loop.getBody())
      callback(def, distance);
  });
}

void Partition::iterateUses(
    scf::ForOp loop,
    function_ref<void(OpResult, OpOperand &, unsigned)> callback) const {
  SmallVector<std::tuple<OpResult, OpOperand *, unsigned>> uses;
  iterateOutputs(loop, [&](Operation *owner, OpOperand &use) {
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

Partition *PartitionSet::getPartition(Operation *op) {
  auto id = getPartitionIds(op);
  assert(id && id->size() == 1);
  return getPartition((*id)[0]);
}

bool PartitionSet::isInRootPartition(Operation *op) {
  auto partitionIds = getPartitionIds(op);
  return !partitionIds || partitionIds->size() == getNumPartitions();
}

FailureOr<PartitionSet> PartitionSet::fromLoop(scf::ForOp loop) {
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
  SmallVector<int> partitions{partition->getIndex()};
  setPartition(op, partitions);
  partition->addOp(op);
}

void setPartition(Operation *op, const SetVector<Partition *> &partitions) {
  SmallVector<int> partitionIds;
  for (auto partition : partitions) {
    partitionIds.push_back(partition->getIndex());
    partition->addOp(op);
  }
  setPartition(op, partitionIds);
}

std::optional<SetVector<int>> getPartitionIds(Operation *op) {
  if (!op) {
    return std::nullopt;
  }
  auto attrs = op->getAttr(kPartitionAttrName);
  if (!attrs) {
    return std::nullopt;
  }

  assert(isa<DenseI32ArrayAttr>(attrs));

  SetVector<int> partitionIds;
  for (auto id : cast<DenseI32ArrayAttr>(attrs).asArrayRef()) {
    partitionIds.insert(id);
  }
  return partitionIds;
}

bool hasPartition(Operation *op) { return getPartitionIds(op) != std::nullopt; }

} // namespace mlir::triton::gpu
