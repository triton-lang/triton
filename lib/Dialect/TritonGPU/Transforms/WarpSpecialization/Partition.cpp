#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Use.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

namespace mlir::triton::gpu {

bool setPartition(Operation *op, Partition *partition) {
  if (op->getAttr(kPartitionAttrName)) {
    // TODO: what to do in this case
  }

  Builder b(op->getContext());
  SmallVector<Attribute, 4> attrs{b.getI32IntegerAttr(partition->getIndex())};
  op->setAttr(kPartitionAttrName, ArrayAttr::get(op->getContext(), attrs));
  partition->addOp(op);
  return true;
}

std::optional<SetVector<int>> getPartitionIds(Operation *op) {
  if (!op) {
    return std::nullopt;
  }
  auto attrs = op->getAttr(kPartitionAttrName);
  if (!attrs) {
    return std::nullopt;
  }
  SetVector<int> partitionIds;
  for (auto attr : cast<ArrayAttr>(attrs)) {
    partitionIds.insert(cast<IntegerAttr>(attr).getInt());
  }
  return partitionIds;
}

bool hasPartition(Operation *op) { return getPartitionIds(op) != std::nullopt; }

bool isOpInPartition(Operation *op, const Partition *partition) {
  auto partitionIds = getPartitionIds(op);
  if (!partitionIds) {
    return false;
  }
  return partitionIds->contains(partition->getIndex());
}
} // namespace mlir::triton::gpu

//===----------------------------------------------------------------------===//
// WarpSchedule
//===----------------------------------------------------------------------===//

Partition *WarpSchedule::addPartition(unsigned stage) {
  partitions.push_back(std::make_unique<Partition>(partitions.size(), stage));
  return partitions.back().get();
}

Partition *WarpSchedule::getPartition(Operation *op) {
  assert(false);
  return opToPartition.lookup(op);
}
const Partition *WarpSchedule::getPartition(Operation *op) const {
  assert(false);
  return opToPartition.lookup(op);
}

Partition *WarpSchedule::getPartition(unsigned idx) {
  return partitions[idx].get();
}
const Partition *WarpSchedule::getPartition(unsigned idx) const {
  return partitions[idx].get();
}

void WarpSchedule::insert(Partition *partition, Operation *op) {
  assert(false);
  partition->ops.push_back(op);
  opToPartition[op] = partition;
}

bool WarpSchedule::isScheduled(Operation *op) const {
  assert(false);
  const Partition *partition = getPartition(op);
  return partition && partition != getRootPartition();
}

bool WarpSchedule::trySchedule(Partition *partition, Operation *op) {
  assert(false);
  if (isScheduled(op))
    return false;
  insert(partition, op);
  return true;
}

FailureOr<WarpSchedule> WarpSchedule::deserialize(scf::ForOp loop) {
  auto stages = loop->getAttrOfType<ArrayAttr>(kPartitionStagesAttrName);
  if (!stages)
    return failure();

  auto tag = loop->getAttrOfType<IntegerAttr>(kWarpSpecializeTagAttrName);
  if (!tag)
    return failure();

  WarpSchedule result;
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

void WarpSchedule::serialize(scf::ForOp loop) const {
  assert(false);
  SmallVector<Attribute> stages;
  Builder b(loop.getContext());
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (Partition *partition = opToPartition.lookup(&op)) {
      if (partition == getRootPartition())
        continue;
      op.setAttr(kPartitionAttrName,
                 b.getI32IntegerAttr(partition->getIndex()));
    }
  }
  for (Partition &partition : getPartitions())
    stages.push_back(b.getI32IntegerAttr(partition.getStage()));
  loop->setAttr(kPartitionStagesAttrName, b.getArrayAttr(stages));
}

LogicalResult WarpSchedule::verify(scf::ForOp loop) const {
  assert(false);
  // The root partition is only allowed to transitively depend on itself.
  bool failed = false;
  iterateInputs(loop, getRootPartition(), [&](OpOperand &input) {
    auto [def, distance] = getDefiningOpAndDistance(loop, input.get());
    // Ignore values defined outside the loop.
    if (!def || def->getParentOp() != loop)
      return;
    const Partition *defPartition = opToPartition.at(def);
    if (defPartition == getRootPartition())
      return;
    InFlightDiagnostic diag = mlir::emitWarning(input.getOwner()->getLoc());
    diag << "operation in the root partition depends on a value that "
            "originates from a non-root partition through operand #"
         << input.getOperandNumber();
    diag.attachNote(def->getLoc())
        << "operand defined here in partition #" << defPartition->getIndex()
        << " at distance " << distance;
    failed = true;
  });
  if (failed)
    return failure();

  return success();
}

void WarpSchedule::eraseFrom(scf::ForOp loop) {
  loop.walk([&](Operation *op) { op->removeAttr(kPartitionAttrName); });
  loop->removeAttr(kPartitionStagesAttrName);
}

void WarpSchedule::iterateInputs(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(OpOperand &)> callback) const {
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

void WarpSchedule::iterateOutputs(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(Operation *, OpOperand &)> callback) const {
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

void WarpSchedule::iterateDefs(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(OpResult, unsigned)> callback) const {
  iterateInputs(loop, partition, [&](OpOperand &input) {
    auto [def, distance] = getDefinitionAndDistance(loop, input.get());
    if (def && def.getParentBlock() == loop.getBody())
      callback(def, distance);
  });
}

void WarpSchedule::iterateUses(
    scf::ForOp loop, const Partition *partition,
    function_ref<void(OpResult, OpOperand &, unsigned)> callback) const {
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

void WarpSchedule::dump() const {
  for (auto [i, partition] :
       llvm::enumerate(llvm::make_pointee_range(partitions))) {
    llvm::errs() << "=== PARTITION #" << i << " ===\n";
    for (Operation *op : partition.getOps()) {
      op->print(llvm::errs(), OpPrintingFlags().skipRegions());
      llvm::errs() << "\n";
    }
    llvm::errs() << "\n";
  }
  llvm::errs() << "=== ROOT PARTITION ===\n";
  for (Operation *op : getRootPartition()->getOps()) {
    op->print(llvm::errs(), OpPrintingFlags().skipRegions());
    llvm::errs() << "\n";
  }
  llvm::errs() << "\n";
}
