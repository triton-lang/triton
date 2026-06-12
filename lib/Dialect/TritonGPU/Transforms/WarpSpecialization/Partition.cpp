#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "PartitionAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Use.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

namespace {

LogicalResult verifyPartitionIdsAttr(Operation *op, StringRef attrName,
                                     Attribute attrValue) {
  auto partitionIdsAttr = dyn_cast<DenseI32ArrayAttr>(attrValue);
  if (!partitionIdsAttr) {
    return op->emitOpError("has invalid attribute ")
           << attrName << "; expected a dense i32 array";
  }

  SetVector<int> idSet;
  for (auto id : partitionIdsAttr.asArrayRef()) {
    if (idSet.contains(id))
      return op->emitOpError("has duplicated partition ids in attribute ")
             << attrName;
    idSet.insert(id);
  }
  if (idSet.empty())
    return op->emitOpError("has no partition ids in attribute ") << attrName;

  auto ids = idSet.takeVector();
  SmallVector<int> sortedIds(ids.begin(), ids.end());
  llvm::sort(sortedIds);
  if (ids != sortedIds) {
    return op->emitOpError("partition ids not in sorted order in attribute ")
           << attrName;
  }
  return success();
}

LogicalResult verifyPartitionAttrs(Operation *op) {
  if (op->hasAttr(kWarpSpecializeAttrName)) {
    if (!isa<scf::ForOp>(op)) {
      return op->emitOpError("has unexpected attribute ")
             << kWarpSpecializeAttrName
             << " which is expected only on `scf.for` ops";
    }

    Operation *failedOp = nullptr;
    op->walk([&](Operation *childOp) {
      if (isa<ub::PoisonOp>(childOp))
        return WalkResult::advance();
      if (!childOp->hasAttr(kPartitionAttrName)) {
        failedOp = childOp;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (failedOp) {
      return failedOp->emitOpError("does not have expected attribute ")
             << kPartitionAttrName
             << " which is expected on all child ops of an op with attribute `"
             << kWarpSpecializeAttrName << "`";
    }
  }

  if (auto partitionAttr = op->getAttr(kPartitionAttrName)) {
    if (failed(verifyPartitionIdsAttr(op, kPartitionAttrName, partitionAttr)))
      return failure();
  }

  if (auto outputsAttr = op->getAttr(kPartitionOutputsAttrName)) {
    auto arrayAttr = dyn_cast<ArrayAttr>(outputsAttr);
    if (!arrayAttr) {
      return op->emitOpError("has invalid attribute ")
             << kPartitionOutputsAttrName << "; expected an array attribute";
    }

    for (Attribute attr : arrayAttr) {
      if (failed(verifyPartitionIdsAttr(op, kPartitionOutputsAttrName, attr))) {
        return failure();
      }
    }
  }

  if (op->hasAttr(kPartitionAttrName) && op->getNumRegions() != 0) {
    SetVector<int> expectedIds;
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (Operation &childOp : block.getOperations()) {
          if (isa<scf::YieldOp, ub::PoisonOp>(childOp))
            continue;
          if (!childOp.hasAttr(kPartitionAttrName)) {
            return childOp.emitOpError("does not have expected attribute ")
                   << kPartitionAttrName
                   << " which is expected for ops whose parent has partitions";
          }
          auto ids = getPartitionIds(&childOp);
          expectedIds.insert(ids.begin(), ids.end());
        }
      }
    }

    auto partitionIds = getPartitionIds(op);
    for (auto id : expectedIds) {
      if (!partitionIds.contains(id)) {
        return op->emitOpError("partition ids in attr ")
               << kPartitionAttrName
               << " does not contain partition ids of all child ops";
      }
    }
  }

  if (auto outputsAttr = op->getAttr(kPartitionOutputsAttrName)) {
    if (!isa<scf::ForOp, scf::IfOp, triton::ReduceOp>(op))
      return op->emitOpError("has unexpected attribute ")
             << kPartitionOutputsAttrName;

    size_t numResults = op->getNumResults();
    auto arrayAttr = cast<ArrayAttr>(outputsAttr);
    if (arrayAttr.size() != numResults) {
      return op->emitOpError("does not have expected number of output "
                             "partition sets in attr ")
             << kPartitionOutputsAttrName << "; should match number of results";
    }

    if (!op->hasAttr(kPartitionAttrName)) {
      return op->emitOpError("does not have expected attribute ")
             << kPartitionAttrName << " which is expected for ops with attr "
             << kPartitionOutputsAttrName;
    }

    auto partitionIds = getPartitionIds(op);
    SetVector<int> outputPartitionIdsUnion;
    for (auto outputPartitionIds : getPartitionOutputs(op)) {
      outputPartitionIdsUnion.insert(outputPartitionIds.begin(),
                                     outputPartitionIds.end());
    }
    if (!llvm::all_of(outputPartitionIdsUnion,
                      [&](int id) { return partitionIds.contains(id); })) {
      return op->emitOpError("partition ids in attr ")
             << kPartitionAttrName
             << " must be the union of all partition ids in "
             << kPartitionOutputsAttrName;
    }
  }

  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Partition
//===----------------------------------------------------------------------===//

bool Partition::hasOp(Operation *op) const {
  if (!hasPartition(op)) {
    return false;
  }
  auto partitionIds = getPartitionIds(op);
  return partitionIds.contains(getIndex());
}

void Partition::iterateInputs(scf::ForOp loop,
                              function_ref<void(OpOperand &)> callback) const {
  for (Operation *op : getOps()) {
    visitNestedOperands(op, [&](OpOperand &operand) {
      // Ignore implicit captures.
      Value value = operand.get();
      std::optional<SetVector<int>> partitionIds;
      if (hasPartition(value.getDefiningOp()))
        partitionIds = getPartitionIds(value.getDefiningOp());
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
      if (!owner) {
        continue;
      }
      std::optional<SetVector<int>> partitionIds;
      if (hasPartition(owner))
        partitionIds = getPartitionIds(owner);
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
    if (!owner) {
      continue;
    }
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
  assert(id.size() == 1);
  return getPartition(id[0]);
}

FailureOr<PartitionSet> PartitionSet::fromLoop(scf::ForOp loop) {
  if (failed(verifyPartitionedLoop(loop)))
    return failure();

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

  SmallVector<Operation *> annotatedOps;
  loop->walk([&](Operation *op) {
    if (hasPartition(op)) {
      annotatedOps.push_back(op);
    }
  });

  for (auto op : annotatedOps) {
    auto attrs = getPartitionIds(op);
    for (auto idx : attrs) {
      if (idx < 0 || idx >= result.partitions.size())
        return mlir::emitError(op->getLoc(), "invalid partition index ") << idx;
      result.partitions[idx]->addOp(op);
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

SetVector<int> getPartitionIds(Operation *op) {
  auto attrs = op->getAttr(kPartitionAttrName);
  SmallVector<int> partitionIds;
  for (auto id : cast<DenseI32ArrayAttr>(attrs).asArrayRef()) {
    partitionIds.push_back(id);
  }
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

SetVector<int> getPartitionIds(OpOperand *use) {
  auto owner = use->getOwner();
  if (isa<scf::YieldOp>(owner)) {
    return getPartitionOutputs(owner->getParentOp())[use->getOperandNumber()];
  }
  if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
    int idx = use->getOperandNumber() - forOp.getNumControlOperands();
    return idx >= 0 ? getPartitionOutputs(owner)[idx] : getPartitionIds(forOp);
  }
  return getPartitionIds(owner);
}

bool hasPartition(Operation *op) {
  return op && op->hasAttr(kPartitionAttrName);
}

bool hasWarpSpecializeTag(Operation *op) {
  return op && op->hasAttr(kWarpSpecializeTagAttrName);
}

std::optional<int> getWarpSpecializeTag(Operation *op) {
  if (hasWarpSpecializeTag(op))
    return cast<IntegerAttr>(op->getAttr(kWarpSpecializeTagAttrName)).getInt();
  return std::nullopt;
}

LogicalResult verifyPartitionedLoop(scf::ForOp loop) {
  if (failed(verifyPartitionAttrs(loop)))
    return failure();

  LogicalResult result = success();
  loop.walk([&](Operation *op) {
    if (failed(verifyPartitionAttrs(op))) {
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

void setPartition(Operation *op, ArrayRef<int> partitionIds) {
  Builder b(op->getContext());
  auto sorted = llvm::to_vector(partitionIds);
  llvm::sort(sorted);
  op->setAttr(kPartitionAttrName, b.getDenseI32ArrayAttr(sorted));
  for (auto &region : op->getRegions()) {
    for (auto &block : region.getBlocks()) {
      auto terminator = block.getTerminator();
      terminator->setAttr(kPartitionAttrName, b.getDenseI32ArrayAttr(sorted));
    }
  }
}

void setPartitionOutputs(Operation *op,
                         ArrayRef<SetVector<int>> partitionOutputsIds) {
  if (partitionOutputsIds.empty()) {
    op->removeAttr(kPartitionOutputsAttrName);
    return;
  }
  SmallVector<Attribute> attrs;
  Builder b(op->getContext());
  for (auto partitionIds : partitionOutputsIds) {
    auto sorted = llvm::to_vector(partitionIds);
    llvm::sort(sorted);
    attrs.push_back(b.getDenseI32ArrayAttr(sorted));
  }
  op->setAttr(kPartitionOutputsAttrName, b.getArrayAttr(attrs));
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

void setWarpSpecializeTag(Operation *op, int tag) {
  Builder b(op->getContext());
  op->setAttr(kWarpSpecializeTagAttrName, b.getI32IntegerAttr(tag));
}

} // namespace mlir::triton::gpu
