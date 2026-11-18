#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "PartitionAttrs.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

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
    if (!isa<scf::ForOp, scf::WhileOp>(op)) {
      return op->emitOpError("has unexpected attribute ")
             << kWarpSpecializeAttrName
             << " which is expected only on `scf.for` or `scf.while` ops";
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
    if (!isa<scf::ForOp, scf::WhileOp, scf::IfOp, triton::ReduceOp,
             triton::ScanOp>(op))
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

bool triton::gpu::isUserVisibleSideEffect(Operation *op) {
  // Source-level effects that must execute exactly once. Lowering ops (arefs,
  // barriers, tmem, fences), descriptor setup, and idempotent waits
  // (grid_dependency_wait, atomic_poll) may be replicated across partitions
  // and are excluded here.
  if (isa<triton::AtomicRMWOp, triton::AtomicCASOp, triton::AtomicStoreOp,
          triton::StoreOp, triton::DescriptorStoreLikeOpInterface,
          triton::AssertOp, triton::PrintOp, triton::CallOp,
          triton::GridDependencyLaunchDependentsOp>(op))
    return true;
  if (auto asmOp = dyn_cast<triton::ElementwiseInlineAsmOp>(op))
    return !asmOp.getPure();
  return false;
}

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

void Partition::iterateInputs(LoopLikeOpInterface loop,
                              function_ref<void(OpOperand &)> callback) const {
  Value inductionVar;
  if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation()))
    inductionVar = forOp.getInductionVar();

  for (Operation *op : getOps()) {
    visitNestedOperands(op, [&](OpOperand &operand) {
      // Ignore implicit captures.
      Value value = operand.get();
      std::optional<SetVector<int>> partitionIds;
      if (hasPartition(value.getDefiningOp()))
        partitionIds = getPartitionIds(value.getDefiningOp());
      if (value.getParentRegion()->getParentOp() != loop)
        return;
      // Ignore the induction variable.
      if (value == inductionVar)
        return;
      if ((isa<BlockArgument>(value) || !partitionIds ||
           !partitionIds->contains(getIndex()))) {
        // This value originates from a different partition in the same
        // iteration.
        assert(value.getParentRegion()->getParentOp() == loop);
        callback(operand);
      }
    });
  }
}

void Partition::iterateOutputs(
    LoopLikeOpInterface loop,
    function_ref<void(Operation *, OpOperand &)> callback) const {
  for (Operation *op : getOps()) {
    for (OpOperand &use : op->getUses()) {
      Operation *owner = nullptr;
      for (Region *region : loop.getLoopRegions())
        if ((owner = region->front().findAncestorOpInBlock(*use.getOwner())))
          break;
      if (!owner) {
        continue;
      }
      std::optional<SetVector<int>> partitionIds;
      if (hasPartition(owner))
        partitionIds = getPartitionIds(owner);
      if (isa<scf::YieldOp, scf::ConditionOp>(owner) || !partitionIds ||
          !partitionIds->contains(getIndex())) {
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

void Partition::iterateDefs(
    scf::WhileOp loop, function_ref<void(OpResult, unsigned)> callback) const {
  iterateInputs(loop, [&](OpOperand &input) {
    auto value = input.get();
    int distance = 0;
    while (auto arg = dyn_cast<BlockArgument>(value)) {
      value = loop.getYieldOp().getOperand(arg.getArgNumber());
      ++distance;
    }
    auto def = dyn_cast<OpResult>(value);
    if (def && loop->isProperAncestor(def.getDefiningOp()))
      callback(def, distance);
  });
}

void Partition::iterateUses(
    LoopLikeOpInterface loop,
    function_ref<void(OpResult, OpOperand &, unsigned)> callback) const {
  SmallVector<std::tuple<OpResult, OpOperand *, unsigned>> uses;
  iterateOutputs(loop, [&](Operation *, OpOperand &use) {
    uses.emplace_back(cast<OpResult>(use.get()), &use, 0);
  });
  while (!uses.empty()) {
    auto [output, use, distance] = uses.pop_back_val();
    Operation *owner = nullptr;
    for (Region *region : loop.getLoopRegions())
      if ((owner = region->front().findAncestorOpInBlock(*use->getOwner())))
        break;
    if (!owner) {
      continue;
    }
    if (auto yield = dyn_cast<scf::YieldOp>(owner)) {
      auto arg = loop.getRegionIterArgs()[use->getOperandNumber()];
      for (OpOperand &argUse : arg.getUses())
        uses.emplace_back(output, &argUse, distance + 1);
      continue;
    }
    if (auto condition = dyn_cast<scf::ConditionOp>(owner);
        condition && use->getOperandNumber() > 0) {
      auto body = &loop.getLoopRegions().back()->front();
      auto arg = body->getArgument(use->getOperandNumber() - 1);
      for (OpOperand &argUse : arg.getUses())
        uses.emplace_back(output, &argUse, distance);
      continue;
    }
    callback(output, *use, distance);
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

FailureOr<PartitionSet> PartitionSet::fromLoop(LoopLikeOpInterface loop) {
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
      return mlir::emitError(loop->getLoc(), "partition stages attribute '")
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

SetVector<int> getPartitionIds(OpOperand *use) {
  Operation *owner = use->getOwner();
  auto pos = use->getOperandNumber();
  if (isa<scf::YieldOp, scf::ConditionOp>(owner)) {
    unsigned numControlOperands = isa<scf::ConditionOp>(owner) ? 1 : 0;
    if (pos < numControlOperands)
      return getPartitionIds(owner);
    return getPartitionOutputs(owner->getParentOp())[pos - numControlOperands];
  }
  if (auto loop = dyn_cast<LoopLikeOpInterface>(owner)) {
    auto numControlOperands = owner->getNumOperands() - loop.getInits().size();
    if (pos < numControlOperands)
      return getPartitionIds(owner);
    return getPartitionOutputs(owner)[pos - numControlOperands];
  }
  return getPartitionIds(owner);
}

bool hasWarpSpecializeTag(Operation *op) {
  return op && op->hasAttr(kWarpSpecializeTagAttrName);
}

std::optional<int> getWarpSpecializeTag(Operation *op) {
  if (hasWarpSpecializeTag(op))
    return cast<IntegerAttr>(op->getAttr(kWarpSpecializeTagAttrName)).getInt();
  return std::nullopt;
}

LogicalResult verifyPartitionedLoop(LoopLikeOpInterface loop) {
  if (failed(verifyPartitionAttrs(loop)))
    return failure();

  LogicalResult result = success();
  loop.walk([&](Operation *op) {
    if (failed(verifyPartitionAttrs(op))) {
      result = failure();
      return WalkResult::interrupt();
    }
    // PartitionLoops clones an op once per partition, so a side-effecting op
    // on several partitions would run its effect several times.
    if (op->getNumRegions() == 0 && hasPartition(op) &&
        isUserVisibleSideEffect(op) && getPartitionIds(op).size() > 1) {
      op->emitOpError("side-effecting op must have exactly one partition");
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

namespace {

bool isInLoopRegions(LoopLikeOpInterface loop, Operation *op) {
  if (!op)
    return false;
  for (Region *region : loop.getLoopRegions())
    if (region->isAncestor(op->getParentRegion()))
      return true;
  return false;
}

void addDefaultPartition(Operation *op, int resultIdx) {
  if (!hasPartition(op))
    return;
  auto ids = getPartitionIds(op);
  ids.insert(0);
  setPartition(op, ids);
  if (resultIdx < 0 || op->getNumResults() == 0)
    return;
  SmallVector<SetVector<int>, 4> outputs;
  if (op->hasAttr(kPartitionOutputsAttrName))
    outputs = getPartitionOutputs(op);
  else
    outputs.resize(op->getNumResults());
  outputs[resultIdx].insert(0);
  setPartitionOutputs(op, outputs);
}

} // namespace

void assignDefaultPartitionToEscapingValues(LoopLikeOpInterface loop) {
  // Walk the backward slice of each result that must be materialized in the
  // default partition: pure ops are replicated there, structured ops get
  // partition 0 added to their result. The walk stops at arefs (already
  // communicated) and at user-visible effects (must not be recomputed;
  // InsertAref communicates their result).
  DenseSet<Value> visited;
  std::function<void(Value)> visit = [&](Value value) {
    if (!visited.insert(value).second)
      return;

    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      Operation *parent = blockArg.getOwner()->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (forOp->hasAttr(kWarpSpecializeAttrName))
          return;
        // The induction variable is uniform across partitions.
        if (blockArg.getArgNumber() == 0)
          return;
        visit(forOp.getInitArgs()[blockArg.getArgNumber() - 1]);
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
        if (whileOp->hasAttr(kWarpSpecializeAttrName))
          return;
        if (blockArg.getParentRegion() == &whileOp.getBefore()) {
          // A before-region argument is the previous iteration's after-yield.
          auto yield =
              cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
          visit(yield.getOperand(blockArg.getArgNumber()));
        } else {
          // An after-region argument comes from the condition operands.
          auto condition = cast<scf::ConditionOp>(
              whileOp.getBefore().front().getTerminator());
          visit(condition.getArgs()[blockArg.getArgNumber()]);
        }
      }
      return;
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return;

    // Already communicated through an aref.
    if (defOp->getName().getStringRef().starts_with("nvws.aref"))
      return;
    // Executes once; InsertAref communicates its result.
    if (isUserVisibleSideEffect(defOp))
      return;
    // Do not duplicate another warp-specialized loop.
    if (defOp->hasAttr(kWarpSpecializeAttrName))
      return;
    if (!isInLoopRegions(loop, defOp))
      return;

    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      auto pos = cast<OpResult>(value).getResultNumber();
      addDefaultPartition(ifOp, pos);
      visit(ifOp.thenYield()->getOperand(pos));
      if (ifOp.elseBlock())
        visit(ifOp.elseYield()->getOperand(pos));
      visit(ifOp.getCondition());
      return;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
      auto pos = cast<OpResult>(value).getResultNumber();
      addDefaultPartition(forOp, pos);
      visit(forOp.getBody()->getTerminator()->getOperand(pos));
      for (int idx = 0; idx < forOp.getNumControlOperands(); ++idx)
        visit(forOp.getOperand(idx));
      return;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
      auto pos = cast<OpResult>(value).getResultNumber();
      addDefaultPartition(whileOp, pos);
      visit(whileOp.getYieldedValues()[pos]);
      auto condition =
          cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
      visit(condition.getCondition());
      return;
    }
    if (auto reduce = dyn_cast<triton::ReduceOp>(defOp)) {
      auto pos = cast<OpResult>(value).getResultNumber();
      addDefaultPartition(reduce, pos);
      for (Value src : reduce.getSrcs())
        visit(src);
      for (Operation &op : *reduce.getBody())
        addDefaultPartition(&op, -1);
      return;
    }
    if (auto scan = dyn_cast<triton::ScanOp>(defOp)) {
      auto pos = cast<OpResult>(value).getResultNumber();
      addDefaultPartition(scan, pos);
      for (Value src : scan.getSrcs())
        visit(src);
      for (Operation &op : scan.getRegion().front())
        addDefaultPartition(&op, -1);
      return;
    }

    addDefaultPartition(defOp, -1);
    for (Value operand : defOp->getOperands())
      visit(operand);
  };

  for (auto [idx, result] : llvm::enumerate(loop->getResults())) {
    if (!isa<IntegerType, FloatType>(result.getType()) || result.use_empty())
      continue;
    // The default partition must observe results used outside the loop or by
    // another warp-specialized loop.
    bool escapes = llvm::any_of(result.getUsers(), [](Operation *user) {
      return !hasPartition(user) || (isa<scf::ForOp, scf::WhileOp>(user) &&
                                     hasWarpSpecializeTag(user));
    });
    if (!escapes)
      continue;
    addDefaultPartition(loop.getOperation(), idx);
    visit(loop.getYieldedValues()[idx]);
  }
}

} // namespace mlir::triton::gpu
