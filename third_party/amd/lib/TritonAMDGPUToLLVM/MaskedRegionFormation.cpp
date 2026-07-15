#include "MaskedOpsToLLVM.h"

#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
namespace AMD = mlir::triton::AMD;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUFORMMASKEDREGIONS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

static Value getMaskedOpMask(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<triton::amdgpu::MaskedLoadOp>([](auto load) -> Value {
        if (load.getMulticastMask())
          return {};
        return load.getMask();
      })
      .Case<triton::amdgpu::MaskedStoreOp>(
          [](auto store) -> Value { return store.getMask(); })
      .Default([](Operation *) -> Value { return {}; });
}

static Value resolveAggregateMask(Value mask) {
  Value current = mask;
  while (auto extract = current.getDefiningOp<LLVM::ExtractValueOp>()) {
    // Keep this intentionally narrow: only fold an extractvalue that reads a
    // value inserted at the same aggregate position.
    Value aggregate = extract.getContainer();
    ArrayRef<int64_t> position = extract.getPosition();
    Value insertedValue;
    while (auto insert = aggregate.getDefiningOp<LLVM::InsertValueOp>()) {
      ArrayRef<int64_t> insertPosition = insert.getPosition();
      if (insertPosition == position) {
        insertedValue = insert.getValue();
        break;
      }

      size_t commonSize = insertPosition.size() < position.size()
                              ? insertPosition.size()
                              : position.size();
      if (llvm::equal(insertPosition.take_front(commonSize),
                      position.take_front(commonSize)))
        break;
      aggregate = insert.getContainer();
    }
    if (!insertedValue)
      return current;
    current = insertedValue;
  }
  return current;
}

static bool hasUseOutside(Value value,
                          const llvm::SmallPtrSetImpl<Operation *> &ops) {
  return llvm::any_of(value.getUses(), [&](OpOperand &use) {
    return !ops.contains(use.getOwner());
  });
}

static bool hasResultUseOutside(Operation *op,
                                const llvm::SmallPtrSetImpl<Operation *> &ops) {
  for (Value result : op->getResults()) {
    if (hasUseOutside(result, ops))
      return true;
  }
  return false;
}

static bool isMovablePureOp(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
         isSpeculatable(op);
}

static bool isErasedMaskedOpOperand(OpOperand &use) {
  return llvm::TypeSwitch<Operation *, bool>(use.getOwner())
      .Case<triton::amdgpu::MaskedLoadOp>([&](auto load) {
        return &use == &load.getMaskMutable() ||
               &use == &load.getFalseValMutable();
      })
      .Case<triton::amdgpu::MaskedStoreOp>(
          [&](auto store) { return &use == &store.getMaskMutable(); })
      .Default([](Operation *) { return false; });
}

static bool
addBodyDependency(Value value,
                  const llvm::SmallPtrSetImpl<Operation *> &interval,
                  llvm::SmallPtrSetImpl<Operation *> &moveSet,
                  SmallVectorImpl<Operation *> &worklist) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || !interval.contains(defOp))
    return true;
  if (moveSet.contains(defOp))
    return true;
  if (!isMovablePureOp(defOp))
    return false;
  if (moveSet.insert(defOp).second)
    worklist.push_back(defOp);
  return true;
}

static bool
addHoistDependency(Value value,
                   const llvm::SmallPtrSetImpl<Operation *> &interval,
                   const llvm::SmallPtrSetImpl<Operation *> &moveSet,
                   llvm::SmallPtrSetImpl<Operation *> &hoistSet,
                   SmallVectorImpl<Operation *> &worklist) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || !interval.contains(defOp))
    return true;
  if (moveSet.contains(defOp) || !isMovablePureOp(defOp))
    return false;
  if (hoistSet.insert(defOp).second)
    worklist.push_back(defOp);
  return true;
}

struct MaskedRegionPlan {
  Operation *firstMaskedOp;
  SmallVector<Operation *> opsToMove;
  SmallVector<Operation *> opsToHoist;
  bool hasMaskedStore = false;
  SmallVector<triton::amdgpu::MaskedLoadOp> escapingLoads;
};

static bool isStoreValueUse(OpOperand &use) {
  Operation *owner = use.getOwner();
  return llvm::TypeSwitch<Operation *, bool>(owner)
      .Case<LLVM::StoreOp>(
          [&](LLVM::StoreOp store) { return &use == &store.getValueMutable(); })
      .Case<ROCDL::RawPtrBufferStoreOp>([&](ROCDL::RawPtrBufferStoreOp store) {
        return &use == &store.getVdataMutable();
      })
      .Case<triton::amdgpu::MaskedStoreOp>(
          [&](triton::amdgpu::MaskedStoreOp store) {
            return &use == &store.getValueMutable();
          })
      .Default([](Operation *) { return false; });
}

static bool isValueForwardingUse(OpOperand &use) {
  Operation *owner = use.getOwner();
  return llvm::TypeSwitch<Operation *, bool>(owner)
      .Case<LLVM::BitcastOp, LLVM::FPExtOp, LLVM::FPTruncOp, LLVM::SExtOp,
            LLVM::TruncOp, LLVM::ZExtOp>(
          [&](auto castOp) { return &use == &castOp.getArgMutable(); })
      .Case<LLVM::ExtractElementOp>([&](LLVM::ExtractElementOp extract) {
        return &use == &extract.getVectorMutable();
      })
      .Case<LLVM::ExtractValueOp>([&](LLVM::ExtractValueOp extract) {
        return &use == &extract.getContainerMutable();
      })
      .Case<LLVM::InsertElementOp>([&](LLVM::InsertElementOp insert) {
        return &use == &insert.getValueMutable() ||
               &use == &insert.getVectorMutable();
      })
      .Case<LLVM::InsertValueOp>([&](LLVM::InsertValueOp insert) {
        return &use == &insert.getValueMutable() ||
               &use == &insert.getContainerMutable();
      })
      .Case<LLVM::ShuffleVectorOp>([&](LLVM::ShuffleVectorOp shuffle) {
        return &use == &shuffle.getV1Mutable() ||
               &use == &shuffle.getV2Mutable();
      })
      .Default([](Operation *) { return false; });
}

static bool
loadOnlyFeedsStoreValues(triton::amdgpu::MaskedLoadOp load,
                         const llvm::SmallPtrSetImpl<Operation *> &moveSet) {
  SmallVector<Value> worklist{load.getResult()};
  llvm::SmallPtrSet<Operation *, 16> visited;
  bool sawStore = false;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      // Uses moved into the region do not escape through region results.
      if (moveSet.contains(owner))
        continue;

      if (isStoreValueUse(use)) {
        sawStore = true;
        continue;
      }

      if (!isValueForwardingUse(use))
        return false;

      if (visited.insert(owner).second)
        worklist.append(owner->result_begin(), owner->result_end());
    }
  }
  return sawStore;
}

// Region results for loads can expose adjacent loads to LLVM vectorization and
// early waits. Keep them only when all outside uses feed store values.
static bool shouldFormMaskedRegion(const MaskedRegionPlan &plan) {
  if (plan.escapingLoads.empty())
    return plan.hasMaskedStore;

  llvm::SmallPtrSet<Operation *, 16> moveSet(plan.opsToMove.begin(),
                                             plan.opsToMove.end());
  return llvm::all_of(plan.escapingLoads,
                      [&](triton::amdgpu::MaskedLoadOp load) {
                        return loadOnlyFeedsStoreValues(load, moveSet);
                      });
}

static FailureOr<MaskedRegionPlan>
computeMaskedRegionPlan(ArrayRef<Operation *> intervalOps) {
  llvm::SmallPtrSet<Operation *, 16> intervalSet(intervalOps.begin(),
                                                 intervalOps.end());
  llvm::SmallPtrSet<Operation *, 16> moveSet;
  llvm::SmallPtrSet<Operation *, 16> hoistSet;
  SmallVector<Operation *> moveWorklist;
  bool hasMaskedStore = false;

  for (Operation *op : intervalOps) {
    if (getMaskedOpMask(op)) {
      moveSet.insert(op);
      moveWorklist.push_back(op);
      hasMaskedStore |= isa<triton::amdgpu::MaskedStoreOp>(op);
      continue;
    }

    if (!isMovablePureOp(op))
      return failure();
  }

  while (!moveWorklist.empty()) {
    Operation *op = moveWorklist.pop_back_val();
    for (OpOperand &operand : op->getOpOperands()) {
      if (isErasedMaskedOpOperand(operand))
        continue;
      if (!addBodyDependency(operand.get(), intervalSet, moveSet, moveWorklist))
        return failure();
    }
  }

  SmallVector<triton::amdgpu::MaskedLoadOp> escapingLoads;
  for (Operation *op : intervalOps) {
    if (!moveSet.contains(op))
      continue;

    // Moved pure dependencies are not yielded by the region, so all their users
    // must move with them.
    if (!getMaskedOpMask(op) && hasResultUseOutside(op, moveSet))
      return failure();

    auto load = dyn_cast<triton::amdgpu::MaskedLoadOp>(op);
    if (load && hasUseOutside(load.getResult(), moveSet))
      escapingLoads.push_back(load);
  }

  SmallVector<Operation *> hoistWorklist;
  for (triton::amdgpu::MaskedLoadOp load : escapingLoads) {
    if (!addHoistDependency(load.getFalseVal(), intervalSet, moveSet, hoistSet,
                            hoistWorklist))
      return failure();
  }
  while (!hoistWorklist.empty()) {
    Operation *op = hoistWorklist.pop_back_val();
    for (Value operand : op->getOperands()) {
      if (!addHoistDependency(operand, intervalSet, moveSet, hoistSet,
                              hoistWorklist))
        return failure();
    }
  }

  SmallVector<Operation *> opsToMove;
  SmallVector<Operation *> opsToHoist;
  for (Operation *op : intervalOps) {
    if (moveSet.contains(op))
      opsToMove.push_back(op);
    else if (hoistSet.contains(op))
      opsToHoist.push_back(op);
  }

  return MaskedRegionPlan{intervalOps.front(), std::move(opsToMove),
                          std::move(opsToHoist), hasMaskedStore,
                          std::move(escapingLoads)};
}

static FailureOr<MaskedRegionPlan> findMaskedRegion(Operation *first) {
  Value mask = getMaskedOpMask(first);
  if (!mask)
    return failure();

  Value canonicalMask = resolveAggregateMask(mask);
  SmallVector<Operation *> intervalOps;
  intervalOps.push_back(first);
  size_t lastMaskedOpIndex = 0;
  for (Operation *op = first->getNextNode(); op; op = op->getNextNode()) {
    if (op->hasTrait<OpTrait::IsTerminator>())
      break;

    if (Value opMask = getMaskedOpMask(op)) {
      if (resolveAggregateMask(opMask) != canonicalMask)
        break;
      lastMaskedOpIndex = intervalOps.size();
      intervalOps.push_back(op);
      continue;
    }

    if (!isMovablePureOp(op))
      break;

    intervalOps.push_back(op);
  }

  if (lastMaskedOpIndex == 0)
    return failure();
  // Pure ops after the last matching masked op do not need to move.
  intervalOps.resize(lastMaskedOpIndex + 1);

  return computeMaskedRegionPlan(intervalOps);
}

static void formMaskedRegion(const MaskedRegionPlan &plan,
                             IRRewriter &rewriter) {
  Operation *firstMaskedOp = plan.firstMaskedOp;
  Location loc = firstMaskedOp->getLoc();
  Value mask = getMaskedOpMask(firstMaskedOp);
  llvm::SmallPtrSet<Operation *, 16> moveSet(plan.opsToMove.begin(),
                                             plan.opsToMove.end());

  SmallVector<Value> falseValues;
  SmallVector<Type> resultTypes;
  DenseMap<Operation *, unsigned> loadToResultIndex;
  for (triton::amdgpu::MaskedLoadOp load : plan.escapingLoads) {
    loadToResultIndex[load] = resultTypes.size();
    falseValues.push_back(load.getFalseVal());
    resultTypes.push_back(load.getResult().getType());
  }

  for (Operation *op : plan.opsToHoist)
    rewriter.moveOpBefore(op, firstMaskedOp);

  rewriter.setInsertionPoint(firstMaskedOp);
  auto regionOp = triton::amdgpu::MaskedRegionOp::create(
      rewriter, loc, TypeRange(resultTypes), mask, falseValues);
  Block *body = rewriter.createBlock(&regionOp.getBody());

  // Interleaved mask and unused false-value definitions may temporarily stop
  // dominating moved helpers. They are erased by the rewrites immediately
  // below, before the transformed IR is observed.
  for (Operation *op : plan.opsToMove)
    rewriter.moveOpBefore(op, body, body->end());

  SmallVector<Value> yieldValues(resultTypes.size());
  auto isMovedUse = [&](OpOperand &use) {
    return moveSet.contains(use.getOwner());
  };
  for (Operation &op : llvm::make_early_inc_range(*body)) {
    if (auto load = dyn_cast<triton::amdgpu::MaskedLoadOp>(&op)) {
      rewriter.setInsertionPoint(load);
      Value trueValue =
          AMD::createRegularLoadFromMaskedOp(rewriter, load.getLoc(), load);
      auto it = loadToResultIndex.find(load);
      if (it != loadToResultIndex.end()) {
        unsigned resultIndex = it->second;
        yieldValues[resultIndex] = trueValue;
        load.getResult().replaceUsesWithIf(
            regionOp.getResult(resultIndex),
            [&](OpOperand &use) { return !isMovedUse(use); });
      }
      rewriter.replaceOp(load, trueValue);
      continue;
    }

    if (auto store = dyn_cast<triton::amdgpu::MaskedStoreOp>(&op)) {
      rewriter.setInsertionPoint(store);
      AMD::createUnmaskedStoreFromMaskedOp(rewriter, store.getLoc(), store);
      rewriter.eraseOp(store);
    }
  }

  rewriter.setInsertionPointToEnd(body);
  triton::amdgpu::MaskedYieldOp::create(rewriter, loc, yieldValues);
}

static bool tryFormMaskedRegionInBlock(Block &block, IRRewriter &rewriter) {
  for (Operation *parent = block.getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<triton::amdgpu::MaskedRegionOp>(parent))
      return false;
  }

  for (Operation &op : block.without_terminator()) {
    FailureOr<MaskedRegionPlan> plan = findMaskedRegion(&op);
    if (failed(plan) || !shouldFormMaskedRegion(*plan))
      continue;

    formMaskedRegion(*plan, rewriter);
    return true;
  }

  return false;
}

struct TritonAMDGPUFormMaskedRegionsPass final
    : public triton::impl::TritonAMDGPUFormMaskedRegionsBase<
          TritonAMDGPUFormMaskedRegionsPass> {
  void runOnOperation() override { AMD::formMaskedRegions(getOperation()); }
};

} // namespace

namespace mlir::triton::AMD {

void formMaskedRegions(ModuleOp module) {
  IRRewriter rewriter(module.getContext());

  SmallVector<Block *> blocks;
  module.walk([&](Block *block) { blocks.push_back(block); });

  for (Block *block : blocks) {
    while (tryFormMaskedRegionInBlock(*block, rewriter)) {
      // Keep forming regions in this original block until no candidate remains.
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUFormMaskedRegionsPass() {
  return std::make_unique<TritonAMDGPUFormMaskedRegionsPass>();
}

} // namespace mlir::triton::AMD
