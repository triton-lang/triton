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
        unsigned operandNumber = use.getOperandNumber();
        return operandNumber == load.getMaskMutable().getOperandNumber() ||
               operandNumber == load.getFalseValMutable().getOperandNumber();
      })
      .Case<triton::amdgpu::MaskedStoreOp>([&](auto store) {
        return use.getOperandNumber() ==
               store.getMaskMutable().getOperandNumber();
      })
      .Default([](Operation *) { return false; });
}

static bool
hasNoLiveUsesInMovedOps(Operation *op,
                        const llvm::SmallPtrSetImpl<Operation *> &moveSet) {
  for (Value result : op->getResults()) {
    for (OpOperand &use : result.getUses()) {
      if (!moveSet.contains(use.getOwner()))
        continue;
      if (!isErasedMaskedOpOperand(use))
        return false;
    }
  }
  return true;
}

static bool
addRegionBodyDependency(Value value,
                        const llvm::SmallPtrSetImpl<Operation *> &interval,
                        llvm::SmallPtrSetImpl<Operation *> &moveSet,
                        SmallVectorImpl<Operation *> &regionBodyWorklist) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || !interval.contains(defOp))
    return true;
  if (moveSet.contains(defOp))
    return true;
  if (!isMovablePureOp(defOp))
    return false;
  if (moveSet.insert(defOp).second)
    regionBodyWorklist.push_back(defOp);
  return true;
}

static bool
addRegionElseDependency(Value value,
                        const llvm::SmallPtrSetImpl<Operation *> &interval,
                        const llvm::SmallPtrSetImpl<Operation *> &moveSet,
                        llvm::SmallPtrSetImpl<Operation *> &hoistSet,
                        SmallVectorImpl<Operation *> &regionElseWorklist) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || !interval.contains(defOp))
    return true;
  if (moveSet.contains(defOp) || !isMovablePureOp(defOp))
    return false;
  if (hoistSet.insert(defOp).second)
    regionElseWorklist.push_back(defOp);
  return true;
}

struct MaskedRegionPlan {
  SmallVector<Operation *> intervalOps;
  SmallVector<Operation *> opsToMove;
  SmallVector<Operation *> opsToHoist;
};

static FailureOr<MaskedRegionPlan>
computeMaskedRegionPlan(ArrayRef<Operation *> intervalOps) {
  llvm::SmallPtrSet<Operation *, 16> intervalSet(intervalOps.begin(),
                                                 intervalOps.end());
  llvm::SmallPtrSet<Operation *, 16> moveSet;
  llvm::SmallPtrSet<Operation *, 16> hoistSet;
  SmallVector<Operation *> regionBodyWorklist;
  SmallVector<Operation *> regionElseWorklist;

  for (Operation *op : intervalOps) {
    if (Value opMask = getMaskedOpMask(op)) {
      moveSet.insert(op);
      if (auto load = dyn_cast<triton::amdgpu::MaskedLoadOp>(op)) {
        if (!addRegionElseDependency(load.getFalseVal(), intervalSet, moveSet,
                                     hoistSet, regionElseWorklist))
          return failure();
        if (!addRegionBodyDependency(load.getPtr(), intervalSet, moveSet,
                                     regionBodyWorklist))
          return failure();
        continue;
      }
      auto store = cast<triton::amdgpu::MaskedStoreOp>(op);
      if (!addRegionBodyDependency(store.getPtr(), intervalSet, moveSet,
                                   regionBodyWorklist))
        return failure();
      if (!addRegionBodyDependency(store.getValue(), intervalSet, moveSet,
                                   regionBodyWorklist))
        return failure();
      continue;
    }

    if (!isMovablePureOp(op))
      return failure();
  }

  while (!regionBodyWorklist.empty()) {
    Operation *op = regionBodyWorklist.pop_back_val();
    for (Value operand : op->getOperands()) {
      if (!addRegionBodyDependency(operand, intervalSet, moveSet,
                                   regionBodyWorklist))
        return failure();
    }
  }

  while (!regionElseWorklist.empty()) {
    Operation *op = regionElseWorklist.pop_back_val();
    if (moveSet.contains(op))
      return failure();
    for (Value operand : op->getOperands()) {
      if (!addRegionElseDependency(operand, intervalSet, moveSet, hoistSet,
                                   regionElseWorklist))
        return failure();
    }
  }

  SmallVector<Operation *> opsToMove = llvm::to_vector(llvm::make_filter_range(
      intervalOps, [&](Operation *op) { return moveSet.contains(op); }));
  SmallVector<Operation *> opsToHoist = llvm::to_vector(llvm::make_filter_range(
      intervalOps, [&](Operation *op) { return hoistSet.contains(op); }));

  for (Operation *op : intervalOps) {
    if (hoistSet.contains(op))
      continue;

    // Moved pure dependencies are not yielded by the region, so all their users
    // must move with them.
    if (moveSet.contains(op)) {
      if (!getMaskedOpMask(op) && hasResultUseOutside(op, moveSet))
        return failure();
      continue;
    }

    // Ops left behind may only be used by moved ops through operands that are
    // erased while rewriting masked memory ops to unmasked memory ops.
    if (!hasNoLiveUsesInMovedOps(op, moveSet))
      return failure();
  }

  return MaskedRegionPlan{SmallVector<Operation *>(intervalOps),
                          std::move(opsToMove), std::move(opsToHoist)};
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
  Operation *first = plan.intervalOps.front();
  Location loc = first->getLoc();
  Value mask = getMaskedOpMask(first);
  llvm::SmallPtrSet<Operation *, 16> moveSet(plan.opsToMove.begin(),
                                             plan.opsToMove.end());

  SmallVector<Value> falseValues;
  SmallVector<Type> resultTypes;
  DenseMap<Operation *, unsigned> loadToResultIndex;
  for (Operation *op : plan.opsToMove) {
    auto load = dyn_cast<triton::amdgpu::MaskedLoadOp>(op);
    if (load && hasUseOutside(load.getResult(), moveSet)) {
      loadToResultIndex[op] = resultTypes.size();
      falseValues.push_back(load.getFalseVal());
      resultTypes.push_back(load.getResult().getType());
    }
  }

  for (Operation *op : plan.opsToHoist)
    rewriter.moveOpBefore(op, first);

  rewriter.setInsertionPoint(first);
  auto regionOp = triton::amdgpu::MaskedRegionOp::create(
      rewriter, loc, TypeRange(resultTypes), mask, falseValues);
  Block *body = rewriter.createBlock(&regionOp.getBody());

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
      Value outsideValue;
      auto it = loadToResultIndex.find(load);
      if (it != loadToResultIndex.end()) {
        unsigned resultIndex = it->second;
        yieldValues[resultIndex] = trueValue;
        outsideValue = regionOp.getResult(resultIndex);
      }
      if (outsideValue)
        load.getResult().replaceUsesWithIf(
            outsideValue, [&](OpOperand &use) { return !isMovedUse(use); });
      load.getResult().replaceUsesWithIf(trueValue, isMovedUse);
      rewriter.eraseOp(load);
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

  SmallVector<Operation *> ops;
  for (Operation &op : block.without_terminator())
    ops.push_back(&op);

  for (Operation *op : ops) {
    if (!getMaskedOpMask(op))
      continue;

    FailureOr<MaskedRegionPlan> maskedRegion = findMaskedRegion(op);
    if (failed(maskedRegion))
      continue;

    formMaskedRegion(*maskedRegion, rewriter);
    return true;
  }

  return false;
}

struct TritonAMDGPUFormMaskedRegionsPass final
    : public triton::impl::TritonAMDGPUFormMaskedRegionsBase<
          TritonAMDGPUFormMaskedRegionsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    AMD::formMaskedRegions(module);
  }
};

} // namespace

namespace mlir::triton::AMD {

void formMaskedRegions(ModuleOp module) {
  IRRewriter rewriter(module.getContext());

  SmallVector<Block *> blocks;
  module.walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region.getBlocks())
        blocks.push_back(&block);
  });

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
