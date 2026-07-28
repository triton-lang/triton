#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/LayoutAssignment.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZELAYOUTS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-optimize-layouts"

namespace {

using LayoutValue = std::pair<Value, Attribute>;

struct PlannedValue {
  Operation *operation;
  Attribute encoding;
  Attribute operandEncoding;
  bool bypassConversion;
};

/// Plan an entire pure tensor expression before changing the IR. A conversion
/// can be removed only when its complete producer graph can be regenerated in
/// the required layout, every tensor leaf is layout-independent, and none of
/// the original expression needs to remain alive for another user.
class ScalarRootedLayoutPlan {
public:
  explicit ScalarRootedLayoutPlan(ConvertLayoutOp root) : root(root) {}

  LogicalResult analyze() {
    if (failed(plan(root.getSrc(), root.getType().getEncoding()))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] producer graph cannot be assigned the "
                    "required layout for "
                 << root << '\n');
      return failure();
    }

    unsigned originalMaterializations = 0;
    for (Operation *op : originalOperations) {
      if (!isa<ConvertLayoutOp>(op))
        ++originalMaterializations;

      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (user != root.getOperation() &&
              !originalOperations.contains(user)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "[" DEBUG_TYPE "] retaining live shared producer "
                       << *op << '\n');
            return failure();
          }
        }
      }
    }

    unsigned newMaterializations = 0;
    for (const auto &entry : plannedValues)
      newMaterializations += !entry.second.bypassConversion;

    // Do not turn layout optimization into expression duplication. This also
    // keeps graph-shaped joins from growing exponentially.
    if (newMaterializations > originalMaterializations) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] rejecting expression duplication: "
                 << originalMaterializations << " original operations, "
                 << newMaterializations << " planned operations\n");
      return failure();
    }

    return success();
  }

  void rewrite() {
    OpBuilder builder(root);
    Value replacement =
        materialize(root.getSrc(), root.getType().getEncoding(), builder);
    assert(replacement && "a validated layout plan must materialize");

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] rematerializing "
               << originalOperations.size()
               << " scalar-rooted operations without layout conversions\n");

    root.getResult().replaceAllUsesWith(replacement);
    root.erase();

    for (Operation *op : originalOperations) {
      if (isOpTriviallyDead(op))
        op->erase();
    }
  }

private:
  static constexpr unsigned maxPlannedValues = 512;

  LogicalResult plan(Value value, Attribute encoding) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !encoding) {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] expected an encoded tensor: "
                              << value << '\n');
      return failure();
    }

    LayoutValue key{value, encoding};
    if (plannedValues.contains(key))
      return success();
    if (plannedValues.size() >= maxPlannedValues ||
        !active.insert(key).second) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE "] bounded or cyclic producer graph at "
                 << value << '\n');
      return failure();
    }

    Operation *op = value.getDefiningOp();
    if (!op || op->getBlock() != root->getBlock() || op->getNumResults() != 1 ||
        !isMemoryEffectFree(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[" DEBUG_TYPE
                 << "] producer is not a single-result pure operation in the "
                    "conversion block: "
                 << value << '\n');
      return failure();
    }

    originalOperations.insert(op);

    if (auto convert = dyn_cast<ConvertLayoutOp>(op)) {
      if (failed(plan(convert.getSrc(), encoding)))
        return failure();
      plannedValues.try_emplace(key, PlannedValue{op, encoding, {}, true});
      active.erase(key);
      return success();
    }

    Attribute operandEncoding;
    if (isa<triton::SplatOp, triton::MakeRangeOp>(op)) {
      // Their tensor contents are defined by scalar operands or logical
      // indices, so their distributed result layout can be chosen directly.
    } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      if (!isa<DenseElementsAttr>(constant.getValue())) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] unsupported tensor constant: " << *op
                   << '\n');
        return failure();
      }
    } else {
      if (!(op->hasTrait<OpTrait::Elementwise>() ||
            op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
            isa<triton::ReshapeOp, triton::JoinOp, triton::ExpandDimsOp,
                triton::TransOp, triton::BroadcastOp>(op))) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] unsupported producer: " << *op << '\n');
        return failure();
      }

      operandEncoding = inferSrcEncoding(op, encoding);
      if (!operandEncoding) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[" DEBUG_TYPE "] cannot infer a legal source encoding "
                   << "for " << *op << " from " << encoding << '\n');
        return failure();
      }
    }

    for (Value operand : op->getOperands()) {
      if (!isa<RankedTensorType>(operand.getType()))
        continue;
      if (!operandEncoding || failed(plan(operand, operandEncoding)))
        return failure();
    }

    plannedValues.try_emplace(
        key, PlannedValue{op, encoding, operandEncoding, false});
    active.erase(key);
    return success();
  }

  Value materialize(Value value, Attribute encoding, OpBuilder &builder) {
    LayoutValue key{value, encoding};
    if (auto found = materializedValues.find(key);
        found != materializedValues.end())
      return found->second;

    const PlannedValue &planned = plannedValues.find(key)->second;
    if (planned.bypassConversion) {
      auto convert = cast<ConvertLayoutOp>(planned.operation);
      Value replacement = materialize(convert.getSrc(), encoding, builder);
      materializedValues.try_emplace(key, replacement);
      return replacement;
    }

    auto oldType = cast<RankedTensorType>(value.getType());
    RankedTensorType newType = oldType.cloneWithEncoding(encoding);
    if (auto constant = dyn_cast<arith::ConstantOp>(planned.operation)) {
      auto elements = cast<DenseElementsAttr>(constant.getValue());
      Value replacement = arith::ConstantOp::create(
          builder, constant.getLoc(), newType, elements.reshape(newType));
      materializedValues.try_emplace(key, replacement);
      return replacement;
    }

    IRMapping mapping;
    for (Value operand : planned.operation->getOperands()) {
      if (isa<RankedTensorType>(operand.getType()))
        mapping.map(operand,
                    materialize(operand, planned.operandEncoding, builder));
    }

    Operation *replacement = builder.clone(*planned.operation, mapping);
    replacement->getResult(0).setType(newType);
    Value result = replacement->getResult(0);
    materializedValues.try_emplace(key, result);
    return result;
  }

  ConvertLayoutOp root;
  DenseMap<LayoutValue, PlannedValue> plannedValues;
  DenseMap<LayoutValue, Value> materializedValues;
  DenseSet<LayoutValue> active;
  SetVector<Operation *> originalOperations;
};

/// Exact-order joins interleave their operands. Their result layout therefore
/// cannot always be inferred backward: the join dimension may be distributed
/// over lanes. For a graph rooted in scalar splats, constants, and logical
/// ranges, evaluate the join choices from the bits of each logical result
/// index instead. Exact-order reshapes preserve that flat logical index, so
/// the expression can be emitted directly in its required result layout. A
/// range in a join operand observes the remaining high index bits after the
/// join choices along its path have been removed.
class ScalarJoinExpressionPlan {
public:
  explicit ScalarJoinExpressionPlan(ConvertLayoutOp root)
      : insertionPoint(root.getOperation()), rootValue(root.getSrc()),
        target(root.getType()), conversionRoot(root) {}

  explicit ScalarJoinExpressionPlan(triton::StoreOp root)
      : insertionPoint(root.getOperation()), rootValue(root.getValue()),
        target(dyn_cast<RankedTensorType>(root.getValue().getType())),
        storeRoot(root) {}

  LogicalResult analyze() {
    if (!target || !target.getEncoding() || !target.hasStaticShape())
      return failure();
    int64_t size = target.getNumElements();
    if (size <= 0 || size > std::numeric_limits<int32_t>::max() ||
        !llvm::isPowerOf2_64(static_cast<uint64_t>(size)))
      return failure();

    auto *interface =
        target.getEncoding()
            .getDialect()
            .getRegisteredInterface<triton::DialectInferLayoutInterface>();
    if (!interface ||
        failed(interface->inferReshapeOpEncoding(
            target.getShape(), target.getEncoding(), ArrayRef<int64_t>{size},
            flatEncoding, /*allowReorder=*/false, insertionPoint->getLoc())))
      return failure();

    if (storeRoot && !rootValue.hasOneUse())
      return failure();

    if (failed(plan(rootValue, 0)))
      return failure();

    unsigned conversions = conversionRoot ? 1 : 0;
    unsigned warpShuffleConversions = 0;
    unsigned sharedMemoryConversions = 0;
    unsigned joins = 0;
    unsigned originalArithmetic = 0;

    auto classifyConversion = [&](ConvertLayoutOp convert) {
      auto sourceType = convert.getSrc().getType();
      auto resultType = convert.getType();
      if (cvtReordersRegisters(sourceType, resultType))
        return;
      if (cvtNeedsWarpShuffle(sourceType, resultType))
        ++warpShuffleConversions;
      else
        ++sharedMemoryConversions;
    };

    if (conversionRoot)
      classifyConversion(conversionRoot);

    for (Operation *op : originalOperations) {
      if (auto convert = dyn_cast<ConvertLayoutOp>(op)) {
        ++conversions;
        classifyConversion(convert);
      }
      joins += isa<triton::JoinOp>(op);
      originalArithmetic += !isStructuralOrLeaf(op);

      // In production stochastic-rounding kernels these inexpensive leaves
      // can live in an enclosing loop or function block and have other users.
      // Recreating a splat, constant, or logical range does not duplicate the
      // random-number arithmetic or require ownership of the original leaf.
      if (isRematerializableLeaf(op))
        continue;

      for (Value result : op->getResults()) {
        for (Operation *user : result.getUsers()) {
          if (user == insertionPoint) {
            if (result != rootValue)
              return failure();
            continue;
          }
          if (!originalOperations.contains(user))
            return failure();
        }
      }
    }

    // Index evaluation introduces integer instructions and live values. Do
    // not replace a single conversion, or a chain consisting entirely of
    // thread-local register permutations, with that extra arithmetic.
    if (joins == 0 || conversions < 2 ||
        (warpShuffleConversions == 0 && sharedMemoryConversions == 0))
      return failure();

    unsigned plannedArithmetic = 0;
    for (const auto &planned : plannedValues)
      plannedArithmetic += !isStructuralOrLeaf(planned.first.getDefiningOp());

    // A shared random-number producer can feed multiple joins. Keep the
    // incumbent if direct evaluation would duplicate that arithmetic at
    // different join depths.
    if (plannedArithmetic > originalArithmetic)
      return failure();

    return success();
  }

  void rewrite() {
    OpBuilder builder(insertionPoint);
    Location loc = insertionPoint->getLoc();
    auto rangeType = RankedTensorType::get({target.getNumElements()},
                                           builder.getI32Type(), flatEncoding);
    Value range = triton::MakeRangeOp::create(
        builder, loc, rangeType, 0,
        static_cast<int32_t>(target.getNumElements()));
    auto indexType = RankedTensorType::get(
        target.getShape(), builder.getI32Type(), target.getEncoding());
    indices = triton::ReshapeOp::create(builder, loc, indexType, range);

    Value replacement = materialize(rootValue, 0, builder);
    assert(replacement && "a validated join expression must materialize");

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] evaluating " << originalOperations.size()
               << " scalar-rooted operations from exact logical index bits\n");

    if (conversionRoot) {
      conversionRoot.getResult().replaceAllUsesWith(replacement);
      conversionRoot.erase();
    } else {
      storeRoot.getValueMutable().assign(replacement);
    }
    for (Operation *op : originalOperations) {
      if (isOpTriviallyDead(op))
        op->erase();
    }
  }

private:
  using IndexedValue = std::pair<Value, unsigned>;
  static constexpr unsigned maxPlannedValues = 512;
  static constexpr unsigned maxJoinDepth = 30;

  static bool isRematerializableLeaf(Operation *op) {
    return isa<triton::SplatOp, triton::MakeRangeOp, arith::ConstantOp>(op);
  }

  static bool isStructuralOrLeaf(Operation *op) {
    return isRematerializableLeaf(op) ||
           isa<ConvertLayoutOp, triton::ReshapeOp, triton::JoinOp>(op);
  }

  LogicalResult plan(Value value, unsigned depth) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !type.hasStaticShape() || depth > maxJoinDepth ||
        (target.getNumElements() >> depth) != type.getNumElements())
      return failure();

    IndexedValue key{value, depth};
    if (plannedValues.contains(key))
      return success();
    if (plannedValues.size() >= maxPlannedValues || !active.insert(key).second)
      return failure();

    Operation *op = value.getDefiningOp();
    if (!op || op->getNumResults() != 1 || !isMemoryEffectFree(op) ||
        (op->getBlock() != insertionPoint->getBlock() &&
         !isRematerializableLeaf(op)))
      return failure();

    originalOperations.insert(op);

    if (auto convert = dyn_cast<ConvertLayoutOp>(op)) {
      if (failed(plan(convert.getSrc(), depth)))
        return failure();
    } else if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
      if (reshape.getAllowReorder() || failed(plan(reshape.getSrc(), depth)))
        return failure();
    } else if (auto join = dyn_cast<triton::JoinOp>(op)) {
      RankedTensorType resultType = join.getType();
      if (resultType.getShape().empty() || resultType.getShape().back() != 2 ||
          failed(plan(join.getLhs(), depth + 1)) ||
          failed(plan(join.getRhs(), depth + 1)))
        return failure();
    } else if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      if (isa<RankedTensorType>(splat.getSrc().getType()))
        return failure();
    } else if (auto range = dyn_cast<triton::MakeRangeOp>(op)) {
      if (!range.getType().getElementType().isInteger(32))
        return failure();
    } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      auto elements = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!elements || !elements.isSplat())
        return failure();
    } else {
      if (!op->hasTrait<OpTrait::Elementwise>())
        return failure();
      for (Value operand : op->getOperands()) {
        auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
        if (!tensorType)
          continue;
        if (tensorType.getShape() !=
                cast<RankedTensorType>(value.getType()).getShape() ||
            failed(plan(operand, depth)))
          return failure();
      }
    }

    plannedValues.insert(key);
    active.erase(key);
    return success();
  }

  Value getJoinCondition(unsigned depth, OpBuilder &builder, Location loc) {
    if (auto it = joinConditions.find(depth); it != joinConditions.end())
      return it->second;

    auto indexType = cast<RankedTensorType>(indices.getType());
    auto maskAttr = DenseElementsAttr::get(
        indexType, builder.getI32IntegerAttr(int32_t{1} << depth));
    Value mask = arith::ConstantOp::create(builder, loc, indexType, maskAttr);
    auto zeroAttr =
        DenseElementsAttr::get(indexType, builder.getI32IntegerAttr(0));
    Value zero = arith::ConstantOp::create(builder, loc, indexType, zeroAttr);
    Value masked = arith::AndIOp::create(builder, loc, indices, mask);
    Value condition = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ne, masked, zero);
    joinConditions.try_emplace(depth, condition);
    return condition;
  }

  Value materialize(Value value, unsigned depth, OpBuilder &builder) {
    IndexedValue key{value, depth};
    if (auto it = materializedValues.find(key); it != materializedValues.end())
      return it->second;

    Operation *op = value.getDefiningOp();
    Value result;
    if (auto convert = dyn_cast<ConvertLayoutOp>(op)) {
      result = materialize(convert.getSrc(), depth, builder);
    } else if (auto reshape = dyn_cast<triton::ReshapeOp>(op)) {
      result = materialize(reshape.getSrc(), depth, builder);
    } else if (auto join = dyn_cast<triton::JoinOp>(op)) {
      Value lhs = materialize(join.getLhs(), depth + 1, builder);
      Value rhs = materialize(join.getRhs(), depth + 1, builder);
      Value condition = getJoinCondition(depth, builder, join.getLoc());
      result =
          arith::SelectOp::create(builder, join.getLoc(), condition, rhs, lhs);
    } else if (auto range = dyn_cast<triton::MakeRangeOp>(op)) {
      auto indexType = cast<RankedTensorType>(indices.getType());
      result = indices;
      if (depth != 0) {
        auto shiftAttr =
            DenseElementsAttr::get(indexType, builder.getI32IntegerAttr(depth));
        Value shift = arith::ConstantOp::create(builder, range.getLoc(),
                                                indexType, shiftAttr);
        result = arith::ShRUIOp::create(builder, range.getLoc(), result, shift);
      }

      int64_t start = range.getStartAttr().getInt();
      if (start != 0) {
        auto startAttr =
            DenseElementsAttr::get(indexType, builder.getI32IntegerAttr(start));
        Value offset = arith::ConstantOp::create(builder, range.getLoc(),
                                                 indexType, startAttr);
        result = arith::AddIOp::create(builder, range.getLoc(), result, offset);
      }
    } else {
      RankedTensorType oldType = cast<RankedTensorType>(value.getType());
      auto newType = RankedTensorType::get(
          target.getShape(), oldType.getElementType(), target.getEncoding());

      if (auto splat = dyn_cast<triton::SplatOp>(op)) {
        result = triton::SplatOp::create(builder, splat.getLoc(), newType,
                                         splat.getSrc());
      } else if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        auto oldElements = cast<DenseElementsAttr>(constant.getValue());
        auto newElements = DenseElementsAttr::get(
            newType, oldElements.getSplatValue<Attribute>());
        result = arith::ConstantOp::create(builder, constant.getLoc(), newType,
                                           newElements);
      } else {
        IRMapping mapping;
        for (Value operand : op->getOperands()) {
          if (isa<RankedTensorType>(operand.getType()))
            mapping.map(operand, materialize(operand, depth, builder));
        }
        Operation *replacement = builder.clone(*op, mapping);
        replacement->getResult(0).setType(newType);
        result = replacement->getResult(0);
      }
    }

    materializedValues.try_emplace(key, result);
    return result;
  }

  Operation *insertionPoint;
  Value rootValue;
  RankedTensorType target;
  ConvertLayoutOp conversionRoot;
  triton::StoreOp storeRoot;
  Attribute flatEncoding;
  Value indices;
  DenseSet<IndexedValue> plannedValues;
  DenseSet<IndexedValue> active;
  DenseMap<IndexedValue, Value> materializedValues;
  DenseMap<unsigned, Value> joinConditions;
  SetVector<Operation *> originalOperations;
};

/// A layout conversion is pure. One materialization can therefore serve every
/// dominated request for the same source value and destination encoding.
static void shareDominatingConversions(ModuleOp module) {
  DominanceInfo dominance(module);
  DenseMap<LayoutValue, SmallVector<ConvertLayoutOp>> available;
  SmallVector<ConvertLayoutOp> conversions;
  module.walk([&](ConvertLayoutOp convert) { conversions.push_back(convert); });

  for (ConvertLayoutOp convert : conversions) {
    LayoutValue key{convert.getSrc(), convert.getType().getEncoding()};
    auto &candidates = available[key];
    bool reused = false;
    for (ConvertLayoutOp candidate : candidates) {
      if (!dominance.properlyDominates(candidate.getResult(),
                                       convert.getOperation()))
        continue;
      convert.getResult().replaceAllUsesWith(candidate.getResult());
      convert.erase();
      reused = true;
      break;
    }
    if (!reused)
      candidates.push_back(convert);
  }
}

/// A store is a hard layout boundary even when the producer already has the
/// required type and there is no final conversion to seed a backward walk.
/// Plan its entire scalar/index-rooted expression before considering smaller
/// interior slices; otherwise those slices can obscure the globally optimal
/// zero-conversion assignment.
static void optimizeStoreRootedConversions(ModuleOp module) {
  bool changed;
  do {
    changed = false;
    SmallVector<triton::StoreOp> stores;
    module.walk([&](triton::StoreOp store) { stores.push_back(store); });

    for (triton::StoreOp store : llvm::reverse(stores)) {
      ScalarJoinExpressionPlan plan(store);
      if (failed(plan.analyze()))
        continue;
      plan.rewrite();
      changed = true;
      // A successful whole-expression rewrite may erase conversions and
      // producers from the remaining worklist.
      break;
    }
  } while (changed);
}

static void optimizeScalarRootedConversions(ModuleOp module) {
  bool changed;
  do {
    changed = false;
    SmallVector<ConvertLayoutOp> conversions;
    module.walk(
        [&](ConvertLayoutOp convert) { conversions.push_back(convert); });

    for (ConvertLayoutOp convert : llvm::reverse(conversions)) {
      ScalarRootedLayoutPlan plan(convert);
      if (succeeded(plan.analyze())) {
        plan.rewrite();
        changed = true;
      } else {
        ScalarJoinExpressionPlan joinPlan(convert);
        if (succeeded(joinPlan.analyze())) {
          joinPlan.rewrite();
          changed = true;
        }
      }

      if (!changed)
        continue;
      // Rebuild the worklist: an entire successful producer graph, including
      // other conversions from the old worklist, may have been erased.
      break;
    }
  } while (changed);
}

} // namespace

class TritonGPUOptimizeLayoutsPass
    : public impl::TritonGPUOptimizeLayoutsBase<TritonGPUOptimizeLayoutsPass> {
public:
  using impl::TritonGPUOptimizeLayoutsBase<
      TritonGPUOptimizeLayoutsPass>::TritonGPUOptimizeLayoutsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (failed(optimizeDistributedLayouts(module, disableRematSplitting,
                                          LayoutAssignmentStrategy::Global)))
      return signalPassFailure();

    shareDominatingConversions(module);
    optimizeStoreRootedConversions(module);
    optimizeScalarRootedConversions(module);

    RewritePatternSet patterns(&getContext());
    ConvertLayoutOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu
