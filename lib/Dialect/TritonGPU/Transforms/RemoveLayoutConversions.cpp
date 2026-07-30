#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>
#include <limits>

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZELAYOUTS
#define GEN_PASS_DEF_TRITONGPUREMOVELAYOUTCONVERSIONS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

enum class LayoutAssignmentStrategy { Legacy, Global };

#define DEBUG_TYPE "tritongpu-layout-assignment"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

/// Hardware-sensitive profitability decisions are deliberately separate from
/// legality, candidate discovery, graph search, and rewriting. Both the solver
/// and rematerialization can therefore evolve without embedding policy in the
/// fundamental layout constraints.
class LayoutCostModel {
public:
  uint64_t getTransitionCost(Value value, Attribute sourceEncoding,
                             Attribute resultEncoding) const;
  uint64_t getRegisterPressureCost(Value value, Attribute encoding) const;
  uint64_t getReductionCost(Value value, Attribute encoding,
                            unsigned axis) const;
  uint64_t getExecutionWeight(Operation *op) const;

private:
  mutable DenseMap<std::pair<Type, Type>, uint64_t> transitionCosts;
  mutable DenseMap<std::pair<Type, Attribute>, uint64_t> registerPressureCosts;
  mutable DenseMap<std::pair<Type, unsigned>, uint64_t> reductionCosts;
  mutable DenseMap<Operation *, uint64_t> executionWeights;
};

/// Operation contracts describe layout relationships without choosing which
/// legal relationship is fastest. Candidate discovery and profitability policy
/// remain independent consumers of this graph.
class LayoutConstraintGraph {
public:
  enum class ContractKind { Equality, Transform, Boundary, ControlFlow };
  enum class ValueRole { Data, Address, Mask, Hardware };

  struct Node {
    Value value;
    SmallVector<Attribute, 8> candidates;
    bool fixed;
  };

  struct Contract {
    ContractKind kind;
    Operation *operation;
    SmallVector<Value, 4> values;
    ValueRole role;
  };

  void addNode(Value value, ArrayRef<Attribute> candidates, bool fixed) {
    unsigned index = nodes.size();
    indices.try_emplace(value, index);
    nodes.push_back({value, llvm::to_vector<8>(candidates), fixed});
  }

  void addOperation(Operation *operation, bool fixedBoundary,
                    bool protectedReduction);

  ArrayRef<Node> getNodes() const { return nodes; }
  ArrayRef<Contract> getContracts() const { return contracts; }
  SmallVector<SmallVector<unsigned, 8>, 8> getConnectedComponents() const;

  bool hasJoinConstraints() const { return joinConstraints; }
  bool hasSplitConstraints() const { return splitConstraints; }
  bool hasReductionConstraints() const { return reductionConstraints; }
  bool hasProtectedReductionConstraints() const {
    return protectedReductionConstraints;
  }
  bool hasMultiResultReductionConstraints() const {
    return multiResultReductionConstraints;
  }
  bool hasLoopConstraints() const { return loopConstraints; }
  bool hasTensorMemoryConstraints() const { return tensorMemoryConstraints; }
  bool hasStoreBoundaries() const { return storeBoundaries; }
  unsigned getTensorLoadBoundaryCount() const {
    return tensorLoadBoundaryCount;
  }

private:
  bool contains(Value value) const { return indices.contains(value); }
  void addContract(ContractKind kind, Operation *operation,
                   ArrayRef<Value> values,
                   ValueRole role = ValueRole::Data) {
    SmallVector<Value, 4> tracked;
    for (Value value : values)
      if (contains(value) && !llvm::is_contained(tracked, value))
        tracked.push_back(value);
    if (!tracked.empty())
      contracts.push_back({kind, operation, std::move(tracked), role});
  }

  SmallVector<Node, 32> nodes;
  SmallVector<Contract, 32> contracts;
  DenseMap<Value, unsigned> indices;
  bool joinConstraints = false;
  bool splitConstraints = false;
  bool reductionConstraints = false;
  bool protectedReductionConstraints = false;
  bool multiResultReductionConstraints = false;
  bool loopConstraints = false;
  bool tensorMemoryConstraints = false;
  bool storeBoundaries = false;
  unsigned tensorLoadBoundaryCount = 0;
};

/// Search limits and high-cost graph features are policy, not graph legality.
class LayoutSearchPolicy {
public:
  explicit LayoutSearchPolicy(const LayoutConstraintGraph &graph)
      : graph(graph) {}

  bool hasMemoryFanoutConstraints() const {
    return graph.getTensorLoadBoundaryCount() >= 2 &&
           !graph.hasJoinConstraints() && !graph.hasSplitConstraints() &&
           !graph.hasReductionConstraints() && !graph.hasLoopConstraints() &&
           !graph.hasTensorMemoryConstraints();
  }

  bool hasTensorMemoryReductionConstraints() const {
    return graph.hasTensorMemoryConstraints() && graph.hasReductionConstraints();
  }

  bool useFullObjective() const {
    constexpr unsigned maxFullObjectiveValues = 256;
    return (graph.hasJoinConstraints() || graph.hasSplitConstraints() ||
            graph.hasReductionConstraints() ||
            graph.hasMultiResultReductionConstraints() ||
            hasMemoryFanoutConstraints() ||
            hasTensorMemoryReductionConstraints()) &&
           graph.getNodes().size() <= maxFullObjectiveValues;
  }

  bool useComponentSearch() const {
    return graph.hasJoinConstraints() || graph.hasSplitConstraints() ||
           graph.hasReductionConstraints() ||
           graph.hasMultiResultReductionConstraints() ||
           graph.hasLoopConstraints() || hasMemoryFanoutConstraints() ||
           hasTensorMemoryReductionConstraints();
  }

  unsigned maxComponentProposals() const {
    return useFullObjective() ? 128 : 32;
  }

  bool useExactComponentSearch() const {
    constexpr unsigned maxExactGraphValues = 64;
    return useFullObjective() &&
           graph.getTensorLoadBoundaryCount() > 0 &&
           !graph.hasStoreBoundaries() &&
           graph.getNodes().size() <= maxExactGraphValues;
  }

  unsigned maxExactComponentStates() const { return 256; }

  bool pruneRedundantReductionProposals() const {
    return graph.hasReductionConstraints() &&
           !graph.hasProtectedReductionConstraints();
  }

private:
  const LayoutConstraintGraph &graph;
};

struct LayoutAssignmentChange {
  Value value;
  Attribute originalEncoding;
  Attribute proposedEncoding;
};

/// The solver owns only candidate search, objective comparison, convergence,
/// and rollback. Operation legality, graph construction, and physical costs
/// are supplied separately, so heuristics cannot leak into its core algorithm.
class LayoutAssignmentSolver {
public:
  using Assignment = DenseMap<Value, Attribute>;

  struct Callbacks {
    llvm::function_ref<bool(Value, Attribute, const Assignment &)> isLegal;
    llvm::function_ref<uint64_t(Value, Attribute, const Assignment &)>
        valueCost;
    llvm::function_ref<uint64_t(Value, Attribute)> lowerBound;
    llvm::function_ref<uint64_t(const Assignment &)> fullCost;
    llvm::function_ref<uint64_t(ArrayRef<Value>, const Assignment &)>
        affectedCost;
    llvm::function_ref<bool(Value, Attribute, Assignment &,
                            SmallVectorImpl<LayoutAssignmentChange> &)>
        proposeComponent;
  };

  void solve(const LayoutConstraintGraph &graph,
             const LayoutSearchPolicy &policy, Assignment &assignments,
             const Callbacks &callbacks) const;
};

// The current algorithm works by analyzing the IR and doing a one-shot rewrite
// based on the analysis. The algorithm is as follows.
//
// 1. Find all the anchor ops. These are ops that have a layout we want to
//    preserve.
//
// 2. For each anchor, propagate its layout to all its descendants.
//    An op can have multiple ancestors that are anchors, so at this stage an op
//    may have multiple layouts associated with it.
//
// 3. Resolve conflicts by deciding which of the multiple layouts the op should
//    keep, inserting convert-layout ops to resolve conflicts.  After this
//    stage, each value has only one layout associated with it.
//
// 4. Rewrite the IR by walking the function in dominance order. Since we
//    assume the IR is structured we just need to process the regions in the
//    correct order. For each op, rewrite it using the layout decided by the
//    analysis phase.
class LayoutPropagation {
public:
  // Structure to keep track of the layout associated to a value.
  struct LayoutInfo {
    LayoutInfo(Attribute encoding) { encodings.insert(encoding); }
    LayoutInfo() {}
    llvm::SmallSetVector<Attribute, 8> encodings;
  };
  LayoutPropagation(FuncOp F, LayoutAssignmentStrategy strategy)
      : funcOp(F), strategy(strategy) {}
  // Find the anchor ops and set their layout in the data structure.
  void initAnchorLayout();
  // Recursively Propagate the layout to all the users of the anchor ops until
  // we reach a fix point.
  void propagateLayout();
  // Add layouts given in `Info` to the uses of `value`.
  SmallVector<Value> propagateToUsers(Value value, LayoutInfo &info);
  // Propagate consumer layout requirements back to their tensor producers.
  SmallVector<Value> propagateToOperands(Value value, LayoutInfo &info);
  // Set the encoding to all the values and fill out the values with new layout
  // in `changed`.
  void setEncoding(ValueRange values, LayoutInfo &info,
                   SmallVector<Value> &changed, Operation *op);
  bool addEncoding(Value value, Attribute encoding);
  // Resolve cases where a value has multiple layouts associated to it.
  void resolveConflicts();
  void resolveGlobalConflicts();
  // Rewrite the IR for the full module.
  void rewrite();
  // Rewrite the IR for a region.
  void rewriteRegion(Region &R);
  // Rewrite an op based on the layout picked by the analysis.
  void rewriteOp(Operation *op);
  // Rewrite a for op based on the layout picked by the analysis.
  void rewriteForOp(scf::ForOp forOp);
  void rewriteWhileOp(scf::WhileOp whileOp);
  void rewriteIfOp(scf::IfOp ifOp);
  void rewriteYieldOp(scf::YieldOp yieldOp);
  void rewriteConditionOp(scf::ConditionOp conditionOp);
  void rewriteReduceToScalar(Operation *reduceOp);
  void rewriteAssertOp(AssertOp assertOp);
  Attribute getEncodingBeforeRewrite(Value value) const;
  void setEncodingInPlace(Value value, Attribute encoding);
  void rewriteGenericOpInPlace(Operation *op, Attribute encoding);
  // Return the mapped value in the given encoding. This will insert a convert
  // if the encoding is different than the encoding decided at resolve time.
  Value getValueAs(Value value, Attribute encoding);
  // Dump the current stage of layout information.
  void dump();

private:
  Attribute getCachedSourceEncoding(Operation *op, Attribute encoding) const;
  Attribute getCachedDestinationEncoding(Operation *op,
                                         Attribute encoding) const;
  bool canAssignEncoding(Value value, Attribute encoding,
                         const DenseMap<Value, Attribute> &assignments) const;
  Attribute
  getAssignedEncoding(Value value,
                      const DenseMap<Value, Attribute> &assignments) const;
  SmallVector<Attribute, 4>
  getUseEncodings(OpOperand &use,
                  const DenseMap<Value, Attribute> &assignments) const;
  uint64_t getLayoutTransitionCost(Value value, Attribute sourceEncoding,
                                   Attribute resultEncoding) const;
  uint64_t getLayoutRegisterPressureCost(Value value, Attribute encoding) const;
  uint64_t getLayoutReductionCost(Value value, Attribute encoding,
                                  unsigned axis) const;
  uint64_t getLayoutExecutionWeight(Operation *op) const;
  uint64_t
  getAssignmentCost(Value value, Attribute encoding,
                    const DenseMap<Value, Attribute> &assignments) const;
  uint64_t getAssignmentLowerBound(Value value, Attribute encoding) const;
  uint64_t
  getGlobalAssignmentCost(const DenseMap<Value, Attribute> &assignments) const;
  uint64_t getAffectedAssignmentCost(
      ArrayRef<Value> changed,
      const DenseMap<Value, Attribute> &assignments) const;
  bool buildGlobalComponentProposal(
      Value seed, Attribute encoding, DenseMap<Value, Attribute> &assignments,
      SmallVectorImpl<LayoutAssignmentChange> &changes) const;

  // map from value to layout information.
  llvm::MapVector<Value, LayoutInfo> layouts;
  DenseSet<Value> fixedLayouts;
  // original encodings of tensor values rewritten in place.
  DenseMap<Value, Attribute> originalEncodings;
  mutable DenseMap<std::pair<Operation *, Attribute>, Attribute>
      inferredSourceEncodings;
  mutable DenseMap<std::pair<Operation *, Attribute>, Attribute>
      inferredDestinationEncodings;
  LayoutCostModel costModel;
  FuncOp funcOp;
  LayoutAssignmentStrategy strategy;
};

class LayoutRematerialization {
public:
  LayoutRematerialization(FuncOp F,
                          bool preserveSharedReductionRematerialization = false)
      : funcOp(F), preserveSharedReductionRematerialization(
                       preserveSharedReductionRematerialization) {}
  ~LayoutRematerialization();

  // Map the original value to the remat'ed one.
  void addRematValue(Value old, Attribute encoding, Value newV);
  // Get the remat'ed value in the given encoding, if one already exists and
  // is different then the layout conversion root.
  Value getRematValue(Value value, Attribute encoding) const {
    return rematMapping.lookup(value).lookup(encoding);
  }

  bool backwardRematerialization(bool disableRematSplitting);

  /// Rematerialize the backward slice leading up to \p convertOp to produce the
  /// result layout directly if it is possible and profitable to do so.
  /// \return true if \p convertOp was eliminated, false otherwise.
  bool backwardRematerialization(ConvertLayoutOp convertOp,
                                 bool disableRematSplitting);

  // TODO: Merge the three hoistConvert*(); functions as they are duplicate code
  void hoistConvertDotOperand();
  void hoistConvertOnTopOfExtOrBroadcast(bool disableRematSplitting);
  void hoistConvertIntoConditionals();

  /// Attempt to hoist \p convertOp above operations that make the tensor larger
  /// and costlier to convert (e.g. ExtFOp and BroadcastOp). If this is
  /// possible, rematerialize the slice between the convert and that operation
  /// and hoist the convert above it.
  /// \return true if \p convertOp was hoisted, false otherwise.
  bool hoistConvertOnTopOfExtOrBroadcast(ConvertLayoutOp convertOp,
                                         bool disableRematSplitting);

  /// Attempt to hoist \p convertOp into conditionals so the conversion is only
  /// conditionally executed. If this is possible, rematerialize the slice
  /// between the convert and the conditional and move the convert inside.
  /// \return true if \p convertOp was hoisted, false otherwise.
  bool hoistConvertIntoConditionals(ConvertLayoutOp convertOp);

  bool hoistConvertDotOperand(ConvertLayoutOp convertOp);

  void rewriteSlice(
      SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
      const DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
      ConvertLayoutOp convertOp, IRMapping &mapping);
  void rewriteSlice(
      SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
      const DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
      ConvertLayoutOp convertOp);

  /// Invokes the utility function getConvertBackwardSlice with a callback for
  /// checking whether a rematerialization for a particular value already
  /// exists. Any value that has an existing rematerialization for all of its
  /// uses will have that rematerialization inserted in \p existingRemats, and
  /// will not have its operands traversed for inclusion in \p slice.
  LogicalResult getConvertBackwardSlice(
      OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
      DenseMap<Value, Attribute> &layout,
      DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
      std::function<bool(Operation *)> stopPropagation);

  LogicalResult getRematerializableSlice(
      OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
      DenseMap<Value, Attribute> &layout,
      DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
      std::function<bool(Operation *)> stopPropagation = nullptr);

private:
  void updateRematMapping(SmallVector<std::tuple<Value, Value>> &values);
  // Map values to their rematerializations for a given encoding. We have to be
  // careful about what we put in this map because updateRematMapping only
  // updates keys, and doesn't search for rematerialized values that may be
  // replaced. This means it is only safe to add something to the map as a value
  // if it is either guaranteed to outlive the map, or if it is mapped to some
  // key that we know will always be replaced at the same time (e.g. different
  // block args or results of an scf op).
  DenseMap<Value, DenseMap<Attribute, Value>> rematMapping;
  FuncOp funcOp;
  bool preserveSharedReductionRematerialization;
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
};

LayoutRematerialization::~LayoutRematerialization() {
#ifndef NDEBUG
  DenseSet<Value> live;
  funcOp.walk([&](Block *block) {
    live.insert(block->args_begin(), block->args_end());
    for (Operation &op : *block)
      live.insert(op.result_begin(), op.result_end());
  });
  for (const auto &[key, remats] : rematMapping) {
    assert(live.contains(key) && "remat mapping: key not present");
    for (const auto &[encoding, remat] : remats)
      assert(live.contains(remat) && "remat mapping: value not present");
  }
#endif
}

void LayoutRematerialization::addRematValue(Value old, Attribute encoding,
                                            Value newV) {
  LDBG("addRematValue " << old << " encoding " << encoding << " " << newV);
  rematMapping[old][encoding] = newV;
}

// Return true if the op is an op with a layout we don't want to change. We will
// propagate the layout starting from anchor ops.
bool isLayoutAnchor(Operation *op) {
  if (isa<DescriptorOpInterface>(op))
    return true;
  if (isa<LoadOp, StoreOp>(op))
    return isExpensiveLoadOrStore(op);
  if (isa<DotOpInterface, AtomicRMWOp, AtomicCASOp,
          triton::nvidia_gpu::TMEMLoadOp>(op))
    return true;
  if (auto gatherOp = dyn_cast<GatherOp>(op))
    return gatherOp.getEfficientLayout();

  // Heuristic: Mark permuting reshape as a layout anchor.  Its dst can be
  // anything, so it stops forward-propagation of layouts.  We rely on the
  // backwards pass to fix it up if necessary.  (If we didn't do this, then
  // anything following the reshape won't be covered by the forward pass at
  // all.)
  if (auto reshape = dyn_cast<ReshapeOp>(op))
    return reshape.getAllowReorder();

  return false;
}

static bool hasPairwiseFp8ReductionMemoryProtocol(FuncOp funcOp) {
  bool hasPairwiseReduction = false;
  bool hasWideScalarReduction = false;
  bool hasWideFp8Load = false;
  unsigned stores = 0;
  funcOp.walk([&](Operation *op) {
    if (auto reduce = dyn_cast<ReduceOp>(op)) {
      auto type = dyn_cast<RankedTensorType>(reduce->getOperand(0).getType());
      if (!type)
        return;
      hasPairwiseReduction |=
          type.getRank() == 2 && type.getDimSize(reduce.getAxis()) == 2;
      hasWideScalarReduction |=
          type.getRank() == 1 && type.getDimSize(0) >= 128 &&
          !isa<RankedTensorType>(reduce->getResult(0).getType());
    } else if (auto load = dyn_cast<LoadOp>(op)) {
      auto type = dyn_cast<RankedTensorType>(load.getType());
      hasWideFp8Load |=
          type && type.getRank() == 2 && type.getDimSize(1) >= 128 &&
          isa<Float8E5M2Type, Float8E4M3FNType>(type.getElementType());
    } else if (isa<StoreOp>(op)) {
      ++stores;
    }
  });
  return hasPairwiseReduction && hasWideScalarReduction && hasWideFp8Load &&
         stores == 1;
}

static bool hasPackedMemoryAssemblyProtocol(FuncOp funcOp) {
  unsigned rankedTensorLoadCount = 0;
  unsigned storeCount = 0;
  unsigned reshapeCount = 0;
  bool hasPackedJoin = false;
  bool hasFp8TensorLoad = false;
  bool hasI8ScaleLoad = false;
  bool hasComplexPackedBoundary = false;
  funcOp.walk([&](Operation *op) {
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (auto tensor = dyn_cast<RankedTensorType>(load.getType())) {
        ++rankedTensorLoadCount;
        hasFp8TensorLoad |= isa<Float8E4M3FNType>(tensor.getElementType());
        hasI8ScaleLoad |= tensor.getElementType().isInteger(8);
      }
    } else if (isa<StoreOp>(op)) {
      ++storeCount;
    } else if (isa<JoinOp>(op)) {
      hasPackedJoin = true;
    } else if (isa<ReshapeOp>(op)) {
      ++reshapeCount;
    } else if (isa<ReduceOp, scf::ForOp, scf::WhileOp,
                   DescriptorLoadLikeOpInterface>(op)) {
      hasComplexPackedBoundary = true;
    }
  });

  return (rankedTensorLoadCount == 2 ||
          (rankedTensorLoadCount == 3 && hasPackedJoin && hasFp8TensorLoad &&
           hasI8ScaleLoad)) &&
         storeCount == 1 &&
         (hasPackedJoin ||
          (hasFp8TensorLoad && hasI8ScaleLoad && reshapeCount == 4)) &&
         !hasComplexPackedBoundary;
}

static bool isProtectedLayoutReduction(Operation *op) {
  auto reduce = dyn_cast<ReduceOp>(op);
  if (!reduce)
    return false;

  auto sourceType = dyn_cast<RankedTensorType>(reduce->getOperand(0).getType());
  if (!sourceType || sourceType.getRank() < 4)
    return false;
  if (sourceType.getDimSize(reduce.getAxis()) == 2)
    return true;

  // A shared reduction can determine both a compact scale and the tensor
  // quantized with that scale. Keep the complete reduction in the layout
  // established by the memory boundaries instead of duplicating its network
  // for the independent stores.
  DenseSet<Operation *> visited;
  DenseSet<Operation *> stores;
  SmallVector<Value, 16> worklist;
  for (Value result : reduce->getResults())
    worklist.push_back(result);

  constexpr unsigned maxSharedReductionUsers = 128;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto store = dyn_cast<StoreOp>(user)) {
        if (store.getValue() == value && stores.insert(user).second &&
            stores.size() >= 2)
          return true;
        continue;
      }
      if (!isMemoryEffectFree(user) || !visited.insert(user).second)
        continue;
      if (visited.size() > maxSharedReductionUsers)
        return false;
      for (Value result : user->getResults())
        if (isa<RankedTensorType>(result.getType()))
          worklist.push_back(result);
    }
  }
  return false;
}

static bool isFixedLayoutBoundary(Operation *op) {
  if (isa<scf::ForOp, scf::WhileOp, scf::IfOp, scf::YieldOp, scf::ConditionOp>(
          op))
    return false;

  if (auto reshape = dyn_cast<ReshapeOp>(op))
    if (reshape.getAllowReorder() || reshape.getEfficientLayout())
      return true;

  // Packed assembly observes which register elements are grouped together.
  // Relayout across the instruction can therefore change its logical results.
  if (auto inlineAsm = dyn_cast<ElementwiseInlineAsmOp>(op))
    if (inlineAsm.getPackedElement() > 1)
      return true;

  // High-rank pairwise reductions implement register-level sorting networks.
  // Their reduction axes jointly determine how values are distributed across
  // registers, lanes, and warps; changing one stage independently can
  // duplicate the network and introduce a conversion at every stage.
  if (isProtectedLayoutReduction(op))
    return true;

  // Gather has distinct source and index encodings. Preserve both contracts
  // instead of treating its result layout as a requirement on every operand.
  if (isa<ReturnOp, LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp, GatherOp,
          DotOpInterface, DescriptorOpInterface,
          triton::nvidia_gpu::TMEMLoadOp>(op) ||
      !isMemoryEffectFree(op))
    return true;

  if (op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
      op->hasTrait<OpTrait::Elementwise>() ||
      isa<ConvertLayoutOp, arith::ConstantOp, MakeRangeOp, SplatOp,
          ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp, ReduceOp>(op))
    return false;

  // Preserve unfamiliar tensor producers locally. Operations such as
  // concatenate and histogram can still participate when their existing
  // result layout is explicitly supported by the layout legality interface.
  for (Value result : op->getResults())
    if (auto tensorType = dyn_cast<RankedTensorType>(result.getType()))
      if (!canUseResultEncoding(op, tensorType.getEncoding()))
        return true;
  return false;
}

static bool isProtectedLayoutLoop(Operation *op) {
  if (!isa<scf::ForOp, scf::WhileOp>(op))
    return false;

  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    for (auto [index, argument] : llvm::enumerate(whileOp.getBeforeArguments()))
      if (index >= whileOp.getNumResults() &&
          isa<RankedTensorType>(argument.getType()))
        return true;

  unsigned functionLoadCount = 0;
  unsigned functionStoreCount = 0;
  if (auto parentFunction = op->getParentOfType<FuncOp>())
    parentFunction.walk([&](Operation *nested) {
      if (isa<LoadOp>(nested))
        ++functionLoadCount;
      else if (isa<StoreOp>(nested))
        ++functionStoreCount;
    });

  WalkResult body = op->walk([&](Operation *nested) {
    if (nested == op || !isFixedLayoutBoundary(nested))
      return WalkResult::advance();
    if (!isa<LoadOp, StoreOp>(nested))
      return WalkResult::interrupt();

    // A single masked load/store is one memory protocol. Count the complete
    // function so independent copy loops retain their global layout freedom.
    if (functionLoadCount != 1 || functionStoreCount != 1)
      return WalkResult::advance();
    auto store = dyn_cast<StoreOp>(nested);
    if (!store || !store.getMask())
      return WalkResult::advance();
    auto stripLayoutConversions = [](Value value) {
      while (auto convert = value.getDefiningOp<ConvertLayoutOp>())
        value = convert.getSrc();
      return value;
    };
    auto load =
        stripLayoutConversions(store.getValue()).getDefiningOp<LoadOp>();
    if (load && load.getMask() &&
        stripLayoutConversions(load.getMask()) ==
            stripLayoutConversions(store.getMask()))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return body.wasInterrupted();
}

static bool hasProtectedLayoutLoop(FuncOp funcOp) {
  WalkResult result = funcOp.walk([&](Operation *op) {
    if (isProtectedLayoutLoop(op))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

void LayoutConstraintGraph::addOperation(Operation *operation,
                                         bool fixedBoundary,
                                         bool protectedReduction) {
  if (auto load = dyn_cast<LoadOp>(operation)) {
    if (load->getNumResults() == 1 &&
        isa<RankedTensorType>(load->getResult(0).getType()))
      ++tensorLoadBoundaryCount;
    addContract(ContractKind::Boundary, operation, {load.getPtr()},
                ValueRole::Address);
    if (Value mask = load.getMask())
      addContract(ContractKind::Boundary, operation, {mask}, ValueRole::Mask);
  } else if (auto store = dyn_cast<StoreOp>(operation)) {
    storeBoundaries = true;
    addContract(ContractKind::Boundary, operation, {store.getPtr()},
                ValueRole::Address);
    if (Value mask = store.getMask())
      addContract(ContractKind::Boundary, operation, {mask}, ValueRole::Mask);
  }

  joinConstraints |= isa<JoinOp>(operation);
  splitConstraints |= isa<SplitOp>(operation);
  loopConstraints |= isa<scf::ForOp, scf::WhileOp>(operation);
  tensorMemoryConstraints |= isa<triton::nvidia_gpu::TMEMLoadOp>(operation);
  if (auto reduce = dyn_cast<ReduceOp>(operation)) {
    reductionConstraints = true;
    protectedReductionConstraints |= protectedReduction;
    multiResultReductionConstraints |= reduce->getNumResults() > 1;
  }

  if (fixedBoundary) {
    SmallVector<Value, 8> values(operation->getOperands());
    llvm::append_range(values, operation->getResults());
    addContract(ContractKind::Boundary, operation, values,
                ValueRole::Hardware);
    return;
  }

  if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(operation)) {
    for (unsigned index = 0; index < operation->getNumResults(); ++index)
      addContract(ContractKind::ControlFlow, operation,
                  getTiedArgs(operation, index));
    return;
  }

  if (operation->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
      operation->hasTrait<OpTrait::Elementwise>() ||
      isa<JoinOp, SplitOp, ConvertLayoutOp, ReshapeOp, TransOp, ExpandDimsOp,
          ReduceOp>(operation)) {
    SmallVector<Value, 8> values(operation->getOperands());
    llvm::append_range(values, operation->getResults());
    ContractKind kind =
        operation->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
                operation->hasTrait<OpTrait::Elementwise>()
            ? ContractKind::Equality
            : ContractKind::Transform;
    addContract(kind, operation, values);
  }
}

SmallVector<SmallVector<unsigned, 8>, 8>
LayoutConstraintGraph::getConnectedComponents() const {
  SmallVector<unsigned, 32> parents;
  for (unsigned index = 0; index < nodes.size(); ++index)
    parents.push_back(index);

  auto find = [&](unsigned index) {
    while (parents[index] != index) {
      parents[index] = parents[parents[index]];
      index = parents[index];
    }
    return index;
  };

  for (const Contract &contract : contracts) {
    if (contract.kind == ContractKind::Boundary)
      continue;
    std::optional<unsigned> previous;
    for (Value value : contract.values) {
      unsigned current = indices.lookup(value);
      if (nodes[current].fixed)
        continue;
      if (previous) {
        unsigned lhs = find(*previous);
        unsigned rhs = find(current);
        if (lhs != rhs)
          parents[rhs] = lhs;
      }
      previous = current;
    }
  }

  DenseMap<unsigned, unsigned> componentIndices;
  SmallVector<SmallVector<unsigned, 8>, 8> components;
  for (unsigned index = 0; index < nodes.size(); ++index) {
    if (nodes[index].fixed || nodes[index].candidates.size() <= 1)
      continue;
    unsigned root = find(index);
    auto [component, inserted] =
        componentIndices.try_emplace(root, components.size());
    if (inserted)
      components.emplace_back();
    components[component->second].push_back(index);
  }
  return components;
}

void LayoutPropagation::initAnchorLayout() {
  auto addAnchor = [&](Value v) {
    if (auto tensorType = dyn_cast<RankedTensorType>(v.getType())) {
      layouts.insert({v, LayoutInfo(tensorType.getEncoding())});
      fixedLayouts.insert(v);
    }
  };

  // Consider function args as anchors.  This makes it easier to write tests --
  // you can pass a tensor with an encoding as an arg, instead of explicitly
  // calling tt.load.
  for (auto arg : funcOp.getArguments()) {
    addAnchor(arg);
  }

  if (strategy == LayoutAssignmentStrategy::Global) {
    if (hasPairwiseFp8ReductionMemoryProtocol(funcOp)) {
      funcOp.walk([&](Operation *op) {
        for (Value result : op->getResults())
          addAnchor(result);
        for (Region &region : op->getRegions())
          for (Block &block : region)
            for (BlockArgument argument : block.getArguments())
              addAnchor(argument);
      });
    }
    if (hasPackedMemoryAssemblyProtocol(funcOp)) {
      DenseSet<Value> protectedMemoryAddresses;
      SmallVector<Value> addressWorklist;
      auto addProtectedMemoryAddress = [&](Value value) {
        if (!value || !isa<RankedTensorType>(value.getType()) ||
            !protectedMemoryAddresses.insert(value).second)
          return;
        addAnchor(value);
        addressWorklist.push_back(value);
      };

      // Packed MX kernels independently rematerialize the address and mask
      // of each load and store. Preserve those established memory slices while
      // leaving their loaded and stored data available for global assignment.
      funcOp.walk([&](Operation *op) {
        if (auto load = dyn_cast<LoadOp>(op)) {
          addProtectedMemoryAddress(load.getPtr());
          addProtectedMemoryAddress(load.getMask());
        } else if (auto store = dyn_cast<StoreOp>(op)) {
          addProtectedMemoryAddress(store.getPtr());
          addProtectedMemoryAddress(store.getMask());
        }
      });

      while (!addressWorklist.empty()) {
        Value address = addressWorklist.pop_back_val();
        Operation *producer = address.getDefiningOp();
        if (!producer || !isMemoryEffectFree(producer) ||
            isa<ConvertLayoutOp>(producer))
          continue;
        for (Value operand : producer->getOperands())
          addProtectedMemoryAddress(operand);
      }
    }
    funcOp.walk([&](StoreOp store) {
      if (!store->getParentOfType<scf::ForOp>() &&
          !store->getParentOfType<scf::WhileOp>())
        return;
      addAnchor(store.getPtr());
      if (Value mask = store.getMask())
        addAnchor(mask);
    });
    DenseSet<Value> protectedReductionValues;
    SmallVector<Value> reductionWorklist;
    bool hasSharedReductionMemoryBoundaries = false;
    auto addProtectedReductionValue = [&](Value value) {
      auto tensorType = dyn_cast<RankedTensorType>(value.getType());
      if (!tensorType || tensorType.getRank() < 4 ||
          !protectedReductionValues.insert(value).second)
        return;
      addAnchor(value);
      reductionWorklist.push_back(value);
    };

    funcOp.walk([&](ReduceOp reduce) {
      if (!isProtectedLayoutReduction(reduce.getOperation()))
        return;
      auto sourceType =
          cast<RankedTensorType>(reduce->getOperand(0).getType());
      if (sourceType.getDimSize(reduce.getAxis()) != 2)
        hasSharedReductionMemoryBoundaries = true;
      for (Value operand : reduce->getOperands())
        addProtectedReductionValue(operand);
      for (Value result : reduce->getResults())
        addProtectedReductionValue(result);
    });

    auto protectReductionComponent = [&](Operation *op) {
      if (!op || !isMemoryEffectFree(op) || isa<ConvertLayoutOp>(op) ||
          (isFixedLayoutBoundary(op) && !isProtectedLayoutReduction(op)))
        return;
      for (Value operand : op->getOperands())
        addProtectedReductionValue(operand);
      for (Value result : op->getResults())
        addProtectedReductionValue(result);
    };

    while (!reductionWorklist.empty()) {
      Value value = reductionWorklist.pop_back_val();
      protectReductionComponent(value.getDefiningOp());
      for (OpOperand &use : value.getUses())
        protectReductionComponent(use.getOwner());
    }

    if (hasSharedReductionMemoryBoundaries) {
      funcOp.walk([&](LoadOp load) {
        addAnchor(load.getPtr());
        if (Value mask = load.getMask())
          addAnchor(mask);
      });
    }

    funcOp.walk([&](Operation *op) {
      if (!isProtectedLayoutLoop(op))
        return;

      // Hardware, opaque-operation, and structurally constrained while loops
      // have jointly chosen layouts. Preserve the complete established
      // protocol, including loop initializers, results, region arguments, and
      // every tensor value produced inside the protected loop. Independent
      // components remain available to global assignment.
      for (Value operand : op->getOperands())
        addAnchor(operand);

      op->walk([&](Operation *nested) {
        for (Value result : nested->getResults())
          addAnchor(result);
        for (Region &region : nested->getRegions())
          for (Block &block : region)
            for (BlockArgument argument : block.getArguments())
              addAnchor(argument);
      });
    });

    if (hasProtectedLayoutLoop(funcOp)) {
      DenseSet<Value> protectedStoreAddresses;
      SmallVector<Value> addressWorklist;
      auto addProtectedStoreAddress = [&](Value value) {
        if (!value || !isa<RankedTensorType>(value.getType()) ||
            !protectedStoreAddresses.insert(value).second)
          return;
        addAnchor(value);
        addressWorklist.push_back(value);
      };

      // Tensor-core and tensor-memory loops establish separately
      // rematerialized, coalesced store indices. Preserve that complete
      // address-and-mask slice without constraining stored data or unrelated
      // layout components.
      funcOp.walk([&](StoreOp store) {
        addProtectedStoreAddress(store.getPtr());
        addProtectedStoreAddress(store.getMask());
      });

      while (!addressWorklist.empty()) {
        Value address = addressWorklist.pop_back_val();
        Operation *producer = address.getDefiningOp();
        if (!producer || !isMemoryEffectFree(producer) ||
            isa<ConvertLayoutOp>(producer))
          continue;
        for (Value operand : producer->getOperands())
          addProtectedStoreAddress(operand);
      }
    }
  }

  funcOp.walk([&](Operation *op) {
    if (isLayoutAnchor(op)) {
      for (auto result : op->getResults()) {
        addAnchor(result);
      }
    }

    if (strategy != LayoutAssignmentStrategy::Global ||
        !isFixedLayoutBoundary(op))
      return;

    for (Value result : op->getResults())
      addAnchor(result);

    for (Value operand : op->getOperands()) {
      if (isProtectedLayoutReduction(op)) {
        addAnchor(operand);
      } else if (auto tensorType =
                     dyn_cast<RankedTensorType>(operand.getType())) {
        addEncoding(operand, tensorType.getEncoding());
      }
    }
  });
}

bool LayoutPropagation::addEncoding(Value value, Attribute encoding) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || !encoding)
    return false;

  if (strategy == LayoutAssignmentStrategy::Global) {
    if (fixedLayouts.contains(value) && encoding != tensorType.getEncoding())
      return false;

    constexpr unsigned maxGlobalLayoutCandidates = 16;
    LayoutInfo &info = layouts[value];
    if (!info.encodings.contains(encoding) &&
        info.encodings.size() >= maxGlobalLayoutCandidates)
      return false;
  }

  return layouts[value].encodings.insert(encoding);
}

void LayoutPropagation::setEncoding(ValueRange values, LayoutInfo &info,
                                    SmallVector<Value> &changed,
                                    Operation *op) {
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;
    bool hasChanged = false;
    for (auto encoding : info.encodings) {
      Attribute dstEncoding;
      if (isa<ConvertLayoutOp>(op)) {
        // Try to remove the convert by making the dst encoding match the source
        // encoding.
        dstEncoding = encoding;
      } else {
        dstEncoding = inferDstEncoding(op, encoding);
      }
      if (dstEncoding)
        hasChanged |= addEncoding(value, dstEncoding);
    }
    if (hasChanged)
      changed.push_back(value);
  }
}

SmallVector<Value> LayoutPropagation::propagateToUsers(Value value,
                                                       LayoutInfo &info) {
  SmallVector<Value> changed;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      Value arg = forOp.getTiedLoopRegionIterArg(&use);
      Value result = forOp.getTiedLoopResult(&use);
      setEncoding({arg, result}, info, changed, user);
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
      Value arg = whileOp.getBeforeArguments()[use.getOperandNumber()];
      setEncoding({arg}, info, changed, user);
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto parent = yieldOp->getParentOp();
      SmallVector<Value> valuesToPropagate;
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent) &&
          use.getOperandNumber() < parent->getNumResults())
        valuesToPropagate.push_back(parent->getResult(use.getOperandNumber()));
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        valuesToPropagate.push_back(
            forOp.getRegionIterArg(use.getOperandNumber()));
      if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
        valuesToPropagate.push_back(
            whileOp.getBeforeArguments()[use.getOperandNumber()]);
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent))
        setEncoding(valuesToPropagate, info, changed, user);
      continue;
    }
    if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      // Skip arg 0 as it is the condition.
      unsigned argIndex = use.getOperandNumber() - 1;
      Value afterArg = whileOp.getAfterArguments()[argIndex];
      Value result = whileOp->getResult(argIndex);
      setEncoding({afterArg, result}, info, changed, user);
      continue;
    }
    if (auto dotWaitOp = dyn_cast<nvidia_gpu::WarpGroupDotWaitOp>(user)) {
      unsigned opIndex = use.getOperandNumber();
      Value result = dotWaitOp->getResult(opIndex);
      setEncoding(result, info, changed, user);
      continue;
    }
    if (auto gatherOp = dyn_cast<GatherOp>(user)) {
      // Propagate the layout through the indices only, and if the layout does
      // not have an efficient layout set.
      if (!gatherOp.getEfficientLayout() &&
          &use == &gatherOp.getIndicesMutable()) {
        setEncoding(gatherOp.getResult(), info, changed, user);
        continue;
      }
    }
    if (auto reshapeOp = dyn_cast<ReshapeOp>(user);
        reshapeOp && reshapeOp.getEfficientLayout())
      continue;
    if (user->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
        user->hasTrait<OpTrait::Elementwise>() ||
        isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
            ConvertLayoutOp>(user)) {
      setEncoding(user->getResults(), info, changed, user);
      continue;
    }
  }
  return changed;
}

SmallVector<Value> LayoutPropagation::propagateToOperands(Value value,
                                                          LayoutInfo &info) {
  SmallVector<Value> changed;

  auto addCandidates = [&](ValueRange values, Attribute encoding) {
    for (Value operand : values)
      if (addEncoding(operand, encoding))
        changed.push_back(operand);
  };

  for (Attribute encoding : info.encodings) {
    if (auto result = dyn_cast<OpResult>(value)) {
      Operation *op = result.getOwner();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op)) {
        addCandidates(getTiedArgs(op, result.getResultNumber()), encoding);
        continue;
      }

      Attribute operandEncoding =
          isa<ConvertLayoutOp>(op) ? encoding : inferSrcEncoding(op, encoding);
      if (operandEncoding)
        addCandidates(op->getOperands(), operandEncoding);
      continue;
    }

    auto blockArg = dyn_cast<BlockArgument>(value);
    if (!blockArg)
      continue;
    Operation *parent = blockArg.getOwner()->getParentOp();
    if (!isa<scf::ForOp, scf::WhileOp>(parent))
      continue;

    if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
      if (blockArg.getArgNumber() >= whileOp.getNumResults())
        continue;

    unsigned firstIterArg = isa<scf::ForOp>(parent) ? 1 : 0;
    if (blockArg.getArgNumber() < firstIterArg)
      continue;
    addCandidates(getTiedArgs(parent, blockArg.getArgNumber() - firstIterArg),
                  encoding);
  }

  return changed;
}

void LayoutPropagation::propagateLayout() {
  SmallVector<Value> queue;
  for (auto it : layouts) {
    queue.push_back(it.first);
  }
  while (!queue.empty()) {
    Value currentValue = queue.back();
    LayoutInfo info = layouts[currentValue];
    queue.pop_back();
    SmallVector<Value> changed = propagateToUsers(currentValue, info);
    if (strategy == LayoutAssignmentStrategy::Global) {
      SmallVector<Value> producerChanges =
          propagateToOperands(currentValue, info);
      changed.append(producerChanges.begin(), producerChanges.end());
    }

    LLVM_DEBUG({
      DBGS() << "propagateLayout considering " << currentValue << ", which has "
             << info.encodings.size() << " candidate encoding(s):\n";
      for (Attribute encoding : info.encodings)
        DBGS() << "  " << encoding << "\n";
      DBGS() << "changed: " << changed.size() << "\n";
    });

    queue.insert(queue.end(), changed.begin(), changed.end());
  }
}

void LayoutPropagation::resolveConflicts() {
  if (strategy == LayoutAssignmentStrategy::Global)
    return resolveGlobalConflicts();

  for (auto &it : layouts) {
    Operation *op = it.first.getDefiningOp();
    LayoutInfo &info = it.second;
    if (info.encodings.size() <= 1)
      continue;
    // Hacky resolve, prefer block encoding.
    // TODO: add a proper heuristic.
    Attribute encoding = *info.encodings.begin();
    bool isLoadOrStore =
        op && isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(op);
    for (Attribute e : info.encodings) {
      if ((isLoadOrStore && isa<BlockedEncodingAttr>(e)) ||
          (!isLoadOrStore && isa<MmaEncodingTrait>(e))) {
        encoding = e;
        break;
      }
    }
    info.encodings.clear();
    info.encodings.insert(encoding);
  }
}

uint64_t LayoutCostModel::getTransitionCost(
    Value value, Attribute sourceEncoding, Attribute resultEncoding) const {
  if (!sourceEncoding || !resultEncoding || sourceEncoding == resultEncoding)
    return 0;

  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return 0;

  RankedTensorType sourceType = tensorType.cloneWithEncoding(sourceEncoding);
  RankedTensorType resultType = tensorType.cloneWithEncoding(resultEncoding);
  auto [cached, inserted] = transitionCosts.try_emplace(
      std::pair<Type, Type>{sourceType, resultType}, 0);
  if (!inserted)
    return cached->second;

  if (cvtReordersRegisters(sourceType, resultType))
    return cached->second = 1;

  // Cross-lane traffic scales with the physical element width, so prefer a
  // conversion after narrowing when both placements are otherwise equivalent.
  int64_t elementCount = std::max<int64_t>(32, tensorType.getNumElements());
  int64_t elementBitWidth = 32;
  if (tensorType.getElementType().isIntOrFloat())
    elementBitWidth = std::max<int64_t>(
        8, tensorType.getElementType().getIntOrFloatBitWidth());
  else if (isa<PointerType>(tensorType.getElementType()))
    elementBitWidth = 64;
  uint64_t byteCount = (elementCount * elementBitWidth) / 8;

  if (cvtNeedsWarpShuffle(sourceType, resultType))
    return cached->second = 4 * byteCount;
  return cached->second = 32 * byteCount;
}

uint64_t LayoutCostModel::getRegisterPressureCost(Value value,
                                                 Attribute encoding) const {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || !encoding || !value.getDefiningOp())
    return 0;

  auto [cached, inserted] = registerPressureCosts.try_emplace(
      std::pair<Type, Attribute>{tensorType, encoding}, 0);
  if (!inserted)
    return cached->second;

  uint64_t originalElements = getUniqueElemsPerThread(tensorType);
  uint64_t assignedElements =
      getUniqueElemsPerThread(encoding, tensorType.getShape());
  if (assignedElements <= originalElements)
    return cached->second;

  uint64_t elementBytes = 4;
  if (tensorType.getElementType().isIntOrFloat())
    elementBytes = std::max<uint64_t>(
        1, (tensorType.getElementType().getIntOrFloatBitWidth() + 7) / 8);
  else if (isa<PointerType>(tensorType.getElementType()))
    elementBytes = 8;

  // Concentrating a distributed tile into registers serializes work that
  // could otherwise be performed by a warp. Price that work alongside the
  // warp-sized exchange used for physical layout transitions.
  cached->second = 32 * (assignedElements - originalElements) * elementBytes;
  return cached->second;
}

uint64_t LayoutCostModel::getReductionCost(Value value, Attribute encoding,
                                          unsigned axis) const {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || !encoding || axis >= tensorType.getRank())
    return 0;

  int64_t axisSize = tensorType.getDimSize(axis);
  if (axisSize <= 1)
    return 0;

  RankedTensorType assignedType = tensorType.cloneWithEncoding(encoding);
  auto [cached, inserted] = reductionCosts.try_emplace(
      std::pair<Type, unsigned>{assignedType, axis}, 0);
  if (!inserted)
    return cached->second;

  uint64_t elementBytes = 4;
  if (tensorType.getElementType().isIntOrFloat())
    elementBytes = std::max<uint64_t>(
        1, (tensorType.getElementType().getIntOrFloatBitWidth() + 7) / 8);

  uint64_t rows = tensorType.getNumElements() / axisSize;
  uint64_t lanes = getThreadsPerWarp(encoding, tensorType.getShape())[axis];
  uint64_t warps = getWarpsPerCTA(encoding, tensorType.getShape())[axis];

  // A warp-local reduction exchanges only the participating lanes. Splitting
  // its axis across warps additionally requires shared-memory exchange and
  // CTA-wide synchronization, even when the IR has no explicit conversion.
  cached->second =
      rows * elementBytes * ((lanes - 1) + 32 * lanes * (warps - 1));
  return cached->second;
}

uint64_t LayoutCostModel::getExecutionWeight(Operation *op) const {
  auto [cached, inserted] = executionWeights.try_emplace(op, 1);
  if (!inserted)
    return cached->second;

  uint64_t &weight = cached->second;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<scf::ForOp, scf::WhileOp>(parent))
      weight = std::min<uint64_t>(256, 4 * weight);
  }
  return weight;
}

uint64_t LayoutPropagation::getLayoutTransitionCost(
    Value value, Attribute sourceEncoding, Attribute resultEncoding) const {
  return costModel.getTransitionCost(value, sourceEncoding, resultEncoding);
}

uint64_t
LayoutPropagation::getLayoutRegisterPressureCost(Value value,
                                                 Attribute encoding) const {
  return costModel.getRegisterPressureCost(value, encoding);
}

uint64_t LayoutPropagation::getLayoutReductionCost(Value value,
                                                   Attribute encoding,
                                                   unsigned axis) const {
  return costModel.getReductionCost(value, encoding, axis);
}

uint64_t LayoutPropagation::getLayoutExecutionWeight(Operation *op) const {
  return costModel.getExecutionWeight(op);
}

Attribute LayoutPropagation::getCachedSourceEncoding(Operation *op,
                                                     Attribute encoding) const {
  auto [cached, inserted] = inferredSourceEncodings.try_emplace(
      std::pair<Operation *, Attribute>{op, encoding}, Attribute{});
  if (inserted)
    cached->second = inferSrcEncoding(op, encoding);
  return cached->second;
}

Attribute
LayoutPropagation::getCachedDestinationEncoding(Operation *op,
                                                Attribute encoding) const {
  auto [cached, inserted] = inferredDestinationEncodings.try_emplace(
      std::pair<Operation *, Attribute>{op, encoding}, Attribute{});
  if (inserted)
    cached->second = inferDstEncoding(op, encoding);
  return cached->second;
}

bool LayoutPropagation::canAssignEncoding(
    Value value, Attribute encoding,
    const DenseMap<Value, Attribute> &assignments) const {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || !encoding)
    return false;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *parent = blockArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (blockArg.getArgNumber() == 0)
        return encoding == tensorType.getEncoding();
      return getAssignedEncoding(forOp.getResult(blockArg.getArgNumber() - 1),
                                 assignments) == encoding;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      if (blockArg.getArgNumber() >= whileOp.getNumResults())
        return encoding == tensorType.getEncoding();
      return getAssignedEncoding(whileOp.getResult(blockArg.getArgNumber()),
                                 assignments) == encoding;
    }
    return encoding == tensorType.getEncoding();
  }

  if (auto result = dyn_cast<OpResult>(value)) {
    if (auto forOp = dyn_cast<scf::ForOp>(result.getOwner()))
      return getAssignedEncoding(
                 forOp.getRegionIterArg(result.getResultNumber()),
                 assignments) == encoding;
    if (auto whileOp = dyn_cast<scf::WhileOp>(result.getOwner()))
      return getAssignedEncoding(
                 whileOp.getBeforeArguments()[result.getResultNumber()],
                 assignments) == encoding &&
             getAssignedEncoding(
                 whileOp.getAfterArguments()[result.getResultNumber()],
                 assignments) == encoding;
    if (auto splitOp = dyn_cast<SplitOp>(result.getOwner())) {
      Value sibling = result.getResultNumber() == 0 ? splitOp.getOutRHS()
                                                    : splitOp.getOutLHS();
      return getAssignedEncoding(sibling, assignments) == encoding &&
             static_cast<bool>(getCachedSourceEncoding(splitOp, encoding));
    }
    if (auto reduceOp = dyn_cast<ReduceOp>(result.getOwner())) {
      for (Value sibling : reduceOp->getResults())
        if (isa<RankedTensorType>(sibling.getType()) &&
            getAssignedEncoding(sibling, assignments) != encoding)
          return false;
      return static_cast<bool>(getCachedSourceEncoding(reduceOp, encoding));
    }
  }

  if (encoding == tensorType.getEncoding())
    return true;

  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  if (auto constant = dyn_cast<arith::ConstantOp>(op))
    return isa<DenseElementsAttr>(constant.getValue());
  if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op) ||
      canUseResultEncoding(op, encoding) ||
      getCachedSourceEncoding(op, encoding))
    return true;

  for (Value operand : op->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    if (getCachedDestinationEncoding(
            op, getAssignedEncoding(operand, assignments)) == encoding)
      return true;
  }
  return false;
}

Attribute LayoutPropagation::getAssignedEncoding(
    Value value, const DenseMap<Value, Attribute> &assignments) const {
  if (auto it = assignments.find(value); it != assignments.end())
    return it->second;
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType()))
    return tensorType.getEncoding();
  return {};
}

bool reduceToScalar(Operation *op) {
  // For reductions returning a scalar we can change the src encoding without
  // affecting the output.
  return isa<ReduceOp>(op) && !isa<RankedTensorType>(op->getResultTypes()[0]);
}

SmallVector<Attribute, 4> LayoutPropagation::getUseEncodings(
    OpOperand &use, const DenseMap<Value, Attribute> &assignments) const {
  Operation *user = use.getOwner();
  SmallVector<Attribute, 4> encodings;

  auto addEncoding = [&](Value value) {
    if (Attribute encoding = getAssignedEncoding(value, assignments))
      if (!llvm::is_contained(encodings, encoding))
        encodings.push_back(encoding);
  };

  if (auto forOp = dyn_cast<scf::ForOp>(user)) {
    if (Value result = forOp.getTiedLoopResult(&use))
      addEncoding(result);
    return encodings;
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
    if (use.getOperandNumber() < whileOp.getBeforeArguments().size())
      addEncoding(whileOp.getBeforeArguments()[use.getOperandNumber()]);
    return encodings;
  }
  if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
    Operation *parent = yieldOp->getParentOp();
    unsigned index = use.getOperandNumber();
    if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
      addEncoding(whileOp.getBeforeArguments()[index]);
    else if (isa<scf::ForOp, scf::IfOp>(parent))
      addEncoding(parent->getResult(index));
    return encodings;
  }
  if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
    if (use.getOperandNumber() != 0) {
      auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
      addEncoding(whileOp.getResult(use.getOperandNumber() - 1));
    }
    return encodings;
  }

  // A scalar reduction has no result layout to impose on its input.
  if (reduceToScalar(user))
    return encodings;

  if (isFixedLayoutBoundary(user)) {
    if (auto type = dyn_cast<RankedTensorType>(use.get().getType()))
      encodings.push_back(type.getEncoding());
    return encodings;
  }

  for (Value result : user->getResults()) {
    if (!isa<RankedTensorType>(result.getType()))
      continue;
    Attribute resultEncoding = getAssignedEncoding(result, assignments);
    Attribute operandEncoding = isa<ConvertLayoutOp>(user)
                                    ? resultEncoding
                                    : getCachedSourceEncoding(user,
                                                              resultEncoding);
    if (operandEncoding && !llvm::is_contained(encodings, operandEncoding))
      encodings.push_back(operandEncoding);
  }

  if (encodings.empty())
    if (auto type = dyn_cast<RankedTensorType>(use.get().getType()))
      encodings.push_back(type.getEncoding());
  return encodings;
}

uint64_t LayoutPropagation::getAssignmentCost(
    Value value, Attribute encoding,
    const DenseMap<Value, Attribute> &assignments) const {
  uint64_t cost = 0;
  if (Operation *definingOp = value.getDefiningOp())
    cost += getLayoutRegisterPressureCost(value, encoding) *
            getLayoutExecutionWeight(definingOp);
  llvm::SmallDenseMap<Attribute, uint64_t, 8> userWeights;

  for (OpOperand &use : value.getUses()) {
    uint64_t weight = getLayoutExecutionWeight(use.getOwner());
    if (auto reduce = dyn_cast<ReduceOp>(use.getOwner()))
      cost +=
          getLayoutReductionCost(value, encoding, reduce.getAxis()) * weight;
    for (Attribute required : getUseEncodings(use, assignments)) {
      auto [it, inserted] = userWeights.try_emplace(required, weight);
      if (!inserted)
        it->second = std::max(it->second, weight);
    }
  }

  for (const auto &[required, weight] : userWeights)
    cost += getLayoutTransitionCost(value, encoding, required) * weight;

  Operation *definingOp = value.getDefiningOp();
  if (!definingOp)
    return cost;

  uint64_t weight = getLayoutExecutionWeight(definingOp);
  if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
    auto result = cast<OpResult>(value);
    for (Value tied : getTiedArgs(definingOp, result.getResultNumber())) {
      if (tied == value || !isa<RankedTensorType>(tied.getType()))
        continue;
      cost += getLayoutTransitionCost(
                  tied, getAssignedEncoding(tied, assignments), encoding) *
              weight;
    }
    return cost;
  }

  Attribute operandEncoding = isa<ConvertLayoutOp>(definingOp)
                                  ? encoding
                                  : getCachedSourceEncoding(definingOp,
                                                            encoding);
  if (!operandEncoding)
    return cost;

  for (Value operand : definingOp->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    cost += getLayoutTransitionCost(operand,
                                    getAssignedEncoding(operand, assignments),
                                    operandEncoding) *
            weight;
  }
  return cost;
}

uint64_t LayoutPropagation::getAssignmentLowerBound(Value value,
                                                   Attribute encoding) const {
  uint64_t cost = 0;
  if (Operation *definingOp = value.getDefiningOp())
    cost += getLayoutRegisterPressureCost(value, encoding) *
            getLayoutExecutionWeight(definingOp);
  for (OpOperand &use : value.getUses())
    if (auto reduce = dyn_cast<ReduceOp>(use.getOwner()))
      cost += getLayoutReductionCost(value, encoding, reduce.getAxis()) *
              getLayoutExecutionWeight(use.getOwner());
  return cost;
}

uint64_t LayoutPropagation::getGlobalAssignmentCost(
    const DenseMap<Value, Attribute> &assignments) const {
  uint64_t cost = 0;
  for (const auto &[value, info] : layouts) {
    Attribute encoding = getAssignedEncoding(value, assignments);
    if (encoding)
      cost += getAssignmentCost(value, encoding, assignments);
  }
  return cost;
}

uint64_t LayoutPropagation::getAffectedAssignmentCost(
    ArrayRef<Value> changedValues,
    const DenseMap<Value, Attribute> &assignments) const {
  llvm::SmallSetVector<Value, 32> affected;
  auto addValue = [&](Value value) {
    if (layouts.find(value) != layouts.end())
      affected.insert(value);
  };
  auto addOperation = [&](Operation *op) {
    if (!op)
      return;
    for (Value operand : op->getOperands())
      addValue(operand);
    for (Value result : op->getResults())
      addValue(result);
  };
  auto addControlFlow = [&](Operation *op, unsigned resultIndex) {
    if (!op || resultIndex >= op->getNumResults())
      return;
    for (Value tied : getTiedArgs(op, resultIndex))
      addValue(tied);
  };

  for (Value changed : changedValues) {
    addValue(changed);
    if (Operation *producer = changed.getDefiningOp()) {
      addOperation(producer);
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(producer))
        addControlFlow(producer, cast<OpResult>(changed).getResultNumber());
    } else if (auto blockArgument = dyn_cast<BlockArgument>(changed)) {
      Operation *parent = blockArgument.getOwner()->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (blockArgument.getArgNumber() != 0)
          addControlFlow(forOp, blockArgument.getArgNumber() - 1);
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
        addControlFlow(whileOp, blockArgument.getArgNumber());
      }
    }

    for (OpOperand &use : changed.getUses()) {
      Operation *user = use.getOwner();
      addOperation(user);
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        if (Value result = forOp.getTiedLoopResult(&use))
          addControlFlow(forOp, cast<OpResult>(result).getResultNumber());
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
        addControlFlow(whileOp, use.getOperandNumber());
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        addControlFlow(yieldOp->getParentOp(), use.getOperandNumber());
      } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
        if (use.getOperandNumber() != 0)
          addControlFlow(conditionOp->getParentOp(),
                         use.getOperandNumber() - 1);
      }
    }
  }

  uint64_t cost = 0;
  for (Value value : affected)
    if (Attribute encoding = getAssignedEncoding(value, assignments))
      cost += getAssignmentCost(value, encoding, assignments);
  return cost;
}

bool LayoutPropagation::buildGlobalComponentProposal(
    Value seed, Attribute encoding, DenseMap<Value, Attribute> &assignments,
    SmallVectorImpl<LayoutAssignmentChange> &changes) const {
  constexpr unsigned maxComponentValues = 512;
  DenseMap<Value, Attribute> requested;
  SmallVector<std::pair<Value, Attribute>, 32> worklist;

  auto rollback = [&]() {
    for (const LayoutAssignmentChange &change : llvm::reverse(changes))
      assignments[change.value] = change.originalEncoding;
    changes.clear();
    return false;
  };

  auto request = [&](Value value, Attribute candidate) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !candidate)
      return true;

    auto found = layouts.find(value);
    if (found == layouts.end() || !found->second.encodings.contains(candidate))
      return true;

    // A physical input or memory boundary remains fixed. Its disagreement is
    // represented by a conversion on the component boundary.
    if (fixedLayouts.contains(value) && candidate != type.getEncoding())
      return true;

    if (auto existing = requested.find(value); existing != requested.end())
      return existing->second == candidate;
    if (requested.size() >= maxComponentValues)
      return false;

    requested.try_emplace(value, candidate);
    worklist.emplace_back(value, candidate);
    return true;
  };

  auto isLayoutComponent = [](Operation *op) {
    return op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
           op->hasTrait<OpTrait::Elementwise>() ||
           isa<JoinOp, SplitOp, ConvertLayoutOp, ReshapeOp, TransOp,
               ExpandDimsOp, ReduceOp>(op);
  };

  auto requestLoopComponent = [&](Operation *loopOp, unsigned resultIndex,
                                  Attribute candidate) {
    for (Value tied : getTiedArgs(loopOp, resultIndex))
      if (!request(tied, candidate))
        return false;
    if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
      auto yield =
          cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
      if (!request(yield.getOperand(resultIndex), candidate))
        return false;
    }
    return true;
  };

  if (!request(seed, encoding) || !requested.contains(seed))
    return false;

  while (!worklist.empty()) {
    auto [value, candidate] = worklist.pop_back_val();
    Attribute original = assignments.lookup(value);
    if (original != candidate) {
      changes.push_back({value, original, candidate});
      assignments[value] = candidate;
    }

    if (Operation *producer = value.getDefiningOp()) {
      if (auto splitOp = dyn_cast<SplitOp>(producer)) {
        if (!request(splitOp.getOutLHS(), candidate) ||
            !request(splitOp.getOutRHS(), candidate))
          return rollback();
      }
      if (auto reduceOp = dyn_cast<ReduceOp>(producer))
        for (Value sibling : reduceOp->getResults())
          if (!request(sibling, candidate))
            return rollback();
      if (isa<scf::ForOp, scf::WhileOp>(producer)) {
        auto result = cast<OpResult>(value);
        if (!requestLoopComponent(producer, result.getResultNumber(),
                                  candidate))
          return rollback();
      }
      if (isLayoutComponent(producer) && !isFixedLayoutBoundary(producer)) {
        Attribute source = isa<ConvertLayoutOp>(producer)
                               ? candidate
                               : getCachedSourceEncoding(producer, candidate);
        if (source) {
          for (Value operand : producer->getOperands())
            if (!request(operand, source))
              return rollback();
        }
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      Operation *parent = blockArg.getOwner()->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (blockArg.getArgNumber() != 0 &&
            !requestLoopComponent(forOp, blockArg.getArgNumber() - 1,
                                  candidate))
          return rollback();
      } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
        if (blockArg.getArgNumber() < whileOp.getNumResults() &&
            !requestLoopComponent(whileOp, blockArg.getArgNumber(), candidate))
          return rollback();
      }
    }

    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        if (Value result = forOp.getTiedLoopResult(&use)) {
          if (!requestLoopComponent(
                  forOp, cast<OpResult>(result).getResultNumber(), candidate))
            return rollback();
        }
        continue;
      }
      if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
        if (use.getOperandNumber() < whileOp.getNumResults() &&
            !requestLoopComponent(whileOp, use.getOperandNumber(), candidate))
          return rollback();
        continue;
      }
      if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
        Operation *parent = yieldOp->getParentOp();
        if (isa<scf::ForOp, scf::WhileOp>(parent) &&
            use.getOperandNumber() < parent->getNumResults()) {
          if (!requestLoopComponent(parent, use.getOperandNumber(), candidate))
            return rollback();
        }
        continue;
      }
      if (auto condition = dyn_cast<scf::ConditionOp>(user)) {
        if (use.getOperandNumber() != 0) {
          auto whileOp = cast<scf::WhileOp>(condition->getParentOp());
          if (!requestLoopComponent(whileOp, use.getOperandNumber() - 1,
                                    candidate))
            return rollback();
        }
        continue;
      }
      if (!isLayoutComponent(user) || isFixedLayoutBoundary(user))
        continue;

      Attribute destination = isa<ConvertLayoutOp>(user)
                                  ? candidate
                                  : getCachedDestinationEncoding(user,
                                                                 candidate);
      if (!destination)
        continue;

      for (Value result : user->getResults())
        if (!request(result, destination))
          return rollback();
    }
  }

  for (const auto &[value, candidate] : requested)
    if (!canAssignEncoding(value, candidate, assignments))
      return rollback();
  return true;
}

void LayoutAssignmentSolver::solve(const LayoutConstraintGraph &graph,
                                   const LayoutSearchPolicy &policy,
                                   Assignment &assignments,
                                   const Callbacks &callbacks) const {
  const bool useFullObjective = policy.useFullObjective();
  if (policy.useComponentSearch()) {
    constexpr unsigned maxComponentIterations = 4;
    const unsigned maxComponentProposals = policy.maxComponentProposals();
    const bool pruneRedundantReductionProposals =
        policy.pruneRedundantReductionProposals();
    uint64_t currentCost = callbacks.fullCost(assignments);

    for (unsigned iteration = 0; iteration < maxComponentIterations;
         ++iteration) {
      bool changed = false;
      unsigned proposalCount = 0;
      for (const LayoutConstraintGraph::Node &node : graph.getNodes()) {
        Value value = node.value;
        if (pruneRedundantReductionProposals &&
            (node.fixed || node.candidates.size() <= 1))
          continue;
        for (Attribute candidate : node.candidates) {
          if (pruneRedundantReductionProposals &&
              candidate == assignments.lookup(value))
            continue;
          if (proposalCount++ >= maxComponentProposals)
            break;
          if (node.fixed &&
              candidate !=
                  cast<RankedTensorType>(value.getType()).getEncoding())
            continue;

          SmallVector<LayoutAssignmentChange, 32> proposal;
          if (!callbacks.proposeComponent(value, candidate, assignments,
                                          proposal) ||
              proposal.empty())
            continue;

          SmallVector<Value, 32> changedValues;
          for (const LayoutAssignmentChange &change : proposal)
            changedValues.push_back(change.value);
          uint64_t affectedProposalCost =
              callbacks.affectedCost(changedValues, assignments);
          for (const LayoutAssignmentChange &change : llvm::reverse(proposal))
            assignments[change.value] = change.originalEncoding;
          uint64_t previousCost =
              callbacks.affectedCost(changedValues, assignments);
          uint64_t proposalCost =
              currentCost - previousCost + affectedProposalCost;
          if (proposalCost >= currentCost)
            continue;

          for (const LayoutAssignmentChange &change : proposal)
            assignments[change.value] = change.proposedEncoding;
          currentCost = proposalCost;
          changed = true;
        }
        if (proposalCount >= maxComponentProposals)
          break;
      }
      if (!changed)
        break;
    }
  }

  auto improve = [&](auto &&nodes) {
    bool changed = false;
    for (const LayoutConstraintGraph::Node &node : nodes) {
      Value value = node.value;
      if (node.fixed || node.candidates.size() <= 1)
        continue;

      Attribute original = assignments.lookup(value);
      Attribute best = original;
      uint64_t bestCost = useFullObjective
                              ? callbacks.affectedCost({value}, assignments)
                              : callbacks.valueCost(value, best, assignments);
      for (Attribute candidate : node.candidates) {
        if (candidate == original ||
            !callbacks.isLegal(value, candidate, assignments))
          continue;
        uint64_t candidateCost;
        if (useFullObjective) {
          assignments[value] = candidate;
          candidateCost = callbacks.affectedCost({value}, assignments);
          assignments[value] = original;
        } else {
          candidateCost = callbacks.valueCost(value, candidate, assignments);
        }
        if (candidateCost < bestCost) {
          best = candidate;
          bestCost = candidateCost;
        }
      }
      if (best != assignments.lookup(value)) {
        assignments[value] = best;
        changed = true;
      }
    }
    return changed;
  };

  constexpr unsigned maxAssignmentIterations = 8;
  for (unsigned iteration = 0; iteration < maxAssignmentIterations;
       ++iteration) {
    bool changed = improve(llvm::reverse(graph.getNodes()));
    changed |= improve(graph.getNodes());
    if (!changed)
      break;
  }

  // Keep convergence local to the constrained values; original encodings are
  // verifier-proven and remain the safe incumbent for any invalid assignment.
  for (unsigned iteration = 0; iteration < maxAssignmentIterations;
       ++iteration) {
    bool changed = false;
    for (const LayoutConstraintGraph::Node &node : graph.getNodes()) {
      Value value = node.value;
      if (callbacks.isLegal(value, assignments.lookup(value), assignments))
        continue;
      assignments[value] =
          cast<RankedTensorType>(value.getType()).getEncoding();
      changed = true;
    }
    if (!changed)
      break;
  }

  if (!policy.useExactComponentSearch())
    return;

  // Coordinate descent cannot change jointly constrained encodings when every
  // intermediate assignment is illegal. Solve bounded connected components
  // exactly, pruning with the nonnegative register/reduction objective floor.
  for (const SmallVector<unsigned, 8> &component :
       graph.getConnectedComponents()) {
    if (component.size() < 2)
      continue;
    unsigned states = 1;
    for (unsigned index : component) {
      unsigned candidates = graph.getNodes()[index].candidates.size();
      if (states > policy.maxExactComponentStates() / candidates) {
        states = policy.maxExactComponentStates() + 1;
        break;
      }
      states *= candidates;
    }
    if (states > policy.maxExactComponentStates())
      continue;

    DenseSet<Value> active;
    for (unsigned index : component)
      active.insert(graph.getNodes()[index].value);

    uint64_t fixedLowerBound = 0;
    uint64_t variableLowerBound = 0;
    SmallVector<uint64_t, 8> minimumCosts;
    for (const LayoutConstraintGraph::Node &node : graph.getNodes()) {
      if (!active.contains(node.value)) {
        fixedLowerBound +=
            callbacks.lowerBound(node.value, assignments.lookup(node.value));
        continue;
      }
      uint64_t minimum = std::numeric_limits<uint64_t>::max();
      for (Attribute candidate : node.candidates)
        minimum = std::min(minimum,
                           callbacks.lowerBound(node.value, candidate));
      minimumCosts.push_back(minimum);
      variableLowerBound += minimum;
    }

    uint64_t bestCost = callbacks.fullCost(assignments);
    SmallVector<Attribute, 8> bestAssignments;
    for (unsigned index : component)
      bestAssignments.push_back(
          assignments.lookup(graph.getNodes()[index].value));

    auto search = [&](auto &&self, unsigned position,
                      uint64_t assignedLowerBound,
                      uint64_t remainingLowerBound) -> void {
      if (fixedLowerBound + assignedLowerBound + remainingLowerBound >=
          bestCost)
        return;
      if (position == component.size()) {
        for (const LayoutConstraintGraph::Node &node : graph.getNodes())
          if (!callbacks.isLegal(node.value, assignments.lookup(node.value),
                                 assignments))
            return;
        uint64_t candidateCost = callbacks.fullCost(assignments);
        if (candidateCost >= bestCost)
          return;
        bestCost = candidateCost;
        for (auto [index, value] : llvm::enumerate(component))
          bestAssignments[index] =
              assignments.lookup(graph.getNodes()[value].value);
        return;
      }

      const LayoutConstraintGraph::Node &node =
          graph.getNodes()[component[position]];
      Attribute original = assignments.lookup(node.value);
      remainingLowerBound -= minimumCosts[position];
      for (Attribute candidate : node.candidates) {
        assignments[node.value] = candidate;
        self(self, position + 1,
             assignedLowerBound + callbacks.lowerBound(node.value, candidate),
             remainingLowerBound);
      }
      assignments[node.value] = original;
    };
    search(search, 0, 0, variableLowerBound);
    for (auto [index, value] : llvm::enumerate(component))
      assignments[graph.getNodes()[value].value] = bestAssignments[index];
  }
}

void LayoutPropagation::resolveGlobalConflicts() {
  DenseMap<Value, Attribute> assignments;
  bool hasFlexibleLayouts = false;
  for (auto &[value, info] : layouts) {
    if (info.encodings.empty())
      continue;
    Attribute original = cast<RankedTensorType>(value.getType()).getEncoding();
    info.encodings.insert(original);
    assignments.try_emplace(value, original);
    hasFlexibleLayouts |=
        !fixedLayouts.contains(value) && info.encodings.size() > 1;
  }

  if (hasFlexibleLayouts) {
    LayoutConstraintGraph graph;
    for (const auto &[value, info] : layouts) {
      SmallVector<Attribute, 8> candidates;
      llvm::append_range(candidates, info.encodings);
      graph.addNode(value, candidates, fixedLayouts.contains(value));
    }
    funcOp.walk([&](Operation *op) {
      graph.addOperation(op, isFixedLayoutBoundary(op),
                         isProtectedLayoutReduction(op));
    });
    LayoutSearchPolicy policy(graph);
    LDBG("resolving " << layouts.size() << " layout values with "
                      << (policy.useFullObjective() ? "the full"
                                                    : "the bounded")
                      << " global objective across "
                      << graph.getContracts().size() << " constraints");

    auto isLegal = [&](Value value, Attribute encoding,
                       const auto &current) {
      return canAssignEncoding(value, encoding, current);
    };
    auto valueCost = [&](Value value, Attribute encoding,
                         const auto &current) {
      return getAssignmentCost(value, encoding, current);
    };
    auto lowerBound = [&](Value value, Attribute encoding) {
      return getAssignmentLowerBound(value, encoding);
    };
    auto fullCost = [&](const auto &current) {
      return getGlobalAssignmentCost(current);
    };
    auto affectedCost = [&](ArrayRef<Value> changed, const auto &current) {
      return getAffectedAssignmentCost(changed, current);
    };
    auto propose = [&](Value value, Attribute encoding, auto &current,
                       SmallVectorImpl<LayoutAssignmentChange> &changes) {
      return buildGlobalComponentProposal(value, encoding, current, changes);
    };
    LayoutAssignmentSolver::Callbacks callbacks{
        isLegal, valueCost, lowerBound, fullCost, affectedCost, propose};
    LayoutAssignmentSolver().solve(graph, policy, assignments, callbacks);
  }

  for (auto &[value, info] : layouts) {
    Attribute encoding = assignments.lookup(value);
    if (!encoding)
      continue;
    info.encodings.clear();
    info.encodings.insert(encoding);
  }
}

void LayoutPropagation::dump() {
  for (auto it : layouts) {
    llvm::errs() << "Value: ";
    OpPrintingFlags flags;
    flags.skipRegions();
    it.first.print(llvm::errs(), flags);
    llvm::errs() << " \n encoding:\n";
    for (auto encoding : it.second.encodings) {
      encoding.print(llvm::errs());
      llvm::errs() << "\n";
    }
    llvm::errs() << "--\n";
  }
}

void LayoutPropagation::rewrite() { rewriteRegion(funcOp->getRegion(0)); }

void LayoutPropagation::rewriteRegion(Region &region) {
  std::deque<Region *> queue = {&region};
  while (!queue.empty()) {
    Region *currentRegion = queue.front();
    queue.pop_front();
    for (Operation &op : currentRegion->getOps()) {
      bool needRewrite = false;
      SmallVector<Value> results = op.getResults();
      for (Value result : results) {
        auto it = layouts.find(result);
        // If we haven't mapped this value skip.
        if (it == layouts.end())
          continue;
        LayoutInfo &info = it->second;
        assert(info.encodings.size() == 1 &&
               "we should have resolved to a single encoding");
        auto encoding = cast<RankedTensorType>(result.getType()).getEncoding();
        // If the encoding is already what we want skip.
        if (encoding == *info.encodings.begin())
          continue;
        needRewrite = true;
      }
      if (needRewrite) {
        rewriteOp(&op);
        for (Region &R : op.getRegions())
          queue.push_back(&R);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
        rewriteYieldOp(yieldOp);
      } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(&op)) {
        rewriteConditionOp(conditionOp);
      } else if (reduceToScalar(&op)) {
        rewriteReduceToScalar(&op);
      } else if (auto assertOp = dyn_cast<AssertOp>(&op)) {
        rewriteAssertOp(assertOp);
      } else {
        // If we don't need to rewrite the op we still need to remap the
        // operands.
        for (OpOperand &operand : op.getOpOperands()) {
          auto it = layouts.find(operand.get());
          if (it == layouts.end())
            continue;
          Attribute encoding = getEncodingBeforeRewrite(operand.get());
          Value newOperand = getValueAs(operand.get(), encoding);
          op.setOperand(operand.getOperandNumber(), newOperand);
        }
        for (Region &R : op.getRegions())
          queue.push_back(&R);
      }
    }
  }
}

Value LayoutPropagation::getValueAs(Value value, Attribute encoding) {
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    if (cast<RankedTensorType>(value.getType()).getEncoding() == encoding)
      return value;
    OpBuilder rewriter(value.getContext());
    rewriter.setInsertionPointAfterValue(value);
    auto tmpType = tensorType.cloneWithEncoding(encoding);
    Value converted =
        ConvertLayoutOp::create(rewriter, value.getLoc(), tmpType, value);
    // TODO: we could cache the conversion.
    return converted;
  }
  return value;
}

Attribute LayoutPropagation::getEncodingBeforeRewrite(Value value) const {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return {};
  if (auto it = originalEncodings.find(value); it != originalEncodings.end())
    return it->second;
  return tensorType.getEncoding();
}

void LayoutPropagation::setEncodingInPlace(Value value, Attribute encoding) {
  auto tensorType = cast<RankedTensorType>(value.getType());
  if (!originalEncodings.count(value))
    originalEncodings[value] = tensorType.getEncoding();
  value.setType(tensorType.cloneWithEncoding(encoding));
}

void LayoutPropagation::rewriteGenericOpInPlace(Operation *op,
                                                Attribute encoding) {
  Attribute operandEnc;
  if (op->getNumOperands() > 0) {
    for (Value operand : op->getOperands()) {
      auto it = layouts.find(operand);
      if (it == layouts.end())
        continue;
      Attribute enc = it->second.encodings[0];
      if (inferDstEncoding(op, enc) == encoding) {
        operandEnc = enc;
        break;
      }
    }
    if (!operandEnc)
      operandEnc = inferSrcEncoding(op, encoding);
    assert(operandEnc);
  }
  for (OpOperand &operand : op->getOpOperands()) {
    op->setOperand(operand.getOperandNumber(),
                   getValueAs(operand.get(), operandEnc));
  }
  for (Value result : op->getResults()) {
    auto tensorType = dyn_cast<RankedTensorType>(result.getType());
    if (!tensorType)
      continue;
    setEncodingInPlace(result, encoding);
  }
}

void LayoutPropagation::rewriteForOp(scf::ForOp forOp) {
  for (auto [i, operand, result, regionArg] :
       llvm::enumerate(forOp.getInitArgs(), forOp.getResults(),
                       forOp.getRegionIterArgs())) {
    auto resultTy = dyn_cast<RankedTensorType>(result.getType());
    if (!resultTy)
      continue;
    auto it = layouts.find(result);
    if (it == layouts.end())
      continue;
    Attribute encoding = it->second.encodings[0];
    Value convertedOperand = getValueAs(operand, encoding);
    forOp.getInitArgsMutable()[i].assign(convertedOperand);
    setEncodingInPlace(result, encoding);
    setEncodingInPlace(regionArg, encoding);
  }
}

void LayoutPropagation::rewriteWhileOp(scf::WhileOp whileOp) {
  for (auto [i, operand, beforeArg] :
       llvm::enumerate(whileOp->getOperands(), whileOp.getBeforeArguments())) {
    auto it = layouts.find(beforeArg);
    if (it == layouts.end())
      continue;
    Attribute encoding = it->second.encodings[0];
    Value convertedOperand = getValueAs(operand, encoding);
    whileOp->setOperand(i, convertedOperand);
    setEncodingInPlace(beforeArg, encoding);
  }

  for (auto [result, afterArg] :
       llvm::zip(whileOp.getResults(), whileOp.getAfterArguments())) {
    auto it = layouts.find(result);
    if (it == layouts.end())
      continue;
    Attribute encoding = it->second.encodings[0];
    setEncodingInPlace(result, encoding);
    setEncodingInPlace(afterArg, encoding);
  }
}

void LayoutPropagation::rewriteIfOp(scf::IfOp ifOp) {
  for (unsigned i = 0, e = ifOp->getNumResults(); i < e; ++i) {
    auto it = layouts.find(ifOp.getResult(i));
    if (it == layouts.end())
      continue;
    Attribute encoding = *(it->second.encodings.begin());
    setEncodingInPlace(ifOp.getResult(i), encoding);
  }
}

void LayoutPropagation::rewriteYieldOp(scf::YieldOp yieldOp) {
  Operation *parentOp = yieldOp->getParentOp();
  for (OpOperand &operand : yieldOp->getOpOperands()) {
    Type yieldType = operand.get().getType();
    if (isa<scf::ForOp, scf::IfOp>(parentOp))
      yieldType = parentOp->getResult(operand.getOperandNumber()).getType();
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp))
      yieldType =
          whileOp.getBeforeArguments()[operand.getOperandNumber()].getType();
    auto tensorType = dyn_cast<RankedTensorType>(yieldType);
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    yieldOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteConditionOp(scf::ConditionOp conditionOp) {
  scf::WhileOp whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
  for (unsigned i = 1; i < conditionOp->getNumOperands(); ++i) {
    OpOperand &operand = conditionOp->getOpOperand(i);
    Type argType = whileOp->getResult(operand.getOperandNumber() - 1).getType();
    auto tensorType = dyn_cast<RankedTensorType>(argType);
    if (!tensorType)
      continue;
    Value newOperand = getValueAs(operand.get(), tensorType.getEncoding());
    conditionOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteReduceToScalar(Operation *reduceOp) {
  OpBuilder rewriter(reduceOp);
  Attribute srcEncoding;
  // Since all the operands need to have the same encoding pick the first one
  // and use it for all the operands.
  for (Value operand : reduceOp->getOperands()) {
    auto it = layouts.find(operand);
    if (it != layouts.end()) {
      srcEncoding = it->second.encodings[0];
      break;
    }
  }
  if (!srcEncoding)
    return;
  for (OpOperand &operand : reduceOp->getOpOperands()) {
    Value newOperand = getValueAs(operand.get(), srcEncoding);
    reduceOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutPropagation::rewriteAssertOp(AssertOp assertOp) {
  Attribute srcEncoding;
  // Only need to deal with the first operand which is the condition tensor.
  Value operand = assertOp->getOperand(0);
  auto it = layouts.find(operand);
  if (it == layouts.end())
    return;
  srcEncoding = it->second.encodings[0];
  Value newOperand = getValueAs(operand, srcEncoding);
  assertOp->setOperand(0, newOperand);
}

void LayoutPropagation::rewriteOp(Operation *op) {
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    rewriteForOp(forOp);
  else if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    rewriteWhileOp(whileOp);
  else if (auto ifOp = dyn_cast<scf::IfOp>(op))
    rewriteIfOp(ifOp);
  else {
    Attribute encoding = *layouts[op->getResult(0)].encodings.begin();
    if (canUseResultEncoding(op, encoding)) {
      setEncodingInPlace(op->getResult(0), encoding);
      if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        auto elements = cast<DenseElementsAttr>(constant.getValue());
        auto resultType = cast<RankedTensorType>(constant.getType());
        constant.setValueAttr(elements.reshape(resultType));
      }
    } else if (op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
               op->hasTrait<OpTrait::Elementwise>() ||
               isa<ReduceOp, ExpandDimsOp, ReshapeOp, TransOp, JoinOp, SplitOp,
                   GatherOp, ConvertLayoutOp, nvidia_gpu::WarpGroupDotWaitOp>(
                   op)) {
      rewriteGenericOpInPlace(op, encoding);
    } else {
      llvm::report_fatal_error("unexpected op in rewrite");
    }
  }
}

bool canBeRemat(Operation *op) {
  if (isa<LoadOp, StoreOp>(op))
    return !isExpensiveLoadOrStore(op);
  if (isa<AtomicRMWOp, AtomicCASOp, DotOpInterface>(op))
    return false;
  if (auto gather = dyn_cast<GatherOp>(op))
    return !gather.getEfficientLayout();
  if (auto reshape = dyn_cast<ReshapeOp>(op))
    return !reshape.getEfficientLayout();

  if (isa<scf::WhileOp, scf::ConditionOp>(op))
    return false;

  return true;
}

void LayoutRematerialization::updateRematMapping(
    SmallVector<std::tuple<Value, Value>> &values) {
  for (auto [old, newV] : values) {
    auto it = rematMapping.find(old);
    if (it == rematMapping.end())
      continue;
    auto remats = std::move(it->second);
    rematMapping.erase(it);
    auto &newRemats = rematMapping[newV];
    for (auto [encoding, replacedValue] : remats) {
      // Loop through the replacement value to find the new version of remat
      // value. This should be okay as the number of values should be small.
      for (auto [before, after] : values) {
        if (before == replacedValue) {
          replacedValue = after;
          break;
        }
      }
      newRemats[encoding] = replacedValue;
    }
  }
}

void LayoutRematerialization::rewriteSlice(
    SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
    const DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
    ConvertLayoutOp convertOp, IRMapping &mapping) {
  for (const auto &[value, encoding] : layout) {
    if (Value remat = existingRemats.lookup({value, encoding}))
      mapping.map(value, remat);
  }

  SetVector<Operation *> opsToRewrite;
  // Keep track of yield operands that need to be duplicated.
  DenseMap<Operation *, SmallVector<int>> yieldOperandsMap;
  for (Value v : slice) {
    if (v.getDefiningOp()) {
      opsToRewrite.insert(v.getDefiningOp());
      if (auto ifOp = v.getDefiningOp<scf::IfOp>()) {
        unsigned operandIdx = cast<OpResult>(v).getResultNumber();
        opsToRewrite.insert(ifOp.thenYield().getOperation());
        yieldOperandsMap[ifOp.thenYield()].push_back(operandIdx);
        opsToRewrite.insert(ifOp.elseYield().getOperation());
        yieldOperandsMap[ifOp.elseYield()].push_back(operandIdx);
      }
    } else {
      BlockArgument blockArg = cast<BlockArgument>(v);
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (auto loopOp = cast<LoopLikeOpInterface>(parentOp)) {
        opsToRewrite.insert(loopOp.getOperation());
        OpOperand *operand = loopOp.getTiedLoopYieldedValue(blockArg);
        auto yieldOp = blockArg.getOwner()->getTerminator();
        yieldOperandsMap[yieldOp].push_back(operand->getOperandNumber());
        opsToRewrite.insert(yieldOp);
      }
    }
  }
  opsToRewrite = mlir::topologicalSort(opsToRewrite);

  // replaceAllUsesWith calls delayed until after initial rewrite.
  // This is required for slice.count(value) to work mid rewrite.
  SmallVector<std::tuple<Value, Value>> replacements;

  SmallVector<Operation *> deadOps;
  IRRewriter builder(convertOp.getContext());
  for (Operation *op : opsToRewrite) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      // Keep a mapping of the operands index to the new operands index.
      SmallVector<std::pair<size_t, size_t>> argMapping;
      SmallVector<Value> newOperands;
      for (auto arg : forOp.getRegionIterArgs()) {
        if (slice.count(arg)) {
          OpOperand &initVal = *forOp.getTiedLoopInit(arg);
          argMapping.push_back(std::make_pair(
              forOp.getTiedLoopResult(&initVal).getResultNumber(),
              forOp.getInitArgs().size() + newOperands.size()));
          newOperands.push_back(mapping.lookup(initVal.get()));
        }
      }
      // Create a new for loop with the new operands.
      scf::ForOp newForOp = replaceForOpWithNewSignature(
          builder, forOp, newOperands, replacements);
      deadOps.push_back(forOp.getOperation());
      Block &loopBody = *newForOp.getBody();
      for (auto m : argMapping) {
        mapping.map(forOp.getResult(m.first), newForOp.getResult(m.second));
        int numIndVars = newForOp.getNumInductionVars();
        mapping.map(loopBody.getArgument(m.first + numIndVars),
                    loopBody.getArgument(m.second + numIndVars));
        LLVM_DEBUG({
          DBGS() << "mapping forOp "
                 << loopBody.getArgument(m.first + numIndVars) << " to "
                 << loopBody.getArgument(m.second + numIndVars) << '\n';
        });
        // The result is not in the layout/slice, the argument is.
        Value oldArg = loopBody.getArgument(m.first + numIndVars);
        addRematValue(newForOp.getResult(m.first), layout[oldArg],
                      newForOp.getResult(m.second));
        addRematValue(oldArg, layout[oldArg],
                      loopBody.getArgument(m.second + numIndVars));
      }
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      SmallVector<Type> newTypes;
      for (auto res : ifOp.getResults()) {
        if (slice.count(res)) {
          auto it = layout.find(res);
          assert(it != layout.end());

          auto oldType = cast<RankedTensorType>(res.getType());
          auto newType = oldType.cloneWithEncoding(it->second);
          newTypes.push_back(newType);
        }
      }
      scf::IfOp newIfOp =
          replaceIfOpWithNewSignature(builder, ifOp, newTypes, replacements);
      unsigned oldIdx = 0;
      unsigned newIdx = ifOp.getNumResults();
      for (auto res : ifOp.getResults()) {
        if (slice.count(res)) {
          // Why can't we use res instead of ifOp.getResult(oldIdx)?
          mapping.map(ifOp.getResult(oldIdx), newIfOp.getResult(newIdx));
          addRematValue(ifOp.getResult(oldIdx), layout[res],
                        newIfOp.getResult(newIdx));
          ++newIdx;
        }
        ++oldIdx;
      }
      deadOps.push_back(ifOp.getOperation());
      continue;
    }
    builder.setInsertionPoint(op);
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      auto yieldOperands = llvm::to_vector(yieldOp.getOperands());
      SmallVector<int> operandsToRewrite = yieldOperandsMap[op];
      // Sort so that operands are added in the same order as the new scf
      // results/arguments.
      std::sort(operandsToRewrite.begin(), operandsToRewrite.end());
      for (int operandIdx : operandsToRewrite) {
        yieldOperands.push_back(mapping.lookup(yieldOp.getOperand(operandIdx)));
      }
      scf::YieldOp::create(builder, op->getLoc(), yieldOperands);
      op->erase();
      continue;
    }
    if (isa<arith::ConstantOp>(op)) {
      Operation *newOp = builder.clone(*op);
      auto tensorType = cast<RankedTensorType>(op->getResult(0).getType());
      auto newType = tensorType.cloneWithEncoding(layout[op->getResult(0)]);
      auto cvt = ConvertLayoutOp::create(builder, op->getLoc(), newType,
                                         newOp->getResult(0));
      mapping.map(op->getResult(0), cvt.getResult());
      addRematValue(op->getResult(0), layout[op->getResult(0)],
                    cvt.getResult());
      continue;
    }
    Operation *newOp = builder.clone(*op, mapping);
    for (auto [old, newV] : llvm::zip(op->getResults(), newOp->getResults())) {
      auto it = layout.find(old);
      if (it == layout.end())
        continue;
      auto newType =
          cast<RankedTensorType>(old.getType()).cloneWithEncoding(it->second);
      newV.setType(newType);
      addRematValue(old, it->second, newV);
    }
  }
  // Add the rewritten convert to the replacements so it is removed from the
  // remat maps and has its uses replaced like the other ops we delete.
  replacements.emplace_back(convertOp.getResult(),
                            mapping.lookup(convertOp.getSrc()));

  updateRematMapping(replacements);
  for (auto &kv : replacements) {
    builder.replaceAllUsesWith(std::get<0>(kv), std::get<1>(kv));
  }

  convertOp->erase();
  for (Operation *op : deadOps)
    op->erase();
}

void LayoutRematerialization::rewriteSlice(
    SetVector<Value> &slice, DenseMap<Value, Attribute> &layout,
    const DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
    ConvertLayoutOp convertOp) {
  IRMapping mapping;
  rewriteSlice(slice, layout, existingRemats, convertOp, mapping);
}

LogicalResult LayoutRematerialization::getConvertBackwardSlice(
    OpOperand &root, Attribute rootEncoding, SetVector<Value> &slice,
    DenseMap<Value, Attribute> &layout,
    DenseMap<std::pair<Value, Attribute>, Value> &existingRemats,
    std::function<bool(Operation *)> stopPropagation) {
  // Allow re-using existing conversions for a value if it dominates the use.
  auto getExistingConversion = [&](OpOperand &value, Attribute encoding) {
    Value remat = getRematValue(value.get(), encoding);
    if (!remat)
      return Value();
    // `value` can be replaced with an existing rematerialization if it
    // dominates the current use of value.
    Operation *user = value.getOwner();
    if (domInfo.properlyDominates(remat, user)) {
      existingRemats.try_emplace({value.get(), encoding}, remat);
      return remat;
    }
    // FIXME: If the current user is a conversion, then we know it will become
    // a no-op when its operand is replaced with `remat`, but we need to check
    // that its users are all dominated by `remat` so the IR is valid.
    // if (isa<ConvertLayoutOp>(user) && remat.getDefiningOp() &&
    //     domInfo.properlyDominates(user, remat.getDefiningOp())) {
    //   for (Operation *op : user->getUsers()) {
    //     if (!domInfo.dominates(remat, op))
    //       return Value();
    //   }
    //   return remat;
    // }

    // There is an existing rematerialization, but it doesn't dominate all the
    // uses we care about, so ensure it isn't used.
    existingRemats[{value.get(), encoding}] = Value();
    return Value();
  };

  return mlir::getConvertBackwardSlice(root, slice, rootEncoding, layout,
                                       stopPropagation, getExistingConversion);
}

LogicalResult LayoutRematerialization::getRematerializableSlice(
    OpOperand &root, Attribute rootEncoding, SetVector<Value> &sliceArg,
    DenseMap<Value, Attribute> &layoutArg,
    DenseMap<std::pair<Value, Attribute>, Value> &existingRematsArg,
    std::function<bool(Operation *)> stopPropagation) {
  // Operate on copies of the input, we do not want to modify them unless we
  // have succeeded.
  auto slice = sliceArg;
  auto layout = layoutArg;
  auto existingRemats = existingRematsArg;
  LogicalResult result = getConvertBackwardSlice(
      root, rootEncoding, slice, layout, existingRemats, stopPropagation);
  if (result.failed())
    return failure();

  // Check if all the operations in the slice can be rematerialized.
  for (Value v : slice) {
    if (Operation *op = v.getDefiningOp()) {
      if (!canBeRemat(op))
        return failure();
    }
  }
  sliceArg = std::move(slice);
  layoutArg = std::move(layout);
  existingRematsArg = std::move(existingRemats);
  return success();
}

bool LayoutRematerialization::backwardRematerialization(
    bool disableRematSplitting) {
  bool changed = false;
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    if (!backwardRematerialization(convertOp, disableRematSplitting)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    } else {
      changed = true;
    }
  }
  return changed;
}

void LayoutRematerialization::hoistConvertOnTopOfExtOrBroadcast(
    bool disableRematSplitting) {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    if (!hoistConvertOnTopOfExtOrBroadcast(convertOp, disableRematSplitting)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

void LayoutRematerialization::hoistConvertIntoConditionals() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    if (!hoistConvertIntoConditionals(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

static bool isExpensiveMathOp(Operation *op) {
  // These operations are either multiple instructions or have throughput
  // lower than 16 according to the arithmetic instructions table in:
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions
  return isa<arith::DivFOp, math::ErfcOp, math::SinhOp, math::CoshOp,
             math::TanhOp, math::AsinhOp, math::AcoshOp, math::AtanhOp,
             math::CtPopOp, math::CountLeadingZerosOp,
             math::CountTrailingZerosOp, math::ExpOp, math::Exp2Op,
             math::ExpM1Op, math::LogOp, math::Log2Op, math::Log10Op,
             math::Log1pOp, math::SinOp, math::CosOp, math::TanOp, math::AsinOp,
             math::AcosOp, math::AtanOp, math::Atan2Op, math::PowFOp,
             math::SqrtOp, math::RsqrtOp, math::ErfOp, math::CbrtOp>(op);
}

static int64_t getByteCount(Value result, int64_t minElementCount = 0,
                            int64_t minBitWidth = 0) {
  int64_t elementCount = 0;
  int64_t dtypeBitWidth = 0;
  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    elementCount = tensorTy.getNumElements();
    auto elemType = tensorTy.getElementType();
    if (elemType.isIntOrFloat()) {
      dtypeBitWidth = elemType.getIntOrFloatBitWidth();
    }
  }
  if (elementCount < minElementCount) {
    elementCount = minElementCount;
  }
  if (dtypeBitWidth < minBitWidth) {
    dtypeBitWidth = minBitWidth;
  }
  return (elementCount * dtypeBitWidth) >> 3;
}

/// Compute the cost of a ConvertLayoutOp with source \p convertSrc and result
/// encoding \p resultEncoding.
int64_t getConvertCost(Value convertSrc, Attribute resultEncoding) {
  auto srcType = cast<RankedTensorType>(convertSrc.getType());
  auto resultType = srcType.cloneWithEncoding(resultEncoding);
  if (cvtReordersRegisters(srcType, resultType))
    return 0;

  // Measure the number of bytes that we're manipulating with the
  // ConvertLayoutOp. We pessimistically assume that we round-trip
  // through shared memory and that we cannot vectorise sub-register
  // loads/stores, so we set a minimum element count of 32 (the warp
  // size and number of shared memory banks) and minimum bitwidth of
  // 32 (the width per bank of the shared memory load/store unit).
  auto convertLayoutBytes = getByteCount(convertSrc, 32, 32);
  // We measure costs in standardised milli-SM-cycles. The smem load
  // and store each cost 8 * convertLayoutBytes, and then we double
  // it to account for extra cost due to synchronisation.
  return 32 * convertLayoutBytes;
}

static unsigned getCostFactor(Value result, Attribute rematEncoding) {
  auto tensorType = cast<RankedTensorType>(result.getType());
  unsigned oldElemsPerThread = getUniqueElemsPerThread(tensorType);
  unsigned newElemsPerThread =
      getUniqueElemsPerThread(rematEncoding, tensorType.getShape());
  return std::max(1u, newElemsPerThread / oldElemsPerThread);
}

/// Determine whether rematerializing \p slice is beneficial given that it will
/// eliminate \p convertOp and require creating new convert ops with cost \p
/// newCvtCost.
bool isRematBeneficial(ConvertLayoutOp convertOp, const SetVector<Value> &slice,
                       const DenseMap<Value, Attribute> &layout,
                       int64_t newCvtCost, bool disableRematSplitting,
                       bool preserveSharedReductionRematerialization = false) {
  // Identify all operations in the slice
  SetVector<Operation *> sliceOps;
  for (Value v : slice) {
    if (Operation *op = v.getDefiningOp()) {
      sliceOps.insert(op);
    }
  }

  // Determine which values used by operations outside the slice. We can use
  // this to determine whether they will actually survive and therefore need to
  // contribute to the cost.
  SetVector<Value> nonSliceOnlyValues;

  // Identify values that directly have uses outside the slice.
  for (Value v : slice) {
    for (auto &use : v.getUses()) {
      auto *user = use.getOwner();
      if (user == convertOp || sliceOps.contains(user))
        continue;
      // For region branch ops, check whether the values they flow into are in
      // the slice or unused instead.
      if (isa<RegionBranchTerminatorOpInterface>(user))
        user = user->getParentOp();
      if (auto rbi = dyn_cast<RegionBranchOpInterface>(user)) {
        RegionBranchSuccessorMapping mapping;
        rbi.getSuccessorOperandInputMapping(mapping);
        auto it = mapping.find(&use);
        if (it != mapping.end()) {
          // We have found the values this use flows into, check if they are
          // used outside the slice.
          bool isSliceOnly = llvm::all_of(it->second, [&](Value v) {
            return slice.contains(v) || v.use_empty();
          });
          if (isSliceOnly)
            continue;
        }
      }
      nonSliceOnlyValues.insert(v);
      break;
    }
  }

  // Expand the set to all transitive operands in the slice.
  for (size_t i = 0; i < nonSliceOnlyValues.size(); ++i) {
    Value v = nonSliceOnlyValues[i];
    auto *op = v.getDefiningOp();
    // If the operand is a block argument, get the enclosing op.
    op = op ? op : v.getParentBlock()->getParentOp();
    if (auto rbi = dyn_cast<RegionBranchOpInterface>(op)) {
      // Try to determine the operands that flow into this value, and mark them
      // as being used outside the slice.
      RegionBranchInverseSuccessorMapping mapping;
      rbi.getSuccessorInputOperandMapping(mapping);
      auto it = mapping.find(v);
      if (it != mapping.end()) {
        for (auto tiedOperand : it->second)
          if (slice.contains(tiedOperand->get()))
            nonSliceOnlyValues.insert(tiedOperand->get());
        continue;
      }
    }
    // In the general case, propagate to all operands of the op.
    for (auto operand : op->getOperands())
      if (slice.contains(operand))
        nonSliceOnlyValues.insert(operand);
  }

  if (disableRematSplitting && !nonSliceOnlyValues.empty()) {
    LDBG("  skipped rematerialization because it would split the slice");
    return false;
  }

  int64_t convertLayoutCost =
      getConvertCost(convertOp.getSrc(), convertOp.getType().getEncoding());
  int64_t rematerialisationCost = newCvtCost;

  // Evaluate single-use status for every operation in slice
  for (Operation *op : sliceOps) {
    auto dialect = op->getDialect();
    bool isOpUsedOutsideSlice = llvm::any_of(op->getResults(), [&](Value v) {
      return nonSliceOnlyValues.contains(v);
    });

    if (preserveSharedReductionRematerialization &&
        isOpUsedOutsideSlice && isa<ReduceOp>(op)) {
      LDBG("  skipped rematerialization because it would duplicate a shared "
           "reduction");
      return false;
    }

    if (isa<arith::ConstantOp>(op)) {
      // special-case: arith.constant has zero cost
      continue;
    } else if (isa<arith::ArithDialect, math::MathDialect>(dialect)) {
      // this is an arithmetic operation; we distinguish between cheap
      // operations (such as floating point add/mul which can be fused
      // as halves of a single-cycle FMA instruction) and expensive
      // operations which use the special function unit and/or involve
      // multiple instructions.
      int64_t multiplier = isExpensiveMathOp(op) ? 8 : 1;
      for (Value result : op->getResults()) {
        Attribute rematEncoding = layout.lookup(result);
        int64_t cost = multiplier * getByteCount(result);
        // If the new layout increases the amount of work that needs to happen
        // on each thread, account for that.
        unsigned factor = getCostFactor(result, rematEncoding);
        if (!isOpUsedOutsideSlice)
          factor -= 1;
        rematerialisationCost += cost * factor;
      }
      continue;
    }

    // If all of the results of the op are only used within the slice, when we
    // rematerialise, this operation does not get duplicated so it does not
    // contribute to our cost model.
    if (!isOpUsedOutsideSlice)
      continue;

    if (isa<LoadOp>(op) || isa<LocalLoadOp>(op)) {
      // optimistically assume L1-cached:
      for (Value result : op->getResults()) {
        rematerialisationCost += 8 * getByteCount(result);
      }
    } else if (isa<ReduceOp>(op)) {
      // Reduce op introduce much cost.
      auto reduceOp = dyn_cast<ReduceOp>(op);
      ReduceOpHelper helper(reduceOp);
      if (!helper.isAssociative()) {
        // We shouldn't rematerize a no associative reduce op if it has multiple
        // use chain.
        LDBG("  skipped rematerialization due to non-associative reduce in the "
             "slice");
        return false;
      }
      rematerialisationCost += helper.getIntraWarpSizeWithUniqueData();
      rematerialisationCost += 8 * helper.getInterWarpSizeWithUniqueData();
    }
  }

  LLVM_DEBUG({
    DBGS() << "  convert layout cost: " << convertLayoutCost << "\n";
    DBGS() << "  rematerialisation cost: " << rematerialisationCost << "\n";
  });

  return convertLayoutCost >= rematerialisationCost;
}

bool LayoutRematerialization::backwardRematerialization(
    ConvertLayoutOp convertOp, bool disableRematSplitting) {
  // DotOperand is hoisted by hoistDotOperand
  RankedTensorType targetType = convertOp.getType();
  if (isa<DotOperandEncodingAttr>(targetType.getEncoding()))
    return false;
  Value oldV = convertOp.getSrc();
  LDBG("check backward remat with source " << oldV << " encoding "
                                           << targetType.getEncoding());
  // 1. Take a backward slice of all the tensor dependencies that can be
  // rematerialized.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  DenseMap<std::pair<Value, Attribute>, Value> existingRemats;
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout,
      existingRemats);
  if (result.failed()) {
    LDBG("  getRematerializableSlice failed");
    return false;
  }

  // 2. Determine whether rematerialisation is beneficial.
  if (!isRematBeneficial(convertOp, slice, layout, /*newCvtCost=*/0,
                         disableRematSplitting,
                         preserveSharedReductionRematerialization)) {
    LDBG("  skipped rematerialization because it is not beneficial");
    return false;
  }

  LLVM_DEBUG({
    DBGS() << "  remat convert op " << convertOp << '\n';
    for (Value v : slice)
      DBGS() << "    " << v << '\n';
  });

  // 3. Rewrite the slice.
  rewriteSlice(slice, layout, existingRemats, convertOp);
  return true;
}

void LayoutRematerialization::hoistConvertDotOperand() {
  // Go through each ConvertLayoutOp.
  SmallVector<ConvertLayoutOp> convertOps;
  funcOp.walk(
      [&](ConvertLayoutOp convertOp) { convertOps.push_back(convertOp); });
  for (ConvertLayoutOp convertOp : convertOps) {
    if (!hoistConvertDotOperand(convertOp)) {
      // If the conversion didn't get removed, consider it for reuse in future
      // backward slices.
      addRematValue(convertOp.getSrc(), convertOp.getType().getEncoding(),
                    convertOp.getResult());
    }
  }
}

bool LayoutRematerialization::hoistConvertDotOperand(
    ConvertLayoutOp convertOp) {
  auto targetType = convertOp.getType();
  // The pass is targeted to MMA dot operands

  auto canBePipelined = [&](ConvertLayoutOp convertOp) {
    // FIXME: Check that the parent is a for loop
    auto parent = convertOp->getParentOp();
    if (!parent)
      return false;

    // Find all the dot-like ops in the for loop that have a dot operand
    // encoding on the lhs and check if any of them post-dominates the load +
    // cvt
    SmallVector<Operation *> dotLikeOps;
    parent->walk([&](Operation *op) {
      if (!isa<mlir::triton::DotOpInterface>(op))
        return;
      auto opType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      if (!opType)
        return;
      auto dotEnc = dyn_cast<DotOperandEncodingAttr>(opType.getEncoding());
      if (!dotEnc)
        return;
      if (isa<MmaEncodingTrait>(dotEnc.getParent()))
        dotLikeOps.push_back(op);
    });
    if (dotLikeOps.empty())
      return false;
    return llvm::any_of(dotLikeOps, [&](Operation *dot) {
      return postDomInfo.postDominates(dot, convertOp);
    });
  };

  // We move convert #dot_operand next to their loads. This is done
  // so that it's then easy to pipeline these loads
  if (!canBePipelined(convertOp))
    return false;

  // We hoist over any operation that can be done without data movement between
  // threads We do views and elementwise pure ops for now
  auto noDataMovement = [](Operation *op) {
    return (op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)) ||
           isa<BroadcastOp, Fp4ToFpOp, ConvertLayoutOp, UpcastFpOpInterface>(
               op) ||
           isView(op);
  };
  // Stop the slice as soon as we find an operation that cannot be done without
  // data movement between threads
  auto stop = std::not_fn(noDataMovement);

  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  DenseMap<std::pair<Value, Attribute>, Value> existingRemats;
  // Set-up the conversion "cache"
  LogicalResult result = getConvertBackwardSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout,
      existingRemats, stop);
  if (result.failed())
    return false;

  IRMapping mapping;
  OpBuilder builder(convertOp.getContext());
  SetVector<Value> innerSlice;
  for (Value v : slice) {
    if (!v.getDefiningOp()) {
      LLVM_DEBUG(
          { DBGS() << "  Block arguments not supported. Got " << v << "\n"; });
      return false;
    }

    // We expect the leaves of the slice to be Load, descriptor load-like ops,
    // or arith::Constant. This could be generalised if necessary.
    if (!isa<LoadOp, DescriptorLoadLikeOpInterface>(v.getDefiningOp())) {
      auto op = v.getDefiningOp();
      if (isa<arith::ConstantOp>(op) || noDataMovement(op)) {
        innerSlice.insert(v);
        continue;
      } else {
        LLVM_DEBUG({
          DBGS() << "  Leaves must be Load, descriptor load-like ops, or "
                    "Constant. Got "
                 << v << "\n";
        });
        return false;
      }
    }
    Operation *loadOp = v.getDefiningOp();
    builder.setInsertionPointAfter(loadOp);
    auto type = dyn_cast<RankedTensorType>(loadOp->getResult(0).getType());
    if (!type)
      continue;
    auto newType = type.cloneWithEncoding(layout[loadOp->getResult(0)]);
    auto newConvertOp = ConvertLayoutOp::create(builder, convertOp.getLoc(),
                                                newType, loadOp->getResult(0));
    mapping.map(loadOp->getResult(0), newConvertOp.getResult());
  }

  if (innerSlice.empty()) {
    return false;
  }

  LLVM_DEBUG({
    DBGS() << "  Hoisting " << convertOp << '\n';
    for (Value v : innerSlice)
      DBGS() << "    " << v << '\n';
  });

  rewriteSlice(innerSlice, layout, existingRemats, convertOp, mapping);
  return true;
}

// For convert left we try to hoist them above type extension to reduce the cost
// of the convert.
bool LayoutRematerialization::hoistConvertOnTopOfExtOrBroadcast(
    ConvertLayoutOp convertOp, bool disableRematSplitting) {
  // DotOperand is hoisted by hoistDotOperand
  RankedTensorType targetType = convertOp.getType();
  if (isa<DotOperandEncodingAttr>(targetType.getEncoding()))
    return false;

  auto isExtOrBroadcastOp = [](Operation *op) {
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, BroadcastOp,
            ExpandDimsOp>(op)) {
      return true;
    }
    if (auto fpToFpOp = dyn_cast<FpToFpOp>(op)) {
      auto srcType = cast<RankedTensorType>(fpToFpOp.getOperand().getType());
      return getElementBitWidth(srcType) <
             getElementBitWidth(cast<RankedTensorType>(fpToFpOp.getType()));
    }
    return false;
  };
  // 1. Take a backward slice of all the tensor dependencies.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  DenseMap<std::pair<Value, Attribute>, Value> existingRemats;
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), slice, layout,
      existingRemats, isExtOrBroadcastOp);
  if (result.failed())
    return false;

  Operation *extOrBroadcastOp = nullptr;
  unsigned sliceSize = slice.size();
  for (unsigned i = 0; i < sliceSize; i++) {
    Value v = slice[i];
    Operation *op = v.getDefiningOp();
    if (!op || !isExtOrBroadcastOp(op))
      continue;

    Attribute srcEncoding = inferSrcEncoding(op, layout[v]);
    if (!srcEncoding)
      return false;

    // If we can rematerialize the rest of the ext slice we can ignore this ext
    // as it won't need a convert.
    if (succeeded(getRematerializableSlice(op->getOpOperand(0), srcEncoding,
                                           slice, layout, existingRemats)))
      continue;

    // Only apply it if there is a single ext op otherwise we would have to
    // duplicate the convert.
    if (extOrBroadcastOp != nullptr)
      return false;
    extOrBroadcastOp = op;
  }

  if (extOrBroadcastOp == nullptr)
    return false;
  Attribute dstEncoding = layout[extOrBroadcastOp->getResult(0)];
  Attribute srcEncoding = inferSrcEncoding(extOrBroadcastOp, dstEncoding);
  if (!srcEncoding)
    return false;
  int64_t newCvtCost =
      getConvertCost(extOrBroadcastOp->getOperand(0), srcEncoding);
  if (!isRematBeneficial(convertOp, slice, layout, newCvtCost,
                         /*disableRematSplitting=*/disableRematSplitting))
    return false;
  // Move the convert before the ext op and rewrite the slice.
  OpBuilder builder(extOrBroadcastOp);
  auto tensorType =
      cast<RankedTensorType>(extOrBroadcastOp->getOperand(0).getType());
  auto newType = tensorType.cloneWithEncoding(srcEncoding);
  auto newConvertOp = ConvertLayoutOp::create(
      builder, convertOp.getLoc(), newType, extOrBroadcastOp->getOperand(0));
  Operation *newExtOrBroadcast = builder.clone(*extOrBroadcastOp);
  newExtOrBroadcast->setOperand(0, newConvertOp.getResult());
  auto oldExtOrBroadcastType =
      cast<RankedTensorType>(extOrBroadcastOp->getResult(0).getType());
  Type newExtOrBroadcastType =
      oldExtOrBroadcastType.cloneWithEncoding(dstEncoding);
  newExtOrBroadcast->getResult(0).setType(newExtOrBroadcastType);
  IRMapping mapping;
  mapping.map(extOrBroadcastOp->getResult(0), newExtOrBroadcast->getResult(0));
  slice.remove(extOrBroadcastOp->getResult(0));
  // 3. Rewrite the slice.
  rewriteSlice(slice, layout, existingRemats, convertOp, mapping);
  return true;
}

bool LayoutRematerialization::hoistConvertIntoConditionals(
    ConvertLayoutOp convertOp) {
  // Take the backward slice of tensor dependencies rooted at the conversion,
  // stopping at conditionals. This subslice is used to initialize the analysis.
  SetVector<Value> slice;
  DenseMap<Value, Attribute> layout;
  DenseMap<std::pair<Value, Attribute>, Value> existingRemats;
  auto isIfOp = [](Operation *op) { return isa<scf::IfOp>(op); };
  if (failed(getRematerializableSlice(convertOp.getSrcMutable(),
                                      convertOp.getType().getEncoding(), slice,
                                      layout, existingRemats, isIfOp)))
    return false;

  // These are the conditional edges above which conversions should be hoisted.
  // The value represents the `scf.if` op result and the operand represents the
  // edge into one of the branches.
  SmallVector<std::pair<Value, OpOperand *>> hoistAbove;

  // The list of `scf.if` op results in the slice that are not rematerializable.
  // Hoisting is terminated at these values.
  SmallVector<OpResult> terminals;

  // This loop recurses through the subslices of the backwards dependencies, so
  // re-query the size of `slice`.
  for (unsigned i = 0; i != slice.size(); ++i) {
    Value v = slice[i];
    auto ifOp = v.getDefiningOp<scf::IfOp>();
    if (!ifOp)
      continue;

    Attribute rootLayout = layout.at(v);
    unsigned resIdx = cast<OpResult>(v).getResultNumber();

    // Take the backward slice along each branch.
    auto thenYield =
        cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
    auto elseYield =
        cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

    OpOperand &thenRes = thenYield.getResultsMutable()[resIdx];
    OpOperand &elseRes = elseYield.getResultsMutable()[resIdx];

    auto newSlice = slice;
    auto newLayout = layout;
    auto newExistingRemats = existingRemats;

    LogicalResult thenResult = getRematerializableSlice(
        thenRes, rootLayout, newSlice, newLayout, newExistingRemats, isIfOp);
    LogicalResult elseResult = getRematerializableSlice(
        elseRes, rootLayout, newSlice, newLayout, newExistingRemats, isIfOp);

    // If propagation across both edges of this conditional succeeded, then we
    // don't need to hoist across it. Merge into the current slice.
    if (succeeded(thenResult) && succeeded(elseResult)) {
      slice = std::move(newSlice);
      layout = std::move(newLayout);
      existingRemats = std::move(newExistingRemats);
      continue;
    }

    // If propagation across both edges failed, then this conditional
    // terminates backwards rematerialization.
    if (failed(thenResult) && failed(elseResult)) {
      terminals.push_back(cast<OpResult>(v));
      continue;
    }

    // Only hoist into conditionals inside loops. The assumption is that an if
    // inside a loop executes fewer than the total number of loop iterations,
    // making this hoist profitable.
    if (!isa<scf::ForOp>(ifOp->getParentOp())) {
      terminals.push_back(cast<OpResult>(v));
      continue;
    }

    slice = std::move(newSlice);
    layout = std::move(newLayout);
    existingRemats = std::move(newExistingRemats);
    // The layout conversion can be rematerialized along one edge but not the
    // other. We can hoist the conversion into the other branch. Push this
    // into the subslice list for analysis.
    if (succeeded(thenResult)) {
      hoistAbove.emplace_back(v, &elseRes);
    } else {
      hoistAbove.emplace_back(v, &thenRes);
    }
  }

  // Exit early if there is nothing to do.
  if (hoistAbove.empty())
    return false;

  // Rematerialize failed hoists right before the condtional, and hoist those
  // that succeeded into the branch and then rewrite the slice.
  IRMapping mapping;
  auto hoistRemat = [&](OpBuilder &b, Value v, Attribute encoding) {
    auto tensorType = cast<RankedTensorType>(v.getType());
    auto newType = tensorType.cloneWithEncoding(encoding);
    Value newCvt = ConvertLayoutOp::create(b, convertOp.getLoc(), newType, v);

    mapping.map(v, newCvt);
    slice.remove(v);
  };
  for (Value v : terminals) {
    OpBuilder b(v.getContext());
    b.setInsertionPointAfter(v.getDefiningOp());
    hoistRemat(b, v, layout.at(v));
  }
  for (auto [result, edge] : hoistAbove) {
    OpBuilder b(edge->getOwner());
    hoistRemat(b, edge->get(), layout.at(result));
  }
  rewriteSlice(slice, layout, existingRemats, convertOp, mapping);
  return true;
}

bool backwardRematerialization(ModuleOp module, bool disableRematSplitting,
                               bool preserveSharedReductionRematerialization) {
  bool changed = false;
  module.walk([&](FuncOp funcOp) {
    LayoutRematerialization layoutRemat(funcOp,
                                        preserveSharedReductionRematerialization);
    changed |= layoutRemat.backwardRematerialization(disableRematSplitting);
  });
  return changed;
}

void hoistConvert(ModuleOp module, bool disableRematSplitting) {
  SmallVector<ConvertLayoutOp> convertOps;
  module.walk([&](FuncOp funcOp) {
    LayoutRematerialization(funcOp).hoistConvertOnTopOfExtOrBroadcast(
        disableRematSplitting);
    if (disableRematSplitting)
      return;

    LayoutRematerialization(funcOp).hoistConvertIntoConditionals();
    LayoutRematerialization(funcOp).hoistConvertDotOperand();
  });
}

LogicalResult cleanupLayoutConversions(ModuleOp module) {
  MLIRContext *context = module.getContext();
  RewritePatternSet patterns(context);
  ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
  if (failed(applyPatternsGreedily(module, std::move(patterns))))
    return failure();

  LLVM_DEBUG({
    DBGS() << "Module after canonicalizing:\n";
    module.dump();
  });
  return success();
}

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

LogicalResult optimizeDistributedLayouts(ModuleOp module,
                                         bool disableRematSplitting,
                                         LayoutAssignmentStrategy strategy) {
  MLIRContext *context = module.getContext();

  // Propagate fixed anchor layouts across complete functions and structured
  // control flow before resolving competing assignments.
  WalkResult propagationResult = module.walk([&](FuncOp funcOp) -> WalkResult {
    bool hasProtectedLoop = strategy == LayoutAssignmentStrategy::Global &&
                            hasProtectedLayoutLoop(funcOp);
    bool hasProtectedPackedAssembly =
        strategy == LayoutAssignmentStrategy::Global &&
        funcOp
            .walk([](ElementwiseInlineAsmOp inlineAsm) {
              if (inlineAsm.getPackedElement() > 1)
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted();
    bool hasProtectedStore =
        hasProtectedLoop &&
        funcOp
            .walk([](Operation *op) {
              if (isa<StoreOp, DescriptorStoreLikeOpInterface,
                      triton::nvidia_gpu::TMAStoreLikeOpInterface>(op))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted();
    bool hasProtectedReductionNetwork =
        strategy == LayoutAssignmentStrategy::Global &&
        funcOp
            .walk([](ReduceOp reduce) {
              if (isProtectedLayoutReduction(reduce.getOperation()))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted();
    bool hasDescriptorLayoutBoundary =
        strategy == LayoutAssignmentStrategy::Global &&
        funcOp
            .walk([](Operation *op) {
              if (isa<DescriptorLoadLikeOpInterface>(op))
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted();
    bool hasPackedMemoryAssembly =
        strategy == LayoutAssignmentStrategy::Global &&
        hasPackedMemoryAssemblyProtocol(funcOp);
    bool hasConvertiblePermutingReshape =
        strategy == LayoutAssignmentStrategy::Global &&
        funcOp
            .walk([](ReshapeOp reshape) {
              if (reshape.getAllowReorder() && !reshape.getEfficientLayout() &&
                  reshape.getSrc().getDefiningOp<ConvertLayoutOp>())
                return WalkResult::interrupt();
              return WalkResult::advance();
            })
            .wasInterrupted();

    if (strategy == LayoutAssignmentStrategy::Global &&
        (hasProtectedLoop || hasProtectedPackedAssembly ||
         hasProtectedReductionNetwork || hasConvertiblePermutingReshape ||
         hasDescriptorLayoutBoundary || hasPackedMemoryAssembly ||
         hasPairwiseFp8ReductionMemoryProtocol(funcOp))) {
      // Establish the incumbent layout for protected hardware, packed
      // register and memory assembly, reduction protocols, descriptor loads,
      // and permuting views before optimizing the remaining components.
      LayoutPropagation legacyPropagation(funcOp,
                                          LayoutAssignmentStrategy::Legacy);
      legacyPropagation.initAnchorLayout();
      legacyPropagation.propagateLayout();
      legacyPropagation.resolveConflicts();
      legacyPropagation.rewrite();

      if (hasProtectedStore || hasPackedMemoryAssembly ||
          hasConvertiblePermutingReshape) {
        // Expose the incumbent's rematerializable memory addresses and
        // explicitly permuting reshapes before fixing their boundaries.
        RewritePatternSet patterns(context);
        ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
        if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
          return WalkResult::interrupt();
      }

      if (hasProtectedStore || hasPackedMemoryAssembly) {
        bool changed;
        do {
          // Establish the complete incumbent before fixing hardware-store and
          // packed-memory addresses. The standard rematerialization traversal
          // also keeps its conversion-reuse and IR-mapping invariants.
          {
            LayoutRematerialization rematerialization(funcOp);
            changed = rematerialization.backwardRematerialization(
                disableRematSplitting);
          }
          if (changed) {
            RewritePatternSet patterns(context);
            ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
            if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
              return WalkResult::interrupt();
          }
        } while (changed);
      }
    }

    LayoutPropagation propagation(funcOp, strategy);
    propagation.initAnchorLayout();
    propagation.propagateLayout();
    propagation.resolveConflicts();
    propagation.rewrite();

    return WalkResult::advance();
  });
  if (propagationResult.wasInterrupted())
    return failure();

  LLVM_DEBUG({
    DBGS() << "Module after propagating layouts forward:\n";
    module.dump();
  });

  if (failed(cleanupLayoutConversions(module)))
    return failure();

  bool changed;
  do {
    changed = backwardRematerialization(
        module, disableRematSplitting,
        strategy == LayoutAssignmentStrategy::Global);
    LLVM_DEBUG({
      DBGS() << "Module after backward remat:\n";
      module.dump();
    });

    if (failed(cleanupLayoutConversions(module)))
      return failure();
  } while (changed);

  hoistConvert(module, disableRematSplitting);
  LLVM_DEBUG({
    DBGS() << "Module after hoisting converts:\n";
    module.dump();
  });

  runDeadIterArgElimination(module);

  RewritePatternSet convertCleanup(context);
  ConvertLayoutOp::getCanonicalizationPatterns(convertCleanup, context);
  if (failed(applyPatternsGreedily(module, std::move(convertCleanup))))
    return failure();

  // Structured-control-flow canonicalization is best effort.
  RewritePatternSet scfCleanup(context);
  scf::ForOp::getCanonicalizationPatterns(scfCleanup, context);
  scf::IfOp::getCanonicalizationPatterns(scfCleanup, context);
  if (failed(applyPatternsGreedily(module, std::move(scfCleanup))))
    LLVM_DEBUG(DBGS() << "scf cleanup did not converge\n");

  if (strategy == LayoutAssignmentStrategy::Global) {
    SmallVector<Block *> blocks;
    module.walk([&](Block *block) { blocks.push_back(block); });
    for (Block *block : blocks) {
      SmallVector<std::pair<std::string, arith::ConstantOp>, 16> constants;
      for (Operation &op : *block) {
        auto constant = dyn_cast<arith::ConstantOp>(op);
        if (!constant)
          continue;
        std::string key;
        llvm::raw_string_ostream stream(key);
        constant.getType().print(stream);
        stream << '\n';
        constant.getValue().print(stream);
        constants.emplace_back(stream.str(), constant);
      }
      llvm::stable_sort(constants, [](const auto &lhs, const auto &rhs) {
        return lhs.first < rhs.first;
      });
      for (auto &entry : llvm::reverse(constants)) {
        Operation *constant = entry.second.getOperation();
        if (constant != &block->front())
          constant->moveBefore(&block->front());
      }
    }
  }

  LLVM_DEBUG({
    DBGS() << "Module after final cleanups:\n";
    module.dump();
  });

  return success();
}

class TritonGPUOptimizeLayoutsPass
    : public impl::TritonGPUOptimizeLayoutsBase<TritonGPUOptimizeLayoutsPass> {
public:
  using impl::TritonGPUOptimizeLayoutsBase<
      TritonGPUOptimizeLayoutsPass>::TritonGPUOptimizeLayoutsBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (unsigned iteration = 0; iteration < 2; ++iteration) {
      unsigned originalConversions = 0;
      module.walk([&](ConvertLayoutOp) { ++originalConversions; });

      if (failed(optimizeDistributedLayouts(module, disableRematSplitting,
                                            LayoutAssignmentStrategy::Global)))
        return signalPassFailure();

      shareDominatingConversions(module);
      optimizeStoreRootedConversions(module);
      optimizeScalarRootedConversions(module);

      RewritePatternSet patterns(&getContext());
      ConvertLayoutOp::getCanonicalizationPatterns(patterns, &getContext());
      if (failed(applyPatternsGreedily(module, std::move(patterns))))
        return signalPassFailure();

      unsigned remainingConversions = 0;
      module.walk([&](ConvertLayoutOp) { ++remainingConversions; });
      if (remainingConversions == 0 ||
          remainingConversions >= originalConversions)
        break;
    }
  }
};

class TritonGPURemoveLayoutConversionsPass
    : public impl::TritonGPURemoveLayoutConversionsBase<
          TritonGPURemoveLayoutConversionsPass> {
public:
  using impl::TritonGPURemoveLayoutConversionsBase<
      TritonGPURemoveLayoutConversionsPass>::
      TritonGPURemoveLayoutConversionsBase;

  void runOnOperation() override {
    if (failed(optimizeDistributedLayouts(getOperation(), disableRematSplitting,
                                          LayoutAssignmentStrategy::Legacy)))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu
