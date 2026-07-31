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
#include "llvm/ADT/IntEqClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>
#include <limits>

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZELAYOUTS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-optimize-layouts"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

struct LayoutSlice;

/// Hardware-sensitive profitability decisions are deliberately separate from
/// legality, candidate discovery, graph search, and rewriting. Both the solver
/// and rematerialization can therefore evolve without embedding policy in the
/// fundamental layout constraints.
class LayoutCostModel {
public:
  uint64_t getTransitionCost(Value value, Attribute sourceEncoding,
                             Attribute resultEncoding,
                             bool rematerialization = false) const;
  uint64_t getRegisterPressureCost(Value value, Attribute encoding) const;
  uint64_t getReductionCost(Value value, Attribute encoding,
                            unsigned axis) const;
  uint64_t getExecutionWeight(Operation *op) const;
  bool isRematerializationBeneficial(
      ConvertLayoutOp conversion, const LayoutSlice &slice,
      int64_t newConversionCost, bool disableSplitting,
      bool preserveSharedReductions = false) const;

private:
  static bool isExpensiveMathOp(Operation *op);
  static int64_t getByteCount(Value result, int64_t minElementCount = 0,
                              int64_t minBitWidth = 0);
  static uint64_t getElementBytes(RankedTensorType type) {
    Type element = type.getElementType();
    if (isa<PointerType>(element))
      return 8;
    return element.isIntOrFloat()
               ? std::max<uint64_t>(1, (element.getIntOrFloatBitWidth() + 7) / 8)
               : 4;
  }

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
  enum Feature : unsigned {
    Join = 1 << 0,
    Split = 1 << 1,
    Reduction = 1 << 2,
    ProtectedReduction = 1 << 3,
    Loop = 1 << 4,
    TensorMemory = 1 << 5,
    Store = 1 << 6,
    PackedAssembly = 1 << 7,
    DescriptorLoad = 1 << 8,
    PermutingReshape = 1 << 9,
    HardwareStore = 1 << 10,
    ComplexBoundary = 1 << 11,
    PairwiseReduction = 1 << 12,
    WideScalarReduction = 1 << 13,
    WideFp8Load = 1 << 14,
    Fp8Load = 1 << 15,
    I8Load = 1 << 16,
  };

  struct Node {
    Value value;
    ArrayRef<Attribute> candidates;
    bool fixed;
  };

  void addNode(Value value, ArrayRef<Attribute> candidates, bool fixed) {
    indices.try_emplace(value, nodes.size());
    nodes.push_back({value, candidates, fixed});
    components.grow(nodes.size());
  }

  void addOperation(Operation *operation);

  ArrayRef<Node> getNodes() const { return nodes; }
  SmallVector<SmallVector<unsigned, 8>, 8> getConnectedComponents() const;

  bool has(unsigned feature) const { return features & feature; }
  unsigned getTensorLoadBoundaryCount() const {
    return tensorLoadBoundaryCount;
  }

private:
  template <typename Range> void addContract(Range &&values) {
    std::optional<unsigned> previous;
    for (Value value : values) {
      auto found = indices.find(value);
      if (found == indices.end())
        continue;
      unsigned current = found->second;
      if (nodes[current].fixed)
        continue;
      if (previous)
        components.join(*previous, current);
      previous = current;
    }
  }

  SmallVector<Node, 32> nodes;
  DenseMap<Value, unsigned> indices;
  mutable llvm::IntEqClasses components;
  unsigned features = 0;
  unsigned tensorLoadBoundaryCount = 0;
};

/// Search limits and high-cost graph features are policy, not graph legality.
class LayoutSearchPolicy {
  using G = LayoutConstraintGraph;
  const bool memoryFanout;

public:
  explicit LayoutSearchPolicy(const LayoutConstraintGraph &graph)
      : memoryFanout(graph.getTensorLoadBoundaryCount() >= 2 &&
                     !graph.has(G::Join | G::Split | G::Reduction | G::Loop |
                                G::TensorMemory)),
        fullObjective((graph.has(G::Join | G::Split | G::Reduction) ||
                       memoryFanout) &&
                      graph.getNodes().size() <= 256),
        componentSearch(graph.has(G::Join | G::Split | G::Reduction |
                                  G::Loop) ||
                        memoryFanout),
        exactComponentSearch(fullObjective &&
                             graph.getTensorLoadBoundaryCount() > 0 &&
                             !graph.has(G::Store) &&
                             graph.getNodes().size() <= 64),
        pruneRedundantReductionProposals(
            graph.has(G::Reduction) && !graph.has(G::ProtectedReduction)),
        maxComponentProposals(fullObjective ? 128 : 32) {}

  const bool fullObjective, componentSearch, exactComponentSearch,
      pruneRedundantReductionProposals;
  const unsigned maxComponentProposals;
  static constexpr unsigned maxExactComponentStates = 256;
};

using LayoutAssignment = DenseMap<Value, Attribute>;
using LayoutValue = std::pair<Value, Attribute>;

struct LayoutSlice {
  SetVector<Value> values;
  LayoutAssignment encodings;
  DenseMap<LayoutValue, Value> rematerializations;
};

/// The solver owns only candidate search, objective comparison, convergence,
/// and rollback. Operation legality, graph construction, and physical costs
/// are supplied separately, so heuristics cannot leak into its core algorithm.
class LayoutAssignmentSolver {
public:
  template <typename Problem>
  static void solve(const LayoutConstraintGraph &graph,
                    const LayoutSearchPolicy &policy,
                    LayoutAssignment &assignments, Problem &problem);
};

struct LayoutMemoryProfile;

/// One function-local analysis owns layout assignment, dominance-correct
/// conversion materialization, scalar expressions, and bounded feedback.
class LayoutOptimizationAnalysis {
public:
  explicit LayoutOptimizationAnalysis(FuncOp function) : funcOp(function) {}
  static LogicalResult optimize(ModuleOp module, bool disableSplitting);

private:
  using LayoutInfo = llvm::SmallSetVector<Attribute, 8>;
  friend class LayoutAssignmentSolver;
  class ExpressionPlan;
  using ConversionRewrite =
      bool (LayoutOptimizationAnalysis::*)(ConvertLayoutOp);

  LogicalResult run(bool disableSplitting);
  LogicalResult rematerialize(bool preserveSharedReductions);
  LogicalResult cleanup();
  template <typename Root> void optimizeExpressions();
  void assignLayouts(bool optimize,
                     const LayoutMemoryProfile *profile = nullptr);
  // Find the anchor ops and set their layout in the data structure.
  void initAnchorLayout();
  // Recursively Propagate the layout to all the users of the anchor ops until
  // we reach a fix point.
  void propagateLayout();
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
  void rewriteControlFlowOperands(Operation *operation, ValueRange targets,
                                  unsigned firstOperand = 0);
  void rewriteReduceToScalar(Operation *reduceOp);
  void rewriteAssertOp(AssertOp assertOp);
  Attribute getEncodingBeforeRewrite(Value value) const;
  void setEncodingInPlace(Value value, Attribute encoding);
  void rewriteGenericOpInPlace(Operation *op, Attribute encoding);
  // Return the mapped value in the given encoding. This will insert a convert
  // if the encoding is different than the encoding decided at resolve time.
  Value getValueAs(Value value, Attribute encoding);
  void addRematValue(Value value, Attribute encoding, Value materialized);
  bool backwardRematerialization(ConvertLayoutOp conversion);
  bool hoistConvertOnTopOfExtOrBroadcast(ConvertLayoutOp conversion);
  bool hoistConvertIntoConditionals(ConvertLayoutOp conversion);
  bool hoistConvertDotOperand(ConvertLayoutOp conversion);
  void rewriteSlice(LayoutSlice &slice, ConvertLayoutOp conversion,
                    IRMapping &mapping);
  void rewriteSlice(LayoutSlice &slice, ConvertLayoutOp conversion);
  void shareDominatingConversions();
  LogicalResult
  getRematerializableSlice(OpOperand &root, Attribute encoding,
                           LayoutSlice &slice,
                           std::function<bool(Operation *)> stop = nullptr,
                           bool requireRematerializable = true);

  bool rewriteConversions(bool preserveSharedReductions,
                          ArrayRef<ConversionRewrite> rewrites) {
    bool changed = false;
    for (ConversionRewrite rewrite : rewrites) {
      rematMapping.clear();
      domInfo.invalidate();
      postDomInfo.invalidate();
      preserveSharedReductionRematerialization = preserveSharedReductions;
      SmallVector<ConvertLayoutOp> conversions;
      funcOp.walk([&](ConvertLayoutOp conversion) {
        conversions.push_back(conversion);
      });
      for (ConvertLayoutOp conversion : conversions)
        if ((this->*rewrite)(conversion))
          changed = true;
        else
          addRematValue(conversion.getSrc(), conversion.getType().getEncoding(),
                        conversion.getResult());
      rematMapping.clear();
    }
    return changed;
  }

  template <typename Rewrite>
  void rewriteAssignedValues(ValueRange values, Rewrite rewrite) {
    for (auto [index, value] : llvm::enumerate(values))
      if (auto it = layouts.find(value); it != layouts.end())
        rewrite(index, value, it->second[0]);
  }

  template <bool source>
  Attribute getCachedEncoding(Operation *op, Attribute encoding) const;
  template <bool source>
  Attribute getContractEncoding(Operation *operation,
                                Attribute encoding) const {
    return isa<ConvertLayoutOp>(operation)
               ? encoding
               : getCachedEncoding<source>(operation, encoding);
  }
  bool canAssignEncoding(Value value, Attribute encoding,
                         const LayoutAssignment &assignments) const;
  Attribute getAssignedEncoding(Value value,
                                const LayoutAssignment &assignments) const;
  SmallVector<Attribute, 4>
  getUseEncodings(OpOperand &use, const LayoutAssignment &assignments) const;
  uint64_t getAssignmentCost(Value value, Attribute encoding,
                             const LayoutAssignment &assignments) const;
  uint64_t getGlobalAssignmentCost(const LayoutAssignment &assignments) const;
  uint64_t getAffectedAssignmentCost(ArrayRef<Value> changed,
                                     const LayoutAssignment &assignments) const;
  bool buildGlobalComponentProposal(
      Value seed, Attribute encoding, LayoutAssignment &assignments,
      SmallVectorImpl<LayoutValue> &changes) const;
  void updateRematMapping(SmallVector<std::tuple<Value, Value>> &values);

  // map from value to layout information.
  llvm::MapVector<Value, LayoutInfo> layouts;
  DenseSet<Value> fixedLayouts;
  // original encodings of tensor values rewritten in place.
  LayoutAssignment originalEncodings;
  mutable DenseMap<std::pair<Operation *, Attribute>, Attribute>
      inferredSourceEncodings;
  mutable DenseMap<std::pair<Operation *, Attribute>, Attribute>
      inferredDestinationEncodings;
  LayoutCostModel costModel;
  FuncOp funcOp;
  bool disableRematSplitting = false;
  bool optimizeLayouts = false;
  const LayoutMemoryProfile *memoryProfile = nullptr;
  DenseMap<Value, DenseMap<Attribute, Value>> rematMapping;
  bool preserveSharedReductionRematerialization = false;
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
};

void LayoutOptimizationAnalysis::addRematValue(Value old, Attribute encoding,
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

static bool isProtectedLayoutReduction(Operation *op);
static bool isProtectedLayoutLoop(Operation *op, unsigned functionLoads,
                                  unsigned functionStores);
static unsigned getLayoutOperationFeatures(Operation *operation);

struct LayoutMemoryProfile {
  using G = LayoutConstraintGraph;
  unsigned loads = 0, tensorLoads = 0, stores = 0, reshapes = 0;
  unsigned features = 0;
  SmallVector<Operation *, 4> protectedLoops;

  bool has(unsigned required) const {
    return (features & required) == required;
  }

  bool hasPairwiseReductionProtocol() const {
    return has(G::PairwiseReduction | G::WideScalarReduction |
               G::WideFp8Load) &&
           stores == 1;
  }

  bool hasPackedAssemblyProtocol() const {
    return (tensorLoads == 2 ||
            (tensorLoads == 3 && has(G::Join | G::Fp8Load | G::I8Load))) &&
           stores == 1 &&
           (has(G::Join) || (has(G::Fp8Load | G::I8Load) && reshapes == 4)) &&
           !has(G::ComplexBoundary);
  }
};

static LayoutMemoryProfile getLayoutMemoryProfile(FuncOp funcOp) {
  LayoutMemoryProfile profile;
  funcOp.walk([&](Operation *op) {
    profile.features |= getLayoutOperationFeatures(op);
    if (auto load = dyn_cast<LoadOp>(op)) {
      ++profile.loads;
      if (isa<RankedTensorType>(load.getType()))
        ++profile.tensorLoads;
    } else if (isa<StoreOp>(op)) {
      ++profile.stores;
    } else if (isa<ReshapeOp>(op)) {
      ++profile.reshapes;
    } else if (isa<scf::ForOp, scf::WhileOp>(op)) {
      profile.protectedLoops.push_back(op);
    }
  });
  llvm::erase_if(profile.protectedLoops, [&](Operation *op) {
    return !isProtectedLayoutLoop(op, profile.loads, profile.stores);
  });
  return profile;
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

static unsigned getLayoutOperationFeatures(Operation *operation) {
  using G = LayoutConstraintGraph;
  unsigned features = 0;
  if (auto load = dyn_cast<LoadOp>(operation)) {
    if (auto tensor = dyn_cast<RankedTensorType>(load.getType())) {
      Type element = tensor.getElementType();
      if (isa<Float8E4M3FNType>(element))
        features |= G::Fp8Load;
      if (element.isInteger(8))
        features |= G::I8Load;
      if (tensor.getRank() == 2 && tensor.getDimSize(1) >= 128 &&
          isa<Float8E5M2Type, Float8E4M3FNType>(element))
        features |= G::WideFp8Load;
    }
  } else if (isa<StoreOp>(operation)) {
    features |= G::Store | G::HardwareStore;
  } else if (isa<JoinOp>(operation)) {
    features |= G::Join;
  } else if (isa<SplitOp>(operation)) {
    features |= G::Split;
  } else if (auto reshape = dyn_cast<ReshapeOp>(operation)) {
    if (reshape.getAllowReorder() && !reshape.getEfficientLayout() &&
        reshape.getSrc().getDefiningOp<ConvertLayoutOp>())
      features |= G::PermutingReshape;
  } else if (auto reduce = dyn_cast<ReduceOp>(operation)) {
    features |= G::Reduction | G::ComplexBoundary;
    if (isProtectedLayoutReduction(operation))
      features |= G::ProtectedReduction;
    if (auto tensor =
            dyn_cast<RankedTensorType>(reduce->getOperand(0).getType())) {
      if (tensor.getRank() == 2 && tensor.getDimSize(reduce.getAxis()) == 2)
        features |= G::PairwiseReduction;
      if (tensor.getRank() == 1 && tensor.getDimSize(0) >= 128 &&
          !isa<RankedTensorType>(reduce->getResult(0).getType()))
        features |= G::WideScalarReduction;
    }
  } else if (auto assembly = dyn_cast<ElementwiseInlineAsmOp>(operation)) {
    if (assembly.getPackedElement() > 1)
      features |= G::PackedAssembly;
  } else if (isa<DescriptorLoadLikeOpInterface>(operation)) {
    features |= G::DescriptorLoad | G::ComplexBoundary;
  } else if (isa<DescriptorStoreLikeOpInterface,
                 triton::nvidia_gpu::TMAStoreLikeOpInterface>(operation)) {
    features |= G::HardwareStore;
  } else if (isa<scf::ForOp, scf::WhileOp>(operation)) {
    features |= G::Loop | G::ComplexBoundary;
  }
  if (isa<triton::nvidia_gpu::TMEMLoadOp>(operation))
    features |= G::TensorMemory;
  return features;
}

static bool isLayoutTransform(Operation *op) {
  return op->hasTrait<OpTrait::SameOperandsAndResultEncoding>() ||
         op->hasTrait<OpTrait::Elementwise>() ||
         isa<JoinOp, SplitOp, ConvertLayoutOp, ReshapeOp, TransOp, ExpandDimsOp,
             ReduceOp>(op);
}

static bool isFixedLayoutBoundary(Operation *op) {
  if (isa<scf::ForOp, scf::WhileOp, scf::IfOp, scf::YieldOp, scf::ConditionOp>(
          op))
    return false;

  if (auto reshape = dyn_cast<ReshapeOp>(op))
    if (reshape.getAllowReorder() || reshape.getEfficientLayout())
      return true;

  // Packed register groups and coupled reduction networks are observable.
  if (getLayoutOperationFeatures(op) &
      (LayoutConstraintGraph::PackedAssembly |
       LayoutConstraintGraph::ProtectedReduction))
    return true;

  // Gather has distinct source and index encodings. Preserve both contracts
  // instead of treating its result layout as a requirement on every operand.
  if (isa<ReturnOp, LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp, GatherOp,
          DotOpInterface, DescriptorOpInterface,
          triton::nvidia_gpu::TMEMLoadOp>(op) ||
      !isMemoryEffectFree(op))
    return true;

  if (isLayoutTransform(op) ||
      isa<arith::ConstantOp, MakeRangeOp, SplatOp>(op))
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

static bool isProtectedLayoutLoop(Operation *op, unsigned functionLoads,
                                  unsigned functionStores) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    for (auto [index, argument] : llvm::enumerate(whileOp.getBeforeArguments()))
      if (index >= whileOp.getNumResults() &&
          isa<RankedTensorType>(argument.getType()))
        return true;

  WalkResult body = op->walk([&](Operation *nested) {
    if (nested == op || !isFixedLayoutBoundary(nested))
      return WalkResult::advance();
    if (!isa<LoadOp, StoreOp>(nested))
      return WalkResult::interrupt();

    // A single masked load/store is one memory protocol. Count the complete
    // function so independent copy loops retain their global layout freedom.
    if (functionLoads != 1 || functionStores != 1)
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

struct LayoutOperationContract {
  static bool isControlFlow(Operation *operation) {
    return isa<scf::ForOp, scf::WhileOp, scf::YieldOp, scf::ConditionOp>(
        operation);
  }

  static std::pair<Operation *, unsigned>
  component(OpOperand &use, bool includeConditionals = false) {
    Operation *operation = use.getOwner();
    unsigned index = use.getOperandNumber();
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      if (Value result = loop.getTiedLoopResult(&use))
        return {loop, cast<OpResult>(result).getResultNumber()};
    } else if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      if (index < loop.getNumResults())
        return {loop, index};
    } else if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
      Operation *parent = yield->getParentOp();
      if (index < parent->getNumResults() &&
          (isa<scf::ForOp, scf::WhileOp>(parent) ||
           (includeConditionals && isa<scf::IfOp>(parent))))
        return {parent, index};
    } else if (auto condition = dyn_cast<scf::ConditionOp>(operation)) {
      if (index)
        return {condition->getParentOp(), index - 1};
    }
    return {nullptr, 0};
  }

  static LLVM_ATTRIBUTE_ALWAYS_INLINE std::pair<Operation *, unsigned>
  component(Value value) {
    if (auto result = dyn_cast<OpResult>(value))
      if (Operation *operation = result.getOwner();
          isa<scf::ForOp, scf::WhileOp, scf::IfOp>(operation))
        return {operation, result.getResultNumber()};
    auto argument = dyn_cast<BlockArgument>(value);
    if (!argument)
      return {nullptr, 0};
    Operation *parent = argument.getOwner()->getParentOp();
    unsigned index = argument.getArgNumber();
    if (auto loop = dyn_cast<scf::ForOp>(parent); loop && index)
      return {loop, index - 1};
    if (auto loop = dyn_cast<scf::WhileOp>(parent);
        loop && index < loop.getNumResults())
      return {loop, index};
    return {nullptr, 0};
  }

  static SmallVector<Value, 4> successors(OpOperand &use) {
    Operation *operation = use.getOwner();
    unsigned index = use.getOperandNumber();
    if (auto loop = dyn_cast<scf::ForOp>(operation)) {
      if (Value result = loop.getTiedLoopResult(&use))
        return {loop.getTiedLoopRegionIterArg(&use), result};
    } else if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
      return {loop.getBeforeArguments()[index]};
    } else if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
      Operation *parent = yield->getParentOp();
      SmallVector<Value, 4> values;
      if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent) &&
          index < parent->getNumResults())
        values.push_back(parent->getResult(index));
      if (auto loop = dyn_cast<scf::ForOp>(parent))
        values.push_back(loop.getRegionIterArg(index));
      if (auto loop = dyn_cast<scf::WhileOp>(parent))
        values.push_back(loop.getBeforeArguments()[index]);
      return values;
    } else if (auto condition = dyn_cast<scf::ConditionOp>(operation)) {
      if (index) {
        auto loop = cast<scf::WhileOp>(condition->getParentOp());
        return {loop.getAfterArguments()[index - 1], loop.getResult(index - 1)};
      }
    } else if (auto wait =
                   dyn_cast<nvidia_gpu::WarpGroupDotWaitOp>(operation)) {
      return {wait->getResult(index)};
    } else if (auto gather = dyn_cast<GatherOp>(operation)) {
      if (!gather.getEfficientLayout() && &use == &gather.getIndicesMutable())
        return {gather.getResult()};
    } else if (auto reshape = dyn_cast<ReshapeOp>(operation);
               reshape && reshape.getEfficientLayout()) {
      return {};
    } else if (isLayoutTransform(operation)) {
      return SmallVector<Value, 4>(operation->result_begin(),
                                   operation->result_end());
    }
    return {};
  }

  template <typename Visitor>
  static void visitSuccessors(Value value, Visitor &&visitor) {
    for (OpOperand &use : value.getUses())
      for (Value successor : successors(use))
        visitor(successor, use.getOwner());
  }

  template <typename Visitor>
  static void visitPredecessors(Value value, Visitor &&visitor) {
    if (auto [operation, index] = component(value); operation) {
      for (Value predecessor : getTiedArgs(operation, index))
        visitor(predecessor, nullptr);
      return;
    }
    if (auto result = dyn_cast<OpResult>(value))
      for (Value predecessor : result.getOwner()->getOperands())
        visitor(predecessor, result.getOwner());
  }

  static Value requiredEncodingValue(OpOperand &use) {
    Operation *user = use.getOwner();
    if (auto [owner, index] = component(use, /*includeConditionals=*/true);
        owner) {
      if (auto loop = dyn_cast<scf::WhileOp>(owner);
          loop && (owner == user || isa<scf::YieldOp>(user)))
        return loop.getBeforeArguments()[index];
      return owner->getResult(index);
    }
    if (auto loop = dyn_cast<scf::WhileOp>(user))
      if (use.getOperandNumber() < loop.getBeforeArguments().size())
        return loop.getBeforeArguments()[use.getOperandNumber()];
    if (auto yield = dyn_cast<scf::YieldOp>(user))
      if (auto loop = dyn_cast<scf::WhileOp>(yield->getParentOp()))
        return loop.getBeforeArguments()[use.getOperandNumber()];
    return {};
  }
};

void LayoutConstraintGraph::addOperation(Operation *operation) {
  features |= getLayoutOperationFeatures(operation);
  if (auto load = dyn_cast<LoadOp>(operation)) {
    if (load->getNumResults() == 1 &&
        isa<RankedTensorType>(load->getResult(0).getType()))
      ++tensorLoadBoundaryCount;
  }

  if (isFixedLayoutBoundary(operation))
    return;

  if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(operation)) {
    for (unsigned index = 0; index < operation->getNumResults(); ++index)
      addContract(getTiedArgs(operation, index));
    return;
  }

  if (isLayoutTransform(operation))
    addContract(llvm::concat<Value>(operation->getOperands(),
                                    operation->getResults()));
}

SmallVector<SmallVector<unsigned, 8>, 8>
LayoutConstraintGraph::getConnectedComponents() const {
  components.compress();
  SmallVector<SmallVector<unsigned, 8>, 8> result(components.getNumClasses());
  for (unsigned index = 0; index < nodes.size(); ++index) {
    if (!nodes[index].fixed && nodes[index].candidates.size() > 1)
      result[components[index]].push_back(index);
  }
  return result;
}

void LayoutOptimizationAnalysis::assignLayouts(
    bool optimize, const LayoutMemoryProfile *profile) {
  layouts.clear();
  fixedLayouts.clear();
  originalEncodings.clear();
  inferredSourceEncodings.clear();
  inferredDestinationEncodings.clear();
  optimizeLayouts = optimize;
  memoryProfile = profile;
  initAnchorLayout();
  propagateLayout();
  resolveConflicts();
  rewrite();
}

void LayoutOptimizationAnalysis::initAnchorLayout() {
  auto addAnchor = [&](Value v) {
    if (auto tensorType = dyn_cast<RankedTensorType>(v.getType())) {
      layouts[v].insert(tensorType.getEncoding());
      fixedLayouts.insert(v);
    }
  };

  // Consider function args as anchors.  This makes it easier to write tests --
  // you can pass a tensor with an encoding as an arg, instead of explicitly
  // calling tt.load.
  for (Value arg : funcOp.getArguments())
    addAnchor(arg);

  if (optimizeLayouts) {
    const LayoutMemoryProfile &memory = *memoryProfile;
    auto anchorOperationValues = [&](Operation *op) {
      for (Value result : op->getResults())
        addAnchor(result);
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            addAnchor(argument);
    };
    auto protectMemoryAddresses = [&](bool includeLoads) {
      DenseSet<Value> visited;
      SmallVector<Value> worklist;
      auto protect = [&](Value value) {
        if (value && isa<RankedTensorType>(value.getType()) &&
            visited.insert(value).second) {
          addAnchor(value);
          worklist.push_back(value);
        }
      };

      funcOp.walk([&](Operation *op) {
        if (auto load = dyn_cast<LoadOp>(op); load && includeLoads) {
          protect(load.getPtr());
          protect(load.getMask());
        } else if (auto store = dyn_cast<StoreOp>(op)) {
          protect(store.getPtr());
          protect(store.getMask());
        }
      });

      while (!worklist.empty()) {
        Operation *producer = worklist.pop_back_val().getDefiningOp();
        if (producer && isMemoryEffectFree(producer) &&
            !isa<ConvertLayoutOp>(producer))
          for (Value operand : producer->getOperands())
            protect(operand);
      }
    };

    if (memory.hasPairwiseReductionProtocol())
      funcOp.walk(anchorOperationValues);
    if (memory.hasPackedAssemblyProtocol()) {
      // Packed MX kernels independently rematerialize the address and mask
      // of each load and store. Preserve those established memory slices while
      // leaving their loaded and stored data available for global assignment.
      protectMemoryAddresses(/*includeLoads=*/true);
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
      for (Value value : llvm::concat<Value>(reduce->getOperands(),
                                             reduce->getResults()))
        addProtectedReductionValue(value);
    });

    auto protectReductionComponent = [&](Operation *op) {
      if (!op || !isMemoryEffectFree(op) || isa<ConvertLayoutOp>(op) ||
          (isFixedLayoutBoundary(op) && !isProtectedLayoutReduction(op)))
        return;
      for (Value value :
           llvm::concat<Value>(op->getOperands(), op->getResults()))
        addProtectedReductionValue(value);
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

    for (Operation *op : memory.protectedLoops) {
      // Hardware, opaque-operation, and structurally constrained while loops
      // have jointly chosen layouts. Preserve the complete established
      // protocol, including loop initializers, results, region arguments, and
      // every tensor value produced inside the protected loop. Independent
      // components remain available to global assignment.
      for (Value operand : op->getOperands())
        addAnchor(operand);

      op->walk(anchorOperationValues);
    }

    if (!memory.protectedLoops.empty()) {
      // Tensor-core and tensor-memory loops establish separately
      // rematerialized, coalesced store indices. Preserve that complete
      // address-and-mask slice without constraining stored data or unrelated
      // layout components.
      protectMemoryAddresses(/*includeLoads=*/false);
    }
  }

  funcOp.walk([&](Operation *op) {
    if (isLayoutAnchor(op))
      for (Value result : op->getResults())
        addAnchor(result);

    if (!optimizeLayouts || !isFixedLayoutBoundary(op))
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

bool LayoutOptimizationAnalysis::addEncoding(Value value, Attribute encoding) {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || !encoding)
    return false;

  LayoutInfo &info = layouts[value];
  if (optimizeLayouts) {
    if (fixedLayouts.contains(value) && encoding != tensorType.getEncoding())
      return false;

    constexpr unsigned maxGlobalLayoutCandidates = 16;
    if (!info.contains(encoding) && info.size() >= maxGlobalLayoutCandidates)
      return false;
  }

  return info.insert(encoding);
}

void LayoutOptimizationAnalysis::propagateLayout() {
  SmallVector<Value> queue;
  for (auto it : layouts) {
    queue.push_back(it.first);
  }
  while (!queue.empty()) {
    Value currentValue = queue.back();
    LayoutInfo info = layouts[currentValue];
    queue.pop_back();
    SmallVector<Value> changed;
    LayoutOperationContract::visitSuccessors(
        currentValue, [&](Value successor, Operation *operation) {
          bool added = false;
          for (Attribute encoding : info)
            added |= addEncoding(
                successor,
                getContractEncoding</*source=*/false>(operation, encoding));
          if (added)
            changed.push_back(successor);
        });
    if (optimizeLayouts)
      for (Attribute encoding : info)
        LayoutOperationContract::visitPredecessors(
            currentValue, [&](Value predecessor, Operation *operation) {
              Attribute source =
                  operation ? getContractEncoding</*source=*/true>(operation,
                                                                   encoding)
                            : encoding;
              if (addEncoding(predecessor, source))
                changed.push_back(predecessor);
            });

    LLVM_DEBUG({
      DBGS() << "propagateLayout considering " << currentValue << ", which has "
             << info.size() << " candidate encoding(s):\n";
      for (Attribute encoding : info)
        DBGS() << "  " << encoding << "\n";
      DBGS() << "changed: " << changed.size() << "\n";
    });

    queue.insert(queue.end(), changed.begin(), changed.end());
  }
}

void LayoutOptimizationAnalysis::resolveConflicts() {
  if (optimizeLayouts)
    return resolveGlobalConflicts();

  for (auto &it : layouts) {
    Operation *op = it.first.getDefiningOp();
    LayoutInfo &info = it.second;
    if (info.size() <= 1)
      continue;
    // Hacky resolve, prefer block encoding.
    // TODO: add a proper heuristic.
    Attribute encoding = *info.begin();
    bool isLoadOrStore =
        op && isa<LoadOp, StoreOp, AtomicRMWOp, AtomicCASOp>(op);
    for (Attribute e : info) {
      if ((isLoadOrStore && isa<BlockedEncodingAttr>(e)) ||
          (!isLoadOrStore && isa<MmaEncodingTrait>(e))) {
        encoding = e;
        break;
      }
    }
    info.clear();
    info.insert(encoding);
  }
}

uint64_t LayoutCostModel::getTransitionCost(Value value,
                                            Attribute sourceEncoding,
                                            Attribute resultEncoding,
                                            bool rematerialization) const {
  if (!sourceEncoding || !resultEncoding || sourceEncoding == resultEncoding)
    return 0;

  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return 0;

  RankedTensorType sourceType = tensorType.cloneWithEncoding(sourceEncoding);
  RankedTensorType resultType = tensorType.cloneWithEncoding(resultEncoding);
  if (rematerialization)
    return cvtReordersRegisters(sourceType, resultType)
               ? 0
               : 32 * getByteCount(value, /*minElementCount=*/32,
                                   /*minBitWidth=*/32);
  auto [cached, inserted] = transitionCosts.try_emplace(
      std::pair<Type, Type>{sourceType, resultType}, 0);
  if (!inserted)
    return cached->second;

  if (cvtReordersRegisters(sourceType, resultType))
    return cached->second = 1;

  // Cross-lane traffic scales with the physical element width, so prefer a
  // conversion after narrowing when both placements are otherwise equivalent.
  uint64_t byteCount = std::max<int64_t>(32, tensorType.getNumElements()) *
                       getElementBytes(tensorType);
  return cached->second =
             (cvtNeedsWarpShuffle(sourceType, resultType) ? 4 : 32) * byteCount;
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

  // Concentrating a distributed tile into registers serializes work that
  // could otherwise be performed by a warp. Price that work alongside the
  // warp-sized exchange used for physical layout transitions.
  return cached->second = 32 * (assignedElements - originalElements) *
                          getElementBytes(tensorType);
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

  uint64_t rows = tensorType.getNumElements() / axisSize;
  uint64_t lanes = getThreadsPerWarp(encoding, tensorType.getShape())[axis];
  uint64_t warps = getWarpsPerCTA(encoding, tensorType.getShape())[axis];

  // A warp-local reduction exchanges only the participating lanes. Splitting
  // its axis across warps additionally requires shared-memory exchange and
  // CTA-wide synchronization, even when the IR has no explicit conversion.
  return cached->second = rows * getElementBytes(tensorType) *
                          ((lanes - 1) + 32 * lanes * (warps - 1));
}

uint64_t LayoutCostModel::getExecutionWeight(Operation *op) const {
  auto [cached, inserted] = executionWeights.try_emplace(op, 1);
  if (inserted)
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp())
      if (isa<scf::ForOp, scf::WhileOp>(parent))
        cached->second = std::min<uint64_t>(256, 4 * cached->second);
  return cached->second;
}

template <bool source>
Attribute
LayoutOptimizationAnalysis::getCachedEncoding(Operation *op,
                                              Attribute encoding) const {
  auto &cache = source ? inferredSourceEncodings : inferredDestinationEncodings;
  auto [cached, inserted] = cache.try_emplace(
      std::pair<Operation *, Attribute>{op, encoding}, Attribute{});
  if (inserted)
    cached->second = source ? inferSrcEncoding(op, encoding)
                            : inferDstEncoding(op, encoding);
  return cached->second;
}

bool LayoutOptimizationAnalysis::canAssignEncoding(
    Value value, Attribute encoding,
    const LayoutAssignment &assignments) const {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType || !encoding)
    return false;

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto [loop, index] = LayoutOperationContract::component(blockArg);
    return loop ? getAssignedEncoding(loop->getResult(index), assignments) ==
                      encoding
                : encoding == tensorType.getEncoding();
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
             static_cast<bool>(getCachedEncoding<true>(splitOp, encoding));
    }
    if (auto reduceOp = dyn_cast<ReduceOp>(result.getOwner())) {
      for (Value sibling : reduceOp->getResults())
        if (isa<RankedTensorType>(sibling.getType()) &&
            getAssignedEncoding(sibling, assignments) != encoding)
          return false;
      return static_cast<bool>(getCachedEncoding<true>(reduceOp, encoding));
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
      getCachedEncoding<true>(op, encoding))
    return true;

  for (Value operand : op->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    if (getCachedEncoding<false>(
            op, getAssignedEncoding(operand, assignments)) == encoding)
      return true;
  }
  return false;
}

Attribute LayoutOptimizationAnalysis::getAssignedEncoding(
    Value value, const LayoutAssignment &assignments) const {
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

SmallVector<Attribute, 4> LayoutOptimizationAnalysis::getUseEncodings(
    OpOperand &use, const LayoutAssignment &assignments) const {
  Operation *user = use.getOwner();
  SmallVector<Attribute, 4> encodings;

  auto addEncoding = [&](Value value) {
    if (Attribute encoding = getAssignedEncoding(value, assignments))
      if (!llvm::is_contained(encodings, encoding))
        encodings.push_back(encoding);
  };

  if (Value required = LayoutOperationContract::requiredEncodingValue(use)) {
    addEncoding(required);
    return encodings;
  }
  if (LayoutOperationContract::isControlFlow(user))
    return encodings;

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
    Attribute operandEncoding =
        getContractEncoding</*source=*/true>(user, resultEncoding);
    if (operandEncoding && !llvm::is_contained(encodings, operandEncoding))
      encodings.push_back(operandEncoding);
  }

  if (encodings.empty())
    if (auto type = dyn_cast<RankedTensorType>(use.get().getType()))
      encodings.push_back(type.getEncoding());
  return encodings;
}

uint64_t LayoutOptimizationAnalysis::getAssignmentCost(
    Value value, Attribute encoding,
    const LayoutAssignment &assignments) const {
  uint64_t cost = 0;
  Operation *definingOp = value.getDefiningOp();
  if (definingOp)
    cost += costModel.getRegisterPressureCost(value, encoding) *
            costModel.getExecutionWeight(definingOp);
  llvm::SmallDenseMap<Attribute, uint64_t, 8> userWeights;

  for (OpOperand &use : value.getUses()) {
    uint64_t weight = costModel.getExecutionWeight(use.getOwner());
    if (auto reduce = dyn_cast<ReduceOp>(use.getOwner()))
      cost +=
          costModel.getReductionCost(value, encoding, reduce.getAxis()) *
          weight;
    for (Attribute required : getUseEncodings(use, assignments)) {
      auto [it, inserted] = userWeights.try_emplace(required, weight);
      if (!inserted)
        it->second = std::max(it->second, weight);
    }
  }

  for (const auto &[required, weight] : userWeights)
    cost += costModel.getTransitionCost(value, encoding, required) * weight;

  if (!definingOp)
    return cost;

  uint64_t weight = costModel.getExecutionWeight(definingOp);
  if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
    auto result = cast<OpResult>(value);
    for (Value tied : getTiedArgs(definingOp, result.getResultNumber())) {
      if (tied == value || !isa<RankedTensorType>(tied.getType()))
        continue;
      cost +=
          costModel.getTransitionCost(tied,
                                      getAssignedEncoding(tied, assignments),
                                      encoding) *
          weight;
    }
    return cost;
  }

  Attribute operandEncoding =
      getContractEncoding</*source=*/true>(definingOp, encoding);
  if (!operandEncoding)
    return cost;

  for (Value operand : definingOp->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    cost += costModel.getTransitionCost(
                operand, getAssignedEncoding(operand, assignments),
                operandEncoding) *
            weight;
  }
  return cost;
}

uint64_t LayoutOptimizationAnalysis::getGlobalAssignmentCost(
    const LayoutAssignment &assignments) const {
  uint64_t cost = 0;
  for (const auto &[value, info] : layouts) {
    Attribute encoding = getAssignedEncoding(value, assignments);
    if (encoding)
      cost += getAssignmentCost(value, encoding, assignments);
  }
  return cost;
}

uint64_t LayoutOptimizationAnalysis::getAffectedAssignmentCost(
    ArrayRef<Value> changedValues, const LayoutAssignment &assignments) const {
  llvm::SmallSetVector<Value, 32> affected;
  auto addValue = [&](Value value) {
    if (layouts.find(value) != layouts.end())
      affected.insert(value);
  };
  auto addOperation = [&](Operation *op) {
    if (!op)
      return;
    for (Value value : llvm::concat<Value>(op->getOperands(), op->getResults()))
      addValue(value);
  };
  auto addControlFlow = [&](Operation *op, unsigned resultIndex) {
    if (!op || resultIndex >= op->getNumResults())
      return;
    for (Value tied : getTiedArgs(op, resultIndex))
      addValue(tied);
  };

  for (Value changed : changedValues) {
    addValue(changed);
    if (Operation *producer = changed.getDefiningOp())
      addOperation(producer);
    if (auto [controlFlow, index] = LayoutOperationContract::component(changed);
        controlFlow)
      addControlFlow(controlFlow, index);

    for (OpOperand &use : changed.getUses()) {
      Operation *user = use.getOwner();
      addOperation(user);
      auto [controlFlow, index] =
          LayoutOperationContract::component(use, /*includeConditionals=*/true);
      addControlFlow(controlFlow, index);
    }
  }

  uint64_t cost = 0;
  for (Value value : affected)
    if (Attribute encoding = getAssignedEncoding(value, assignments))
      cost += getAssignmentCost(value, encoding, assignments);
  return cost;
}

bool LayoutOptimizationAnalysis::buildGlobalComponentProposal(
    Value seed, Attribute encoding, LayoutAssignment &assignments,
    SmallVectorImpl<LayoutValue> &changes) const {
  constexpr unsigned maxComponentValues = 512;
  LayoutAssignment requested;
  SmallVector<Value, 32> worklist;

  auto rollback = [&]() {
    for (const auto &[value, original] : llvm::reverse(changes))
      assignments[value] = original;
    changes.clear();
    return false;
  };

  auto request = [&](Value value, Attribute candidate) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !candidate)
      return true;

    auto found = layouts.find(value);
    if (found == layouts.end() || !found->second.contains(candidate))
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
    worklist.push_back(value);
    return true;
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
    Value value = worklist.pop_back_val();
    Attribute candidate = requested.lookup(value);
    Attribute original = assignments.lookup(value);
    if (original != candidate) {
      changes.push_back({value, original});
      assignments[value] = candidate;
    }

    if (auto [loop, index] = LayoutOperationContract::component(value);
        loop && isa<scf::ForOp, scf::WhileOp>(loop) &&
        !requestLoopComponent(loop, index, candidate))
      return rollback();

    if (Operation *producer = value.getDefiningOp()) {
      if (isa<SplitOp, ReduceOp>(producer))
        for (Value sibling : producer->getResults())
          if (!request(sibling, candidate))
            return rollback();
      if (isLayoutTransform(producer) && !isFixedLayoutBoundary(producer)) {
        Attribute source =
            getContractEncoding</*source=*/true>(producer, candidate);
        if (source) {
          for (Value operand : producer->getOperands())
            if (!request(operand, source))
              return rollback();
        }
      }
    }

    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      auto [loop, index] = LayoutOperationContract::component(use);
      if (loop && !requestLoopComponent(loop, index, candidate))
        return rollback();
      if (loop || LayoutOperationContract::isControlFlow(user))
        continue;
      if (!isLayoutTransform(user) || isFixedLayoutBoundary(user))
        continue;

      Attribute destination =
          getContractEncoding</*source=*/false>(user, candidate);
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

template <typename Problem>
void LayoutAssignmentSolver::solve(const LayoutConstraintGraph &graph,
                                   const LayoutSearchPolicy &policy,
                                   LayoutAssignment &assignments,
                                   Problem &problem) {
  const bool useFullObjective = policy.fullObjective;
  if (policy.componentSearch) {
    constexpr unsigned maxComponentIterations = 4;
    const unsigned maxComponentProposals = policy.maxComponentProposals;
    const bool pruneRedundantReductionProposals =
        policy.pruneRedundantReductionProposals;
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

          SmallVector<LayoutValue, 32> proposal;
          if (!problem.buildGlobalComponentProposal(value, candidate,
                                                    assignments, proposal) ||
              proposal.empty())
            continue;

          SmallVector<Value, 32> changedValues;
          for (const auto &[changed, original] : proposal)
            changedValues.push_back(changed);
          uint64_t affectedProposalCost =
              problem.getAffectedAssignmentCost(changedValues, assignments);
          for (auto &[changed, original] : proposal)
            std::swap(assignments[changed], original);
          uint64_t previousCost =
              problem.getAffectedAssignmentCost(changedValues, assignments);
          if (affectedProposalCost >= previousCost)
            continue;

          for (auto &[changed, original] : proposal)
            std::swap(assignments[changed], original);
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
      auto getCost = [&](Attribute candidate) {
        if (!useFullObjective)
          return problem.getAssignmentCost(value, candidate, assignments);
        std::swap(assignments[value], candidate);
        uint64_t result =
            problem.getAffectedAssignmentCost({value}, assignments);
        std::swap(assignments[value], candidate);
        return result;
      };
      uint64_t bestCost = getCost(original);
      for (Attribute candidate : node.candidates) {
        if (candidate == original ||
            !problem.canAssignEncoding(value, candidate, assignments))
          continue;
        uint64_t candidateCost = getCost(candidate);
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
      if (problem.canAssignEncoding(value, assignments.lookup(value),
                                    assignments))
        continue;
      assignments[value] =
          cast<RankedTensorType>(value.getType()).getEncoding();
      changed = true;
    }
    if (!changed)
      break;
  }

  if (!policy.exactComponentSearch)
    return;

  // Coordinate descent cannot change jointly constrained encodings when every
  // intermediate assignment is illegal. Search the bounded components exactly.
  for (const SmallVector<unsigned, 8> &component :
       graph.getConnectedComponents()) {
    if (component.size() < 2)
      continue;
    unsigned states = 1;
    for (unsigned index : component)
      if ((states *= graph.getNodes()[index].candidates.size()) >
          policy.maxExactComponentStates)
        break;
    if (states > policy.maxExactComponentStates)
      continue;

    uint64_t bestCost = problem.getGlobalAssignmentCost(assignments);
    SmallVector<Attribute, 8> bestAssignments;
    for (unsigned index : component)
      bestAssignments.push_back(
          assignments.lookup(graph.getNodes()[index].value));

    auto search = [&](auto &&self, unsigned position) -> void {
      if (position == component.size()) {
        for (const LayoutConstraintGraph::Node &node : graph.getNodes())
          if (!problem.canAssignEncoding(
                  node.value, assignments.lookup(node.value), assignments))
            return;
        uint64_t candidateCost = problem.getGlobalAssignmentCost(assignments);
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
      for (Attribute candidate : node.candidates) {
        assignments[node.value] = candidate;
        self(self, position + 1);
      }
      assignments[node.value] = original;
    };
    search(search, 0);
    for (auto [index, value] : llvm::enumerate(component))
      assignments[graph.getNodes()[value].value] = bestAssignments[index];
  }
}

void LayoutOptimizationAnalysis::resolveGlobalConflicts() {
  LayoutAssignment assignments;
  bool hasFlexibleLayouts = false;
  for (auto &[value, info] : layouts) {
    Attribute original = cast<RankedTensorType>(value.getType()).getEncoding();
    info.insert(original);
    assignments.try_emplace(value, original);
    hasFlexibleLayouts |=
        !fixedLayouts.contains(value) && info.size() > 1;
  }

  if (hasFlexibleLayouts) {
    LayoutConstraintGraph graph;
    for (const auto &[value, info] : layouts)
      graph.addNode(value, info.getArrayRef(), fixedLayouts.contains(value));
    funcOp.walk([&](Operation *op) { graph.addOperation(op); });
    LayoutSearchPolicy policy(graph);
    LDBG("resolving " << layouts.size() << " layout values with "
                      << (policy.fullObjective ? "the full" : "the bounded")
                      << " global objective");

    LayoutAssignmentSolver::solve(graph, policy, assignments, *this);
  }

  for (auto &[value, info] : layouts) {
    Attribute encoding = assignments.lookup(value);
    info.clear();
    info.insert(encoding);
  }
}

void LayoutOptimizationAnalysis::rewrite() {
  rewriteRegion(funcOp->getRegion(0));
}

void LayoutOptimizationAnalysis::rewriteRegion(Region &region) {
  std::deque<Region *> queue = {&region};
  while (!queue.empty()) {
    Region *currentRegion = queue.front();
    queue.pop_front();
    for (Operation &op : currentRegion->getOps()) {
      bool needRewrite = llvm::any_of(op.getResults(), [&](Value result) {
        auto it = layouts.find(result);
        if (it == layouts.end())
          return false;
        LayoutInfo &info = it->second;
        assert(info.size() == 1 &&
               "we should have resolved to a single encoding");
        return cast<RankedTensorType>(result.getType()).getEncoding() !=
               *info.begin();
      });
      if (needRewrite) {
        rewriteOp(&op);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(&op)) {
        Operation *parent = yieldOp->getParentOp();
        if (auto whileOp = dyn_cast<scf::WhileOp>(parent))
          rewriteControlFlowOperands(yieldOp, whileOp.getBeforeArguments());
        else if (isa<scf::ForOp, scf::IfOp>(parent))
          rewriteControlFlowOperands(yieldOp, parent->getResults());
      } else if (auto conditionOp = dyn_cast<scf::ConditionOp>(&op)) {
        auto whileOp = cast<scf::WhileOp>(conditionOp->getParentOp());
        rewriteControlFlowOperands(conditionOp, whileOp.getResults(), 1);
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
      }
      for (Region &nested : op.getRegions())
        queue.push_back(&nested);
    }
  }
}

Value LayoutOptimizationAnalysis::getValueAs(Value value, Attribute encoding) {
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    if (tensorType.getEncoding() == encoding)
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

Attribute
LayoutOptimizationAnalysis::getEncodingBeforeRewrite(Value value) const {
  auto tensorType = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorType)
    return {};
  if (auto it = originalEncodings.find(value); it != originalEncodings.end())
    return it->second;
  return tensorType.getEncoding();
}

void LayoutOptimizationAnalysis::setEncodingInPlace(Value value,
                                                    Attribute encoding) {
  auto tensorType = cast<RankedTensorType>(value.getType());
  originalEncodings.try_emplace(value, tensorType.getEncoding());
  value.setType(tensorType.cloneWithEncoding(encoding));
}

void LayoutOptimizationAnalysis::rewriteGenericOpInPlace(Operation *op,
                                                         Attribute encoding) {
  Attribute operandEnc;
  if (op->getNumOperands() > 0) {
    for (Value operand : op->getOperands()) {
      auto it = layouts.find(operand);
      if (it == layouts.end())
        continue;
      Attribute enc = it->second[0];
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
  for (Value result : op->getResults())
    if (isa<RankedTensorType>(result.getType()))
      setEncodingInPlace(result, encoding);
}

void LayoutOptimizationAnalysis::rewriteForOp(scf::ForOp forOp) {
  rewriteAssignedValues(forOp.getResults(), [&](unsigned index, Value result,
                                                Attribute encoding) {
    forOp.getInitArgsMutable()[index].assign(
        getValueAs(forOp.getInitArgs()[index], encoding));
    setEncodingInPlace(result, encoding);
    setEncodingInPlace(forOp.getRegionIterArg(index), encoding);
  });
}

void LayoutOptimizationAnalysis::rewriteWhileOp(scf::WhileOp whileOp) {
  rewriteAssignedValues(
      whileOp.getBeforeArguments(),
      [&](unsigned index, Value argument, Attribute encoding) {
        whileOp->setOperand(index,
                            getValueAs(whileOp->getOperand(index), encoding));
        setEncodingInPlace(argument, encoding);
      });
  rewriteAssignedValues(whileOp.getResults(), [&](unsigned index, Value result,
                                                  Attribute encoding) {
    setEncodingInPlace(result, encoding);
    setEncodingInPlace(whileOp.getAfterArguments()[index], encoding);
  });
}

void LayoutOptimizationAnalysis::rewriteIfOp(scf::IfOp ifOp) {
  rewriteAssignedValues(ifOp.getResults(),
                        [&](unsigned, Value result, Attribute encoding) {
                          setEncodingInPlace(result, encoding);
                        });
}

void LayoutOptimizationAnalysis::rewriteControlFlowOperands(
    Operation *operation, ValueRange targets, unsigned firstOperand) {
  for (auto [index, target] : llvm::enumerate(targets))
    if (auto type = dyn_cast<RankedTensorType>(target.getType())) {
      unsigned operand = firstOperand + index;
      operation->setOperand(operand, getValueAs(operation->getOperand(operand),
                                                type.getEncoding()));
    }
}

void LayoutOptimizationAnalysis::rewriteReduceToScalar(Operation *reduceOp) {
  Attribute srcEncoding;
  // Since all the operands need to have the same encoding pick the first one
  // and use it for all the operands.
  for (Value operand : reduceOp->getOperands()) {
    auto it = layouts.find(operand);
    if (it != layouts.end()) {
      srcEncoding = it->second[0];
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

void LayoutOptimizationAnalysis::rewriteAssertOp(AssertOp assertOp) {
  // Only need to deal with the first operand which is the condition tensor.
  Value operand = assertOp->getOperand(0);
  auto it = layouts.find(operand);
  if (it == layouts.end())
    return;
  assertOp->setOperand(0, getValueAs(operand, it->second[0]));
}

void LayoutOptimizationAnalysis::rewriteOp(Operation *op) {
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    rewriteForOp(forOp);
  else if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    rewriteWhileOp(whileOp);
  else if (auto ifOp = dyn_cast<scf::IfOp>(op))
    rewriteIfOp(ifOp);
  else {
    Attribute encoding = *layouts[op->getResult(0)].begin();
    if (canUseResultEncoding(op, encoding)) {
      setEncodingInPlace(op->getResult(0), encoding);
      if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
        auto elements = cast<DenseElementsAttr>(constant.getValue());
        auto resultType = cast<RankedTensorType>(constant.getType());
        constant.setValueAttr(elements.reshape(resultType));
      }
    } else if (isLayoutTransform(op) ||
               isa<GatherOp, nvidia_gpu::WarpGroupDotWaitOp>(op)) {
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
  return !isa<scf::WhileOp, scf::ConditionOp>(op);
}

void LayoutOptimizationAnalysis::updateRematMapping(
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

void LayoutOptimizationAnalysis::rewriteSlice(LayoutSlice &state,
                                              ConvertLayoutOp convertOp,
                                              IRMapping &mapping) {
  auto &[slice, layout, existingRemats] = state;
  for (const auto &[value, encoding] : layout)
    if (Value remat = existingRemats.lookup({value, encoding}))
      mapping.map(value, remat);

  SetVector<Operation *> opsToRewrite;
  // Keep track of yield operands that need to be duplicated.
  DenseMap<Operation *, SmallVector<int>> yieldOperandsMap;
  for (Value v : slice) {
    if (Operation *producer = v.getDefiningOp()) {
      opsToRewrite.insert(producer);
      if (auto ifOp = dyn_cast<scf::IfOp>(producer)) {
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
          argMapping.emplace_back(
              forOp.getTiedLoopResult(&initVal).getResultNumber(),
              forOp.getInitArgs().size() + newOperands.size());
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
      for (Value result : ifOp.getResults())
        if (slice.count(result))
          newTypes.push_back(cast<RankedTensorType>(result.getType())
                                 .cloneWithEncoding(layout.at(result)));
      scf::IfOp newIfOp =
          replaceIfOpWithNewSignature(builder, ifOp, newTypes, replacements);
      unsigned newIdx = ifOp.getNumResults();
      for (Value result : ifOp.getResults())
        if (slice.count(result)) {
          mapping.map(result, newIfOp.getResult(newIdx));
          addRematValue(result, layout[result], newIfOp.getResult(newIdx));
          ++newIdx;
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

void LayoutOptimizationAnalysis::rewriteSlice(LayoutSlice &slice,
                                              ConvertLayoutOp convertOp) {
  IRMapping mapping;
  rewriteSlice(slice, convertOp, mapping);
}

LogicalResult LayoutOptimizationAnalysis::getRematerializableSlice(
    OpOperand &root, Attribute rootEncoding, LayoutSlice &state,
    std::function<bool(Operation *)> stopPropagation,
    bool requireRematerializable) {
  LayoutSlice candidate = state;
  auto &[slice, layout, existingRemats] = candidate;
  // Allow re-using existing conversions for a value if it dominates the use.
  auto getExistingConversion = [&](OpOperand &value, Attribute encoding) {
    Value remat = rematMapping.lookup(value.get()).lookup(encoding);
    if (!remat)
      return Value();
    // `value` can be replaced with an existing rematerialization if it
    // dominates the current use of value.
    Operation *user = value.getOwner();
    if (domInfo.properlyDominates(remat, user)) {
      existingRemats.try_emplace({value.get(), encoding}, remat);
      return remat;
    }
    // There is an existing rematerialization, but it doesn't dominate all the
    // uses we care about, so ensure it isn't used.
    existingRemats[{value.get(), encoding}] = Value();
    return Value();
  };

  if (failed(mlir::getConvertBackwardSlice(root, slice, rootEncoding, layout,
                                           stopPropagation,
                                           getExistingConversion)))
    return failure();

  if (requireRematerializable)
    for (Value value : slice)
      if (Operation *op = value.getDefiningOp(); op && !canBeRemat(op))
        return failure();
  state = std::move(candidate);
  return success();
}

bool LayoutCostModel::isExpensiveMathOp(Operation *op) {
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

int64_t LayoutCostModel::getByteCount(Value result, int64_t minElementCount,
                                      int64_t minBitWidth) {
  int64_t elementCount = 0, bitWidth = 0;
  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    elementCount = tensorTy.getNumElements();
    if (Type element = tensorTy.getElementType(); element.isIntOrFloat())
      bitWidth = element.getIntOrFloatBitWidth();
  }
  return (std::max(elementCount, minElementCount) *
          std::max(bitWidth, minBitWidth)) >>
         3;
}

/// Determine whether rematerializing \p slice is beneficial given that it will
/// eliminate \p convertOp and require creating new convert ops with cost \p
/// newCvtCost.
bool LayoutCostModel::isRematerializationBeneficial(
    ConvertLayoutOp convertOp, const LayoutSlice &state, int64_t newCvtCost,
    bool disableRematSplitting,
    bool preserveSharedReductionRematerialization) const {
  const SetVector<Value> &slice = state.values;
  const LayoutAssignment &layout = state.encodings;
  // Identify all operations in the slice
  SetVector<Operation *> sliceOps;
  for (Value value : slice)
    if (Operation *operation = value.getDefiningOp())
      sliceOps.insert(operation);

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

  int64_t convertLayoutCost = getTransitionCost(
      convertOp.getSrc(),
      cast<RankedTensorType>(convertOp.getSrc().getType()).getEncoding(),
      convertOp.getType().getEncoding(), /*rematerialization=*/true);
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
        auto type = cast<RankedTensorType>(result.getType());
        unsigned factor = std::max(
            1u, getUniqueElemsPerThread(rematEncoding, type.getShape()) /
                    getUniqueElemsPerThread(type));
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

bool LayoutOptimizationAnalysis::backwardRematerialization(
    ConvertLayoutOp convertOp) {
  // DotOperand is hoisted by hoistDotOperand
  RankedTensorType targetType = convertOp.getType();
  if (isa<DotOperandEncodingAttr>(targetType.getEncoding()))
    return false;
  LDBG("check backward remat with source " << convertOp.getSrc() << " encoding "
                                           << targetType.getEncoding());
  // 1. Take a backward slice of all the tensor dependencies that can be
  // rematerialized.
  LayoutSlice state;
  auto &[slice, layout, existingRemats] = state;
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), state);
  if (result.failed()) {
    LDBG("  getRematerializableSlice failed");
    return false;
  }

  // 2. Determine whether rematerialisation is beneficial.
  if (!costModel.isRematerializationBeneficial(
          convertOp, state, /*newConversionCost=*/0, disableRematSplitting,
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
  rewriteSlice(state, convertOp);
  return true;
}

bool LayoutOptimizationAnalysis::hoistConvertDotOperand(
    ConvertLayoutOp convertOp) {
  auto targetType = convertOp.getType();
  // The pass is targeted to MMA dot operands

  auto canBePipelined = [&](ConvertLayoutOp convertOp) {
    // FIXME: Check that the parent is a for loop
    auto parent = convertOp->getParentOp();
    if (!parent)
      return false;

    // Stop at the first MMA dot that post-dominates the load and conversion.
    return parent
        ->walk([&](Operation *operation) {
          if (isa<mlir::triton::DotOpInterface>(operation))
            if (auto type = dyn_cast<RankedTensorType>(
                    operation->getOperand(0).getType()))
              if (auto encoding =
                      dyn_cast<DotOperandEncodingAttr>(type.getEncoding()))
                if (isa<MmaEncodingTrait>(encoding.getParent()) &&
                    postDomInfo.postDominates(operation, convertOp))
                  return WalkResult::interrupt();
          return WalkResult::advance();
        })
        .wasInterrupted();
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

  LayoutSlice state;
  auto &[slice, layout, existingRemats] = state;
  // Set-up the conversion "cache"
  LogicalResult result = getRematerializableSlice(
      convertOp.getSrcMutable(), targetType.getEncoding(), state, stop,
      /*requireRematerializable=*/false);
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

  if (innerSlice.empty())
    return false;

  LLVM_DEBUG({
    DBGS() << "  Hoisting " << convertOp << '\n';
    for (Value v : innerSlice)
      DBGS() << "    " << v << '\n';
  });

  state.values = std::move(innerSlice);
  rewriteSlice(state, convertOp, mapping);
  return true;
}

// For convert left we try to hoist them above type extension to reduce the cost
// of the convert.
bool LayoutOptimizationAnalysis::hoistConvertOnTopOfExtOrBroadcast(
    ConvertLayoutOp convertOp) {
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
  LayoutSlice state;
  auto &[slice, layout, existingRemats] = state;
  LogicalResult result = getRematerializableSlice(convertOp.getSrcMutable(),
                                                  targetType.getEncoding(),
                                                  state, isExtOrBroadcastOp);
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
    if (succeeded(
            getRematerializableSlice(op->getOpOperand(0), srcEncoding, state)))
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
  Value operand = extOrBroadcastOp->getOperand(0);
  int64_t newCvtCost = costModel.getTransitionCost(
      operand, cast<RankedTensorType>(operand.getType()).getEncoding(),
      srcEncoding, /*rematerialization=*/true);
  if (!costModel.isRematerializationBeneficial(convertOp, state, newCvtCost,
                                               disableRematSplitting))
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
  rewriteSlice(state, convertOp, mapping);
  return true;
}

bool LayoutOptimizationAnalysis::hoistConvertIntoConditionals(
    ConvertLayoutOp convertOp) {
  // Take the backward slice of tensor dependencies rooted at the conversion,
  // stopping at conditionals. This subslice is used to initialize the analysis.
  LayoutSlice state;
  auto &[slice, layout, existingRemats] = state;
  auto isIfOp = [](Operation *op) { return isa<scf::IfOp>(op); };
  if (failed(getRematerializableSlice(convertOp.getSrcMutable(),
                                      convertOp.getType().getEncoding(), state,
                                      isIfOp)))
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
    SmallVector<OpOperand *, 2> edges;
    for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()})
      edges.push_back(&cast<scf::YieldOp>(region->front().getTerminator())
                           .getResultsMutable()[resIdx]);
    LayoutSlice candidate = state;
    bool first = succeeded(
        getRematerializableSlice(*edges[0], rootLayout, candidate, isIfOp));
    bool second = succeeded(
        getRematerializableSlice(*edges[1], rootLayout, candidate, isIfOp));
    if (first == second) {
      if (first)
        state = std::move(candidate);
      else
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

    state = std::move(candidate);
    // The layout conversion can be rematerialized along one edge but not the
    // other. We can hoist the conversion into the other branch. Push this
    // into the subslice list for analysis.
    hoistAbove.emplace_back(v, edges[first]);
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
  rewriteSlice(state, convertOp, mapping);
  return true;
}

LogicalResult LayoutOptimizationAnalysis::cleanup() {
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  ConvertLayoutOp::getCanonicalizationPatterns(patterns, context);
  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    return failure();

  LLVM_DEBUG({
    DBGS() << "Function after canonicalizing:\n";
    funcOp.dump();
  });
  return success();
}

/// Plan scalar-rooted tensor expressions in either distributed-layout or
/// logical-index coordinates. Logical-index evaluation additionally supports
/// exact-order joins and reshapes whose layouts cannot be inferred backward.
class LayoutOptimizationAnalysis::ExpressionPlan {
public:
  explicit ExpressionPlan(ConvertLayoutOp root, bool indexed = false)
      : insertionPoint(root.getOperation()), rootValue(root.getSrc()),
        target(root.getType()), indexed(indexed) {}

  explicit ExpressionPlan(triton::StoreOp root)
      : insertionPoint(root.getOperation()), rootValue(root.getValue()),
        target(dyn_cast<RankedTensorType>(root.getValue().getType())),
        indexed(true) {}

  LogicalResult analyze() {
    if (!target || !target.getEncoding())
      return failure();
    if (indexed) {
      if (!target.hasStaticShape())
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
              flatEncoding, /*allowReorder=*/false,
              insertionPoint->getLoc())) ||
          (isa<triton::StoreOp>(insertionPoint) && !rootValue.hasOneUse()))
        return failure();
    }

    if (failed(plan(rootValue, target.getEncoding(), 0)))
      return failure();

    unsigned conversions = 0, joins = 0;
    bool hasCommunication = false;
    auto classifyConversion = [&](ConvertLayoutOp convert) {
      ++conversions;
      hasCommunication |=
          !cvtReordersRegisters(convert.getSrc().getType(), convert.getType());
    };
    if (indexed)
      if (auto conversion = dyn_cast<ConvertLayoutOp>(insertionPoint))
        classifyConversion(conversion);

    for (Operation *operation : originalOperations) {
      if (indexed) {
        if (auto conversion = dyn_cast<ConvertLayoutOp>(operation))
          classifyConversion(conversion);
        joins += isa<triton::JoinOp>(operation);
      }
      // Inexpensive scalar leaves may belong to an enclosing expression.
      if (!indexed || !isRematerializableLeaf(operation))
        if (llvm::any_of(operation->getUsers(), [&](Operation *user) {
              return user == insertionPoint
                         ? indexed && operation->getResult(0) != rootValue
                         : !originalOperations.contains(user);
            }))
          return failure();
    }

    if (indexed && (joins == 0 || conversions < 2 || !hasCommunication))
      return failure();

    auto isArithmetic = [&](Operation *operation) {
      return indexed ? !isStructuralOrLeaf(operation)
                     : !isa<ConvertLayoutOp>(operation);
    };
    return success(llvm::count_if(plannedValues, [&](const auto &planned) {
                     return isArithmetic(planned.first.first.getDefiningOp());
                   }) <= llvm::count_if(originalOperations, isArithmetic));
  }

  void rewrite() {
    OpBuilder builder(insertionPoint);
    if (indexed) {
      Location loc = insertionPoint->getLoc();
      auto rangeType = RankedTensorType::get(
          {target.getNumElements()}, builder.getI32Type(), flatEncoding);
      Value range = triton::MakeRangeOp::create(
          builder, loc, rangeType, 0,
          static_cast<int32_t>(target.getNumElements()));
      auto indexType = RankedTensorType::get(
          target.getShape(), builder.getI32Type(), target.getEncoding());
      indices = triton::ReshapeOp::create(builder, loc, indexType, range);
    }

    Value replacement =
        materialize(rootValue, target.getEncoding(), 0, builder);
    assert(replacement && "a validated expression plan must materialize");

    if (auto conversion = dyn_cast<ConvertLayoutOp>(insertionPoint)) {
      conversion.getResult().replaceAllUsesWith(replacement);
      conversion.erase();
    } else {
      cast<triton::StoreOp>(insertionPoint).getValueMutable().assign(
          replacement);
    }
    for (Operation *operation : originalOperations)
      if (isOpTriviallyDead(operation))
        operation->erase();
  }

private:
  using Coordinate = std::pair<Value, std::pair<Attribute, unsigned>>;
  static constexpr unsigned maxJoinDepth = 30;

  static bool isRematerializableLeaf(Operation *operation) {
    return isa<triton::SplatOp, triton::MakeRangeOp, arith::ConstantOp>(
        operation);
  }

  static bool isStructuralOrLeaf(Operation *operation) {
    return isRematerializableLeaf(operation) ||
           isa<ConvertLayoutOp, triton::ReshapeOp, triton::JoinOp>(operation);
  }

  LogicalResult plan(Value value, Attribute encoding, unsigned depth) {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || !encoding ||
        (indexed &&
         (!type.hasStaticShape() || depth > maxJoinDepth ||
          (target.getNumElements() >> depth) != type.getNumElements())))
      return failure();

    Coordinate key{value, {encoding, depth}};
    if (plannedValues.contains(key))
      return success();
    Operation *operation = value.getDefiningOp();
    if (plannedValues.size() >= 512 || !operation ||
        operation->getNumResults() != 1 || !isMemoryEffectFree(operation) ||
        (operation->getBlock() != insertionPoint->getBlock() &&
         (!indexed || !isRematerializableLeaf(operation))))
      return failure();
    originalOperations.insert(operation);

    Attribute operandEncoding;
    bool joined = false, sameShape = false;
    if (isa<ConvertLayoutOp>(operation) ||
        (indexed && isa<triton::ReshapeOp>(operation))) {
      if (auto reshape = dyn_cast<triton::ReshapeOp>(operation);
          reshape && reshape.getAllowReorder())
        return failure();
      operandEncoding = encoding;
    } else if (indexed && isa<triton::JoinOp>(operation)) {
      auto join = cast<triton::JoinOp>(operation);
      RankedTensorType resultType = join.getType();
      if (resultType.getShape().empty() || resultType.getShape().back() != 2 ||
          failed(plan(join.getLhs(), encoding, depth + 1)) ||
          failed(plan(join.getRhs(), encoding, depth + 1)))
        return failure();
      joined = true;
    } else if (auto splat = dyn_cast<triton::SplatOp>(operation)) {
      if (indexed && isa<RankedTensorType>(splat.getSrc().getType()))
        return failure();
    } else if (auto range = dyn_cast<triton::MakeRangeOp>(operation)) {
      if (indexed && !range.getType().getElementType().isInteger(32))
        return failure();
    } else if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      auto elements = dyn_cast<DenseElementsAttr>(constant.getValue());
      if (!elements || (indexed && !elements.isSplat()))
        return failure();
    } else if (indexed) {
      if (!operation->hasTrait<OpTrait::Elementwise>())
        return failure();
      operandEncoding = encoding;
      sameShape = true;
    } else {
      if ((!isLayoutTransform(operation) &&
           !isa<triton::BroadcastOp>(operation)) ||
          isa<ReduceOp, SplitOp>(operation) ||
          !(operandEncoding = inferSrcEncoding(operation, encoding)))
        return failure();
    }

    if (!joined)
      for (Value operand : operation->getOperands()) {
        auto operandType = dyn_cast<RankedTensorType>(operand.getType());
        if (!operandType)
          continue;
        if (!operandEncoding ||
            (sameShape && operandType.getShape() != type.getShape()) ||
            failed(plan(operand, operandEncoding, depth)))
          return failure();
      }

    plannedValues.try_emplace(key, operandEncoding, Value{});
    return success();
  }

  Value createIndexConstant(OpBuilder &builder, Location loc, int32_t value) {
    auto type = cast<RankedTensorType>(indices.getType());
    return arith::ConstantOp::create(
        builder, loc, type,
        DenseElementsAttr::get(type, builder.getI32IntegerAttr(value)));
  }

  Value getJoinCondition(unsigned depth, OpBuilder &builder, Location loc) {
    if (auto found = joinConditions.find(depth); found != joinConditions.end())
      return found->second;
    Value mask = createIndexConstant(builder, loc, int32_t{1} << depth);
    Value zero = createIndexConstant(builder, loc, 0);
    Value masked = arith::AndIOp::create(builder, loc, indices, mask);
    Value condition = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ne, masked, zero);
    joinConditions.try_emplace(depth, condition);
    return condition;
  }

  Value materialize(Value value, Attribute encoding, unsigned depth,
                    OpBuilder &builder) {
    Coordinate key{value, {encoding, depth}};
    auto &[operandEncoding, materialized] = plannedValues.find(key)->second;
    if (materialized)
      return materialized;

    Operation *operation = value.getDefiningOp();
    Value result;
    if (isa<ConvertLayoutOp>(operation) ||
        (indexed && isa<triton::ReshapeOp>(operation))) {
      result = materialize(operation->getOperand(0), encoding, depth, builder);
    } else if (indexed && isa<triton::JoinOp>(operation)) {
      auto join = cast<triton::JoinOp>(operation);
      Value lhs = materialize(join.getLhs(), encoding, depth + 1, builder);
      Value rhs = materialize(join.getRhs(), encoding, depth + 1, builder);
      Value condition = getJoinCondition(depth, builder, join.getLoc());
      result =
          arith::SelectOp::create(builder, join.getLoc(), condition, rhs, lhs);
    } else if (indexed && isa<triton::MakeRangeOp>(operation)) {
      auto range = cast<triton::MakeRangeOp>(operation);
      result = indices;
      if (depth != 0) {
        Value shift = createIndexConstant(builder, range.getLoc(), depth);
        result = arith::ShRUIOp::create(builder, range.getLoc(), result, shift);
      }
      if (int64_t start = range.getStartAttr().getInt(); start != 0) {
        Value offset = createIndexConstant(builder, range.getLoc(), start);
        result = arith::AddIOp::create(builder, range.getLoc(), result, offset);
      }
    } else {
      RankedTensorType oldType = cast<RankedTensorType>(value.getType());
      RankedTensorType newType =
          indexed ? RankedTensorType::get(target.getShape(),
                                          oldType.getElementType(), encoding)
                  : oldType.cloneWithEncoding(encoding);
      if (indexed && isa<triton::SplatOp>(operation)) {
        auto splat = cast<triton::SplatOp>(operation);
        result = triton::SplatOp::create(builder, splat.getLoc(), newType,
                                         splat.getSrc());
      } else if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
        auto elements = cast<DenseElementsAttr>(constant.getValue());
        auto newElements =
            indexed ? DenseElementsAttr::get(
                          newType, elements.getSplatValue<Attribute>())
                    : elements.reshape(newType);
        result = arith::ConstantOp::create(builder, constant.getLoc(), newType,
                                           newElements);
      } else {
        IRMapping mapping;
        for (Value operand : operation->getOperands())
          if (isa<RankedTensorType>(operand.getType()))
            mapping.map(operand,
                        materialize(operand, operandEncoding, depth, builder));
        Operation *replacement = builder.clone(*operation, mapping);
        replacement->getResult(0).setType(newType);
        result = replacement->getResult(0);
      }
    }
    return materialized = result;
  }

  Operation *insertionPoint;
  Value rootValue;
  RankedTensorType target;
  bool indexed;
  Attribute flatEncoding;
  Value indices;
  DenseMap<Coordinate, std::pair<Attribute, Value>> plannedValues;
  SetVector<Operation *> originalOperations;
  DenseMap<unsigned, Value> joinConditions;
};

/// A layout conversion is pure. One materialization can therefore serve every
/// dominated request for the same source value and destination encoding.
void LayoutOptimizationAnalysis::shareDominatingConversions() {
  domInfo.invalidate();
  DenseMap<LayoutValue, SmallVector<ConvertLayoutOp>> available;
  funcOp.walk([&](ConvertLayoutOp convert) {
    LayoutValue key{convert.getSrc(), convert.getType().getEncoding()};
    auto &candidates = available[key];
    auto candidate = llvm::find_if(candidates, [&](ConvertLayoutOp existing) {
      return domInfo.properlyDominates(existing.getResult(),
                                       convert.getOperation());
    });
    if (candidate == candidates.end()) {
      candidates.push_back(convert);
      return;
    }
    convert.getResult().replaceAllUsesWith(candidate->getResult());
    convert.erase();
  });
}

/// Rebuild the root worklist after each rewrite because a successful whole-
/// expression plan can erase other roots and their producers.
template <typename Root>
void LayoutOptimizationAnalysis::optimizeExpressions() {
  auto rewrite = [](auto &&plan) {
    if (failed(plan.analyze()))
      return false;
    plan.rewrite();
    return true;
  };
  for (;;) {
    SmallVector<Root> roots;
    funcOp.walk([&](Root root) { roots.push_back(root); });
    if (!llvm::any_of(llvm::reverse(roots), [&](Root root) {
          if constexpr (std::is_same_v<Root, triton::StoreOp>)
            return rewrite(ExpressionPlan(root));
          else
            return rewrite(ExpressionPlan(root)) ||
                   rewrite(ExpressionPlan(root, /*indexed=*/true));
        }))
      return;
  }
}

LogicalResult
LayoutOptimizationAnalysis::rematerialize(bool preserveSharedReductions) {
  bool changed;
  do {
    changed = rewriteConversions(
        preserveSharedReductions,
        {&LayoutOptimizationAnalysis::backwardRematerialization});
    if ((changed || preserveSharedReductions) && failed(cleanup()))
      return failure();
  } while (changed);
  return success();
}

LogicalResult LayoutOptimizationAnalysis::run(bool disableSplitting) {
  disableRematSplitting = disableSplitting;
  costModel = LayoutCostModel();
  LayoutMemoryProfile memory = getLayoutMemoryProfile(funcOp);
  bool hasProtectedStore = !memory.protectedLoops.empty() &&
                           memory.has(LayoutConstraintGraph::HardwareStore);
  bool hasPackedMemoryAssembly = memory.hasPackedAssemblyProtocol();

  if (!memory.protectedLoops.empty() ||
      (memory.features & (LayoutConstraintGraph::PackedAssembly |
                          LayoutConstraintGraph::ProtectedReduction |
                          LayoutConstraintGraph::PermutingReshape |
                          LayoutConstraintGraph::DescriptorLoad)) ||
      hasPackedMemoryAssembly || memory.hasPairwiseReductionProtocol()) {
    assignLayouts(/*optimize=*/false);
    if ((hasProtectedStore || hasPackedMemoryAssembly ||
         memory.has(LayoutConstraintGraph::PermutingReshape)) &&
        failed(cleanup()))
      return failure();

    if ((hasProtectedStore || hasPackedMemoryAssembly) &&
        failed(rematerialize(/*preserveSharedReductions=*/false)))
      return failure();
    memory = getLayoutMemoryProfile(funcOp);
  }

  assignLayouts(/*optimize=*/true, &memory);
  if (failed(cleanup()))
    return failure();

  if (failed(rematerialize(/*preserveSharedReductions=*/true)))
    return failure();

  constexpr ConversionRewrite hoisting[] = {
      &LayoutOptimizationAnalysis::hoistConvertOnTopOfExtOrBroadcast,
      &LayoutOptimizationAnalysis::hoistConvertIntoConditionals,
      &LayoutOptimizationAnalysis::hoistConvertDotOperand};
  rewriteConversions(/*preserveSharedReductions=*/false,
                     ArrayRef(hoisting).take_front(disableSplitting ? 1 : 3));

  runDeadIterArgElimination(funcOp);
  if (failed(cleanup()))
    return failure();

  RewritePatternSet controlFlow(funcOp.getContext());
  scf::ForOp::getCanonicalizationPatterns(controlFlow, funcOp.getContext());
  scf::IfOp::getCanonicalizationPatterns(controlFlow, funcOp.getContext());
  if (failed(applyPatternsGreedily(funcOp, std::move(controlFlow))))
    LLVM_DEBUG(DBGS() << "scf cleanup did not converge\n");

  shareDominatingConversions();
  optimizeExpressions<triton::StoreOp>();
  optimizeExpressions<ConvertLayoutOp>();
  return cleanup();
}

LogicalResult LayoutOptimizationAnalysis::optimize(ModuleOp module,
                                                   bool disableSplitting) {
  SmallVector<LayoutOptimizationAnalysis, 1> analyses;
  module.walk([&](FuncOp function) { analyses.emplace_back(function); });
  for (unsigned iteration = 0; iteration < 2; ++iteration) {
    unsigned originalConversions = 0, remainingConversions = 0;
    module.walk([&](ConvertLayoutOp) { ++originalConversions; });
    for (LayoutOptimizationAnalysis &analysis : analyses)
      if (failed(analysis.run(disableSplitting)))
        return failure();
    module.walk([&](ConvertLayoutOp) { ++remainingConversions; });
    if (!remainingConversions || remainingConversions >= originalConversions)
      break;
  }
  return success();
}

} // namespace

class TritonGPUOptimizeLayoutsPass
    : public impl::TritonGPUOptimizeLayoutsBase<TritonGPUOptimizeLayoutsPass> {
public:
  using impl::TritonGPUOptimizeLayoutsBase<
      TritonGPUOptimizeLayoutsPass>::TritonGPUOptimizeLayoutsBase;

  void runOnOperation() override {
    if (failed(LayoutOptimizationAnalysis::optimize(getOperation(),
                                                    disableRematSplitting)))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu
