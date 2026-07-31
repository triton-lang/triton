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
#include <limits>

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUOPTIMIZELAYOUTS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tritongpu-optimize-layouts"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using LayoutAssignment = DenseMap<Value, Attribute>;
using LayoutValue = std::pair<Value, Attribute>;

struct LayoutSlice {
  SetVector<Value> values;
  LayoutAssignment encodings;
  DenseMap<LayoutValue, Value> rematerializations;
};

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
  uint64_t getProducerCost(Operation *producer, Value value,
                           Attribute encoding) const;
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

class LayoutOptimizationAnalysis;
class LayoutOptimizationPolicy;

/// A physical tensor value is not identified by SSA value alone: multiple
/// encodings can coexist at different dominance-safe insertion points. This
/// graph solves those alternatives independently of the rewrite policy.
class LayoutMaterializationGraph {
public:
  enum class Action : uint8_t {
    BackwardSlice,
    Narrowing,
    Conditional,
    DotOperand,
    DominatingReuse,
    StoreExpression,
    ConversionExpression,
  };

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

  enum class Kind : uint8_t {
    Existing,
    Conversion,
    Rematerialization,
    Expression,
  };

  explicit LayoutMaterializationGraph(LayoutOptimizationAnalysis &problem)
      : problem(problem) {}

  void build();
  void solveAssignments(const LayoutOptimizationPolicy &policy,
                        LayoutAssignment &assignments);
  unsigned size() const { return constraints.size(); }
  bool has(unsigned feature) const { return features & feature; }
  unsigned tensorLoads() const { return tensorLoadBoundaryCount; }

  void clear();
  void analyze();
  Value materialize(Value value, Attribute encoding, Operation *placement);
  LogicalResult materialize(ArrayRef<Action> actions,
                            bool preserveSharedReductions = false);

private:
  using Key = std::pair<std::pair<LayoutValue, unsigned>, Operation *>;

  struct Alternative {
    Kind kind;
    SmallVector<unsigned, 4> operands;
    uint64_t cost = 0;
    Value existing;
  };

  struct Node {
    Value value;
    ArrayRef<Attribute> candidates;
    bool fixed;
  };

  struct Materialization {
    Value value;
    Attribute encoding;
    Operation *placement;
    unsigned depth = 0;
    SmallVector<Alternative, 4> alternatives;
    uint64_t cost = std::numeric_limits<uint64_t>::max();
    unsigned choice = 0;
    Value materialized;
  };

  struct Expression {
    Operation *root;
    RankedTensorType target;
    bool indexed;
    Attribute flatEncoding;
    Value indices;
    SetVector<Operation *> operations;
    DenseMap<unsigned, Value> joinConditions{};
  };

  void remember(Value value, Attribute encoding, Value materialized);
  void updateExisting(ArrayRef<std::tuple<Value, Value>> replacements);
  bool reuse(ConvertLayoutOp conversion);
  Value materialize(OpBuilder &builder, Location location, Value value,
                    Attribute encoding);
  LogicalResult plan(OpOperand &root, Attribute encoding, LayoutSlice &slice,
                     std::function<bool(Operation *)> stop = nullptr,
                     bool requireRematerializable = true);
  void materialize(ConvertLayoutOp conversion, LayoutSlice &slice,
                   IRMapping *mapping = nullptr);
  bool materialize(Operation *root, bool indexed);
  bool materialize(Operation *root, Action action,
                   bool preserveSharedReductions);
  unsigned request(Value value, Attribute encoding, Operation *placement,
                   const LayoutAssignment &assignments, unsigned depth = 0);
  LogicalResult requestExpression(Value value, Attribute encoding,
                                  unsigned depth);
  Value expressionConstant(OpBuilder &builder, Location location,
                           int32_t value);
  Value materializeExpression(Materialization &node,
                              const Alternative &alternative,
                              OpBuilder &builder);
  bool feedsMemoryProtocol(Value value) const;
  bool hoist(ConvertLayoutOp conversion, Action action);
  void solve();
  Value materialize(unsigned node, OpBuilder *builder = nullptr);

  template <typename Range> void addContract(Range &&values) {
    std::optional<unsigned> previous;
    for (Value value : values) {
      auto found = indices.find({{{value, Attribute{}}, 0}, nullptr});
      if (found == indices.end())
        continue;
      unsigned current = found->second;
      if (constraints[current].fixed)
        continue;
      if (previous)
        components.join(*previous, current);
      previous = current;
    }
  }

  LayoutOptimizationAnalysis &problem;
  SmallVector<Node, 32> constraints;
  mutable llvm::IntEqClasses components;
  unsigned features = 0;
  unsigned tensorLoadBoundaryCount = 0;
  DenseMap<Key, unsigned> indices;
  DenseMap<LayoutValue, SmallVector<unsigned, 4>> candidates;
  DenseMap<Value, DenseMap<Attribute, Value>> existing;
  DenseMap<LayoutValue, SmallVector<ConvertLayoutOp>> sharedConversions;
  SmallVector<Materialization, 32> nodes;
  std::optional<Expression> expression;
};

/// One interchangeable policy controls assignment search, hardware contracts,
/// and physical materialization priorities; neither graph owns heuristics.
class LayoutOptimizationPolicy {
  using G = LayoutMaterializationGraph;

public:
  LayoutOptimizationPolicy() = default;
  explicit LayoutOptimizationPolicy(const LayoutMaterializationGraph &graph) {
    bool memoryFanout = graph.tensorLoads() >= 2 &&
                        !graph.has(G::Join | G::Split | G::Reduction | G::Loop |
                                   G::TensorMemory);
    fullObjective =
        (graph.has(G::Join | G::Split | G::Reduction) || memoryFanout) &&
        graph.size() <= 256;
    componentSearch =
        graph.has(G::Join | G::Split | G::Reduction | G::Loop) || memoryFanout;
    exactComponentSearch = fullObjective && graph.tensorLoads() > 0 &&
                           !graph.has(G::Store) && graph.size() <= 64;
    pruneRedundantReductionProposals =
        graph.has(G::Reduction) && !graph.has(G::ProtectedReduction);
    maxComponentProposals = fullObjective ? 128 : 32;
  }

  bool allowProducer(Value value, Operation *producer, Operation *placement,
                     bool fixed, bool memoryProtocol, unsigned depth,
                     unsigned states) const;
  bool allowAlternative(LayoutMaterializationGraph::Kind kind,
                        Operation *placement) const;

  bool fullObjective = false;
  bool componentSearch = false;
  bool exactComponentSearch = false;
  bool pruneRedundantReductionProposals = false;
  unsigned maxComponentProposals = 32;
  static constexpr unsigned maxExactComponentStates = 256;
};

struct LayoutMemoryProfile;

/// One function-local analysis owns layout assignment, dominance-correct
/// conversion materialization, scalar expressions, and bounded feedback.
class LayoutOptimizationAnalysis {
public:
  explicit LayoutOptimizationAnalysis(FuncOp function)
      : funcOp(function), graph(*this) {}
  LogicalResult run(bool disableSplitting);

private:
  using LayoutInfo = llvm::SmallSetVector<Attribute, 8>;
  friend class LayoutMaterializationGraph;
  using Action = LayoutMaterializationGraph::Action;

  LogicalResult cleanup();
  void assignLayouts(const LayoutMemoryProfile *profile = nullptr);
  // Find the anchor ops and set their layout in the data structure.
  void initAnchorLayout();
  // Recursively Propagate the layout to all the users of the anchor ops until
  // we reach a fix point.
  void propagateLayout();
  bool addEncoding(Value value, Attribute encoding);
  // Resolve cases where a value has multiple layouts associated to it.
  void resolveConflicts();
  void resolveGlobalConflicts();
  void rewrite();
  // Rewrite an op based on the layout picked by the analysis.
  void rewriteOp(Operation *op);
  void rewriteControlFlow(Operation *operation);
  void rewriteControlFlowOperands(Operation *operation, ValueRange targets,
                                  unsigned firstOperand = 0);
  void rewriteReduceToScalar(Operation *reduceOp);
  void setEncodingInPlace(Value value, Attribute encoding);
  void rewriteGenericOpInPlace(Operation *op, Attribute encoding);

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
  uint64_t getAffectedAssignmentCost(ArrayRef<Value> changed,
                                     const LayoutAssignment &assignments) const;
  bool buildGlobalComponentProposal(
      Value seed, Attribute encoding, LayoutAssignment &assignments,
      SmallVectorImpl<LayoutValue> &changes) const;

  // map from value to layout information.
  llvm::MapVector<Value, LayoutInfo> layouts;
  DenseSet<Value> fixedLayouts;
  // original encodings of tensor values rewritten in place.
  LayoutAssignment originalEncodings;
  mutable DenseMap<std::pair<std::pair<Operation *, Attribute>, unsigned>,
                   Attribute>
      inferredEncodings;
  LayoutCostModel costModel;
  LayoutOptimizationPolicy policy;
  FuncOp funcOp;
  bool disableRematSplitting = false;
  const LayoutMemoryProfile *memoryProfile = nullptr;
  LayoutMaterializationGraph graph;
  DominanceInfo domInfo;
  PostDominanceInfo postDomInfo;
};

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
  using G = LayoutMaterializationGraph;
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
      (LayoutMaterializationGraph::PackedAssembly |
       LayoutMaterializationGraph::ProtectedReduction))
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

struct LayoutMemoryProfile {
  using G = LayoutMaterializationGraph;
  unsigned loads = 0, tensorLoads = 0, stores = 0, reshapes = 0;
  unsigned features = 0;
  SmallVector<Operation *, 4> protectedLoops;

  explicit LayoutMemoryProfile(FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      features |= getLayoutOperationFeatures(op);
      if (auto load = dyn_cast<LoadOp>(op)) {
        ++loads;
        if (isa<RankedTensorType>(load.getType()))
          ++tensorLoads;
      } else if (isa<StoreOp>(op)) {
        ++stores;
      } else if (isa<ReshapeOp>(op)) {
        ++reshapes;
      } else if (isa<scf::ForOp, scf::WhileOp>(op)) {
        protectedLoops.push_back(op);
      }
    });
    llvm::erase_if(protectedLoops, [&](Operation *op) {
      return !isProtectedLayoutLoop(op, loads, stores);
    });
  }

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

void LayoutMaterializationGraph::build() {
  clear();
  constraints.clear();
  components.clear();
  features = problem.memoryProfile->features;
  tensorLoadBoundaryCount = problem.memoryProfile->tensorLoads;
  for (const auto &[value, layouts] : problem.layouts) {
    indices.try_emplace({{{value, Attribute{}}, 0}, nullptr},
                        constraints.size());
    constraints.push_back(
        {value, layouts.getArrayRef(), problem.fixedLayouts.contains(value)});
    components.grow(constraints.size());
  }

  problem.funcOp.walk([&](Operation *operation) {
    if (isFixedLayoutBoundary(operation))
      return;
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(operation)) {
      for (unsigned index = 0; index < operation->getNumResults(); ++index)
        addContract(getTiedArgs(operation, index));
    } else if (isLayoutTransform(operation)) {
      addContract(llvm::concat<Value>(operation->getOperands(),
                                      operation->getResults()));
    }
  });
}

void LayoutOptimizationAnalysis::assignLayouts(
    const LayoutMemoryProfile *profile) {
  layouts.clear();
  fixedLayouts.clear();
  originalEncodings.clear();
  inferredEncodings.clear();
  memoryProfile = profile;
  initAnchorLayout();
  propagateLayout();
  resolveConflicts();
  if (memoryProfile)
    graph.analyze();
  else
    graph.clear();
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

  if (memoryProfile) {
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

    if (!memoryProfile || !isFixedLayoutBoundary(op))
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
  if (memoryProfile) {
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
    if (memoryProfile)
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
  if (memoryProfile)
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

uint64_t LayoutCostModel::getProducerCost(Operation *producer, Value value,
                                          Attribute encoding) const {
  if (isa<arith::ConstantOp, SplatOp, MakeRangeOp>(producer))
    return 0;
  uint64_t arithmetic =
      (isExpensiveMathOp(producer) ? 8 : 1) * getByteCount(value);
  return (arithmetic + getRegisterPressureCost(value, encoding)) *
         getExecutionWeight(producer);
}

template <bool source>
Attribute
LayoutOptimizationAnalysis::getCachedEncoding(Operation *op,
                                              Attribute encoding) const {
  auto [cached, inserted] = inferredEncodings.try_emplace(
      std::pair<std::pair<Operation *, Attribute>, unsigned>{{op, encoding},
                                                             source},
      Attribute{});
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

void LayoutMaterializationGraph::solveAssignments(
    const LayoutOptimizationPolicy &policy, LayoutAssignment &assignments) {
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
      for (const Node &node : constraints) {
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
    for (const Node &node : nodes) {
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
    bool changed = improve(llvm::reverse(constraints));
    changed |= improve(constraints);
    if (!changed)
      break;
  }

  // Keep convergence local to the constrained values; original encodings are
  // verifier-proven and remain the safe incumbent for any invalid assignment.
  for (unsigned iteration = 0; iteration < maxAssignmentIterations;
       ++iteration) {
    bool changed = false;
    for (const Node &node : constraints) {
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

  auto globalCost = [&] {
    uint64_t cost = 0;
    for (const Node &node : constraints)
      if (Attribute encoding = assignments.lookup(node.value))
        cost += problem.getAssignmentCost(node.value, encoding, assignments);
    return cost;
  };

  components.compress();
  SmallVector<SmallVector<unsigned, 8>, 8> connected(
      components.getNumClasses());
  for (auto [index, node] : llvm::enumerate(constraints))
    if (!node.fixed && node.candidates.size() > 1)
      connected[components[index]].push_back(index);

  // Coordinate descent cannot change jointly constrained encodings when every
  // intermediate assignment is illegal. Search the bounded components exactly.
  for (const SmallVector<unsigned, 8> &component : connected) {
    if (component.size() < 2)
      continue;
    unsigned states = 1;
    for (unsigned index : component)
      if ((states *= constraints[index].candidates.size()) >
          policy.maxExactComponentStates)
        break;
    if (states > policy.maxExactComponentStates)
      continue;

    uint64_t bestCost = globalCost();
    SmallVector<Attribute, 8> bestAssignments;
    for (unsigned index : component)
      bestAssignments.push_back(assignments.lookup(constraints[index].value));

    auto search = [&](auto &&self, unsigned position) -> void {
      if (position == component.size()) {
        for (const Node &node : constraints)
          if (!problem.canAssignEncoding(
                  node.value, assignments.lookup(node.value), assignments))
            return;
        uint64_t candidateCost = globalCost();
        if (candidateCost >= bestCost)
          return;
        bestCost = candidateCost;
        for (auto [index, value] : llvm::enumerate(component))
          bestAssignments[index] = assignments.lookup(constraints[value].value);
        return;
      }

      const Node &node = constraints[component[position]];
      Attribute original = assignments.lookup(node.value);
      for (Attribute candidate : node.candidates) {
        assignments[node.value] = candidate;
        self(self, position + 1);
      }
      assignments[node.value] = original;
    };
    search(search, 0);
    for (auto [index, value] : llvm::enumerate(component))
      assignments[constraints[value].value] = bestAssignments[index];
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
    graph.build();
    policy = LayoutOptimizationPolicy(graph);
    LDBG("resolving " << layouts.size() << " layout values with "
                      << (policy.fullObjective ? "the full" : "the bounded")
                      << " global objective");

    graph.solveAssignments(policy, assignments);
  }

  for (auto &[value, info] : layouts) {
    Attribute encoding = assignments.lookup(value);
    info.clear();
    info.insert(encoding);
  }
}

void LayoutOptimizationAnalysis::rewrite() {
  SmallVector<Region *, 8> regions{&funcOp->getRegion(0)};
  for (unsigned index = 0; index < regions.size(); ++index) {
    for (Operation &op : regions[index]->getOps()) {
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
        Value operand = assertOp->getOperand(0);
        if (auto it = layouts.find(operand); it != layouts.end())
          assertOp->setOperand(
              0, graph.materialize(operand, it->second[0], assertOp));
      } else {
        // If we don't need to rewrite the op we still need to remap the
        // operands.
        for (OpOperand &operand : op.getOpOperands()) {
          auto it = layouts.find(operand.get());
          if (it == layouts.end())
            continue;
          Attribute encoding = originalEncodings.lookup(operand.get());
          if (!encoding)
            encoding =
                cast<RankedTensorType>(operand.get().getType()).getEncoding();
          Value newOperand = graph.materialize(operand.get(), encoding, &op);
          op.setOperand(operand.getOperandNumber(), newOperand);
        }
      }
      for (Region &nested : op.getRegions())
        regions.push_back(&nested);
    }
  }
}

Value LayoutMaterializationGraph::materialize(Value value, Attribute encoding,
                                              Operation *placement) {
  if (auto tensorType = dyn_cast<RankedTensorType>(value.getType())) {
    if (tensorType.getEncoding() == encoding)
      return value;
    if (problem.memoryProfile && placement)
      if (auto found = indices.find({{{value, encoding}, 0}, placement});
          found != indices.end())
        return materialize(found->second);
    OpBuilder rewriter(value.getContext());
    rewriter.setInsertionPointAfterValue(value);
    return materialize(rewriter, value.getLoc(), value, encoding);
  }
  return value;
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
                   graph.materialize(operand.get(), operandEnc, op));
  }
  for (Value result : op->getResults())
    if (isa<RankedTensorType>(result.getType()))
      setEncodingInPlace(result, encoding);
}

void LayoutOptimizationAnalysis::rewriteControlFlow(Operation *operation) {
  if (auto loop = dyn_cast<scf::ForOp>(operation)) {
    rewriteAssignedValues(loop.getResults(), [&](unsigned index, Value result,
                                                 Attribute encoding) {
      loop.getInitArgsMutable()[index].assign(
          graph.materialize(loop.getInitArgs()[index], encoding, loop));
      setEncodingInPlace(result, encoding);
      setEncodingInPlace(loop.getRegionIterArg(index), encoding);
    });
    return;
  }

  if (auto loop = dyn_cast<scf::WhileOp>(operation)) {
    rewriteAssignedValues(
        loop.getBeforeArguments(),
        [&](unsigned index, Value argument, Attribute encoding) {
          loop->setOperand(index, graph.materialize(loop->getOperand(index),
                                                    encoding, loop));
          setEncodingInPlace(argument, encoding);
        });
    rewriteAssignedValues(loop.getResults(), [&](unsigned index, Value result,
                                                 Attribute encoding) {
      setEncodingInPlace(result, encoding);
      setEncodingInPlace(loop.getAfterArguments()[index], encoding);
    });
    return;
  }

  rewriteAssignedValues(operation->getResults(),
                        [&](unsigned, Value result, Attribute encoding) {
                          setEncodingInPlace(result, encoding);
                        });
}

void LayoutOptimizationAnalysis::rewriteControlFlowOperands(
    Operation *operation, ValueRange targets, unsigned firstOperand) {
  for (auto [index, target] : llvm::enumerate(targets))
    if (auto type = dyn_cast<RankedTensorType>(target.getType())) {
      unsigned operand = firstOperand + index;
      operation->setOperand(operand,
                            graph.materialize(operation->getOperand(operand),
                                              type.getEncoding(), operation));
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
    Value newOperand = graph.materialize(operand.get(), srcEncoding, reduceOp);
    reduceOp->setOperand(operand.getOperandNumber(), newOperand);
  }
}

void LayoutOptimizationAnalysis::rewriteOp(Operation *op) {
  if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(op)) {
    rewriteControlFlow(op);
  } else {
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

bool LayoutOptimizationPolicy::allowProducer(Value value, Operation *producer,
                                             Operation *placement, bool fixed,
                                             bool memoryProtocol,
                                             unsigned depth,
                                             unsigned states) const {
  return depth < 32 && states < 4096 && producer && placement &&
         producer->getBlock() == placement->getBlock() &&
         producer->getNumResults() == 1 && producer->getNumRegions() == 0 &&
         isMemoryEffectFree(producer) && canBeRemat(producer) &&
         !isFixedLayoutBoundary(producer) && !fixed && value.hasOneUse() &&
         !memoryProtocol &&
         !isa<ConvertLayoutOp, ReduceOp, SplitOp, JoinOp>(producer);
}

bool LayoutOptimizationPolicy::allowAlternative(
    LayoutMaterializationGraph::Kind kind, Operation *placement) const {
  return kind != LayoutMaterializationGraph::Kind::Rematerialization ||
         isa<ConvertLayoutOp>(placement);
}

void LayoutMaterializationGraph::clear() {
  indices.clear();
  candidates.clear();
  nodes.clear();
  existing.clear();
  sharedConversions.clear();
  expression.reset();
}

Value LayoutMaterializationGraph::materialize(OpBuilder &builder,
                                              Location location, Value value,
                                              Attribute encoding) {
  auto type = cast<RankedTensorType>(value.getType());
  return ConvertLayoutOp::create(builder, location,
                                 type.cloneWithEncoding(encoding), value)
      .getResult();
}

void LayoutMaterializationGraph::remember(Value value, Attribute encoding,
                                          Value materialized) {
  LDBG("materialized " << value << " encoding " << encoding << " "
                       << materialized);
  existing[value][encoding] = materialized;
}

bool LayoutMaterializationGraph::reuse(ConvertLayoutOp conversion) {
  LayoutValue key{conversion.getSrc(), conversion.getType().getEncoding()};
  auto &available = sharedConversions[key];
  auto candidate = llvm::find_if(available, [&](ConvertLayoutOp existing) {
    return problem.domInfo.properlyDominates(existing.getResult(),
                                             conversion.getOperation());
  });
  if (candidate == available.end()) {
    available.push_back(conversion);
    return false;
  }
  conversion.getResult().replaceAllUsesWith(candidate->getResult());
  conversion.erase();
  return true;
}

void LayoutMaterializationGraph::analyze() {
  clear();
  problem.domInfo.invalidate();

  LayoutAssignment assignments;
  for (const auto &[value, layouts] : problem.layouts)
    if (!layouts.empty())
      assignments.try_emplace(value, layouts[0]);

  problem.funcOp.walk([&](Operation *placement) {
    for (OpOperand &use : placement->getOpOperands())
      if (isa<RankedTensorType>(use.get().getType()))
        for (Attribute encoding : problem.getUseEncodings(use, assignments))
          request(use.get(), encoding, placement, assignments);
  });
  solve();
}

unsigned LayoutMaterializationGraph::request(
    Value value, Attribute encoding, Operation *placement,
    const LayoutAssignment &assignments, unsigned depth) {
  Key key{{{value, encoding}, 0}, placement};
  auto [entry, inserted] = indices.try_emplace(key, nodes.size());
  if (!inserted)
    return entry->second;

  unsigned index = entry->second;
  nodes.push_back({value, encoding, placement});
  Attribute source = problem.getAssignedEncoding(value, assignments);
  if (!source || source == encoding) {
    nodes[index].alternatives.push_back({Kind::Existing, {}, 0, value});
    return index;
  }

  unsigned original = request(value, source, placement, assignments, depth + 1);
  nodes[index].alternatives.push_back(
      {Kind::Conversion,
       {original},
       problem.costModel.getTransitionCost(value, source, encoding),
       {}});

  auto &available = candidates[{value, encoding}];
  for (unsigned candidate : available) {
    Operation *previous = nodes[candidate].placement;
    if (previous && placement &&
        problem.domInfo.properlyDominates(previous, placement)) {
      nodes[index].alternatives.push_back({Kind::Existing, {candidate}, 0, {}});
      break;
    }
  }
  available.push_back(index);

  for (OpOperand &use : value.getUses()) {
    auto existing = dyn_cast<ConvertLayoutOp>(use.getOwner());
    if (existing && existing.getType().getEncoding() == encoding && placement &&
        problem.domInfo.properlyDominates(existing.getResult(), placement)) {
      nodes[index].alternatives.push_back(
          {Kind::Existing, {}, 0, existing.getResult()});
      break;
    }
  }

  Operation *producer = value.getDefiningOp();
  bool memoryProtocol =
      placement && isa<ConvertLayoutOp>(placement) &&
      feedsMemoryProtocol(cast<ConvertLayoutOp>(placement).getResult());
  if (!problem.policy.allowProducer(value, producer, placement,
                                    problem.fixedLayouts.contains(value),
                                    memoryProtocol, depth, nodes.size()))
    return index;

  Attribute operandEncoding =
      problem.getCachedEncoding<true>(producer, encoding);
  SmallVector<unsigned, 4> operands;
  for (Value operand : producer->getOperands()) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    if (!operandEncoding)
      return index;
    operands.push_back(
        request(operand, operandEncoding, placement, assignments, depth + 1));
  }

  uint64_t cloneCost =
      problem.costModel.getProducerCost(producer, value, encoding);
  nodes[index].alternatives.push_back(
      {Kind::Rematerialization, std::move(operands), cloneCost, {}});
  return index;
}

void LayoutMaterializationGraph::solve() {
  constexpr uint64_t infinite = std::numeric_limits<uint64_t>::max();
  SmallVector<uint8_t, 32> state(nodes.size(), 0);
  auto evaluate = [&](auto &&self, unsigned index) -> uint64_t {
    Materialization &node = nodes[index];
    if (state[index] == 2)
      return node.cost;
    if (state[index] == 1)
      return infinite;
    state[index] = 1;

    for (auto [choice, alternative] : llvm::enumerate(node.alternatives)) {
      if (!problem.policy.allowAlternative(alternative.kind, node.placement))
        continue;
      uint64_t candidate = alternative.cost;
      for (unsigned operand : alternative.operands) {
        uint64_t cost = self(self, operand);
        if (cost == infinite || candidate > infinite - cost) {
          candidate = infinite;
          break;
        }
        candidate += cost;
      }
      if (candidate < node.cost) {
        node.cost = candidate;
        node.choice = choice;
      }
    }

    state[index] = 2;
    return node.cost;
  };

  for (unsigned index = 0; index < nodes.size(); ++index)
    evaluate(evaluate, index);
}

bool LayoutMaterializationGraph::feedsMemoryProtocol(Value value) const {
  DenseSet<Value> visited;
  SmallVector<Value, 16> worklist{value};
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (isa<StoreOp>(user)) {
        if (use.getOperandNumber() != 1)
          return true;
        continue;
      }
      if (isa<LoadOp, AtomicRMWOp, AtomicCASOp, DescriptorLoadLikeOpInterface,
              DescriptorStoreLikeOpInterface>(user))
        return true;
      if (isMemoryEffectFree(user))
        for (Value result : user->getResults())
          if (isa<RankedTensorType>(result.getType()))
            worklist.push_back(result);
    }
  }
  return false;
}

Value LayoutMaterializationGraph::materialize(unsigned index,
                                              OpBuilder *expressionBuilder) {
  Materialization &node = nodes[index];
  if (node.materialized)
    return node.materialized;

  Alternative &alternative = node.alternatives[node.choice];
  if (alternative.kind == Kind::Existing)
    return node.materialized = alternative.existing
                                   ? alternative.existing
                                   : materialize(alternative.operands.front());

  if (alternative.kind == Kind::Expression) {
    assert(expressionBuilder && "an expression requires its root builder");
    return node.materialized =
               materializeExpression(node, alternative, *expressionBuilder);
  }

  if (alternative.kind == Kind::Conversion) {
    Value source = materialize(alternative.operands.front());
    auto type = cast<RankedTensorType>(source.getType());
    if (type.getEncoding() == node.encoding)
      return node.materialized = source;
    OpBuilder builder(source.getContext());
    builder.setInsertionPointAfterValue(source);
    return node.materialized =
               materialize(builder, source.getLoc(), source, node.encoding);
  }

  Operation *producer = node.value.getDefiningOp();
  OpBuilder builder(node.placement);
  IRMapping mapping;
  auto operands = alternative.operands.begin();
  for (Value operand : producer->getOperands())
    if (isa<RankedTensorType>(operand.getType()))
      mapping.map(operand, materialize(*operands++));
  Operation *replacement = builder.clone(*producer, mapping);
  auto type = cast<RankedTensorType>(node.value.getType());
  replacement->getResult(0).setType(type.cloneWithEncoding(node.encoding));
  if (auto constant = dyn_cast<arith::ConstantOp>(replacement)) {
    auto elements = cast<DenseElementsAttr>(constant.getValue());
    constant.setValueAttr(
        elements.reshape(cast<RankedTensorType>(constant.getType())));
  }
  return node.materialized = replacement->getResult(0);
}

void LayoutMaterializationGraph::updateExisting(
    ArrayRef<std::tuple<Value, Value>> replacements) {
  for (auto [old, replacement] : replacements) {
    auto it = existing.find(old);
    if (it == existing.end())
      continue;
    auto previous = std::move(it->second);
    existing.erase(it);
    auto &updated = existing[replacement];
    for (auto [encoding, materialized] : previous) {
      for (auto [before, after] : replacements) {
        if (before == materialized) {
          materialized = after;
          break;
        }
      }
      updated[encoding] = materialized;
    }
  }
}

void LayoutMaterializationGraph::materialize(ConvertLayoutOp convertOp,
                                             LayoutSlice &state,
                                             IRMapping *providedMapping) {
  IRMapping local;
  IRMapping &mapping = providedMapping ? *providedMapping : local;
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
        for (scf::YieldOp yield : {ifOp.thenYield(), ifOp.elseYield()}) {
          opsToRewrite.insert(yield);
          yieldOperandsMap[yield].push_back(operandIdx);
        }
      }
    } else {
      BlockArgument blockArg = cast<BlockArgument>(v);
      auto loopOp =
          cast<LoopLikeOpInterface>(blockArg.getOwner()->getParentOp());
      opsToRewrite.insert(loopOp.getOperation());
      OpOperand *operand = loopOp.getTiedLoopYieldedValue(blockArg);
      auto yieldOp = blockArg.getOwner()->getTerminator();
      yieldOperandsMap[yieldOp].push_back(operand->getOperandNumber());
      opsToRewrite.insert(yieldOp);
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
        remember(newForOp.getResult(m.first), layout[oldArg],
                 newForOp.getResult(m.second));
        remember(oldArg, layout[oldArg],
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
          remember(result, layout[result], newIfOp.getResult(newIdx));
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
      Value cvt = materialize(builder, op->getLoc(), newOp->getResult(0),
                              layout[op->getResult(0)]);
      mapping.map(op->getResult(0), cvt);
      remember(op->getResult(0), layout[op->getResult(0)], cvt);
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
      remember(old, it->second, newV);
    }
  }
  // Add the rewritten convert to the replacements so it is removed from the
  // remat maps and has its uses replaced like the other ops we delete.
  replacements.emplace_back(convertOp.getResult(),
                            mapping.lookup(convertOp.getSrc()));

  updateExisting(replacements);
  for (auto [old, replacement] : replacements)
    builder.replaceAllUsesWith(old, replacement);

  convertOp->erase();
  for (Operation *op : deadOps)
    op->erase();
}

LogicalResult LayoutMaterializationGraph::plan(
    OpOperand &root, Attribute rootEncoding, LayoutSlice &state,
    std::function<bool(Operation *)> stopPropagation,
    bool requireRematerializable) {
  LayoutSlice candidate = state;
  auto &[slice, layout, existingRemats] = candidate;
  // Allow re-using existing conversions for a value if it dominates the use.
  auto getExistingConversion = [&](OpOperand &value, Attribute encoding) {
    Value remat = existing.lookup(value.get()).lookup(encoding);
    if (!remat)
      return Value();
    // `value` can be replaced with an existing rematerialization if it
    // dominates the current use of value.
    Operation *user = value.getOwner();
    if (problem.domInfo.properlyDominates(remat, user)) {
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
    bool disableRematSplitting, bool preserveSharedReductions) const {
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

    if (preserveSharedReductions && isOpUsedOutsideSlice && isa<ReduceOp>(op)) {
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

bool LayoutMaterializationGraph::hoist(ConvertLayoutOp convertOp,
                                       Action action) {
  auto noDataMovement = [](Operation *op) {
    return (op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)) ||
           isa<BroadcastOp, Fp4ToFpOp, ConvertLayoutOp, UpcastFpOpInterface>(
               op) ||
           isView(op);
  };
  auto isNarrowing = [](Operation *op) {
    if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp, BroadcastOp,
            ExpandDimsOp>(op))
      return true;
    if (auto castOp = dyn_cast<FpToFpOp>(op))
      return getElementBitWidth(
                 cast<RankedTensorType>(castOp.getOperand().getType())) <
             getElementBitWidth(cast<RankedTensorType>(castOp.getType()));
    return false;
  };

  Attribute target = convertOp.getType().getEncoding();
  if (action == Action::Narrowing && isa<DotOperandEncodingAttr>(target))
    return false;
  if (action == Action::DotOperand) {
    Operation *parent = convertOp->getParentOp();
    if (!parent ||
        !parent
             ->walk([&](Operation *operation) {
               if (isa<triton::DotOpInterface>(operation))
                 if (auto type = dyn_cast<RankedTensorType>(
                         operation->getOperand(0).getType()))
                   if (auto layout =
                           dyn_cast<DotOperandEncodingAttr>(type.getEncoding()))
                     if (isa<MmaEncodingTrait>(layout.getParent()) &&
                         problem.postDomInfo.postDominates(operation,
                                                           convertOp))
                       return WalkResult::interrupt();
               return WalkResult::advance();
             })
             .wasInterrupted())
      return false;
  }

  auto stop = [&](Operation *op) {
    if (action == Action::Narrowing)
      return isNarrowing(op);
    if (action == Action::Conditional)
      return isa<scf::IfOp>(op);
    return !noDataMovement(op);
  };

  LayoutSlice state;
  auto &[slice, layout, existingRemats] = state;
  if (failed(plan(convertOp.getSrcMutable(), target, state, stop,
                  /*requireRematerializable=*/action != Action::DotOperand)))
    return false;

  IRMapping mapping;
  if (action == Action::DotOperand) {
    OpBuilder builder(convertOp.getContext());
    SetVector<Value> innerSlice;
    for (Value v : slice) {
      Operation *producer = v.getDefiningOp();
      if (!producer) {
        LLVM_DEBUG({
          DBGS() << "  Block arguments not supported. Got " << v << "\n";
        });
        return false;
      }

      // We expect the leaves of the slice to be Load, descriptor load-like ops,
      // or arith::Constant. This could be generalised if necessary.
      if (!isa<LoadOp, DescriptorLoadLikeOpInterface>(producer)) {
        if (!isa<arith::ConstantOp>(producer) && !noDataMovement(producer)) {
          LLVM_DEBUG({
            DBGS() << "  Leaves must be Load, descriptor load-like ops, or "
                      "Constant. Got "
                   << v << "\n";
          });
          return false;
        }
        innerSlice.insert(v);
        continue;
      }
      builder.setInsertionPointAfter(producer);
      if (!isa<RankedTensorType>(producer->getResult(0).getType()))
        continue;
      mapping.map(producer->getResult(0),
                  materialize(builder, convertOp.getLoc(),
                              producer->getResult(0),
                              layout[producer->getResult(0)]));
    }

    if (innerSlice.empty())
      return false;

    LLVM_DEBUG({
      DBGS() << "  Hoisting " << convertOp << '\n';
      for (Value v : innerSlice)
        DBGS() << "    " << v << '\n';
    });

    state.values = std::move(innerSlice);
  } else if (action == Action::Narrowing) {
    Operation *extOrBroadcastOp = nullptr;
    unsigned sliceSize = slice.size();
    for (unsigned i = 0; i < sliceSize; i++) {
      Value v = slice[i];
      Operation *op = v.getDefiningOp();
      if (!op || !isNarrowing(op))
        continue;

      Attribute srcEncoding = inferSrcEncoding(op, layout[v]);
      if (!srcEncoding)
        return false;

      // If we can rematerialize the rest of the ext slice we can ignore this
      // ext as it won't need a convert.
      if (succeeded(plan(op->getOpOperand(0), srcEncoding, state)))
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
    int64_t newCvtCost = problem.costModel.getTransitionCost(
        operand, cast<RankedTensorType>(operand.getType()).getEncoding(),
        srcEncoding, /*rematerialization=*/true);
    if (!problem.costModel.isRematerializationBeneficial(
            convertOp, state, newCvtCost, problem.disableRematSplitting))
      return false;
    // Move the convert before the ext op and rewrite the slice.
    OpBuilder builder(extOrBroadcastOp);
    Value newConvertOp =
        materialize(builder, convertOp.getLoc(),
                    extOrBroadcastOp->getOperand(0), srcEncoding);
    Operation *newExtOrBroadcast = builder.clone(*extOrBroadcastOp);
    newExtOrBroadcast->setOperand(0, newConvertOp);
    newExtOrBroadcast->getResult(0).setType(
        cast<RankedTensorType>(extOrBroadcastOp->getResult(0).getType())
            .cloneWithEncoding(dstEncoding));
    mapping.map(extOrBroadcastOp->getResult(0),
                newExtOrBroadcast->getResult(0));
    slice.remove(extOrBroadcastOp->getResult(0));
  } else {
    // These are the conditional edges above which conversions should be
    // hoisted. The value represents the `scf.if` op result and the operand
    // represents the edge into one of the branches.
    SmallVector<std::pair<Value, OpOperand *>> hoistAbove;

    // The list of `scf.if` op results in the slice that are not
    // rematerializable. Hoisting is terminated at these values.
    SmallVector<OpResult> terminals;

    // This loop recurses through the subslices of the backwards dependencies,
    // so re-query the size of `slice`.
    for (unsigned i = 0; i != slice.size(); ++i) {
      Value v = slice[i];
      auto ifOp = v.getDefiningOp<scf::IfOp>();
      if (!ifOp)
        continue;

      Attribute rootLayout = layout.at(v);
      unsigned resIdx = cast<OpResult>(v).getResultNumber();

      // Take the backward slice along each branch.
      OpOperand *edges[] = {&ifOp.thenYield().getResultsMutable()[resIdx],
                            &ifOp.elseYield().getResultsMutable()[resIdx]};
      LayoutSlice candidate = state;
      bool first = succeeded(plan(*edges[0], rootLayout, candidate, stop));
      bool second = succeeded(plan(*edges[1], rootLayout, candidate, stop));
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
    auto hoistRemat = [&](OpBuilder &b, Value v, Attribute encoding) {
      Value newCvt = materialize(b, convertOp.getLoc(), v, encoding);
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
  }
  materialize(convertOp, state, &mapping);
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

static bool isExpressionLeaf(Operation *operation) {
  return isa<SplatOp, MakeRangeOp, arith::ConstantOp>(operation);
}

LogicalResult LayoutMaterializationGraph::requestExpression(Value value,
                                                            Attribute encoding,
                                                            unsigned depth) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !encoding ||
      (expression->indexed && (!type.hasStaticShape() || depth > 30 ||
                               (expression->target.getNumElements() >> depth) !=
                                   type.getNumElements())))
    return failure();

  Key key{{{value, encoding}, depth}, expression->root};
  if (indices.contains(key))
    return success();
  Operation *operation = value.getDefiningOp();
  if (nodes.size() >= 512 || !operation || operation->getNumResults() != 1 ||
      !isMemoryEffectFree(operation) ||
      (operation->getBlock() != expression->root->getBlock() &&
       (!expression->indexed || !isExpressionLeaf(operation))))
    return failure();
  expression->operations.insert(operation);

  Attribute operandEncoding;
  SmallVector<unsigned, 4> operands;
  bool joined = false, sameShape = false;
  auto addOperand = [&](Value operand, Attribute layout,
                        unsigned operandDepth) {
    if (failed(requestExpression(operand, layout, operandDepth)))
      return failure();
    operands.push_back(
        indices.find({{{operand, layout}, operandDepth}, expression->root})
            ->second);
    return success();
  };

  if (isa<ConvertLayoutOp>(operation) ||
      (expression->indexed && isa<ReshapeOp>(operation))) {
    if (auto reshape = dyn_cast<ReshapeOp>(operation);
        reshape && reshape.getAllowReorder())
      return failure();
    operandEncoding = encoding;
  } else if (expression->indexed && isa<JoinOp>(operation)) {
    auto join = cast<JoinOp>(operation);
    RankedTensorType resultType = join.getType();
    if (resultType.getShape().empty() || resultType.getShape().back() != 2 ||
        failed(addOperand(join.getLhs(), encoding, depth + 1)) ||
        failed(addOperand(join.getRhs(), encoding, depth + 1)))
      return failure();
    joined = true;
  } else if (auto splat = dyn_cast<SplatOp>(operation)) {
    if (expression->indexed && isa<RankedTensorType>(splat.getSrc().getType()))
      return failure();
  } else if (auto range = dyn_cast<MakeRangeOp>(operation)) {
    if (expression->indexed && !range.getType().getElementType().isInteger(32))
      return failure();
  } else if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
    auto elements = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (!elements || (expression->indexed && !elements.isSplat()))
      return failure();
  } else if (expression->indexed) {
    if (!operation->hasTrait<OpTrait::Elementwise>())
      return failure();
    operandEncoding = encoding;
    sameShape = true;
  } else if ((!isLayoutTransform(operation) && !isa<BroadcastOp>(operation)) ||
             isa<ReduceOp, SplitOp>(operation) ||
             !(operandEncoding = inferSrcEncoding(operation, encoding))) {
    return failure();
  }

  if (!joined)
    for (Value operand : operation->getOperands()) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType)
        continue;
      if (!operandEncoding ||
          (sameShape && operandType.getShape() != type.getShape()) ||
          failed(addOperand(operand, operandEncoding, depth)))
        return failure();
    }

  unsigned index = nodes.size();
  indices.try_emplace(key, index);
  nodes.push_back({value, encoding, expression->root, depth});
  nodes.back().alternatives.push_back(
      {Kind::Expression, std::move(operands), 0, {}});
  return success();
}

Value LayoutMaterializationGraph::expressionConstant(OpBuilder &builder,
                                                     Location location,
                                                     int32_t value) {
  auto type = cast<RankedTensorType>(expression->indices.getType());
  return arith::ConstantOp::create(
      builder, location, type,
      DenseElementsAttr::get(type, builder.getI32IntegerAttr(value)));
}

Value LayoutMaterializationGraph::materializeExpression(
    Materialization &node, const Alternative &alternative, OpBuilder &builder) {
  Operation *operation = node.value.getDefiningOp();
  if (isa<ConvertLayoutOp>(operation) ||
      (expression->indexed && isa<ReshapeOp>(operation)))
    return materialize(alternative.operands.front(), &builder);

  if (expression->indexed && isa<JoinOp>(operation)) {
    auto join = cast<JoinOp>(operation);
    Value lhs = materialize(alternative.operands[0], &builder);
    Value rhs = materialize(alternative.operands[1], &builder);
    auto [condition, inserted] =
        expression->joinConditions.try_emplace(node.depth, Value());
    if (inserted) {
      Value mask =
          expressionConstant(builder, join.getLoc(), int32_t{1} << node.depth);
      Value zero = expressionConstant(builder, join.getLoc(), 0);
      Value masked = arith::AndIOp::create(builder, join.getLoc(),
                                           expression->indices, mask);
      condition->second = arith::CmpIOp::create(
          builder, join.getLoc(), arith::CmpIPredicate::ne, masked, zero);
    }
    return arith::SelectOp::create(builder, join.getLoc(), condition->second,
                                   rhs, lhs);
  }

  if (expression->indexed && isa<MakeRangeOp>(operation)) {
    auto range = cast<MakeRangeOp>(operation);
    Value result = expression->indices;
    if (node.depth != 0)
      result = arith::ShRUIOp::create(
          builder, range.getLoc(), result,
          expressionConstant(builder, range.getLoc(), node.depth));
    if (int64_t start = range.getStartAttr().getInt(); start != 0)
      result = arith::AddIOp::create(
          builder, range.getLoc(), result,
          expressionConstant(builder, range.getLoc(), start));
    return result;
  }

  RankedTensorType oldType = cast<RankedTensorType>(node.value.getType());
  RankedTensorType newType =
      expression->indexed
          ? RankedTensorType::get(expression->target.getShape(),
                                  oldType.getElementType(), node.encoding)
          : oldType.cloneWithEncoding(node.encoding);
  if (expression->indexed && isa<SplatOp>(operation)) {
    auto splat = cast<SplatOp>(operation);
    return SplatOp::create(builder, splat.getLoc(), newType, splat.getSrc());
  }
  if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
    auto elements = cast<DenseElementsAttr>(constant.getValue());
    auto values = expression->indexed
                      ? DenseElementsAttr::get(
                            newType, elements.getSplatValue<Attribute>())
                      : elements.reshape(newType);
    return arith::ConstantOp::create(builder, constant.getLoc(), newType,
                                     values);
  }

  IRMapping mapping;
  auto next = alternative.operands.begin();
  for (Value operand : operation->getOperands())
    if (isa<RankedTensorType>(operand.getType()))
      mapping.map(operand, materialize(*next++, &builder));
  Operation *replacement = builder.clone(*operation, mapping);
  replacement->getResult(0).setType(newType);
  return replacement->getResult(0);
}

bool LayoutMaterializationGraph::materialize(Operation *root, bool indexed) {
  clear();
  Value value;
  RankedTensorType target;
  if (auto conversion = dyn_cast<ConvertLayoutOp>(root)) {
    value = conversion.getSrc();
    target = conversion.getType();
  } else {
    auto store = cast<StoreOp>(root);
    value = store.getValue();
    target = dyn_cast<RankedTensorType>(value.getType());
  }
  if (!target || !target.getEncoding())
    return false;
  expression.emplace(Expression{root, target, indexed});

  if (indexed) {
    if (!target.hasStaticShape())
      return false;
    int64_t size = target.getNumElements();
    if (size <= 0 || size > std::numeric_limits<int32_t>::max() ||
        !llvm::isPowerOf2_64(static_cast<uint64_t>(size)))
      return false;
    auto *interface =
        target.getEncoding()
            .getDialect()
            .getRegisteredInterface<triton::DialectInferLayoutInterface>();
    if (!interface ||
        failed(interface->inferReshapeOpEncoding(
            target.getShape(), target.getEncoding(), ArrayRef<int64_t>{size},
            expression->flatEncoding, /*allowReorder=*/false,
            root->getLoc())) ||
        (isa<StoreOp>(root) && !value.hasOneUse()))
      return false;
  }

  if (failed(requestExpression(value, target.getEncoding(), 0)))
    return false;

  unsigned conversions = 0, joins = 0;
  bool hasCommunication = false;
  auto classify = [&](ConvertLayoutOp conversion) {
    ++conversions;
    hasCommunication |= !cvtReordersRegisters(conversion.getSrc().getType(),
                                              conversion.getType());
  };
  if (indexed)
    if (auto conversion = dyn_cast<ConvertLayoutOp>(root))
      classify(conversion);
  for (Operation *operation : expression->operations) {
    if (indexed) {
      if (auto conversion = dyn_cast<ConvertLayoutOp>(operation))
        classify(conversion);
      joins += isa<JoinOp>(operation);
    }
    if (!indexed || !isExpressionLeaf(operation))
      if (llvm::any_of(operation->getUsers(), [&](Operation *user) {
            return user == root ? indexed && operation->getResult(0) != value
                                : !expression->operations.contains(user);
          }))
        return false;
  }
  if (indexed && (joins == 0 || conversions < 2 || !hasCommunication))
    return false;

  auto isArithmetic = [&](Operation *operation) {
    return indexed ? !isExpressionLeaf(operation) &&
                         !isa<ConvertLayoutOp, ReshapeOp, JoinOp>(operation)
                   : !isa<ConvertLayoutOp>(operation);
  };
  if (llvm::count_if(nodes, [&](const Materialization &node) {
        return isArithmetic(node.value.getDefiningOp());
      }) > llvm::count_if(expression->operations, isArithmetic))
    return false;

  solve();
  OpBuilder builder(root);
  if (indexed) {
    auto rangeType =
        RankedTensorType::get({target.getNumElements()}, builder.getI32Type(),
                              expression->flatEncoding);
    Value range =
        MakeRangeOp::create(builder, root->getLoc(), rangeType, 0,
                            static_cast<int32_t>(target.getNumElements()));
    auto indexType = RankedTensorType::get(
        target.getShape(), builder.getI32Type(), target.getEncoding());
    expression->indices =
        ReshapeOp::create(builder, root->getLoc(), indexType, range);
  }
  unsigned rootNode =
      indices.find({{{value, target.getEncoding()}, 0}, root})->second;
  Value replacement = materialize(rootNode, &builder);
  if (auto conversion = dyn_cast<ConvertLayoutOp>(root)) {
    conversion.getResult().replaceAllUsesWith(replacement);
    conversion.erase();
  } else {
    cast<StoreOp>(root).getValueMutable().assign(replacement);
  }
  for (Operation *operation : expression->operations)
    if (isOpTriviallyDead(operation))
      operation->erase();
  return true;
}

/// Every physical rewrite is an alternative in the same policy-ordered
/// worklist. Root collection restarts only when an expression consumes other
/// roots; conversion sweeps retain dominance-safe reuse between alternatives.
bool LayoutMaterializationGraph::materialize(Operation *root, Action action,
                                             bool preserveSharedReductions) {
  auto conversion = dyn_cast<ConvertLayoutOp>(root);
  switch (action) {
  case Action::BackwardSlice: {
    RankedTensorType target = conversion.getType();
    if (isa<DotOperandEncodingAttr>(target.getEncoding()))
      return false;
    LayoutSlice slice;
    if (failed(plan(conversion.getSrcMutable(), target.getEncoding(), slice)) ||
        !problem.costModel.isRematerializationBeneficial(
            conversion, slice, /*newConversionCost=*/0,
            problem.disableRematSplitting, preserveSharedReductions))
      return false;
    materialize(conversion, slice);
    return true;
  }
  case Action::Narrowing:
  case Action::Conditional:
  case Action::DotOperand:
    return hoist(conversion, action);
  case Action::DominatingReuse:
    return reuse(conversion);
  case Action::ConversionExpression:
    return materialize(conversion, /*indexed=*/false) ||
           materialize(conversion, /*indexed=*/true);
  case Action::StoreExpression:
    return materialize(root, /*indexed=*/true);
  }
  llvm_unreachable("unknown materialization action");
}

LogicalResult
LayoutMaterializationGraph::materialize(ArrayRef<Action> actions,
                                        bool preserveSharedReductions) {
  for (Action action : actions) {
    bool repeat;
    do {
      problem.domInfo.invalidate();
      problem.postDomInfo.invalidate();
      bool changed = false;

      bool expression = action == Action::StoreExpression ||
                        action == Action::ConversionExpression;
      SmallVector<Operation *> roots;
      problem.funcOp.walk([&](Operation *root) {
        if (action == Action::StoreExpression ? isa<StoreOp>(root)
                                              : isa<ConvertLayoutOp>(root))
          roots.push_back(root);
      });
      if (expression)
        std::reverse(roots.begin(), roots.end());

      for (Operation *root : roots) {
        bool rewritten = materialize(root, action, preserveSharedReductions);
        if (!rewritten && action != Action::DominatingReuse)
          if (auto conversion = dyn_cast<ConvertLayoutOp>(root))
            remember(conversion.getSrc(), conversion.getType().getEncoding(),
                     conversion.getResult());
        changed |= rewritten;
        if (expression && rewritten)
          break;
      }

      clear();
      repeat = changed && (action == Action::BackwardSlice || expression);
      if (action == Action::BackwardSlice &&
          (changed || preserveSharedReductions) && failed(problem.cleanup()))
        return failure();
    } while (repeat);
  }
  return success();
}

LogicalResult LayoutOptimizationAnalysis::run(bool disableSplitting) {
  using G = LayoutMaterializationGraph;
  disableRematSplitting = disableSplitting;
  constexpr Action backward[] = {Action::BackwardSlice};
  for (unsigned iteration = 0; iteration < 2; ++iteration) {
    unsigned original = 0, remaining = 0;
    funcOp.walk([&](ConvertLayoutOp) { ++original; });
    costModel = LayoutCostModel();
    LayoutMemoryProfile memory(funcOp);
    bool hasProtectedStore =
        !memory.protectedLoops.empty() && memory.has(G::HardwareStore);
    bool hasPackedMemoryAssembly = memory.hasPackedAssemblyProtocol();

    if (!memory.protectedLoops.empty() ||
        (memory.features & (G::PackedAssembly | G::ProtectedReduction |
                            G::PermutingReshape | G::DescriptorLoad)) ||
        hasPackedMemoryAssembly || memory.hasPairwiseReductionProtocol()) {
      assignLayouts();
      if ((hasProtectedStore || hasPackedMemoryAssembly ||
           memory.has(G::PermutingReshape)) &&
          failed(cleanup()))
        return failure();

      if ((hasProtectedStore || hasPackedMemoryAssembly) &&
          failed(graph.materialize(backward,
                                   /*preserveSharedReductions=*/false)))
        return failure();
      memory = LayoutMemoryProfile(funcOp);
    }

    assignLayouts(&memory);
    if (failed(cleanup()))
      return failure();

    if (failed(graph.materialize(backward,
                                 /*preserveSharedReductions=*/true)))
      return failure();

    constexpr Action hoisting[] = {Action::Narrowing, Action::Conditional,
                                   Action::DotOperand};
    if (failed(graph.materialize(
            ArrayRef(hoisting).take_front(disableSplitting ? 1 : 3))))
      return failure();

    runDeadIterArgElimination(funcOp);
    if (failed(cleanup()))
      return failure();

    RewritePatternSet controlFlow(funcOp.getContext());
    scf::ForOp::getCanonicalizationPatterns(controlFlow, funcOp.getContext());
    scf::IfOp::getCanonicalizationPatterns(controlFlow, funcOp.getContext());
    if (failed(applyPatternsGreedily(funcOp, std::move(controlFlow))))
      LLVM_DEBUG(DBGS() << "scf cleanup did not converge\n");

    constexpr Action finishing[] = {Action::DominatingReuse,
                                    Action::StoreExpression,
                                    Action::ConversionExpression};
    if (failed(graph.materialize(finishing)) || failed(cleanup()))
      return failure();
    funcOp.walk([&](ConvertLayoutOp) { ++remaining; });
    if (!remaining || remaining >= original)
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
    if (getOperation()
            .walk([&](FuncOp function) {
              return failed(LayoutOptimizationAnalysis(function).run(
                         disableRematSplitting))
                         ? WalkResult::interrupt()
                         : WalkResult::advance();
            })
            .wasInterrupted())
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu
