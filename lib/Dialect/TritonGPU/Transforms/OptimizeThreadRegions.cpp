#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

#include <array>

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUOPTIMIZETHREADREGIONS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {
using Domain = ThreadDomainAttr;

Domain fullDomain(MLIRContext *ctx) { return Domain::get(ctx, 0, 0, 0); }

Domain ownerDomain(Operation *op, bool atomicDependent = false) {
  return Domain::get(
      op->getContext(),
      TritonGPUDialect::getThreadsPerWarp(op->getParentOfType<ModuleOp>()) - 1,
      lookupNumWarps(op) - 1, atomicDependent ? lookupNumCTAs(op) - 1 : 0);
}

scf::IfOp physicalGuard(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->getAttrOfType<Domain>("ttg.execution_domain"))
      return cast<scf::IfOp>(parent);
  return {};
}

Domain executionDomain(Operation *op) {
  if (auto guard = physicalGuard(op))
    return guard->getAttrOfType<Domain>("ttg.execution_domain");
  return fullDomain(op->getContext());
}

bool isScalar(Type type) {
  return type.isIntOrFloat() || isa<PointerType>(type);
}

bool isArithmetic(Operation *op) {
  return isa<arith::ArithDialect>(op->getDialect()) ||
         isa<AddPtrOp, PtrToIntOp, IntToPtrOp, BitcastOp>(op);
}

bool isMovableArithmetic(Operation *op) {
  return isArithmetic(op) && isSpeculatable(op);
}

struct RegionEffects {
  bool hasMemory = false;
  bool hasAtomic = false;
  bool hasProtocol = false;
  bool before = false;
  bool after = false;
};

// Keep scalar control flow intact, including arithmetic that cannot speculate.
// A rejected subtree must not contribute effects to the preceding region.
bool analyzeScalarOperation(Operation *root, RegionEffects &effects) {
  RegionEffects candidate = effects;
  auto result = root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!llvm::all_of(
            llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes()),
            isScalar))
      return WalkResult::interrupt();
    bool memory = isa<LoadOp, StoreOp, AtomicLoadOp, AtomicStoreOp, AtomicCASOp,
                      AtomicRMWOp>(op);
    bool loop = isa<scf::ForOp, scf::WhileOp>(op);
    bool controlFlow = loop || isa<scf::IfOp>(op);
    bool barrier = isa<BarrierOp>(op);
    auto atomic = dyn_cast<AtomicOpInterface>(op);
    if (auto load = dyn_cast<LoadOp>(op); load && load.getIsVolatile())
      return WalkResult::interrupt();
    if (!memory && !controlFlow && !barrier &&
        !isa<scf::ConditionOp, scf::YieldOp>(op) && !isArithmetic(op))
      return WalkResult::interrupt();

    candidate.hasMemory |= memory;
    candidate.hasProtocol |= controlFlow || barrier || bool(atomic);
    // Other threads must observe completion even if no value leaves a loop.
    candidate.after |= loop || barrier;
    candidate.before |= barrier;
    if (atomic) {
      candidate.hasAtomic = true;
      auto stages = getAtomicBarrierStages(atomic.getMemSemantic(),
                                           /*hasResultBarrier=*/false);
      candidate.before |= stages.beforeMemoryEffects;
      candidate.after |= stages.afterMemoryEffects;
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return false;
  effects = candidate;
  return true;
}

llvm::SetVector<Operation *> getAtomicDependencies(Block *block) {
  llvm::SetVector<Operation *> slice;
  if (lookupNumCTAs(block->getParentOp()) == 1)
    return slice;
  ForwardSliceOptions options;
  options.inclusive = true;
  options.filter = [&](Operation *op) {
    Operation *root = block->findAncestorOpInBlock(*op);
    RegionEffects effects;
    if (!root ||
        (!slice.contains(root) && !analyzeScalarOperation(root, effects)))
      return false;
    slice.insert(root);
    return true;
  };
  // In SSA order, revisit marked region ops to propagate captured uses through
  // their results.
  for (Operation &op : *block) {
    bool reached = slice.contains(&op);
    if (reached && op.getNumRegions() == 0)
      continue;
    RegionEffects effects;
    if (reached ||
        (analyzeScalarOperation(&op, effects) && effects.hasAtomic)) {
      slice.insert(&op);
      getForwardSlice(&op, &slice, options);
    }
  }
  return slice;
}

void hoistUniformArithmetic(SmallVectorImpl<Operation *> &ops) {
  DenseSet<Operation *> retainedOps;
  unsigned retained = 0;
  for (Operation *op : ops) {
    bool uniform = isMovableArithmetic(op) &&
                   llvm::all_of(op->getOperands(), [&](Value operand) {
                     return !retainedOps.contains(operand.getDefiningOp()) &&
                            !operand.getDefiningOp<ThreadHandoffOp>();
                   });
    if (!uniform) {
      retainedOps.insert(op);
      ops[retained++] = op;
      continue;
    }
    // Leading uniform operations already precede the first retained operation.
    if (retained)
      op->moveBefore(ops.front());
  }
  assert(retained);
  ops.resize(retained);
}

bool isTensorTailOperation(Operation *op) {
  return isArithmetic(op) || isa<SplatOp, MakeRangeOp, StoreOp>(op);
}

struct TensorTail {
  llvm::SetVector<Operation *> ops;
  Domain domain;
};

struct RegionOutputs {
  SmallVector<Value> values;
  SmallVector<Domain> destinations;
};

std::array<uint32_t, 3> getTensorThreadMasks(RankedTensorType type) {
  auto masks = toLinearLayout(type).getFreeVariableMasks();
  auto *ctx = type.getContext();
  return {uint32_t(masks.lookup(StringAttr::get(ctx, "lane"))),
          uint32_t(masks.lookup(StringAttr::get(ctx, "warp"))),
          uint32_t(masks.lookup(StringAttr::get(ctx, "block")))};
}

RegionOutputs getRegionOutputs(ArrayRef<Operation *> ops,
                               const TensorTail &tail, Domain owner) {
  RegionOutputs outputs;
  Block *block = ops.front()->getBlock();
  assert(llvm::all_of(ops,
                      [&](Operation *op) { return op->getBlock() == block; }));
  DenseSet<Operation *> roots(ops.begin(), ops.end());
  bool fullDestination =
      tail.ops.empty() || !tail.domain.hasSameWarpAndBlock(owner);
  for (Operation *op : ops)
    for (Value result : op->getResults()) {
      Domain destination;
      for (Operation *user : result.getUsers()) {
        if (roots.contains(block->findAncestorOpInBlock(*user)))
          continue;
        destination = !fullDestination && tail.ops.contains(user)
                          ? tail.domain
                          : fullDomain(owner.getContext());
        if (destination.isFull())
          break;
      }
      if (destination) {
        outputs.values.push_back(result);
        outputs.destinations.push_back(destination);
      }
    }
  return outputs;
}

// Restrict the tensor tail to its owners only when no result escapes.
// Preserve each layout's physical dimensions and logical element placement.
TensorTail findTensorTail(Operation *first, Domain owner) {
  assert(first && "expected an operation following the scalar region");
  TensorTail tail;
  uint32_t lane = owner.getLane(), warp = owner.getWarp();
  uint32_t block = owner.getBlock();
  bool hasStore = false, hasTensor = false;
  for (Operation *op = first; isTensorTailOperation(op);
       op = op->getNextNode()) {
    tail.ops.insert(op);
    hasStore |= isa<StoreOp>(op);
    for (Type type :
         llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        hasTensor = true;
        auto [laneMask, warpMask, blockMask] = getTensorThreadMasks(tensor);
        lane &= laneMask;
        warp &= warpMask;
        block &= blockMask;
      }
    }
  }
  // The shuffle's member mask includes every lane in the warp. Keep all lanes
  // participating even when the tensor needs fewer than 32.
  if (lane != owner.getLane())
    lane = 0;
  if (!hasStore || !hasTensor || !(lane | warp | block))
    return {};
  for (Operation *op : tail.ops)
    for (Operation *user : op->getUsers())
      if (!tail.ops.contains(user))
        return {};
  tail.domain = Domain::get(owner.getContext(), lane, warp, block);
  return tail;
}

Value zero(OpBuilder &b, Location loc, Type type) {
  if (isa<PointerType>(type)) {
    Value bits = arith::ConstantIntOp::create(b, loc, 0, 64);
    return IntToPtrOp::create(b, loc, type, bits);
  }
  return arith::ConstantOp::create(b, loc, type, b.getZeroAttr(type));
}

scf::IfOp wrapThreadRegion(ArrayRef<Operation *> ops, ValueRange outputs,
                           Domain domain) {
  OpBuilder b(ops.front());
  Location loc = ops.front()->getLoc();
  Value pred = ThreadPredicateOp::create(b, loc, b.getI1Type(), domain);
  auto guard = scf::IfOp::create(b, loc, outputs.getTypes(), pred);
  guard->setAttr("ttg.execution_domain", domain);
  Block *body = b.createBlock(&guard.getThenRegion());
  for (Operation *op : ops)
    op->moveBefore(body, body->end());
  scf::YieldOp::create(b, loc, outputs);
  if (outputs.empty())
    return guard;
  b.createBlock(&guard.getElseRegion());
  SmallVector<Value> inactive;
  for (Type type : outputs.getTypes())
    inactive.push_back(zero(b, loc, type));
  scf::YieldOp::create(b, loc, inactive);
  return guard;
}

void wrapScalarRegion(ArrayRef<Operation *> ops, const RegionEffects &effects,
                      const TensorTail &tail, const RegionOutputs &results,
                      Domain owner) {
  ValueRange outputs = results.values;

  OpBuilder b(ops.front());
  Location loc = ops.front()->getLoc();
  auto emitBarrier = [&] {
    if (owner.getBlock())
      nvidia_gpu::ClusterBarrierOp::create(b, loc);
    else
      BarrierOp::create(b, loc, AddrSpace::All);
  };
  if (effects.before)
    emitBarrier();
  auto scope = ThreadScopeOp::create(b, loc, outputs.getTypes());
  Block *body = b.createBlock(&scope.getBody());
  for (Operation *op : ops)
    op->moveBefore(body, body->end());
  ThreadYieldOp::create(b, loc, outputs);
  scope.walk([](BarrierOp barrier) { barrier.erase(); });
  auto guard = wrapThreadRegion({scope}, scope.getResults(), owner);
  // Cross-warp and cross-CTA handoffs require the full region's participants.
  // A single-warp tail in the source CTA needs only its own shuffle.
  OpBuilder tailBuilder(b.getContext());
  if (!tail.ops.empty()) {
    auto consumer = wrapThreadRegion(tail.ops.getArrayRef(), {}, tail.domain);
    tailBuilder.setInsertionPointToStart(consumer.thenBlock());
  }
  b.setInsertionPointAfter(guard);
  if (effects.after)
    emitBarrier();
  for (auto [output, destination, result] :
       llvm::zip_equal(outputs, results.destinations, guard.getResults())) {
    Value value =
        ThreadHandoffOp::create(destination.isFull() ? b : tailBuilder, loc,
                                result, owner, destination);
    output.replaceUsesWithIf(value, [&](OpOperand &use) {
      return !guard->isProperAncestor(use.getOwner());
    });
  }
}

void restrictHandoffConsumers(ArrayRef<ThreadHandoffOp> handoffs) {
  for (ThreadHandoffOp handoff : handoffs) {
    if (!handoff.getDestination().isFull())
      continue;
    for (Operation *consumer : handoff.getResult().getUsers()) {
      if (!executionDomain(consumer).isFull())
        continue;
      auto tail = findTensorTail(consumer, ownerDomain(consumer));
      if (!tail.ops.empty())
        wrapThreadRegion(tail.ops.getArrayRef(), {}, tail.domain);
    }
  }
}

// A loop's completion rendezvous remains at its exit. Only its data handoff
// moves to a later consumer, after any unrelated full-block work.
void sinkHandoffs(ArrayRef<ThreadHandoffOp> handoffs) {
  for (ThreadHandoffOp handoff : handoffs) {
    assert(!handoff.getResult().use_empty() && "handoffs have escaping users");
    if (!handoff.getDestination().isFull())
      continue;
    auto users = handoff.getResult().getUsers();
    auto consumer = physicalGuard(*users.begin());
    if (!consumer ||
        !llvm::all_of(llvm::drop_begin(users), [&](Operation *user) {
          return physicalGuard(user) == consumer;
        }))
      continue;
    auto destination = consumer->getAttrOfType<Domain>("ttg.execution_domain");
    auto source = handoff.getSource();
    if (!destination.hasSameWarpAndBlock(source))
      continue;
    assert((destination == source || !destination.getLane()) &&
           "generated regions use either the owner lane or every lane");
    handoff.setDestinationAttr(destination);
    Block *body = consumer.thenBlock();
    handoff->moveBefore(body, body->begin());
  }
}

void sinkConditionalArithmetic(ModuleOp module) {
  DominanceInfo dominance(module);
  module.walk<WalkOrder::PreOrder>([&](scf::IfOp branch) {
    if (executionDomain(branch).isFull())
      return;
    controlFlowSink(
        branch->getRegions(), dominance,
        [&](Operation *op, Region *region) {
          // Keep loop-invariant work outside loops by sinking only from the
          // branch's block.
          return op->getBlock() == region->getParentOp()->getBlock() &&
                 isMovableArithmetic(op);
        },
        [](Operation *op, Region *region) {
          op->moveBefore(&region->front(), region->front().begin());
        });
  });
}

#ifndef NDEBUG
void assertValidRegions(ModuleOp module) {
  module.walk([&](Operation *op) {
    if (auto domain = op->getAttrOfType<Domain>("ttg.execution_domain")) {
      auto guard = cast<scf::IfOp>(op);
      auto predicate = guard.getCondition().getDefiningOp<ThreadPredicateOp>();
      assert(predicate && predicate.getDomain() == domain &&
             "physical region requires its declared thread predicate");
      assert(executionDomain(op).isFull() &&
             "physical regions are inserted in full execution domains");
      for (Operation *user : guard->getUsers()) {
        auto handoff = dyn_cast<ThreadHandoffOp>(user);
        assert(handoff && handoff.getSource() == domain &&
               "physical region results require a matching handoff");
      }
    }
    if (auto handoff = dyn_cast<ThreadHandoffOp>(op)) {
      auto source = handoff.getSrc().getDefiningOp<scf::IfOp>();
      assert(source &&
             source->getAttrOfType<Domain>("ttg.execution_domain") ==
                 handoff.getSource() &&
             "handoff source must belong to its declared physical region");
      assert(handoff.getDestination() == executionDomain(op) &&
             "handoff must execute in its destination domain");
    }
    Domain domain = executionDomain(op);
    assert((!isa<ThreadScopeOp>(op) ||
            domain == ownerDomain(op, domain.getBlock() != 0)) &&
           "thread scope requires the canonical thread region");
    if (domain.isFull())
      return;
    assert((!isa<BarrierOp, nvidia_gpu::ClusterBarrierOp>(op)) &&
           "block rendezvous must remain outside physical regions");
    auto covered = [&](Type type) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (!tensor)
        return true;
      auto [laneMask, warpMask, blockMask] = getTensorThreadMasks(tensor);
      return domain.hasMasksWithin(laneMask, warpMask, blockMask);
    };
    bool coversTypes = llvm::all_of(
        llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes()),
        covered);
    assert(coversTypes && "physical region must retain every tensor element");
  });
}
#endif

void optimizeBlock(Block *block) {
  assert(!block->empty() && "frontend and generated blocks are nonempty");
  if (!executionDomain(&block->front()).isFull())
    return;
  // Classify before wrapping replaces operands with handoff results.
  auto atomicDependencies = getAtomicDependencies(block);
  Operation *next = &block->front();
  while (next) {
    SmallVector<Operation *> ops;
    RegionEffects effects;
    bool atomicDependent = atomicDependencies.contains(next);
    while (!next->hasTrait<OpTrait::IsTerminator>() &&
           (atomicDependencies.contains(next) == atomicDependent ||
            (atomicDependent && isMovableArithmetic(next))) &&
           analyzeScalarOperation(next, effects)) {
      ops.push_back(next);
      next = next->getNextNode();
      assert(next && "scalar operations must precede a block terminator");
    }
    if (ops.empty()) {
      next = next->getNextNode();
      continue;
    }
    if (!effects.hasMemory && !atomicDependent)
      continue;
    hoistUniformArithmetic(ops);
    Domain owner = ownerDomain(ops.front(), atomicDependent);
    auto tail = !atomicDependent && atomicDependencies.contains(next)
                    ? TensorTail{}
                    : findTensorTail(next, owner);
    Operation *resume =
        tail.ops.empty() ? next : tail.ops.back()->getNextNode();
    if (!tail.ops.empty() && tail.domain == owner && !effects.after) {
      // Tail results cannot escape, so fusion preserves the profitability
      // check: only scalar uses outside the tail still require a handoff.
      llvm::append_range(ops, tail.ops);
      tail = {};
    }
    auto outputs = getRegionOutputs(ops, tail, owner);
    // Ordinary scalar loads already produce full copies without shared
    // memory. Restrict them only when all consumers stay in one warp.
    if (!effects.hasProtocol && !atomicDependent &&
        llvm::any_of(outputs.destinations,
                     [](Domain domain) { return domain.isFull(); }))
      continue;
    next = resume;
    wrapScalarRegion(ops, effects, tail, outputs, owner);
  }
}

llvm::SetVector<Operation *> getPerElementRoots(ModuleOp module) {
  llvm::SetVector<Operation *> roots;
  module.walk([&](MapElementwiseOp op) { roots.insert(op); });
  SymbolTableCollection symbolTable;
  for (unsigned i = 0; i < roots.size(); ++i) {
    Operation *root = roots[i];
    root->walk([&](CallOpInterface call) {
      if (Operation *callee = call.resolveCallableInTable(&symbolTable))
        roots.insert(callee);
    });
  }
  return roots;
}

class OptimizeThreadRegions
    : public impl::TritonGPUOptimizeThreadRegionsBase<OptimizeThreadRegions> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Map callback scalars vary per element, including in their callees.
    auto perElementRoots = getPerElementRoots(module);
    SmallVector<ThreadHandoffOp> handoffs;
    module.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
      if (perElementRoots.contains(op))
        return WalkResult::skip();
      for (Region &region : op->getRegions())
        for (Block &block : region)
          optimizeBlock(&block);
      if (auto handoff = dyn_cast<ThreadHandoffOp>(op))
        handoffs.push_back(handoff);
      return WalkResult::advance();
    });
    restrictHandoffConsumers(handoffs);
    sinkHandoffs(handoffs);
    sinkConditionalArithmetic(module);
#ifndef NDEBUG
    assertValidRegions(module);
#endif
  }
};
} // namespace
} // namespace mlir::triton::gpu
