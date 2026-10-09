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
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUOPTIMIZETHREADREGIONS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {
using Domain = ThreadDomainAttr;

Domain fullDomain(MLIRContext *ctx) { return Domain::get(ctx, 0, 0); }

Domain ownerDomain(Operation *op) {
  return Domain::get(
      op->getContext(),
      TritonGPUDialect::getThreadsPerWarp(op->getParentOfType<ModuleOp>()) - 1,
      lookupNumWarps(op) - 1);
}

Domain executionDomain(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (auto domain = parent->getAttrOfType<Domain>("ttg.execution_domain"))
      return domain;
  return fullDomain(op->getContext());
}

bool isFull(Domain domain) { return !(domain.getLane() | domain.getWarp()); }

bool isScalar(Type type) {
  return type.isIntOrFloat() || isa<PointerType>(type);
}

bool isArithmetic(Operation *op) {
  return op->getDialect()->getNamespace() == "arith" ||
         isa<AddPtrOp, PtrToIntOp, IntToPtrOp, BitcastOp>(op);
}

struct RegionEffects {
  bool hasMemory = false;
  bool hasProtocol = false;
  bool before = false;
  bool after = false;
};

// Keep scalar control flow intact, including arithmetic that cannot speculate.
// A rejected subtree must not contribute effects to the preceding region.
bool analyzeScalarOperation(Operation *root, RegionEffects &effects) {
  RegionEffects candidate = effects;
  auto result = root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!llvm::all_of(op->getOperandTypes(), isScalar) ||
        !llvm::all_of(op->getResultTypes(), isScalar))
      return WalkResult::interrupt();
    bool memory = isa<LoadOp, StoreOp, AtomicLoadOp, AtomicStoreOp, AtomicCASOp,
                      AtomicRMWOp>(op);
    if (auto load = dyn_cast<LoadOp>(op); load && load.getIsVolatile())
      return WalkResult::interrupt();
    if (!memory &&
        !isa<scf::IfOp, scf::WhileOp, scf::ForOp, scf::ConditionOp,
             scf::YieldOp, BarrierOp>(op) &&
        !isArithmetic(op))
      return WalkResult::interrupt();

    candidate.hasMemory |= memory;
    candidate.hasProtocol |=
        isa<scf::IfOp, scf::ForOp, scf::WhileOp, BarrierOp, AtomicOpInterface>(
            op);
    // Other threads must observe completion even if no value leaves a loop.
    candidate.after |= isa<scf::ForOp, scf::WhileOp>(op);
    if (isa<BarrierOp>(op))
      candidate.before = candidate.after = true;
    if (auto atomic = dyn_cast<AtomicOpInterface>(op)) {
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

void hoistUniformArithmetic(SmallVectorImpl<Operation *> &ops) {
  DenseSet<Operation *> retainedOps;
  unsigned retained = 0;
  for (Operation *op : ops) {
    bool uniform = isArithmetic(op) && isSpeculatable(op) &&
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

RegionOutputs getRegionOutputs(ArrayRef<Operation *> ops,
                               const TensorTail &tail, Domain owner) {
  RegionOutputs outputs;
  Block *block = ops.front()->getBlock();
  assert(llvm::all_of(ops,
                      [&](Operation *op) { return op->getBlock() == block; }));
  DenseSet<Operation *> roots(ops.begin(), ops.end());
  for (Operation *op : ops)
    for (Value result : op->getResults()) {
      bool escapes = false;
      bool materializeFull =
          tail.ops.empty() || tail.domain.getWarp() != owner.getWarp();
      for (Operation *user : result.getUsers()) {
        if (roots.contains(block->findAncestorOpInBlock(*user)))
          continue;
        escapes = true;
        materializeFull |= !tail.ops.contains(user);
      }
      if (escapes) {
        outputs.values.push_back(result);
        outputs.destinations.push_back(
            materializeFull ? fullDomain(owner.getContext()) : tail.domain);
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
  bool hasStore = false, hasTensor = false;
  for (Operation *op = first; isTensorTailOperation(op);
       op = op->getNextNode()) {
    tail.ops.insert(op);
    hasStore |= isa<StoreOp>(op);
    auto restrictDomain = [&](Type type) {
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
        hasTensor = true;
        auto masks = toLinearLayout(tensor).getFreeVariableMasks();
        lane &= masks.lookup(StringAttr::get(op->getContext(), "lane"));
        warp &= masks.lookup(StringAttr::get(op->getContext(), "warp"));
      }
    };
    llvm::for_each(op->getOperandTypes(), restrictDomain);
    llvm::for_each(op->getResultTypes(), restrictDomain);
  }
  // The shuffle's member mask includes every lane in the warp. Keep all lanes
  // participating even when the tensor needs fewer than 32.
  if (lane != owner.getLane())
    lane = 0;
  if (!hasStore || !hasTensor || !(lane | warp))
    return {};
  for (Operation *op : tail.ops)
    for (Operation *user : op->getUsers())
      if (!tail.ops.contains(user))
        return {};
  tail.domain = Domain::get(owner.getContext(), lane, warp);
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
  b.createBlock(&guard.getElseRegion());
  SmallVector<Value> inactive;
  for (Type type : outputs.getTypes())
    inactive.push_back(zero(b, loc, type));
  scf::YieldOp::create(b, loc, inactive);
  return guard;
}

void wrapScalarRegion(ArrayRef<Operation *> ops, const RegionEffects &effects,
                      const TensorTail &tail, RegionOutputs results) {
  Domain owner = ownerDomain(ops.front());
  ValueRange outputs = results.values;

  OpBuilder b(ops.front());
  Location loc = ops.front()->getLoc();
  if (effects.before)
    BarrierOp::create(b, loc, AddrSpace::All);
  auto scope = ThreadScopeOp::create(b, loc, outputs.getTypes());
  Block *body = b.createBlock(&scope.getBody());
  for (Operation *op : ops)
    op->moveBefore(body, body->end());
  ThreadYieldOp::create(b, loc, outputs);
  scope.walk([](BarrierOp barrier) { barrier.erase(); });
  auto guard = wrapThreadRegion({scope}, scope.getResults(), owner);
  // Cross-warp handoffs execute in the full region so every barrier has all
  // its participants. A single-warp tail needs only its own shuffle.
  OpBuilder tailBuilder(b.getContext());
  if (!tail.ops.empty()) {
    auto consumer = wrapThreadRegion(tail.ops.getArrayRef(), {}, tail.domain);
    tailBuilder.setInsertionPointToStart(&consumer.getThenRegion().front());
  }
  b.setInsertionPointAfter(guard);
  if (effects.after)
    BarrierOp::create(b, loc, AddrSpace::All);
  for (auto [output, destination, result] :
       llvm::zip_equal(outputs, results.destinations, guard.getResults())) {
    Value value = ThreadHandoffOp::create(isFull(destination) ? b : tailBuilder,
                                          loc, result, owner, destination);
    output.replaceUsesWithIf(value, [&](OpOperand &use) {
      return !guard->isProperAncestor(use.getOwner());
    });
  }
}

void restrictHandoffConsumers(ArrayRef<ThreadHandoffOp> handoffs) {
  for (ThreadHandoffOp handoff : handoffs) {
    if (!isFull(handoff.getDestination()))
      continue;
    for (Operation *consumer : handoff.getResult().getUsers()) {
      if (!isFull(executionDomain(consumer)))
        continue;
      auto tail = findTensorTail(consumer, ownerDomain(consumer));
      if (!tail.ops.empty())
        wrapThreadRegion(tail.ops.getArrayRef(), {}, tail.domain);
    }
  }
}

scf::IfOp physicalGuard(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (parent->hasAttr("ttg.execution_domain"))
      return cast<scf::IfOp>(parent);
  return {};
}

// A loop's completion rendezvous remains at its exit. Only its data handoff
// moves to a later consumer, after any unrelated full-block work.
void sinkHandoffs(ArrayRef<ThreadHandoffOp> handoffs) {
  for (ThreadHandoffOp handoff : handoffs) {
    assert(!handoff.getResult().use_empty() && "handoffs have escaping users");
    if (!isFull(handoff.getDestination()))
      continue;
    scf::IfOp consumer;
    bool oneRegion = true;
    for (Operation *user : handoff.getResult().getUsers()) {
      auto guard = physicalGuard(user);
      if (!guard || (consumer && consumer != guard)) {
        oneRegion = false;
        break;
      }
      consumer = guard;
    }
    if (!oneRegion)
      continue;
    auto destination = consumer->getAttrOfType<Domain>("ttg.execution_domain");
    auto source = handoff.getSource();
    if (destination.getWarp() != source.getWarp())
      continue;
    assert((destination == source || !destination.getLane()) &&
           "generated regions use either the owner lane or every lane");
    handoff->setAttr("destination", destination);
    handoff->moveBefore(&consumer.getThenRegion().front(),
                        consumer.getThenRegion().front().begin());
  }
}

void sinkConditionalArithmetic(ModuleOp module) {
  DominanceInfo dominance(module);
  module.walk<WalkOrder::PreOrder>([&](scf::IfOp branch) {
    if (isFull(executionDomain(branch)))
      return;
    controlFlowSink(
        branch->getRegions(), dominance,
        [&](Operation *op, Region *region) {
          // Keep loop-invariant work outside loops by sinking only from the
          // branch's block.
          return op->getBlock() == region->getParentOp()->getBlock() &&
                 isArithmetic(op) && isSpeculatable(op);
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
      assert(isFull(executionDomain(op)) &&
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
    assert((!isa<ThreadScopeOp>(op) || domain == ownerDomain(op)) &&
           "thread scope requires the canonical thread region");
    if (isFull(domain))
      return;
    assert(!isa<BarrierOp>(op) &&
           "block rendezvous must remain outside physical regions");
    auto covered = [&](Type type) {
      auto tensor = dyn_cast<RankedTensorType>(type);
      if (!tensor)
        return true;
      auto masks = toLinearLayout(tensor).getFreeVariableMasks();
      return !(domain.getLane() &
               ~masks.lookup(StringAttr::get(op->getContext(), "lane"))) &&
             !(domain.getWarp() &
               ~masks.lookup(StringAttr::get(op->getContext(), "warp")));
    };
    bool coversTypes = llvm::all_of(op->getOperandTypes(), covered) &&
                       llvm::all_of(op->getResultTypes(), covered);
    assert(coversTypes && "physical region must retain every tensor element");
  });
}
#endif

void optimizeBlock(Block *block) {
  assert(!block->empty() && "frontend and generated blocks are nonempty");
  if (!isFull(executionDomain(&block->front())))
    return;
  Operation *next = &block->front();
  while (next) {
    SmallVector<Operation *> ops;
    RegionEffects effects;
    while (!next->hasTrait<OpTrait::IsTerminator>() &&
           analyzeScalarOperation(next, effects)) {
      ops.push_back(next);
      next = next->getNextNode();
      assert(next && "scalar operations must precede a block terminator");
    }
    if (ops.empty()) {
      next = next->getNextNode();
      continue;
    }
    if (!effects.hasMemory)
      continue;
    hoistUniformArithmetic(ops);
    Domain owner = ownerDomain(ops.front());
    auto tail = findTensorTail(next, owner);
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
    if (!effects.hasProtocol && llvm::any_of(outputs.destinations, isFull))
      continue;
    next = resume;
    wrapScalarRegion(ops, effects, tail, std::move(outputs));
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
    if (TritonGPUDialect::getNumCTAs(module) != 1)
      return;
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
