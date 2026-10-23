#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

#include <array>
#include <optional>

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

Domain guardDomain(Operation *op) {
  if (auto guard = dyn_cast<scf::IfOp>(op))
    if (auto predicate =
            guard.getCondition().getDefiningOp<ThreadPredicateOp>())
      return predicate.getDomain();
  return {};
}

scf::IfOp physicalGuard(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (guardDomain(parent))
      return cast<scf::IfOp>(parent);
  return {};
}

Domain executionDomain(Operation *op) {
  if (auto guard = physicalGuard(op))
    return guardDomain(guard);
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
  bool hasMemory = false; // contains loads/stores/atomics
  bool hasAtomic = false; // contains an atomic
  bool hasSCF = false;    // contains ifs/loops
  bool needsBarrierBefore = false;
  bool needsBarrierAfter = false;

  void merge(const RegionEffects &other) {
    hasMemory |= other.hasMemory;
    hasAtomic |= other.hasAtomic;
    hasSCF |= other.hasSCF;
    needsBarrierBefore |= other.needsBarrierBefore;
    needsBarrierAfter |= other.needsBarrierAfter;
  }
};

// Get memory and synchronization requirements for the entire scalar subtree.
// Return std::nullopt if any operation is unsupported.
std::optional<RegionEffects> getScalarRegionEffects(Operation *root) {
  RegionEffects effects;
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

    effects.hasMemory |= memory;
    effects.hasSCF |= controlFlow;
    // Other threads must observe completion even if no value leaves a loop.
    effects.needsBarrierAfter |= loop || barrier;
    effects.needsBarrierBefore |= barrier;
    if (atomic) {
      effects.hasAtomic = true;
      auto stages = getAtomicBarrierStages(atomic.getMemSemantic(),
                                           /*hasResultBarrier=*/false);
      effects.needsBarrierBefore |= stages.beforeMemoryEffects;
      effects.needsBarrierAfter |= stages.afterMemoryEffects;
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return std::nullopt;
  return effects;
}

// Find scalar atomics and their transitive scalar users so dependent regions
// run on CTA 0 in multi-CTA blocks, while unrelated regions stay per CTA.
DenseSet<Operation *> getAtomicDependencies(Block *block) {
  DenseSet<Operation *> slice;
  // This is a heuristic for `num_ctas > 1`
  if (lookupNumCTAs(block->getParentOp()) == 1)
    return slice;
  for (Operation &op : *block) {
    auto effects = getScalarRegionEffects(&op);
    if (!effects)
      continue;

    // Only earlier ops in this block enter slice; nested matches are captures.
    bool dependent = effects->hasAtomic;
    if (!dependent && !slice.empty()) {
      auto result = op.walk<WalkOrder::PreOrder>([&](Operation *nested) {
        for (Value operand : nested->getOperands())
          if (slice.contains(operand.getDefiningOp()))
            return WalkResult::interrupt();
        return WalkResult::advance();
      });
      dependent = result.wasInterrupted();
    }
    if (dependent)
      slice.insert(&op);
  }
  return slice;
}

void hoistMovableArithmetic(SmallVectorImpl<Operation *> &ops) {
  llvm::SetVector<Operation *> retainedOps;
  for (Operation *op : ops) {
    bool uniform = isMovableArithmetic(op) &&
                   llvm::all_of(op->getOperands(), [&](Value operand) {
                     return !retainedOps.contains(operand.getDefiningOp()) &&
                            !operand.getDefiningOp<ThreadHandoffOp>();
                   });
    // Leading uniform operations already precede the first retained operation.
    if (!uniform)
      retainedOps.insert(op);
    else if (!retainedOps.empty())
      op->moveBefore(retainedOps.front());
  }
  assert(!retainedOps.empty());
  ops = retainedOps.takeVector();
}

bool isTensorTailOperation(Operation *op) {
  return isArithmetic(op) || isa<SplatOp, MakeRangeOp, StoreOp>(op);
}

struct TensorTail {
  llvm::SetVector<Operation *> ops;
  Domain domain;
};

std::array<uint32_t, 3> getTensorThreadMasks(RankedTensorType type) {
  auto masks = toLinearLayout(type).getFreeVariableMasks();
  auto *ctx = type.getContext();
  return {uint32_t(masks.lookup(StringAttr::get(ctx, "lane"))),
          uint32_t(masks.lookup(StringAttr::get(ctx, "warp"))),
          uint32_t(masks.lookup(StringAttr::get(ctx, "block")))};
}

// Collect escaping values, rejecting full-domain handoffs when disallowed.
std::optional<SmallVector<Value>>
getEscapingValues(ArrayRef<Operation *> ops, const TensorTail &tail,
                  Domain owner, bool allowFullDomainHandoff) {
  SmallVector<Value> outputs;
  Block *block = ops.front()->getBlock();
  assert(llvm::all_of(ops,
                      [&](Operation *op) { return op->getBlock() == block; }));
  DenseSet<Operation *> roots(ops.begin(), ops.end());
  auto isExternalUser = [&](Operation *user) {
    return !roots.contains(block->findAncestorOpInBlock(*user));
  };
  bool canUseTail = !tail.ops.empty() && tail.domain.hasSameWarpAndBlock(owner);
  for (Operation *op : ops)
    for (Value result : op->getResults()) {
      auto externalUsers =
          llvm::make_filter_range(result.getUsers(), isExternalUser);
      if (externalUsers.empty())
        continue;

      if (!allowFullDomainHandoff &&
          (!canUseTail || !llvm::all_of(externalUsers, [&](Operation *user) {
            return tail.ops.contains(user);
          })))
        return std::nullopt;
      outputs.push_back(result);
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
  for (Operation *op = first; isTensorTailOperation(op);
       op = op->getNextNode()) {
    tail.ops.insert(op);
    for (Type type :
         llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
      if (auto tensor = dyn_cast<RankedTensorType>(type)) {
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
  tail.domain = Domain::get(owner.getContext(), lane, warp, block);
  if (tail.domain.isFull())
    return {};
  for (Operation *op : tail.ops)
    for (Operation *user : op->getUsers())
      if (!tail.ops.contains(user))
        return {};
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
                      const TensorTail &tail, ValueRange outputs,
                      Domain owner) {
  OpBuilder b(ops.front());
  Location loc = ops.front()->getLoc();
  auto emitBarrier = [&] {
    if (owner.getBlock())
      nvidia_gpu::ClusterBarrierOp::create(b, loc);
    else
      BarrierOp::create(b, loc, AddrSpace::All);
  };
  if (effects.needsBarrierBefore)
    emitBarrier();
  auto scope = ThreadScopeOp::create(b, loc, outputs.getTypes());
  Block *body = b.createBlock(&scope.getBody());
  for (Operation *op : ops)
    op->moveBefore(body, body->end());
  ThreadYieldOp::create(b, loc, outputs);
  scope.walk([](BarrierOp barrier) { barrier.erase(); });
  auto guard = wrapThreadRegion({scope}, scope.getResults(), owner);
  if (!tail.ops.empty())
    wrapThreadRegion(tail.ops.getArrayRef(), {}, tail.domain);
  b.setInsertionPointAfter(guard);
  // Kernel completion joins all threads, so no exit barrier is needed.
  auto ret = dyn_cast<ReturnOp>(guard->getNextNode());
  bool exitsKernel = ret && isKernel(ret->getParentOfType<FuncOp>());
  if (effects.needsBarrierAfter && !exitsKernel)
    emitBarrier();
  // Cross-warp and cross-CTA handoffs require the full region's participants.
  // Sinking narrows handoffs whose users share a compatible guard.
  for (auto [output, result] : llvm::zip_equal(outputs, guard.getResults())) {
    Value value = ThreadHandoffOp::create(b, loc, result, owner,
                                          fullDomain(owner.getContext()));
    output.replaceUsesWithIf(value, [&](OpOperand &use) {
      return !guard->isProperAncestor(use.getOwner());
    });
  }
}

void restrictHandoffConsumers(ArrayRef<ThreadHandoffOp> handoffs) {
  for (ThreadHandoffOp handoff : handoffs) {
    assert(handoff.getDestination().isFull() &&
           "only handoffs outside physical guards are collected");
    for (Operation *consumer : handoff.getResult().getUsers()) {
      if (!executionDomain(consumer).isFull())
        continue;
      auto tail = findTensorTail(consumer, ownerDomain(consumer));
      if (!tail.ops.empty())
        wrapThreadRegion(tail.ops.getArrayRef(), {}, tail.domain);
    }
  }
}

Region *findSinkRegion(Operation *op) {
  if (op->use_empty())
    return nullptr;
  Operation *firstUser = *op->getUsers().begin();
  scf::IfOp branch;
  auto handoff = dyn_cast<ThreadHandoffOp>(op);
  if (handoff) {
    if (!handoff.getDestination().isFull())
      return nullptr;
    branch = physicalGuard(firstUser);
    if (!branch ||
        !guardDomain(branch).hasSameWarpAndBlock(handoff.getSource()))
      return nullptr;
  } else {
    if (!isMovableArithmetic(op))
      return nullptr;
    // Keep arithmetic in its current loop and physical execution domain.
    branch = dyn_cast_or_null<scf::IfOp>(
        op->getBlock()->findAncestorOpInBlock(*firstUser));
    if (!branch || executionDomain(branch).isFull())
      return nullptr;
  }
  for (Region &region : branch->getRegions()) {
    if (llvm::all_of(op->getUsers(), [&](Operation *user) {
          return region.isAncestor(user->getParentRegion());
        }))
      return &region;
  }
  return nullptr;
}

// Sink movable arithmetic into inner conditionals containing all its uses.
//
// a = x + 1
// if ready:
//     use(a)
// ===>
// if ready:
//     a = x + 1
//     use(a)
//
// Sink handoffs into compatible thread guards and narrow their destinations.
//
// value = handoff(thread_0_result, destination=all_128_threads)
// if warp_id == 0:
//     consume(value)
// ===>
// if warp_id == 0:
//     value = handoff(thread_0_result, destination=warp_0)
//     consume(value)
void sinkRegionInputs(ModuleOp mod) {
  // Process users before producers to propagate sinking without a worklist.
  mod.walk<WalkOrder::PostOrder, ReverseIterator>([&](Operation *op) {
    while (Region *region = findSinkRegion(op)) {
      if (auto handoff = dyn_cast<ThreadHandoffOp>(op)) {
        Domain destination = guardDomain(region->getParentOp());
        assert((destination == handoff.getSource() || !destination.getLane()) &&
               "generated regions use either the owner lane or every lane");
        handoff.setDestinationAttr(destination);
      }
      op->moveBefore(&region->front(), region->front().begin());
    }
  });
}

void optimizeBlock(Block *block) {
  assert(!block->empty() && "frontend and generated blocks are nonempty");
  // For num_ctas > 1 find the ops that depend on atomics
  // For num_ctas == 1 it's the empty set
  auto atomicDependencies = getAtomicDependencies(block);
  Operation *next = &block->front();
  while (next) {
    SmallVector<Operation *> ops;
    RegionEffects effects;
    // This is always false for num_ctas == 1
    bool atomicDependent = atomicDependencies.contains(next);
    // Collect consecutive scalar ops until a terminator or unsupported op.
    // for num_ctas > 1:
    //   keep atomic-dependent (CTA 0) and per-CTA regions separate,
    //   allowing movable arithmetic between atomic-dependent ops.
    while (!next->hasTrait<OpTrait::IsTerminator>() &&
           (atomicDependencies.contains(next) == atomicDependent ||
            (atomicDependent && isMovableArithmetic(next)))) {
      auto opEffects = getScalarRegionEffects(next);
      if (!opEffects)
        break;
      effects.merge(*opEffects);
      ops.push_back(next);
      next = next->getNextNode();
      assert(next && "scalar operations must precede a block terminator");
    }
    if (ops.empty()) {
      next = next->getNextNode();
      continue;
    }
    // Keep regions with memory accesses or atomic-dependent work.
    if (!effects.hasMemory && !atomicDependent)
      continue;
    hoistMovableArithmetic(ops);
    Domain owner = ownerDomain(ops.front(), atomicDependent);
    // Keep CTA-0 atomic-dependent work out of a preceding per-CTA region.
    auto tail = !atomicDependent && atomicDependencies.contains(next)
                    ? TensorTail{}
                    : findTensorTail(next, owner);
    Operation *resume =
        tail.ops.empty() ? next : tail.ops.back()->getNextNode();
    // Fuse tail
    if (!tail.ops.empty() && tail.domain == owner &&
        !effects.needsBarrierAfter) {
      llvm::append_range(ops, tail.ops);
      tail = {};
    }
    // Heuristic: atomic/barrier sequences and executing control flow on fewer
    // threads may avoid intermediate broadcasts and justify a full-domain
    // handoff.
    bool allowFullDomainHandoff = atomicDependent || effects.hasSCF ||
                                  effects.hasAtomic ||
                                  effects.needsBarrierBefore;
    // Ordinary scalar loads already produce full copies without shared
    // memory. Restrict them only when all consumers stay in one warp.
    auto outputs = getEscapingValues(ops, tail, owner, allowFullDomainHandoff);
    if (!outputs)
      continue;
    next = resume;
    wrapScalarRegion(ops, effects, tail, *outputs, owner);
  }
}

// Return tensor callback ops and their transitive noinline callees.
// Map bodies and reduce/scan combiners have per-element scalar values.
llvm::SetVector<Operation *> getTensorCallbackOps(ModuleOp mod) {
  llvm::SetVector<Operation *> roots;
  mod.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<MapElementwiseOp, ReduceOp, ScanOp>(op))
      return WalkResult::advance();
    roots.insert(op);
    return WalkResult::skip();
  });
  SymbolTableCollection symbolTable;
  for (unsigned i = 0; i < roots.size(); ++i) {
    Operation *root = roots[i];
    root->walk([&](CallOpInterface call) {
      Operation *callee = call.resolveCallableInTable(&symbolTable);
      assert(callee && "verified Gluon calls must resolve to a function");
      roots.insert(callee);
    });
  }
  return roots;
}

class OptimizeThreadRegions
    : public impl::TritonGPUOptimizeThreadRegionsBase<OptimizeThreadRegions> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // Callback scalars vary per element, including in their callees.
    auto tensorCallbackOps = getTensorCallbackOps(mod);
    SmallVector<ThreadHandoffOp> handoffs;
    mod.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
      if (tensorCallbackOps.contains(op) || guardDomain(op))
        return WalkResult::skip();
      for (Region &region : op->getRegions())
        for (Block &block : region)
          optimizeBlock(&block);
      if (auto handoff = dyn_cast<ThreadHandoffOp>(op))
        handoffs.push_back(handoff);
      return WalkResult::advance();
    });
    // Make eligible consumers of handoffs execute on fewer threads.
    restrictHandoffConsumers(handoffs);
    sinkRegionInputs(mod);
  }
};
} // namespace
} // namespace mlir::triton::gpu
