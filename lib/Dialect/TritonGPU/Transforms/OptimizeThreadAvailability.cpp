#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUOPTIMIZETHREADAVAILABILITY
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {
using Domain = ThreadAvailabilityAttr;
using AvailabilityMap = DenseMap<Value, Domain>;
using Materializations = llvm::SmallPtrSet<Operation *, 16>;

Domain available(Value v, const AvailabilityMap &availability) {
  auto it = availability.find(v);
  return it == availability.end() ? gpu::getThreadAvailability(v.getType())
                                  : it->second;
}

Domain executionDomain(Operation *op) {
  for (auto *parent = op->getParentOp(); parent; parent = parent->getParentOp())
    if (auto domain = parent->getAttrOfType<Domain>("ttg.execution_domain"))
      return domain;
  return Domain::get(op->getContext(), 0, 0, 0);
}

Value materialize(OpBuilder &b, Location loc, Value value, Domain domain,
                  AvailabilityMap &availability, Materializations &casts) {
  auto op = UnrealizedConversionCastOp::create(
      b, loc, TypeRange{value.getType()}, ValueRange{value});
  availability[op.getResult(0)] = domain;
  casts.insert(op);
  return op.getResult(0);
}

Domain join(Domain a, Domain b) {
  // The smallest coordinate subspace containing both sets of demanded copies.
  return Domain::get(a.getContext(), a.getLane() & b.getLane(),
                     a.getWarp() & b.getWarp(), a.getBlock() & b.getBlock());
}
bool isFull(Domain a) { return !(a.getLane() | a.getWarp() | a.getBlock()); }

bool isLocalArithmetic(Operation *op) {
  // Speculatable operations can still produce poison (e.g. oversized shifts).
  // Such poison must not reach a side-effect predicate on an inactive copy.
  if (isa<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp, arith::FPToSIOp,
          arith::FPToUIOp, arith::DivSIOp, arith::DivUIOp, arith::RemSIOp,
          arith::RemUIOp>(op))
    return false;
  if (auto overflow = dyn_cast<arith::ArithIntegerOverflowFlagsInterface>(op);
      overflow && overflow.getOverflowAttr().getValue() !=
                      arith::IntegerOverflowFlags::none)
    return false;
  if (auto fastmath = dyn_cast<arith::ArithFastMathInterface>(op);
      fastmath &&
      fastmath.getFastMathFlagsAttr().getValue() != arith::FastMathFlags::none)
    return false;
  if (auto nonneg = dyn_cast<arith::ArithNonNegFlagInterface>(op);
      nonneg && nonneg.getNonNeg())
    return false;
  // Speculation must be safe on inactive copies too. In particular, division,
  // calls and inline assembly must not consume unavailable values.
  return op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
         isSpeculatable(op) &&
         (op->getDialect()->getNamespace() == "arith" ||
          isa<SplatOp, AddPtrOp>(op));
}

bool isScalar(Type t) { return t.isIntOrFloat(); }
bool isUnmasked(AtomicLoadOp load) {
  return !load.getMask() || matchPattern(load.getMask(), m_One());
}

// A scalar protocol can include writes and atomics as long as every operation
// and every branch belongs to the same canonical thread. Tensor accesses,
// asynchronous operations, calls and unknown effects terminate these regions.
// Entry/exit rendezvous carry the block's memory ordering across the region;
// barriers entirely inside it have no other participating thread to order.
bool isScalarProtocolOperation(Operation *root) {
  bool eligible = true;
  root->walk([&](Operation *op) {
    auto scalar = [](Type type) {
      return isScalar(type) || isa<PointerType>(type);
    };
    if (!llvm::all_of(op->getOperandTypes(), scalar) ||
        !llvm::all_of(op->getResultTypes(), scalar)) {
      eligible = false;
      return;
    }
    if (isa<scf::IfOp, scf::WhileOp, scf::ForOp, scf::ConditionOp, scf::YieldOp,
            BarrierOp>(op))
      return;
    if (auto load = dyn_cast<LoadOp>(op)) {
      eligible &= !load.getIsVolatile();
      return;
    }
    if (isa<StoreOp, AtomicLoadOp, AtomicStoreOp, AtomicCASOp, AtomicRMWOp>(op))
      return;
    eligible &= op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
                (op->getDialect()->getNamespace() == "arith" ||
                 isa<AddPtrOp, PtrToIntOp, IntToPtrOp, BitcastOp>(op));
  });
  return eligible;
}

void isolateScalarProtocols(ModuleOp module, AvailabilityMap &availability,
                            Materializations &casts) {
  SmallVector<Block *> blocks;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        blocks.push_back(&block);
  });
  for (Block *block : blocks) {
    if (block->empty() || !isFull(executionDomain(&block->front())))
      continue;
    SmallVector<SmallVector<Operation *>> groups(1);
    for (Operation &op : *block) {
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          !isScalarProtocolOperation(&op)) {
        if (!groups.back().empty())
          groups.emplace_back();
        continue;
      }
      groups.back().push_back(&op);
    }
    for (auto &group : groups) {
      if (group.empty())
        continue;
      bool hasAtomic = false, hasAtomicWrite = false, hasBarrier = false;
      bool hasControl = false;
      DenseSet<Operation *> members;
      SmallVector<Operation *> barriers;
      for (Operation *root : group)
        root->walk([&](Operation *op) {
          members.insert(op);
          hasAtomic |=
              isa<AtomicLoadOp, AtomicStoreOp, AtomicCASOp, AtomicRMWOp>(op);
          hasAtomicWrite |= isa<AtomicStoreOp, AtomicCASOp, AtomicRMWOp>(op);
          hasControl |= isa<scf::WhileOp, scf::ForOp, scf::IfOp>(op);
          if (isa<BarrierOp>(op)) {
            hasBarrier = true;
            barriers.push_back(op);
          }
        });
      // Keep simple scalar load/arithmetic/store sequences on the cheaper
      // predicated path. Read-only polls are handled separately below.
      if (!hasAtomic || !(hasBarrier || (hasAtomicWrite && hasControl)))
        continue;

      SmallVector<Value> outputs;
      SmallVector<Type> types;
      SmallVector<SmallVector<OpOperand *>> uses;
      for (Operation *root : group)
        for (Value value : root->getResults()) {
          SmallVector<OpOperand *> external;
          for (OpOperand &use : value.getUses())
            if (!members.contains(use.getOwner()))
              external.push_back(&use);
          if (!external.empty()) {
            outputs.push_back(value);
            types.push_back(value.getType());
            uses.push_back(std::move(external));
          }
        }
      OpBuilder b(group.front());
      Location loc = group.front()->getLoc();
      Domain owner = getOwnerAvailability(b.getI32Type(), group.front());
      BarrierOp::create(b, loc, AddrSpace::All);
      auto pred = ThreadPredicateOp::create(b, loc, b.getI1Type(), owner);
      auto guard = scf::IfOp::create(b, loc, types, pred, true);
      guard->setAttr("ttg.execution_domain", owner);
      auto *thenBlock = &guard.getThenRegion().front();
      auto *elseBlock = &guard.getElseRegion().front();
      if (!thenBlock->empty())
        thenBlock->back().erase();
      if (!elseBlock->empty())
        elseBlock->back().erase();
      for (auto [index, result] : llvm::enumerate(guard.getResults())) {
        availability[result] = owner;
        for (OpOperand *use : uses[index])
          use->set(result);
      }
      for (Operation *op : group)
        op->moveBefore(thenBlock, thenBlock->end());
      for (Operation *barrier : barriers)
        barrier->erase();
      guard.getThenRegion().walk([&](Operation *op) {
        if (isa<AtomicLoadOp, AtomicStoreOp, AtomicCASOp, AtomicRMWOp>(op)) {
          op->setAttr("ttg.thread_local", b.getUnitAttr());
          for (Value value : op->getResults())
            availability[value] = owner;
        }
      });
      b.setInsertionPointToEnd(thenBlock);
      scf::YieldOp::create(b, loc, outputs);
      b.setInsertionPointToEnd(elseBlock);
      SmallVector<Value> inactive;
      for (Type type : types) {
        Value zero;
        if (isa<PointerType>(type)) {
          Value bits = arith::ConstantIntOp::create(b, loc, 0, 64);
          zero = IntToPtrOp::create(b, loc, type, bits);
        } else {
          zero = arith::ConstantOp::create(b, loc, type, b.getZeroAttr(type));
        }
        inactive.push_back(zero);
      }
      scf::YieldOp::create(b, loc, inactive);
      b.setInsertionPointAfter(guard);
      BarrierOp::create(b, loc, AddrSpace::All);
      for (Value result : guard.getResults()) {
        Value converted = materialize(b, loc, result,
                                      Domain::get(module.getContext(), 0, 0, 0),
                                      availability, casts);
        result.replaceAllUsesExcept(converted, converted.getDefiningOp());
      }
    }
  }
}

// Isolate complete scalar polling loops. No side effects other than atomic
// reads are admitted, so the original per-load acquire handoff can be deferred
// until the whole loop completes. The per-load acquire fences remain in place.
void isolatePollingLoops(ModuleOp module, AvailabilityMap &availability,
                         Materializations &casts) {
  SmallVector<scf::WhileOp> loops;
  module.walk([&](scf::WhileOp loop) { loops.push_back(loop); });
  for (scf::WhileOp loop : llvm::reverse(loops)) {
    if (!isFull(executionDomain(loop)) || lookupNumCTAs(loop) != 1 ||
        !llvm::all_of(loop.getResultTypes(), isScalar))
      continue;
    bool hasAtomic = false;
    bool eligible = true;
    loop.walk([&](Operation *op) {
      if (op == loop.getOperation() || isa<scf::ConditionOp, scf::YieldOp>(op))
        return;
      if (auto branch = dyn_cast<scf::IfOp>(op)) {
        // The condition is scalar in the logical block program. Both branches
        // are visited below, so each must satisfy the same read-only contract.
        eligible &= branch.getCondition().getType().isInteger(1) &&
                    llvm::all_of(branch.getResultTypes(), isScalar);
        return;
      }
      if (auto load = dyn_cast<AtomicLoadOp>(op)) {
        hasAtomic = true;
        eligible &= !isa<RankedTensorType>(load.getType()) && isUnmasked(load);
      } else {
        eligible &= isLocalArithmetic(op) &&
                    llvm::none_of(op->getOperandTypes(),
                                  [](Type t) { return isa<ShapedType>(t); }) &&
                    llvm::all_of(op->getResultTypes(), [](Type type) {
                      return isScalar(type) || isa<PointerType>(type);
                    });
      }
    });
    if (!eligible || !hasAtomic)
      continue;

    OpBuilder b(loop);
    auto loc = loop.getLoc();
    Domain owner = getOwnerAvailability(b.getI32Type(), loop);
    // Include an adjacent initial poll in the same completion obligation.
    // Its only consumer is the loop, and nothing can observe its acquire
    // between this load and the region's final rendezvous.
    if (auto initial = dyn_cast_or_null<AtomicLoadOp>(loop->getPrevNode());
        initial && initial.getResult().hasOneUse() && isUnmasked(initial) &&
        !isa<RankedTensorType>(initial.getType()) &&
        (initial.getResult().use_begin()->getOwner() == loop.getOperation() ||
         loop->isProperAncestor(initial.getResult().use_begin()->getOwner()))) {
      initial->setAttr("ttg.thread_local", b.getUnitAttr());
      availability[initial.getResult()] = owner;
    }
    auto pred = ThreadPredicateOp::create(b, loc, b.getI1Type(), owner);
    auto guard = scf::IfOp::create(b, loc, loop.getResultTypes(), pred, true);
    guard->setAttr("ttg.execution_domain", owner);
    for (Value result : guard.getResults())
      availability[result] = owner;
    for (Value result : loop.getResults())
      availability[result] = owner;
    for (Region &region : loop->getRegions())
      for (BlockArgument arg : region.front().getArguments())
        availability[arg] = owner;
    // Result replacement is performed before the original loop is nested.
    SmallVector<OpOperand *> uses;
    for (Value result : loop.getResults())
      for (OpOperand &use : result.getUses())
        uses.push_back(&use);
    for (OpOperand *use : uses)
      use->set(guard.getResult(cast<OpResult>(use->get()).getResultNumber()));
    auto *thenBlock = &guard.getThenRegion().front();
    auto *elseBlock = &guard.getElseRegion().front();
    // Result-bearing scf.if builders leave the blocks unterminated.
    if (!thenBlock->empty())
      thenBlock->back().erase();
    if (!elseBlock->empty())
      elseBlock->back().erase();
    loop->moveBefore(thenBlock, thenBlock->end());
    b.setInsertionPointToEnd(thenBlock);
    scf::YieldOp::create(b, loc, loop.getResults());
    b.setInsertionPointToEnd(elseBlock);
    SmallVector<Value> inactive;
    for (Type type : guard.getResultTypes()) {
      Value zero = arith::ConstantOp::create(b, loc, type, b.getZeroAttr(type));
      availability[zero] = owner;
      inactive.push_back(zero);
    }
    scf::YieldOp::create(b, loc, inactive);
    loop.walk([&](AtomicLoadOp load) {
      load->setAttr("ttg.thread_local", b.getUnitAttr());
      availability[load.getResult()] = owner;
    });
    loop.walk([&](scf::IfOp branch) {
      for (Value result : branch.getResults())
        availability[result] = owner;
    });
    b.setInsertionPointAfter(guard);
    // This rendezvous preserves completion and acquired visibility even when
    // the loop has no live-out payload. It remains outside divergent control.
    BarrierOp::create(b, loc, AddrSpace::All);
    for (Value result : guard.getResults()) {
      if (result.use_empty())
        continue;
      Value converted =
          materialize(b, loc, result, Domain::get(module.getContext(), 0, 0, 0),
                      availability, casts);
      result.replaceAllUsesExcept(converted, converted.getDefiningOp());
    }
  }
}

// A scalar uses a rank-zero linear layout. Its lane and warp bases have no
// output components, retaining the execution domain and one logical element.
RankedTensorType layoutType(Type original, Domain domain, Operation *context) {
  auto *ctx = context->getContext();
  Attribute placement;
  Type element;
  SmallVector<int64_t> shape;
  if (auto tensor = dyn_cast<RankedTensorType>(original)) {
    placement = getPlacementEncoding(tensor.getEncoding());
    element = tensor.getElementType();
    shape.assign(tensor.getShape().begin(), tensor.getShape().end());
  } else {
    element = original;
    auto blocked = BlockedEncodingAttr::get(
        ctx, ArrayRef<unsigned>{1},
        ArrayRef<unsigned>{unsigned(TritonGPUDialect::getThreadsPerWarp(
            context->getParentOfType<ModuleOp>()))},
        ArrayRef<unsigned>{unsigned(lookupNumWarps(context))},
        ArrayRef<unsigned>{0}, CGAEncodingAttr::get1CTALayout(ctx, 1));
    auto scalar = toLinearLayout(ArrayRef<int64_t>{1}, blocked)
                      .squeezeOuts(StringAttr::get(ctx, "dim0"));
    placement = LinearEncodingAttr::get(ctx, std::move(scalar));
  }
  Attribute encoding = isFull(domain)
                           ? placement
                           : PartialEncodingAttr::get(ctx, placement, domain);
  return RankedTensorType::get(shape, element, encoding);
}

// This bridge changes only representation. Expanding availability is always an
// explicit convert_layout, including when both values have the same placement.
Value adaptValue(OpBuilder &b, Location loc, Value value, Type expected,
                 Domain execution, Operation *context) {
  if (value.getType() == expected)
    return value;
  auto src = dyn_cast<RankedTensorType>(value.getType());
  auto dst = dyn_cast<RankedTensorType>(expected);
  if (!dst) {
    assert(src && src.getRank() == 0);
    auto needed = layoutType(src, execution, context);
    if (!availabilityCovers(src, needed))
      value = ConvertLayoutOp::create(b, loc, needed, value);
    return ExtractScalarOp::create(b, loc, expected, value, execution);
  }
  if (!src)
    return triton::SplatOp::create(b, loc, dst, value);
  if (src.getRank() == 0 && dst.getRank() != 0) {
    auto needed = layoutType(src, gpu::getThreadAvailability(dst), context);
    if (!availabilityCovers(src, needed))
      value = ConvertLayoutOp::create(b, loc, needed, value);
    return ScalarSplatOp::create(b, loc, dst, value);
  }
  return ConvertLayoutOp::create(b, loc, dst, value);
}

void rewriteLayoutTypes(ModuleOp module, const AvailabilityMap &availability,
                        const Materializations &casts) {
  DenseMap<Value, Type> original;
  SmallVector<Operation *> ops;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    ops.push_back(op);
    for (Value value : op->getResults())
      original[value] = value.getType();
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          original[arg] = arg.getType();
  });
  for (auto [value, domain] : availability) {
    Operation *context = value.getDefiningOp();
    if (!context)
      context = value.getParentRegion()->getParentOp();
    value.setType(layoutType(original.lookup(value), domain, context));
  }
  for (Operation *op : ops) {
    OpBuilder b(op);
    auto loc = op->getLoc();
    auto domain = executionDomain(op);
    auto adapt = [&](unsigned index, Type expected) {
      Value value =
          adaptValue(b, loc, op->getOperand(index), expected, domain, op);
      op->setOperand(index, value);
    };
    if (casts.contains(op)) {
      auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
      Value converted = op->getOperand(0);
      if (converted.getType() != resultType)
        converted = ConvertLayoutOp::create(b, loc, resultType, converted);
      original[converted] = original.lookup(op->getResult(0));
      op->getResult(0).replaceAllUsesWith(converted);
      op->erase();
      continue;
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      if (auto tensor = dyn_cast<RankedTensorType>(constant.getType());
          tensor && original.lookup(constant.getResult()) != tensor) {
        Attribute value = constant.getValue();
        if (auto dense = dyn_cast<DenseElementsAttr>(value))
          constant.setValueAttr(dense.reshape(tensor));
        else
          constant.setValueAttr(
              DenseElementsAttr::get(tensor, ArrayRef<Attribute>{value}));
      }
      continue;
    }
    if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      if (isa<RankedTensorType>(splat.getSrc().getType())) {
        Value value =
            adaptValue(b, loc, splat.getSrc(), splat.getType(), domain, op);
        original[value] = original.lookup(splat.getResult());
        splat.getResult().replaceAllUsesWith(value);
        op->erase();
      }
      continue;
    }
    if (auto loop = dyn_cast<scf::WhileOp>(op)) {
      for (unsigned i = 0; i < loop.getNumOperands(); ++i)
        adapt(i, loop.getBefore().front().getArgument(i).getType());
      continue;
    }
    if (auto condition = dyn_cast<scf::ConditionOp>(op)) {
      adapt(0, b.getI1Type());
      auto loop = cast<scf::WhileOp>(op->getParentOp());
      for (unsigned i = 1; i < op->getNumOperands(); ++i)
        adapt(i, loop.getResult(i - 1).getType());
      continue;
    }
    if (auto yield = dyn_cast<scf::YieldOp>(op)) {
      auto *parent = op->getParentOp();
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        Type expected;
        if (auto loop = dyn_cast<scf::WhileOp>(parent))
          expected = loop.getBefore().front().getArgument(i).getType();
        else
          expected = parent->getResult(i).getType();
        adapt(i, expected);
      }
      continue;
    }
    if (isa<triton::StoreOp, AtomicStoreOp>(op)) {
      bool changed = llvm::any_of(op->getOperands(), [&](Value v) {
        return original.count(v) && original.lookup(v) != v.getType();
      });
      if (changed) {
        auto owner =
            getOwnerAvailability(original.lookup(op->getOperand(0)), op);
        for (unsigned i = 0; i < op->getNumOperands(); ++i)
          adapt(i, layoutType(original.lookup(op->getOperand(i)), owner, op));
        continue;
      }
    }
    if ((isLocalArithmetic(op) ||
         isa<LoadOp, AtomicLoadOp, AtomicCASOp, AtomicRMWOp>(op)) &&
        op->getNumResults() == 1 &&
        original.lookup(op->getResult(0)) != op->getResult(0).getType()) {
      auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
      for (unsigned i = 0; i < op->getNumOperands(); ++i) {
        Type element = getElementTypeOrSelf(original.lookup(op->getOperand(i)));
        adapt(i, RankedTensorType::get(resultTy.getShape(), element,
                                       resultTy.getEncoding()));
      }
      continue;
    }
    // Unknown users keep their original contract. If necessary, make all
    // copies available before unwrapping a rank-zero scalar for that user.
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      auto it = original.find(op->getOperand(i));
      if (it != original.end())
        adapt(i, it->second);
    }
  }
}

class OptimizeThreadAvailability
    : public impl::TritonGPUOptimizeThreadAvailabilityBase<
          OptimizeThreadAvailability> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (module->hasAttr("ttg.thread_availability_planned"))
      return;
    // Cluster and instrumentation contracts are deliberately left to their
    // existing lowering until availability is integrated with those domains.
    if (TritonGPUDialect::getNumCTAs(module) != 1)
      return;
    AvailabilityMap availability;
    Materializations casts;
    isolateScalarProtocols(module, availability, casts);
    isolatePollingLoops(module, availability, casts);
    module->setAttr("ttg.thread_availability_planned",
                    UnitAttr::get(module.getContext()));
    DenseMap<Value, Domain> demand;
    SmallVector<Value> worklist;
    auto full = Domain::get(module.getContext(), 0, 0, 0);
    auto require = [&](Value value, Domain requested, Operation *context) {
      if (!value.getType().isIntOrFloat() &&
          !isa<PointerType, RankedTensorType>(value.getType()))
        return;
      // Restrict only redundant coordinates: every tensor element retains an
      // available representative. This also handles splats and slices.
      Domain owner = getOwnerAvailability(value.getType(), context);
      requested = join(requested, owner);
      auto [it, inserted] = demand.try_emplace(value, requested);
      Domain merged = inserted ? requested : join(it->second, requested);
      if (inserted || merged != it->second) {
        it->second = merged;
        worklist.push_back(value);
      }
    };
    // Unknown users and ordinary control flow require full availability.
    module.walk([&](Operation *op) {
      if (auto load = dyn_cast<AtomicLoadOp>(op)) {
        for (Value operand : op->getOperands())
          require(operand, getOwnerAvailability(load.getType(), op), op);
        return;
      }
      if (auto load = dyn_cast<LoadOp>(op)) {
        // An unused load can still execute. Keep its address and mask valid.
        if (load.getResult().use_empty())
          for (Value operand : op->getOperands())
            require(operand, full, op);
        return;
      }
      if (isLocalArithmetic(op) || casts.contains(op))
        return;
      Domain needed = full;
      if (auto store = dyn_cast<StoreOp>(op)) {
        needed = getOwnerAvailability(store.getPtr().getType(), op);
        if (store.getIgnoreCta())
          needed = Domain::get(module.getContext(), needed.getLane(),
                               needed.getWarp(), 0);
      }
      if (auto store = dyn_cast<AtomicStoreOp>(op))
        needed = getOwnerAvailability(store.getPtr().getType(), op);
      if (auto rmw = dyn_cast<AtomicRMWOp>(op))
        needed = getOwnerAvailability(rmw.getPtr().getType(), op);
      if (auto cas = dyn_cast<AtomicCASOp>(op))
        needed = getOwnerAvailability(cas.getPtr().getType(), op);
      for (auto *parent = op->getParentOp(); parent;
           parent = parent->getParentOp())
        if (auto domain =
                parent->getAttrOfType<Domain>("ttg.execution_domain")) {
          needed = domain;
          break;
        }
      for (Value operand : op->getOperands())
        require(operand, needed, op);
    });
    while (!worklist.empty()) {
      Value value = worklist.pop_back_val();
      Operation *op = value.getDefiningOp();
      if (!op)
        continue;
      Domain needed = demand.lookup(value);
      if (casts.contains(op))
        needed = getOwnerAvailability(value.getType(), op);
      else if (auto load = dyn_cast<AtomicLoadOp>(op))
        needed = getOwnerAvailability(load.getType(), op);
      else if (auto load = dyn_cast<LoadOp>(op)) {
        if (!isa<RankedTensorType>(load.getType()) && !load.getIsVolatile() &&
            !isFull(needed))
          needed = getOwnerAvailability(load.getType(), op);
        else
          needed = full;
      } else if (!isLocalArithmetic(op)) {
        continue;
      }
      for (Value operand : op->getOperands())
        require(operand, needed, op);
    }
    SmallVector<Operation *> producers;
    module.walk([&](Operation *op) {
      if (isa<LoadOp, AtomicLoadOp>(op) || casts.contains(op))
        producers.push_back(op);
    });
    for (Operation *op : producers) {
      Value result = op->getResult(0);
      Domain needed = demand.count(result) ? demand.lookup(result) : full;
      if (casts.contains(op)) {
        availability[result] = needed;
        continue;
      }
      if (op->hasAttr("ttg.thread_local"))
        continue;
      // Keep volatile and tensor loads on their existing path. Atoms can
      // always expose their canonical representatives without repeating them.
      if (auto load = dyn_cast<LoadOp>(op)) {
        if (load.getIsVolatile() || isa<RankedTensorType>(load.getType()) ||
            isFull(needed))
          continue;
      }
      Domain owner = getOwnerAvailability(result.getType(), op);
      if (isFull(needed) || owner == full)
        continue;
      availability[result] = owner;
      if (needed == owner || result.use_empty())
        continue;
      OpBuilder b(op);
      b.setInsertionPointAfter(op);
      Value converted =
          materialize(b, op->getLoc(), result, needed, availability, casts);
      result.replaceAllUsesExcept(converted, converted.getDefiningOp());
    }
    // Publish forward guarantees for straight-line intermediates as well as
    // producers. The planner has already expanded operands where a splat or
    // another consumer needs additional copies. Unknown control-flow results
    // remain fully replicated, except for the explicit isolated regions.
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!isLocalArithmetic(op) || op->getNumResults() != 1)
        return;
      uint32_t lane = 0, warp = 0, block = 0;
      for (Value operand : op->getOperands()) {
        auto a = available(operand, availability);
        lane |= a.getLane();
        warp |= a.getWarp();
        block |= a.getBlock();
      }
      auto domain = executionDomain(op);
      lane |= domain.getLane();
      warp |= domain.getWarp();
      block |= domain.getBlock();
      if (lane || warp || block)
        availability[op->getResult(0)] =
            Domain::get(module.getContext(), lane, warp, block);
    });
    rewriteLayoutTypes(module, availability, casts);
  }
};
} // namespace
} // namespace mlir::triton::gpu
