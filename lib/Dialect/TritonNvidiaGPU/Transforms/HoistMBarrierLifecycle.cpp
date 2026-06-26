#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Analysis/Alias.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUHOISTMBARRIERLIFECYCLEPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

// This pass makes cross-CTA mbarrier initialization visible before users that
// rely on CTA cooperation, such as multicast TMA, 2CTA MMA, and CLC. Those ops
// may observe the barrier from another CTA, so a loop-local inval is too
// expensive when the barrier is initialized inside the producing loop body. The
// pass hoists alloc/init to the function entry, carries
// the wait phase through loops when the lifecycle repeats, and recreates
// invalidations at function exits. For example:
//   scf.for ... {
//     %bar = ttg.local_alloc
//     ttng.init_barrier %bar, 1
//     ttng.barrier_expect %bar, ...
//     ttng.async_tma_copy_global_to_local ... %bar ... {multicast}
//     ttng.wait_barrier %bar, 0
//     ttng.inval_barrier %bar
//   }
// becomes:
//   %bar = ttg.local_alloc
//   ttng.init_barrier %bar, 1
//   scf.for ... iter_args(%phase = 0) {
//     ttng.barrier_expect %bar, ...
//     ttng.async_tma_copy_global_to_local ... %bar ... {multicast}
//     ttng.wait_barrier %bar, %phase
//     %next_phase = arith.xori %phase, 1
//   }
//   ttng.inval_barrier %bar
namespace {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

struct BarrierLifecycle {
  Value barrier;
  ttg::LocalAllocOp alloc;
  ttng::InitBarrierOp init;
  SmallVector<ttng::InitBarrierOp> inits;
  SmallVector<ttng::WaitBarrierOp> waits;
  Value initialPhase;
  SmallVector<ttng::InvalBarrierOp> invals;
};

class BarrierAliases {
public:
  BarrierAliases(Value barrier, SharedMemoryAliasAnalysis &aliasAnalysis)
      : aliasAnalysis(aliasAnalysis) {
    collectAliasRoots(barrier, roots);
  }

  bool contains(Value value) const { return aliasesRoot(value); }

private:
  void collectAliasRoots(Value value, llvm::DenseSet<Value> &values) const {
    auto *lattice = aliasAnalysis.getLatticeElement(value);
    if (!lattice)
      return;
    for (Value alloc : lattice->getValue().getAllocs())
      values.insert(alloc);
  }

  bool aliasesRoot(Value value) const {
    if (!value)
      return false;

    llvm::DenseSet<Value> valueRoots;
    collectAliasRoots(value, valueRoots);
    for (Value root : valueRoots)
      if (roots.contains(root))
        return true;

    return roots.contains(value);
  }

  SharedMemoryAliasAnalysis &aliasAnalysis;
  llvm::DenseSet<Value> roots;
};

class MBarrierLifecycleHoister {
public:
  MBarrierLifecycleHoister(ModuleOp mod, int numCTAs)
      : mod(mod), builder(mod.getContext()), numCTAs(numCTAs) {}

  void run() {
    for (auto funcOp : mod.getOps<FunctionOpInterface>()) {
      std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
      SharedMemoryAliasAnalysis *aliasAnalysis =
          solver->load<SharedMemoryAliasAnalysis>();
      if (failed(solver->initializeAndRun(funcOp)))
        continue;

      llvm::SmallPtrSet<Value, 8> seen;
      SmallVector<Value> candidates;
      funcOp.walk([&](ttng::InitBarrierOp init) {
        Value barrier = init.getAlloc();
        BarrierAliases aliases(barrier, *aliasAnalysis);
        if (requiresCrossCTAMBarrierInitSync(
                funcOp, barrier, numCTAs,
                [&](Value value) { return aliases.contains(value); }) &&
            seen.insert(barrier).second)
          candidates.push_back(barrier);
      });

      for (Value barrier : candidates) {
        BarrierAliases aliases(barrier, *aliasAnalysis);
        BarrierLifecycle lifecycle;
        if (failed(collectLifecycle(funcOp, barrier, aliases, lifecycle)))
          continue;
        if (failed(rewriteLoopPhases(lifecycle)))
          continue;

        moveInitToFunctionEntry(lifecycle, funcOp);
        moveInvalidationToFunctionExits(lifecycle, funcOp);
      }
    }
  }

private:
  bool isKnownBarrierUser(Operation *op, const BarrierAliases &aliases) {
    if (auto iface = dyn_cast<ttg::MBarrierOpInterface>(op)) {
      return llvm::any_of(iface.getBarriers(),
                          [&](Value value) { return aliases.contains(value); });
    }
    // Memdesc views and arith.select only forward the descriptor for the same
    // underlying mbarrier storage. They do not initialize, wait, invalidate, or
    // signal the barrier, so they are transparent to the lifecycle.
    if (op->hasTrait<OpTrait::MemDescViewTrait>() || isa<arith::SelectOp>(op))
      return llvm::any_of(op->getResults(), [&](Value result) {
        return aliases.contains(result);
      });
    return false;
  }

  // "Opaque" means the barrier is passed to an op this pass does not model as
  // an mbarrier user.
  bool hasOpaqueBarrierUse(FunctionOpInterface funcOp,
                           const BarrierAliases &aliases) {
    return funcOp
        ->walk<WalkOrder::PreOrder>([&](Operation *op) {
          bool usesAlias = llvm::any_of(op->getOperands(), [&](Value operand) {
            return aliases.contains(operand);
          });
          if (!usesAlias)
            return WalkResult::advance();
          if (isKnownBarrierUser(op, aliases))
            return WalkResult::advance();
          return WalkResult::interrupt();
        })
        .wasInterrupted();
  }

  LogicalResult collectLifecycle(FunctionOpInterface funcOp, Value barrier,
                                 const BarrierAliases &aliases,
                                 BarrierLifecycle &lifecycle) {
    lifecycle.barrier = barrier;
    lifecycle.alloc = barrier.getDefiningOp<ttg::LocalAllocOp>();
    if (!lifecycle.alloc || lifecycle.alloc->getNumOperands() != 0)
      return failure();

    if (hasOpaqueBarrierUse(funcOp, aliases))
      return failure();

    funcOp.walk([&](Operation *op) {
      if (auto init = dyn_cast<ttng::InitBarrierOp>(op)) {
        if (init.getAlloc() == barrier) {
          if (!lifecycle.init)
            lifecycle.init = init;
          lifecycle.inits.push_back(init);
        }
        return;
      }
      if (auto wait = dyn_cast<ttng::WaitBarrierOp>(op)) {
        if (wait.getAlloc() == barrier)
          lifecycle.waits.push_back(wait);
        return;
      }
      if (auto inval = dyn_cast<ttng::InvalBarrierOp>(op)) {
        if (inval.getAlloc() == barrier)
          lifecycle.invals.push_back(inval);
        return;
      }
    });

    if (!lifecycle.init || lifecycle.waits.empty() || lifecycle.invals.empty())
      return failure();

    for (ttng::InitBarrierOp init : lifecycle.inits)
      if (init.getCount() != lifecycle.init.getCount())
        return failure();

    lifecycle.initialPhase = lifecycle.waits.front().getPhase();
    if (!isa_and_nonnull<arith::ConstantOp>(
            lifecycle.initialPhase.getDefiningOp()))
      return failure();

    for (ttng::WaitBarrierOp wait : lifecycle.waits)
      if (wait.getPhase() != lifecycle.initialPhase)
        return failure();

    return success();
  }

  void getEnclosingLoops(Operation *op, SmallVectorImpl<Operation *> &loops) {
    for (op = op->getParentOp(); op; op = op->getParentOp()) {
      if (!isa<scf::ForOp, scf::WhileOp>(op))
        continue;
      loops.push_back(op);
    }
  }

  void moveInitialPhaseBeforeLoop(Value initialPhase, Operation *loopOp) {
    Operation *def = initialPhase.getDefiningOp();
    if (!def)
      return;
    if (def->getBlock() == loopOp->getBlock() && def->isBeforeInBlock(loopOp))
      return;
    def->moveBefore(loopOp);
  }

  Block *getLoopBodyBlock(Operation *loopOp) {
    if (auto forOp = dyn_cast<scf::ForOp>(loopOp))
      return forOp.getBody();
    if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp))
      return &whileOp.getAfter().front();
    return nullptr;
  }

  Value getLoopResultPhase(Operation *loopOp) {
    return loopOp->getResults().back();
  }

  bool isAvailableAt(Value value, Operation *op) {
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      Block *argBlock = arg.getOwner();
      for (Operation *ancestor = op; ancestor;
           ancestor = ancestor->getParentOp())
        if (ancestor->getBlock() == argBlock)
          return true;
      return argBlock == op->getBlock();
    }

    Operation *def = value.getDefiningOp();
    if (!def)
      return false;

    Operation *ancestor = op;
    while (ancestor && ancestor->getBlock() != def->getBlock())
      ancestor = ancestor->getParentOp();
    return ancestor && def->isBeforeInBlock(ancestor);
  }

  FailureOr<Value> getPhaseAdvancePredicate(ArrayRef<ttng::WaitBarrierOp> waits,
                                            Operation *op) {
    Value pred;
    for (ttng::WaitBarrierOp wait : waits) {
      Value waitPred = wait.getPred();
      if (!waitPred || matchPattern(waitPred, m_One()))
        return Value();
      if (!isAvailableAt(waitPred, op))
        return failure();
      if (!pred) {
        pred = waitPred;
        continue;
      }
      if (pred != waitPred)
        pred = arith::OrIOp::create(builder, op->getLoc(), pred, waitPred);
    }
    return pred;
  }

  FailureOr<Value> createPhaseAdvance(ttng::InvalBarrierOp inval, Value phase,
                                      Value phaseOne,
                                      ArrayRef<ttng::WaitBarrierOp> waits) {
    builder.setInsertionPointAfter(inval);
    FailureOr<Value> pred = getPhaseAdvancePredicate(waits, inval);
    if (failed(pred))
      return failure();

    // A hoisted loop-local barrier alternates between phase 0 and 1 when a
    // lifecycle completes. Predicated waits only consume the phase when their
    // predicate is true, so the carried phase must advance under the same
    // predicate instead of toggling unconditionally.
    Value nextPhase =
        arith::XOrIOp::create(builder, inval.getLoc(), phase, phaseOne);
    if (*pred)
      nextPhase = arith::SelectOp::create(builder, inval.getLoc(), *pred,
                                          nextPhase, phase);
    return nextPhase;
  }

  scf::ForOp addForPhaseArg(scf::ForOp forOp, Value initialPhase,
                            Value &phase) {
    forOp = mlir::addIterArgsToLoop(builder, forOp, initialPhase);
    phase = forOp.getRegionIterArg(forOp.getNumRegionIterArgs() - 1);
    return forOp;
  }

  scf::WhileOp addWhilePhaseArg(scf::WhileOp whileOp, Value initialPhase,
                                Value &phase) {
    whileOp = mlir::addIterArgsToLoop(builder, whileOp, initialPhase);
    phase = whileOp.getAfterArguments().back();
    return whileOp;
  }

  void appendToLoopYield(Operation *loopOp, Value phase) {
    if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
      mlir::appendToForOpYield(forOp, phase);
      return;
    }
    appendToWhileYield(cast<scf::WhileOp>(loopOp), phase);
  }

  LogicalResult rewriteLoopPhases(BarrierLifecycle &lifecycle) {
    SmallVector<Operation *> loops;
    ttng::InvalBarrierOp inval = lifecycle.invals.front();
    getEnclosingLoops(inval, loops);

    for (ttng::InvalBarrierOp otherInval : llvm::drop_begin(lifecycle.invals)) {
      SmallVector<Operation *> otherLoops;
      getEnclosingLoops(otherInval, otherLoops);
      if (!llvm::equal(loops, otherLoops))
        return failure();
    }

    // If invalidation is already outside loops, moving it to function exits
    // does not remove a loop-local phase reset.
    if (loops.empty())
      return success();

    if (lifecycle.invals.size() != 1)
      return failure();

    std::reverse(loops.begin(), loops.end());

    OpBuilder::InsertionGuard guard(builder);
    moveInitialPhaseBeforeLoop(lifecycle.initialPhase, loops.front());

    builder.setInsertionPoint(loops.front());
    Value phaseOne =
        arith::ConstantIntOp::create(builder, loops.front()->getLoc(), 1, 32);

    SmallVector<std::pair<Operation *, Value>> loopPhases;
    Value initialPhase = lifecycle.initialPhase;
    for (auto [idx, loopOp] : llvm::enumerate(loops)) {
      builder.setInsertionPoint(loopOp);
      Value phase;
      if (auto forOp = dyn_cast<scf::ForOp>(loopOp))
        loopOp = addForPhaseArg(forOp, initialPhase, phase).getOperation();
      else
        loopOp =
            addWhilePhaseArg(cast<scf::WhileOp>(loopOp), initialPhase, phase)
                .getOperation();

      loops[idx] = loopOp;
      loopPhases.push_back({loopOp, phase});
      initialPhase = phase;
    }

    Operation *innerLoop = loopPhases.back().first;
    Value innerPhase = loopPhases.back().second;
    for (ttng::WaitBarrierOp wait : lifecycle.waits)
      wait.getPhaseMutable().assign(innerPhase);

    FailureOr<Value> maybeNextPhase =
        createPhaseAdvance(inval, innerPhase, phaseOne, lifecycle.waits);
    if (failed(maybeNextPhase))
      return failure();
    Value nextPhase = *maybeNextPhase;
    if (inval->getBlock() != getLoopBodyBlock(innerLoop))
      nextPhase = mlir::triton::sinkValueRedefinition(
          builder, innerPhase, nextPhase, inval->getBlock());

    appendToLoopYield(innerLoop, nextPhase);
    Value loopResult = getLoopResultPhase(innerLoop);

    for (int i = static_cast<int>(loopPhases.size()) - 2; i >= 0; --i) {
      Operation *loopOp = loopPhases[i].first;
      Value loopPhase = loopPhases[i].second;
      if (loopResult.getParentBlock() != getLoopBodyBlock(loopOp))
        loopResult = mlir::triton::sinkValueRedefinition(
            builder, loopPhase, loopResult, loopResult.getParentBlock());
      appendToLoopYield(loopOp, loopResult);
      loopResult = getLoopResultPhase(loopOp);
    }

    return success();
  }

  void appendToWhileYield(scf::WhileOp whileOp, Value phase) {
    auto yieldOp =
        cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
    SmallVector<Value> operands(yieldOp->getOperands());
    operands.push_back(phase);

    builder.setInsertionPoint(yieldOp);
    scf::YieldOp::create(builder, yieldOp.getLoc(), operands);
    yieldOp->erase();
  }

  void moveInitToFunctionEntry(BarrierLifecycle &lifecycle,
                               FunctionOpInterface funcOp) {
    Block &entry = funcOp->getRegion(0).front();
    lifecycle.alloc->moveBefore(&entry.front());
    lifecycle.init->moveAfter(lifecycle.alloc);
    for (ttng::InitBarrierOp init : lifecycle.inits)
      if (init != lifecycle.init)
        init->erase();
  }

  void moveInvalidationToFunctionExits(BarrierLifecycle &lifecycle,
                                       FunctionOpInterface funcOp) {
    for (ttng::InvalBarrierOp inval : lifecycle.invals)
      inval->erase();

    SmallVector<Operation *> returns;
    funcOp.walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::ReturnLike>() &&
          op->getParentOp() == funcOp.getOperation())
        returns.push_back(op);
    });

    // Once alloc/init move to function entry, the matching invalidation must
    // live on every function exit instead of at the former loop-local position.
    for (Operation *ret : returns) {
      builder.setInsertionPoint(ret);
      ttng::InvalBarrierOp::create(builder, ret->getLoc(), lifecycle.barrier);
    }
  }

  ModuleOp mod;
  IRRewriter builder;
  int numCTAs;
};

class HoistMBarrierLifecyclePass
    : public impl::TritonNvidiaGPUHoistMBarrierLifecyclePassBase<
          HoistMBarrierLifecyclePass> {
public:
  using impl::TritonNvidiaGPUHoistMBarrierLifecyclePassBase<
      HoistMBarrierLifecyclePass>::
      TritonNvidiaGPUHoistMBarrierLifecyclePassBase;

  void runOnOperation() override {
    if (computeCapability < 90)
      return;

    ModuleOp mod = getOperation();
    int numCTAs = ttg::TritonGPUDialect::getNumCTAs(mod);
    if (numCTAs == 1)
      return;

    MBarrierLifecycleHoister(mod, numCTAs).run();
  }
};

} // namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
