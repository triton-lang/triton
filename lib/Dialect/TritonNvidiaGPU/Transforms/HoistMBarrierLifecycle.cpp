#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
  int initCount = 0;
  SmallVector<ttng::WaitBarrierOp> waits;
  ttng::WaitBarrierOp wait;
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
          lifecycle.init = init;
          ++lifecycle.initCount;
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

    if (lifecycle.initCount != 1 || !lifecycle.init ||
        lifecycle.waits.size() != 1 || lifecycle.invals.size() != 1)
      return failure();

    lifecycle.wait = lifecycle.waits.front();
    lifecycle.initialPhase = lifecycle.wait.getPhase();
    if (!isa_and_nonnull<arith::ConstantOp>(
            lifecycle.initialPhase.getDefiningOp()))
      return failure();

    return success();
  }

  LogicalResult getSingleEnclosingLoop(ttng::WaitBarrierOp wait,
                                       Operation *&loopOp) {
    loopOp = nullptr;
    for (Operation *op = wait->getParentOp(); op; op = op->getParentOp()) {
      if (!isa<scf::ForOp, scf::WhileOp>(op))
        continue;
      if (loopOp)
        return failure();
      loopOp = op;
    }
    return success();
  }

  void moveInitialPhaseBeforeLoop(Value initialPhase, Operation *loopOp) {
    Operation *def = initialPhase.getDefiningOp();
    if (!def)
      return;
    if (def->getBlock() == loopOp->getBlock() && def->isBeforeInBlock(loopOp))
      return;
    def->moveBefore(loopOp);
  }

  Value createPhaseAdvance(ttng::WaitBarrierOp wait, Value phase,
                           Value phaseOne) {
    builder.setInsertionPointAfter(wait);
    // A hoisted loop-local barrier alternates between phase 0 and 1 on each
    // completed transaction. Predicated waits only complete when their
    // predicate is true, so preserve the current phase otherwise.
    Value toggledPhase =
        arith::XOrIOp::create(builder, wait.getLoc(), phase, phaseOne);
    if (Value pred = wait.getPred())
      return arith::SelectOp::create(builder, wait.getLoc(), pred, toggledPhase,
                                     phase);
    return toggledPhase;
  }

  LogicalResult rewriteForPhase(BarrierLifecycle &lifecycle, scf::ForOp forOp) {
    OpBuilder::InsertionGuard guard(builder);
    moveInitialPhaseBeforeLoop(lifecycle.initialPhase, forOp);

    builder.setInsertionPoint(forOp);
    Value phaseOne =
        arith::ConstantIntOp::create(builder, forOp.getLoc(), 1, 32);

    forOp = mlir::addIterArgsToLoop(builder, forOp, lifecycle.initialPhase);
    Value phase = forOp.getRegionIterArg(forOp.getNumRegionIterArgs() - 1);
    lifecycle.wait.getPhaseMutable().assign(phase);

    Value nextPhase = createPhaseAdvance(lifecycle.wait, phase, phaseOne);
    if (lifecycle.wait->getBlock() != forOp.getBody())
      nextPhase = mlir::triton::sinkValueRedefinition(
          builder, phase, nextPhase, lifecycle.wait->getBlock());

    mlir::appendToForOpYield(forOp, nextPhase);
    return success();
  }

  void appendToWhileCondition(scf::WhileOp whileOp, Value phase) {
    auto conditionOp =
        cast<scf::ConditionOp>(whileOp.getBefore().front().getTerminator());
    SmallVector<Value> args(conditionOp.getArgs());
    args.push_back(phase);

    builder.setInsertionPoint(conditionOp);
    scf::ConditionOp::create(builder, conditionOp.getLoc(),
                             conditionOp.getCondition(), args);
    conditionOp->erase();
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

  LogicalResult rewriteWhilePhase(BarrierLifecycle &lifecycle,
                                  scf::WhileOp whileOp) {
    OpBuilder::InsertionGuard guard(builder);
    moveInitialPhaseBeforeLoop(lifecycle.initialPhase, whileOp);

    builder.setInsertionPoint(whileOp);
    Value phaseOne =
        arith::ConstantIntOp::create(builder, whileOp.getLoc(), 1, 32);

    scf::WhileOp oldWhileOp = whileOp;
    whileOp = mlir::replaceWhileOpWithNewSignature(
        builder, whileOp, lifecycle.initialPhase,
        lifecycle.initialPhase.getType());

    Value beforePhase = whileOp.getBeforeArguments().back();
    Value afterPhase = whileOp.getAfterArguments().back();
    appendToWhileCondition(whileOp, beforePhase);

    lifecycle.wait.getPhaseMutable().assign(afterPhase);
    Value nextPhase = createPhaseAdvance(lifecycle.wait, afterPhase, phaseOne);
    if (lifecycle.wait->getBlock() != &whileOp.getAfter().front())
      nextPhase = mlir::triton::sinkValueRedefinition(
          builder, afterPhase, nextPhase, lifecycle.wait->getBlock());
    appendToWhileYield(whileOp, nextPhase);

    oldWhileOp->erase();
    return success();
  }

  LogicalResult rewriteLoopPhases(BarrierLifecycle &lifecycle) {
    Operation *loopOp = nullptr;
    if (failed(getSingleEnclosingLoop(lifecycle.wait, loopOp)))
      return failure();

    // Straight-line single-use barriers keep their original constant phase.
    if (!loopOp)
      return success();

    if (auto forOp = dyn_cast<scf::ForOp>(loopOp))
      return rewriteForPhase(lifecycle, forOp);
    if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp))
      return rewriteWhilePhase(lifecycle, whileOp);
    return failure();
  }

  void moveInitToFunctionEntry(BarrierLifecycle &lifecycle,
                               FunctionOpInterface funcOp) {
    Block &entry = funcOp->getRegion(0).front();
    lifecycle.alloc->moveBefore(&entry.front());
    lifecycle.init->moveAfter(lifecycle.alloc);
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
