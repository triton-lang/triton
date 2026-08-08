#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
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

// The chain of operations is:
//  barrier=alloc -> init(barrier) -> wait(barrier) -> inval(barrier)
struct BarrierLifecycle {
  Value barrier;
  ttg::LocalAllocOp alloc;
  SmallVector<ttng::InitBarrierOp> inits;
  SmallVector<ttng::WaitBarrierOp> waits;
  Value initialPhase;
  SmallVector<ttng::InvalBarrierOp> invals;
};

// We declare a new class here to track alias instead of allocation's alias
// buffer id sets because memory allocation hasn't been done yet when this pass
// is run.
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

  // Collect a lifecycle only when:
  // - No barrier alias has an opaque use.
  // - Exactly one init, one inval, and at least one wait
  // - Every wait uses the same constant-zero phase.
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
        if (init.getAlloc() == barrier)
          lifecycle.inits.push_back(init);
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

    if (lifecycle.waits.empty() || lifecycle.invals.size() != 1 ||
        lifecycle.inits.size() != 1)
      return failure();

    lifecycle.initialPhase = lifecycle.waits.front().getPhase();
    if (!matchPattern(lifecycle.initialPhase, m_Zero()))
      return failure();

    for (ttng::WaitBarrierOp wait : lifecycle.waits)
      if (wait.getPhase() != lifecycle.initialPhase)
        return failure();

    return success();
  }

  void getEnclosingLoops(Operation *op,
                         SmallVectorImpl<LoopLikeOpInterface> &loops) {
    for (op = op->getParentOp(); op; op = op->getParentOp()) {
      if (!isa<scf::ForOp, scf::WhileOp>(op))
        continue;
      loops.push_back(cast<LoopLikeOpInterface>(op));
    }
    std::reverse(loops.begin(), loops.end());
  }

  void moveInitialPhaseBeforeLoop(Value initialPhase,
                                  LoopLikeOpInterface loop) {
    Operation *loopOp = loop.getOperation();
    Operation *def = initialPhase.getDefiningOp();
    if (!def)
      return;
    if (def->getBlock() == loopOp->getBlock() && def->isBeforeInBlock(loopOp))
      return;
    def->moveBefore(loopOp);
  }

  Block *getLoopBodyBlock(LoopLikeOpInterface loop) {
    return &loop.getLoopRegions().back()->front();
  }

  Value getLoopResultPhase(LoopLikeOpInterface loop) {
    return loop->getResults().back();
  }

  Value createPhaseAdvance(ttng::WaitBarrierOp wait, Value phase,
                           Value phaseOne) {
    builder.setInsertionPointAfter(wait);
    // A hoisted loop-local barrier alternates between phase 0 and 1 when the
    // lifecycle completes. The phase is consumed by wait_barrier, so advance it
    // next to that wait and under the wait predicate when the wait is
    // predicated.
    Value nextPhase =
        arith::XOrIOp::create(builder, wait.getLoc(), phase, phaseOne);
    Value pred = wait.getPred();
    if (pred && !matchPattern(pred, m_One()))
      nextPhase = arith::SelectOp::create(builder, wait.getLoc(), pred,
                                          nextPhase, phase);
    return nextPhase;
  }

  void appendToYieldAndReplace(scf::YieldOp yield, Value value) {
    SmallVector<Value> operands(yield->getOperands());
    operands.push_back(value);
    builder.setInsertionPoint(yield);
    scf::YieldOp::create(builder, yield.getLoc(), operands);
    yield->erase();
  }

  Value rewriteIfPhase(scf::IfOp ifOp, Value phase, Value phaseOne,
                       const llvm::SmallPtrSetImpl<Operation *> &waits) {
    Value thenPhase =
        rewriteBlockPhases(ifOp.thenBlock(), phase, phaseOne, waits);
    Value elsePhase = phase;
    if (ifOp.elseBlock())
      elsePhase = rewriteBlockPhases(ifOp.elseBlock(), phase, phaseOne, waits);

    // This if does not contain a tracked wait in either branch, so it does not
    // need to yield an updated phase.
    if (thenPhase == phase && elsePhase == phase)
      return phase;

    scf::IfOp newIfOp =
        mlir::replaceIfOpWithNewSignature(builder, ifOp, phase.getType());
    appendToYieldAndReplace(newIfOp.thenYield(), thenPhase);
    appendToYieldAndReplace(newIfOp.elseYield(), elsePhase);
    builder.eraseOp(ifOp);
    return newIfOp.getResults().back();
  }

  Value rewriteBlockPhases(Block *block, Value phase, Value phaseOne,
                           const llvm::SmallPtrSetImpl<Operation *> &waits) {
    for (Operation &op :
         llvm::make_early_inc_range(block->without_terminator())) {
      if (waits.contains(&op)) {
        auto wait = cast<ttng::WaitBarrierOp>(op);
        wait.getPhaseMutable().assign(phase);
        phase = createPhaseAdvance(wait, phase, phaseOne);
        continue;
      }

      if (auto ifOp = dyn_cast<scf::IfOp>(op))
        phase = rewriteIfPhase(ifOp, phase, phaseOne, waits);
    }
    return phase;
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

  void appendToLoopYield(LoopLikeOpInterface loop, Value phase) {
    if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation())) {
      mlir::appendToForOpYield(forOp, phase);
      return;
    }
    appendToWhileYield(cast<scf::WhileOp>(loop.getOperation()), phase);
  }

  // First move the phase initialization op before the outermost loop,
  // then add a phase argument to each loop,
  // xor the phase after each wait,
  // and in the end thread the updated phase through the loop yields.
  LogicalResult rewriteLoopPhases(BarrierLifecycle &lifecycle) {
    if (llvm::none_of(lifecycle.invals, [&](ttng::InvalBarrierOp inval) {
          SmallVector<LoopLikeOpInterface> loops;
          getEnclosingLoops(inval, loops);
          return !loops.empty();
        }))
      // Only loop-local invalidations require phase threading. If every
      // invalidation is already outside loops, moving them to function exits
      // does not change the phase seen by repeated loop iterations.
      return success();

    SmallVector<LoopLikeOpInterface> loops;
    getEnclosingLoops(lifecycle.invals.front(), loops);

    llvm::SmallPtrSet<Operation *, 4> waits;
    for (ttng::WaitBarrierOp wait : lifecycle.waits)
      waits.insert(wait);

    OpBuilder::InsertionGuard guard(builder);
    moveInitialPhaseBeforeLoop(lifecycle.initialPhase, loops.front());

    builder.setInsertionPoint(loops.front());
    Value phaseOne =
        arith::ConstantIntOp::create(builder, loops.front()->getLoc(), 1, 32);

    SmallVector<std::pair<LoopLikeOpInterface, Value>> loopPhases;
    Value initialPhase = lifecycle.initialPhase;
    for (auto [idx, loop] : llvm::enumerate(loops)) {
      builder.setInsertionPoint(loop);
      Value phase;
      if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation()))
        loop = addForPhaseArg(forOp, initialPhase, phase);
      else
        loop = addWhilePhaseArg(cast<scf::WhileOp>(loop.getOperation()),
                                initialPhase, phase);

      loops[idx] = loop;
      loopPhases.push_back({loop, phase});
      initialPhase = phase;
    }

    LoopLikeOpInterface innerLoop = loopPhases.back().first;
    Value innerPhase = loopPhases.back().second;
    Value nextPhase = rewriteBlockPhases(getLoopBodyBlock(innerLoop),
                                         innerPhase, phaseOne, waits);

    appendToLoopYield(innerLoop, nextPhase);
    Value loopResult = getLoopResultPhase(innerLoop);

    for (int i = loopPhases.size() - 2; i >= 0; --i) {
      LoopLikeOpInterface loop = loopPhases[i].first;
      Value loopPhase = loopPhases[i].second;
      if (loopResult.getParentBlock() != getLoopBodyBlock(loop))
        loopResult = mlir::triton::sinkValueRedefinition(
            builder, loopPhase, loopResult, loopResult.getParentBlock());
      appendToLoopYield(loop, loopResult);
      loopResult = getLoopResultPhase(loop);
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
    lifecycle.inits.front()->moveAfter(lifecycle.alloc);
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
