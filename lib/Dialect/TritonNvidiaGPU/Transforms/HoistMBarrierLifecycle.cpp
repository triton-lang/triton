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
#include <cassert>

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
      : aliasAnalysis(aliasAnalysis),
        roots(
            aliasAnalysis.getLatticeElement(barrier)->getValue().getAllocs()) {}

  bool contains(Value value) const {
    return value &&
           llvm::any_of(
               aliasAnalysis.getLatticeElement(value)->getValue().getAllocs(),
               [&](Value root) { return roots.contains(root); });
  }

private:
  SharedMemoryAliasAnalysis &aliasAnalysis;
  const llvm::DenseSet<Value> &roots;
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
        if (!seen.insert(barrier).second)
          return;
        BarrierAliases aliases(barrier, *aliasAnalysis);
        if (requiresCrossCTAMBarrierInitSync(
                funcOp, barrier, numCTAs,
                [&](Value value) { return aliases.contains(value); }))
          candidates.push_back(barrier);
      });

      for (Value barrier : candidates) {
        BarrierAliases aliases(barrier, *aliasAnalysis);
        BarrierLifecycle lifecycle;
        if (failed(collectLifecycle(funcOp, barrier, aliases, lifecycle)))
          continue;
        rewriteLoopPhases(lifecycle);

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
    lifecycle.alloc = barrier.getDefiningOp<ttg::LocalAllocOp>();
    if (!lifecycle.alloc || lifecycle.alloc->getNumOperands() != 0)
      return failure();

    if (hasOpaqueBarrierUse(funcOp, aliases))
      return failure();

    funcOp.walk([&](ttg::MBarrierOpInterface user) {
      if (!aliases.contains(user.getBarrier()))
        return;
      Operation *op = user.getOperation();
      if (auto init = dyn_cast<ttng::InitBarrierOp>(op))
        lifecycle.inits.push_back(init);
      else if (auto wait = dyn_cast<ttng::WaitBarrierOp>(op))
        lifecycle.waits.push_back(wait);
      else if (auto inval = dyn_cast<ttng::InvalBarrierOp>(op))
        lifecycle.invals.push_back(inval);
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

  Block *getLoopBodyBlock(LoopLikeOpInterface loop, Operation *nested) {
    if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation()))
      return forOp.getBody();
    auto whileOp = cast<scf::WhileOp>(loop.getOperation());
    Region &region = whileOp.getBefore().isAncestor(nested->getParentRegion())
                         ? whileOp.getBefore()
                         : whileOp.getAfter();
    return &region.front();
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
    if (Value pred = wait.getPred(); pred && !matchPattern(pred, m_One()))
      nextPhase = arith::SelectOp::create(builder, wait.getLoc(), pred,
                                          nextPhase, phase);
    return nextPhase;
  }

  bool containsTrackedWait(Block *block,
                           const llvm::SmallPtrSetImpl<Operation *> &waits) {
    for (Operation &op : block->without_terminator()) {
      if (waits.contains(&op))
        return true;
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        if (containsTrackedWait(ifOp.thenBlock(), waits) ||
            (ifOp.elseBlock() && containsTrackedWait(ifOp.elseBlock(), waits)))
          return true;
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (containsTrackedWait(forOp.getBody(), waits))
          return true;
      }
    }
    return false;
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
    newIfOp.thenYield().getResultsMutable().append(thenPhase);
    newIfOp.elseYield().getResultsMutable().append(elsePhase);
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

      if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (!containsTrackedWait(forOp.getBody(), waits))
          continue;
        forOp = mlir::addIterArgsToLoop(builder, forOp, phase);
        Value loopPhase = forOp.getRegionIterArgs().back();
        Value nextPhase =
            rewriteBlockPhases(forOp.getBody(), loopPhase, phaseOne, waits);
        cast<scf::YieldOp>(forOp.getBody()->getTerminator())
            .getResultsMutable()
            .append(nextPhase);
        phase = forOp.getResults().back();
        continue;
      }

      if (auto ifOp = dyn_cast<scf::IfOp>(op))
        phase = rewriteIfPhase(ifOp, phase, phaseOne, waits);
    }
    return phase;
  }

  Value appendToLoopYield(LoopLikeOpInterface loop, Block *body, Value phase) {
    if (auto whileOp = dyn_cast<scf::WhileOp>(loop.getOperation())) {
      if (body == &whileOp.getBefore().front()) {
        // addIterArgsToLoop initially forwards the input phase through the
        // condition. Replace it with the phase updated in the before region,
        // then carry the corresponding after-region argument through its yield.
        auto conditionOp = whileOp.getConditionOp();
        conditionOp->setOperand(conditionOp->getNumOperands() - 1, phase);
        phase = whileOp.getAfterArguments().back();
      }
      body = &whileOp.getAfter().front();
    }
    cast<scf::YieldOp>(body->getTerminator()).getResultsMutable().append(phase);
    return loop->getResults().back();
  }

  // First move the phase initialization op before the outermost loop,
  // then add a phase argument to each loop,
  // xor the phase after each wait,
  // and in the end thread the updated phase through the loop yields.
  void rewriteLoopPhases(BarrierLifecycle &lifecycle) {
    SmallVector<LoopLikeOpInterface> loops;
    getEnclosingLoops(lifecycle.invals.front(), loops);
    if (loops.empty())
      // Only loop-local invalidations require phase threading. If every
      // invalidation is already outside loops, moving them to function exits
      // does not change the phase seen by repeated loop iterations.
      return;

    llvm::SmallPtrSet<Operation *, 4> waits;
    for (ttng::WaitBarrierOp wait : lifecycle.waits)
      waits.insert(wait);

    OpBuilder::InsertionGuard guard(builder);
    moveInitialPhaseBeforeLoop(lifecycle.initialPhase, loops.front());

    builder.setInsertionPoint(loops.front());
    Value phaseOne =
        arith::ConstantIntOp::create(builder, loops.front()->getLoc(), 1, 32);

    Operation *inval = lifecycle.invals.front().getOperation();
    Value phase = lifecycle.initialPhase;
    for (LoopLikeOpInterface &loop : loops) {
      builder.setInsertionPoint(loop);
      if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation()))
        loop = mlir::addIterArgsToLoop(builder, forOp, phase);
      else
        loop = mlir::addIterArgsToLoop(
            builder, cast<scf::WhileOp>(loop.getOperation()), phase);
      phase = getLoopBodyBlock(loop, inval)->getArguments().back();
    }

    LoopLikeOpInterface innerLoop = loops.back();
    Block *innerBody = getLoopBodyBlock(innerLoop, inval);
    Value nextPhase = rewriteBlockPhases(innerBody, phase, phaseOne, waits);

    Value loopResult = appendToLoopYield(innerLoop, innerBody, nextPhase);

    for (int i = loops.size() - 2; i >= 0; --i) {
      LoopLikeOpInterface loop = loops[i];
      Block *body = getLoopBodyBlock(loop, inval);
      Value loopPhase = body->getArguments().back();
      if (loopResult.getParentBlock() != body)
        loopResult = mlir::triton::sinkValueRedefinition(
            builder, loopPhase, loopResult, loopResult.getParentBlock());
      loopResult = appendToLoopYield(loop, body, loopResult);
    }
  }

  void moveInitToFunctionEntry(BarrierLifecycle &lifecycle,
                               FunctionOpInterface funcOp) {
    bool isInIsolatedPartition =
        lifecycle.alloc->getParentOfType<ttg::WarpSpecializePartitionsOp>() !=
        nullptr;
    assert(!isInIsolatedPartition &&
           "cannot hoist mbarrier lifecycle out of an isolated "
           "warp-specialization partition");
    Block &entry = funcOp->getRegion(0).front();
    lifecycle.alloc->moveBefore(&entry.front());
    lifecycle.inits.front()->moveAfter(lifecycle.alloc);
  }

  void moveInvalidationToFunctionExits(BarrierLifecycle &lifecycle,
                                       FunctionOpInterface funcOp) {
    for (ttng::InvalBarrierOp inval : lifecycle.invals)
      inval->erase();

    // Once alloc/init move to function entry, the matching invalidation must
    // live on every function exit instead of at the former loop-local position.
    for (Block &block : funcOp.getFunctionBody()) {
      Operation *ret = block.getTerminator();
      if (!ret->hasTrait<OpTrait::ReturnLike>())
        continue;
      builder.setInsertionPoint(ret);
      ttng::InvalBarrierOp::create(builder, ret->getLoc(),
                                   lifecycle.alloc.getResult());
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
    if (getModuleTwoCTAs(mod)) {
      assert(false &&
             "HoistMBarrierLifecyclePass does not handle the two-CTA mode yet");
    }

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
