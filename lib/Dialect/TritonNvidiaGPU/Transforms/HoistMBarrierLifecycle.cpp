#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"

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
// the wait phase through scf.for when the lifecycle repeats, and recreates
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
  int expectCount = 0;
  SmallVector<Operation *> transactionsAndWaits;
  SmallVector<ttng::WaitBarrierOp> waits;
  SmallVector<ttng::InvalBarrierOp> invals;
};

class MBarrierLifecycleHoister {
public:
  MBarrierLifecycleHoister(ModuleOp mod, int numCTAs)
      : mod(mod), builder(mod.getContext()), numCTAs(numCTAs) {}

  LogicalResult run() {
    for (auto funcOp : mod.getOps<FunctionOpInterface>()) {
      // scf only now and scf.while is not supported yet.
      if (funcOp->getNumRegions() != 1 || funcOp->getRegion(0).empty())
        continue;

      llvm::SmallPtrSet<Value, 8> seen;
      SmallVector<Value> candidates;
      funcOp.walk([&](ttng::InitBarrierOp init) {
        Value barrier = init.getAlloc();
        if (requiresCrossCTAMBarrierInitSync(funcOp, barrier, numCTAs) &&
            seen.insert(barrier).second)
          candidates.push_back(barrier);
      });

      for (Value barrier : candidates) {
        BarrierLifecycle lifecycle;
        if (failed(collectLifecycle(funcOp, barrier, lifecycle)))
          continue;
        if (failed(verifyTransactionWaitPairs(funcOp, lifecycle)))
          continue;
        if (failed(rewriteLoopPhases(lifecycle)))
          continue;

        moveInitToFunctionEntry(lifecycle, funcOp);
        moveInvalidationToFunctionExits(lifecycle, funcOp);
      }
    }
    return success();
  }

private:
  bool isKnownBarrierUser(Operation *op, Value barrier) {
    if (auto iface = dyn_cast<ttg::MBarrierOpInterface>(op)) {
      return llvm::any_of(iface.getBarriers(),
                          [&](Value value) { return value == barrier; });
    }
    return false;
  }

  bool isTransactionOp(Operation *op, Value barrier) {
    if (!isKnownBarrierUser(op, barrier))
      return false;
    // BarrierExpectOp only records the expected byte count for the next
    // transfer. The following producer/consumer op is the transaction that must
    // be paired with a wait.
    return !isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::WaitBarrierOp,
                ttng::BarrierExpectOp>(op);
  }

  bool isConstTrue(Value value) {
    if (!value)
      return true;
    if (matchPattern(value, m_One()))
      return true;
    if (auto constOp = value.getDefiningOp<arith::ConstantOp>())
      if (auto attr = dyn_cast<BoolAttr>(constOp.getValueAttr()))
        return attr.getValue();
    return false;
  }

  bool isUnconditionallyTrueTransaction(Operation *op) {
    if (auto predicated = dyn_cast<triton::PredicatedOpInterface>(op))
      return isConstTrue(predicated.getPredicateOperand());
    return true;
  }

  // "Opaque" means the barrier is passed to an op this pass does not model as
  // an mbarrier user. Skip hoisting rather than moving the lifecycle across an
  // unknown use.
  bool hasOpaqueBarrierUse(Value barrier) {
    for (Operation *user : barrier.getUsers()) {
      if (!isKnownBarrierUser(user, barrier))
        return true;
    }
    return false;
  }

  LogicalResult collectLifecycle(FunctionOpInterface funcOp, Value barrier,
                                 BarrierLifecycle &lifecycle) {
    lifecycle.barrier = barrier;
    lifecycle.alloc = barrier.getDefiningOp<ttg::LocalAllocOp>();
    if (!lifecycle.alloc || lifecycle.alloc->getNumOperands() != 0)
      return failure();

    if (hasOpaqueBarrierUse(barrier))
      return failure();

    funcOp.walk([&](Operation *op) {
      if (auto init = dyn_cast<ttng::InitBarrierOp>(op)) {
        if (init.getAlloc() == barrier) {
          lifecycle.init = init;
          ++lifecycle.initCount;
        }
        return;
      }
      if (auto expect = dyn_cast<ttng::BarrierExpectOp>(op)) {
        if (expect.getAlloc() == barrier)
          ++lifecycle.expectCount;
        return;
      }
      if (auto wait = dyn_cast<ttng::WaitBarrierOp>(op)) {
        if (wait.getAlloc() == barrier) {
          lifecycle.waits.push_back(wait);
          lifecycle.transactionsAndWaits.push_back(op);
        }
        return;
      }
      if (auto inval = dyn_cast<ttng::InvalBarrierOp>(op)) {
        if (inval.getAlloc() == barrier)
          lifecycle.invals.push_back(inval);
        return;
      }
      if (isTransactionOp(op, barrier))
        lifecycle.transactionsAndWaits.push_back(op);
    });

    if (lifecycle.initCount != 1 || !lifecycle.init ||
        lifecycle.expectCount > 1 || lifecycle.waits.empty() ||
        lifecycle.invals.size() != 1)
      return failure();
    return success();
  }

  LogicalResult verifyTransactionWaitPairs(FunctionOpInterface funcOp,
                                           BarrierLifecycle &lifecycle) {
    DominanceInfo domInfo(funcOp);
    PostDominanceInfo postDomInfo(funcOp);
    SmallVector<Operation *> pendingTransactions;

    for (Operation *op : lifecycle.transactionsAndWaits) {
      auto wait = dyn_cast<ttng::WaitBarrierOp>(op);
      if (!wait) {
        pendingTransactions.push_back(op);
        continue;
      }

      // A wait can cover one or more transactions, but it must not observe a
      // transaction that was already consumed by an earlier wait.
      if (pendingTransactions.empty())
        return failure();

      for (Operation *transaction : pendingTransactions) {
        // The transaction must run on every path to the wait. A transaction in
        // a conditional branch followed by a wait after the branch would
        // otherwise leave paths that wait on work that was never issued.
        if (!domInfo.dominates(transaction, wait.getOperation()))
          return failure();
        // The wait must run on every path after the transaction. Otherwise a
        // transaction before a conditional wait could be left unmatched.
        if (!postDomInfo.postDominates(wait.getOperation(), transaction))
          return failure();
        if (!isUnconditionallyTrueTransaction(transaction) ||
            !isConstTrue(wait.getPred()))
          return failure();
      }

      pendingTransactions.clear();
    }

    if (!pendingTransactions.empty())
      return failure();
    return success();
  }

  bool hasUnsupportedWhile(ttng::WaitBarrierOp wait, scf::ForOp forOp) {
    for (Operation *op = wait->getParentOp(); op && op != forOp;
         op = op->getParentOp()) {
      if (isa<scf::WhileOp>(op))
        return true;
    }
    return false;
  }

  LogicalResult rewriteLoopPhases(BarrierLifecycle &lifecycle) {
    llvm::MapVector<scf::ForOp, SmallVector<ttng::WaitBarrierOp>> waitsByLoop;
    SmallVector<ttng::WaitBarrierOp> nonLoopWaits;

    for (ttng::WaitBarrierOp wait : lifecycle.waits) {
      scf::ForOp forOp = wait->getParentOfType<scf::ForOp>();
      if (!forOp) {
        if (wait->getParentOfType<scf::WhileOp>())
          return failure();
        nonLoopWaits.push_back(wait);
        continue;
      }
      if (hasUnsupportedWhile(wait, forOp))
        return failure();
      waitsByLoop[forOp].push_back(wait);
    }

    // Straight-line single-use barriers keep phase 0.
    if (waitsByLoop.empty())
      return success();

    if (!nonLoopWaits.empty())
      return failure();

    for (auto &entry : waitsByLoop) {
      SmallVector<ttng::WaitBarrierOp> &waits = entry.second;
      if (waits.size() != 1)
        return failure();
    }

    for (auto &entry : waitsByLoop) {
      scf::ForOp forOp = entry.first;
      SmallVector<ttng::WaitBarrierOp> &waits = entry.second;
      ttng::WaitBarrierOp wait = waits.front();
      Location loc = wait.getLoc();
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(forOp);
      Value phase0 = arith::ConstantIntOp::create(builder, loc, 0, 32);
      Value phase1 = arith::ConstantIntOp::create(builder, loc, 1, 32);

      forOp = mlir::addIterArgsToLoop(builder, forOp, phase0);
      Value phase = forOp.getRegionIterArg(forOp.getNumRegionIterArgs() - 1);
      wait.getPhaseMutable().assign(phase);

      builder.setInsertionPointAfter(wait);
      // A hoisted loop-local barrier alternates between phase 0 and 1 on each
      // completed transaction, so carry phase through the scf.for and toggle it
      // after the wait.
      Value nextPhase = arith::XOrIOp::create(builder, loc, phase, phase1);
      if (wait->getBlock() != forOp.getBody())
        nextPhase = mlir::triton::sinkValueRedefinition(
            builder, phase, nextPhase, wait->getBlock());
      mlir::appendToForOpYield(forOp, nextPhase);
    }

    return success();
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

    if (failed(MBarrierLifecycleHoister(mod, numCTAs).run()))
      return signalPassFailure();
  }
};

} // namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
