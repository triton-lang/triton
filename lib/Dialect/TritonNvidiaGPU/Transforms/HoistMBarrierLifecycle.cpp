#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUHOISTMBARRIERLIFECYCLEPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

struct BarrierLifecycle {
  Value barrier;
  ttg::LocalAllocOp alloc;
  ttng::InitBarrierOp init;
  int initCount = 0;
  SmallVector<ttng::WaitBarrierOp> waits;
  SmallVector<ttng::InvalBarrierOp> invals;
};

static bool isCrossCTABarrier(Value barrier, int numCTAs) {
  auto barrierTy = dyn_cast<ttg::MemDescType>(barrier.getType());
  return barrierTy && barrierTy.getShape()[0] != numCTAs;
}

static bool isCrossCTAConsumer(Operation *op, Value barrier) {
  auto aliasesBarrier = [&](Value value) { return value == barrier; };

  if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(op)) {
    auto barrierOp = cast<ttg::MBarrierOpInterface>(op);
    return mma.getTwoCtas() &&
           llvm::any_of(barrierOp.getBarriers(), aliasesBarrier);
  }
  if (auto commit = dyn_cast<ttng::TCGen5CommitOp>(op))
    return ttng::getModuleTwoCTAs(op) && aliasesBarrier(commit.getBarrier());
  if (auto tma = dyn_cast<ttng::TMALoadLikeOpInterface>(op))
    return tma.getMulticast() && aliasesBarrier(tma.getBarrier());
  if (auto clc = dyn_cast<ttng::CLCTryCancelOp>(op))
    return aliasesBarrier(clc.getMbarrier());
  return false;
}

static bool requiresCrossCTAInitSync(FunctionOpInterface funcOp, Value barrier,
                                     int numCTAs) {
  if (isCrossCTABarrier(barrier, numCTAs))
    return true;

  return funcOp
      ->walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (isCrossCTAConsumer(op, barrier))
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

static bool isKnownBarrierUser(Operation *op, Value barrier) {
  if (auto iface = dyn_cast<ttg::MBarrierOpInterface>(op)) {
    return llvm::any_of(iface.getBarriers(),
                        [&](Value value) { return value == barrier; });
  }
  return false;
}

static bool isTransactionOp(Operation *op, Value barrier) {
  if (!isKnownBarrierUser(op, barrier))
    return false;
  return !isa<ttng::InitBarrierOp, ttng::InvalBarrierOp, ttng::WaitBarrierOp,
              ttng::BarrierExpectOp>(op);
}

static bool isConstTrue(Value value) {
  if (!value)
    return true;
  if (matchPattern(value, m_One()))
    return true;
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>())
    if (auto attr = dyn_cast<BoolAttr>(constOp.getValueAttr()))
      return attr.getValue();
  return false;
}

static bool isUnconditionallyTrueTransaction(Operation *op) {
  if (auto predicated = dyn_cast<triton::PredicatedOpInterface>(op))
    return isConstTrue(predicated.getPredicateOperand());
  return true;
}

static bool hasOpaqueBarrierUse(Value barrier) {
  for (Operation *user : barrier.getUsers()) {
    if (!isKnownBarrierUser(user, barrier))
      return true;
  }
  return false;
}

static LogicalResult collectLifecycle(FunctionOpInterface funcOp, Value barrier,
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

  if (lifecycle.initCount != 1 || !lifecycle.init || lifecycle.waits.empty() ||
      lifecycle.invals.empty())
    return failure();
  return success();
}

static LogicalResult verifySingleTransactionPerWait(FunctionOpInterface funcOp,
                                                    Value barrier) {
  int pendingTransactions = 0;
  Operation *firstPendingTransaction = nullptr;

  WalkResult result = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (isTransactionOp(op, barrier)) {
      if (pendingTransactions == 0)
        firstPendingTransaction = op;
      ++pendingTransactions;
      return WalkResult::advance();
    }

    auto wait = dyn_cast<ttng::WaitBarrierOp>(op);
    if (!wait || wait.getAlloc() != barrier)
      return WalkResult::advance();

    if (pendingTransactions != 1) {
      InFlightDiagnostic diag = wait.emitOpError()
                                << "cannot hoist mbarrier lifecycle: expected "
                                   "exactly one transaction before this wait, "
                                   "but found "
                                << pendingTransactions;
      if (firstPendingTransaction)
        diag.attachNote(firstPendingTransaction->getLoc())
            << "first transaction covered by this wait";
      return WalkResult::interrupt();
    }
    if (firstPendingTransaction->getBlock() != wait->getBlock()) {
      wait.emitOpError() << "cannot hoist mbarrier lifecycle: transaction and "
                            "wait must be in the same block";
      return WalkResult::interrupt();
    }
    if (!isUnconditionallyTrueTransaction(firstPendingTransaction) ||
        !isConstTrue(wait.getPred())) {
      wait.emitOpError() << "cannot hoist mbarrier lifecycle for predicated "
                            "transactions";
      return WalkResult::interrupt();
    }

    pendingTransactions = 0;
    firstPendingTransaction = nullptr;
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();
  if (pendingTransactions != 0) {
    InFlightDiagnostic diag =
        funcOp.emitOpError()
        << "cannot hoist mbarrier lifecycle: found " << pendingTransactions
        << " transaction(s) without a matching wait";
    if (firstPendingTransaction)
      diag.attachNote(firstPendingTransaction->getLoc())
          << "first transaction without a matching wait";
    return failure();
  }
  return success();
}

static bool hasUnsupportedWhile(ttng::WaitBarrierOp wait, scf::ForOp forOp) {
  for (Operation *op = wait->getParentOp(); op && op != forOp;
       op = op->getParentOp()) {
    if (isa<scf::WhileOp>(op))
      return true;
  }
  return false;
}

static LogicalResult rewriteLoopPhases(BarrierLifecycle &lifecycle,
                                       RewriterBase &builder) {
  llvm::MapVector<scf::ForOp, SmallVector<ttng::WaitBarrierOp>> waitsByLoop;
  SmallVector<ttng::WaitBarrierOp> nonLoopWaits;

  for (ttng::WaitBarrierOp wait : lifecycle.waits) {
    scf::ForOp forOp = wait->getParentOfType<scf::ForOp>();
    if (!forOp) {
      if (wait->getParentOfType<scf::WhileOp>()) {
        wait.emitOpError() << "cannot hoist mbarrier lifecycle through "
                              "scf.while yet";
        return failure();
      }
      nonLoopWaits.push_back(wait);
      continue;
    }
    if (hasUnsupportedWhile(wait, forOp)) {
      wait.emitOpError() << "cannot hoist mbarrier lifecycle through this "
                            "control-flow shape yet";
      return failure();
    }
    waitsByLoop[forOp].push_back(wait);
  }

  // Straight-line single-use barriers keep phase 0.
  if (waitsByLoop.empty())
    return success();

  if (!nonLoopWaits.empty())
    return failure();

  for (auto &entry : waitsByLoop) {
    scf::ForOp forOp = entry.first;
    SmallVector<ttng::WaitBarrierOp> &waits = entry.second;
    if (waits.size() != 1) {
      forOp.emitOpError()
          << "cannot hoist mbarrier lifecycle with multiple waits for the "
             "same barrier in one loop";
      return failure();
    }

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
    Value nextPhase = arith::XOrIOp::create(builder, loc, phase, phase1);
    if (wait->getBlock() != forOp.getBody())
      nextPhase = mlir::triton::sinkValueRedefinition(builder, phase, nextPhase,
                                                      wait->getBlock());
    mlir::appendToForOpYield(forOp, nextPhase);
  }

  return success();
}

static Operation *getFunctionEntryInsertionPoint(FunctionOpInterface funcOp) {
  Block &entry = funcOp->getRegion(0).front();
  return &entry.front();
}

static void moveInitToFunctionEntry(BarrierLifecycle &lifecycle,
                                    FunctionOpInterface funcOp) {
  Operation *entry = getFunctionEntryInsertionPoint(funcOp);
  lifecycle.alloc->moveBefore(entry);
  lifecycle.init->moveAfter(lifecycle.alloc);
}

static void moveInvalidationToFunctionExits(BarrierLifecycle &lifecycle,
                                            FunctionOpInterface funcOp,
                                            RewriterBase &builder) {
  for (ttng::InvalBarrierOp inval : lifecycle.invals)
    inval->erase();

  SmallVector<Operation *> returns;
  funcOp.walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::ReturnLike>() &&
        op->getParentOp() == funcOp.getOperation())
      returns.push_back(op);
  });

  for (Operation *ret : returns) {
    builder.setInsertionPoint(ret);
    ttng::InvalBarrierOp::create(builder, ret->getLoc(), lifecycle.barrier);
  }
}

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

    IRRewriter builder(mod.getContext());
    for (auto funcOp : mod.getOps<FunctionOpInterface>()) {
      if (funcOp->getNumRegions() != 1 || funcOp->getRegion(0).empty())
        continue;

      llvm::SmallPtrSet<Value, 8> seen;
      SmallVector<Value> candidates;
      funcOp.walk([&](ttng::InitBarrierOp init) {
        Value barrier = init.getAlloc();
        if (requiresCrossCTAInitSync(funcOp, barrier, numCTAs) &&
            seen.insert(barrier).second)
          candidates.push_back(barrier);
      });

      for (Value barrier : candidates) {
        BarrierLifecycle lifecycle;
        if (failed(collectLifecycle(funcOp, barrier, lifecycle)))
          continue;
        if (failed(verifySingleTransactionPerWait(funcOp, barrier)))
          return signalPassFailure();
        if (failed(rewriteLoopPhases(lifecycle, builder)))
          return signalPassFailure();

        moveInitToFunctionEntry(lifecycle, funcOp);
        moveInvalidationToFunctionExits(lifecycle, funcOp, builder);
      }
    }
  }
};

} // namespace

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
