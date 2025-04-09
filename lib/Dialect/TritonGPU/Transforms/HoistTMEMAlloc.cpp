#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUHOISTTMEMALLOC
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// If the operation access tensor memory, return the tensor memory values.
void getTensorMemoryAccesses(Operation *op, DenseSet<Value> &tmemAccesses) {
  for (Value operand : op->getOperands()) {
    if (auto memdesc = dyn_cast<MemDescType>(operand.getType())) {
      if (isa<ttng::TensorMemorySpaceAttr>(memdesc.getMemorySpace())) {
        tmemAccesses.insert(operand);
      }
    }
  }
}

bool mayAliasTMEMOp(const DenseSet<Value> &sinkAccesses, Operation *op) {
  // Treat barriers as aliasing ops because they may be protecting tensory
  // memory buffers.
  if (isa<ttng::ArriveBarrierOp, ttng::WaitBarrierOp>(op)) {
    return true;
  }

  // Check if the operation may alias the sink.
  DenseSet<Value> tmemAccesses;
  getTensorMemoryAccesses(op, tmemAccesses);
  if (mayAliasAllocations(tmemAccesses, sinkAccesses)) {
    return true;
  }
  return false;
}

// Returns the earliest operation that may alias `sink` through tensor memory
// starting at, but not including, `lhs`. Returns nullptr if no such op exists.
Operation *findTMEMAliasingOpInBetween(Operation *lhs, Operation *sink) {
  DenseSet<Value> sinkAccesses;
  getTensorMemoryAccesses(sink, sinkAccesses);

  Operation *prevNode = sink;
  Operation *curAliasingOp = nullptr;
  while (prevNode != lhs) {
    if (prevNode->getPrevNode() == nullptr) {
      prevNode = prevNode->getParentOp();
      if (prevNode == nullptr) {
        return nullptr;
      }
    } else {
      prevNode = prevNode->getPrevNode();
    }
    if (prevNode == lhs) {
      break;
    }

    // Check if this op may alias tensor memory.
    if (mayAliasTMEMOp(sinkAccesses, prevNode)) {
      curAliasingOp = prevNode;
    }
  }
  return curAliasingOp;
}

class CombineTMEMStoreAndSelect : public OpRewritePattern<ttng::TMEMStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMStoreOp store,
                                PatternRewriter &rewriter) const override {
    Value src = store.getSrc();
    auto select = src.getDefiningOp<arith::SelectOp>();
    if (!select) {
      return failure();
    }
    enum { kTrue, kFalse, kUnknown } valueFromTMEM = kUnknown;
    Value trueSrc = select.getTrueValue();
    Value falseSrc = select.getFalseValue();
    if (auto load = trueSrc.getDefiningOp<ttng::TMEMLoadOp>()) {
      if (store.getDst() == load.getSrc() &&
          !findTMEMAliasingOpInBetween(load, store)) {
        valueFromTMEM = kTrue;
      }
    }
    if (auto load = falseSrc.getDefiningOp<ttng::TMEMLoadOp>()) {
      if (store.getDst() == load.getSrc() &&
          !findTMEMAliasingOpInBetween(load, store)) {
        valueFromTMEM = valueFromTMEM == kTrue ? kUnknown : kFalse;
      }
    }
    if (valueFromTMEM == kUnknown) {
      return failure();
    }
    Value pred = select.getCondition();
    // In case the false operand is overwriting, we need to negate the predicate
    // (owerwrite when select would be false)
    if (valueFromTMEM == kTrue) {
      Value one = rewriter.create<arith::ConstantIntOp>(select.getLoc(), 1, 1);
      pred = rewriter.create<arith::XOrIOp>(select.getLoc(), pred, one);
    }
    // Store the selected value with the updated predicate
    Value overwritingValue = valueFromTMEM == kTrue ? falseSrc : trueSrc;
    rewriter.create<ttng::TMEMStoreOp>(store.getLoc(), store.getDst(),
                                       overwritingValue, pred);
    store.erase();
    return success();
  }
};

class CombineTMEMLoadAndStore : public OpRewritePattern<ttng::TMEMLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMLoadOp load,
                                PatternRewriter &rewriter) const override {
    bool foundStore = false;
    for (auto user : llvm::make_early_inc_range(load->getUsers())) {
      if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
        if (store.getDst() != load.getSrc()) {
          continue;
        }
        // Can't have other stores to the same tmem_alloc in between the load
        // and the store
        if (findTMEMAliasingOpInBetween(load, store)) {
          continue;
        }
        rewriter.eraseOp(store);
        foundStore = true;
      }
    }
    if (!foundStore) {
      return failure();
    }
    return success();
  }
};

class SinkTMEMLoad : public OpRewritePattern<ttng::TMEMLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMLoadOp load,
                                PatternRewriter &rewriter) const override {
    auto forOp = load->getParentOfType<scf::ForOp>();
    if (!forOp) {
      return failure();
    }
    // If the load is used by a yield, it may have uses in the next loop
    // iteration so we can't sink it
    if (llvm::any_of(load->getUsers(),
                     [](Operation *op) { return isa<scf::YieldOp>(op); })) {
      return failure();
    }
    DominanceInfo domInfo(forOp);
    Operation *domOp =
        findNearestCommonDominator(llvm::to_vector(load->getUsers()), domInfo);
    if (!domOp || !domInfo.properlyDominates(load.getOperation(), domOp)) {
      return failure();
    }
    // Don't sink past potentially aliasing ops.
    if (Operation *dst = findTMEMAliasingOpInBetween(load, domOp)) {
      domOp = dst;
    }
    if (domOp == load->getNextNode()) {
      // The load wasn't moved.
      return failure();
    }
    rewriter.moveOpBefore(load, domOp);
    return success();
  }
};

class RotateTMEMStoreInLoop : public OpRewritePattern<ttng::TMEMStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMStoreOp store,
                                PatternRewriter &rewriter) const override {
    scf::ForOp forOp = dyn_cast<scf::ForOp>(store->getParentOp());
    if (!forOp) {
      return failure();
    }
    if (!forOp.isDefinedOutsideOfLoop(store.getPred()) ||
        !forOp.isDefinedOutsideOfLoop(store.getDst())) {
      return failure();
    }
    BlockArgument src = dyn_cast<BlockArgument>(store.getSrc());
    if (!src || !src.hasOneUse()) {
      return failure();
    }
    // Create two copies of the store: one before the loop, storing the initial
    // value, and one before the yield, storing the value carried by the loop
    // arg.
    int argNo = src.getArgNumber() - 1;
    Value initVal = forOp.getInitArgs()[argNo];
    rewriter.setInsertionPoint(forOp);
    rewriter.create<ttng::TMEMStoreOp>(store.getLoc(), store.getDst(), initVal,
                                       store.getPred());
    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    auto attributes = store->getAttrDictionary();
    rewriter.moveOpBefore(store, yield);
    Value yieldArgValue = yield.getOperand(argNo);
    store.getSrcMutable().assign(yieldArgValue);
    store->setAttrs(attributes);

    // Load from the tmem after the loop, and use it instead of the loop carried
    // value.
    rewriter.setInsertionPointAfter(forOp);
    auto load = rewriter.create<ttng::TMEMLoadOp>(
        store.getLoc(), store.getSrc().getType(), store.getDst());
    forOp->getResult(argNo).replaceAllUsesWith(load.getResult());
    // Loop carried value is no longer used, short-circuit it.
    yield.setOperand(argNo, forOp.getRegionIterArg(argNo));
    return success();
  }
};

class RotateTMEMLoadInLoop : public OpRewritePattern<ttng::TMEMLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::TMEMLoadOp load,
                                PatternRewriter &rewriter) const override {
    scf::ForOp forOp = dyn_cast<scf::ForOp>(load->getParentOp());
    if (!forOp) {
      return failure();
    }
    if (!forOp.isDefinedOutsideOfLoop(load.getSrc())) {
      return failure();
    }
    if (!load.getResult().hasOneUse()) {
      return failure();
    }
    OpOperand &use = *load.getResult().getUses().begin();
    auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
    if (!yield) {
      return failure();
    }
    // Create two copies of the store: one before the loop, storing the initial
    // value, and one before the yield, storing the value carried by the loop
    // arg.
    int argNo = use.getOperandNumber();
    Value initVal = forOp.getInitArgs()[argNo];
    rewriter.setInsertionPoint(forOp);
    auto vTrue = rewriter.create<arith::ConstantIntOp>(load.getLoc(), 1, 1);
    rewriter.create<ttng::TMEMStoreOp>(load.getLoc(), load.getSrc(), initVal,
                                       vTrue);
    auto attributes = load->getAttrDictionary();
    rewriter.moveOpBefore(load, &forOp.getBody()->front());
    forOp.getRegionIterArg(argNo).replaceAllUsesWith(load.getResult());
    load->setAttrs(attributes);

    // Load from the tmem after the loop, and use it instead of the loop carried
    // value.
    rewriter.setInsertionPointAfter(forOp);
    auto loadAfterLoop = rewriter.create<ttng::TMEMLoadOp>(
        load.getLoc(), load.getResult().getType(), load.getSrc());
    forOp->getResult(argNo).replaceAllUsesWith(loadAfterLoop.getResult());
    // Loop carried value is no longer used, short-circuit it.
    yield.setOperand(argNo, forOp.getRegionIterArg(argNo));
    return success();
  }
};

ttng::TMEMAllocOp hoistTMEMAlloc(ttng::TMEMAllocOp alloc, scf::ForOp forOp) {
  OpBuilder builder(alloc);
  builder.setInsertionPoint(forOp);
  Value vTrue = builder.create<arith::ConstantIntOp>(alloc.getLoc(), 1, 1);
  auto src = alloc.getSrc();
  auto newAlloc = ttng::createTMemAlloc(builder, alloc, false, 0);
  if (src != nullptr) {
    builder.setInsertionPoint(alloc);
    builder.create<ttng::TMEMStoreOp>(alloc.getLoc(), newAlloc.getResult(), src,
                                      vTrue);
  }
  alloc.replaceAllUsesWith(newAlloc.getResult());
  alloc.erase();

  return newAlloc;
}

// Hoist invariant tmem_alloc. This could technically be done as general LICM
// but controlling tmem liveranga more precisley is likely to be important.
static void hoistInvariantInputs(Operation *mmaOp, scf::ForOp forOp) {
  for (auto operand : mmaOp->getOperands()) {
    if (forOp.isDefinedOutsideOfLoop(operand))
      continue;
    auto tmemAllocOp = operand.getDefiningOp<ttng::TMEMAllocOp>();
    if (!tmemAllocOp || tmemAllocOp.getType().getMutableMemory())
      continue;
    assert(tmemAllocOp.getSrc());
    Value src = tmemAllocOp.getSrc();
    SmallVector<Operation *> opToHoist = {tmemAllocOp.getOperation()};
    // Also hoist simple unary elementwise that may have sinked into the loop.
    while (Operation *defOp = src.getDefiningOp()) {
      if (forOp.isDefinedOutsideOfLoop(src))
        break;
      if (!(isPure(defOp) && defOp->getNumOperands() == 1))
        break;
      opToHoist.push_back(defOp);
      src = defOp->getOperand(0);
    }
    if (!forOp.isDefinedOutsideOfLoop(src))
      continue;
    for (auto op : llvm::reverse(opToHoist)) {
      forOp.moveOutOfLoop(op);
    }
  }
}
} // namespace

struct HoistTMEMAlloc
    : public impl::TritonGPUHoistTMEMAllocBase<HoistTMEMAlloc> {
  using impl::TritonGPUHoistTMEMAllocBase<
      HoistTMEMAlloc>::TritonGPUHoistTMEMAllocBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    SmallVector<ttng::MMAv5OpInterface> mmaOps;
    m.walk([&](ttng::MMAv5OpInterface mmaOp) { mmaOps.push_back(mmaOp); });
    for (auto mmaOp : mmaOps) {
      auto forOp = dyn_cast<scf::ForOp>(mmaOp->getParentOp());
      if (!forOp) {
        continue;
      }
      hoistInvariantInputs(mmaOp, forOp);

      auto alloc = mmaOp.getAccumulator().getDefiningOp<ttng::TMEMAllocOp>();
      if (!alloc || alloc->getParentRegion() != mmaOp->getParentRegion()) {
        continue;
      }
      hoistTMEMAlloc(alloc, forOp);
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns
        .add<RotateTMEMStoreInLoop, RotateTMEMLoadInLoop,
             CombineTMEMLoadAndStore, CombineTMEMStoreAndSelect, SinkTMEMLoad>(
            &getContext());
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      llvm_unreachable("Failed to hoist tmem_store");
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
