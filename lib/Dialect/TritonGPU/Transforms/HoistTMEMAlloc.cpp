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

// Returns true if between the op and the store there is another store to the
// same TMEMAlloc.
// Assumes that op dominates store.
bool aliasingStoresBetween(Operation *op, ttng::TMEMStoreOp store) {
  Operation *prevNode = store;
  while (prevNode != op) {
    if (prevNode->getPrevNode() == nullptr) {
      prevNode = prevNode->getParentOp();
      if (prevNode == nullptr) {
        return false;
      }
    } else {
      prevNode = prevNode->getPrevNode();
    }
    if (auto otherStore = dyn_cast<ttng::TMEMStoreOp>(prevNode)) {
      if (otherStore.getDst() == store.getDst()) {
        return true;
      }
    }
    if (prevNode == op) {
      break;
    }
  }
  return false;
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
          !aliasingStoresBetween(load, store)) {
        valueFromTMEM = kTrue;
      }
    }
    if (auto load = falseSrc.getDefiningOp<ttng::TMEMLoadOp>()) {
      if (store.getDst() == load.getSrc() &&
          !aliasingStoresBetween(load, store)) {
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
    for (auto user : load->getUsers()) {
      if (auto store = dyn_cast<ttng::TMEMStoreOp>(user)) {
        if (store.getDst() != load.getSrc()) {
          continue;
        }
        // Can't have other stores to the same tmem_alloc in between the load
        // and the store
        if (aliasingStoresBetween(load, store)) {
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
    if (!domOp || !domInfo.properlyDominates(load.getOperation(), domOp) ||
        domOp == load->getNextNode()) {
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

  ModuleOp module = forOp->getParentOfType<ModuleOp>();
  mlir::RewritePatternSet patterns(module.getContext());
  patterns.add<RotateTMEMStoreInLoop, CombineTMEMLoadAndStore,
               CombineTMEMStoreAndSelect, SinkTMEMLoad>(module.getContext());
  scf::ForOp::getCanonicalizationPatterns(patterns, module.getContext());
  if (applyPatternsGreedily(module, std::move(patterns)).failed()) {
    llvm_unreachable("Failed to hoist tmem_store");
  }

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
      if (!(isMemoryEffectFree(defOp) && isSpeculatable(defOp) &&
            defOp->getNumOperands() == 1))
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
        return;
      }
      hoistInvariantInputs(mmaOp, forOp);

      auto allocAndLoadOpt = getTMemAllocAndLoad(mmaOp);
      if (!allocAndLoadOpt) {
        return;
      }
      auto [alloc, load] = allocAndLoadOpt.value();
      hoistTMEMAlloc(alloc, forOp);
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
