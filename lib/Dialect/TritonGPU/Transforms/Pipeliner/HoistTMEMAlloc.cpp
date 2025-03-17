#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
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
// Returns the TMEMAllocOp and TMEMLoadOp that are used to allocate and load the
// accumulator for the given MMA operation. The TMEMAllocOp and TMEMLoadOp must
// be in the same region as the MMA operation.
std::optional<std::pair<ttng::TMEMAllocOp, ttng::TMEMLoadOp>>
getTMemAllocAndLoad(ttng::MMAv5OpInterface mmaOp) {
  auto acc = mmaOp->getOperand(2).getDefiningOp<ttng::TMEMAllocOp>();
  if (!acc || acc->getParentRegion() != mmaOp->getParentRegion()) {
    return std::nullopt;
  }
  for (auto user : acc->getUsers()) {
    if (auto load = dyn_cast<ttng::TMEMLoadOp>(user)) {
      if (load->getParentRegion() == mmaOp->getParentRegion()) {
        return std::make_pair(acc, load);
      }
    }
  }
  return std::nullopt;
}

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

Operation *findNearestCommonDominator(ArrayRef<Operation *> ops,
                                      DominanceInfo &domInfo) {
  if (ops.size() == 0) {
    return nullptr;
  }
  if (ops.size() == 1) {
    return ops[0];
  }
  llvm::SmallPtrSet<Block *, 16> blocks;
  for (auto op : ops) {
    blocks.insert(op->getBlock());
  }
  Block *domBlock = domInfo.findNearestCommonDominator(blocks);
  if (domBlock == nullptr) {
    return nullptr;
  }
  SmallVector<Operation *> ancestorOps;
  for (auto op : ops) {
    ancestorOps.push_back(domBlock->findAncestorOpInBlock(*op));
  }
  Operation *dom = ancestorOps[0];
  for (unsigned i = 1; i < ops.size(); i++) {
    if (ancestorOps[i]->isBeforeInBlock(dom)) {
      dom = ancestorOps[i];
    }
  }
  return dom;
}

ttng::TMEMAllocOp createTMemAlloc(OpBuilder &builder,
                                  ttng::TMEMAllocOp oldTMemAllocOp,
                                  bool multiBufferred, int numStages) {
  Location loc = oldTMemAllocOp.getLoc();
  auto oldRetType = oldTMemAllocOp.getType();
  SmallVector<int64_t> shape = {oldRetType.getShape().begin(),
                                oldRetType.getShape().end()};
  if (multiBufferred) {
    shape.insert(shape.begin(), numStages);
  }
  Type accMemDescType = triton::gpu::MemDescType::get(
      shape, oldRetType.getElementType(), oldRetType.getEncoding(),
      oldRetType.getMemorySpace(), /*mutableMemory=*/true);
  return builder.create<ttng::TMEMAllocOp>(oldTMemAllocOp.getLoc(),
                                           accMemDescType, nullptr);
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
  auto src = alloc->getOperand(0);
  auto newAlloc = createTMemAlloc(builder, alloc, false, 0);
  builder.setInsertionPoint(alloc);
  builder.create<ttng::TMEMStoreOp>(alloc.getLoc(), newAlloc.getResult(), src,
                                    vTrue);
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
      auto allocAndLoadOpt = getTMemAllocAndLoad(mmaOp);
      if (!allocAndLoadOpt) {
        return;
      }
      auto [alloc, load] = allocAndLoadOpt.value();
      auto forOp = dyn_cast<scf::ForOp>(load->getParentOp());
      if (!forOp) {
        return;
      }
      hoistTMEMAlloc(alloc, forOp);
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
