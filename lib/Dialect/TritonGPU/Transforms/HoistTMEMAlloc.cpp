#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
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

// This CRTP class is an operation type constraint that checks that it has TMEM
// dependency tokens present. HoistTMEMAlloc requires that TMEM tokens are
// present to check aliasing for its transformations.
template <typename OpT> struct HasToken : public OpT {
  using OpT::OpT;

  static bool classof(Operation *op) {
    if (auto tmemOp = dyn_cast<OpT>(op))
      return !!tmemOp.getToken();
    return false;
  }
};

using TMEMTokenLoadOp = HasToken<ttng::TMEMLoadOp>;
using TMEMTokenStoreOp = HasToken<ttng::TMEMStoreOp>;
using TMEMTokenAllocOp = HasToken<ttng::TMEMAllocOp>;

class CombineTMEMStoreAndSelect : public OpRewritePattern<TMEMTokenStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMTokenStoreOp store,
                                PatternRewriter &rewriter) const override {
    Value src = store.getSrc();
    auto select = src.getDefiningOp<arith::SelectOp>();
    if (!select) {
      return failure();
    }
    enum { kTrue, kFalse, kUnknown } valueFromTMEM = kUnknown;
    Value trueSrc = select.getTrueValue();
    Value falseSrc = select.getFalseValue();
    if (auto load = trueSrc.getDefiningOp<TMEMTokenLoadOp>()) {
      if (store.getDst() == load.getSrc() && load.getToken() == store.getDep())
        valueFromTMEM = kTrue;
    }
    if (auto load = falseSrc.getDefiningOp<TMEMTokenLoadOp>()) {
      if (store.getDst() == load.getSrc() && load.getToken() == store.getDep())
        valueFromTMEM = valueFromTMEM == kTrue ? kUnknown : kFalse;
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
    rewriter.replaceOpWithNewOp<ttng::TMEMStoreOp>(
        store, rewriter.getType<AsyncTokenType>(), store.getDst(),
        store.getDep(), overwritingValue, pred);
    return success();
  }
};

class RemoveUnusedTMEMLoad : public OpRewritePattern<TMEMTokenLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMTokenLoadOp load,
                                PatternRewriter &rewriter) const override {
    if (!load.getResult().use_empty())
      return failure();
    rewriter.replaceAllUsesWith(load.getToken(), load.getDep());
    return success();
  }
};

// Load-store forwarding pattern.
class CombineTMEMLoadAndStore : public OpRewritePattern<TMEMTokenStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMTokenStoreOp store,
                                PatternRewriter &rewriter) const override {
    auto load = store.getDep().getDefiningOp<HasToken<ttng::TMEMLoadOp>>();
    if (!load || load.getResult() != store.getSrc() ||
        load.getSrc() != store.getDst())
      return failure();
    rewriter.replaceOp(store, load.getToken());
    return success();
  }
};

class SinkTMEMLoad : public OpRewritePattern<TMEMTokenLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMTokenLoadOp load,
                                PatternRewriter &rewriter) const override {
    auto forOp = load->getParentOfType<scf::ForOp>();
    if (!forOp) {
      return failure();
    }
    DominanceInfo domInfo(forOp);
    Operation *domOp = findNearestCommonDominator(
        llvm::to_vector(load.getResult().getUsers()), domInfo);
    if (!domOp || !domInfo.properlyDominates(load.getOperation(), domOp)) {
      return failure();
    }
    // Don't sink past potentially aliasing ops.
    PostDominanceInfo postDomInfo(forOp);
    SmallVector<OpOperand *> uses;
    for (OpOperand &use : load.getToken().getUses())
      uses.push_back(&use);
    if (!llvm::all_of(uses, [&](OpOperand *use) {
          return postDomInfo.properlyPostDominates(use->getOwner(), domOp);
        }))
      return failure();
    if (domOp == load->getNextNode()) {
      // The load wasn't moved.
      return failure();
    }
    rewriter.moveOpBefore(load, domOp);
    Value newToken = sinkValueRedefinition(rewriter, load.getDep(),
                                           load.getToken(), domOp->getBlock());
    if (newToken != load.getToken()) {
      for (OpOperand *use : uses)
        use->set(newToken);
    }
    return success();
  }
};

// Remove loop-carried tensor dependencies if they are fed immediately into a
// TMEM store by pulling the store into the previous iteration.
class RotateTMEMStoreInLoop : public OpRewritePattern<TMEMTokenStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMTokenStoreOp store,
                                PatternRewriter &rewriter) const override {
    // Pattern match stores whose source comes from a loop region argument and
    // whose predicate is loop-invariant.
    scf::ForOp forOp = dyn_cast<scf::ForOp>(store->getParentOp());
    if (!forOp || !forOp.isDefinedOutsideOfLoop(store.getPred()) ||
        !forOp.isDefinedOutsideOfLoop(store.getDst())) {
      return failure();
    }
    auto getAsLoopArg = [&](Value v) -> BlockArgument {
      auto arg = dyn_cast<BlockArgument>(v);
      if (arg && arg.getOwner() == forOp.getBody())
        return arg;
      return {};
    };
    BlockArgument src = getAsLoopArg(store.getSrc());
    if (!src || !src.hasOneUse()) {
      return failure();
    }

    // Check that rotating the store into the past won't violate any
    // write-after-read dependencies.
    BlockArgument storeTok = getAsLoopArg(store.getDep());
    if (!storeTok)
      return failure();
    int tokArgNo = storeTok.getArgNumber() - 1;

    // Create two copies of the store: one before the loop, storing the initial
    // value, and one before the yield, storing the value carried by the loop
    // arg.
    int argNo = src.getArgNumber() - 1;
    Value initVal = forOp.getInitArgs()[argNo];
    rewriter.setInsertionPoint(forOp);
    auto tokType = rewriter.getType<AsyncTokenType>();
    auto initStore = rewriter.create<ttng::TMEMStoreOp>(
        store.getLoc(), tokType, store.getDst(), forOp.getInitArgs()[tokArgNo],
        initVal, store.getPred());
    forOp.getInitArgsMutable()[tokArgNo].assign(initStore.getToken());

    auto yield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    store.getToken().replaceAllUsesWith(forOp.getRegionIterArg(tokArgNo));
    rewriter.moveOpBefore(store, yield);
    store.getDepMutable().assign(yield.getOperand(tokArgNo));
    yield.setOperand(tokArgNo, store.getToken());
    store.getSrcMutable().assign(yield.getOperand(argNo));

    // Load from the tmem after the loop, and use it instead of the loop carried
    // value.
    rewriter.setInsertionPointAfter(forOp);
    auto load = rewriter.create<ttng::TMEMLoadOp>(
        store.getLoc(), store.getSrc().getType(), tokType, store.getDst(),
        forOp.getResult(tokArgNo));
    forOp->getResult(argNo).replaceAllUsesWith(load.getResult());
    // Loop carried value is no longer used, short-circuit it.
    yield.setOperand(argNo, forOp.getRegionIterArg(argNo));
    return success();
  }
};

// Remove loop-carried tensor dependencies if they are the result of TMEM loads
// at the end of the loop by pushing the load into the next iteration.
class RotateTMEMLoadInLoop : public OpRewritePattern<TMEMTokenLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TMEMTokenLoadOp load,
                                PatternRewriter &rewriter) const override {
    // Pattern match loads whose results are only passed into the next iteration
    // of a loop.
    scf::ForOp forOp = dyn_cast<scf::ForOp>(load->getParentOp());
    if (!forOp || !forOp.isDefinedOutsideOfLoop(load.getSrc()) ||
        !load.getResult().hasOneUse()) {
      return failure();
    }
    OpOperand &use = *load.getResult().use_begin();
    auto yield = dyn_cast<scf::YieldOp>(use.getOwner());
    if (!yield)
      return failure();

    // By rotating the load into the future, we are essentially merging the
    // loop-carried tensor value into the same TMEM allocation as the load.
    // Thus, they cannot be live at the same time. Check this by ensuring we
    // won't clobber the memory.

    // 1. There are no aliasing stores between the load and the end of the loop.
    if (!llvm::is_contained(load.getToken().getUsers(), yield))
      return failure();
    // 2. The TMEM variable is live into the loop with an undefined value.
    int tokArgNo = load.getToken().use_begin()->getOperandNumber();
    Value initTok = forOp.getInitArgs()[tokArgNo];
    auto initAlloc = initTok.getDefiningOp<TMEMTokenAllocOp>();
    if (!initAlloc || initAlloc.getSrc())
      return failure();
    // TODO: 3. The live-in value of the TMEM variable is never read.

    // Create a store before the loop to write the initial value.
    int argNo = use.getOperandNumber();
    Value initVal = forOp.getInitArgs()[argNo];
    rewriter.setInsertionPoint(forOp);
    auto vTrue = rewriter.create<arith::ConstantIntOp>(load.getLoc(), 1, 1);
    auto tokType = rewriter.getType<AsyncTokenType>();
    auto initStore = rewriter.create<ttng::TMEMStoreOp>(
        load.getLoc(), tokType, load.getSrc(), initAlloc.getToken(), initVal,
        vTrue);
    forOp.getInitArgsMutable()[tokArgNo].assign(initStore.getToken());

    // Move the load to the beginning of the loop to load the tensor value.
    yield.setOperand(tokArgNo, load.getDep());
    rewriter.moveOpBefore(load, &forOp.getBody()->front());
    Value tokArg = forOp.getRegionIterArg(tokArgNo);
    load.getDepMutable().assign(tokArg);
    tokArg.replaceAllUsesExcept(load.getToken(), load);
    forOp.getRegionIterArg(argNo).replaceAllUsesWith(load.getResult());

    // Load from the tmem after the loop, and use it instead of the loop carried
    // value.
    rewriter.setInsertionPointAfter(forOp);
    auto loadAfterLoop = rewriter.create<ttng::TMEMLoadOp>(
        load.getLoc(), load.getResult().getType(), tokType, load.getSrc(),
        forOp.getResult(tokArgNo));
    forOp->getResult(argNo).replaceAllUsesWith(loadAfterLoop.getResult());
    // Loop carried value is no longer used, short-circuit it.
    yield.setOperand(argNo, forOp.getRegionIterArg(argNo));
    return success();
  }
};

// Given an operation that uses a token, return its forwarded token. This
// assumes the memory variable is not loop carried.
static Value getTokenFromOp(Operation *op) {
  if (auto mmaOp = dyn_cast<HasToken<ttng::MMAv5OpInterface>>(op)) {
    return mmaOp.getToken();
  } else if (auto loadOp = dyn_cast<TMEMTokenLoadOp>(op)) {
    return loadOp.getToken();
  } else if (auto storeOp = dyn_cast<TMEMTokenStoreOp>(op)) {
    return storeOp.getToken();
  }
  assert(!isa<scf::YieldOp>(op) && "unexpected loop carried token");
  llvm_unreachable("unknown TMEM memory user");
}

// Find all the last uses of a memory variable in a loop body. This traces the
// token lattice to its leaves.
static void findLastMemoryUses(OpResult token,
                               SmallVectorImpl<OpResult> &lastUses,
                               DenseSet<Value> &seen) {
  if (!seen.insert(token).second)
    return;
  if (token.use_empty()) {
    lastUses.push_back(token);
    return;
  }
  for (Operation *user : token.getUsers())
    findLastMemoryUses(cast<OpResult>(getTokenFromOp(user)), lastUses, seen);
}

// Find the last uses of a memory variable, joining them into a single token if
// necessary. This token can be carried into the next loop iteration.
static Value joinLastMemoryUses(OpBuilder &b, Value token) {
  SmallVector<OpResult> lastUses;
  DenseSet<Value> seenTokens;
  findLastMemoryUses(cast<OpResult>(token), lastUses, seenTokens);
  assert(!lastUses.empty());

  if (lastUses.size() == 1 && lastUses.front().getDefiningOp()->getBlock() ==
                                  token.getDefiningOp()->getBlock())
    return lastUses.front();
  // We can handle this case as needed. Right now it never happens.
  llvm::report_fatal_error(
      "FIXME: can't hoist TMEM alloc with multiple or conditional uses");
}

ttng::TMEMAllocOp hoistTMEMAlloc(TMEMTokenAllocOp alloc, scf::ForOp &forOp) {
  OpBuilder builder(alloc);
  builder.setInsertionPoint(forOp);
  Value vTrue = builder.create<arith::ConstantIntOp>(alloc.getLoc(), 1, 1);
  auto src = alloc.getSrc();
  auto newAlloc = cast<ttng::TMEMAllocOp>(builder.clone(*alloc));
  newAlloc.getSrcMutable().clear();

  // By hoisting the allocation out of the loop, we need to turn the underlying
  // memory variable into a loop-carried depdendency.
  auto tokType = builder.getType<AsyncTokenType>();
  Value newTok = addIterArgsToLoop(builder, forOp, newAlloc.getToken()).front();
  appendToForOpYield(forOp, joinLastMemoryUses(builder, alloc.getToken()));

  if (src != nullptr) {
    builder.setInsertionPoint(alloc);
    // Write the initial value of the allocation and replace the token.
    auto initStoreOp = builder.create<ttng::TMEMStoreOp>(
        alloc.getLoc(), tokType, newAlloc.getResult(), newTok, src, vTrue);
    newTok = initStoreOp.getToken();
  }
  alloc.replaceAllUsesWith(ValueRange{newAlloc.getResult(), newTok});
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

      // Only hoist the TMEM alloc feeding into the accumulator. Leave the ones
      // for the scales in the loop.
      auto alloc = mmaOp.getAccumulator().getDefiningOp<TMEMTokenAllocOp>();
      if (!alloc || alloc->getParentRegion() != mmaOp->getParentRegion()) {
        continue;
      }
      hoistTMEMAlloc(alloc, forOp);
    }

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RotateTMEMStoreInLoop, RotateTMEMLoadInLoop,
                 CombineTMEMLoadAndStore, CombineTMEMStoreAndSelect,
                 SinkTMEMLoad, RemoveUnusedTMEMLoad>(&getContext());
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      llvm_unreachable("Failed to hoist tmem_store");
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
