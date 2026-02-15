#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/EquivalenceClasses.h"

using namespace mlir;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONLOOPAWARECSE
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"
} // namespace mlir::triton

namespace {
class ValueEquivalence {
public:
  std::optional<bool> getKnownEquivalence(Value a, Value b) {
    if (auto it = equalValues.find(normalizeKey(a, b)); it != equalValues.end())
      return it->second;
    return std::nullopt;
  }
  void setKnownEquivalence(Value a, Value b, bool eq) {
    equalValues.insert_or_assign(normalizeKey(a, b), eq);
  }

private:
  // Commutatively query the equivalence of two values by sorting the key by
  // pointer value.
  std::pair<Value, Value> normalizeKey(Value a, Value b) {
    if ((uintptr_t)a.getAsOpaquePointer() < (uintptr_t)b.getAsOpaquePointer())
      return {a, b};
    return {b, a};
  }

  DenseMap<std::pair<Value, Value>, bool> equalValues;
};

struct LoopCSEDriver {
  LoopCSEDriver(scf::ForOp loop) : loop(loop) {}

  bool areIterArgsEqual(int i, int j);
  bool areEqualInLoop(Value a, Value b);

  scf::ForOp loop;
  SmallVector<std::pair<int, int>> argStack;
};
} // namespace

bool LoopCSEDriver::areIterArgsEqual(int i, int j) {
  if (i == j)
    return true;
  if (loop.getInitArgs()[i] != loop.getInitArgs()[j])
    return false;
  if (llvm::is_contained(argStack, std::make_pair(i, j)))
    return true;

  // First, assume the arguments are equal. This is how recursion is broken.
  argStack.push_back({i, j});
  bool result =
      areEqualInLoop(loop.getYieldedValues()[i], loop.getYieldedValues()[j]);
  argStack.pop_back();
  return result;
}

bool LoopCSEDriver::areEqualInLoop(Value a, Value b) {
  // Check trivial case.
  if (a == b)
    return true;
  if (a.getType() != b.getType())
    return false;

  Block *aBlock = a.getParentBlock();
  Block *bBlock = b.getParentBlock();
  // Values from outside the loop must have been equal.
  if (aBlock != loop.getBody() || bBlock != loop.getBody()) {
    return false;
  }
  // Both must be block arguments or not.
  if (isa<BlockArgument>(a) != isa<BlockArgument>(b))
    return false;
  // Both must be the inductor var or not.
  if (a == loop.getInductionVar() || b == loop.getInductionVar())
    return false;

  if (auto aArg = dyn_cast<BlockArgument>(a)) {
    auto bArg = cast<BlockArgument>(b);
    bool result =
        areIterArgsEqual(aArg.getArgNumber() - 1, bArg.getArgNumber() - 1);
    return result;
  }

  Operation *aDef = a.getDefiningOp();
  Operation *bDef = b.getDefiningOp();
  if (cast<OpResult>(a).getResultNumber() !=
      cast<OpResult>(b).getResultNumber())
    return false;
  // For it to be known that the operation results have the same value, they
  // must be side effect free.
  if (!isMemoryEffectFree(aDef) || !isMemoryEffectFree(bDef))
    return false;
  // Don't bother with operations with regions.
  if (aDef->getNumRegions() || bDef->getNumRegions())
    return false;

  bool result = OperationEquivalence::isEquivalentTo(
      aDef, bDef,
      [&](Value a, Value b) { return success(areEqualInLoop(a, b)); },
      /*markEquivalent=*/nullptr, OperationEquivalence::IgnoreLocations);
  return result;
}

static void loopCSE(scf::ForOp loop) {
  int numIterArgs = loop.getNumRegionIterArgs();
  // Group equivalent iter args together.
  llvm::EquivalenceClasses<int> equivalentArgs;
  LoopCSEDriver driver(loop);
  for (int i = 0; i != numIterArgs; ++i) {
    for (int j = i + 1; j != numIterArgs; ++j) {
      if (driver.areIterArgsEqual(i, j))
        equivalentArgs.unionSets(i, j);
    }
  }

  // For each equivalence class, replace all other args in the class with one.
  for (auto it = equivalentArgs.begin(), end = equivalentArgs.end(); it != end;
       ++it) {
    if (!(*it)->isLeader())
      continue;
    SmallVector<int> eqArgs;
    for (auto mIt = equivalentArgs.member_begin(**it);
         mIt != equivalentArgs.member_end(); ++mIt)
      eqArgs.push_back(*mIt);
    assert(eqArgs.size() > 1);
    // Sort the indices so the pass is deterministic.
    llvm::sort(eqArgs);
    BlockArgument unique = loop.getRegionIterArg(eqArgs.front());
    Value uniqueResult = loop.getResult(eqArgs.front());
    for (int j : llvm::drop_begin(eqArgs)) {
      BlockArgument other = loop.getRegionIterArg(j);
      other.replaceAllUsesWith(unique);
      // Short-circuit the value. The canonicalizer will clean this up. Leftover
      // subcomputations can now be removed by normal CSE.
      (*loop.getYieldedValuesMutable())[j].set(other);
      loop.getResult(j).replaceAllUsesWith(uniqueResult);
    }
  }
}

namespace {
struct LoopAwareCSE
    : public triton::impl::TritonLoopAwareCSEBase<LoopAwareCSE> {
  using TritonLoopAwareCSEBase::TritonLoopAwareCSEBase;

  void runOnOperation() override {
    // LoopAwareCSE doesn't recursively CSE ops outside of loops, so run CSE
    // first to make sure values from outside loops that are equivalent are made
    // pointer equal.
    IRRewriter rewriter(&getContext());
    auto &domInfo = getAnalysis<DominanceInfo>();
    eliminateCommonSubExpressions(rewriter, domInfo, getOperation());

    // CSE region iter args within loop bodies.
    getOperation().walk(loopCSE);

    // Now that equivalent iter args have been made pointer equal, run CSE again
    // to clean up the loop body.
    eliminateCommonSubExpressions(rewriter, domInfo, getOperation());

    // Run the `scf.for` canonicalizer to clean up the loops (short-circuited
    // values, unused results, etc.).
    RewritePatternSet patterns(&getContext());
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
