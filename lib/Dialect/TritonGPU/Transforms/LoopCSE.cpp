#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/SetOperations.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPULOOPCSE
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

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
  ValueEquivalence equalValues;
};
} // namespace

bool LoopCSEDriver::areIterArgsEqual(int i, int j) {
  if (i == j)
    return true;
  if (loop.getInitArgs()[i] != loop.getInitArgs()[j])
    return false;
  BlockArgument aArg = loop.getRegionIterArg(i);
  BlockArgument bArg = loop.getRegionIterArg(j);
  // First, assume the arguments are equal. This is how recursion is broken.
  equalValues.setKnownEquivalence(aArg, bArg, true);
  bool result =
      areEqualInLoop(loop.getYieldedValues()[i], loop.getYieldedValues()[j]);
  // Now update the equivalence based on the actual result.
  equalValues.setKnownEquivalence(aArg, bArg, result);
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

  if (std::optional<bool> eq = equalValues.getKnownEquivalence(a, b))
    return *eq;

  if (auto aArg = dyn_cast<BlockArgument>(a)) {
    auto bArg = cast<BlockArgument>(b);
    bool result =
        areIterArgsEqual(aArg.getArgNumber() - 1, bArg.getArgNumber() - 1);
    equalValues.setKnownEquivalence(a, b, result);
    return result;
  }

  Operation *aDef = a.getDefiningOp();
  Operation *bDef = b.getDefiningOp();
  bool result = OperationEquivalence::isEquivalentTo(
      aDef, bDef,
      [&](Value a, Value b) { return success(areEqualInLoop(a, b)); },
      [&](Value a, Value b) { equalValues.setKnownEquivalence(a, b, true); },
      OperationEquivalence::IgnoreLocations);
  equalValues.setKnownEquivalence(a, b, result);
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
struct LoopCSE : public triton::gpu::impl::TritonGPULoopCSEBase<LoopCSE> {
  using TritonGPULoopCSEBase::TritonGPULoopCSEBase;

  void runOnOperation() override {
    // LoopCSE doesn't recursively CSE ops outside of loops, so run CSE first to
    // make sure values from outside loops that are equivalent are made pointer
    // equal.
    OpPassManager pm;
    pm.addPass(createCSEPass());
    if (failed(runPipeline(pm, getOperation())))
      return signalPassFailure();

    // CSE region iter args within loop bodies.
    getOperation().walk(loopCSE);

    // Now that equivalent iter args have been made pointer equal, run CSE again
    // to clean up the loop body.
    if (failed(runPipeline(pm, getOperation())))
      return signalPassFailure();

    // Run the `scf.for` canonicalizer to clean up the loops (short-circuited
    // values, unused results, etc.).
    RewritePatternSet patterns(&getContext());
    scf::ForOp::getCanonicalizationPatterns(patterns, &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
