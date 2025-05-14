#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
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

static std::optional<DenseSet<Operation *>> getArgDefOps(scf::ForOp loop,
                                                         unsigned idx) {
  Value root = loop.getYieldedValues()[idx];
  SmallVector<Value> worklist{root};
  DenseSet<Operation *> result;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    auto [op, distance] = getDefiningOpAndDistance(loop, value);
    if (!op || !result.insert(op).second)
      continue;
    // Don't CSE ops with regions.
    if (op->getNumRegions())
      return std::nullopt;
    // Don't need to use `getNestedOperands` since the op has no regions.
    llvm::append_range(worklist, op->getOperands());
  }
  return std::move(result);
}

namespace {
struct ValueDefSet {
  DenseSet<Operation *> ops;
  SmallVector<int> iterArgIndices;
};
} // namespace

static void loopCSE(scf::ForOp loop) {
  SmallVector<std::optional<DenseSet<Operation *>>> argDefOps;
  for (auto i : llvm::seq(loop.getNumRegionIterArgs()))
    argDefOps.push_back(getArgDefOps(loop, i));

  llvm::EquivalenceClasses<int> merged;
  for (auto [i, defOps] : llvm::enumerate(argDefOps)) {
    if (!defOps)
      continue;
    merged.insert(i);
    for (auto [j, otherDefOps] :
         llvm::enumerate(llvm::drop_begin(argDefOps, i + 1))) {
      assert(&otherDefOps != &defOps);
      if (!otherDefOps)
        continue;
      if (!llvm::set_intersects(*defOps, *otherDefOps))
        continue;
      merged.unionSets(i, j);
    }
  }

  SmallVector<ValueDefSet> valueDefSets;
  for (auto it = merged.begin(), e = merged.end(); it != e; ++it) {
    auto *ec = *it;
    if (!ec->isLeader())
      continue;

    ValueDefSet valueDefSet;
    for (auto mIt = merged.member_begin(*ec); mIt != merged.member_end();
         ++mIt) {
      valueDefSet.iterArgIndices.push_back(*mIt);
      valueDefSet.ops.insert_range(*argDefOps[*mIt]);
    }
  }
}

namespace {
struct LoopCSE : public triton::gpu::impl::TritonGPULoopCSEBase<LoopCSE> {
  using TritonGPULoopCSEBase::TritonGPULoopCSEBase;

  void runOnOperation() override { getOperation().walk(loopCSE); }
};
} // namespace
