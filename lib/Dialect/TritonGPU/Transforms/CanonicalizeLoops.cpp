#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::triton;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

struct CanonicalizePass
    : public TritonGPUCanonicalizeLoopsBase<CanonicalizePass> {
  CanonicalizePass() = default;

  void runOnOperation() override {

    // Canonicalize pass may have created dead code that
    // standard scf.for canonicalization cannot handle
    // as of LLVM 14. For example, the iteration arguments
    // for the pointer of the synchronous loads that are
    // discarded.
    // The following piece of code is a workaround to
    // very crudely remove dead code, by making an iteration
    // argument yield itself if it is not used to create
    // side effects anywhere.
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      for (size_t i = 0; i < forOp.getNumResults(); ++i) {
        // condition 1: no other iter arguments depend on it
        SetVector<Operation *> fwdSlice;
        mlir::getForwardSlice(forOp.getRegionIterArgs()[i], &fwdSlice);
        Operation *yieldOp = forOp.getBody()->getTerminator();
        bool noOtherDependency = std::all_of(
            yieldOp->operand_begin(), yieldOp->operand_end(), [&](Value arg) {
              return arg == yieldOp->getOperand(i) ||
                     !fwdSlice.contains(arg.getDefiningOp());
            });
        // condition 2: final value is not used after the loop
        auto retVal = forOp.getResult(i);
        bool noUserAfterLoop = retVal.getUsers().empty();
        // yielding the region iter arg will cause loop canonicalization
        // to clean up the dead code
        if (noOtherDependency && noUserAfterLoop) {
          yieldOp->setOperand(i, forOp.getRegionIterArgs()[i]);
        }
      }
    });
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUCanonicalizeLoopsPass() {
  return std::make_unique<CanonicalizePass>();
}