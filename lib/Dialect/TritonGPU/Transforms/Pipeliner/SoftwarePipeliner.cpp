#include "PipelineExpander.h"
#include "Schedule.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create async operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

// The main loop pipelining. Return the new ForOp if it was successfully
// pipelined.
static void pipelineLoop(scf::ForOp forOp, int numStages) {
  mlir::triton::PipeliningOption options;
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) {
                     Operation *def = operand.getDefiningOp();
                     return !def;
                   }))
    return;

  bool foundSchedule = false;
  foundSchedule = preProcessLoopAndGetSchedule(forOp, numStages, options);

  // TODO: add more pipelines strategy.
  if (!foundSchedule)
    return;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::pipelineForLoop(rewriter, forOp, options);

  if (succeeded(newForOp)) {
    mlir::triton::asyncLaunchDots(newForOp.value());
  }
}

namespace {
struct PipelinePass : public TritonGPUPipelineBase<PipelinePass> {
  PipelinePass() = default;
  PipelinePass(int numStages, int numWarps, int numCTAs,
               int computeCapability) {
    this->numStages = numStages;
    this->numWarps = numWarps;
    this->numCTAs = numCTAs;
    this->computeCapability = computeCapability;
  }

  void runOnOperation() override {
    if (this->numStages <= 1)
      return;
    SmallVector<scf::ForOp> loops;

    getOperation()->walk<WalkOrder::PostOrder>(
        [&loops](scf::ForOp forOp) { loops.push_back(forOp); });

    // All for ops that have a forOp as descendant will be tombstoned.
    for (scf::ForOp forOp : loops) {
      if (!forOp)
        continue;
      while ((forOp = forOp.getOperation()->getParentOfType<scf::ForOp>())) {
        scf::ForOp *it = std::find(loops.begin(), loops.end(), forOp);
        if (it != loops.end()) {
          *it = nullptr;
        }
      }
    }

    // Pipeline the loops that were not found to be ancestors of other loops.
    for (scf::ForOp forOp : loops) {
      if (!forOp)
        continue;
      pipelineLoop(forOp, numStages);
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUPipelinePass(int numStages,
                                                        int numWarps,
                                                        int numCTAs,
                                                        int computeCapability) {
  return std::make_unique<PipelinePass>(numStages, numWarps, numCTAs,
                                        computeCapability);
}
