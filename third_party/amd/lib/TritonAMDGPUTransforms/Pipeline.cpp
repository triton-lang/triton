#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUTransforms/PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#define DEBUG_TYPE "tritonamdgpu-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {
#define GEN_PASS_DEF_TRITONAMDGPUPIPELINE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {
Operation *streamPredication(RewriterBase &rewriter, Operation *op,
                             Value pred) {
  // The epilogue peeling generates a select for the stage output. This causes
  // too much register pressure with the loop result and the epilogue-dot in
  // regs for the select. Conditionally executing the dot will allow the backend
  // to optimize the select away as redundant.
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(op)) {
    auto loc = dotOp->getLoc();
    auto ifOp = rewriter.create<scf::IfOp>(loc, dotOp->getResult(0).getType(),
                                           pred, /*withElseRegion=*/true);
    auto thenB = ifOp.getThenBodyBuilder();
    auto yield = thenB.create<scf::YieldOp>(loc, dotOp->getResult(0));
    dotOp->moveBefore(yield);
    ifOp.getElseBodyBuilder().create<scf::YieldOp>(loc, dotOp->getOperand(2));
    return ifOp;
  }
  return tt::wrapInMaskOp(rewriter, op, pred);
}

void expandLoops(ModuleOp moduleOp, bool useAsyncCopy) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  for (scf::ForOp forOp : loops) {
    tt::CoarseSchedule schedule;
    if (failed(schedule.deSerialize(forOp)))
      continue;

    // Create the final schedule for the kernel loop. This will dictate the
    // stages and order of operations to the pipeline expander.
    auto coarseSchedule = schedule.createFinalSchedule(forOp);

    tt::PipeliningOption options;
    options.supportDynamicLoops = true;
    options.peelEpilogue = true;
    options.predicateFn = streamPredication;
    // Annotate loadOp in prologue for further moving up
    options.annotateFn = [](Operation *op,
                            tt::PipeliningOption::PipelinerPart part,
                            unsigned stage) {
      if (part != tt::PipeliningOption::PipelinerPart::Prologue)
        return;

      auto annotateLoad = [](Operation *loadOp) {
        loadOp->setAttr("amd.pipeliner_part",
                        StringAttr::get(loadOp->getContext(), "prologue"));
      };

      if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
        annotateLoad(loadOp);
        return;
      }
      // loadOp may be wrapped by a MaskOp as predicateFn execution
      // precedes annotation
      if (auto maskOp = dyn_cast<ttg::MaskOp>(op)) {
        for (auto &innerOp : maskOp.getBody()->without_terminator()) {
          if (auto loadOp = dyn_cast<tt::LoadOp>(&innerOp)) {
            annotateLoad(loadOp);
            return;
          }
        }
      }
    };
    // Set the final schedule as our scheduling function
    options.getScheduleFn =
        [coarseSchedule](scf::ForOp,
                         std::vector<std::pair<Operation *, unsigned>> &s) {
          s = std::move(coarseSchedule);
        };

    LDBG("Loop before sending to expander:\n" << *forOp);

    IRRewriter rewriter(forOp);
    FailureOr<scf::ForOp> newForOp =
        tt::pipelineForLoop(rewriter, forOp, options);

    if (failed(newForOp))
      continue;

    forOp = *newForOp;
  }

  // NOTE: Leave empty for now, until we utilize customEpiloguePeeling
  DenseSet<ttg::MaskOp> peeledMaskOps;
  tt::resolveMaskOp(moduleOp, peeledMaskOps);

  if (useAsyncCopy) {
    llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
    moduleOp.walk([&](ttg::AsyncWaitOp waitOp) { waitOps.insert(waitOp); });
    tt::combineRedundantWaitOps(waitOps);
  }
}
} // namespace

struct PipelinePass : impl::TritonAMDGPUPipelineBase<PipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    expandLoops(moduleOp, useAsyncCopy);

    tt::removePipeliningAttributes(moduleOp);
  }
};
} // namespace mlir
