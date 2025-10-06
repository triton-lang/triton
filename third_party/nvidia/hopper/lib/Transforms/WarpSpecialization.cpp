#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#define DEBUG_TYPE "nvgpu-warp-specialization"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

void doTaskPartition(triton::FuncOp &funcOp, unsigned numWarpGroups);
int doTaskIdPropagate(triton::FuncOp &funcOp);
bool doDataPartition(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doCodePartition(triton::FuncOp &funcOp, unsigned numBuffers);
void doTokenLowering(triton::FuncOp &funcOp, unsigned numConsumerGroups);
void doPingPongSync(triton::FuncOp &funcOp, unsigned numWarpGroups);

#define GEN_PASS_DEF_NVGPUWARPSPECIALIZATION
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUWarpSpecializationPass
    : public impl::NVGPUWarpSpecializationBase<NVGPUWarpSpecializationPass> {
public:
  using impl::NVGPUWarpSpecializationBase<
      NVGPUWarpSpecializationPass>::NVGPUWarpSpecializationBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    SmallVector<scf::ForOp> loops;
    funcOp->walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(mlir::triton::kWarpSpecializeAttrName))
        loops.push_back(forOp);
    });
    if (loops.empty())
      return;

    int numWarps = mlir::triton::gpu::lookupNumWarps(funcOp);
    if (numWarps != 4)
      return;

    // FIXME: skip warpspec if there is else block. Need to improve
    // CodePartitioning to correctly handle channels in else block.
    bool hasElse = false;
    funcOp->walk([&](scf::IfOp ifOp) {
      if (ifOp.elseBlock()) {
        for (Operation &op : ifOp.elseBlock()->getOperations()) {
          hasElse = true;
        }
      }
    });
    if (hasElse)
      return;

    OpBuilder builder(funcOp);
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    unsigned numWarpGroups = 3;
    // FIXME: skip data partitioning with on-host TMA.
    bool success = false;
    for (; numWarpGroups >= 2; numWarpGroups--) {
      // Partition key ops into multiple async tasks.
      doTaskPartition(funcOp, numWarpGroups);
      if (dumpIntermediateSteps) {
        llvm::dbgs()
            << "// -----// WarpSpec internal IR Dump After: doTaskPartition\n"
            << moduleOp << "\n\n\n";
      }
      // Propagate taskId.
      int retCode = doTaskIdPropagate(funcOp);
      if (retCode == -1)
        continue;
      if (dumpIntermediateSteps) {
        llvm::dbgs()
            << "// -----// WarpSpec internal IR Dump After: doTaskIdPropagate\n"
            << moduleOp << "\n\n\n";
      }

      // Partition ops into parallel sub ops.
      if (doDataPartition(funcOp, numWarpGroups - 1)) {
        if (dumpIntermediateSteps) {
          llvm::dbgs()
              << "// -----// WarpSpec internal IR Dump After: doDataPartition\n"
              << moduleOp << "\n\n\n";
        }
        success = true;
        break;
      }
      // Clear async_task.
    }
    if (!success)
      signalPassFailure();

    doCodePartition(funcOp, numStages);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doCodePartition\n"
          << moduleOp << "\n\n\n";
    }
    doPingPongSync(funcOp, numWarpGroups);
    if (dumpIntermediateSteps) {
      llvm::dbgs()
          << "// -----// WarpSpec internal IR Dump After: doPingPongSync\n"
          << moduleOp << "\n\n\n";
    }
    doTokenLowering(funcOp, numWarpGroups - 1);
    // Clear num_stages to disable SWP.
    funcOp->walk([&](scf::ForOp forOp) {
      forOp->setAttr(mlir::triton::kNumStagesAttrName,
                     builder.getI32IntegerAttr(0));
    });
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
