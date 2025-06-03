#include "TaskIdPropagation.h"
#include "Utility.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#define DEBUG_TYPE "nvgpu-ws-task-id-propagate"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir::dataflow;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {

#define GEN_PASS_DEF_NVGPUTESTWSTASKIDPROPAGATE
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestWSTaskIdPropagatePass
    : public impl::NVGPUTestWSTaskIdPropagateBase<
          NVGPUTestWSTaskIdPropagatePass> {
public:
  using impl::NVGPUTestWSTaskIdPropagateBase<
      NVGPUTestWSTaskIdPropagatePass>::NVGPUTestWSTaskIdPropagateBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    llvm::DenseSet<Operation *> anchorOps;
    funcOp.walk([&](mlir::Operation *op) {
      auto asyncTasks = getAsyncTaskIds(op);
      if (!asyncTasks.empty()) {
        std::sort(asyncTasks.begin(), asyncTasks.end());
        setAsyncTaskIds(op, asyncTasks);
        if (!isa<arith::ConstantOp, arith::ConstantIntOp>(op))
          anchorOps.insert(op);
        if (numWarpGroups == 0)
          op->removeAttr("async_task_id");
      }
    });
    if (numWarpGroups == 0 || anchorOps.empty())
      return;

    SymbolTableCollection symbolTable;
    Operation *op = getOperation();
    DataFlowSolver solver;

    solver.load<DeadCodeAnalysis>();
    solver.load<SparseConstantPropagation>();
    solver.load<ttg::TaskIdBackwardPropagation>(symbolTable);
    if (failed(solver.initializeAndRun(op)))
      return signalPassFailure();

    // Annotate the ops with the results from the dataflow analysis.
    getOperation()->walk([&](triton::FuncOp funcOp) {
      funcOp.walk([&](mlir::Operation *op) {
        auto taskIds = ttg::TaskId::getUninitialized();
        // Get the union of the results
        for (auto result : op->getResults()) {
          auto *lattice = solver.lookupState<ttg::TaskIdLattice>(result);
          if (!lattice)
            llvm_unreachable("Lattice not found.");
          taskIds = taskIds.meet(taskIds, lattice->getValue());
        }
        // Get the union of the operands
        if (op->getNumResults() == 0) {
          for (auto operand : op->getOperands()) {
            auto *lattice = solver.lookupState<ttg::TaskIdLattice>(operand);
            if (!lattice)
              llvm_unreachable("Lattice not found.");
            taskIds = taskIds.meet(taskIds, lattice->getValue());
          }
        }
        // TODO(Arda): Ideally front-end should not allow constant ops to be
        // annotated. Anchor constants cause problems.
        if (!taskIds.isUninitialized() &&
            (isa<arith::ConstantOp>(op) || !op->hasAttr("async_task_id"))) {
          op->setAttr("async_task_id", taskIds.getTaskIds());
          labelParentOps(op);
        }
      });
    });
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
