#include "TaskIdPropagation.h"
#include "Utility.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "nvgpu-ws-task-id-propagate"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir::dataflow;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

namespace mlir {

int doTaskIdPropagate(triton::FuncOp &funcOp) {
  SymbolTableCollection symbolTable;
  Operation *op = funcOp.getOperation();
  DataFlowSolver solver;

  // An op with side effects and no used results has no user to infer a task id
  // from. Leaving it unannotated drops it when its enclosing region is
  // specialized. Seed *result-bearing* ops with the producer (default) task
  // before running the analysis, so the backward propagation assigns the same
  // task to the values they consume (e.g. the pointer and program id of an
  // atomic) and to the enclosing control flow. Result-less ops are handled
  // after the analysis, where their operands' task ids are already known.
  auto isOrphanSideEffect = [](mlir::Operation *candidate) {
    auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(candidate);
    return memoryEffects && !memoryEffects.hasNoEffect() &&
           !candidate->hasAttr("async_task_id") &&
           llvm::all_of(candidate->getResults(),
                        [](Value result) { return result.use_empty(); });
  };
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getNumResults() == 0 || !isOrphanSideEffect(op))
      return;
    setAsyncTaskIds(op, {0});
    labelParentOps(op);
  });

  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<ttg::TaskIdBackwardPropagation>(symbolTable);
  if (failed(solver.initializeAndRun(op)))
    return -1;

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
    // A side-effecting op with no results and no annotated operands cannot
    // infer a task id either. Keep it in the producer (default) task.
    if (taskIds.isUninitialized() && isOrphanSideEffect(op)) {
      setAsyncTaskIds(op, {0});
      labelParentOps(op);
      return;
    }
    // TODO(Arda): Ideally front-end should not allow constant ops to be
    // annotated. Anchor constants cause problems.
    if (!taskIds.isUninitialized() &&
        (isa<arith::ConstantOp>(op) || !op->hasAttr("async_task_id"))) {
      op->setAttr("async_task_id", taskIds.getTaskIds());
      labelParentOps(op);
    }
  });
  return 0;
}

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
    int retCode = doTaskIdPropagate(funcOp);
    if (retCode != 0)
      signalPassFailure();
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
