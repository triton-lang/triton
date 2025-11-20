#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

using namespace mlir;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

struct TestBufferRegionPass
    : public PassWrapper<TestBufferRegionPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBufferRegionPass);

  static void emit(Location loc, StringRef name,
                   const tt::RegionInfo &regionInfo) {
    InFlightDiagnostic diag = mlir::emitRemark(loc);
    diag << name << ": ";
    regionInfo.print(diag);
  }

  StringRef getArgument() const final { return "test-print-buffer-region"; }
  StringRef getDescription() const final {
    return "print the result of the buffer region analysis pass";
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    std::unique_ptr<DataFlowSolver> solver = createDataFlowSolver();
    triton::BufferRegionAnalysis *analysis =
        solver->load<triton::BufferRegionAnalysis>();
    if (failed(solver->initializeAndRun(moduleOp)))
      return signalPassFailure();
    analysis->calculateUsedBufferRegions(moduleOp);

    moduleOp.walk([&](Operation *op) {
      if (!isa<ttg::LocalLoadOp, ttg::LocalStoreOp, ttng::TMEMLoadOp,
               ttng::TMEMStoreOp, ttng::InitBarrierOp>(op))
        return;

      auto maybeMemDesc = llvm::find_if(op->getOperands(), [](Value operand) {
        return isa<ttg::MemDescType>(operand.getType());
      });

      if (maybeMemDesc == op->operand_end())
        return;

      StringRef label = "Shared";
      if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp>(op))
        label = "Tensor";
      else if (isa<ttng::InitBarrierOp>(op))
        label = "Barrier";

      emit(op->getLoc(), label,
           analysis->getLatticeElement(*maybeMemDesc)->getValue());
    });
  }
};

} // namespace

namespace mlir {
namespace test {
void registerTestBufferRegionPass() {
  PassRegistration<TestBufferRegionPass>();
}
} // namespace test
} // namespace mlir
