#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

struct TestBufferRegionPass
    : public PassWrapper<TestBufferRegionPass, OperationPass<ModuleOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBufferRegionPass);

  static void emitRegionInfo(Location loc, StringRef name,
                             const tt::RegionInfo &regionInfo) {
    InFlightDiagnostic diag = mlir::emitRemark(loc);
    diag << name << ": ";
    regionInfo.print(diag);
  }

  static void emitRegionList(Location loc, StringRef name,
                             llvm::ArrayRef<tt::BufferRegion> regions) {
    if (regions.empty())
      return;

    InFlightDiagnostic diag = mlir::emitRemark(loc);
    diag << name << ": ";
    llvm::interleaveComma(regions, diag, [&](const tt::BufferRegion &region) {
      region.print(diag);
    });
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
      if (!triton::BufferRegionAnalysis::isMemoryAccessOperation(op))
        return;

      auto maybeMemDesc = llvm::find_if(op->getOperands(), [](Value operand) {
        return isa<ttg::MemDescType>(operand.getType());
      });

      if (maybeMemDesc == op->operand_end())
        return;

      emitRegionInfo(op->getLoc(), "Buffers",
                     analysis->getLatticeElement(*maybeMemDesc)->getValue());
    });

    llvm::SmallVector<Operation *> anchors;
    moduleOp.walk([&](Operation *op) {
      if (op->hasAttr("test.print_all_used_regions"))
        anchors.push_back(op);
    });

    for (Operation *anchor : anchors) {
      auto emitAllRegions = [&](tt::BufferRegionAnalysis::RegionType type,
                                StringRef label) {
        emitRegionList(anchor->getLoc(), label,
                       analysis->getAllUsedBufferRegions(type));
      };

      emitAllRegions(tt::BufferRegionAnalysis::SHARED_MEMORY,
                     "All Shared Regions");
      emitAllRegions(tt::BufferRegionAnalysis::TENSOR_MEMORY,
                     "All Tensor Regions");
      emitAllRegions(tt::BufferRegionAnalysis::BARRIER, "All Barrier Regions");
    }
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
