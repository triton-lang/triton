#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ALLOCATEPROTONSMEMBUFFER
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

struct AllocateProtonSMEMBuffer
    : public mlir::triton::impl::AllocateProtonSMEMBufferBase<
          AllocateProtonSMEMBuffer> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    ModuleAllocation allocation(mod);
    llvm::errs() <<  "Shared Memory: " << allocation.getSharedMemorySize() << "\n";

  }
};

} // namespace

namespace mlir {

namespace triton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>> createAllocateProtonSMEMBufferPass(StringRef targetArch, int customLDSLimit) {
  return std::make_unique<AllocateProtonSMEMBuffer>();
}

} // namespace gpu

} // namespace triton

} // namespace mlir
