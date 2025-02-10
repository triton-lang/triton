#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_PROTONLOWERINGPASS
#include "../third_party/proton/dialect/include/ProtonGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {
struct ProtonLoweringPass
    : public mlir::triton::impl::ProtonLoweringPassBase<ProtonLoweringPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    ModuleAllocation allocation(mod);

    OpBuilder b(mod.getBodyRegion());
    MLIRContext *context = &getContext();
    auto loc = mod.getLoc();

    /*Add Proton Op Lowerings Here*/
  }
};

} // namespace

namespace mlir {

namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createProtonLoweringPass() {
  return std::make_unique<ProtonLoweringPass>();
}

} // namespace triton

} // namespace mlir
