#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/TargetInfo.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton {
#define GEN_PASS_DEF_CONVERTPROTONAMDGPUTOLLVM
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h.inc"
} // namespace triton::proton
} // namespace mlir

namespace {

struct ConvertProtonAMDGPUToLLVM
    : public mlir::triton::proton::impl::ConvertProtonAMDGPUToLLVMBase<
          ConvertProtonAMDGPUToLLVM> {
  explicit ConvertProtonAMDGPUToLLVM(std::string arch) { this->arch = arch; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    auto tritonTargetInfo = mlir::triton::AMD::TargetInfo(arch);
    auto protonTargetInfo =
        mlir::triton::proton::AMD::TargetInfo(tritonTargetInfo);
    return;
  }
};

} // namespace

namespace mlir {

namespace triton::proton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertProtonAMDGPUToLLVMPass(std::string arch) {
  return std::make_unique<ConvertProtonAMDGPUToLLVM>(arch);
}

} // namespace gpu

} // namespace triton::proton

} // namespace mlir
