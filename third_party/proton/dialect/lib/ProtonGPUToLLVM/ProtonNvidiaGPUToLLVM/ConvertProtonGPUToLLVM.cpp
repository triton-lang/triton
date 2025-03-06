#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#include "third_party/proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/TargetInfo.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton {
#define GEN_PASS_DEF_CONVERTPROTONNVIDIAGPUTOLLVM
#include "proton/dialect/include/Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h.inc"
} // namespace triton::proton
} // namespace mlir

namespace {

struct ConvertProtonNvidiaGPUToLLVM
    : public mlir::triton::proton::impl::ConvertProtonNvidiaGPUToLLVMBase<
          ConvertProtonNvidiaGPUToLLVM> {
  explicit ConvertProtonNvidiaGPUToLLVM(int32_t computeCapability,
                                        int32_t ptxVersion) {
    this->computeCapability = computeCapability;
    this->ptxVersion = ptxVersion;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    auto tritonTargetInfo =
        mlir::triton::NVIDIA::TargetInfo(computeCapability, ptxVersion);
    auto protonTargetInfo =
        mlir::triton::proton::NVIDIA::TargetInfo(tritonTargetInfo);
    return;
  }
};

} // namespace

namespace mlir {

namespace triton::proton {

namespace gpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertProtonNvidiaGPUToLLVMPass(int32_t computeCapability,
                                       int32_t ptxVersion) {
  return std::make_unique<ConvertProtonNvidiaGPUToLLVM>(computeCapability,
                                                        ptxVersion);
}

} // namespace gpu

} // namespace triton::proton

} // namespace mlir
