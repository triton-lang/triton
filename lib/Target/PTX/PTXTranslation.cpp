#include "triton/Target/PTX/PTXTranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "triton/Target/LLVMIR/LLVMIRTranslation.h"
#include "triton/driver/dispatch.h"
#include "triton/driver/llvm.h"

namespace triton {

void getCuCCAndVersionFromDevice(uint64_t device, int *cc, int *version,
                                 std::string *ptxasPath) {
  CUdevice dev = (CUdevice)device;
  size_t major = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR>(dev);
  size_t minor = cuGetInfo<CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR>(dev);
  *cc = major * 10 + minor;
  *ptxasPath = driver::path_to_ptxas(*version); // assign version
}

std::tuple<std::string, size_t, int, std::string>
translateTritonGPUToPTX(mlir::ModuleOp module, uint64_t device) {
  int cc;
  int version;
  std::string ptxasPath;
  getCuCCAndVersionFromDevice(device, &cc, &version, &ptxasPath);

  llvm::LLVMContext ctx;
  auto llModule = mlir::triton::translateTritonGPUToLLVMIR(&ctx, module);
  auto ptxCode = driver::llir_to_ptx(llModule.get(), cc, version);
  return std::make_tuple(ptxCode, cc, version, ptxasPath);
}

} // namespace triton
