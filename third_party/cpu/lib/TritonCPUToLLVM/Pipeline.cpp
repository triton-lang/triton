#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace triton {
namespace cpu {

void tritonCPUToLLVMPipelineBuilder(OpPassManager &pm) {
  pm.addPass(mlir::triton::cpu::createFuncOpToLLVMPass());
  pm.addPass(mlir::triton::cpu::createGetProgramIdOpToLLVMPass());
  pm.addPass(mlir::triton::cpu::createMemoryOpToLLVMPass());
  // pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

void registerTritonCPUToLLVMPipeline() {
  PassPipelineRegistration<>("triton-cpu-to-llvmir",
                             "TritonCPU to LLVM conversion pipeline.",
                             tritonCPUToLLVMPipelineBuilder);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
