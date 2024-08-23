#ifndef AMDGPU_CONVERSION_PASSES
#define AMDGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertAMDGPUToLLVM : Pass<"convert-amd-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert AMDGPU to LLVM";
    let description = [{
    }];
    let constructor = "mlir::triton::createConvertAMDGPUToLLVMPass()";

    let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif
