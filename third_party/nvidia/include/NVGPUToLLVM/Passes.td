#ifndef NVGPU_CONVERSION_PASSES
#define NVGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertNVGPUToLLVM : Pass<"convert-nv-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert NVGPU to LLVM";
    let options = [
        Option<"conSan", "consan", "bool", /*default*/"false",
               "Use normal region simplification for ConSan-instrumented IR">
    ];
    let description = [{

    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::nvgpu::NVGPUDialect"];
}

#endif // NVGPU_CONVERSION_PASSES
