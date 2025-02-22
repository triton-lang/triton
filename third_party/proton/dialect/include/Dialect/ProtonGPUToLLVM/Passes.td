#ifndef PROTONGPU_CONVERSION_PASSES
#define PROTONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertProtonGPUToLLVM : Pass<"convert-proton-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert ProtonGPU to LLVM";
    let constructor = "mlir::triton::createConvertProtonGPUToLLVMPass(\"\")";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "target device architecture, e.g., gfx942">,
    ];
}

#endif
