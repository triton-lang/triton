#ifndef PROTONAMDGPU_TO_LLVM_PASSES
#define PROTONAMDGPU_TO_LLVM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertProtonAMDGPUToLLVM : Pass<"convert-proton-amd-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert ProtonGPU to LLVM";
    let description = [{
        Convert ProtonGPU to LLVM using AMD-specific lowering patterns.
    }];
    let constructor = "mlir::triton::proton::gpu::createConvertProtonAMDGPUToLLVMPass(\"\")";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::amdgpu::TritonAMDGPUDialect",
                             "mlir::triton::proton::ProtonDialect",
                             "mlir::triton::proton::gpu::ProtonGPUDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">
    ];
}

#endif // PROTONAMDGPU_TO_LLVM_PASSES
