#ifndef PROTONNVIDIAGPU_TO_LLVM_PASSES
#define PROTONNVIDIAGPU_TO_LLVM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertProtonNvidiaGPUToLLVM : Pass<"convert-proton-nvidia-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert ProtonGPU to LLVM";
    let description = [{
        Convert ProtonGPU to LLVM using Nvidia-specific lowering patterns.
    }];
    let constructor = "mlir::triton::proton::gpu::createConvertProtonNvidiaGPUToLLVMPass(80, 80)";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::proton::ProtonDialect",
                             "mlir::triton::proton::gpu::ProtonGPUDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"ptxVersion", "ptx-version",
               "int32_t", /*default*/"80",
               "PTX version">,
    ];
}

#endif // PROTONNVIDIAGPU_TO_LLVM_PASSES
