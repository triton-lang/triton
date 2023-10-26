#ifndef CONVERSION_TRITONGPUROCM_TO_LLVM_PASSES
#define CONVERSION_TRITONGPUROCM_TO_LLVM_PASSES

include "mlir/Pass/PassBase.td"


def ConvertTritonGPUROCMToLLVM : Pass<"convert-triton-gpurocm-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPUROCM to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonGPUROCMToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu_rocm::TritonGPUROCMDialect",
                             "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"tmaMetadata", "tma-metadata",
               "mlir::triton::gpu::TMAMetadataTy*", /*default*/"nullptr",
               "tma metadata to the runtime">,
        Option<"target", "target", "enum Target", "mlir::triton::Target::Default",
               "compile for target compatible LLVM",
               "llvm::cl::values("
               "clEnumValN(mlir::triton::Target::NVVM, \"nvvm\", \"compile for "
               "NVVM-compatible LLVM\"), "
               "clEnumValN(mlir::triton::Target::ROCDL, \"rocdl\", \"compile for "
               "ROCDL-compatible LLVM\"))">,
    ];
}

#endif
