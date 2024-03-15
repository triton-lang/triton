#ifndef TRITONAMDGPU_CONVERSION_PASSES
#define TRITONAMDGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def DecomposeUnsupportedAMDConversions : Pass<"decompose-unsupported-amd-conversions", "mlir::ModuleOp"> {
    let summary = "Decompose conversions that are not supported by TritonGPU -> LLVM";
    let constructor = "mlir::triton::gpu::createDecomposeUnsupportedAMDConversionsPass()";
}

def ConvertTritonAMDGPUToLLVM : Pass<"convert-triton-amdgpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonAMDGPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
    ];
}

#endif
