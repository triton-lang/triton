#ifndef NVGPU_CONVERSION_PASSES
#define NVGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertNVGPUToLLVM : Pass<"convert-nv-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert NVGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertNVGPUToLLVMPass()";

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
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::nvgpu::NVGPUDialect"];
}

#endif
