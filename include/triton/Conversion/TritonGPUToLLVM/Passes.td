#ifndef TRITONGPU_CONVERSION_PASSES
#define TRITONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertTritonGPUToLLVM : Pass<"convert-triton-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonGPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"target", "target", "enum Target", "mlir::triton::Target::Default",
               "compile for target compatible LLVM",
               "llvm::cl::values("
               "clEnumValN(mlir::triton::Target::NVVM, \"nvvm\", \"compile for "
               "NVVM-compatible LLVM\"), "
               "clEnumValN(mlir::triton::Target::ROCDL, \"rocdl\", \"compile for "
               "ROCDL-compatible LLVM\"))">,
    ];
    let statistics = [
        Statistic<"numMemXfer", "num-mem-xfer", "Number of memory tranfer ops">,
        Statistic<"memXferBytesWithSMem", "mem-xfer-bytes-smem", "Bytes of memory tranfer involving smem">,
        Statistic<"memXferBytesWithGlobal", "mem-xfer-bytes-global", "Bytes of memory tranfer involving global mem">,
  ];
}

#endif
