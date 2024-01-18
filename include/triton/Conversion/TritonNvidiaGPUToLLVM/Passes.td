#ifndef TRITONNVIDIAGPU_CONVERSION_PASSES
#define TRITONNVIDIAGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def ConvertTritonNvidiaGPUToLLVM : Pass<"convert-tritonnvidia-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonNvidiaGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonNvidiaGPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];
}

#endif
