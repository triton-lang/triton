#ifndef TRITONMTGPU_CONVERSION_PASSES
#define TRITONMTGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonMTGPUToLLVM : Pass<"convert-triton-mtgpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonMTGPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::MTGPU::MTGPUDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
    ];
}

def ConvertMTGPUBuiltinFuncToLLVM : Pass<"convert-mtgpu-builtin-func-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert MTGPU Builtin Func to LLVM";
    let constructor = "mlir::triton::createConvertMTGPUBuiltinFuncToLLVMPass()";

    let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

#endif // TRITONMTGPU_CONVERSION_PASSES
