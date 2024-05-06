#ifndef TRITONCPU_CONVERSION_PASSES
#define TRITONCPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonCPUToLLVM : Pass<"convert-triton-cpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonCPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonCPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::math::MathDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::tensor::TensorDialect",
                             "mlir::triton::cpu::TritonCPUDialect",
                             "mlir::triton::TritonDialect"];

    let options = [
    ];
}

#endif
