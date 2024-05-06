#ifndef TRITON_CONVERSION_TO_CPU_PASSES
#define TRITON_CONVERSION_TO_CPU_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonCPU: Pass<"convert-triton-to-tritoncpu", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonCPU";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonToTritonCPUPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::triton::cpu::TritonCPUDialect",
                             "mlir::triton::TritonDialect"];

   let options = [
   ];
}

#endif
