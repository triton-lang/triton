#ifndef TRITON_CONVERSION_PASSES
#define TRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonGPU: Pass<"convert-triton-to-tritongpu", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonGPU";
    let description = [{
        
    }];
    let constructor = "mlir::triton::createConvertTritonToTritonGPUPass()";
}

#endif
