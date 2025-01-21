#ifndef TRITONPROTON_CONVERSION_PASSES
#define TRITONPROTON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateProtonSMEMBuffer : Pass<"allocate-proton-smem-buffer", "mlir::ModuleOp"> {
    let constructor = "mlir::triton::createAllocateProtonSMEMBuffer()";

}
#endif
