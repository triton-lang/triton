#ifndef TRITONPROTON_CONVERSION_PASSES
#define TRITONPROTON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateProtonDeviceBuffer : Pass<"allocate-proton-device-buffer", "mlir::ModuleOp"> {
    let constructor = "mlir::triton::createAllocateProtonDeviceBuffer()";

}
#endif
