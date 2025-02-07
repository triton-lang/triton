#ifndef TRITONPROTON_CONVERSION_PASSES
#define TRITONPROTON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ProtonLoweringPass : Pass<"proton-lowering-pass", "mlir::ModuleOp"> {
    let constructor = "mlir::triton::createProtonLoweringPass()";

}
#endif
