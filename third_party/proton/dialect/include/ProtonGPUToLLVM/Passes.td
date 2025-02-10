#ifndef PROTON_GPU_CONVERSION_PASSES
#define PROTON_GPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ProtonLoweringPass : Pass<"proton-lowering-pass", "mlir::ModuleOp"> {
    let constructor = "mlir::triton::createProtonLoweringPass()";

    let summary = "Lower Proton dialect to LLVM dialect";

    let description = [{
        "This pass lowers the Proton dialect to the LLVM dialect."
    }];
}
#endif
