#ifndef TRITONGPU_PLUGIN_PASSES
#define TRITONGPU_PLUGIN_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUMLIRPlugin : Pass<"tritongpu-plugin", "mlir::ModuleOp"> {
  let summary = "Triton MLIR Plugin Pass";
}
#endif
