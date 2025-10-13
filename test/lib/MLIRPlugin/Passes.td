#ifndef TRITONGPU_MLIRPLUGIN_PASSES
#define TRITONGPU_MLIRPLUGIN_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUHelloMLIRPlugin : Pass<"tritongpu-HelloMLIRPlugin", "mlir::ModuleOp"> {
  let summary = "Hello World Triton MLIR Plugin Pass";
}
#endif
