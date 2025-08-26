#ifndef TRITONGPU_EXTENSION_PASSES
#define TRITONGPU_EXTENSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUHelloExtension : Pass<"tritongpu-HelloExtension", "mlir::ModuleOp"> {
  let summary = "Hello World Extension";
  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}
#endif
