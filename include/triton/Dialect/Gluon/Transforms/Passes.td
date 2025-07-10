#ifndef GLUON_PASSES
#define GLUON_PASSES

include "mlir/Pass/PassBase.td"

def GluonResolveAutoEncodingsPass : Pass<"gluon-resolve-auto-encodings", "mlir::ModuleOp"> {
  let summary = "Resolve automatic encodings";
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
  ];

}

#endif
