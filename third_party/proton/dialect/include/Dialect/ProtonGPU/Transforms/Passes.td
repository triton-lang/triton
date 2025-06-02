#ifndef PROTONGPU_TRANSFORMS_PASSES
#define PROTONGPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MoveProtonStoresToEndPass: Pass<"move-proton-stores-to-end", "mlir::ModuleOp"> {
  let summary = "Pass to move all Proton stores to the end of the function";

  let description = "This pass makes the measurement more accurate by moving the expensive "
                    "shared memory stores to the end of the measured reagion.";

  let dependentDialects = ["gpu::ProtonGPUDialect"];
}

#endif  // PROTONGPU_TRANSFORMS_PASSES
