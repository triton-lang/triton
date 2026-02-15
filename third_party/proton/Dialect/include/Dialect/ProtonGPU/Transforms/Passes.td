#ifndef PROTONGPU_TRANSFORMS_PASSES
#define PROTONGPU_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def ScheduleBufferStorePass: Pass<"proton-schedule-buffer-store", "mlir::ModuleOp"> {
  let summary = "Pass to move all Proton buffer stores to the end of the function";

  let description = "This pass makes the measurement more accurate by moving the expensive "
                    "shared memory stores to the end of the measured region after the measurements.";

  let dependentDialects = ["gpu::ProtonGPUDialect"];
}

#endif  // PROTONGPU_TRANSFORMS_PASSES
