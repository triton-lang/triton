#ifndef PROTON_PASSES
#define PROTON_PASSES

include "mlir/Pass/PassBase.td"

def ProtonLowering: Pass<"proton-lowering", "mlir::ModuleOp"> {
  let summary = "Lower Proton's RecordOp with scaffolding code (e.g., resource allocation).";

  let description = "Allocate memory for local profiling buffers and convert the RecordOp into "
                "LocalRecordOp with resource allocated and binded. Before exiting, the local profiling buffers "
                "(with metadata) are copied back to GPU's global memory.";

  let constructor = "mlir::triton::proton::createProtonLowering(0, 0, 0)";

  let dependentDialects = ["mlir::triton::proton::ProtonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

    let options = [
    Option<"maxSharedMem", "max-shared-mem",
           "int32_t", /*default*/"0",
           "max shared memory size per CTA">,
    Option<"scratchMem", "scratch-mem",
           "int32_t", /*default*/"0",
           "profiler's global scratch memory size per CTA">,
    Option<"alignment", "alignment",
           "int32_t", /*default*/"0",
           "profiler's global scratch memory alignment">,
  ];
}

#endif
