#ifndef TRITONCOMMONGPU_CONVERSION_PASSES
#define TRITONCOMMONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateSharedMemory : Pass<"allocate-shared-memory", "mlir::ModuleOp"> {
    let summary = "Add metadata for shared memory allocation";
    let description = [{
      This pass uses the `ModuleAllocation` analysis to:
        - Annotate modules with an attribute with the amount of shared/local
          memory used.
        - Annotate operations with an offset into the total shared/local memory.
     }];

    let constructor = "mlir::triton::gpu::createAllocateSharedMemoryPass()";
}


def AllocateProtonSMEMBuffer : Pass<"allocate-proton-smem-buffer", "mlir::ModuleOp"> {
    let constructor = "mlir::triton::gpu::createAllocateProtonSMEMBufferPass(\"\")";

    let options = [
        Option<"targetArch", "target-arch", "std::string", /*default*/"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"customLDSLimit", "lds-limit", "int", /*default*/"0",
               "custom limit of LDS consumption, if not provided, maximum LDS size is used">,
    ];
}


def TritonGPUGlobalScratchAllocationPass : Pass<"tritongpu-global-scratch-memory-allocation", "mlir::ModuleOp"> {
  let summary = "Assign global scratch memory allocation";

  let description = [{
    Decide on global scratch space memory allocation and assign attributes to each allocation.
  }];

  let constructor = "mlir::triton::gpu::createTritonGPUGlobalScratchAllocationPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

#endif
