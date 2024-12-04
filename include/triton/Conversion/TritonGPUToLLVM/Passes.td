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
