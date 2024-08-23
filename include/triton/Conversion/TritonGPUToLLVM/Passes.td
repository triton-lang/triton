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

#endif
