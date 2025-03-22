#ifndef PROTONGPU_TO_LLVM_PASSES
#define PROTONGPU_TO_LLVM_PASSES

include "mlir/Pass/PassBase.td"

def AllocateProtonSharedMemory : Pass<"allocate-proton-shared-memory", "mlir::ModuleOp"> {
    let summary = "Update metadata for proton shared memory allocation";
    let description = [{
      This pass updates the amount of shared/local memory used due to
      proton intra kernel profiling.
     }];

    let constructor = "mlir::triton::proton::gpu::createAllocateProtonSharedMemoryPass()";
}

def AllocateProtonGlobalScratchBuffer : Pass<"allocate-proton-global-scratch-buffer", "mlir::ModuleOp"> {
    let constructor = "mlir::triton::proton::gpu::createAllocateProtonGlobalScratchBufferPass()";
}

#endif // PROTONGPU_TO_LLVM_PASSES
