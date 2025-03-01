#ifndef PROTONCOMMONGPU_CONVERSION_PASSES
#define PROTONCOMMONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateProtonSharedMemory : Pass<"allocate-proton-shared-memory", "mlir::ModuleOp"> {
    let summary = "Update metadata for proton shared memory allocation";
    let description = [{
      This pass updates the amount of shared/local memory used due to
      proton intra kernel profiling.
     }];

    let constructor = "mlir::triton::proton::gpu::createAllocateProtonSharedMemoryPass()";
}


def AddProtonKernelArg : Pass<"add-proton-kernel-arg", "mlir::ModuleOp"> {

    let constructor = "mlir::triton::proton::gpu::createAddProtonKernelArgPass()";
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}
#endif
