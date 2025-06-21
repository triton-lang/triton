#ifndef PROTONGPU_TO_LLVM_PASSES
#define PROTONGPU_TO_LLVM_PASSES

include "mlir/Pass/PassBase.td"

def AllocateProtonSharedMemoryPass : Pass<"allocate-proton-shared-memory", "mlir::ModuleOp"> {
    let summary = "Update metadata for proton shared memory allocation";
    let description = [{
      This pass updates the amount of shared/local memory used by
      proton intra kernel profiling.
     }];

    let dependentDialects = ["ProtonDialect",
                             "gpu::ProtonGPUDialect"];
}

def AllocateProtonGlobalScratchBufferPass : Pass<"allocate-proton-global-scratch-buffer", "mlir::ModuleOp"> {
    let summary = "Update metadata for proton global scratch buffer allocation";
    let description = [{
      This pass updates the amount of global memory used by
      proton intra kernel profiling.
     }];

    let dependentDialects = ["ProtonDialect",
                             "gpu::ProtonGPUDialect"];
}

def AddSchedBarriers : Pass<"add-sched-barriers", "mlir::ModuleOp"> {
    let constructor = "mlir::triton::proton::gpu::createAddSchedBarriersPass()";
    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::ROCDL::ROCDLDialect"];
}

#endif // PROTONGPU_TO_LLVM_PASSES
