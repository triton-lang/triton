#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUPipeline : Pass<"tritongpu-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Replace `LoadOp` in loops by `InsertSliceAsyncOp` instructions that asynchronously construct the data
    needed at the next iteration
  }];

  let constructor = "mlir::createTritonGPUPipelinePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages">
  ];
}

def TritonGPUPrefetch : Pass<"tritongpu-prefetch", "mlir::ModuleOp"> {
  let summary = "prefetch";

  let description = [{
    Decompose `DotOp` instructions in loops into several finer-grained `DotOp`
    that may have their operands constructed at the end of the previous iteration
  }];

  let constructor = "mlir::createTritonGPUPrefetchPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonGPUAccelerateMatmul : Pass<"tritongpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., Nvidia tensor cores)
  }];

  let constructor = "mlir::createTritonGPUAccelerateMatmulPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "device compute capability">
  ];

}

def TritonGPUOptimizeDotOperands : Pass<"tritongpu-optimize-dot-operands", "mlir::ModuleOp"> {
  let summary = "fuse transpositions";

  let description = [{
    Re-arranged layouts of tensors used as matrix multiplication operands so as to promote the use of
    hardware-accelerated transpositions.
  }];

  let constructor = "mlir::createTritonGPUOptimizeDotOperandsPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUCoalesce: Pass<"tritongpu-coalesce", "mlir::ModuleOp"> {
  let summary = "coalesce";

  let description = [{
    TODO
  }];

  let constructor = "mlir::createTritonGPUCoalescePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}


def TritonGPURemoveLayoutConversions : Pass<"tritongpu-remove-layout-conversions", "mlir::ModuleOp"> {
  let summary = "remove superfluous layout conversions";

  let description = [{
  }];

  let constructor = "mlir::createTritonGPURemoveLayoutConversionsPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUReorderInstructions: Pass<"tritongpu-reorder-instructions", "mlir::ModuleOp"> {
  let summary = "Reorder instructions";

  let description = "This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving "
                    "conversions from shared memory before their first use) and (2) promote LLVM instruction "
                    "order more friendly to `ptxas`.";

  let constructor = "mlir::createTritonGPUReorderInstructionsPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUDecomposeConversions: Pass<"tritongpu-decompose-conversions", "mlir::ModuleOp"> {
  let summary = "Decompose convert[distributed -> dotOperand] into convert[distributed -> shared -> dotOperand]";

  let description = "Decomposing conversions this way makes it possible to use CSE and re-use #shared tensors";

  let constructor = "mlir::createTritonGPUDecomposeConversionsPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

#endif
