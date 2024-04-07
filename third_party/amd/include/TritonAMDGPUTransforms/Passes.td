#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonAMDGPUPipeline : Pass<"tritonamdgpu-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Applies software pipelining to loops in the module based on number of stages.
    This may convert some load into asynchronous loads, and multi-buffer the data.
  }];

  let constructor = "mlir::createTritonAMDGPUPipelinePass()";

  let dependentDialects = [];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages">,
    Option<"numWarps", "num-warps",
           "int32_t", /*default*/"4",
           "number of warps per block">,
    Option<"numCTAs", "num-ctas",
           "int32_t", /*default*/"1",
           "number of CTAs per CGA">,
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "device compute capability">
  ];
}

def TritonAMDGPUStreamPipeline : Pass<"tritonamdgpu-stream-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Pipeline global loads through registers to shared memory while computing on previous
    tile
  }];

  let constructor = "mlir::createTritonAMDGPUStreamPipelinePass()";

  let dependentDialects = [];
}

def TritonAMDGPUPrefetch : Pass<"tritonamdgpu-prefetch", "mlir::ModuleOp"> {
  let summary = "prefetch";

  let description = [{
    Decompose `DotOp` instructions in loops into several finer-grained `DotOp`
    that may have their operands constructed at the end of the previous iteration
  }];

  let constructor = "mlir::createTritonAMDGPUPrefetchPass()";

  let dependentDialects = [];
}

def TritonAMDGPUAccelerateMatmul : Pass<"tritonamdgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., AMD matrix cores)
  }];

  let constructor = "mlir::createTritonAMDGPUAccelerateMatmulPass()";

  let dependentDialects = [];

  let options = [
    Option<"archGenerationName", "arch-generation-name",
           "std::string", /*default=*/"std::string{}",
           "GFX generation name of target device.">,
    Option<"matrixInstructionSize", "matrix-instruction-size",
           "int32_t", /*default*/"0",
           "enforce matrix instruction MN size">
  ];
}

def TritonAMDGPUOptimizeDotOperands : Pass<"tritonamdgpu-optimize-dot-operands", "mlir::ModuleOp"> {
  let summary = "fuse transpositions";

  let description = [{
    Re-arranged layouts of tensors used as matrix multiplication operands so as to promote the use of
    hardware-accelerated transpositions.
  }];

  let constructor = "mlir::createTritonAMDGPUOptimizeDotOperandsPass()";

  let dependentDialects = [];
}

def TritonAMDGPUCoalesce: Pass<"tritonamdgpu-coalesce", "mlir::ModuleOp"> {
  let summary = "coalesce";

  let description = [{
    TODO
  }];

  let constructor = "mlir::createTritonAMDGPUCoalescePass()";

  let dependentDialects = [];
}


def TritonAMDGPURemoveLayoutConversions : Pass<"tritonamdgpu-remove-layout-conversions", "mlir::ModuleOp"> {
  let summary = "remove superfluous layout conversions";

  let description = [{
  }];

  let constructor = "mlir::createTritonAMDGPURemoveLayoutConversionsPass()";

  let dependentDialects = [];

}

def TritonAMDGPUOptimizeEpilogue : Pass<"tritonamdgpu-optimize-epilogue", "mlir::ModuleOp"> {
  let summary = "Optimize epilogue: (1) Store accumulators directly without going thorough SMEM in epilogue.";

  let description = [{
  }];

  let constructor = "mlir::createTritonAMDGPUOptimizeEpiloguePass()";

  let dependentDialects = [];

}

def TritonAMDGPUReorderInstructions: Pass<"tritonamdgpu-reorder-instructions", "mlir::ModuleOp"> {
  let summary = "Reorder instructions";

  let description = "This pass reorder instructions so as to (1) decrease register pressure (e.g., by moving "
                    "conversions from shared memory before their first use) and (2) promote LLVM instruction "
                    "order more friendly to `ptxas`.";

  let constructor = "mlir::createTritonAMDGPUReorderInstructionsPass()";

  let dependentDialects = [];
}

def TritonAMDGPUDecomposeConversions: Pass<"tritonamdgpu-decompose-conversions", "mlir::ModuleOp"> {
  let summary = "Decompose convert[distributed -> dotOperand] into convert[distributed -> shared -> dotOperand]";

  let description = "Decomposing conversions this way makes it possible to use CSE and re-use #shared tensors";

  let constructor = "mlir::createTritonAMDGPUDecomposeConversionsPass()";

  let dependentDialects = [];
}

#endif
