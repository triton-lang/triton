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
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

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

def TritonGPUStreamPipeline : Pass<"tritongpu-stream-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Pipeline global loads through registers to shared memory while computing on previous
    tile
  }];

  let constructor = "mlir::createTritonGPUStreamPipelinePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];
}

def TritonAMDGPUDotSlicing: Pass<"tritonamdgpu-dot-slicing", "mlir::ModuleOp"> {
  let summary = "'DotOp' instruction slicing";

  let description = [{
    Slice 'DotOp' instruction into multiple smaller 'DotOp' instructions
    in order to improve scheduling and latency hiding.
  }];

  let constructor = "mlir::createTritonAMDGPUDotSlicingPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"sliceKTile", "slice-k-tile",
           "int32_t", /*default*/"0",
           "slice size in k dimension">
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
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "device compute capability">
  ];
}

def TritonAMDGPUAccelerateMatmul : Pass<"tritonamdgpu-accelerate-matmul", "mlir::ModuleOp"> {
  let summary = "accelerate matmul";

  let description = [{
    Optimize the input/output layout of `dot` instruction to make them compatible hardware accelerators
    (e.g., AMD matrix cores)
  }];

  let constructor = "mlir::createTritonAMDGPUAccelerateMatmulPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"archGenerationName", "arch-generation-name",
           "std::string", /*default=*/"std::string{}",
           "GFX generation name of target device.">,
    Option<"matrixInstructionSize", "matrix-instruction-size",
           "int32_t", /*default*/"0",
           "enforce matrix instruction MN size">,
    Option<"enableWmmaTransform", "enable-wmma-transform",
           "bool", /*default*/"false",
           "temporary option, required for lit tests only">
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
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
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

def TritonGPUOptimizeEpilogue : Pass<"tritongpu-optimize-epilogue", "mlir::ModuleOp"> {
  let summary = "Optimize epilogue: (1) Store accumulators directly without going thorough SMEM in epilogue.";

  let description = [{
  }];

  let constructor = "mlir::createTritonGPUOptimizeEpiloguePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

}

def TritonGPUOptimizeThreadLocality : Pass<"tritongpu-optimize-thread-locality", "mlir::ModuleOp"> {
  let summary = "Reduce the cost of synchronization between threads in an SM";

  let description = [{
    Today, this optimizes reduction yielded by loop to be thread-local until after the loop completes.
  }];

  let constructor = "mlir::createTritonGPUOptimizeThreadLocalityPass()";

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
