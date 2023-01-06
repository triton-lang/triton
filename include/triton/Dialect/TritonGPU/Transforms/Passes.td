#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUPipeline : Pass<"tritongpu-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    Unroll loops to hide global memory -> shared memory latency.
  }];

  let constructor = "mlir::createTritonGPUPipelinePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithmeticDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"2",
           "number of pipeline stages">
  ];
}

def TritonGPUPrefetch : Pass<"tritongpu-prefetch", "mlir::ModuleOp"> {
  let summary = "prefetch";

  let description = [{
    Prefetch operands (a and b) of tt.dot into shared memory to hide shared memory -> register latency.
  }];

  let constructor = "mlir::createTritonGPUPrefetchPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithmeticDialect"];
}

def TritonGPUCoalesce: Pass<"tritongpu-coalesce", "mlir::ModuleOp"> {
  let summary = "coalesce";

  let description = [{
    TODO
  }];

  let constructor = "mlir::createTritonGPUCoalescePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

def TritonGPUCombineOps : Pass<"tritongpu-combine", "mlir::ModuleOp"> {
  let summary = "combine triton gpu ops";

  let description = [{
    convert_layout(convert_layout(%src, #LAYOUT_0), #LAYOUT_1) =>
      convert_layout(%src, #LAYOUT_1)

    convert_layout(%src, #LAYOUT) => %src if %src.layout() == #LAYOUT
  }];

  let constructor = "mlir::createTritonGPUCombineOpsPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "device compute capability">
  ];
}

def TritonGPUOptimizeLoadConvert: Pass<"tritongpu-optimize-load-convert", "mlir::ModuleOp"> {
  let summary = "Optimize load + convert into insert_slice_async + wait + extract_slice + convert";

  let description = "Transform load + convert into insert_slice_async + wait + extract_slice + convert."
                    "This decreases registers pressure on architecture with direct pathways between DRAM "
                    "and shared memory";

  let constructor = "mlir::createTritonGPUOptimizeLoadConvertPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUSinkConversionsFromShared: Pass<"tritongpu-sink-conversions-from-shared", "mlir::ModuleOp"> {
  let summary = "Sink conversions from shared into loops";

  let description = "This pass sinks conversions from shared memory into loops. This will lead the codegen "
                    "to keep data in shared memory throughout loops, which will reduce register pressure.";

  let constructor = "mlir::createTritonGPUSinkConversionsFromSharedPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUCanonicalizeLoops: Pass<"tritongpu-canonicalize-loops", "mlir::ModuleOp"> {
  let summary = "canonicalize scf.ForOp ops";

  let description = [{
    This implements some optimizations that are missing in the standard scf.ForOp
    canonicalizer.
  }];

  let constructor = "mlir::createTritonGPUCanonicalizeLoopsPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

#endif
