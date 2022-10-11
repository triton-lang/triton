#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUPipeline : Pass<"tritongpu-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    TODO  
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
}

def TritonGPUSwizzle : Pass<"tritongpu-swizzle", "mlir::ModuleOp"> {
  let summary = "swizzle";

  let description = [{
    Inserts conversions to swizzled layout so as to avoid shared memory bank conflicts.
  }];

  let constructor = "mlir::createTritonGPUSwizzlePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
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
