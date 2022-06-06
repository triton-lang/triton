#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUPipeline : Pass<"tritongpu-pipeline", "mlir::ModuleOp"> {
  let summary = "pipeline";

  let description = [{
    scf.for() {
      %a = load %a_ptr;
      %b = load %b_ptr;

      %d = dot %a, %b, %c;
    }

    =>

    ...
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

def TritonGPUCombineOps : Pass<"tritongpu-combine", "mlir::ModuleOp"> {
  let summary = "combine triton gpu ops";

  let description = [{
    convert_layout(load(%ptr, %mask, %other), #SMEM_LAYOUT) => 
      copy_async(%ptr, %mask, %other), barrier

    convert_layout(convert_layout(%src, #LAYOUT_0), #LAYOUT_1) =>
      convert_layout(%src, #LAYOUT_1)

    convert_layout(%src, #LAYOUT) => %src if %src.layout() == #LAYOUT
  }];

  let constructor = "mlir::createTritonGPUCombineOpsPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonGPUVerifier : Pass<"tritongpu-verifier", "mlir::ModuleOp"> {
  let summary = "verify TritonGPU IR";

  let description = [{}];

  let constructor = "mlir::createTritonGPUVerifier()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
}

#endif
