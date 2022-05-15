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

  let constructor = "mlir::triton::gpu::createPipelinePass";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithmeticDialect"];
}

#endif
