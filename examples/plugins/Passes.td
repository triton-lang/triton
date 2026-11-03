#ifndef TRITONGPU_PLUGIN_PASSES
#define TRITONGPU_PLUGIN_PASSES

include "mlir/Pass/PassBase.td"

def TritonGPUMLIRPlugin : Pass<"tritongpu-plugin", "mlir::ModuleOp"> {
  let summary = "Triton MLIR Plugin Pass";

     let options = [
        Option<"num_warps", "num-warps",
               "int32_t", /*default*/"4",
               "Number of warps">,
     ];

}
#endif
