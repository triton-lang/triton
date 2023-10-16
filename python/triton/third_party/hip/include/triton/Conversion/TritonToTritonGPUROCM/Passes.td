#ifndef CONVERSION_TRITON_TO_TRITONGPUROCM_PASSES
#define CONVERSION_TRITON_TO_TRITONGPUROCM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonGPUROCM: Pass<"convert-triton-to-tritongpu", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonGPUROCM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonToTritonGPUROCMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             // TODO: Does this pass depend on SCF?
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu_rocm::TritonGPUROCMDialect"];

   let options = [
       Option<"numWarps", "num-warps",
              "int32_t", /*default*/"4",
              "number of warps">,

       Option<"threadsPerWarp", "threads-per-warp",
              "int32_t", /*default*/"64",
              "number of threads per warp">,
        Option<"numCTAs", "num-ctas",
              "int32_t", /*default*/"1",
              "number of ctas in a cga">,
        Option<"computeCapability", "compute-capability",
              "int32_t", /*default*/"80",
              "compute capability">
   ];
}

#endif
