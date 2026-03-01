#ifndef DIALECTPLUGIN_PASS
#define DIALECTPLUGIN_PASS

include "mlir/Pass/PassBase.td"


def DialectPluginMagicOp: Pass<"convert-plugin-gpu-to-llvm", "mlir::ModuleOp"> {
  let summary = "Converts PluginGPU Ops to LLVM";
  let constructor = "mlir::triton::plugin::createConvertPluginGPUToLLVMPass(80, 80)";

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
   let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"ptxVersion", "ptx-version",
               "int32_t", /*default*/"80",
               "PTX version">,
    ];
}

def DialectPluginFMagicOp: Pass<"convert-plugin-gpu-to-triton-gpu", "mlir::ModuleOp"> {
  let summary = "Converts PluginGPU Ops to TritonGPU Ops";
  let constructor = "mlir::triton::plugin::createConvertPluginGPUToTritonGPUPass(4, 32, 1)";

   let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                            "mlir::triton::plugin::DialectPluginDialect"];
   let options = [
        Option<"num_warps", "num-warps",
               "int32_t", /*default*/"4",
               "Number of warps">,
        Option<"threadsPerWarp", "threads-per-warp",
               "int32_t", /*default*/"32",
               "Threads per warp">,
        Option<"numCTAs", "num-ctas",
               "int32_t", /*default*/"1",
               "Number of CTAs">,
    ];
}

#endif // DIALECTPLUGIN_PASS
