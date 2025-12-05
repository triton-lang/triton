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

#endif // DIALECTPLUGIN_PASS
