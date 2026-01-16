#ifndef DIALECTPLUGIN_PASS
#define DIALECTPLUGIN_PASS

include "mlir/Pass/PassBase.td"


def DialectPluginMagicOp: Pass<"convert-plugin-gpu-to-llvm", "mlir::ModuleOp"> {
  let summary = "Converts PluginGPU Ops to LLVM";
  let constructor = "mlir::triton::plugin::createConvertPluginGPUToLLVMPass(4)";

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
   let options = [
        Option<"num_warps", "num-warps",
               "int32_t", /*default*/"4",
               "device compute capability">,
    ];
}

#endif // DIALECTPLUGIN_PASS
