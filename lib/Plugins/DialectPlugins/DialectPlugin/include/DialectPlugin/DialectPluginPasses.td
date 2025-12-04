#ifndef DIALECTPLUGIN_PASS
#define DIALECTPLUGIN_PASS

include "mlir/Pass/PassBase.td"


def DialectPluginMagicOp: Pass<"convert-plugin-gpu-to-llvm", "mlir::ModuleOp"> {
  let summary = "Converts a MagicOp to LLVM";
  let constructor = "mlir::triton::dialectplugin::createConvertPluginGPUToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect"];
}

#endif // DIALECTPLUGIN_PASS
