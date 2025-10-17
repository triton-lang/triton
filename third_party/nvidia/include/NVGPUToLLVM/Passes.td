#ifndef NVG_CONVERSION_PASSES
#define NVG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertNVGToLLVM : Pass<"convert-nv-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert NVG to LLVM";
    let description = [{

    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::nvg::NVGDialect"];
}

#endif // NVG_CONVERSION_PASSES
