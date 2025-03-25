//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TTX2LLVM_CONVERSION_PASSES
#define TTX2LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonXPUToLLVM: Pass<"convert-triton-xpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonXPU to LLVM";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonXPUToLLVMPass()";

    let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

   let options = [
        Option<"xpu_arch", "xpu_arch",
               "uint32_t", /*default*/"3",
               "XPU Architecture">,
        Option<"buffer_size", "buffer_size",
               "uint32_t", /*default*/"128",
               "Buffer Size of LM">,
   ];
}


#endif // TTX2LLVM_CONVERSION_PASSES
