//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TT2TTX_CONVERSION_PASSES
#define TT2TTX_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonToTritonXPU: Pass<"convert-triton-to-triton-xpu", "mlir::ModuleOp"> {
    let summary = "Convert Triton to TritonXPU";
    let description = [{

    }];
    let constructor = "mlir::triton::createConvertTritonToTritonXPUPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             // TODO: Does this pass depend on SCF?
                             "mlir::scf::SCFDialect",
                             "mlir::triton::TritonDialect",
                             // "mlir::triton::gpu::TritonGPUDialect", // Does this pass depend on TritonGPU in Triton 3.0?
                             "mlir::triton::xpu::TritonXPUDialect"];

   let options = [
       Option<"xpu_arch", "xpu_arch",
               "uint32_t", /*default*/"3",
               "XPU Architecture">,
       Option<"buffer_size", "buffer_size",
              "uint32_t", /*default*/"512",
              "bytes for local memory buffer">,
       Option<"core_num", "core_num",
              "uint32_t", /*default*/"64",
              "xpu spec core_num">
   ];
}


#endif // TT2TTX_CONVERSION_PASSES
