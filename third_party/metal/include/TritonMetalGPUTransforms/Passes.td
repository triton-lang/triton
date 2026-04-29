#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

// def TritonMetalGPUAccelerateMatmul : Pass<"accelerate-metal-matmul", "mlir::ModuleOp"> {
//   let summary = "accelerate matmul";

//   let dependentDialects = ["mlir::triton::TritonDialect",
//                            "mlir::triton::gpu::TritonGPUDialect"];
// }

def TritonMetalGPUInjectTensorStrideArgs : Pass<"inject-tensor-stride-args", "mlir::ModuleOp"> {
  let summary = "inject implicit stride pointer args after each tensor pointer arg in kernel functions";
  let dependentDialects = ["mlir::triton::TritonDialect"];
}

def TritonMetalGPUPrepareSimdgroupMatmul : Pass<"prepare-simdgroup-matmul", "mlir::ModuleOp"> {
  let summary = "prepare simdgroup matmul";
  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::metalgpu::TritonMetalGPUDialect"];
}

#endif // TRITONGPU_PASSES