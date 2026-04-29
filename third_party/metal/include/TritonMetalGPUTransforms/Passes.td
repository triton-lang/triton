#ifndef TRITONGPU_PASSES
#define TRITONGPU_PASSES

include "mlir/Pass/PassBase.td"

// def TritonMetalGPUAccelerateMatmul : Pass<"accelerate-metal-matmul", "mlir::ModuleOp"> {
//   let summary = "accelerate matmul";

//   let dependentDialects = ["mlir::triton::TritonDialect",
//                            "mlir::triton::gpu::TritonGPUDialect"];
// }

def TritonMetalGPUAllocateSmemForSimdgroupMatmul : Pass<"allocate-smem-for-simdgroup-matmul", "mlir::ModuleOp"> {
  let summary = "allocate shared memory for simdgroup matmul";
  let dependentDialects = ["mlir::triton::TritonDialect",
                           "mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::metalgpu::TritonMetalGPUDialect"];
}

#endif // TRITONGPU_PASSES