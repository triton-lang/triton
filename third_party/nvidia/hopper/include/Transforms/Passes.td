#ifndef NV_TRANSFORMS_PASSES
#define NV_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def NVGPUWarpSpecialization : Pass<"nvgpu-warp-specialization", "mlir::ModuleOp"> {
  let summary = "Automaticl Warp specialization for NVIDIA GPU";

  let description = [{
    This pass automatically partitions user-defined kernels into
    warp-specialized kernels, enabling finer-grained scheduling
    and improved utilization of hardware resources.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">
  ];
}

def NVGPUTestWSTaskPartition : Pass<"nvgpu-test-ws-task-partition", "mlir::ModuleOp"> {
  let summary = "test warp specialization task partition";

  let description = "This pass computes a warp schedule partition by annoating anchor operations with async task ids";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">
  ];
}

#endif // NV_TRANSFORMS_PASSES
