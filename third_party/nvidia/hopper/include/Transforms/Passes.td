#ifndef NV_TRANSFORMS_PASSES
#define NV_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def NVGWarpSpecialization : Pass<"nvg-warp-specialization", "mlir::ModuleOp"> {
  let summary = "Automatic Warp specialization for NVIDIA GPU";

  let description = [{
    This pass automatically partitions user-defined kernels into
    warp-specialized kernels, enabling finer-grained scheduling
    and improved utilization of hardware resources.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"0",
           "number of buffers for warp specialization">,
    Option<"dumpIntermediateSteps", "dump-intermediate-steps",
           "bool", /*default*/"false",
           "Dump intermediate steps">
  ];
}

def NVGTestWSTaskPartition : Pass<"nvg-test-ws-task-partition", "mlir::ModuleOp"> {
  let summary = "test warp specialization task partition";

  let description = "This pass computes a warp schedule partition by annoating anchor operations with async task ids";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">
  ];
}

def NVGTestWSTaskIdPropagate : Pass<"nvg-test-taskid-propagate", "mlir::ModuleOp"> {
  let summary = "test warp specialization task id propagation";

  let description = [{
    This pass propagates the `async_task_id` annotation to the dependencies
    of any op that has it set.  This has the functional effect of partitioning
    the graph into multiple async tasks, based on the initial annotation.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];

  let options = [
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">
  ];
}

def NVGTestWSDataPartition : Pass<"nvg-test-ws-data-partition", "mlir::ModuleOp"> {
  let summary = "test warp specialization data partition";

  let description = "This pass partitions operations into multiple suboperations which operate on smaller data shapes";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect"];
  let options = [
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">
  ];
}

def NVGTestWSCodePartition: Pass<"nvg-test-ws-code-partition", "mlir::ModuleOp"> {
  let summary = "test warp specialization code partition";

  let description = "This pass generates warp specialized code baed on task id attributes.";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::TritonDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::nvws::NVWSDialect"];
  let options = [
    Option<"numBuffers", "num-buffers",
           "int32_t", /*default*/"0",
           "number of buffering for producer-consumer">,
    Option<"numWarpGroups", "num-warp-groups",
           "int32_t", /*default*/"0",
           "number of warp groups for warp specialization">,
    Option<"requestedRegisters", "requested-registers",
           "int32_t", /*default*/"232",
           "number of register requested for computation group">
  ];
}

#endif // NV_TRANSFORMS_PASSES
