// Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#ifndef TRITONNVIDIAGPU_PASSES
#define TRITONNVIDIAGPU_PASSES

include "mlir/Pass/PassBase.td"

def MaterializeLoadStore : Pass<"triton-nvidia-gpu-materialize-load-store", "mlir::ModuleOp"> {
  let summary = "materialize load & store";

  let description = [{
    This pass works after pipeline pass, converting the remaining tt.LoadOp taking
    ptr<tensor> as input into ttg.InsertSliceAsyncOp and emit proper barriers
  }];

  let constructor = "mlir::createTritonNvidiaGPUMaterializeLoadStorePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"numWarps", "num-warps",
           "int32_t", /*default*/"4",
           "number of warps per block">,
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "device compute capability">
  ];
}

def TritonGPUPlanCTAPass : Pass<"triton-nvidia-gpu-plan-cta", "mlir::ModuleOp"> {
  let summary = "plan CTA";

  let description = [{
    Plan CTAs in CGA
  }];

  let constructor = "mlir::createTritonNvidiaGPUPlanCTAPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonGPUWSFeasibilityChecking : Pass<"triton-nvidia-gpu-ws-feasibility-checking", "mlir::ModuleOp"> {
  let summary = "Attach attr named TritonNvidiaGPUDialect::getWSSupportedAttrName() if auto WS supported";

  let description = [{
    Since not every legal triton kernels can be auto WS, this pass does some (conservative) check
    and attaches an attribute named TritonNvidiaGPUDialect::getWSSupportedAttrName() on
    the input module op if the kernel is supported.
  }];

  let constructor = "mlir::createTritonNvidiaGPUWSFeasibilityCheckingPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"90",
           "device compute capability">
  ];
}

def TritonGPUWSDecomposing : Pass<"triton-nvidia-gpu-ws-decomposing", "mlir::ModuleOp"> {
  let summary = "Clustering on the ops according to their performance hotspots";

  let description = [{
    Based on compute capability and heuristics,
    this pass will identify some operations to be executed in different agents,
    by marking them with async 'label'. E.g.,
    input:
      %1 = tt,load %0 ...
      %4 = tt.dot %1, %2, %3 ...
    output:
      %1 = tt,load %0 {async_agent = 0} ...
      %4 = tt.dot %1, %2, %3 {async_agent = 1} : ...
  }];

  let constructor = "mlir::createTritonNvidiaGPUWSDecomposingPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"80",
           "device compute capability">
  ];
}

def TritonGPUWSPipeline : Pass<"triton-nvidia-gpu-ws-pipeline", "mlir::ModuleOp"> {
  let summary = "Warp specialization pipeline";

  let description = [{
  }];

  let constructor = "mlir::createTritonNvidiaGPUWSPipelinePass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::scf::SCFDialect",
                           "mlir::arith::ArithDialect"];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of pipeline stages">,
    Option<"numWarps", "num-warps",
           "int32_t", /*default*/"12",
           "number of warps per block">,
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"90",
           "device compute capability">
  ];
}

def TritonGPUWSMaterialization : Pass<"triton-nvidia-gpu-ws-materialization", "mlir::ModuleOp"> {
  let summary = "Warp specialization materialization";

  let description = [{
  }];

  let constructor = "mlir::createTritonNvidiaGPUWSMaterializationPass()";

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"];

  let options = [
    Option<"computeCapability", "compute-capability",
           "int32_t", /*default*/"90",
           "device compute capability">
  ];
}

#endif
