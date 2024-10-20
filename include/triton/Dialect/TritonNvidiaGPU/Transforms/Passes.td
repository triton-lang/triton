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

def TritonGPUPlanCTAPass : Pass<"triton-nvidia-gpu-plan-cta", "mlir::ModuleOp"> {
  let summary = "plan CTA";

  let description = [{
    This pass computes and applies "optimized" CTA tilings to DotOp, ReduceOp
    and StoreLikeOps operations.
  }];

  let constructor = "mlir::createTritonNvidiaGPUPlanCTAPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonGPUFenceInsertion : Pass<"triton-nvidia-gpu-fence-insertion", "mlir::ModuleOp"> {
  let summary = "Insert fences across generic and async proxy";

  let description = [{
    This pass is to insert memory fences to ensure that memory operations are
    properly ordered across generic and async operations.
  }];

  let constructor = "mlir::createTritonNvidiaGPUFenceInsertionPass()";

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


def TritonNvidiaGPUTMALoweringPass : Pass<"triton-nvidia-tma-lowering", "mlir::ModuleOp"> {
  let summary = "lower to TMA load/store operations";

  let description = [{
    Lower Triton experimental descriptor load to TMA load/store operations in TritonNvidiaGPUDialect.
  }];

  let constructor = "mlir::createTritonNvidiaGPUTMALoweringPass()";

  let dependentDialects = [
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonNvidiaGPUGlobalScratchAllocationPass : Pass<"triton-nvidia-global-scratch-memory-allocation", "mlir::ModuleOp"> {
  let summary = "Assign global scratch memory allocation";

  let description = [{
    Decide on global scratch space memory allocation and assign attributes to each allocation.
  }];

  let constructor = "mlir::createTritonNvidiaGPUGlobalScratchAllocationPass()";

  let dependentDialects = [
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

#endif
