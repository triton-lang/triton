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

  let constructor = "mlir::triton::nvidia_gpu::createTritonNvidiaGPUPlanCTAPass()";

  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonGPUFenceInsertion : Pass<"triton-nvidia-gpu-fence-insertion", "mlir::ModuleOp"> {
  let summary = "Insert fences across generic and async proxy.";

  let description = [{
    This pass is to insert memory fences to ensure that memory operations are
    properly ordered across generic and async operations.
    This pass inserts fences at optimized location.
    There is a pass later to handle all the functional requirements
  }];

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

def TritonGPUProxyFenceInsertion : Pass<"triton-nvidia-gpu-proxy-fence-insertion", "mlir::ModuleOp"> {
  let summary = "Insert fences across generic and async proxy";

  let description = [{
    This pass is to insert memory fences to ensure that memory operations are
    properly ordered across generic and async operations.
  }];

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
    Lower Triton descriptor load to TMA load/store operations in TritonNvidiaGPUDialect.
  }];

  let dependentDialects = [
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonTensorMemoryAllocationPass : Pass<"triton-tensor-memory-allocation", "mlir::ModuleOp"> {
  let summary = "Assign tensor memory allocation";

  let description = [{
    Decide on tensor memory allocation and assign attributes to each allocation.
  }];

  let dependentDialects = [
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonNvidiaGPUMMALoweringPass : Pass<"triton-nvidia-mma-lowering", "mlir::ModuleOp"> {
  let summary = "lower mma operations if needed";

  let description = [{
    Lower MMA ops to prepare for conversion to LLVM.
  }];

  let dependentDialects = [
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def TritonNvidiaGPUPromoteLHSToTMemPass : Pass<"tritongpu-promote-lhs-to-tmem", "mlir::ModuleOp"> {
  let summary = "Promote LHS operand of MMAv5 op to Tensor Memory";

  let description = [{
    Promote LHS operand of MMAv5 op to Tensor Memory.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonNvidiaGPUOptimizeDescriptorEncodingPass : Pass<"triton-nvidia-optimize-descriptor-encoding", "mlir::ModuleOp"> {
  let summary = "Set encodings on tensor descriptor types";

  let description = [{
    Set shared memory encoding on tensor descriptors, which decides the swizzling mode and message size of the tma descriptor.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonNvidiaGPUOptimizeTMemLayoutsPass : Pass<"triton-nvidia-optimize-tmem-layouts", "mlir::ModuleOp"> {
  let summary = "Optimize TMEM layouts.";

  let description = [{
    Optimize TMEM layouts by selecting a layouts to enable better subtiling,
    reduction performance, etc.
  }];

  let dependentDialects = ["mlir::triton::gpu::TritonGPUDialect",
                           "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                           "mlir::triton::TritonDialect"];
}

def TritonNvidiaGPUInterleaveTMemPass : Pass<"triton-nvidia-interleave-tmem", "mlir::ModuleOp"> {
  let summary = "Interleave TMEM loads/stores.";

  let description = [{
    The `triton-nvidia-interleave-tmem` pass attempts to sink TMEM loads and
    hoist TMEM stores, and potentially interleave them, to reduce register
    pressure.
  }];
}

def TritonNvidiaGPURemoveTMEMTokensPass : Pass<"triton-nvidia-gpu-remove-tmem-tokens", "mlir::ModuleOp"> {
  let summary = "remove TMEM tokens";

  let description = [{
    The `triton-nvidia-gpu-remove-tmem-tokens` pass removes TMEM memory
    dependency tokens from the IR, after they are no longer needed.
  }];
}

#endif
