// Copyright (c) 2025 NVIDIA Corporation & Affiliates. All rights reserved.
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

#ifndef NVWS_PASSES
#define NVWS_PASSES

include "mlir/Pass/PassBase.td"

def NVWSLowerWarpGroup : Pass<"nvws-lower-warp-group", "mlir::ModuleOp"> {
  let summary = "Convert nvws.warp_group to ttg.warp_specialize.";

  let description = [{
    Convert nvws.warp_group to ttg.warp_specialize.

    If the first group of nvws.warp_group matches the global
    ttg.num_warps, it will be come the default region of ttg.warp_specialize.
    If not, the ttg.warp_specialize default region will be empty, and all
    warp groups will become isolated regions.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def NVWSAssignStagePhase : Pass<"nvws-assign-stage-phase", "mlir::ModuleOp"> {
  let summary = "Assign buffer stage to nvws.aref.*.";

  let description = [{
    Assign buffer stage & phase to nvws.aref.*

    The pass will assign buffer stage to each aref op, and phase for enter ops.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSLowerAref : Pass<"nvws-lower-aref", "mlir::ModuleOp"> {
  let summary = "Convert nvws.aref.* to ttng.*barrier* ops.";

  let description = [{
    Convert nvws.aref.* to ttng.*barrier* ops.

    The pass will convert each aref to a matched value and barrier set,
    and will determined appropriate waits/signalling for values being
    "empty" or "full" from the use/def chain of aref get/put.

    This lowering may yield non-ideal parallelism in certain cases,
    which will be optimized by follow up peephole passes.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSInsertAref: Pass<"nvws-insert-aref", "mlir::ModuleOp"> {
  let summary = "Insert arefs between producer and consumer partitions.";

  let description = [{
    To automate barrier synchronizations between producer and consumer
    partitions, arefs are introduced in the IR. Currently, this pass handles
    only SMEM producers and consumers.

    Specifically, for producer partitions, a producing operation is
    wrapped in an ArefPutEnterOp and ArefPutExitOp pair. A descriptor load
    op is replaced with the corresponding NVWS op, to store its result
    into the SMEM buffer owned by an aref. For consumer partitions, a reference
    to the original SMEM buffer is replaced with an indirection via ArefGetEnterOp on
    the SMEM buffer owned by an aref. ArefGetExitOp is placed after the post-dominant
    consumer operation.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

}

#endif // NVWS_PASSES
