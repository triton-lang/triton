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

def NVWSAssignGroups : Pass<"nvws-assign-groups", "mlir::ModuleOp"> {
  let summary = "Assign initial group information";

  let description = [{
    Determines the warp group strategy for the program, and produces the initial
    group assignments for ops.
  }];

  let constructor = "mlir::createNVWSAssignGroupsPass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def NVWSPropagateGroups : Pass<"nvws-propagate-groups", "mlir::ModuleOp"> {
  let summary = "Propagates group information";

  let description = [{
    Propagates group information throughout the program, based on
    the initial group assignment.
  }];

  let constructor = "mlir::createNVWSPropagateGroupsPass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def NVWSLowerWarpGroup : Pass<"nvws-lower-warp-group", "mlir::ModuleOp"> {
  let summary = "Convert nvws.warp_group to ttg.warp_specialize.";

  let description = [{
    Convert nvws.warp_group to ttg.warp_specialize.

    If the first group of nvws.warp_group matches the global
    ttg.num_warps, it will be come the default region of ttg.warp_specialize.
    If not, the ttg.warp_specialize default region will be empty, and all
    warp groups will become isolated regions.
  }];

  let constructor = "mlir::createNVWSLowerWarpGroupPass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
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

  let constructor = "mlir::createNVWSLowerArefPass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect"
  ];
}

def NVWSArefCanonicalize : Pass<"nvws-aref-canonicalize", "mlir::ModuleOp"> {
  let summary = "Canonicalize IR for aref insertion";

  let description = [{
    Lower aref operations in TritonNvidiaGPUDialect.
  }];

  let constructor = "mlir::createNVWSArefCanonicalizePass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSArefInsertion: Pass<"nvws-aref-insertion", "mlir::ModuleOp"> {
  let summary = "insert arefs into the code";

  let description = [{
    Lower aref operations in TritonNvidiaGPUDialect.
  }];

  let constructor = "mlir::createNVWSArefInsertionPass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSArefCodeSplit : Pass<"nvws-aref-code-split", "mlir::ModuleOp"> {
  let summary = "split code with arefs into warp-groups";

  let description = [{
    Lower aref operations in TritonNvidiaGPUDialect.
  }];

  let constructor = "mlir::createNVWSArefCodeSplitPass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSArefCopyElimination : Pass<"nvws-aref-copy-elimination", "mlir::ModuleOp"> {
  let summary = "elminate aref_copies";

  let description = [{
    Lower aref operations in TritonNvidiaGPUDialect.
  }];

  let constructor = "mlir::createNVWSArefCopyEliminationPass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSArefCopyLowering : Pass<"nvws-aref-copy-lowering", "mlir::ModuleOp"> {
  let summary = "lower remaining aref copies";

  let description = [{
    Lower aref operations in TritonNvidiaGPUDialect.
  }];

  let constructor = "mlir::createNVWSArefCopyLoweringPass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSArefOptimize : Pass<"nvws-aref-optimize", "mlir::ModuleOp"> {
  let summary = "aref optimization passes";

  let description = [{
    Lower aref operations in TritonNvidiaGPUDialect.
  }];

  let constructor = "mlir::createNVWSArefOptimizePass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSArefAsyncOps : Pass<"nvws-aref-async-ops", "mlir::ModuleOp"> {
  let summary = "aref optimization passes";

  let description = [{
    Lower aref operations in TritonNvidiaGPUDialect.
  }];

  let constructor = "mlir::createNVWSArefAsyncOpsPass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}

def NVWSArefIndex : Pass<"nvws-aref-index", "mlir::ModuleOp"> {
  let summary = "aref optimization passes";

  let description = [{
    Lower aref operations in TritonNvidiaGPUDialect.
  }];

  let constructor = "mlir::createNVWSArefIndexPass()";

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];
}


def NVWSArefDepthPass : Pass<"nvws-aref-depth", "mlir::ModuleOp"> {
  let summary = "add depth to arefs";

  let description = [{
    Lower aref operations in TritonNvidiaGPUDialect.
  }];

  let dependentDialects = [
    "mlir::triton::nvws::NVWSDialect",
    "mlir::triton::TritonDialect",
    "mlir::triton::gpu::TritonGPUDialect",
    "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect"
  ];

  let options = [
    Option<"numStages", "num-stages",
           "int32_t", /*default*/"3",
           "number of warp specialization pipeline stages">,
    Option<"mmaDepth", "mma-depth",
           "int32_t", /*default*/"2",
           "number of MMA pipeline stages">
  ];
}

#endif // NVWS_PASSES
