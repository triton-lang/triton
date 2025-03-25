//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef TRITONXPU_PASSES
#define TRITONXPU_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Functionality Pass
//===----------------------------------------------------------------------===//

def TritonXPUCreateGM2LM : Pass<"tritonxpu-create-gm2lm", "mlir::ModuleOp"> {
  let summary = "Create GM2LM for XPU.";

  let description = [{
    tt.load => triton_xpu.gm2lm + triton_xpu.load;
    tt.store => triton_xpu.store + triton_xpu.lm2gm.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"xpuArch", "xpu-arch",
           "uint32_t", /*default*/"3",
           "XPU architecture">,
    Option<"atomicSim", "atomic-sim",
           "bool", /*default*/"1",
           "Atomic Simulation">
  ];
}

def TritonXPULegalize : Pass<"tritonxpu-legalize", "mlir::ModuleOp"> {
  let summary = "Legalize for XPU.";

  let description = [{
    Insert scf.for when local memory is smaller than block size.
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"bufferSize", "lm-buflen",
            "uint32_t", /*default*/"128",
            "buffer size for local memory">,
    Option<"coreNum", "core-num",
            "uint32_t", /*default*/"64",
            "core num">
  ];
}

def TritonXPUMask : Pass<"tritonxpu-mask", "mlir::ModuleOp"> {
  let summary = "Mask for Calculation.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPUAlloca : Pass<"tritonxpu-alloca", "mlir::ModuleOp"> {
  let summary = "Alloca buffer for gm2lm and lm2gm.";

  let description = [{
    triton_xpu.gm2lm => triton_xpu.alloca + triton_xpu.gm2lm
    triton_xpu.lm2gm => triton_xpu.alloca + triton_xpu.lm2gm
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}


def TritonXPUDtypeConvert : Pass<"tritonxpu-dtype-convert", "mlir::ModuleOp"> {
  let summary = "Dtype Convert for XPU.";

  let description = [{
    XPU2: FP16 => FP32
    XPU3: BF16 => FP32
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
  let options = [
    Option<"xpuArch", "xpu-arch",
           "uint32_t", /*default*/"3",
           "XPU architecture">
  ];
}

def TritonXPULoopGrid : Pass<"tritonxpu-loop-grid", "mlir::ModuleOp"> {
  let summary = "Create loop on triton programs for out-of-bounds grid_size.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPUUnrollControl : Pass<"tritonxpu-unroll-control", "mlir::ModuleOp"> {
  let summary = "Control the unroll size.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

//===----------------------------------------------------------------------===//
// Optimization Pass
//===----------------------------------------------------------------------===//

def TritonXPUOffsetAnalysis : Pass<"tritonxpu-offset-analysis", "mlir::ModuleOp"> {
  let summary = "Analysis Ptr's Offset State.";

  let description = [{
    Given buffer_size = 8

    offsets = [0, 1, 2, 3, 4, 5, 6, 7] -> Continuous
    offsets = [0, 0, 0, 0, 0, 0, 0, 0] -> DiscreteSame
    offsets = [4, 5, 6, 7, 0, 1, 2 ,3] -> Discrete
    offsets = [0, 1, 1001, 1002, 3, 4, 2001, 2002] -> Unknown
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"dumpFlag", "dump-flag",
           "bool", /*default*/"0",
           "detail dump flag">
  ];
}

def TritonXPUCoreTiling : Pass<"tritonxpu-core-tiling", "mlir::ModuleOp"> {
  let summary = "Core Tiling Optimization.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"dumpFlag", "dump-flag",
           "bool", /*default*/"0",
           "detail dump flag">,
    Option<"bufferSize", "lm-buflen",
            "uint32_t", /*default*/"128",
            "buffer size for local memory">
  ];
}


def TritonXPUVectorize : Pass<"tritonxpu-vectorize", "mlir::ModuleOp"> {
  let summary = "Vectorize Calculation.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"dumpFlag", "dump-flag",
           "bool", /*default*/"0",
           "detail dump flag">
  ];
}

def TritonXPUMemoryAsync : Pass<"tritonxpu-memory-async", "mlir::ModuleOp"> {
  let summary = "Memory Async Optimization.";

  let description = [{
  }];

  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];

  let options = [
    Option<"dumpFlag", "dump-flag",
           "bool", /*default*/"0",
           "detail dump flag">,
  ];
}

def TritonXPUInterleave : Pass<"tritonxpu-interleave", "mlir::ModuleOp"> {
  let summary = "Interleave for XPU.";
  let description = [{
    Convert triton.make_range triton_xpu.interleave.
  }];
  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPUStoreControl : Pass<"tritonxpu-store-control", "mlir::ModuleOp"> {
  let summary = "Store Control for XPU.";
  let description = [{
    Only Store isCoreId0InsideGroup=0 for ReduceOp
  }];
  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

def TritonXPUOtherSim : Pass<"tritonxpu-other-sim", "mlir::ModuleOp"> {
  let summary = "Simulate Other for XPU.";
  let description = [{
    Simulate Other in LoadOp/StoreOp.
  }];
  let dependentDialects = ["mlir::triton::xpu::TritonXPUDialect"];
}

#endif // TRITONXPU_PASSES
