#ifndef TRITONGPU_CONVERSION_PASSES
#define TRITONGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertTritonGPUToLLVM : Pass<"convert-triton-gpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let description = [{

    }];

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::instrument::TritonInstrumentDialect",
                             "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                             "mlir::triton::nvgpu::NVGPUDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"ptxVersion", "ptx-version",
               "int32_t", /*default*/"80",
               "PTX version">,
        Option<"enableConcurrencySanitizer", "enable-concurrency-sanitizer",
               "bool", /*default*/"false",
               "run ConSan after fence and membar preparation">,
    ];
}

def InitializeWSClusterBarriers : Pass<"initialize-ws-cluster-barriers", "mlir::ModuleOp"> {
    let summary = "Initialize warp-specialized cluster barrier mbarriers";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::NVVM::NVVMDialect",
                             "mlir::triton::gpu::TritonGPUDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"ptxVersion", "ptx-version",
               "int32_t", /*default*/"80",
               "PTX version">,
    ];
}

def AllocateSharedMemoryNv : Pass<"allocate-shared-memory-nv", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation for Nvidia";

  let description = [{
    See `allocate-shared-memory` for more details.
  }];

  let options = [
      Option<"computeCapability", "compute-capability",
             "int32_t", /*default*/"80",
             "device compute capability">,
      Option<"ptxVersion", "ptx-version",
             "int32_t", /*default*/"80",
             "PTX version">,
  ];
}

def SetMinimumSharedMemory : Pass<"set-minimum-shared-memory", "mlir::ModuleOp"> {
  let summary = "Set a lower bound on the module shared memory size";

  let options = [
      Option<"minimumSize", "minimum-size",
             "int32_t", /*default*/"0",
             "minimum shared memory size in bytes">,
  ];
}

def NvidiaOptimizeConditionalXor
    : Pass<"nvidia-optimize-conditional-xor", "mlir::ModuleOp"> {
  let summary = "Fuse conditional XORs of one-hot reduction shuffles into LOP3";
  let description = [{
    After TritonGPU to LLVM conversion and LLVM canonicalization, replace
    conditional XORs of shuffles originating from one-hot reductions with
    NVIDIA LOP3 instructions. Freeze the selected value to preserve poison
    masking, and remove the temporary gather provenance attributes.
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}


def ConvertWarpSpecializeToLLVM : Pass<"convert-warp-specialize-to-llvm", "mlir::ModuleOp"> {
  let summary = "lower `ttg.warp_specialize` to LLVM";
  let description = [{
    The `convert-warp-specialize-to-llvm` pass performs codegen for warp
    specialization. It is a function-level transformation that rewrites
    warp-specialized kernels by using shared memory and barriers to communicate
    states between the default warpgroup and the worker warps.
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::NVVM::NVVMDialect"];
}

#endif // TRITONGPU_CONVERSION_PASSES
