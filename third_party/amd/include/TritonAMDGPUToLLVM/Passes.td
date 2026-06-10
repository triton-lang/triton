#ifndef TRITONAMDGPU_CONVERSION_PASSES
#define TRITONAMDGPU_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AllocateAMDGPUSharedMemory : Pass<"allocate-amdgpu-shared-memory", "mlir::ModuleOp"> {
  let summary = "Add metadata for shared memory allocation";
  let constructor = "mlir::triton::createAllocateAMDGPUSharedMemoryPass(\"\")";

  let options = [
      Option<"arch", "arch", "std::string", /*default*/"\"\"",
             "gfx target device architecture, e.g., gfx942">,
  ];

  let description = [{
    This pass uses the `ModuleAllocation` analysis to:
      - Annotate modules with an attribute with the amount of shared/local
        memory used.
      - Annotate operations with an offset into the total shared/local memory.
  }];
}

def ConvertTritonAMDGPUToLLVM : Pass<"convert-triton-amdgpu-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonGPU to LLVM";
    let constructor = "mlir::triton::createConvertTritonAMDGPUToLLVMPass(\"\", /*ftz=*/true)";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::ROCDL::ROCDLDialect"];

    let options = [
        Option<"gfxArch", "gfx-arch", "std::string", /*default*/"std::string{}",
               "Specify the GFX architecture name, e.g., gfx942, gfx950, etc.">,
        Option<"ftz", "ftz", "bool", /*default*/"true",
               "flush denorms for math functions">,
    ];
}

def ConvertBuiltinFuncToLLVM : Pass<"convert-builtin-func-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Builtin Func to LLVM";
    let constructor = "mlir::triton::createConvertBuiltinFuncToLLVMPass(\"\", /*ftz=*/true)";

    let dependentDialects = ["mlir::LLVM::LLVMDialect"];

    let options = [
        Option<"gfxArch", "gfx-arch", "std::string", /*default*/"std::string{}",
               "Specify the GFX architecture name, e.g., gfx942, gfx950, etc.">,
        Option<"ftz", "ftz", "bool", /*default*/"true",
               "flush denorms for math functions">,
    ];
}

def ConvertWarpPipeline : Pass<"convert-warp-pipeline", "mlir::ModuleOp"> {
    let summary = "Emit conditional barrier and inlines scf.execute_region for warp-pipeline";
    let constructor = "mlir::triton::AMD::createConvertWarpPipelinePass(\"\")";

    let dependentDialects = ["mlir::LLVM::LLVMDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::triton::amdgpu::TritonAMDGPUDialect"];

    let options = [
        Option<"gfxArch", "gfx-arch", "std::string", /*default*/"std::string{}",
               "Specify the GFX architecture name, e.g., gfx942, gfx950, etc.">,
    ];
}

def TritonAMDGPUConvertWarpSpecializeToLLVM : Pass<"triton-amdgpu-convert-warp-specialize-to-llvm", "mlir::ModuleOp"> {
  let summary = "lower `ttg.warp_specialize` to LLVM";
  let constructor = "mlir::triton::AMD::createTritonAMDGPUConvertWarpSpecializeToLLVMPass(\"\")";
  let description = [{
    The `triton-amdgpu-convert-warp-specialize-to-llvm` pass performs codegen for warp
    specialization. It is a function-level transformation that rewrites
    warp-specialized kernels by using shared memory and barriers to communicate
    states between the default warpgroup and the worker warps.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::ROCDL::ROCDLDialect"];

  let options = [
    Option<"gfxArch", "gfx-arch", "std::string", /*default*/"std::string{}",
           "Specify the GFX architecture name, e.g., gfx942, gfx950, etc.">,
  ];
}

#endif
