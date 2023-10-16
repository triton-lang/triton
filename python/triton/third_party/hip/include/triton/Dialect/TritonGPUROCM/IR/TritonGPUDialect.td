#ifndef TRITON_DIALECT_TRITONGPUROCM_DIALECT
#define TRITON_DIALECT_TRITONGPUROCM_DIALECT

include "mlir/IR/OpBase.td"

def TritonGPUROCM_Dialect : Dialect {
  let name = "triton_gpu_rocm";

  let cppNamespace = "::mlir::triton::gpu_rocm";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton GPU ROCM Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "mlir::triton::nvgpu::NVGPUDialect",
    "mlir::gpu::GPUDialect",
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    static std::string getNumWarpsAttrName() { return "triton_gpu_rocm.num-warps"; }
    static int getNumWarps(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu_rocm.num-warps"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu_rocm.num-warps attribute");
      return mod->getAttr("triton_gpu_rocm.num-warps").cast<IntegerAttr>().getInt();
    }
    static int getNumCTAs(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu_rocm.num-ctas"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu_rocm.num-ctas attribute");
      return mod->getAttr("triton_gpu_rocm.num-ctas").cast<IntegerAttr>().getInt();
    }
    static int getComputeCapability(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu_rocm.compute-capability"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu_rocm.compute-capability attribute");
      return mod->getAttrOfType<IntegerAttr>("triton_gpu_rocm.compute-capability").getInt();
    }
    void registerTypes();

    static std::string getThreadsPerWarpAttrName() { return "triton_gpu_rocm.threads-per-warp"; }

    static int getThreadsPerWarp(ModuleOp mod) {
      Attribute threadsPerWarp = mod->getDiscardableAttr("triton_gpu_rocm.threads-per-warp");
      if(!threadsPerWarp) {
        return 64;
      }
      return threadsPerWarp.cast<IntegerAttr>().getInt();
    }
    static int getSharedSize(ModuleOp mod) {
      Attribute sharedAttr = mod->getDiscardableAttr("triton_gpu_rocm.shared");
      if(!sharedAttr) {
        return 0;
      }
      return sharedAttr.cast<IntegerAttr>().getInt();
    }

  }];

  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
