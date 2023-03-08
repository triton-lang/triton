#ifndef TRITONGPU_DIALECT
#define TRITONGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonGPU_Dialect : Dialect {
  let name = "triton_gpu";

  let cppNamespace = "::mlir::triton::gpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton GPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "mlir::gpu::GPUDialect",
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    static std::string getNumWarpsAttrName() { return "triton_gpu.num-warps"; }
    static int getNumWarps(ModuleOp mod) { 
      if(!mod->hasAttr("triton_gpu.num-warps"))
        llvm::report_fatal_error(
            "TritonGPU module should contain a triton_gpu.num-warps attribute");
      return mod->getAttr("triton_gpu.num-warps").cast<IntegerAttr>().getInt();
    }

    static std::string getThreadsPerWarpAttrName() { return "triton_gpu.threads-per-warp"; }
    static int getThreadsPerWarp(ModuleOp mod) {
      if(!mod->hasAttr("triton_gpu.threads-per-warp"))
        llvm::report_fatal_error(
                "TritonGPU module should contain a triton_gpu.threads-per-warp attribute");
      return mod->getAttr("triton_gpu.threads-per-warp").cast<IntegerAttr>().getInt();
    }

  }];
  
  let useDefaultAttributePrinterParser = 1;
}

#endif
