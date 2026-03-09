#ifndef TRITON_APPLE_GPU_DIALECT
#define TRITON_APPLE_GPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonAppleGPU_Dialect : Dialect {
  let name = "applegpu";
  let cppNamespace = "::mlir::triton::applegpu";

  let description = [{
    TritonAppleGPU Dialect hosts Apple-specific ops and encodings
    at the TritonGPU abstraction level.

    The primary encoding is AppleMmaEncodingAttr, which represents
    tensors distributed across simdgroup_multiply_accumulate tiles.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect"
  ];

  let useDefaultAttributePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // TRITON_APPLE_GPU_DIALECT
