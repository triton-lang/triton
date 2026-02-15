#ifndef PROTONGPU_DIALECT
#define PROTONGPU_DIALECT

include "mlir/IR/OpBase.td"

def ProtonGPU_Dialect : Dialect {
  let name = "proton_gpu";
  let cppNamespace = "::mlir::triton::proton::gpu";

  let description = [{
    Proton GPU dialect.
  }];

  let dependentDialects = [
    "triton::gpu::TritonGPUDialect",
		"triton::proton::ProtonDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif  // PROTONGPU_DIALECT
