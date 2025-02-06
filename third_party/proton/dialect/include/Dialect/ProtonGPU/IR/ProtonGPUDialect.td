#ifndef PROTON_GPU_DIALECT
#define PROTON_GPU_DIALECT

include "mlir/IR/OpBase.td"

def Proton_GPU_Dialect : Dialect {
  let name = "proton_gpu";
  let cppNamespace = "::mlir::triton::proton::gpu";

  let description = [{
    Proton GPU dialect.
  }];

  let dependentDialects = [
    "triton::TritonGPUDialect"
		"triton::proton::ProtonDialect",
  ];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif  // PROTON_GPU_DIALECT
