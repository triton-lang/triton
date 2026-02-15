#ifndef GLUON_DIALECT
#define GLUON_DIALECT

include "mlir/IR/OpBase.td"

def Gluon_Dialect : Dialect {
  let name = "gluon";
  let cppNamespace = "::mlir::triton::gluon";
  let description = [{
    Gluon dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
    "mlir::gpu::GPUDialect",
  ];
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
