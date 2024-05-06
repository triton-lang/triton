#ifndef TRITONCPU_DIALECT
#define TRITONCPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonCPU_Dialect : Dialect {
  let name = "triton_cpu";

  let cppNamespace = "::mlir::triton::cpu";

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton CPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "tensor::TensorDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif
