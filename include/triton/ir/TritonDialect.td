#ifndef TRITON_DIALECT
#define TRITON_DIALECT

include "mlir/IR/OpBase.td"

def Triton_Dialect : Dialect {
  let name = "triton";

  let cppNamespace = "::mlir::triton";

  let summary = "The Triton IR in MLIR";

  let description = [{
    Triton Dialect.

    Dependent Dialects:
      * Arithmetic:
        * addf, addi, andi, cmpf, cmpi, divf, fptosi, ...
      * Tensor:
        * reshape (?)
      * ControlFlow:
        * bf, cond_bf
      * Func:
        * call, return
  }];

  let dependentDialects = [
    "arith::ArithmeticDialect",
    "tensor::TensorDialect",
    "StandardOpsDialect",
    "scf::SCFDialect"

    // Since LLVM 15
    // "cf::ControlFlowDialect",
    // "func::FuncDialect"
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

#endif // TRITON_DIALECT
