#ifndef TRITONINSTRUMENT_DIALECT
#define TRITONINSTRUMENT_DIALECT

include "mlir/IR/OpBase.td"

def TritonInstrument_Dialect : Dialect {
  let name = "tti";
  let cppNamespace = "::mlir::triton::instrument";
}

#endif // TRITONINSTRUMENT_DIALECT
