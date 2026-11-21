#ifndef TRITONINSTRUMENT_DIALECT
#define TRITONINSTRUMENT_DIALECT

include "mlir/IR/OpBase.td"

def TritonInstrument_Dialect : Dialect {
  let name = "tti";
  let cppNamespace = "::mlir::triton::instrument";

  // Preserve the existing attr-dict syntax for inherent attributes.
  let useStrictPropertiesInAssemblyFormat = 0;
}

#endif // TRITONINSTRUMENT_DIALECT
