#ifndef TRITON_DIALECT_TRITONGPUROCM_TYPES
#define TRITON_DIALECT_TRITONGPUROCM_TYPES

include "triton/Dialect/TritonGPUROCM/IR/TritonGPUDialect.td"
include "mlir/IR/AttrTypeBase.td"

class TTG_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TritonGPUROCM_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def TTG_TokenType : TTG_TypeDef<"Token", "token"> {
  let parameters = (ins "int32_t":$type);

  let builders = [
    TypeBuilder<(ins "unsigned":$type), [{
      return $_get($_ctxt, type);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let skipDefaultBuilders = 1;
}

#endif
