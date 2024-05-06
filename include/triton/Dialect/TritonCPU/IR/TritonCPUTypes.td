#ifndef TRITONCPU_TYPES
#define TRITONCPU_TYPES

include "triton/Dialect/TritonCPU/IR/TritonCPUDialect.td"
include "mlir/IR/AttrTypeBase.td"

class TTC_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TritonCPU_Dialect, name, traits> {
    let mnemonic = _mnemonic;
}

def TTC_TokenType : TTC_TypeDef<"Token", "token"> {
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
