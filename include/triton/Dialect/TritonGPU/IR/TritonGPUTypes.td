#ifndef TRITONGPU_TYPES
#define TRITONGPU_TYPES

include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"
include "mlir/IR/AttrTypeBase.td"

class TTG_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<TritonGPU_Dialect, name, traits> {
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

def TTG_AsyncToken : TTG_TypeDef<"AsyncToken",
                                    "async.token", []> {
  let summary = "async token type";
  let description = [{
    `ttg.async.token` is a type returned by an asynchronous operation.
    It is used to establish an SSA-based link between async operations
    and operations that group or synchronize the async operations.
  }];
}

#endif
