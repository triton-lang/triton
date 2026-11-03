#ifndef DIALECTPLUGIN_TYPES
#define DIALECTPLUGIN_TYPES

include "mlir/IR/AttrTypeBase.td"
include "DialectPlugin/DialectPluginDialect.td"

//===----------------------------------------------------------------------===//
// DialectPlugin type definitions
//===----------------------------------------------------------------------===//

class DialectPlugin_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<DialectPlugin_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def DialectPlugin_CustomType : DialectPlugin_Type<"Custom", "custom"> {
    let summary = "DialectPlugin custom type";
    let description = "Custom type in plugin dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // DIALECTPLUGIN_TYPES
