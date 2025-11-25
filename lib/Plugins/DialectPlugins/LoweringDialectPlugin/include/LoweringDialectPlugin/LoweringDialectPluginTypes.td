#ifndef LOWERINGDIALECTPLUGIN_TYPES
#define LOWERINGDIALECTPLUGIN_TYPES

include "mlir/IR/AttrTypeBase.td"
include "LoweringDialectPlugin/LoweringDialectPluginDialect.td"

//===----------------------------------------------------------------------===//
// LoweringDialectPlugin type definitions
//===----------------------------------------------------------------------===//

class LoweringDialectPlugin_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<LoweringDialectPlugin_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def LoweringDialectPlugin_CustomType : LoweringDialectPlugin_Type<"Custom", "custom"> {
    let summary = "LoweringDialectPlugin custom type";
    let description = "Custom type in loweringdialectplugin dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // LOWERINGDIALECTPLUGIN_TYPES
