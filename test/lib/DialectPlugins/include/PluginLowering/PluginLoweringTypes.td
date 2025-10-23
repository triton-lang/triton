//===- PluginLoweringTypes.td - PluginLowering dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PLUGINLOWERING_TYPES
#define PLUGINLOWERING_TYPES

include "mlir/IR/AttrTypeBase.td"
include "PluginLoweringDialect.td"

// FIXME
// include "PluginLowering/PluginLoweringDialect.td"

//===----------------------------------------------------------------------===//
// PluginLowering type definitions
//===----------------------------------------------------------------------===//

class PluginLowering_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<PluginLowering_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def PluginLowering_CustomType : PluginLowering_Type<"Custom", "custom"> {
    let summary = "PluginLowering custom type";
    let description = "Custom type in pluginlowering dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // PLUGINLOWERING_TYPES
