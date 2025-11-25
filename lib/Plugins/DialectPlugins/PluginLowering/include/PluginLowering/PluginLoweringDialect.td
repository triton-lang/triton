#ifndef PLUGINLOWERING_DIALECT
#define PLUGINLOWERING_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// PluginLowering dialect definition.
//===----------------------------------------------------------------------===//

def PluginLowering_Dialect : Dialect {
    let name = "pluginlowering";
    let summary = "A pluginlowering out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::triton::pluginlowering";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base pluginlowering operation definition.
//===----------------------------------------------------------------------===//

class PluginLowering_Op<string mnemonic, list<Trait> traits = []> :
        Op<PluginLowering_Dialect, mnemonic, traits>;

#endif // PLUGINLOWERING_DIALECT
