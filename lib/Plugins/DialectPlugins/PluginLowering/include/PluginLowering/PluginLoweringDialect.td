#ifndef LOWERINGDIALECTPLUGIN_DIALECT
#define LOWERINGDIALECTPLUGIN_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// LoweringDialectPlugin dialect definition.
//===----------------------------------------------------------------------===//

def LoweringDialectPlugin_Dialect : Dialect {
    let name = "loweringdialectplugin";
    let summary = "A loweringdialectplugin out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::triton::loweringdialectplugin";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base loweringdialectplugin operation definition.
//===----------------------------------------------------------------------===//

class LoweringDialectPlugin_Op<string mnemonic, list<Trait> traits = []> :
        Op<LoweringDialectPlugin_Dialect, mnemonic, traits>;

#endif // LOWERINGDIALECTPLUGIN_DIALECT
