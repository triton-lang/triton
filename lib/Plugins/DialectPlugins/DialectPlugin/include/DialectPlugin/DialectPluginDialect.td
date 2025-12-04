#ifndef DIALECTPLUGIN_DIALECT
#define DIALECTPLUGIN_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DialectPlugin dialect definition.
//===----------------------------------------------------------------------===//

def DialectPlugin_Dialect : Dialect {
    let name = "dialectplugin";
    let summary = "A dialectplugin out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::triton::dialectplugin";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base dialectplugin operation definition.
//===----------------------------------------------------------------------===//

class DialectPlugin_Op<string mnemonic, list<Trait> traits = []> :
        Op<DialectPlugin_Dialect, mnemonic, traits>;

#endif // DIALECTPLUGIN_DIALECT
