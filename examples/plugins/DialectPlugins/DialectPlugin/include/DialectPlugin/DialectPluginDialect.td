#ifndef DIALECTPLUGIN_DIALECT
#define DIALECTPLUGIN_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DialectPlugin dialect definition.
//===----------------------------------------------------------------------===//

def DialectPlugin_Dialect : Dialect {
    let name = "plugin";
    let summary = "A plugin out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::triton::plugin";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];

}

#endif // DIALECTPLUGIN_DIALECT
