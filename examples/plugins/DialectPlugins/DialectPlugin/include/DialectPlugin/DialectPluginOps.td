#ifndef DIALECTPLUGIN_OPS
#define DIALECTPLUGIN_OPS

include "DialectPlugin/DialectPluginTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class DialectPlugin_Op<string mnemonic, list<Trait> traits = []> :
        Op<DialectPlugin_Dialect, mnemonic, traits>;

def DialectPlugin_MagicOp : DialectPlugin_Op<"magic", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define a custom operation.";
    let description = [{
        The `plugin.magic` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the magic operation to %0
        %1 = plugin.magic %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

def DialectPlugin_FMagicOp : DialectPlugin_Op<"fmagic", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define a custom operation.";
    let description = [{
        The `plugin.fmagic` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the magic operation to tensor %0 and return %1
        %1 = plugin.fmagic %0 : i32
        ```
    }];

    let arguments = (ins FloatLike:$input);
    let results = (outs FloatLike:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // DIALECTPLUGIN_OPS
