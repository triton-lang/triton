#ifndef LOWERINGDIALECTPLUGIN_OPS
#define LOWERINGDIALECTPLUGIN_OPS

include "LoweringDialectPlugin/LoweringDialectPluginTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LoweringDialectPlugin_MagicOp : LoweringDialectPlugin_Op<"magic", [Pure,
                                             SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `loweringdialectplugin.magic` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the magic operation to %0
        %1 = loweringdialectplugin.magic %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

#endif // LOWERINGDIALECTPLUGIN_OPS
