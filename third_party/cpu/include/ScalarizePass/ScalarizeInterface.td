#ifndef MLIR_SCALARIZEINTERFACE
#define MLIR_SCALARIZEINTERFACE

include "mlir/IR/OpBase.td"

def ScalarizeInterface : OpInterface<"ScalarizeInterface"> {
  let description = [{
    Interface for allowing operations to expose information needed to
    scalarize them or in simpler terms inserts SCF loops to reduce amount of
    generated ir. Similar with checking operands of specific operations for
    constancy - to understand is it possible to put it inside of loop's body.
  }];
  let cppNamespace = "mlir::triton::cpu";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Checks operand and is ScalarizeInterface registered for this operation.
        }],
        /*retType=*/"bool",
        /*methodName=*/"canComputeScalarValue",
        /*args=*/(ins
          "mlir::Value ":$vals)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns value that can be  put inside of generated cycle and creates required constants.
          Can go throught operands to check type of passed values. Implementation for static indeces.
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"computeScalarValue",
        /*args=*/(ins
          "mlir::Value ":$vals,
          "mlir::ArrayRef<int64_t>  ":$indices,
          "mlir::PatternRewriter &":$rewriter)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns value that can be  put inside of generated cycle and creates required constants.
          Can go throught operands to check type of passed values. Implementation for dynamic indices
          which is in common used in loops to iterate with Inductional Variable.
        }],
        /*retType=*/"mlir::Value",
        /*methodName=*/"computeScalarValueForLoop",
        /*args=*/(ins
          "mlir::Value ":$vals,
          "mlir::ValueRange  ":$indices,
          "mlir::PatternRewriter &":$rewriter)
      >
  ];
}

#endif // MLIR_SCALARIZEINTERFACE
