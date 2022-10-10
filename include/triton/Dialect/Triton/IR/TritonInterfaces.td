#ifndef TRITON_INTERFACES
#define TRITON_INTERFACES

include "mlir/IR/OpBase.td"

def LayoutVerificationAttrInterface : AttrInterface<"LayoutVerificationAttrInterface"> {

  let description = [{
    An interface that ov errides hooks for verifying if this attribute is a valid as a layout
    for the given operation.
  }];

  let methods = [
    InterfaceMethod<
      "This is an example of a non-static hook to an operation.",
      "::mlir::LogicalResult", "verifyLayoutForArg", (ins "::mlir::Operation *":$op, "unsigned":$argNo)
    >,
  ];
}

#endif // TRITON_INTERFACES